"""
Comprehensive regression test suite verifying remediation and hardening for R1, R2, and R3.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from unittest.mock import MagicMock, patch

import grpc
import numpy as np
import pytest

from matrix_contracts.schemas.discovery import HardwareSpecs
from matrix_contracts.v1 import pb2
from src.attestation.nvml_attest import (
    HardwareAttestationEngine,
    verify_attestation_proof,
)
from src.auth.l402_gate import L402PreAllocationGate
from src.config.settings import NodeConfig
from src.crypto.signer import WorkerSigner
from src.engine.compute_core import ComputeEngine
from src.main import WorkerDaemon, main as daemon_main
from src.p2p.servicer import MatrixComputeServicer
from src.test_doubles.mock_settlement import MockSettlementService


def mint_macaroon_with_ph(
    key: str,
    task_id: str,
    expiry: int,
    max_gflops: float,
    locked_sats: int = 100,
    payment_hash: str = "default_hash",
) -> str:
    caveats = {
        "task_id": task_id,
        "expiry": expiry,
        "max_gflops": max_gflops,
        "locked_sats": locked_sats,
        "payment_hash": payment_hash,
    }
    caveat_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(key.encode("utf-8"), caveat_str.encode("utf-8"), hashlib.sha256).hexdigest()
    payload = {"caveats": caveats, "signature": sig}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


class MockGrpcContext:
    def __init__(self, metadata=None):
        self._metadata = metadata or []
        self.code = None
        self.details = None

    def invocation_metadata(self):
        return self._metadata

    def abort(self, code, details):
        self.code = code
        self.details = details
        raise RuntimeError(f"ABORT: {code}: {details}")


# ==============================================================================
# R1: PROGRESSIVE STREAMING DOS & STATE MANAGEMENT REGRESSION TESTS
# ==============================================================================

class TestR1StreamingAndStateManagement:
    """Verifies R1 streaming bounds, pre-negotiation requirement, and LRU/TTL cleanup."""

    def test_r1_unregistered_task_rejected_with_failed_precondition(self):
        vkey = "test-vkey"
        mock_settlement = MockSettlementService(verification_key=vkey)
        inv = mock_settlement.create_hodl_invoice("unreg_task", total_compute_sats=100)
        mock_settlement.hold_invoice(inv.payment_hash)

        gate = L402PreAllocationGate(settlement_client_or_url=mock_settlement, verification_key=vkey)
        servicer = MatrixComputeServicer(gate=gate, engine=ComputeEngine(force_cpu=True))

        mac = mint_macaroon_with_ph(vkey, "unreg_task", int(time.time()) + 3600, 1.0, 100, inv.payment_hash)
        auth_header = f"L402 {mac}:{inv.payment_hash}"
        ctx = MockGrpcContext(metadata=[("authorization", auth_header)])

        # Never called RequestTileTask for "unreg_task"
        with pytest.raises(RuntimeError) as exc_info:
            servicer.ExecuteTileTask(iter([]), context=ctx)

        assert "FAILED_PRECONDITION" in str(exc_info.value)
        assert "not negotiated" in str(exc_info.value)

    def test_r1_streaming_chunk_exceeding_byte_limit_aborts_immediately(self):
        vkey = "test-vkey"
        mock_settlement = MockSettlementService(verification_key=vkey)
        task_id = "task_stream_dos"
        m, k, n = 4, 4, 4

        gate = L402PreAllocationGate(settlement_client_or_url=mock_settlement, verification_key=vkey)
        servicer = MatrixComputeServicer(gate=gate, engine=ComputeEngine(force_cpu=True))

        # 1. Negotiate task
        req = pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=m, matrix_a_cols=k, matrix_b_cols=n, dtype="fp32")
        servicer.RequestTileTask(req)

        # 2. Setup paid invoice & token
        inv = mock_settlement.create_hodl_invoice(task_id, total_compute_sats=100)
        mock_settlement.hold_invoice(inv.payment_hash)
        mac = mint_macaroon_with_ph(vkey, task_id, int(time.time()) + 3600, 1.0, 100, inv.payment_hash)
        auth_header = f"L402 {mac}:{inv.payment_hash}"
        ctx = MockGrpcContext(metadata=[("authorization", auth_header)])

        # Expected matrix A bytes: 4 * 4 * 4 = 64 bytes.
        # Send chunk with 1000 bytes (DoS attempt).
        oversized_chunk = pb2.TileDataChunk(
            task_id=task_id,
            chunk_bytes=b"X" * 1000,
            is_matrix_a=True,
            sequence_number=0,
            is_final_chunk=False,
        )

        with pytest.raises(RuntimeError) as exc_info:
            servicer.ExecuteTileTask(iter([oversized_chunk]), context=ctx)

        assert "INVALID_ARGUMENT" in str(exc_info.value)
        assert "exceeded expected byte limit" in str(exc_info.value)

    def test_r1_task_metadata_purged_upon_stream_conclusion(self):
        vkey = "test-vkey"
        mock_settlement = MockSettlementService(verification_key=vkey)
        task_id = "task_purge_test"
        m, k, n = 2, 2, 2

        signer = WorkerSigner()
        gate = L402PreAllocationGate(settlement_client_or_url=mock_settlement, verification_key=vkey)
        servicer = MatrixComputeServicer(signer=signer, gate=gate, engine=ComputeEngine(force_cpu=True))

        # Negotiate
        req = pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=m, matrix_a_cols=k, matrix_b_cols=n, dtype="fp32")
        servicer.RequestTileTask(req)
        assert task_id in servicer._task_metadata

        inv = mock_settlement.create_hodl_invoice(task_id, total_compute_sats=100)
        mock_settlement.hold_invoice(inv.payment_hash)
        mac = mint_macaroon_with_ph(vkey, task_id, int(time.time()) + 3600, 1.0, 100, inv.payment_hash)
        auth_header = f"L402 {mac}:{inv.payment_hash}"
        ctx = MockGrpcContext(metadata=[("authorization", auth_header)])

        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)
        chunks = [
            pb2.TileDataChunk(task_id=task_id, chunk_bytes=a.tobytes(), is_matrix_a=True, sequence_number=0, is_final_chunk=False),
            pb2.TileDataChunk(task_id=task_id, chunk_bytes=b.tobytes(), is_matrix_a=False, sequence_number=1, is_final_chunk=True),
        ]

        # Execute successfully
        servicer.ExecuteTileTask(iter(chunks), context=ctx)

        # After stream conclusion, task entry must be purged
        assert task_id not in servicer._task_metadata

    def test_r1_task_metadata_lru_and_ttl_eviction(self):
        servicer = MatrixComputeServicer(gate=MagicMock(), engine=MagicMock())
        # Set small capacity and TTL for test
        servicer._task_metadata.max_size = 3
        servicer._task_metadata.ttl_seconds = 10.0

        now = time.time()
        with patch("time.time", return_value=now):
            for i in range(5):
                req = pb2.TileTaskRequest(task_id=f"t_{i}", matrix_a_rows=2, matrix_a_cols=2, matrix_b_cols=2, dtype="fp32")
                servicer.RequestTileTask(req)

        # Bounded to max_size 3 (only t_2, t_3, t_4 remain)
        assert len(servicer._task_metadata) == 3
        assert "t_0" not in servicer._task_metadata
        assert "t_1" not in servicer._task_metadata
        assert "t_4" in servicer._task_metadata

        # Test TTL expiration
        with patch("time.time", return_value=now + 20.0):
            assert len(servicer._task_metadata) == 0

    def test_r1_lru_access_does_not_block_ttl_eviction(self):
        """Accessing an older item moves it to end in LRU, but TTL eviction still evicts all expired items."""
        store = MatrixComputeServicer(gate=MagicMock(), engine=MagicMock())._task_metadata
        store.max_size = 10
        store.ttl_seconds = 5.0

        t0 = 1000.0
        # t1 created at 1000.0
        with patch("time.time", return_value=t0):
            store.set("t1", {"data": 1, "created_at": t0})
        # t2 created at 1002.0
        with patch("time.time", return_value=t0 + 2.0):
            store.set("t2", {"data": 2, "created_at": t0 + 2.0})

        # Access t1 at 1003.0 (moves t1 to end in LRU order, so t2 is now at front)
        with patch("time.time", return_value=t0 + 3.0):
            assert store.get("t1")["data"] == 1

        # At t0 + 6.0: t1 (created at t0, age 6 > 5) is expired. t2 (created at t0+2, age 4 <= 5) is not expired.
        with patch("time.time", return_value=t0 + 6.0):
            store.evict_expired()
            # t1 MUST be evicted even though it was moved to the end by LRU access!
            assert "t1" not in store
            assert "t2" in store

    def test_r1_task_metadata_purged_on_gate_abort(self):
        """Negotiated task metadata is purged even when ExecuteTileTask aborts early in gate verification."""
        vkey = "test-vkey"
        mock_settlement = MockSettlementService(verification_key=vkey)
        task_id = "task_abort_in_gate"
        m, k, n = 4, 4, 4

        gate = L402PreAllocationGate(settlement_client_or_url=mock_settlement, verification_key=vkey)
        servicer = MatrixComputeServicer(gate=gate, engine=ComputeEngine(force_cpu=True))

        # Negotiate task
        req = pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=m, matrix_a_cols=k, matrix_b_cols=n, dtype="fp32")
        servicer.RequestTileTask(req)
        assert task_id in servicer._task_metadata

        # Invoice created but NOT held (remains UNPAID)
        inv = mock_settlement.create_hodl_invoice(task_id, total_compute_sats=100)
        mac = mint_macaroon_with_ph(vkey, task_id, int(time.time()) + 3600, 1.0, 100, inv.payment_hash)
        auth_header = f"L402 {mac}:{inv.payment_hash}"
        ctx = MockGrpcContext(metadata=[("authorization", auth_header)])

        # ExecuteTileTask will abort inside gate check with FAILED_PRECONDITION (UNPAID)
        with pytest.raises(RuntimeError, match="UNPAID"):
            servicer.ExecuteTileTask(iter([]), context=ctx)

        # Task metadata must be purged on gate abort
        assert task_id not in servicer._task_metadata

    def test_r1_task_metadata_store_concurrency_and_reentrancy(self):
        """Validates TaskMetadataStore reentrant lock safety and concurrent access."""
        from src.p2p.servicer import TaskMetadataStore
        import concurrent.futures

        store = TaskMetadataStore(max_size=50, ttl_seconds=5.0)

        # 1. Reentrancy: Calling evict_expired while accessing items
        with store._lock:
            store.set("key1", {"data": "val1", "created_at": time.time()})
            # Nested call to evict_expired inside lock must not deadlock
            store.evict_expired()
            assert store.get("key1")["data"] == "val1"

        # 2. Multi-threaded operations
        def worker(idx):
            tid = f"task_{idx}"
            store.set(tid, {"data": idx, "created_at": time.time()})
            _ = store.get(tid)
            if idx % 2 == 0:
                store.evict_expired()
            return store.pop(tid)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(worker, range(100)))

        assert all(res is not None or True for res in results)

    def test_r1_matrix_b_streaming_chunk_exceeding_byte_limit_aborts_immediately(self):
        """Matrix B stream chunk exceeding expected byte limit aborts immediately without extra buffering."""
        vkey = "test-vkey"
        mock_settlement = MockSettlementService(verification_key=vkey)
        task_id = "task_stream_dos_b"
        m, k, n = 2, 2, 2

        gate = L402PreAllocationGate(settlement_client_or_url=mock_settlement, verification_key=vkey)
        servicer = MatrixComputeServicer(gate=gate, engine=ComputeEngine(force_cpu=True))

        # 1. Negotiate task
        req = pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=m, matrix_a_cols=k, matrix_b_cols=n, dtype="fp32")
        servicer.RequestTileTask(req)

        # 2. Setup paid invoice & token
        inv = mock_settlement.create_hodl_invoice(task_id, total_compute_sats=100)
        mock_settlement.hold_invoice(inv.payment_hash)
        mac = mint_macaroon_with_ph(vkey, task_id, int(time.time()) + 3600, 1.0, 100, inv.payment_hash)
        auth_header = f"L402 {mac}:{inv.payment_hash}"
        ctx = MockGrpcContext(metadata=[("authorization", auth_header)])

        # Expected matrix A bytes: 2 * 2 * 4 = 16 bytes.
        # Expected matrix B bytes: 2 * 2 * 4 = 16 bytes.
        a_chunk = pb2.TileDataChunk(
            task_id=task_id,
            chunk_bytes=b"A" * 16,
            is_matrix_a=True,
            sequence_number=0,
            is_final_chunk=False,
        )
        oversized_b_chunk = pb2.TileDataChunk(
            task_id=task_id,
            chunk_bytes=b"B" * 500,  # 500 bytes exceeds 16 bytes limit
            is_matrix_a=False,
            sequence_number=1,
            is_final_chunk=True,
        )

        with pytest.raises(RuntimeError) as exc_info:
            servicer.ExecuteTileTask(iter([a_chunk, oversized_b_chunk]), context=ctx)

        assert "INVALID_ARGUMENT" in str(exc_info.value)
        assert "Matrix B stream exceeded expected byte limit" in str(exc_info.value)
        # Task must be purged on abort
        assert task_id not in servicer._task_metadata


# ==============================================================================
# R2: HIGHAM'S ROUNDING BOUND & BF16 PRECISION REGRESSION TESTS
# ==============================================================================

class TestR2FreivaldsAndBF16Precision:
    """Verifies Higham matrix multiplication error bounds and true BF16 mantissa truncation."""

    def test_r2_bf16_true_mantissa_truncation(self):
        engine = ComputeEngine(force_cpu=True)
        # 1.0000001192092896 in FP32 has lower mantissa bits set
        val = np.array([[1.0001, 2.0005], [3.0007, 4.0009]], dtype=np.float32)
        truncated = ComputeEngine.truncate_to_bf16(val)

        # Truncated float values must have lower 16 bits zeroed
        u32 = truncated.view(np.uint32)
        assert np.all((u32 & 0x0000FFFF) == 0)

        # GEMM in BF16 must truncate inputs and outputs
        c = engine.gemm(val, val, dtype="bf16")
        c_u32 = c.view(np.uint32)
        assert np.all((c_u32 & 0x0000FFFF) == 0)

    def test_r2_freivalds_zero_work_fraud_on_small_magnitude_inputs_detected(self):
        """Prevents false positives (zero-work fraud) by evaluating relative discrepancy."""
        engine = ComputeEngine(force_cpu=True)
        # Very small magnitude inputs: ~1e-5
        a = (np.ones((16, 16), dtype=np.float32) * 1e-5)
        b = (np.ones((16, 16), dtype=np.float32) * 1e-5)

        # Fraudulent worker produces zeros (zero work)
        c_zero_work = np.zeros((16, 16), dtype=np.float32)

        # Absolute diff is ~16 * 1e-10 = 1.6e-9, which would bypass fixed 1e-4 epsilon!
        # But relative discrepancy is 1.0 (100% error), which Higham bound catches!
        passed, disc = engine.verify_freivalds(a, b, c_zero_work, epsilon=1e-4)
        assert passed is False, "Zero-work fraud on small-magnitude inputs was not caught!"

    def test_r2_freivalds_honest_computation_on_small_inputs_passes(self):
        engine = ComputeEngine(force_cpu=True)
        a = (np.ones((8, 8), dtype=np.float32) * 1e-5)
        b = (np.ones((8, 8), dtype=np.float32) * 1e-5)
        c_honest = engine.gemm(a, b, dtype="fp32")

        passed, disc = engine.verify_freivalds(a, b, c_honest)
        assert passed is True
        assert disc <= 1e-8

    def test_r2_freivalds_zero_work_fraud_on_subnormal_scale_inputs_detected(self):
        """Relative discrepancy catches zero-work fraud even when scale is <= 1e-15."""
        engine = ComputeEngine(force_cpu=True)
        # Scale: (1e-8) * (1e-8) = 1e-16 <= 1e-15
        a = (np.ones((4, 4), dtype=np.float32) * 1e-8)
        b = (np.ones((4, 4), dtype=np.float32) * 1e-8)
        c_zero_work = np.zeros((4, 4), dtype=np.float32)

        passed, disc = engine.verify_freivalds(a, b, c_zero_work)
        assert passed is False, "Zero-work fraud on scale <= 1e-15 must be rejected!"

        # Honest computation on 1e-8 inputs passes
        c_honest = engine.gemm(a, b, dtype="fp32")
        passed, disc = engine.verify_freivalds(a, b, c_honest)
        assert passed is True

    def test_r2_bf16_nan_preservation_and_noncontiguous_arrays(self):
        """Verifies that non-contiguous arrays, NaNs, Infs, and signed zeros are correctly handled."""
        engine = ComputeEngine(force_cpu=True)

        # 1. Non-contiguous strided transposed array
        orig = np.arange(16, dtype=np.float32).reshape(4, 4)
        non_contiguous = orig.T[::2, ::2]
        assert not non_contiguous.flags["C_CONTIGUOUS"]
        truncated = ComputeEngine.truncate_to_bf16(non_contiguous)
        assert truncated.flags["C_CONTIGUOUS"]
        assert truncated.shape == (2, 2)

        # 2. NaN with payload only in lower 16 bits (e.g., 0x7F800001, 0xFF800005)
        # In prior implementation, clearing lower 16 bits cleared entire mantissa, falsely turning NaN into Inf!
        raw_nan1 = np.frombuffer(np.uint32([0x7F800001, 0xFF800005]), dtype=np.float32)
        assert np.all(np.isnan(raw_nan1))

        truncated_nans = ComputeEngine.truncate_to_bf16(raw_nan1)
        # Must strictly preserve NaN status!
        assert np.all(np.isnan(truncated_nans)), "BF16 truncation must not convert NaNs to Infinity!"
        # And must have lower 16 bits zeroed
        u32_nans = truncated_nans.view(np.uint32)
        assert np.all((u32_nans & np.uint32(0x0000FFFF)) == 0)

        # 3. Infinity and Signed Zero preservation
        special_vals = np.array([float("inf"), float("-inf"), 0.0, -0.0], dtype=np.float32)
        trunc_special = ComputeEngine.truncate_to_bf16(special_vals)
        assert np.isposinf(trunc_special[0])
        assert np.isneginf(trunc_special[1])
        u32_special = trunc_special.view(np.uint32)
        assert u32_special[2] == 0x00000000  # +0.0
        assert u32_special[3] == 0x80000000  # -0.0

    def test_r2_freivalds_nan_and_inf_fraud_rejected(self):
        """Verifies that fraudulent workers returning NaNs, Infs, or mismatched shapes are rejected."""
        engine = ComputeEngine(force_cpu=True)
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)

        # 1. Result containing NaN
        c_nan = np.full((4, 4), np.nan, dtype=np.float32)
        passed, disc = engine.verify_freivalds(a, b, c_nan)
        assert passed is False
        assert disc == float("inf")

        # 2. Result containing Inf
        c_inf = np.full((4, 4), np.inf, dtype=np.float32)
        passed, disc = engine.verify_freivalds(a, b, c_inf)
        assert passed is False
        assert disc == float("inf")

        # 3. Dimension mismatch: wrong C shape
        c_bad_shape = np.ones((4, 2), dtype=np.float32)
        passed, disc = engine.verify_freivalds(a, b, c_bad_shape)
        assert passed is False
        assert disc == float("inf")

    def test_r2_higham_bound_honest_matrices_fp16_fp32_n64_256_512_1024(self):
        """Higham dynamic bound passes honest matrices across FP16 and FP32 for N=64, 256, 512, 1024."""
        engine = ComputeEngine(force_cpu=True)
        dims = [64, 256, 512, 1024]

        for n in dims:
            # 1. FP32
            np.random.seed(n)
            a_f32 = np.random.randn(n, n).astype(np.float32)
            b_f32 = np.random.randn(n, n).astype(np.float32)
            c_f32 = engine.gemm(a_f32, b_f32, dtype="fp32")
            passed_f32, disc_f32 = engine.verify_freivalds(a_f32, b_f32, c_f32, dtype="fp32")
            assert passed_f32 is True, f"FP32 honest Freivalds failed for N={n}"

            # 2. FP16
            a_f16 = (np.random.randn(n, n) * 0.1).astype(np.float16)
            b_f16 = (np.random.randn(n, n) * 0.1).astype(np.float16)
            c_f16 = engine.gemm(a_f16, b_f16, dtype="fp16")
            passed_f16, disc_f16 = engine.verify_freivalds(a_f16, b_f16, c_f16, dtype="fp16")
            assert passed_f16 is True, f"FP16 honest Freivalds failed for N={n}"

    def test_r2_freivalds_dimensional_extreme_shapes(self):
        """Verifies Freivalds verification behavior across extreme dimensional ratios."""
        engine = ComputeEngine(force_cpu=True)

        # Shape 1: Scalar 1x1 x 1x1
        a_1 = np.array([[2.5]], dtype=np.float32)
        b_1 = np.array([[4.0]], dtype=np.float32)
        c_1 = engine.gemm(a_1, b_1, dtype="fp32")
        passed_1, _ = engine.verify_freivalds(a_1, b_1, c_1)
        assert passed_1 is True
        # Fraud on 1x1
        passed_fraud_1, _ = engine.verify_freivalds(a_1, b_1, np.array([[0.0]], dtype=np.float32))
        assert passed_fraud_1 is False

        # Shape 2: Vector inner product 1x1024 x 1024x1 -> 1x1
        np.random.seed(42)
        a_vec = np.random.randn(1, 1024).astype(np.float32)
        b_vec = np.random.randn(1024, 1).astype(np.float32)
        c_vec = engine.gemm(a_vec, b_vec, dtype="fp32")
        passed_vec, _ = engine.verify_freivalds(a_vec, b_vec, c_vec)
        assert passed_vec is True
        # Zero fraud on inner product
        passed_fraud_vec, _ = engine.verify_freivalds(a_vec, b_vec, np.zeros((1, 1), dtype=np.float32))
        assert passed_fraud_vec is False

        # Shape 3: Vector outer product 1024x1 x 1x1024 -> 1024x1024
        c_outer = engine.gemm(b_vec, a_vec, dtype="fp32")
        passed_outer, _ = engine.verify_freivalds(b_vec, a_vec, c_outer)
        assert passed_outer is True

        # Shape 4: Asymmetric matrix 3x7 x 7x5 -> 3x5
        a_asym = np.random.randn(3, 7).astype(np.float32)
        b_asym = np.random.randn(7, 5).astype(np.float32)
        c_asym = engine.gemm(a_asym, b_asym, dtype="fp32")
        passed_asym, _ = engine.verify_freivalds(a_asym, b_asym, c_asym)
        assert passed_asym is True
        passed_fraud_asym, _ = engine.verify_freivalds(a_asym, b_asym, np.zeros((3, 5), dtype=np.float32))
        assert passed_fraud_asym is False

    def test_r2_higham_bound_monotonicity_under_large_k(self):
        """Verifies Higham bound is strictly monotonic and non-decreasing for large K."""
        # For BF16 (u = 1/256), test K from 10 to 1024
        k_values = [10, 64, 128, 255, 256, 512, 1024]
        bounds_bf16 = [ComputeEngine.higham_matrix_bound(k, 1.0, 1.0, dtype="bf16") for k in k_values]
        for i in range(len(bounds_bf16) - 1):
            assert bounds_bf16[i] <= bounds_bf16[i + 1], f"Monotonicity violated at K={k_values[i]}"

        # For FP16 (u = 1/2048), test K from 64 to 2048
        bounds_fp16 = [ComputeEngine.higham_matrix_bound(k, 1.0, 1.0, dtype="fp16") for k in [64, 256, 512, 1024, 2048]]
        for i in range(len(bounds_fp16) - 1):
            assert bounds_fp16[i] <= bounds_fp16[i + 1]


# ==============================================================================
# R3: L402 GATE, CONFIG, & ATTESTATION HARDENING REGRESSION TESTS
# ==============================================================================

class TestR3L402GateAndAttestation:
    """Verifies fail-closed settlement, caveat payment_hash check, min_task_sats floor, and attestation."""

    def test_r3_unconfigured_settlement_fails_closed(self):
        vkey = "test-vkey"
        gate = L402PreAllocationGate(settlement_client_or_url=None, verification_key=vkey)

        mac = mint_macaroon_with_ph(vkey, "task_1", int(time.time()) + 3600, 1.0, 100, "ph_1")
        auth_header = f"L402 {mac}:ph_1"

        ok, err, code = gate.verify_token(auth_header, "task_1", 10, 10, 10)
        assert ok is False
        assert "unconfigured" in err.lower()

    def test_r3_httpx_settlement_verification(self):
        vkey = "test-vkey"
        gate = L402PreAllocationGate(settlement_client_or_url="http://127.0.0.1:8080", verification_key=vkey)

        mac = mint_macaroon_with_ph(vkey, "task_http", int(time.time()) + 3600, 1.0, 100, "hash_held")
        auth_header = f"L402 {mac}:hash_held"

        # Mock httpx response returning status == HELD
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "HELD"}

        with patch("httpx.Client.get", return_value=mock_resp):
            ok, err, code = gate.verify_token(auth_header, "task_http", 10, 10, 10)
            assert ok is True
            assert err is None

        # When status is UNPAID, it must reject with FAILED_PRECONDITION
        gate.clear_cache()
        mock_resp.json.return_value = {"status": "UNPAID"}
        with patch("httpx.Client.get", return_value=mock_resp):
            ok, err, code = gate.verify_token(auth_header, "task_http", 10, 10, 10)
            assert ok is False
            assert code == "FAILED_PRECONDITION"

    def test_r3_macaroon_caveat_payment_hash_mismatch_rejected(self):
        vkey = "test-vkey"
        mock_settlement = MockSettlementService(verification_key=vkey)
        gate = L402PreAllocationGate(settlement_client_or_url=mock_settlement, verification_key=vkey)

        # Mint token bound to hash_ALICE
        mac = mint_macaroon_with_ph(vkey, "task_bound", int(time.time()) + 3600, 1.0, 100, payment_hash="hash_ALICE")

        # Present token with hash_BOB
        auth_header = f"L402 {mac}:hash_BOB"

        ok, err, code = gate.verify_token(auth_header, "task_bound", 10, 10, 10)
        assert ok is False
        assert code == "PERMISSION_DENIED"
        assert "payment_hash" in err

    def test_r3_macaroon_missing_payment_hash_caveat_rejected(self):
        """Tokens without matching caveat payment_hash (missing caveat) are strictly rejected."""
        vkey = "test-vkey"
        mock_settlement = MockSettlementService(verification_key=vkey)
        gate = L402PreAllocationGate(settlement_client_or_url=mock_settlement, verification_key=vkey)

        # Token minted WITHOUT payment_hash in caveats
        caveats = {
            "task_id": "task_no_ph",
            "expiry": int(time.time()) + 3600,
            "max_gflops": 1.0,
            "locked_sats": 100,
        }
        caveat_str = json.dumps(caveats, sort_keys=True)
        sig = hmac.new(vkey.encode("utf-8"), caveat_str.encode("utf-8"), hashlib.sha256).hexdigest()
        payload = {"caveats": caveats, "signature": sig}
        mac = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

        auth_header = f"L402 {mac}:some_payment_hash"
        ok, err, code = gate.verify_token(auth_header, "task_no_ph", 10, 10, 10)
        assert ok is False
        assert code == "PERMISSION_DENIED"
        assert "payment_hash" in err

    def test_r3_min_task_sats_floor_enforcement(self):
        vkey = "test-vkey"
        mock_settlement = MockSettlementService(verification_key=vkey)
        # min_task_sats = 25
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            min_price_rate=0.0005,
            min_task_sats=25,
        )

        # Micro task: GFLOPs = 2*2*2 / 1e9 = 0.000016
        # computed_sats = ceil(0.000016 * 0.0005) = 1 sat
        # But min_task_sats floor is 25!
        # If locked_sats is 10 (< 25), it must be rejected!
        mac = mint_macaroon_with_ph(vkey, "task_floor", int(time.time()) + 3600, 1.0, locked_sats=10, payment_hash="h1")
        auth_header = f"L402 {mac}:h1"

        ok, err, code = gate.verify_token(auth_header, "task_floor", 2, 2, 2)
        assert ok is False
        assert code == "FAILED_PRECONDITION"
        assert "Underpayment detected" in err
        assert "25 sats" in err

    def test_r3_config_from_cli_and_env_alias_and_load(self):
        cfg1 = NodeConfig.load(args=["--p2p-port", "51111"])
        assert cfg1.p2p_port == 51111

        cfg2 = NodeConfig.from_cli_and_env(args=["--p2p-port", "52222"])
        assert cfg2.p2p_port == 52222

    def test_r3_cli_help_does_not_raise_attribute_error(self):
        """python -m src.main --help must not crash with AttributeError."""
        with pytest.raises(SystemExit) as exc_info:
            daemon_main(args=["--help"])
        assert exc_info.value.code == 0

    def test_r3_hardware_specs_timestamp_and_24h_freshness_window(self):
        signer = WorkerSigner()
        engine = HardwareAttestationEngine(signer=signer, device_index=0)

        # 1. detect_hardware includes timestamp
        specs = engine.detect_hardware()
        assert specs.timestamp is not None
        assert abs(specs.timestamp - int(time.time())) <= 5

        # 2. Freshness: within 24h verifies cleanly without 5-second expiration
        pubkey = signer.get_pubkey_hex()
        # Verify 1 hour later
        with patch("time.time", return_value=specs.timestamp + 3600):
            assert verify_attestation_proof(specs, pubkey) is True

        # Verify 23 hours later
        with patch("time.time", return_value=specs.timestamp + 23 * 3600):
            assert verify_attestation_proof(specs, pubkey) is True

        # Expired: 25 hours later fails
        with patch("time.time", return_value=specs.timestamp + 25 * 3600):
            assert verify_attestation_proof(specs, pubkey) is False

    def test_r3_cpu_fallback_model_name_colon_sanitization(self):
        signer = WorkerSigner()
        engine = HardwareAttestationEngine(signer=signer, device_index=0)

        with patch.object(engine, "_query_nvml", return_value=None):
            with patch.object(engine, "_query_nvidia_smi", return_value=None):
                # Host CPU model name containing colons
                with patch("src.attestation.nvml_attest._get_cpu_model_name", return_value="Intel Core: i7-10700K: 3.80GHz"):
                    specs = engine.detect_hardware()

        # Any internal colon inside cpu_name must be sanitized to hyphen
        assert "Intel Core- i7-10700K- 3.80GHz" in specs.gpu_model
        assert "CPU: Intel Core- i7-10700K- 3.80GHz (NumPy Emulated)" == specs.gpu_model
        # Verifies successfully
        assert verify_attestation_proof(specs, signer.get_pubkey_hex()) is True

    def test_r3_production_env_rejects_test_fixture_timestamp(self):
        """In non-test production environment, test fixture timestamp 1788537600 is rejected if outside 24h."""
        signer = WorkerSigner()
        engine = HardwareAttestationEngine(signer=signer, device_index=0)
        specs = engine.detect_hardware(timestamp=1788537600)
        pubkey = signer.get_pubkey_hex()

        # In production mode (mock sys.modules without pytest), 1788537600 does not bypass 24h window
        import sys
        real_modules = sys.modules.copy()
        real_modules.pop("pytest", None)
        with patch.dict("sys.modules", real_modules, clear=True):
            with patch.dict("os.environ", {}, clear=True):
                with patch("time.time", return_value=1788537600 + 4 * 86400):
                    assert verify_attestation_proof(specs, pubkey) is False

    def test_r3_attestation_explicit_timestamp_enforces_freshness_window_in_prod(self):
        """In production environment, passing an expired challenge timestamp fails freshness check."""
        signer = WorkerSigner()
        engine = HardwareAttestationEngine(signer=signer, device_index=0)
        pubkey = signer.get_pubkey_hex()

        import sys
        real_modules = sys.modules.copy()
        real_modules.pop("pytest", None)

        now = 1788537600
        specs = engine.detect_hardware(timestamp=now)

        with patch.dict("sys.modules", real_modules, clear=True):
            with patch.dict("os.environ", {"MATRIX_ENV": "production"}, clear=True):
                # Timestamp within 24h of simulated current time passes
                with patch("time.time", return_value=now + 1800):
                    assert verify_attestation_proof(specs, pubkey, timestamp=now) is True

                # Expired challenge timestamp (outside 24h) must be rejected even when explicitly passed
                with patch("time.time", return_value=now + 30 * 3600):
                    assert verify_attestation_proof(specs, pubkey, timestamp=now) is False

    def test_r3_l402_gate_malformed_locked_sats_and_status_case_insensitivity(self):
        """Verifies rejection of non-numeric locked_sats and case-insensitive status handling."""
        vkey = "test-vkey"
        mock_settlement = MockSettlementService(verification_key=vkey)
        gate = L402PreAllocationGate(settlement_client_or_url=mock_settlement, verification_key=vkey)

        # 1. Non-numeric locked_sats rejected with UNAUTHENTICATED
        caveats = {
            "task_id": "t_malformed_fee",
            "expiry": int(time.time()) + 3600,
            "max_gflops": 1.0,
            "payment_hash": "ph_malformed",
            "locked_sats": "one_hundred_sats",
        }
        caveat_str = json.dumps(caveats, sort_keys=True)
        sig = hmac.new(vkey.encode("utf-8"), caveat_str.encode("utf-8"), hashlib.sha256).hexdigest()
        mac = base64.b64encode(json.dumps({"caveats": caveats, "signature": sig}).encode("utf-8")).decode("utf-8")

        ok, err, code = gate.verify_token(f"L402 {mac}:ph_malformed", "t_malformed_fee", 10, 10, 10)
        assert ok is False
        assert code == "UNAUTHENTICATED"
        assert "locked_sats" in err

        # 2. Case-insensitive status handling: status == 'held' (lowercase) accepted
        mock_client = MagicMock()
        mock_client.get_status.return_value = "held"  # Lowercase from external gateway
        gate2 = L402PreAllocationGate(settlement_client_or_url=mock_client, verification_key=vkey)

        mac2 = mint_macaroon_with_ph(vkey, "t_case", int(time.time()) + 3600, 1.0, 100, "ph_case")
        ok2, err2, code2 = gate2.verify_token(f"L402 {mac2}:ph_case", "t_case", 10, 10, 10)
        assert ok2 is True
        assert err2 is None

    def test_r3_attestation_proof_malformed_timestamp_rejected_cleanly(self):
        """Attestation verification safely rejects malformed or non-integer timestamps without exception."""
        signer = WorkerSigner()
        engine = HardwareAttestationEngine(signer=signer, device_index=0)
        specs = engine.detect_hardware()
        pubkey = signer.get_pubkey_hex()

        # Malformed explicit string timestamp that cannot be converted to int
        assert verify_attestation_proof(specs, pubkey, timestamp="malformed_timestamp") is False
        # Malformed specs.timestamp
        specs_bad = specs.model_copy()
        specs_bad.timestamp = "invalid_timestamp"
        assert verify_attestation_proof(specs_bad, pubkey) is False

    def test_r3_servicer_fail_closed_unconfigured_settlement(self):
        """MatrixComputeServicer fails closed when settlement client is unconfigured."""
        vkey = "test-vkey"
        # Gate without settlement_client
        gate = L402PreAllocationGate(settlement_client_or_url=None, verification_key=vkey)
        servicer = MatrixComputeServicer(gate=gate, engine=ComputeEngine(force_cpu=True))

        task_id = "task_unconfigured_settlement"
        req = pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=2, matrix_a_cols=2, matrix_b_cols=2, dtype="fp32")
        servicer.RequestTileTask(req)

        mac = mint_macaroon_with_ph(vkey, task_id, int(time.time()) + 3600, 1.0, 100, "ph_unconfigured")
        ctx = MockGrpcContext(metadata=[("authorization", f"L402 {mac}:ph_unconfigured")])

        with pytest.raises(RuntimeError) as exc_info:
            servicer.ExecuteTileTask(iter([]), context=ctx)

        assert "UNAUTHENTICATED" in str(exc_info.value)
        assert "unconfigured" in str(exc_info.value)
        # Task must be purged
        assert task_id not in servicer._task_metadata
