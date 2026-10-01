"""
Adversarial Challenge & Stress-Test Suite for Milestone 3:
`matrix-compute-node` Enclave Compute Execution & Servicer Integration.

Challenger: challenger_m3_1 (role: teamwork_preview_challenger)

Empirically tests four adversarial challenge vectors against `EnclaveComputeCore` and `MatrixComputeServicer`:
1. Chunk injection and tampering attacks (1-bit ciphertext corruption, IV corruption, tag corruption, AAD alteration).
2. Chunk ordering, duplication, truncation, and buffer overflow attacks.
3. Enclave memory boundary isolation (plaintext float32 leakage prevention, zeroization, security policy enforcement).
4. Concurrency stress, multi-threaded session isolation, and key/buffer cross-contamination prevention.
"""

from __future__ import annotations

import base64
import collections
import concurrent.futures
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519
import grpc
import numpy as np
import pytest

from src.attestation.simulated import SimulatedAttestationProvider
from src.auth.l402_gate import L402PreAllocationGate
from src.crypto.signer import WorkerSigner
from src.engine.compute_core import ComputeEngine
from src.engine.enclave_compute import (
    EnclaveComputeCore,
    EnclaveDecryptionError,
    EnclaveError,
    EnclaveExecutionError,
    EnclaveFramingError,
    EnclaveKeyExchangeError,
    EnclaveSecurityPolicyError,
    EnclaveSession,
    SessionKeys,
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    canonical_chunk_aad,
    canonical_result_aad,
    derive_session_keys,
)
from src.p2p.servicer import MatrixComputeServicer
from src.test_doubles.mock_settlement import MockSettlementService

try:
    from matrix_contracts.v1 import pb2
except ImportError:
    from matrix_contracts.proto import compute_service_pb2 as pb2


# ==============================================================================
# TEST FIXTURES & HELPERS
# ==============================================================================

def compute_shannon_entropy(data: bytes) -> float:
    """Calculates Shannon entropy in bits per byte (0.0 to 8.0)."""
    if not data:
        return 0.0
    counts = collections.Counter(data)
    total = len(data)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


class MockServicerGrpcContext:
    """Mock gRPC context capturing abort status code and details."""

    def __init__(self, metadata: Optional[List[Tuple[str, str]]] = None):
        self._metadata = metadata or []
        self.code: Optional[grpc.StatusCode] = None
        self.details: Optional[str] = None

    def invocation_metadata(self) -> List[Tuple[str, str]]:
        return self._metadata

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details
        raise RuntimeError(f"ABORT: {code}: {details}")


def create_mock_servicer_and_auth(
    task_id: str,
    m: int = 4,
    k: int = 4,
    n: int = 4,
    dtype: str = "fp32",
    require_confidential: bool = True,
    vkey: str = "adversarial-vkey-m3",
) -> Tuple[MatrixComputeServicer, str, SessionKeys, x25519.X25519PrivateKey, bytes, pb2.TileTaskAck]:
    """Helper to initialize MatrixComputeServicer, perform RequestTileTask, and mint valid L402 auth header."""
    signer = WorkerSigner()
    mock_settlement = MockSettlementService(verification_key=vkey)
    gate = L402PreAllocationGate(
        settlement_client_or_url=mock_settlement,
        verification_key=vkey,
        min_price_rate=0.0005,
    )
    provider = SimulatedAttestationProvider()
    engine = ComputeEngine(force_cpu=True)
    servicer = MatrixComputeServicer(
        signer=signer,
        gate=gate,
        engine=engine,
        attestation_provider=provider,
    )

    c_priv = x25519.X25519PrivateKey.generate()
    c_pub = c_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    nonce = secrets.token_bytes(32)

    req = pb2.TileTaskRequest(
        task_id=task_id,
        matrix_a_rows=m,
        matrix_a_cols=k,
        matrix_b_cols=n,
        dtype=dtype,
        require_confidential_compute=require_confidential,
        client_ephemeral_pubkey=c_pub if require_confidential else b"",
        attestation_nonce=nonce if require_confidential else b"",
        required_tee_type="simulated-cca" if require_confidential else "",
    )
    ack = servicer.RequestTileTask(req)

    c_keys = None
    if require_confidential:
        c_keys = derive_session_keys(c_priv, ack.worker_ephemeral_pubkey, nonce)

    # Mint L402 token
    inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
    mock_settlement.hold_invoice(inv.payment_hash)
    caveats = {
        "task_id": task_id,
        "expiry": int(time.time()) + 3600,
        "max_gflops": 1.0,
        "locked_sats": 100,
        "payment_hash": inv.payment_hash,
    }
    cav_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(vkey.encode("utf-8"), cav_str.encode("utf-8"), hashlib.sha256).hexdigest()
    mac = base64.b64encode(json.dumps({"caveats": caveats, "signature": sig}).encode("utf-8")).decode("utf-8")
    auth_header = f"L402 {mac}:{inv.payment_hash}"

    return servicer, auth_header, c_keys, c_priv, nonce, ack


# ==============================================================================
# CHALLENGE 1: CHUNK INJECTION & TAMPERING ATTACKS
# ==============================================================================

class TestChunkInjectionAndTamperingAttacks:
    """
    Adversarial attacks mutating ciphertext bits, IVs, GCM authentication tags,
    and AAD header metadata. Verifies immediate fail-closed decryption rejection.
    """

    def test_ciphertext_single_bit_flip_tampering(self) -> None:
        """Flips 1 bit at various offsets in chunk ciphertext and verifies decryption rejection."""
        k_c2w = secrets.token_bytes(32)
        k_w2c = secrets.token_bytes(32)
        session = EnclaveSession(task_id="task-tamper-ct", m=4, k=4, n=4, k_c2w=k_c2w, k_w2c=k_w2c)

        raw_data = np.ones((4, 4), dtype=np.float32).tobytes()
        aad = canonical_chunk_aad("task-tamper-ct", 0, is_matrix_a=True, is_final_chunk=False)
        ct, iv, tag = aes_gcm_encrypt(k_c2w, raw_data, aad=aad)

        # Test bit flips at start, middle, and end of ciphertext
        offsets_to_test = [0, len(ct) // 2, len(ct) - 1]
        for offset in offsets_to_test:
            for bit in range(8):
                tampered_ct = bytearray(ct)
                tampered_ct[offset] ^= (1 << bit)

                chunk = pb2.TileDataChunk(
                    task_id="task-tamper-ct",
                    chunk_bytes=bytes(tampered_ct),
                    is_matrix_a=True,
                    sequence_number=0,
                    is_final_chunk=False,
                    is_encrypted=True,
                    iv=iv,
                    tag=tag,
                )
                with pytest.raises(EnclaveDecryptionError, match="AES-256-GCM authentication failed"):
                    session.ingest_chunk(chunk)

    def test_iv_corruption_and_length_tampering(self) -> None:
        """Tests corrupted IVs (bit flips) and invalid IV lengths."""
        k_c2w = secrets.token_bytes(32)
        session = EnclaveSession(task_id="task-tamper-iv", m=4, k=4, n=4, k_c2w=k_c2w, k_w2c=secrets.token_bytes(32))
        raw_data = np.ones((4, 4), dtype=np.float32).tobytes()
        aad = canonical_chunk_aad("task-tamper-iv", 0, is_matrix_a=True, is_final_chunk=False)
        ct, iv, tag = aes_gcm_encrypt(k_c2w, raw_data, aad=aad)

        # 1. Invalid IV lengths must fail with EnclaveFramingError
        invalid_iv_lengths = [0, 1, 8, 11, 13, 16, 32]
        for bad_len in invalid_iv_lengths:
            bad_iv = secrets.token_bytes(bad_len)
            chunk = pb2.TileDataChunk(
                task_id="task-tamper-iv",
                chunk_bytes=ct,
                is_matrix_a=True,
                sequence_number=0,
                is_final_chunk=False,
                is_encrypted=True,
                iv=bad_iv,
                tag=tag,
            )
            with pytest.raises(EnclaveFramingError, match="IV must be exactly 12 bytes"):
                session.ingest_chunk(chunk)

        # 2. Bit flips in 12-byte IV must fail with EnclaveDecryptionError
        for bit in [0, 3, 7]:
            tampered_iv = bytearray(iv)
            tampered_iv[0] ^= (1 << bit)
            chunk = pb2.TileDataChunk(
                task_id="task-tamper-iv",
                chunk_bytes=ct,
                is_matrix_a=True,
                sequence_number=0,
                is_final_chunk=False,
                is_encrypted=True,
                iv=bytes(tampered_iv),
                tag=tag,
            )
            with pytest.raises(EnclaveDecryptionError):
                session.ingest_chunk(chunk)

    def test_gcm_tag_corruption_and_length_tampering(self) -> None:
        """Tests corrupted authentication tags (bit flips) and invalid tag lengths."""
        k_c2w = secrets.token_bytes(32)
        session = EnclaveSession(task_id="task-tamper-tag", m=4, k=4, n=4, k_c2w=k_c2w, k_w2c=secrets.token_bytes(32))
        raw_data = np.ones((4, 4), dtype=np.float32).tobytes()
        aad = canonical_chunk_aad("task-tamper-tag", 0, is_matrix_a=True, is_final_chunk=False)
        ct, iv, tag = aes_gcm_encrypt(k_c2w, raw_data, aad=aad)

        # 1. Invalid tag lengths must fail with EnclaveFramingError
        invalid_tag_lengths = [0, 8, 12, 15, 17, 32]
        for bad_len in invalid_tag_lengths:
            bad_tag = secrets.token_bytes(bad_len)
            chunk = pb2.TileDataChunk(
                task_id="task-tamper-tag",
                chunk_bytes=ct,
                is_matrix_a=True,
                sequence_number=0,
                is_final_chunk=False,
                is_encrypted=True,
                iv=iv,
                tag=bad_tag,
            )
            with pytest.raises(EnclaveFramingError, match="Tag must be exactly 16 bytes"):
                session.ingest_chunk(chunk)

        # 2. Bit flips in 16-byte tag must fail with EnclaveDecryptionError
        for offset in [0, 8, 15]:
            tampered_tag = bytearray(tag)
            tampered_tag[offset] ^= 0x01
            chunk = pb2.TileDataChunk(
                task_id="task-tamper-tag",
                chunk_bytes=ct,
                is_matrix_a=True,
                sequence_number=0,
                is_final_chunk=False,
                is_encrypted=True,
                iv=iv,
                tag=bytes(tampered_tag),
            )
            with pytest.raises(EnclaveDecryptionError):
                session.ingest_chunk(chunk)

    def test_aad_metadata_alteration_rejection(self) -> None:
        """Verifies that altering sequence numbers, matrix flag, or task_id breaks AAD authentication."""
        k_c2w = secrets.token_bytes(32)
        session = EnclaveSession(task_id="task-aad-01", m=4, k=4, n=4, k_c2w=k_c2w, k_w2c=secrets.token_bytes(32))
        raw_data = np.ones((4, 4), dtype=np.float32).tobytes()

        # Encrypt with canonical AAD: seq=0, matrix_a=True, is_final=False
        aad_valid = canonical_chunk_aad("task-aad-01", 0, is_matrix_a=True, is_final_chunk=False)
        ct, iv, tag = aes_gcm_encrypt(k_c2w, raw_data, aad=aad_valid)

        # 1. Alter task_id in chunk header -> EnclaveFramingError
        chunk_bad_task = pb2.TileDataChunk(
            task_id="task-aad-WRONG",
            chunk_bytes=ct,
            is_matrix_a=True,
            sequence_number=0,
            is_final_chunk=False,
            is_encrypted=True,
            iv=iv,
            tag=tag,
        )
        with pytest.raises(EnclaveFramingError, match="Chunk task_id mismatch"):
            session.ingest_chunk(chunk_bad_task)

        # 2. Alter is_matrix_a (claim Matrix B when encrypted as Matrix A)
        chunk_flipped_matrix = pb2.TileDataChunk(
            task_id="task-aad-01",
            chunk_bytes=ct,
            is_matrix_a=False,  # Flipped!
            sequence_number=0,
            is_final_chunk=False,
            is_encrypted=True,
            iv=iv,
            tag=tag,
        )
        with pytest.raises(EnclaveDecryptionError):
            session.ingest_chunk(chunk_flipped_matrix)

        # 3. Alter is_final_chunk flag
        chunk_flipped_final = pb2.TileDataChunk(
            task_id="task-aad-01",
            chunk_bytes=ct,
            is_matrix_a=True,
            sequence_number=0,
            is_final_chunk=True,  # Flipped!
            is_encrypted=True,
            iv=iv,
            tag=tag,
        )
        with pytest.raises(EnclaveDecryptionError):
            session.ingest_chunk(chunk_flipped_final)

    def test_servicer_gRPC_aborts_on_tampered_chunk(self) -> None:
        """Verifies that MatrixComputeServicer translates decryption/framing errors to gRPC aborts."""
        task_id = "task-servicer-tamper-01"
        servicer, auth_header, c_keys, _, _, _ = create_mock_servicer_and_auth(task_id, m=4, k=4, n=4)

        a_data = np.ones((4, 4), dtype=np.float32).tobytes()
        aad = canonical_chunk_aad(task_id, 0, is_matrix_a=True, is_final_chunk=False)
        ct, iv, tag = aes_gcm_encrypt(c_keys.k_c2w, a_data, aad=aad)

        # Mutate 1 bit in tag
        bad_tag = bytearray(tag)
        bad_tag[0] ^= 0xFF

        chunk = pb2.TileDataChunk(
            task_id=task_id,
            chunk_bytes=ct,
            is_matrix_a=True,
            sequence_number=0,
            is_final_chunk=False,
            is_encrypted=True,
            iv=iv,
            tag=bytes(bad_tag),
        )

        ctx = MockServicerGrpcContext(metadata=[("authorization", auth_header), ("x-task-id", task_id)])
        with pytest.raises(RuntimeError, match="ABORT: StatusCode.INVALID_ARGUMENT"):
            servicer.ExecuteTileTask(iter([chunk]), context=ctx)
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT


# ==============================================================================
# CHALLENGE 2: CHUNK ORDERING, DUPLICATION & TRUNCATION ATTACKS
# ==============================================================================

class TestChunkOrderingDuplicationAndTruncationAttacks:
    """
    Adversarial attacks manipulating chunk sequence streams: out-of-order chunks,
    duplicate replay chunks, prematurely truncated streams, and buffer overflow attempts.
    """

    def test_out_of_order_chunk_rejection(self) -> None:
        """Verifies that chunks arriving out of sequence are rejected immediately."""
        k_c2w = secrets.token_bytes(32)
        session = EnclaveSession(task_id="task-ooo-01", m=4, k=4, n=4, k_c2w=k_c2w, k_w2c=secrets.token_bytes(32))

        # Ingest chunk with seq=1 when seq=0 is expected
        chunk_1 = pb2.TileDataChunk(
            task_id="task-ooo-01",
            chunk_bytes=b"cipher",
            is_matrix_a=True,
            sequence_number=1,
            is_final_chunk=False,
            is_encrypted=True,
            iv=secrets.token_bytes(12),
            tag=secrets.token_bytes(16),
        )
        with pytest.raises(EnclaveFramingError, match="Out of order chunk: expected seq 0, got 1"):
            session.ingest_chunk(chunk_1)

    def test_duplicate_chunk_replay_rejection(self) -> None:
        """Verifies that replaying an already-ingested chunk triggers sequence framing error."""
        k_c2w = secrets.token_bytes(32)
        session = EnclaveSession(task_id="task-replay-01", m=4, k=4, n=4, k_c2w=k_c2w, k_w2c=secrets.token_bytes(32))

        data = np.ones((4, 4), dtype=np.float32).tobytes()
        aad = canonical_chunk_aad("task-replay-01", 0, is_matrix_a=True, is_final_chunk=False)
        ct, iv, tag = aes_gcm_encrypt(k_c2w, data, aad=aad)

        chunk_0 = pb2.TileDataChunk(
            task_id="task-replay-01",
            chunk_bytes=ct,
            is_matrix_a=True,
            sequence_number=0,
            is_final_chunk=False,
            is_encrypted=True,
            iv=iv,
            tag=tag,
        )

        # First ingestion succeeds
        session.ingest_chunk(chunk_0)
        assert session._expected_seq == 1

        # Replaying chunk_0 must be rejected
        with pytest.raises(EnclaveFramingError, match="Out of order chunk: expected seq 1, got 0"):
            session.ingest_chunk(chunk_0)

    def test_missing_final_chunk_flag_truncation(self) -> None:
        """Verifies that stream ending without is_final_chunk=True is rejected upon execution."""
        k_c2w = secrets.token_bytes(32)
        k_w2c = secrets.token_bytes(32)
        session = EnclaveSession(task_id="task-no-final", m=2, k=2, n=2, k_c2w=k_c2w, k_w2c=k_w2c)

        a_data = np.ones((2, 2), dtype=np.float32).tobytes()
        b_data = np.ones((2, 2), dtype=np.float32).tobytes()

        # Both chunks sent with is_final_chunk=False
        aad_a = canonical_chunk_aad("task-no-final", 0, is_matrix_a=True, is_final_chunk=False)
        ct_a, iv_a, tag_a = aes_gcm_encrypt(k_c2w, a_data, aad=aad_a)
        session.ingest_chunk(pb2.TileDataChunk(
            task_id="task-no-final", chunk_bytes=ct_a, is_matrix_a=True, sequence_number=0,
            is_final_chunk=False, is_encrypted=True, iv=iv_a, tag=tag_a
        ))

        aad_b = canonical_chunk_aad("task-no-final", 1, is_matrix_a=False, is_final_chunk=False)
        ct_b, iv_b, tag_b = aes_gcm_encrypt(k_c2w, b_data, aad=aad_b)
        session.ingest_chunk(pb2.TileDataChunk(
            task_id="task-no-final", chunk_bytes=ct_b, is_matrix_a=False, sequence_number=1,
            is_final_chunk=False, is_encrypted=True, iv=iv_b, tag=tag_b
        ))

        with pytest.raises(EnclaveFramingError, match="Stream terminated without final chunk flag"):
            session.execute_and_encrypt()

    def test_byte_count_underflow_truncation(self) -> None:
        """Verifies that a truncated payload (bytes < expected) fails execution even if final flag is set."""
        k_c2w = secrets.token_bytes(32)
        k_w2c = secrets.token_bytes(32)
        # Expected bytes for 4x4 float32: 64 bytes
        session = EnclaveSession(task_id="task-underflow", m=4, k=4, n=4, k_c2w=k_c2w, k_w2c=k_w2c)

        # Ingest only 32 bytes for A (truncated)
        partial_a = np.ones((2, 4), dtype=np.float32).tobytes()
        aad_a = canonical_chunk_aad("task-underflow", 0, is_matrix_a=True, is_final_chunk=False)
        ct_a, iv_a, tag_a = aes_gcm_encrypt(k_c2w, partial_a, aad=aad_a)
        session.ingest_chunk(pb2.TileDataChunk(
            task_id="task-underflow", chunk_bytes=ct_a, is_matrix_a=True, sequence_number=0,
            is_final_chunk=False, is_encrypted=True, iv=iv_a, tag=tag_a
        ))

        # Full B with final chunk flag
        full_b = np.ones((4, 4), dtype=np.float32).tobytes()
        aad_b = canonical_chunk_aad("task-underflow", 1, is_matrix_a=False, is_final_chunk=True)
        ct_b, iv_b, tag_b = aes_gcm_encrypt(k_c2w, full_b, aad=aad_b)
        session.ingest_chunk(pb2.TileDataChunk(
            task_id="task-underflow", chunk_bytes=ct_b, is_matrix_a=False, sequence_number=1,
            is_final_chunk=True, is_encrypted=True, iv=iv_b, tag=tag_b
        ))

        with pytest.raises(EnclaveFramingError, match="Matrix A byte count mismatch"):
            session.execute_and_encrypt()

    def test_buffer_overflow_chunk_injection_rejected(self) -> None:
        """Verifies that injecting more bytes than the declared matrix dimensions triggers EnclaveFramingError."""
        k_c2w = secrets.token_bytes(32)
        session = EnclaveSession(task_id="task-overflow", m=2, k=2, n=2, k_c2w=k_c2w, k_w2c=secrets.token_bytes(32))

        # Expected: 2*2*4 = 16 bytes. Send 32 bytes.
        oversized_data = np.ones((4, 2), dtype=np.float32).tobytes()
        aad = canonical_chunk_aad("task-overflow", 0, is_matrix_a=True, is_final_chunk=False)
        ct, iv, tag = aes_gcm_encrypt(k_c2w, oversized_data, aad=aad)

        chunk_overflow = pb2.TileDataChunk(
            task_id="task-overflow",
            chunk_bytes=ct,
            is_matrix_a=True,
            sequence_number=0,
            is_final_chunk=False,
            is_encrypted=True,
            iv=iv,
            tag=tag,
        )
        with pytest.raises(EnclaveFramingError, match="Matrix A stream exceeded expected byte limit"):
            session.ingest_chunk(chunk_overflow)


# ==============================================================================
# CHALLENGE 3: MEMORY BOUNDARY ISOLATION
# ==============================================================================

class TestEnclaveMemoryBoundaryIsolation:
    """
    Adversarial verification of enclave execution boundaries:
    1. Raw float32 matrix data is never leaked unencrypted.
    2. Result ciphertext exhibits high Shannon entropy (H >= 7.85) and cannot be parsed as matrix floats.
    3. Memory zeroization reliably wipes tensors and cryptographic keys.
    4. Confidentiality policy violations are strictly rejected.
    """

    def test_result_ciphertext_entropy_and_non_leakage(self) -> None:
        """
        Executes confidential GEMM and proves result_bytes is pure ciphertext
        with Shannon entropy H >= 7.85 and no correlation to ground truth floats.
        """
        m, k, n = 32, 32, 32  # 32x32 = 1024 floats = 4096 bytes
        k_c2w = secrets.token_bytes(32)
        k_w2c = secrets.token_bytes(32)
        session = EnclaveSession(task_id="task-leak-test", m=m, k=k, n=n, k_c2w=k_c2w, k_w2c=k_w2c)

        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        expected_c = a @ b
        expected_c_bytes = expected_c.tobytes()

        # Ingest A
        ct_a, iv_a, tag_a = aes_gcm_encrypt(k_c2w, a.tobytes(), aad=canonical_chunk_aad("task-leak-test", 0, True, False))
        session.ingest_chunk(pb2.TileDataChunk(
            task_id="task-leak-test", chunk_bytes=ct_a, is_matrix_a=True, sequence_number=0,
            is_final_chunk=False, is_encrypted=True, iv=iv_a, tag=tag_a
        ))

        # Ingest B
        ct_b, iv_b, tag_b = aes_gcm_encrypt(k_c2w, b.tobytes(), aad=canonical_chunk_aad("task-leak-test", 1, False, True))
        session.ingest_chunk(pb2.TileDataChunk(
            task_id="task-leak-test", chunk_bytes=ct_b, is_matrix_a=False, sequence_number=1,
            is_final_chunk=True, is_encrypted=True, iv=iv_b, tag=tag_b
        ))

        result_bytes, iv, tag, is_encrypted, _ = session.execute_and_encrypt()

        assert is_encrypted is True
        assert len(iv) == 12
        assert len(tag) == 16
        assert len(result_bytes) == len(expected_c_bytes)

        # 1. Byte exact non-leakage check
        assert result_bytes != expected_c_bytes, "Result leaked raw plaintext float32 bytes!"

        # 2. Shannon Entropy Verification: Ciphertext must be pseudorandom with H >= 7.85
        entropy = compute_shannon_entropy(result_bytes)
        assert entropy >= 7.85, f"Entropy too low for AES-256-GCM ciphertext: {entropy:.4f} bits/byte"

        # 3. Attempting to parse result_bytes directly as float32 must fail mathematical equivalence
        leaked_floats = np.frombuffer(result_bytes, dtype=np.float32).reshape((m, n))
        assert not np.allclose(leaked_floats, expected_c, atol=1e-3)

        # 4. Decrypting with valid K_w2c recovers ground truth
        decrypted_c_bytes = aes_gcm_decrypt(k_w2c, iv, result_bytes, tag, aad=canonical_result_aad("task-leak-test", m, n))
        decrypted_c = np.frombuffer(decrypted_c_bytes, dtype=np.float32).reshape((m, n))
        np.testing.assert_allclose(decrypted_c, expected_c, rtol=1e-5)

    def test_session_zeroization_scrubs_buffers_and_keys(self) -> None:
        """Verifies that zeroize() clears internal chunk byte buffers and wipes session keys."""
        k_c2w = secrets.token_bytes(32)
        k_w2c = secrets.token_bytes(32)
        session = EnclaveSession(task_id="task-zeroize", m=2, k=2, n=2, k_c2w=k_c2w, k_w2c=k_w2c)

        a_data = np.ones((2, 2), dtype=np.float32).tobytes()
        ct, iv, tag = aes_gcm_encrypt(k_c2w, a_data, aad=canonical_chunk_aad("task-zeroize", 0, True, True))
        session.ingest_chunk(pb2.TileDataChunk(
            task_id="task-zeroize", chunk_bytes=ct, is_matrix_a=True, sequence_number=0,
            is_final_chunk=True, is_encrypted=True, iv=iv, tag=tag
        ))

        assert len(session._a_chunks) == 1
        assert session.k_c2w is not None
        assert session.k_w2c is not None

        # Execute zeroize
        session.zeroize()

        assert len(session._a_chunks) == 0
        assert len(session._b_chunks) == 0
        assert session.k_c2w is None
        assert session.k_w2c is None

    def test_unencrypted_chunk_in_confidential_session_fails_policy(self) -> None:
        """Verifies that sending an unencrypted chunk to a confidential session raises EnclaveSecurityPolicyError."""
        session = EnclaveSession(task_id="task-sec-policy", m=2, k=2, n=2, is_confidential=True, k_c2w=secrets.token_bytes(32))
        chunk_plain = pb2.TileDataChunk(
            task_id="task-sec-policy",
            chunk_bytes=b"raw_plaintext_floats",
            is_matrix_a=True,
            sequence_number=0,
            is_final_chunk=False,
            is_encrypted=False,  # Policy violation!
        )
        with pytest.raises(EnclaveSecurityPolicyError, match="Unencrypted chunk received for confidential task"):
            session.ingest_chunk(chunk_plain)

    def test_encrypted_chunk_in_non_confidential_session_fails_policy(self) -> None:
        """Verifies that sending an encrypted chunk to a non-confidential session raises EnclaveSecurityPolicyError."""
        session = EnclaveSession(task_id="task-plain-session", m=2, k=2, n=2, is_confidential=False)
        chunk_enc = pb2.TileDataChunk(
            task_id="task-plain-session",
            chunk_bytes=b"ciphertext",
            is_matrix_a=True,
            sequence_number=0,
            is_final_chunk=False,
            is_encrypted=True,  # Policy violation for plaintext session!
            iv=secrets.token_bytes(12),
            tag=secrets.token_bytes(16),
        )
        with pytest.raises(EnclaveSecurityPolicyError, match="Encrypted chunk received for non-confidential task"):
            session.ingest_chunk(chunk_enc)


# ==============================================================================
# CHALLENGE 4: CONCURRENCY STRESS & SESSION ISOLATION
# ==============================================================================

class TestEnclaveConcurrencyStressAndIsolation:
    """
    Adversarial multi-threaded concurrency stress testing:
    1. Multiple concurrent sessions computing simultaneously without key or buffer bleed.
    2. Cross-session key mismatch: thread A's key cannot decrypt thread B's ciphertext.
    3. Concurrent servicer execution across separate gRPC streams.
    4. Bounded LRU and eviction under high thread concurrency.
    """

    def test_concurrent_sessions_without_key_or_buffer_bleed(self) -> None:
        """
        Executes 16 concurrent threads each computing independent confidential GEMM tasks.
        Validates mathematical precision for each thread and checks for cross-contamination.
        """
        core = EnclaveComputeCore(max_sessions=100)
        num_threads = 16
        results_collector: Dict[int, Dict[str, Any]] = {}
        errors: List[Exception] = []

        def worker_task(thread_idx: int) -> None:
            try:
                task_id = f"concurrent-task-{thread_idx}"
                m, k, n = 4, 4, 4
                c_priv = x25519.X25519PrivateKey.generate()
                c_pub = c_priv.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
                nonce = secrets.token_bytes(32)

                session, w_pub = core.create_session(
                    task_id=task_id,
                    m=m,
                    k=k,
                    n=n,
                    dtype="fp32",
                    require_confidential=True,
                    client_ephemeral_pubkey=c_pub,
                    attestation_nonce=nonce,
                )
                c_keys = derive_session_keys(c_priv, w_pub, nonce)

                # Distinct matrix data per thread
                a = (np.ones((m, k), dtype=np.float32) * (thread_idx + 1))
                b = (np.ones((k, n), dtype=np.float32) * 2.0)
                expected_c = a @ b

                # Encrypt and ingest A
                aad_a = canonical_chunk_aad(task_id, 0, True, False)
                ct_a, iv_a, tag_a = aes_gcm_encrypt(c_keys.k_c2w, a.tobytes(), aad=aad_a)
                session.ingest_chunk(pb2.TileDataChunk(
                    task_id=task_id, chunk_bytes=ct_a, is_matrix_a=True, sequence_number=0,
                    is_final_chunk=False, is_encrypted=True, iv=iv_a, tag=tag_a
                ))

                # Encrypt and ingest B
                aad_b = canonical_chunk_aad(task_id, 1, False, True)
                ct_b, iv_b, tag_b = aes_gcm_encrypt(c_keys.k_c2w, b.tobytes(), aad=aad_b)
                session.ingest_chunk(pb2.TileDataChunk(
                    task_id=task_id, chunk_bytes=ct_b, is_matrix_a=False, sequence_number=1,
                    is_final_chunk=True, is_encrypted=True, iv=iv_b, tag=tag_b
                ))

                result_bytes, iv, tag, is_enc, _ = session.execute_and_encrypt()

                # Decrypt result
                aad_res = canonical_result_aad(task_id, m, n)
                dec_c_bytes = aes_gcm_decrypt(c_keys.k_w2c, iv, result_bytes, tag, aad=aad_res)
                computed_c = np.frombuffer(dec_c_bytes, dtype=np.float32).reshape((m, n))

                np.testing.assert_allclose(computed_c, expected_c, rtol=1e-5)

                results_collector[thread_idx] = {
                    "task_id": task_id,
                    "c_keys": c_keys,
                    "result_bytes": result_bytes,
                    "iv": iv,
                    "tag": tag,
                    "expected_c": expected_c,
                }
                core.close_session(task_id)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Encountered errors during concurrent execution: {errors}"
        assert len(results_collector) == num_threads

        # Cross-session key bleed test: Thread i's key MUST NOT decrypt Thread j's result
        for i in range(num_threads):
            for j in range(num_threads):
                if i == j:
                    continue
                k_w2c_wrong = results_collector[j]["c_keys"].k_w2c
                res_i = results_collector[i]
                with pytest.raises(EnclaveDecryptionError):
                    aes_gcm_decrypt(
                        k_w2c_wrong,
                        res_i["iv"],
                        res_i["result_bytes"],
                        res_i["tag"],
                        aad=canonical_result_aad(res_i["task_id"], 4, 4),
                    )

    def test_servicer_concurrent_streaming_tasks(self) -> None:
        """Executes 8 concurrent streams against MatrixComputeServicer with unique L402 tokens."""
        num_tasks = 8
        errors: List[Exception] = []

        def run_stream(idx: int) -> None:
            try:
                task_id = f"servicer-concurrent-{idx}"
                m, k, n = 4, 4, 4
                servicer, auth, c_keys, _, _, _ = create_mock_servicer_and_auth(task_id, m=m, k=k, n=n)

                a = np.random.randn(m, k).astype(np.float32)
                b = np.random.randn(k, n).astype(np.float32)
                expected_c = a @ b

                aad_a = canonical_chunk_aad(task_id, 0, True, False)
                ct_a, iv_a, tag_a = aes_gcm_encrypt(c_keys.k_c2w, a.tobytes(), aad=aad_a)
                chunk_a = pb2.TileDataChunk(
                    task_id=task_id, chunk_bytes=ct_a, is_matrix_a=True, sequence_number=0,
                    is_final_chunk=False, is_encrypted=True, iv=iv_a, tag=tag_a
                )

                aad_b = canonical_chunk_aad(task_id, 1, False, True)
                ct_b, iv_b, tag_b = aes_gcm_encrypt(c_keys.k_c2w, b.tobytes(), aad=aad_b)
                chunk_b = pb2.TileDataChunk(
                    task_id=task_id, chunk_bytes=ct_b, is_matrix_a=False, sequence_number=1,
                    is_final_chunk=True, is_encrypted=True, iv=iv_b, tag=tag_b
                )

                ctx = MockServicerGrpcContext(metadata=[("authorization", auth), ("x-task-id", task_id)])
                result = servicer.ExecuteTileTask(iter([chunk_a, chunk_b]), context=ctx)

                assert result.is_encrypted is True
                dec_bytes = aes_gcm_decrypt(c_keys.k_w2c, result.iv, result.result_bytes, result.tag, aad=canonical_result_aad(task_id, m, n))
                c_mat = np.frombuffer(dec_bytes, dtype=np.float32).reshape((m, n))
                np.testing.assert_allclose(c_mat, expected_c, rtol=1e-5)
            except Exception as exc:
                errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as pool:
            futures = [pool.submit(run_stream, i) for i in range(num_tasks)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Concurrent servicer execution failed: {errors}"

    def test_core_bounded_lru_under_concurrency(self) -> None:
        """Stress-tests EnclaveComputeCore capacity bound and LRU eviction under heavy concurrent load."""
        max_capacity = 25
        core = EnclaveComputeCore(max_sessions=max_capacity)
        total_sessions = 150

        def create_entry(idx: int) -> None:
            core.create_session(f"lru-task-{idx}", 2, 2, 2, require_confidential=False)

        threads = [threading.Thread(target=create_entry, args=(i,)) for i in range(total_sessions)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with core._lock:
            # Must strictly not exceed max_capacity
            assert len(core._sessions) <= max_capacity
