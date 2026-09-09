"""Unit tests for MatrixComputeServicer."""

import base64
import hashlib
import hmac
import json
import time
import numpy as np
import pytest

from matrix_contracts.v1 import pb2
from src.auth.l402_gate import L402PreAllocationGate
from src.crypto.signer import WorkerSigner
from src.engine.compute_core import ComputeEngine
from src.p2p.servicer import MatrixComputeServicer
from src.test_doubles.mock_settlement import MockSettlementService


def mint_macaroon(
    key: str,
    task_id: str,
    expiry: int,
    max_gflops: float,
    locked_sats: int = 100,
    payment_hash: str = "default_servicer_ph",
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


@pytest.fixture
def vkey():
    return "servicer-unit-test-vkey"


@pytest.fixture
def mock_settlement(vkey):
    return MockSettlementService(verification_key=vkey)


@pytest.fixture
def signer():
    return WorkerSigner()


@pytest.fixture
def servicer(signer, mock_settlement, vkey):
    gate = L402PreAllocationGate(
        settlement_client_or_url=mock_settlement,
        verification_key=vkey,
        min_price_rate=0.0005,
    )
    engine = ComputeEngine(force_cpu=True)
    return MatrixComputeServicer(signer=signer, gate=gate, engine=engine)


def test_request_tile_task_valid(servicer, signer):
    req = pb2.TileTaskRequest(
        task_id="task_req_01",
        matrix_a_rows=32,
        matrix_a_cols=16,
        matrix_b_cols=32,
        dtype="fp32",
    )
    ack = servicer.RequestTileTask(req)
    assert ack.task_id == "task_req_01"
    assert ack.is_ready is True
    assert ack.worker_pubkey == signer.get_pubkey_hex()
    expected_gflops = (2.0 * 32 * 16 * 32) / 1e9
    assert ack.estimated_gflops == pytest.approx(expected_gflops)


def test_request_tile_task_invalid_dimension(servicer):
    req = pb2.TileTaskRequest(
        task_id="task_invalid_dim",
        matrix_a_rows=0,
        matrix_a_cols=16,
        matrix_b_cols=32,
        dtype="fp32",
    )
    with pytest.raises(ValueError, match="positive integers"):
        servicer.RequestTileTask(req)


def test_request_tile_task_invalid_dtype(servicer):
    req = pb2.TileTaskRequest(
        task_id="task_invalid_dtype",
        matrix_a_rows=16,
        matrix_a_cols=16,
        matrix_b_cols=16,
        dtype="complex128",
    )
    with pytest.raises(ValueError, match="Unsupported dtype"):
        servicer.RequestTileTask(req)


def test_execute_tile_task_missing_auth_header_aborts(servicer):
    ctx = MockGrpcContext(metadata=[])
    with pytest.raises(RuntimeError, match="Missing Authorization header"):
        servicer.ExecuteTileTask(iter([]), context=ctx)


def test_execute_tile_task_unpaid_invoice_aborts(servicer, mock_settlement, vkey):
    task_id = "task_exec_unpaid"
    inv = mock_settlement.create_hodl_invoice(task_id)
    # Don't hold invoice -> UNPAID

    mac = mint_macaroon(vkey, task_id, int(time.time()) + 3600, 1.0, payment_hash=inv.payment_hash)
    ctx = MockGrpcContext(metadata=[("authorization", f"L402 {mac}:{inv.payment_hash}")])

    with pytest.raises(RuntimeError, match="UNPAID"):
        servicer.ExecuteTileTask(iter([]), context=ctx)


def test_execute_tile_task_full_lifecycle(servicer, mock_settlement, signer, vkey):
    task_id = "task_exec_success"
    m, k, n = 4, 4, 4

    # 1. Handshake quote
    req = pb2.TileTaskRequest(
        task_id=task_id,
        matrix_a_rows=m,
        matrix_a_cols=k,
        matrix_b_cols=n,
        dtype="fp32",
    )
    ack = servicer.RequestTileTask(req)
    assert ack.is_ready is True

    # 2. Settlement invoice and hold
    inv = mock_settlement.create_hodl_invoice(task_id, total_compute_sats=100)
    mock_settlement.hold_invoice(inv.payment_hash)

    # 3. Create L402 token
    mac = mint_macaroon(vkey, task_id, int(time.time()) + 3600, max_gflops=1.0, payment_hash=inv.payment_hash)
    auth_header = f"L402 {mac}:{inv.payment_hash}"

    # 4. Prepare data stream
    np.random.seed(123)
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    chunks = [
        pb2.TileDataChunk(task_id=task_id, chunk_bytes=a.tobytes(), is_matrix_a=True, sequence_number=0, is_final_chunk=False),
        pb2.TileDataChunk(task_id=task_id, chunk_bytes=b.tobytes(), is_matrix_a=False, sequence_number=1, is_final_chunk=True),
    ]

    ctx = MockGrpcContext(metadata=[("authorization", auth_header)])
    result = servicer.ExecuteTileTask(iter(chunks), context=ctx)

    assert result.task_id == task_id
    assert result.rows == m
    assert result.cols == n
    assert len(result.result_bytes) == m * n * 4

    # Verify result correctness
    res_matrix = np.frombuffer(result.result_bytes, dtype=np.float32).reshape((m, n))
    expected = np.matmul(a, b)
    np.testing.assert_allclose(res_matrix, expected, rtol=1e-5)

    # Verify worker signature
    c_hash = hashlib.sha256(result.result_bytes).hexdigest()
    expected_signed_msg = f"{task_id}:{c_hash}".encode("utf-8")
    assert signer.verify_signature(expected_signed_msg, result.worker_signature, signer.get_pubkey_hex()) is True
