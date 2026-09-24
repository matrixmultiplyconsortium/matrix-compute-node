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


def test_execute_tile_task_batch_caching(servicer, mock_settlement, signer, vkey):
    task_1 = "batch_tile_1"
    task_2 = "batch_tile_2"
    m, k, n = 4, 4, 4

    # 1. Negotiate both tasks
    for tid in [task_1, task_2]:
        req = pb2.TileTaskRequest(
            task_id=tid,
            matrix_a_rows=m,
            matrix_a_cols=k,
            matrix_b_cols=n,
            dtype="fp32",
        )
        ack = servicer.RequestTileTask(req)
        assert ack.is_ready is True

    # 2. Create batch invoice covering both tasks
    inv = mock_settlement.create_hodl_invoice(task_ids=[task_1, task_2], total_compute_sats=200)
    mock_settlement.hold_invoice(inv.payment_hash)

    # 3. Create batch Macaroon
    caveats = {
        "task_id": task_1,
        "task_ids": [task_1, task_2],
        "job_id": "job_servicer_batch",
        "expiry": int(time.time()) + 3600,
        "max_gflops": 10.0,
        "locked_sats": 200,
        "payment_hash": inv.payment_hash,
    }
    cav_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(vkey.encode("utf-8"), cav_str.encode("utf-8"), hashlib.sha256).hexdigest()
    mac = base64.b64encode(json.dumps({"caveats": caveats, "signature": sig}).encode("utf-8")).decode("utf-8")
    auth_header = f"L402 {mac}:{inv.payment_hash}"

    # Prepare data stream
    a = np.ones((m, k), dtype=np.float32)
    b = np.ones((k, n), dtype=np.float32)

    # Execute tile 1 -> Cache miss, queries settlement
    chunks_1 = [
        pb2.TileDataChunk(task_id=task_1, chunk_bytes=a.tobytes(), is_matrix_a=True, sequence_number=0, is_final_chunk=False),
        pb2.TileDataChunk(task_id=task_1, chunk_bytes=b.tobytes(), is_matrix_a=False, sequence_number=1, is_final_chunk=True),
    ]
    ctx_1 = MockGrpcContext(metadata=[("authorization", auth_header), ("x-task-id", task_1)])
    res_1 = servicer.ExecuteTileTask(iter(chunks_1), context=ctx_1)
    assert res_1.task_id == task_1
    assert mock_settlement.query_count == 1
    assert servicer.gate.remote_query_count == 1
    assert servicer.gate.cache_hit_count == 0

    # Execute tile 2 -> Cache hit, 0 new remote queries!
    chunks_2 = [
        pb2.TileDataChunk(task_id=task_2, chunk_bytes=a.tobytes(), is_matrix_a=True, sequence_number=0, is_final_chunk=False),
        pb2.TileDataChunk(task_id=task_2, chunk_bytes=b.tobytes(), is_matrix_a=False, sequence_number=1, is_final_chunk=True),
    ]
    ctx_2 = MockGrpcContext(metadata=[("authorization", auth_header), ("x-task-id", task_2)])
    res_2 = servicer.ExecuteTileTask(iter(chunks_2), context=ctx_2)
    assert res_2.task_id == task_2
    assert mock_settlement.query_count == 1
    assert servicer.gate.remote_query_count == 1
    assert servicer.gate.cache_hit_count == 1


def test_execute_tile_task_unauthorized_task_aborts_permission_denied(servicer, mock_settlement, vkey):
    task_auth = "batch_authorized_task"
    task_unauth = "batch_unauthorized_task"
    m, k, n = 4, 4, 4

    req = pb2.TileTaskRequest(
        task_id=task_unauth,
        matrix_a_rows=m,
        matrix_a_cols=k,
        matrix_b_cols=n,
        dtype="fp32",
    )
    servicer.RequestTileTask(req)

    inv = mock_settlement.create_hodl_invoice(task_ids=[task_auth], total_compute_sats=100)
    mock_settlement.hold_invoice(inv.payment_hash)

    caveats = {
        "task_id": task_auth,
        "task_ids": [task_auth],
        "job_id": "job_unauth_test",
        "expiry": int(time.time()) + 3600,
        "max_gflops": 10.0,
        "locked_sats": 100,
        "payment_hash": inv.payment_hash,
    }
    cav_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(vkey.encode("utf-8"), cav_str.encode("utf-8"), hashlib.sha256).hexdigest()
    mac = base64.b64encode(json.dumps({"caveats": caveats, "signature": sig}).encode("utf-8")).decode("utf-8")
    auth_header = f"L402 {mac}:{inv.payment_hash}"

    ctx = MockGrpcContext(metadata=[("authorization", auth_header), ("x-task-id", task_unauth)])

    # Stream generator that records if any chunk was ingested
    stream_read = False
    def stream_gen():
        nonlocal stream_read
        stream_read = True
        yield pb2.TileDataChunk(task_id=task_unauth)

    with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
        servicer.ExecuteTileTask(stream_gen(), context=ctx)

    # DoS protection check: abort occurs before any stream chunks are read
    assert stream_read is False
    assert mock_settlement.query_count == 0


def test_execute_tile_task_case_insensitive_metadata(servicer, mock_settlement, vkey):
    task_id = "task_case_insensitive"
    m, k, n = 4, 4, 4

    req = pb2.TileTaskRequest(
        task_id=task_id,
        matrix_a_rows=m,
        matrix_a_cols=k,
        matrix_b_cols=n,
        dtype="fp32",
    )
    servicer.RequestTileTask(req)

    inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
    mock_settlement.hold_invoice(inv.payment_hash)

    mac = mint_macaroon(vkey, task_id, int(time.time()) + 3600, 1.0, payment_hash=inv.payment_hash)
    auth_header = f"L402 {mac}:{inv.payment_hash}"

    a = np.ones((m, k), dtype=np.float32)
    b = np.ones((k, n), dtype=np.float32)
    chunks = [
        pb2.TileDataChunk(task_id=task_id, chunk_bytes=a.tobytes(), is_matrix_a=True, sequence_number=0, is_final_chunk=False),
        pb2.TileDataChunk(task_id=task_id, chunk_bytes=b.tobytes(), is_matrix_a=False, sequence_number=1, is_final_chunk=True),
    ]

    # Uppercase headers
    ctx = MockGrpcContext(metadata=[
        ("Authorization", auth_header),
        ("X-Task-ID", task_id),
    ])
    result = servicer.ExecuteTileTask(iter(chunks), context=ctx)
    assert result.task_id == task_id
    assert len(result.result_bytes) == m * n * 4

