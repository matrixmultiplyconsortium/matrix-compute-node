"""
Unit tests for EnclaveComputeCore, EnclaveSession, and In-Enclave AES-256-GCM Framing (Milestone 3).
"""

from __future__ import annotations

import secrets
from cryptography.hazmat.primitives.asymmetric import x25519
import numpy as np
import pytest

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
    EnclaveSessionNotFoundError,
    SessionKeys,
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    canonical_chunk_aad,
    canonical_result_aad,
    compute_chunk_aad,
    compute_result_aad,
    derive_session_keys,
)

try:
    from matrix_contracts.v1 import pb2
except ImportError:
    from matrix_contracts.proto import compute_service_pb2 as pb2


# ==============================================================================
# Tier 1: Cryptographic Primitives & Derivation Unit Tests
# ==============================================================================

def test_x25519_symmetric_key_derivation():
    c_priv = x25519.X25519PrivateKey.generate()
    c_pub = c_priv.public_key()
    w_priv = x25519.X25519PrivateKey.generate()
    w_pub = w_priv.public_key()
    nonce = secrets.token_bytes(32)

    c_keys = derive_session_keys(c_priv, w_pub, nonce)
    w_keys = derive_session_keys(w_priv, c_pub, nonce)

    assert isinstance(c_keys, SessionKeys)
    assert c_keys.k_c2w == w_keys.k_c2w
    assert c_keys.k_w2c == w_keys.k_w2c
    assert c_keys.k_c2w != c_keys.k_w2c


def test_x25519_low_order_point_rejected():
    w_priv = x25519.X25519PrivateKey.generate()
    zero_point = b"\x00" * 32
    nonce = secrets.token_bytes(32)

    with pytest.raises(EnclaveKeyExchangeError):
        derive_session_keys(w_priv, zero_point, nonce)


def test_aes_gcm_roundtrip():
    key = secrets.token_bytes(32)
    plaintext = np.random.randn(8, 8).astype(np.float32).tobytes()
    aad = b"task-01:chunk:1:0:0"

    ct, iv, tag = aes_gcm_encrypt(key, plaintext, aad=aad)
    assert len(iv) == 12
    assert len(tag) == 16
    assert ct != plaintext

    decrypted = aes_gcm_decrypt(key, iv, ct, tag, aad=aad)
    assert decrypted == plaintext


def test_aes_gcm_ciphertext_tamper_fails():
    key = secrets.token_bytes(32)
    plaintext = b"Payload to tamper"
    aad = b"aad"

    ct, iv, tag = aes_gcm_encrypt(key, plaintext, aad=aad)
    bad_ct = bytes([ct[0] ^ 0xFF]) + ct[1:]

    with pytest.raises(EnclaveDecryptionError):
        aes_gcm_decrypt(key, iv, bad_ct, tag, aad=aad)


def test_aes_gcm_tag_tamper_fails():
    key = secrets.token_bytes(32)
    plaintext = b"Payload to tamper"
    aad = b"aad"

    ct, iv, tag = aes_gcm_encrypt(key, plaintext, aad=aad)
    bad_tag = bytes([tag[0] ^ 0xFF]) + tag[1:]

    with pytest.raises(EnclaveDecryptionError):
        aes_gcm_decrypt(key, iv, ct, bad_tag, aad=aad)


def test_aes_gcm_aad_tamper_fails():
    key = secrets.token_bytes(32)
    plaintext = b"Payload to tamper"

    ct, iv, tag = aes_gcm_encrypt(key, plaintext, aad=b"valid_aad")

    with pytest.raises(EnclaveDecryptionError):
        aes_gcm_decrypt(key, iv, ct, tag, aad=b"invalid_aad")


def test_fresh_iv_uniqueness():
    key = secrets.token_bytes(32)
    plaintext = b"same plaintext"
    ivs = set()
    for _ in range(500):
        _, iv, _ = aes_gcm_encrypt(key, plaintext)
        assert iv not in ivs
        ivs.add(iv)
    assert len(ivs) == 500


# ==============================================================================
# Tier 2: Enclave Session & Chunk Ingestion Unit Tests
# ==============================================================================

def test_enclave_session_chunk_ingestion_success():
    k_c2w = secrets.token_bytes(32)
    k_w2c = secrets.token_bytes(32)

    m, k, n = 4, 4, 4
    session = EnclaveSession(
        task_id="task-ingest-01",
        m=m,
        k=k,
        n=n,
        dtype="fp32",
        is_confidential=True,
        k_c2w=k_c2w,
        k_w2c=k_w2c,
    )

    a_data = np.ones((m, k), dtype=np.float32).tobytes()
    b_data = np.ones((k, n), dtype=np.float32).tobytes()

    # Chunk A (seq 0)
    aad_a = canonical_chunk_aad("task-ingest-01", 0, is_matrix_a=True, is_final_chunk=False)
    ct_a, iv_a, tag_a = aes_gcm_encrypt(k_c2w, a_data, aad=aad_a)
    chunk_a = pb2.TileDataChunk(
        task_id="task-ingest-01",
        chunk_bytes=ct_a,
        is_matrix_a=True,
        sequence_number=0,
        is_final_chunk=False,
        is_encrypted=True,
        iv=iv_a,
        tag=tag_a,
    )
    session.ingest_chunk(chunk_a)

    # Chunk B (seq 1, final)
    aad_b = canonical_chunk_aad("task-ingest-01", 1, is_matrix_a=False, is_final_chunk=True)
    ct_b, iv_b, tag_b = aes_gcm_encrypt(k_c2w, b_data, aad=aad_b)
    chunk_b = pb2.TileDataChunk(
        task_id="task-ingest-01",
        chunk_bytes=ct_b,
        is_matrix_a=False,
        sequence_number=1,
        is_final_chunk=True,
        is_encrypted=True,
        iv=iv_b,
        tag=tag_b,
    )
    session.ingest_chunk(chunk_b)

    assert session._running_a_bytes == session.expected_a_bytes
    assert session._running_b_bytes == session.expected_b_bytes
    assert session._final_seen is True


def test_enclave_session_out_of_order_rejected():
    session = EnclaveSession(
        task_id="task-ooo",
        m=4,
        k=4,
        n=4,
        is_confidential=False,
    )
    # Send seq 1 instead of 0
    chunk = pb2.TileDataChunk(
        task_id="task-ooo",
        chunk_bytes=b"\x00" * 16,
        is_matrix_a=True,
        sequence_number=1,
        is_final_chunk=False,
        is_encrypted=False,
    )
    with pytest.raises(EnclaveFramingError, match="Out of order chunk"):
        session.ingest_chunk(chunk)


def test_enclave_session_task_id_mismatch_rejected():
    session = EnclaveSession(
        task_id="task-correct",
        m=4,
        k=4,
        n=4,
        is_confidential=False,
    )
    chunk = pb2.TileDataChunk(
        task_id="task-wrong",
        chunk_bytes=b"\x00" * 16,
        is_matrix_a=True,
        sequence_number=0,
        is_final_chunk=False,
        is_encrypted=False,
    )
    with pytest.raises(EnclaveFramingError, match="Chunk task_id mismatch"):
        session.ingest_chunk(chunk)


def test_enclave_session_policy_violation_unencrypted():
    session = EnclaveSession(
        task_id="task-sec-violation",
        m=4,
        k=4,
        n=4,
        is_confidential=True,
        k_c2w=secrets.token_bytes(32),
    )
    chunk = pb2.TileDataChunk(
        task_id="task-sec-violation",
        chunk_bytes=b"\x00" * 16,
        is_matrix_a=True,
        sequence_number=0,
        is_final_chunk=False,
        is_encrypted=False,  # Unencrypted to confidential session!
    )
    with pytest.raises(EnclaveSecurityPolicyError, match="Unencrypted chunk received for confidential task"):
        session.ingest_chunk(chunk)


def test_enclave_session_policy_violation_encrypted_to_plain():
    session = EnclaveSession(
        task_id="task-sec-violation-plain",
        m=4,
        k=4,
        n=4,
        is_confidential=False,
    )
    chunk = pb2.TileDataChunk(
        task_id="task-sec-violation-plain",
        chunk_bytes=b"\x00" * 16,
        is_matrix_a=True,
        sequence_number=0,
        is_final_chunk=False,
        is_encrypted=True,  # Encrypted to unencrypted session!
    )
    with pytest.raises(EnclaveSecurityPolicyError, match="Encrypted chunk received for non-confidential task"):
        session.ingest_chunk(chunk)


def test_enclave_session_byte_overflow_protection():
    session = EnclaveSession(
        task_id="task-overflow",
        m=2,
        k=2,
        n=2,
        dtype="fp32",
        is_confidential=False,
    )
    # expected_a_bytes = 2 * 2 * 4 = 16 bytes. Send 20 bytes.
    chunk = pb2.TileDataChunk(
        task_id="task-overflow",
        chunk_bytes=b"\x00" * 20,
        is_matrix_a=True,
        sequence_number=0,
        is_final_chunk=False,
        is_encrypted=False,
    )
    with pytest.raises(EnclaveFramingError, match="Matrix A stream exceeded expected byte limit"):
        session.ingest_chunk(chunk)


def test_enclave_session_cleanup_zeroization():
    session = EnclaveSession(
        task_id="task-zeroize",
        m=2,
        k=2,
        n=2,
        is_confidential=True,
        k_c2w=secrets.token_bytes(32),
        k_w2c=secrets.token_bytes(32),
    )
    session._a_chunks.append(b"dummy")
    session.zeroize()
    assert len(session._a_chunks) == 0
    assert len(session._b_chunks) == 0
    assert session.k_c2w is None
    assert session.k_w2c is None


# ==============================================================================
# Tier 3: In-Enclave GEMM Execution & Result Encryption Tests
# ==============================================================================

def test_enclave_confidential_gemm_end_to_end():
    # 1. Ephemeral key derivation
    c_priv = x25519.X25519PrivateKey.generate()
    c_pub = c_priv.public_key()
    w_priv = x25519.X25519PrivateKey.generate()
    w_pub = w_priv.public_key()
    nonce = secrets.token_bytes(32)

    c_keys = derive_session_keys(c_priv, w_pub, nonce)
    w_keys = derive_session_keys(w_priv, c_pub, nonce)

    m, k, n = 16, 16, 16
    session = EnclaveSession(
        task_id="task-gemm-e2e",
        m=m,
        k=k,
        n=n,
        dtype="fp32",
        is_confidential=True,
        k_c2w=w_keys.k_c2w,
        k_w2c=w_keys.k_w2c,
    )

    a_mat = np.random.randn(m, k).astype(np.float32)
    b_mat = np.random.randn(k, n).astype(np.float32)
    expected_c = a_mat @ b_mat

    # Encrypt chunks using Client Key K_c2w
    aad_a = canonical_chunk_aad("task-gemm-e2e", 0, is_matrix_a=True, is_final_chunk=False)
    ct_a, iv_a, tag_a = aes_gcm_encrypt(c_keys.k_c2w, a_mat.tobytes(), aad=aad_a)
    session.ingest_chunk(pb2.TileDataChunk(
        task_id="task-gemm-e2e", chunk_bytes=ct_a, is_matrix_a=True,
        sequence_number=0, is_final_chunk=False, is_encrypted=True, iv=iv_a, tag=tag_a
    ))

    aad_b = canonical_chunk_aad("task-gemm-e2e", 1, is_matrix_a=False, is_final_chunk=True)
    ct_b, iv_b, tag_b = aes_gcm_encrypt(c_keys.k_c2w, b_mat.tobytes(), aad=aad_b)
    session.ingest_chunk(pb2.TileDataChunk(
        task_id="task-gemm-e2e", chunk_bytes=ct_b, is_matrix_a=False,
        sequence_number=1, is_final_chunk=True, is_encrypted=True, iv=iv_b, tag=tag_b
    ))

    # In-enclave GEMM execution and result encryption
    result_bytes, iv, tag, is_encrypted, exec_time_ms = session.execute_and_encrypt()
    assert is_encrypted is True
    assert len(iv) == 12
    assert len(tag) == 16
    assert exec_time_ms >= 0

    # Decrypt result using Client Key K_w2c
    aad_res = canonical_result_aad("task-gemm-e2e", m, n)
    c_decrypted_raw = aes_gcm_decrypt(c_keys.k_w2c, iv, result_bytes, tag, aad=aad_res)
    c_decrypted = np.frombuffer(c_decrypted_raw, dtype=np.float32).reshape((m, n))

    np.testing.assert_allclose(c_decrypted, expected_c, rtol=1e-5)


def test_enclave_unencrypted_fallback():
    m, k, n = 8, 8, 8
    session = EnclaveSession(
        task_id="task-plain-fallback",
        m=m,
        k=k,
        n=n,
        dtype="fp32",
        is_confidential=False,
    )

    a_mat = np.random.randn(m, k).astype(np.float32)
    b_mat = np.random.randn(k, n).astype(np.float32)
    expected_c = a_mat @ b_mat

    session.ingest_chunk(pb2.TileDataChunk(
        task_id="task-plain-fallback", chunk_bytes=a_mat.tobytes(), is_matrix_a=True,
        sequence_number=0, is_final_chunk=False, is_encrypted=False
    ))
    session.ingest_chunk(pb2.TileDataChunk(
        task_id="task-plain-fallback", chunk_bytes=b_mat.tobytes(), is_matrix_a=False,
        sequence_number=1, is_final_chunk=True, is_encrypted=False
    ))

    result_bytes, iv, tag, is_encrypted, _ = session.execute_and_encrypt()
    assert is_encrypted is False
    assert iv == b""
    assert tag == b""

    c_res = np.frombuffer(result_bytes, dtype=np.float32).reshape((m, n))
    np.testing.assert_allclose(c_res, expected_c, rtol=1e-5)


def test_enclave_precision_support_fp16_bf16():
    for dtype in ("fp16", "bf16"):
        session = EnclaveSession(
            task_id=f"task-precision-{dtype}",
            m=4,
            k=4,
            n=4,
            dtype=dtype,
            is_confidential=False,
        )
        a_mat = np.ones((4, 4), dtype=np.float32 if dtype == "bf16" else np.float16)
        b_mat = np.ones((4, 4), dtype=np.float32 if dtype == "bf16" else np.float16)

        session.ingest_chunk(pb2.TileDataChunk(
            task_id=f"task-precision-{dtype}", chunk_bytes=a_mat.tobytes(), is_matrix_a=True,
            sequence_number=0, is_final_chunk=False, is_encrypted=False
        ))
        session.ingest_chunk(pb2.TileDataChunk(
            task_id=f"task-precision-{dtype}", chunk_bytes=b_mat.tobytes(), is_matrix_a=False,
            sequence_number=1, is_final_chunk=True, is_encrypted=False
        ))

        res_bytes, _, _, _, _ = session.execute_and_encrypt()
        assert len(res_bytes) == 4 * 4 * (4 if dtype == "bf16" else 2)


# ==============================================================================
# Tier 4: EnclaveComputeCore Manager Lifecycle & Eviction Tests
# ==============================================================================

def test_enclave_compute_core_lifecycle():
    core = EnclaveComputeCore(max_sessions=10, session_ttl_seconds=3600.0)

    from cryptography.hazmat.primitives import serialization
    c_p = x25519.X25519PrivateKey.generate()
    c_pub_bytes = c_p.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    nonce = secrets.token_bytes(32)

    session, w_pub = core.create_session(
        task_id="core-task-1",
        m=8,
        k=8,
        n=8,
        dtype="fp32",
        require_confidential=True,
        client_ephemeral_pubkey=c_pub_bytes,
        attestation_nonce=nonce,
    )

    assert session is not None
    assert len(w_pub) == 32
    assert core.get_session("core-task-1") is session

    core.close_session("core-task-1")
    assert core.get_session("core-task-1") is None


def test_enclave_compute_core_eviction():
    core = EnclaveComputeCore(max_sessions=2, session_ttl_seconds=5.0)

    core.create_session("task-1", 2, 2, 2, require_confidential=False)
    core.create_session("task-2", 2, 2, 2, require_confidential=False)
    # Adding 3rd should evict oldest (task-1)
    core.create_session("task-3", 2, 2, 2, require_confidential=False)

    assert core.get_session("task-1") is None
    assert core.get_session("task-2") is not None
    assert core.get_session("task-3") is not None

    # Test TTL expiration
    core.evict_expired(now=session_start := 1000000000.0 + 10.0)
    assert core.get_session("task-2") is None
    assert core.get_session("task-3") is None


# ==============================================================================
# Tier 5: Servicer Streaming gRPC Integration Tests
# ==============================================================================

class MockServicerGrpcContext:
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


def test_servicer_confidential_streaming_execution():
    import base64
    import hashlib
    import hmac
    import json
    import time
    from src.attestation.simulated import SimulatedAttestationProvider
    from src.auth.l402_gate import L402PreAllocationGate
    from src.crypto.signer import WorkerSigner
    from src.p2p.servicer import MatrixComputeServicer
    from src.test_doubles.mock_settlement import MockSettlementService

    signer = WorkerSigner()
    vkey = "servicer-test-vkey"
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

    task_id = "task-conf-servicer-01"
    m, k, n = 8, 8, 8

    # 1. Client generates ephemeral X25519 keypair and challenger nonce
    c_priv = x25519.X25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization
    c_pub = c_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    nonce = secrets.token_bytes(32)

    # 2. Handshake via RequestTileTask
    req = pb2.TileTaskRequest(
        task_id=task_id,
        matrix_a_rows=m,
        matrix_a_cols=k,
        matrix_b_cols=n,
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=c_pub,
        attestation_nonce=nonce,
        required_tee_type="simulated-cca",
    )
    ack = servicer.RequestTileTask(req)
    assert ack.is_ready is True
    assert len(ack.worker_ephemeral_pubkey) == 32
    assert len(ack.attestation_evidence) > 0

    # 3. Client derives session keys
    c_keys = derive_session_keys(c_priv, ack.worker_ephemeral_pubkey, nonce)

    # 4. Mint valid L402 invoice and auth header
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

    # 5. Prepare and encrypt streaming chunks using K_c2w
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    expected_c = a @ b

    aad_a = canonical_chunk_aad(task_id, 0, is_matrix_a=True, is_final_chunk=False)
    ct_a, iv_a, tag_a = aes_gcm_encrypt(c_keys.k_c2w, a.tobytes(), aad=aad_a)
    chunk_a = pb2.TileDataChunk(
        task_id=task_id,
        chunk_bytes=ct_a,
        is_matrix_a=True,
        sequence_number=0,
        is_final_chunk=False,
        is_encrypted=True,
        iv=iv_a,
        tag=tag_a,
    )

    aad_b = canonical_chunk_aad(task_id, 1, is_matrix_a=False, is_final_chunk=True)
    ct_b, iv_b, tag_b = aes_gcm_encrypt(c_keys.k_c2w, b.tobytes(), aad=aad_b)
    chunk_b = pb2.TileDataChunk(
        task_id=task_id,
        chunk_bytes=ct_b,
        is_matrix_a=False,
        sequence_number=1,
        is_final_chunk=True,
        is_encrypted=True,
        iv=iv_b,
        tag=tag_b,
    )

    ctx = MockServicerGrpcContext(metadata=[("authorization", auth_header), ("x-task-id", task_id)])
    result = servicer.ExecuteTileTask(iter([chunk_a, chunk_b]), context=ctx)

    assert result.task_id == task_id
    assert result.is_encrypted is True
    assert len(result.iv) == 12
    assert len(result.tag) == 16

    # 6. Client decrypts result using K_w2c
    aad_res = canonical_result_aad(task_id, m, n)
    c_decrypted_bytes = aes_gcm_decrypt(c_keys.k_w2c, result.iv, result.result_bytes, result.tag, aad=aad_res)
    c_decrypted = np.frombuffer(c_decrypted_bytes, dtype=np.float32).reshape((m, n))
    np.testing.assert_allclose(c_decrypted, expected_c, rtol=1e-5)

    # 7. Verify worker signature over wire ciphertext payload
    c_hash = hashlib.sha256(result.result_bytes).hexdigest()
    expected_signed_msg = f"{task_id}:{c_hash}".encode("utf-8")
    assert signer.verify_signature(expected_signed_msg, result.worker_signature, signer.get_pubkey_hex()) is True


def test_servicer_tampered_stream_aborts_cleanly():
    import base64
    import hashlib
    import hmac
    import json
    import time
    from src.attestation.simulated import SimulatedAttestationProvider
    from src.auth.l402_gate import L402PreAllocationGate
    from src.crypto.signer import WorkerSigner
    from src.p2p.servicer import MatrixComputeServicer
    from src.test_doubles.mock_settlement import MockSettlementService

    signer = WorkerSigner()
    vkey = "servicer-test-vkey"
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

    task_id = "task-tampered-stream-01"
    m, k, n = 4, 4, 4

    c_priv = x25519.X25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization
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
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=c_pub,
        attestation_nonce=nonce,
    )
    ack = servicer.RequestTileTask(req)
    c_keys = derive_session_keys(c_priv, ack.worker_ephemeral_pubkey, nonce)

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

    a = np.ones((m, k), dtype=np.float32)
    aad_a = canonical_chunk_aad(task_id, 0, is_matrix_a=True, is_final_chunk=False)
    ct_a, iv_a, tag_a = aes_gcm_encrypt(c_keys.k_c2w, a.tobytes(), aad=aad_a)
    # Flip bit in ciphertext
    bad_ct_a = bytes([ct_a[0] ^ 0xFF]) + ct_a[1:]

    chunk_a = pb2.TileDataChunk(
        task_id=task_id,
        chunk_bytes=bad_ct_a,
        is_matrix_a=True,
        sequence_number=0,
        is_final_chunk=False,
        is_encrypted=True,
        iv=iv_a,
        tag=tag_a,
    )

    ctx = MockServicerGrpcContext(metadata=[("authorization", auth_header), ("x-task-id", task_id)])
    with pytest.raises(RuntimeError, match="ABORT: StatusCode.INVALID_ARGUMENT"):
        servicer.ExecuteTileTask(iter([chunk_a]), context=ctx)


def test_servicer_unencrypted_policy_violation_aborts():
    import base64
    import hashlib
    import hmac
    import json
    import time
    from src.attestation.simulated import SimulatedAttestationProvider
    from src.auth.l402_gate import L402PreAllocationGate
    from src.crypto.signer import WorkerSigner
    from src.p2p.servicer import MatrixComputeServicer
    from src.test_doubles.mock_settlement import MockSettlementService

    signer = WorkerSigner()
    vkey = "servicer-test-vkey"
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

    task_id = "task-policy-viol-01"
    m, k, n = 4, 4, 4

    c_priv = x25519.X25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization
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
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=c_pub,
        attestation_nonce=nonce,
    )
    servicer.RequestTileTask(req)

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

    # Send UNENCRYPTED chunk to confidential task
    a = np.ones((m, k), dtype=np.float32)
    chunk_plain = pb2.TileDataChunk(
        task_id=task_id,
        chunk_bytes=a.tobytes(),
        is_matrix_a=True,
        sequence_number=0,
        is_final_chunk=False,
        is_encrypted=False,
    )

    ctx = MockServicerGrpcContext(metadata=[("authorization", auth_header), ("x-task-id", task_id)])
    with pytest.raises(RuntimeError, match="ABORT: StatusCode.PERMISSION_DENIED"):
        servicer.ExecuteTileTask(iter([chunk_plain]), context=ctx)


