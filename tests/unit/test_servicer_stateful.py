"""
Unit and Integration tests for stateful RPC handlers in MatrixComputeServicer:
LeaseMatrix, StepIterative, and ReleaseLease.
Tests plaintext lifecycle, 100 power iterations, Secp256k1 result signatures,
L402 gate authorization, and confidential enclave encryption.
"""

import base64
import hashlib
import hmac
import json
import time
from typing import Optional, Tuple
import numpy as np
import pytest
import grpc

from matrix_contracts.v1 import pb2
from src.auth.l402_gate import L402PreAllocationGate
from src.crypto.signer import WorkerSigner
from src.engine.compute_core import ComputeEngine
from src.engine.enclave_compute import (
    EnclaveComputeCore,
    derive_session_keys,
    aes_gcm_encrypt,
    aes_gcm_decrypt,
)
from src.engine.session_manager import SessionManager
from src.p2p.servicer import MatrixComputeServicer
from src.test_doubles.mock_settlement import MockSettlementService


# ==============================================================================
# Helper Doubles & Utilities
# ==============================================================================

class MockGrpcContext:
    """Mock gRPC context simulating metadata passing and abort handling."""
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


def mint_macaroon(
    key: str,
    task_id: str,
    expiry: int,
    max_gflops: float = 10.0,
    locked_sats: int = 100,
    payment_hash: str = "default_hash",
) -> str:
    """Mints an L402 Macaroon token for testing."""
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


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def vkey():
    return "test-verification-secret-key-12345"


@pytest.fixture
def mock_settlement(vkey):
    return MockSettlementService(verification_key=vkey)


@pytest.fixture
def signer():
    return WorkerSigner()


@pytest.fixture
def session_manager():
    mgr = SessionManager(max_sessions=50, max_memory_bytes=100 * 1024 * 1024, reaper_interval_seconds=1.0)
    yield mgr
    mgr.shutdown()


@pytest.fixture
def servicer(signer, mock_settlement, vkey, session_manager):
    gate = L402PreAllocationGate(
        settlement_client_or_url=mock_settlement,
        verification_key=vkey,
        min_price_rate=0.0005,
    )
    engine = ComputeEngine(force_cpu=True)
    enclave_core = EnclaveComputeCore(compute_engine=engine)
    return MatrixComputeServicer(
        signer=signer,
        gate=gate,
        engine=engine,
        enclave_core=enclave_core,
        session_manager=session_manager,
    )


# ==============================================================================
# 1. Plaintext Stateful Lifecycle Tests
# ==============================================================================

def test_stateful_session_lifecycle_plaintext(servicer, signer):
    """
    Verifies the complete 3-stage stateful lifecycle:
    LeaseMatrix -> StepIterative (multiple steps) -> ReleaseLease.
    """
    lease_id = "lease_plain_01"
    m, k = 4, 4
    np.random.seed(42)
    mat_a = np.random.randn(m, k).astype(np.float32)

    # 1. LeaseMatrix
    lease_req = pb2.LeaseMatrixRequest(
        lease_id=lease_id,
        rows=m,
        cols=k,
        dtype="fp32",
        matrix_bytes=mat_a.tobytes(),
        ttl_seconds=300,
        require_confidential_compute=False,
    )
    lease_resp = servicer.LeaseMatrix(lease_req)

    assert lease_resp.lease_id == lease_id
    assert lease_resp.is_accepted is True
    assert lease_resp.expires_at_unix_ms > int(time.time() * 1000)
    assert lease_resp.worker_pubkey == signer.get_pubkey_hex()

    # 2. StepIterative: Step 0
    v0 = np.random.randn(k).astype(np.float32)
    step_req_0 = pb2.IterativeStepRequest(
        lease_id=lease_id,
        step_index=0,
        vector_bytes=v0.tobytes(),
        vector_length=k,
        dtype="fp32",
        is_encrypted=False,
    )
    step_resp_0 = servicer.StepIterative(step_req_0)

    assert step_resp_0.lease_id == lease_id
    assert step_resp_0.step_index == 0
    assert step_resp_0.vector_length == m

    res_v0 = np.frombuffer(step_resp_0.result_vector_bytes, dtype=np.float32)
    expected_v0 = np.matmul(mat_a, v0)
    np.testing.assert_allclose(res_v0, expected_v0, rtol=1e-5)

    # Verify Secp256k1 worker signature
    res_hash = hashlib.sha256(step_resp_0.result_vector_bytes).hexdigest()
    expected_payload = f"{lease_id}:0:{res_hash}".encode("utf-8")
    assert signer.verify_signature(
        expected_payload,
        step_resp_0.worker_signature,
        signer.get_pubkey_hex(),
    ) is True

    # 3. StepIterative: Step 1 (feeding previous result back)
    step_req_1 = pb2.IterativeStepRequest(
        lease_id=lease_id,
        step_index=1,
        vector_bytes=res_v0.tobytes(),
        vector_length=m,
        dtype="fp32",
        is_encrypted=False,
    )
    step_resp_1 = servicer.StepIterative(step_req_1)
    res_v1 = np.frombuffer(step_resp_1.result_vector_bytes, dtype=np.float32)
    expected_v1 = np.matmul(mat_a, res_v0)
    np.testing.assert_allclose(res_v1, expected_v1, rtol=1e-5)

    # 4. ReleaseLease
    rel_req = pb2.ReleaseLeaseRequest(
        lease_id=lease_id,
        consumer_pubkey="test_consumer",
        reason="Computation complete",
    )
    rel_resp = servicer.ReleaseLease(rel_req)
    assert rel_resp.lease_id == lease_id
    assert rel_resp.is_released is True
    assert rel_resp.bytes_freed == m * k * 4

    # 5. Subsequent StepIterative must fail
    with pytest.raises((RuntimeError, ValueError)):
        servicer.StepIterative(step_req_1)


# ==============================================================================
# 2. 100 Power Iterations Equivalence Test
# ==============================================================================

def test_servicer_100_power_iterations_equivalence(servicer, signer):
    """
    Executes 100 power iterations against the servicer and verifies:
    1. Worker uploads matrix A once via LeaseMatrix.
    2. Streams 100 sequential vector steps v_{t+1} = A * v_t.
    3. Verifies each step's Secp256k1 worker signature.
    4. Proves mathematical equivalence against local NumPy reference.
    5. Cleanly releases lease upon completion.
    """
    np.random.seed(1234)
    dim = 8
    # Generate symmetric positive-definite matrix and scale spectral radius to 1.0
    raw_mat = np.random.randn(dim, dim).astype(np.float64)
    sym_mat = np.dot(raw_mat, raw_mat.T)
    max_eigenvalue = np.max(np.linalg.eigvalsh(sym_mat))
    norm_mat = (sym_mat / max_eigenvalue).astype(np.float32)

    v_init = np.random.randn(dim).astype(np.float32)
    v_init /= np.linalg.norm(v_init)

    lease_id = "power_iter_100_test"

    # Step 1: Lease matrix once
    lease_req = pb2.LeaseMatrixRequest(
        lease_id=lease_id,
        rows=dim,
        cols=dim,
        dtype="fp32",
        matrix_bytes=norm_mat.tobytes(),
        ttl_seconds=600,
    )
    lease_resp = servicer.LeaseMatrix(lease_req)
    assert lease_resp.is_accepted is True

    # Step 2: Run 100 iterations on servicer
    v_curr = v_init.copy()
    v_ref = v_init.copy()

    for step in range(100):
        step_req = pb2.IterativeStepRequest(
            lease_id=lease_id,
            step_index=step,
            vector_bytes=v_curr.tobytes(),
            vector_length=dim,
            dtype="fp32",
        )
        step_resp = servicer.StepIterative(step_req)

        # Verify signature
        res_hash = hashlib.sha256(step_resp.result_vector_bytes).hexdigest()
        msg = f"{lease_id}:{step}:{res_hash}".encode("utf-8")
        assert signer.verify_signature(msg, step_resp.worker_signature, signer.get_pubkey_hex())

        # Update servicer vector
        v_next = np.frombuffer(step_resp.result_vector_bytes, dtype=np.float32)
        norm_val = np.linalg.norm(v_next)
        if norm_val > 0:
            v_curr = v_next / norm_val
        else:
            v_curr = v_next

        # Compute local reference step
        v_ref_next = np.matmul(norm_mat, v_ref)
        ref_norm = np.linalg.norm(v_ref_next)
        if ref_norm > 0:
            v_ref = v_ref_next / ref_norm
        else:
            v_ref = v_ref_next

    # Step 3: Verify mathematical equivalence of 100th iteration
    np.testing.assert_allclose(v_curr, v_ref, rtol=1e-4, atol=1e-4)

    # Step 4: Release lease
    rel_resp = servicer.ReleaseLease(pb2.ReleaseLeaseRequest(lease_id=lease_id))
    assert rel_resp.is_released is True
    assert rel_resp.bytes_freed == dim * dim * 4


# ==============================================================================
# 3. Confidential Enclave Mode Integration Test
# ==============================================================================

def test_servicer_confidential_enclave_mode(servicer, signer):
    """
    Verifies end-to-end confidential computing mode:
    1. Client generates X25519 keypair and requests confidential lease.
    2. Servicer provides worker ephemeral pubkey.
    3. Both derive shared session keys (K_c2w, K_w2c).
    4. Client sends encrypted vector (AES-256-GCM, 12B IV, 16B tag).
    5. Servicer decrypts in enclave, performs GEMV, and returns encrypted result.
    6. Client decrypts and verifies mathematical accuracy against NumPy.
    """
    from cryptography.hazmat.primitives.asymmetric import x25519
    from cryptography.hazmat.primitives import serialization

    # Client ephemeral X25519 keypair
    client_priv = x25519.X25519PrivateKey.generate()
    client_pub_bytes = client_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    lease_id = "enclave_lease_01"
    m, k = 4, 4
    np.random.seed(77)
    mat_a = np.random.randn(m, k).astype(np.float32)

    # Worker ephemeral X25519 keypair for genuine confidential lease upload
    worker_priv = x25519.X25519PrivateKey.generate()
    worker_pub_bytes = worker_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    salt = hashlib.sha256(f"iterative:{lease_id}".encode("utf-8")).digest()
    client_keys = derive_session_keys(
        worker_priv=client_priv,
        client_pub_bytes=worker_pub_bytes,
        client_nonce=salt,
    )
    k_c2w = client_keys.k_c2w
    k_w2c = client_keys.k_w2c

    # Encrypt matrix_a with k_c2w (28-byte wire framing: IV 12B + Tag 16B + Ciphertext)
    ct_mat, iv_mat, tag_mat = aes_gcm_encrypt(key=k_c2w, plaintext=mat_a.tobytes())
    encrypted_matrix_bytes = iv_mat + tag_mat + ct_mat

    # 1. LeaseMatrix with confidential compute requested and encrypted wire framing
    lease_req = pb2.LeaseMatrixRequest(
        lease_id=lease_id,
        rows=m,
        cols=k,
        dtype="fp32",
        matrix_bytes=encrypted_matrix_bytes,
        ttl_seconds=300,
        require_confidential_compute=True,
        client_ephemeral_pubkey=client_pub_bytes,
    )
    from unittest.mock import patch
    with patch("cryptography.hazmat.primitives.asymmetric.x25519.X25519PrivateKey.generate", return_value=worker_priv):
        lease_resp = servicer.LeaseMatrix(lease_req)

    assert lease_resp.is_accepted is True
    assert len(lease_resp.worker_ephemeral_pubkey) == 32
    assert lease_resp.worker_ephemeral_pubkey == worker_pub_bytes

    # 3. Encrypt initial vector v0
    v0 = np.random.randn(k).astype(np.float32)
    ct, iv, tag = aes_gcm_encrypt(key=k_c2w, plaintext=v0.tobytes())

    # 4. Send encrypted step request
    step_req = pb2.IterativeStepRequest(
        lease_id=lease_id,
        step_index=0,
        vector_bytes=ct,
        vector_length=k,
        dtype="fp32",
        is_encrypted=True,
        iv=iv,
        tag=tag,
    )
    step_resp = servicer.StepIterative(step_req)

    assert step_resp.is_encrypted is True
    assert len(step_resp.iv) == 12
    assert len(step_resp.tag) == 16

    # 5. Decrypt result on client side using K_w2c
    decrypted_result_bytes = aes_gcm_decrypt(
        key=k_w2c,
        iv=step_resp.iv,
        ciphertext=step_resp.result_vector_bytes,
        tag=step_resp.tag,
    )
    res_vector = np.frombuffer(decrypted_result_bytes, dtype=np.float32)
    expected = np.matmul(mat_a, v0)

    np.testing.assert_allclose(res_vector, expected, rtol=1e-5)

    # 6. Clean release
    servicer.ReleaseLease(pb2.ReleaseLeaseRequest(lease_id=lease_id))


# ==============================================================================
# 4. L402 Gate Enforcement Tests
# ==============================================================================

def test_lease_matrix_l402_missing_auth_header(servicer):
    """Verifies LeaseMatrix rejects request when L402 header is missing and gate enforces it."""
    ctx = MockGrpcContext(metadata=[])
    req = pb2.LeaseMatrixRequest(
        lease_id="l402_missing",
        rows=4,
        cols=4,
        dtype="fp32",
        matrix_bytes=np.zeros((4, 4), dtype=np.float32).tobytes(),
        ttl_seconds=60,
    )

    with pytest.raises(RuntimeError, match="Missing Authorization|UNAUTHENTICATED"):
        servicer.LeaseMatrix(req, context=ctx)


def test_lease_matrix_l402_unpaid_invoice(servicer, mock_settlement, vkey):
    """Verifies LeaseMatrix rejects request when HODL invoice is UNPAID."""
    lid = "l402_unpaid"
    inv = mock_settlement.create_hodl_invoice(lid)
    mac = mint_macaroon(vkey, lid, int(time.time()) + 3600, payment_hash=inv.payment_hash)

    ctx = MockGrpcContext(metadata=[("authorization", f"L402 {mac}:{inv.payment_hash}")])
    req = pb2.LeaseMatrixRequest(
        lease_id=lid,
        rows=4,
        cols=4,
        dtype="fp32",
        matrix_bytes=np.zeros((4, 4), dtype=np.float32).tobytes(),
        ttl_seconds=60,
    )

    with pytest.raises(RuntimeError, match="UNPAID|PAYMENT_REQUIRED"):
        servicer.LeaseMatrix(req, context=ctx)


def test_lease_matrix_l402_held_invoice_succeeds(servicer, mock_settlement, vkey):
    """Verifies LeaseMatrix succeeds when HODL invoice is in HELD status."""
    lid = "l402_held"
    inv = mock_settlement.create_hodl_invoice(lid)
    mock_settlement.hold_invoice(inv.payment_hash)
    mac = mint_macaroon(vkey, lid, int(time.time()) + 3600, payment_hash=inv.payment_hash)

    ctx = MockGrpcContext(metadata=[("authorization", f"L402 {mac}:{inv.payment_hash}")])
    req = pb2.LeaseMatrixRequest(
        lease_id=lid,
        rows=4,
        cols=4,
        dtype="fp32",
        matrix_bytes=np.zeros((4, 4), dtype=np.float32).tobytes(),
        ttl_seconds=60,
    )

    resp = servicer.LeaseMatrix(req, context=ctx)
    assert resp.is_accepted is True
    assert resp.lease_id == lid


# ==============================================================================
# 5. Error Handling & Edge Cases
# ==============================================================================

def test_step_iterative_unknown_lease_aborts(servicer):
    """Verifies StepIterative aborts with NOT_FOUND / FAILED_PRECONDITION on unknown lease."""
    ctx = MockGrpcContext()
    req = pb2.IterativeStepRequest(
        lease_id="non_existent_lease",
        step_index=0,
        vector_bytes=np.zeros(4, dtype=np.float32).tobytes(),
        vector_length=4,
        dtype="fp32",
    )
    with pytest.raises(RuntimeError, match="not found|expired|FAILED_PRECONDITION"):
        servicer.StepIterative(req, context=ctx)


def test_step_iterative_vector_length_mismatch(servicer):
    """Verifies StepIterative aborts when vector_length does not match leased matrix cols."""
    lid = "mismatch_lease"
    servicer.LeaseMatrix(pb2.LeaseMatrixRequest(
        lease_id=lid,
        rows=4,
        cols=4,
        dtype="fp32",
        matrix_bytes=np.zeros((4, 4), dtype=np.float32).tobytes(),
        ttl_seconds=60,
    ))

    ctx = MockGrpcContext()
    # Matrix expects length 4; pass length 8
    req = pb2.IterativeStepRequest(
        lease_id=lid,
        step_index=0,
        vector_bytes=np.zeros(8, dtype=np.float32).tobytes(),
        vector_length=8,
        dtype="fp32",
    )
    with pytest.raises(RuntimeError, match="dimension mismatch|INVALID_ARGUMENT"):
        servicer.StepIterative(req, context=ctx)


def test_release_lease_non_existent(servicer):
    """Verifies ReleaseLease handles non-existent lease gracefully."""
    req = pb2.ReleaseLeaseRequest(
        lease_id="unknown_release",
        consumer_pubkey="user",
        reason="test",
    )
    resp = servicer.ReleaseLease(req)
    assert resp.is_released is False
    assert resp.bytes_freed == 0
