"""
Adversarial Challenge & Stress Test Suite for Milestone 2:
`matrix-compute-node` Hardware Attestation Subsystem & Servicer Integration.

Challenger: challenger_m2_1 (role: teamwork_preview_challenger)

Empirically tests four adversarial challenge vectors:
1. Evidence serialization/deserialization integrity across binary blobs.
2. Boundary conditions: 31-byte nonces, 33-byte nonces, invalid pubkeys.
3. Simulated attestation provider under concurrency.
4. RequestTileTask error paths (missing nonce, missing pubkey, unencrypted fallback).
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import secrets
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

import grpc
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from matrix_contracts.schemas.attestation import AttestationEvidence, _coerce_bytes_payload
from matrix_contracts.v1 import pb2
from src.attestation.base import (
    AttestationError,
    AttestationValidationError,
    BaseAttestationProvider,
    HardwareAttestationUnavailableError,
    get_attestation_provider,
)
from src.attestation.simulated import (
    DEFAULT_SIMULATED_MEASUREMENTS,
    SimulatedAttestationProvider,
    SyntheticPKICertificateBundle,
    generate_synthetic_pki,
)
from src.p2p.servicer import MatrixComputeServicer, TaskMetadataStore


# ==============================================================================
# CHALLENGE 1: EVIDENCE SERIALIZATION / DESERIALIZATION ACROSS BINARY BLOBS
# ==============================================================================

class TestEvidenceBinarySerializationIntegrity:
    """Stress-tests AttestationEvidence serialization across extreme binary payloads."""

    def test_roundtrip_empty_and_null_binary_fields(self) -> None:
        """Verifies handling of None vs empty bytes across all binary fields."""
        ev_none = AttestationEvidence(
            provider="simulated-cca",
            certificate_chain=["cert1"],
            token_claims={"nonce": "00" * 32},
            token_signature=b"",
            evidence_blob=None,
            user_data=None,
        )
        data_none = ev_none.to_bytes()
        res_none = AttestationEvidence.from_bytes(data_none)
        assert res_none.token_signature == b""
        assert res_none.evidence_blob is None
        assert res_none.user_data is None

        ev_empty = AttestationEvidence(
            provider="simulated-cca",
            certificate_chain=["cert1"],
            token_claims={"nonce": "00" * 32},
            token_signature=b"",
            evidence_blob=b"",
            user_data=b"",
        )
        data_empty = ev_empty.to_bytes()
        res_empty = AttestationEvidence.from_bytes(data_empty)
        assert res_empty.token_signature == b""
        assert res_empty.evidence_blob == b""
        assert res_empty.user_data == b""

    @pytest.mark.parametrize("payload_size", [1, 15, 16, 32, 64, 256, 1024, 65536])
    def test_roundtrip_arbitrary_binary_blobs(self, payload_size: int) -> None:
        """Verifies exact binary preservation of diverse payload sizes containing all byte values."""
        # Include full 0x00..0xFF byte spectrum
        raw_blob = bytes([i % 256 for i in range(payload_size)])
        raw_sig = secrets.token_bytes(64)
        raw_user_data = secrets.token_bytes(32)

        ev = AttestationEvidence(
            provider="simulated-cca",
            certificate_chain=["-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----"],
            token_claims={"nonce": "aa" * 32, "size": payload_size},
            token_signature=raw_sig,
            evidence_blob=raw_blob,
            user_data=raw_user_data,
        )

        serialized = ev.to_bytes()
        assert isinstance(serialized, bytes)

        recovered = AttestationEvidence.from_bytes(serialized)
        assert recovered.provider == "simulated-cca"
        assert recovered.token_signature == raw_sig
        assert recovered.evidence_blob == raw_blob
        assert recovered.user_data == raw_user_data
        assert recovered.token_claims["size"] == payload_size

    def test_binary_blob_with_null_and_high_ascii_bytes(self) -> None:
        """Verifies evidence preserves null bytes, high-bit bytes, and whitespace bytes."""
        tricky_blob = b"\x00\x00\x00\xff\xfe\xfd\x20\x09\x0a\x0d\x7f\x80\x81\x00\xff"
        ev = AttestationEvidence(
            provider="simulated-cca",
            token_signature=tricky_blob,
            evidence_blob=tricky_blob,
            user_data=tricky_blob,
        )
        serialized = ev.to_bytes()
        recovered = AttestationEvidence.from_bytes(serialized)
        assert recovered.token_signature == tricky_blob
        assert recovered.evidence_blob == tricky_blob
        assert recovered.user_data == tricky_blob

    def test_deserialization_from_hex_and_base64_strings(self) -> None:
        """Verifies _coerce_bytes_payload accepts valid hex (with and without 0x) and base64."""
        test_bytes = b"CONFIDENTIAL_COMPUTE_PAYLOAD_32B"

        # 1. Plain hex
        ev1 = AttestationEvidence(
            provider="simulated-cca",
            token_signature=test_bytes.hex(),
            evidence_blob=test_bytes.hex(),
        )
        assert ev1.token_signature == test_bytes
        assert ev1.evidence_blob == test_bytes

        # 2. 0x-prefixed hex
        ev2 = AttestationEvidence(
            provider="simulated-cca",
            token_signature="0x" + test_bytes.hex(),
            evidence_blob="0X" + test_bytes.hex(),
        )
        assert ev2.token_signature == test_bytes
        assert ev2.evidence_blob == test_bytes

        # 3. Base64 encoded string
        b64_str = base64.b64encode(test_bytes).decode("ascii")
        ev3 = AttestationEvidence(
            provider="simulated-cca",
            token_signature=b64_str,
            evidence_blob=b64_str,
        )
        assert ev3.token_signature == test_bytes
        assert ev3.evidence_blob == test_bytes

    def test_malformed_binary_deserialization_rejection(self) -> None:
        """Verifies fail-closed rejection of corrupted, truncated, or invalid binary blobs."""
        # 1. Truncated / malformed JSON
        with pytest.raises(Exception):
            AttestationEvidence.from_bytes(b'{"provider": "simulated-cca", "token_signature": ')

        # 2. Invalid UTF-8 bytes to from_bytes
        with pytest.raises(UnicodeDecodeError):
            AttestationEvidence.from_bytes(b"\x80\xff\xfe\x00")

        # 3. Odd-length hex string for token_signature
        with pytest.raises(ValueError, match="odd length"):
            AttestationEvidence(provider="simulated-cca", token_signature="0x123")

        with pytest.raises(ValueError, match="odd length"):
            AttestationEvidence(provider="simulated-cca", token_signature="abc")

        # 4. Invalid types (int, float, bool)
        with pytest.raises(ValueError, match="Expected bytes or hex/base64 string"):
            AttestationEvidence(provider="simulated-cca", token_signature=12345)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Expected bytes or hex/base64 string"):
            AttestationEvidence(provider="simulated-cca", token_signature=True)  # type: ignore[arg-type]

    def test_tampered_serialized_bytes_invalidates_leaf_signature(self) -> None:
        """Verifies that flipping a single byte in serialized JSON corrupts signature verification."""
        provider = SimulatedAttestationProvider()
        nonce = secrets.token_bytes(32)
        pubkey = secrets.token_bytes(32)

        ev = provider.generate_evidence(nonce=nonce, public_key=pubkey)
        raw_json_str = ev.to_bytes().decode("utf-8")
        data_dict = json.loads(raw_json_str)

        # Tamper with the claims: alter security_version from 1 to 2
        data_dict["token_claims"]["security_version"] = 2
        tampered_bytes = json.dumps(data_dict).encode("utf-8")

        tampered_ev = AttestationEvidence.from_bytes(tampered_bytes)
        leaf_cert = x509.load_pem_x509_certificate(tampered_ev.certificate_chain[0].encode("utf-8"))

        # Re-canonicalize claims
        canonical_claims = json.dumps(
            tampered_ev.token_claims, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        # Signature verification MUST fail
        with pytest.raises(Exception):
            leaf_cert.public_key().verify(  # type: ignore[union-attr]
                tampered_ev.token_signature,
                canonical_claims,
                ec.ECDSA(hashes.SHA256()),
            )


# ==============================================================================
# CHALLENGE 2: BOUNDARY CONDITIONS: 31-BYTE, 33-BYTE NONCES & INVALID PUBKEYS
# ==============================================================================

class TestAttestationBoundaryConditions:
    """Stress-tests boundary conditions for challenger nonces and ephemeral public keys."""

    @pytest.fixture
    def provider(self) -> SimulatedAttestationProvider:
        return SimulatedAttestationProvider()

    @pytest.fixture
    def servicer(self, provider: SimulatedAttestationProvider) -> MatrixComputeServicer:
        return MatrixComputeServicer(attestation_provider=provider)

    @pytest.mark.parametrize(
        "invalid_nonce_len",
        [0, 1, 16, 30, 31, 33, 34, 64, 128, 1024],
    )
    def test_provider_rejects_non_32_byte_nonces(
        self, provider: SimulatedAttestationProvider, invalid_nonce_len: int
    ) -> None:
        """Verifies SimulatedAttestationProvider rejects nonces of length != 32."""
        invalid_nonce = secrets.token_bytes(invalid_nonce_len)
        valid_pubkey = secrets.token_bytes(32)

        with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
            provider.generate_evidence(nonce=invalid_nonce, public_key=valid_pubkey)

    @pytest.mark.parametrize(
        "invalid_pubkey_len",
        [0, 1, 16, 30, 31, 33, 34, 64, 128, 1024],
    )
    def test_provider_rejects_non_32_byte_pubkeys(
        self, provider: SimulatedAttestationProvider, invalid_pubkey_len: int
    ) -> None:
        """Verifies SimulatedAttestationProvider rejects public keys of length != 32."""
        valid_nonce = secrets.token_bytes(32)
        invalid_pubkey = secrets.token_bytes(invalid_pubkey_len)

        with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
            provider.generate_evidence(nonce=valid_nonce, public_key=invalid_pubkey)

    def test_provider_rejects_non_bytes_nonce_and_pubkey(
        self, provider: SimulatedAttestationProvider
    ) -> None:
        """Verifies string, integer, and None inputs are rejected with AttestationValidationError."""
        valid_bytes = secrets.token_bytes(32)

        # String of length 32
        str_32 = "a" * 32
        with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
            provider.generate_evidence(nonce=str_32, public_key=valid_bytes)  # type: ignore[arg-type]

        with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
            provider.generate_evidence(nonce=valid_bytes, public_key=str_32)  # type: ignore[arg-type]

        # None inputs
        with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
            provider.generate_evidence(nonce=None, public_key=valid_bytes)  # type: ignore[arg-type]

        with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
            provider.generate_evidence(nonce=valid_bytes, public_key=None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_nonce_len", [0, 1, 31, 33, 64])
    def test_servicer_rejects_invalid_nonce_lengths(
        self, servicer: MatrixComputeServicer, invalid_nonce_len: int
    ) -> None:
        """Verifies Servicer.RequestTileTask fails closed on 31-byte, 33-byte, and other invalid nonces."""
        req = pb2.TileTaskRequest(
            task_id=f"boundary-nonce-{invalid_nonce_len}",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp32",
            require_confidential_compute=True,
            client_ephemeral_pubkey=secrets.token_bytes(32),
            attestation_nonce=secrets.token_bytes(invalid_nonce_len),
        )
        with pytest.raises(ValueError, match="attestation_nonce must be exactly 32 bytes"):
            servicer.RequestTileTask(req)

    @pytest.mark.parametrize("invalid_pubkey_len", [0, 1, 31, 33, 64])
    def test_servicer_rejects_invalid_pubkey_lengths(
        self, servicer: MatrixComputeServicer, invalid_pubkey_len: int
    ) -> None:
        """Verifies Servicer.RequestTileTask fails closed on 31-byte, 33-byte, and other invalid pubkeys."""
        req = pb2.TileTaskRequest(
            task_id=f"boundary-pubkey-{invalid_pubkey_len}",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp32",
            require_confidential_compute=True,
            client_ephemeral_pubkey=secrets.token_bytes(invalid_pubkey_len),
            attestation_nonce=secrets.token_bytes(32),
        )
        with pytest.raises(ValueError, match="client_ephemeral_pubkey must be exactly 32 bytes"):
            servicer.RequestTileTask(req)

    def test_boundary_weak_and_edge_case_32_byte_keys(
        self, provider: SimulatedAttestationProvider, servicer: MatrixComputeServicer
    ) -> None:
        """
        Verifies behavior with edge-case 32-byte patterns: all zeroes (0x00*32),
        all ones (0xFF*32), and sequential bytes.
        """
        all_zeros = b"\x00" * 32
        all_ones = b"\xff" * 32

        # 1. Provider handles edge-case byte patterns cleanly
        ev1 = provider.generate_evidence(nonce=all_zeros, public_key=all_ones)
        assert ev1.token_claims["nonce"] == "00" * 32
        assert ev1.token_claims["public_key"] == "ff" * 32

        # 2. Servicer processes valid 32-byte edge patterns cleanly in Step 1
        req = pb2.TileTaskRequest(
            task_id="boundary-zeros-ones",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp32",
            require_confidential_compute=True,
            client_ephemeral_pubkey=all_ones,
            attestation_nonce=all_zeros,
        )
        ack = servicer.RequestTileTask(req)
        assert ack.is_ready is True
        assert len(ack.worker_ephemeral_pubkey) == 32

        # 3. Servicer rejects weak null-point key (all zeros) for X25519
        req_null = pb2.TileTaskRequest(
            task_id="boundary-null-key",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp32",
            require_confidential_compute=True,
            client_ephemeral_pubkey=all_zeros,
            attestation_nonce=all_ones,
        )
        with pytest.raises(ValueError, match="Enclave key exchange failed"):
            servicer.RequestTileTask(req_null)


# ==============================================================================
# CHALLENGE 3: SIMULATED ATTESTATION PROVIDER UNDER CONCURRENCY
# ==============================================================================

class TestAttestationProviderConcurrency:
    """Stress-tests SimulatedAttestationProvider and Servicer under high thread concurrency."""

    def test_concurrent_evidence_generation_thread_safety(self) -> None:
        """
        Executes 16 concurrent threads generating attestation evidence (20 iterations each).
        Verifies:
        1. No thread interference or race conditions.
        2. Every evidence has unique nonce, pubkey, and valid ECDSA signature.
        3. No cross-talk of claims or user_data.
        """
        provider = SimulatedAttestationProvider()
        num_threads = 16
        iterations_per_thread = 20
        total_requests = num_threads * iterations_per_thread

        results: List[Dict[str, Any]] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker_task(thread_id: int) -> None:
            for i in range(iterations_per_thread):
                nonce = secrets.token_bytes(32)
                pubkey = secrets.token_bytes(32)
                try:
                    ev = provider.generate_evidence(nonce=nonce, public_key=pubkey)
                    with lock:
                        results.append({
                            "thread_id": thread_id,
                            "iteration": i,
                            "nonce": nonce,
                            "pubkey": pubkey,
                            "evidence": ev,
                        })
                except Exception as exc:
                    with lock:
                        errors.append(exc)

        threads = [
            threading.Thread(target=worker_task, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Encountered {len(errors)} concurrency errors: {errors}"
        assert len(results) == total_requests

        # Verify cryptographic integrity of every generated evidence
        leaf_cert = x509.load_pem_x509_certificate(
            provider.pki_bundle.leaf_cert_pem.encode("utf-8")
        )
        leaf_pubkey = leaf_cert.public_key()

        seen_nonces = set()
        seen_pubkeys = set()

        for item in results:
            ev: AttestationEvidence = item["evidence"]
            nonce: bytes = item["nonce"]
            pubkey: bytes = item["pubkey"]

            # Check uniqueness
            assert nonce not in seen_nonces
            assert pubkey not in seen_pubkeys
            seen_nonces.add(nonce)
            seen_pubkeys.add(pubkey)

            # Check claims binding
            assert ev.token_claims["nonce"] == nonce.hex()
            assert ev.token_claims["public_key"] == pubkey.hex()
            assert ev.token_claims["ephemeral_pubkey_hash"] == hashlib.sha256(pubkey).hexdigest()
            assert ev.user_data == hashlib.sha256(nonce + pubkey).digest()

            # Verify leaf certificate ECDSA signature
            canonical_claims = json.dumps(
                ev.token_claims, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            leaf_pubkey.verify(  # type: ignore[union-attr]
                ev.token_signature,
                canonical_claims,
                ec.ECDSA(hashes.SHA256()),
            )

    def test_concurrent_servicer_request_tile_task(self) -> None:
        """
        Executes concurrent RequestTileTask requests against MatrixComputeServicer.
        Verifies:
        1. Thread-safe ephemeral key generation (unique worker pubkey per task).
        2. Thread-safe TaskMetadataStore insertions.
        3. Zero deadlocks or task collision.
        """
        provider = SimulatedAttestationProvider()
        servicer = MatrixComputeServicer(attestation_provider=provider)
        num_tasks = 100

        def request_job(task_idx: int) -> pb2.TileTaskAck:
            req = pb2.TileTaskRequest(
                task_id=f"concurrent-task-{task_idx}",
                matrix_a_rows=64,
                matrix_a_cols=64,
                matrix_b_cols=64,
                dtype="fp32",
                require_confidential_compute=True,
                client_ephemeral_pubkey=secrets.token_bytes(32),
                attestation_nonce=secrets.token_bytes(32),
            )
            return servicer.RequestTileTask(req)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(request_job, i) for i in range(num_tasks)]
            acks = [f.result() for f in futures]

        assert len(acks) == num_tasks
        worker_pubkeys = set()
        for ack in acks:
            assert ack.is_ready is True
            assert len(ack.worker_ephemeral_pubkey) == 32
            assert ack.worker_ephemeral_pubkey not in worker_pubkeys
            worker_pubkeys.add(ack.worker_ephemeral_pubkey)

        assert len(worker_pubkeys) == num_tasks

    def test_task_metadata_store_bounded_lru_and_eviction_under_concurrency(self) -> None:
        """Verifies TaskMetadataStore enforces max_size and TTL under high concurrent writes."""
        store = TaskMetadataStore(max_size=30, ttl_seconds=0.1)

        def write_entries(thread_id: int) -> None:
            for i in range(50):
                store.set(f"t-{thread_id}-{i}", {"val": i, "created_at": time.time()})

        threads = [threading.Thread(target=write_entries, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Size must never exceed max_size
        assert len(store) <= 30

        # After TTL elapses, evict_expired empties the store
        time.sleep(0.15)
        store.evict_expired()
        assert len(store) == 0


# ==============================================================================
# CHALLENGE 4: REQUESTTILETASK ERROR PATHS & UNENCRYPTED FALLBACK
# ==============================================================================

class TestRequestTileTaskErrorPaths:
    """Stress-tests all error and fallback paths in Servicer.RequestTileTask."""

    def test_error_path_missing_attestation_provider(self) -> None:
        """Verifies FAILED_PRECONDITION when confidential compute is requested without a provider."""
        servicer = MatrixComputeServicer(attestation_provider=None)
        req = pb2.TileTaskRequest(
            task_id="missing-provider-task",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp32",
            require_confidential_compute=True,
            client_ephemeral_pubkey=secrets.token_bytes(32),
            attestation_nonce=secrets.token_bytes(32),
        )

        # Without context (local call): raises RuntimeError
        with pytest.raises(RuntimeError, match="Confidential computing requested but worker has no operational attestation provider"):
            servicer.RequestTileTask(req)

        # With mock gRPC context: invokes context.abort with FAILED_PRECONDITION
        mock_ctx = MagicMock(spec=grpc.ServicerContext)
        with pytest.raises(RuntimeError):
            servicer.RequestTileTask(req, context=mock_ctx)
        mock_ctx.abort.assert_called_once_with(
            grpc.StatusCode.FAILED_PRECONDITION,
            "Confidential computing requested but worker has no operational attestation provider",
        )

    def test_error_path_provider_reports_unavailable(self) -> None:
        """Verifies FAILED_PRECONDITION when provider.is_available() is False."""
        mock_provider = MagicMock(spec=BaseAttestationProvider)
        mock_provider.is_available.return_value = False
        mock_provider.provider_name = "mock-hardware-cca"

        servicer = MatrixComputeServicer(attestation_provider=mock_provider)
        req = pb2.TileTaskRequest(
            task_id="unavailable-provider-task",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp32",
            require_confidential_compute=True,
            client_ephemeral_pubkey=secrets.token_bytes(32),
            attestation_nonce=secrets.token_bytes(32),
        )

        mock_ctx = MagicMock(spec=grpc.ServicerContext)
        with pytest.raises(RuntimeError, match="no operational attestation provider"):
            servicer.RequestTileTask(req, context=mock_ctx)
        mock_ctx.abort.assert_called_once_with(
            grpc.StatusCode.FAILED_PRECONDITION,
            "Confidential computing requested but worker has no operational attestation provider",
        )

    def test_error_path_tee_type_mismatch(self) -> None:
        """Verifies UNIMPLEMENTED when client requests a specific TEE type worker does not have."""
        provider = SimulatedAttestationProvider()
        servicer = MatrixComputeServicer(attestation_provider=provider)

        req = pb2.TileTaskRequest(
            task_id="tee-mismatch-task",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp32",
            require_confidential_compute=True,
            client_ephemeral_pubkey=secrets.token_bytes(32),
            attestation_nonce=secrets.token_bytes(32),
            required_tee_type="nvidia-cca",  # Provider is simulated-cca
        )

        mock_ctx = MagicMock(spec=grpc.ServicerContext)
        with pytest.raises(ValueError, match="Worker TEE type 'simulated-cca' does not match requested 'nvidia-cca'"):
            servicer.RequestTileTask(req, context=mock_ctx)
        mock_ctx.abort.assert_called_once_with(
            grpc.StatusCode.UNIMPLEMENTED,
            "Worker TEE type 'simulated-cca' does not match requested 'nvidia-cca'",
        )

    def test_error_path_missing_nonce_or_pubkey(self) -> None:
        """Verifies INVALID_ARGUMENT when client omits nonce or ephemeral pubkey."""
        provider = SimulatedAttestationProvider()
        servicer = MatrixComputeServicer(attestation_provider=provider)

        # Missing client_ephemeral_pubkey (empty bytes)
        req_no_pubkey = pb2.TileTaskRequest(
            task_id="no-pubkey-task",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp32",
            require_confidential_compute=True,
            client_ephemeral_pubkey=b"",
            attestation_nonce=secrets.token_bytes(32),
        )
        mock_ctx = MagicMock(spec=grpc.ServicerContext)
        with pytest.raises(ValueError, match="client_ephemeral_pubkey must be exactly 32 bytes"):
            servicer.RequestTileTask(req_no_pubkey, context=mock_ctx)
        mock_ctx.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "client_ephemeral_pubkey must be exactly 32 bytes",
        )

        # Missing attestation_nonce (empty bytes)
        req_no_nonce = pb2.TileTaskRequest(
            task_id="no-nonce-task",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp32",
            require_confidential_compute=True,
            client_ephemeral_pubkey=secrets.token_bytes(32),
            attestation_nonce=b"",
        )
        mock_ctx2 = MagicMock(spec=grpc.ServicerContext)
        with pytest.raises(ValueError, match="attestation_nonce must be exactly 32 bytes"):
            servicer.RequestTileTask(req_no_nonce, context=mock_ctx2)
        mock_ctx2.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "attestation_nonce must be exactly 32 bytes",
        )

    def test_error_path_invalid_matrix_dimensions_and_dtype(self) -> None:
        """Verifies INVALID_ARGUMENT for zero or negative dimensions and invalid dtypes."""
        servicer = MatrixComputeServicer(attestation_provider=SimulatedAttestationProvider())

        # Zero rows
        req_dim = pb2.TileTaskRequest(
            task_id="invalid-dim-1",
            matrix_a_rows=0,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp32",
        )
        mock_ctx = MagicMock(spec=grpc.ServicerContext)
        with pytest.raises(ValueError, match="Matrix dimensions must be positive"):
            servicer.RequestTileTask(req_dim, context=mock_ctx)
        mock_ctx.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "Matrix dimensions must be positive integers",
        )

        # Invalid dtype
        req_dtype = pb2.TileTaskRequest(
            task_id="invalid-dtype-1",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="int64",
        )
        mock_ctx2 = MagicMock(spec=grpc.ServicerContext)
        with pytest.raises(ValueError, match="Unsupported dtype: int64"):
            servicer.RequestTileTask(req_dtype, context=mock_ctx2)
        mock_ctx2.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "Unsupported dtype: int64",
        )

    def test_unencrypted_fallback_succeeds_without_attestation_provider(self) -> None:
        """
        Verifies unencrypted backwards compatibility: when require_confidential_compute=False,
        the request succeeds even when no attestation provider is present on the worker.
        """
        servicer = MatrixComputeServicer(attestation_provider=None)

        req = pb2.TileTaskRequest(
            task_id="plain-task-no-provider",
            matrix_a_rows=64,
            matrix_a_cols=128,
            matrix_b_cols=64,
            dtype="fp32",
            require_confidential_compute=False,
        )

        ack = servicer.RequestTileTask(req)
        assert ack.is_ready is True
        assert ack.worker_ephemeral_pubkey == b""
        assert ack.attestation_evidence == b""
        assert ack.tee_type == ""
        assert ack.estimated_gflops > 0.0

        # Verify metadata store record reflects non-confidential task
        meta = servicer._task_metadata["plain-task-no-provider"]
        assert meta["is_confidential"] is False
        assert "worker_ephemeral_priv" not in meta

    def test_unencrypted_fallback_with_provider_present(self) -> None:
        """
        Verifies unencrypted fallback when an attestation provider IS present:
        it must NOT generate ephemeral keys or attestation evidence when require_confidential_compute=False.
        """
        provider = SimulatedAttestationProvider()
        servicer = MatrixComputeServicer(attestation_provider=provider)

        req = pb2.TileTaskRequest(
            task_id="plain-task-with-provider",
            matrix_a_rows=32,
            matrix_a_cols=32,
            matrix_b_cols=32,
            dtype="fp16",
            require_confidential_compute=False,
        )

        ack = servicer.RequestTileTask(req)
        assert ack.is_ready is True
        assert ack.worker_ephemeral_pubkey == b""
        assert ack.attestation_evidence == b""
        assert ack.tee_type == ""

        meta = servicer._task_metadata["plain-task-with-provider"]
        assert meta["is_confidential"] is False
        assert "worker_ephemeral_priv" not in meta
