"""
Unit test suite for matrix-compute-node hardware attestation subsystem.
Covers BaseAttestationProvider, NvidiaCcaProvider, SimulatedAttestationProvider,
registry resolution, parameter validation, fault injection, and servicer integration.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from matrix_contracts.schemas.attestation import AttestationEvidence
from matrix_contracts.v1 import pb2
from src.attestation import (
    AttestationError,
    AttestationGenerationError,
    AttestationValidationError,
    BaseAttestationProvider,
    HardwareAttestationUnavailableError,
    NvidiaCcaProvider,
    SimulatedAttestationProvider,
    SyntheticPKICertificateBundle,
    generate_synthetic_pki,
    get_attestation_provider,
    register_attestation_provider,
)
from src.p2p.servicer import MatrixComputeServicer


def test_base_provider_contract() -> None:
    """Verifies BaseAttestationProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseAttestationProvider()  # type: ignore[abstract]


def test_provider_factory_resolution() -> None:
    """Tests get_attestation_provider factory resolution for various provider types."""
    # Simulated provider
    p1 = get_attestation_provider("simulated-cca")
    assert isinstance(p1, SimulatedAttestationProvider)
    assert p1.provider_name == "simulated-cca"
    assert p1.is_available() is True

    p2 = get_attestation_provider("simulated")
    assert isinstance(p2, SimulatedAttestationProvider)

    # Unknown provider
    with pytest.raises(ValueError, match="Unknown attestation provider type"):
        get_attestation_provider("unknown-enclave-xyz")


def test_simulated_provider_evidence_structure() -> None:
    """Verifies SimulatedAttestationProvider produces valid AttestationEvidence."""
    provider = SimulatedAttestationProvider()
    nonce = secrets.token_bytes(32)
    pubkey = secrets.token_bytes(32)

    evidence = provider.generate_evidence(nonce=nonce, public_key=pubkey)

    assert isinstance(evidence, AttestationEvidence)
    assert evidence.provider == "simulated-cca"
    assert len(evidence.certificate_chain) == 3
    assert evidence.token_claims["nonce"] == nonce.hex()
    assert evidence.token_claims["ephemeral_pubkey_hash"] == hashlib.sha256(pubkey).hexdigest()
    assert evidence.token_claims["public_key"] == pubkey.hex()
    assert evidence.golden_rim == provider.get_measurements()
    assert len(evidence.token_signature) > 0
    assert evidence.evidence_blob is not None
    assert evidence.user_data == hashlib.sha256(nonce + pubkey).digest()


def test_simulated_evidence_serialization_roundtrip() -> None:
    """Verifies that evidence serialize/deserialize preserves all fields without loss."""
    provider = SimulatedAttestationProvider()
    nonce = secrets.token_bytes(32)
    pubkey = secrets.token_bytes(32)

    ev1 = provider.generate_evidence(nonce=nonce, public_key=pubkey)
    raw_bytes = ev1.to_bytes()

    ev2 = AttestationEvidence.from_bytes(raw_bytes)
    assert ev2.provider == ev1.provider
    assert ev2.certificate_chain == ev1.certificate_chain
    assert ev2.token_claims == ev1.token_claims
    assert ev2.token_signature == ev1.token_signature
    assert ev2.golden_rim == ev1.golden_rim
    assert ev2.evidence_blob == ev1.evidence_blob
    assert ev2.user_data == ev1.user_data


def test_simulated_pki_signature_verification() -> None:
    """Verifies that the generated token signature is valid under the leaf cert's public key."""
    provider = SimulatedAttestationProvider()
    nonce = secrets.token_bytes(32)
    pubkey = secrets.token_bytes(32)

    evidence = provider.generate_evidence(nonce=nonce, public_key=pubkey)

    leaf_pem = evidence.certificate_chain[0]
    leaf_cert = x509.load_pem_x509_certificate(leaf_pem.encode("utf-8"))
    leaf_pubkey = leaf_cert.public_key()
    assert isinstance(leaf_pubkey, ec.EllipticCurvePublicKey)

    claims_bytes = json.dumps(evidence.token_claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
    leaf_pubkey.verify(evidence.token_signature, claims_bytes, ec.ECDSA(hashes.SHA256()))


def test_simulated_provider_parameter_validation() -> None:
    """Verifies parameter validation checks on nonce and public key."""
    provider = SimulatedAttestationProvider()
    valid_32 = secrets.token_bytes(32)

    # Invalid nonces
    with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
        provider.generate_evidence(nonce=b"too_short", public_key=valid_32)

    with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
        provider.generate_evidence(nonce=secrets.token_bytes(33), public_key=valid_32)

    # Invalid public keys
    with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
        provider.generate_evidence(nonce=valid_32, public_key=b"bad_len")

    with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
        provider.generate_evidence(nonce=valid_32, public_key=secrets.token_bytes(64))


def test_simulated_provider_fault_injection() -> None:
    """Verifies fault-injection parameters for negative testing."""
    nonce = secrets.token_bytes(32)
    pubkey = secrets.token_bytes(32)

    # 1. Tamper signature
    p_sig = SimulatedAttestationProvider(tamper_signature=True)
    ev_sig = p_sig.generate_evidence(nonce=nonce, public_key=pubkey)
    leaf_cert = x509.load_pem_x509_certificate(ev_sig.certificate_chain[0].encode("utf-8"))
    claims_bytes = json.dumps(ev_sig.token_claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with pytest.raises(Exception):
        leaf_cert.public_key().verify(ev_sig.token_signature, claims_bytes, ec.ECDSA(hashes.SHA256()))  # type: ignore[union-attr]

    # 2. Tamper measurements
    p_meas = SimulatedAttestationProvider(tamper_measurements=True)
    ev_meas = p_meas.generate_evidence(nonce=nonce, public_key=pubkey)
    assert ev_meas.token_claims["measurements"]["runtime_digest"] == "bad0" * 16

    # 3. Tamper nonce
    p_nonce = SimulatedAttestationProvider(tamper_nonce=True)
    ev_nonce = p_nonce.generate_evidence(nonce=nonce, public_key=pubkey)
    assert ev_nonce.token_claims["nonce"] != nonce.hex()

    # 4. Expired cert
    p_exp = SimulatedAttestationProvider(expired_cert=True)
    ev_exp = p_exp.generate_evidence(nonce=nonce, public_key=pubkey)
    cert = x509.load_pem_x509_certificate(ev_exp.certificate_chain[0].encode("utf-8"))
    assert cert.not_valid_after_utc < cert.not_valid_before_utc.replace(year=cert.not_valid_before_utc.year + 2)

    # 5. Untrusted CA
    p_untrusted = SimulatedAttestationProvider(untrusted_ca=True)
    ev_untrusted = p_untrusted.generate_evidence(nonce=nonce, public_key=pubkey)
    root_cert = x509.load_pem_x509_certificate(ev_untrusted.certificate_chain[-1].encode("utf-8"))
    assert "Untrusted Rogue CA" in root_cert.subject.rfc4514_string()


def test_nvidia_cca_provider_mock_available() -> None:
    """Tests NvidiaCcaProvider when NVML CC mode is ON."""
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    mock_handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
    mock_status = MagicMock()
    mock_status.ccMode = 1
    mock_pynvml.nvmlDeviceGetConfidentialComputeStatus.return_value = mock_status
    mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA H100 80GB HBM3"
    mock_pynvml.nvmlDeviceGetConfidentialComputeGpuAttestationReport.return_value = b"MOCK_REPORT_BLOB_CCA"
    mock_pynvml.nvmlDeviceGetConfidentialComputeGpuCertificate.return_value = [
        "-----BEGIN CERTIFICATE-----\nMOCK_LEAF\n-----END CERTIFICATE-----",
        "-----BEGIN CERTIFICATE-----\nMOCK_ROOT\n-----END CERTIFICATE-----",
    ]

    with patch("src.attestation.nvml_attest._get_pynvml", return_value=mock_pynvml):
        provider = NvidiaCcaProvider(device_index=0)
        assert provider.is_available() is True
        assert provider.provider_name == "nvidia-cca"

        nonce = secrets.token_bytes(32)
        pubkey = secrets.token_bytes(32)
        evidence = provider.generate_evidence(nonce=nonce, public_key=pubkey)

        assert evidence.provider == "nvidia-cca"
        assert evidence.token_claims["gpu_model"] == "NVIDIA H100 80GB HBM3"
        assert evidence.token_claims["nonce"] == nonce.hex()
        assert evidence.evidence_blob == b"MOCK_REPORT_BLOB_CCA"
        assert len(evidence.certificate_chain) == 2


def test_nvidia_cca_provider_cc_disabled_unavailable() -> None:
    """Tests NvidiaCcaProvider when CC mode is OFF."""
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    mock_handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
    mock_status = MagicMock()
    mock_status.ccMode = 0  # OFF
    mock_pynvml.nvmlDeviceGetConfidentialComputeStatus.return_value = mock_status

    with patch("src.attestation.nvml_attest._get_pynvml", return_value=mock_pynvml):
        provider = NvidiaCcaProvider(device_index=0)
        assert provider.is_available() is False

        nonce = secrets.token_bytes(32)
        pubkey = secrets.token_bytes(32)
        with pytest.raises(HardwareAttestationUnavailableError):
            provider.generate_evidence(nonce=nonce, public_key=pubkey)


def test_nvidia_cca_provider_missing_library_fallback() -> None:
    """Tests that missing NVML gracefully reports unavailable and falls back."""
    with patch("src.attestation.nvml_attest._get_pynvml", return_value=None):
        provider = NvidiaCcaProvider(device_index=0)
        assert provider.is_available() is False

        # Factory with fallback_to_simulated=True
        fallback_p = get_attestation_provider("nvidia-cca", fallback_to_simulated=True)
        assert isinstance(fallback_p, SimulatedAttestationProvider)

        # Factory with fallback_to_simulated=False
        with pytest.raises(HardwareAttestationUnavailableError):
            get_attestation_provider("nvidia-cca", fallback_to_simulated=False)


def test_servicer_confidential_handshake() -> None:
    """Verifies that MatrixComputeServicer handles confidential compute handshake."""
    provider = SimulatedAttestationProvider()
    servicer = MatrixComputeServicer(attestation_provider=provider)

    nonce = secrets.token_bytes(32)
    client_pubkey = secrets.token_bytes(32)

    req = pb2.TileTaskRequest(
        task_id="task-cc-001",
        matrix_a_rows=64,
        matrix_a_cols=64,
        matrix_b_cols=64,
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=client_pubkey,
        attestation_nonce=nonce,
        required_tee_type="simulated-cca",
    )

    ack = servicer.RequestTileTask(req)
    assert ack.is_ready is True
    assert len(ack.worker_ephemeral_pubkey) == 32
    assert len(ack.attestation_evidence) > 0
    assert ack.tee_type == "simulated-cca"

    # Validate that attestation evidence binds to the nonce and worker ephemeral pubkey
    ev = AttestationEvidence.from_bytes(ack.attestation_evidence)
    assert ev.token_claims["nonce"] == nonce.hex()
    worker_pubkey_hash = hashlib.sha256(ack.worker_ephemeral_pubkey).hexdigest()
    assert ev.token_claims["ephemeral_pubkey_hash"] == worker_pubkey_hash


def test_servicer_unencrypted_fallback() -> None:
    """Verifies unencrypted backwards compatibility when require_confidential_compute=False."""
    servicer = MatrixComputeServicer(attestation_provider=SimulatedAttestationProvider())

    req = pb2.TileTaskRequest(
        task_id="task-plain-001",
        matrix_a_rows=32,
        matrix_a_cols=32,
        matrix_b_cols=32,
        dtype="fp32",
        require_confidential_compute=False,
    )

    ack = servicer.RequestTileTask(req)
    assert ack.is_ready is True
    assert ack.worker_ephemeral_pubkey == b""
    assert ack.attestation_evidence == b""
    assert ack.tee_type == ""


def test_servicer_confidential_validation_errors() -> None:
    """Verifies invalid confidential parameters are rejected fail-closed."""
    provider = SimulatedAttestationProvider()
    servicer = MatrixComputeServicer(attestation_provider=provider)

    valid_32 = secrets.token_bytes(32)

    # 1. Invalid client_ephemeral_pubkey length
    req1 = pb2.TileTaskRequest(
        task_id="task-err-1",
        matrix_a_rows=32,
        matrix_a_cols=32,
        matrix_b_cols=32,
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=b"short",
        attestation_nonce=valid_32,
    )
    with pytest.raises(ValueError, match="client_ephemeral_pubkey must be exactly 32 bytes"):
        servicer.RequestTileTask(req1)

    # 2. Invalid attestation_nonce length
    req2 = pb2.TileTaskRequest(
        task_id="task-err-2",
        matrix_a_rows=32,
        matrix_a_cols=32,
        matrix_b_cols=32,
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=valid_32,
        attestation_nonce=b"invalid_nonce",
    )
    with pytest.raises(ValueError, match="attestation_nonce must be exactly 32 bytes"):
        servicer.RequestTileTask(req2)

    # 3. Mismatched TEE type
    req3 = pb2.TileTaskRequest(
        task_id="task-err-3",
        matrix_a_rows=32,
        matrix_a_cols=32,
        matrix_b_cols=32,
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=valid_32,
        attestation_nonce=valid_32,
        required_tee_type="intel-tdx",
    )
    with pytest.raises(ValueError, match="Worker TEE type 'simulated-cca' does not match requested 'intel-tdx'"):
        servicer.RequestTileTask(req3)

    # 4. Servicer without attestation provider
    servicer_no_tee = MatrixComputeServicer(attestation_provider=None)
    req4 = pb2.TileTaskRequest(
        task_id="task-err-4",
        matrix_a_rows=32,
        matrix_a_cols=32,
        matrix_b_cols=32,
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=valid_32,
        attestation_nonce=valid_32,
    )
    with pytest.raises(RuntimeError, match="Confidential computing requested but worker has no operational attestation provider"):
        servicer_no_tee.RequestTileTask(req4)
