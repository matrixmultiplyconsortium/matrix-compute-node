"""
Comprehensive Unit Test Suite for Intel TDX Attestation Provider (matrix-compute-node).

Verifies:
1. IntelTdxProvider contract, registry registration, constants, and ctypes structure sizes.
2. Factory instantiation and alias resolution ("intel-tdx", "tdx", "intel", case-insensitivity).
3. Availability probing and safe non-hardware error handling.
4. Fail-closed parameter validation for nonces, public keys, and device indices.
5. Evidence generation structure, token claims, and 64-byte REPORTDATA dual-field binding.
6. Roundtrip serialization via AttestationEvidence.to_bytes() and from_bytes().
7. Direct verification with matrix_compute.crypto.IntelTdxVerifier (component-level).
8. Full end-to-end appraisal with matrix_compute.crypto.AttestationVerifier policy pipeline.
9. Single-bit signature tampering, nonce mismatch, and pubkey substitution rejection.
10. MatrixComputeServicer gRPC confidential computing handshake integration.
11. NodeConfig settings CLI and environment variable resolution.
12. AF_VSOCK QGS socket framing and error code handling.
13. Sensitive buffer zeroization using zeroize_buffer.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import secrets
import struct
from unittest.mock import MagicMock, mock_open, patch

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
    DEFAULT_TDX_GOLDEN_RIM,
    HardwareAttestationUnavailableError,
    INTEL_TDX_GOLDEN_RIM,
    IntelTdxProvider,
    TDREPORT_STRUCT_LEN,
    TDX_CMD_GET_QUOTE,
    TDX_CMD_GET_REPORT0,
    TDX_GUEST_DEV_NODES,
    TDX_REPORTDATA_LEN,
    TdxQuoteReq,
    TdxReportReq,
    VSOCK_HOST_CID,
    VSOCK_QGS_PORT,
    get_attestation_provider,
    parse_tdx_quote_binary,
    tdx_quote_req,
    tdx_report_req,
    zeroize_buffer,
)
from src.config.settings import NodeConfig
from src.p2p.servicer import MatrixComputeServicer

# Import SDK verification components for direct verification
from matrix_compute.crypto import (
    AttestationVerifier,
    IntelTdxVerifier,
    VerificationPolicy,
    VerificationResult,
    get_default_intel_pki,
)
from matrix_compute.exceptions import (
    AttestationBindingError,
    AttestationNonceMismatchError,
    AttestationSignatureError,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def intel_pki():
    """Cached singleton synthetic Intel PKI hierarchy."""
    return get_default_intel_pki()


@pytest.fixture
def sample_nonce() -> bytes:
    """32-byte cryptographically secure random nonce."""
    return secrets.token_bytes(32)


@pytest.fixture
def sample_pubkey() -> bytes:
    """32-byte ephemeral public key."""
    return secrets.token_bytes(32)


# ==============================================================================
# 1. CONTRACT, CONSTANTS & CTYPES STRUCTURES
# ==============================================================================

def test_intel_tdx_provider_contract() -> None:
    """Verifies IntelTdxProvider inherits from BaseAttestationProvider."""
    assert issubclass(IntelTdxProvider, BaseAttestationProvider)
    provider = IntelTdxProvider(simulate=True)
    assert provider.provider_name == "intel-tdx"
    assert provider.device_index == 0


def test_intel_tdx_constants_and_types() -> None:
    """Verifies binary protocol constants, ioctls, and ctypes structures."""
    assert TDX_REPORTDATA_LEN == 64
    assert TDREPORT_STRUCT_LEN == 1024
    assert TDX_CMD_GET_REPORT0 == 0xC4405401
    assert TDX_CMD_GET_QUOTE == 0xC0105402
    assert VSOCK_QGS_PORT == 4050
    assert VSOCK_HOST_CID == 2
    assert "/dev/tdx_guest" in TDX_GUEST_DEV_NODES
    assert "/dev/tdx-guest" in TDX_GUEST_DEV_NODES

    # Structure checks
    assert TdxReportReq is tdx_report_req
    assert TdxQuoteReq is tdx_quote_req
    assert ctypes.sizeof(tdx_report_req) == 1088
    assert ctypes.sizeof(tdx_quote_req) == 16

    req = tdx_report_req()
    assert hasattr(req, "reportdata")
    assert hasattr(req, "tdreport")
    assert len(req.reportdata) == 64
    assert len(req.tdreport) == 1024

    qreq = tdx_quote_req()
    assert hasattr(qreq, "buf")
    assert hasattr(qreq, "len")


# ==============================================================================
# 2. FACTORY RESOLUTION & ALIASES
# ==============================================================================

def test_intel_tdx_factory_resolution() -> None:
    """Tests get_attestation_provider factory resolution for canonical name and aliases."""
    # Canonical name
    p1 = get_attestation_provider("intel-tdx", simulate=True)
    assert isinstance(p1, IntelTdxProvider)
    assert p1.provider_name == "intel-tdx"
    assert p1.is_available() is True

    # Aliases
    p2 = get_attestation_provider("tdx", simulate=True)
    assert isinstance(p2, IntelTdxProvider)

    p3 = get_attestation_provider("intel", simulate=True)
    assert isinstance(p3, IntelTdxProvider)

    # Case-insensitivity & whitespace trimming
    p4 = get_attestation_provider("  INTEL-TDX  ", simulate=True)
    assert isinstance(p4, IntelTdxProvider)

    p5 = get_attestation_provider("TDX", simulate=True)
    assert isinstance(p5, IntelTdxProvider)


def test_intel_tdx_factory_fallback_behavior() -> None:
    """Verifies fallback to simulated provider when hardware is missing."""
    with patch.object(IntelTdxProvider, "is_available", return_value=False):
        # Fail-closed when fallback_to_simulated=False
        with pytest.raises(HardwareAttestationUnavailableError):
            get_attestation_provider("intel-tdx", fallback_to_simulated=False)

        # Graceful fallback when fallback_to_simulated=True
        p_fallback = get_attestation_provider("intel-tdx", fallback_to_simulated=True)
        assert isinstance(p_fallback, IntelTdxProvider)
        assert getattr(p_fallback, "simulate", False) is True


# ==============================================================================
# 3. AVAILABILITY PROBING & NON-HARDWARE SAFETY
# ==============================================================================

def test_intel_tdx_availability_probing() -> None:
    """Verifies hardware probing across device nodes, cpuinfo, and VSOCK transport."""
    # Simulation mode always available
    sim_provider = IntelTdxProvider(simulate=True)
    assert sim_provider.is_available() is True

    # Non-hardware safety: without mocks or hardware, returns False safely
    real_provider = IntelTdxProvider(simulate=False)
    avail = real_provider.is_available()
    assert isinstance(avail, bool)

    # Mock device node presence (/dev/tdx_guest)
    with patch("os.path.exists", side_effect=lambda p: p == "/dev/tdx_guest"):
        with patch("os.access", return_value=True):
            p_dev = IntelTdxProvider(simulate=False)
            assert p_dev.is_available() is True

    # Mock device node presence (/dev/tdx-guest)
    with patch("os.path.exists", side_effect=lambda p: p == "/dev/tdx-guest"):
        with patch("os.access", return_value=True):
            p_dev2 = IntelTdxProvider(simulate=False)
            assert p_dev2.is_available() is True

    # Mock /proc/cpuinfo with tdx_guest flag and device node
    cpuinfo_mock = "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic tdx_guest\n"
    with patch("os.path.exists", side_effect=lambda p: p in ("/proc/cpuinfo", "/dev/tdx_guest")):
        with patch("builtins.open", mock_open(read_data=cpuinfo_mock)):
            p_cpu = IntelTdxProvider(simulate=False)
            assert p_cpu.is_available() is True

    # Mock VSOCK socket connection
    with patch("src.attestation.tdx_attest._probe_tdx_guest_dev", return_value=None):
        with patch("src.attestation.tdx_attest._probe_cpuinfo_tdx", return_value=False):
            with patch("src.attestation.tdx_attest._probe_vsock_qgs", return_value=True):
                p_vsock = IntelTdxProvider(simulate=False)
                assert p_vsock.is_available() is True


def test_intel_tdx_hardware_unavailable_raises_when_not_simulated(sample_nonce, sample_pubkey) -> None:
    """Verifies HardwareAttestationUnavailableError when device is absent and simulate=False."""
    with patch.object(IntelTdxProvider, "is_available", return_value=False):
        provider = IntelTdxProvider(simulate=False)
        assert provider.is_available() is False
        with pytest.raises(HardwareAttestationUnavailableError, match="Intel TDX hardware guest environment is unavailable"):
            provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)


# ==============================================================================
# 4. FAIL-CLOSED PARAMETER VALIDATION
# ==============================================================================

def test_intel_tdx_parameter_validation(sample_nonce, sample_pubkey) -> None:
    """Verifies parameter validation checks on challenger nonce, public key, and device_index."""
    provider = IntelTdxProvider(simulate=True)

    # Invalid nonces (short, long, wrong type)
    with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
        provider.generate_evidence(nonce=b"too_short", public_key=sample_pubkey)

    with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
        provider.generate_evidence(nonce=secrets.token_bytes(33), public_key=sample_pubkey)

    with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
        provider.generate_evidence(nonce="string_not_bytes", public_key=sample_pubkey)  # type: ignore[arg-type]

    # Invalid public keys (short, long, wrong type)
    with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
        provider.generate_evidence(nonce=sample_nonce, public_key=b"bad_len")

    with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
        provider.generate_evidence(nonce=sample_nonce, public_key=secrets.token_bytes(64))

    with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
        provider.generate_evidence(nonce=sample_nonce, public_key=12345)  # type: ignore[arg-type]

    # Invalid device_index
    with pytest.raises(ValueError, match="device_index must be non-negative"):
        IntelTdxProvider(device_index=-1)


# ==============================================================================
# 5. EVIDENCE GENERATION & REPORTDATA BINDING
# ==============================================================================

def test_intel_tdx_evidence_generation_valid(sample_nonce, sample_pubkey) -> None:
    """Verifies valid authentic DCAP Quote and AttestationEvidence container generation."""
    provider = IntelTdxProvider(simulate=True)
    evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    assert isinstance(evidence, AttestationEvidence)
    assert evidence.provider == "intel-tdx"
    assert evidence.evidence_blob is not None
    assert len(evidence.evidence_blob) >= 636
    assert len(evidence.certificate_chain) == 3

    # Claims inspection
    claims = evidence.token_claims
    assert claims["provider"] == "intel-tdx"
    assert claims["debug"] is False
    assert claims["nonce"] == sample_nonce.hex()
    assert claims["ephemeral_pubkey_hash"] == hashlib.sha256(sample_pubkey).hexdigest()

    # Measurement registers
    measurements = claims["measurements"]
    assert "mrtd" in measurements
    assert "rtmr0" in measurements
    assert "rtmr1" in measurements
    assert "rtmr2" in measurements
    assert "rtmr3" in measurements
    assert len(measurements["mrtd"]) == 96  # 48 bytes in hex


def test_intel_tdx_reportdata_cryptographic_binding(sample_nonce, sample_pubkey) -> None:
    """Verifies 64-byte REPORTDATA wire embedding in Quote Body."""
    provider = IntelTdxProvider(simulate=True)
    evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    blob = evidence.evidence_blob
    assert blob is not None
    version = struct.unpack_from("<H", blob, 0)[0]
    body_offset = 48 if version == 4 else 54

    # REPORTDATA is at offset body_offset + 520 (length 64 bytes)
    reportdata = blob[body_offset + 520 : body_offset + 584]
    assert len(reportdata) == 64

    # Check dual-field binding: reportdata[:32] == nonce, reportdata[32:64] == SHA256(pubkey)
    pubkey_hash = hashlib.sha256(sample_pubkey).digest()
    assert reportdata[:32] == sample_nonce
    assert reportdata[32:64] == pubkey_hash
    assert evidence.user_data == reportdata


def test_intel_tdx_evidence_serialization_roundtrip(sample_nonce, sample_pubkey) -> None:
    """Verifies evidence serialization to bytes and deserialization roundtrip."""
    provider = IntelTdxProvider(simulate=True)
    ev1 = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    raw_bytes = ev1.to_bytes()
    ev2 = AttestationEvidence.from_bytes(raw_bytes)

    assert ev2.provider == ev1.provider
    assert ev2.evidence_blob == ev1.evidence_blob
    assert ev2.user_data == ev1.user_data
    assert ev2.certificate_chain == ev1.certificate_chain
    assert ev2.token_claims == ev1.token_claims
    assert ev2.token_signature == ev1.token_signature
    assert ev2.golden_rim == ev1.golden_rim


# ==============================================================================
# 6. DIRECT VERIFICATION WITH INTEL TDX VERIFIER (SDK BRIDGING)
# ==============================================================================

def test_direct_verification_with_intel_tdx_verifier(sample_nonce, sample_pubkey) -> None:
    """Direct component-level verification of generated evidence using IntelTdxVerifier."""
    provider = IntelTdxProvider(simulate=True)
    evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    leaf_pem = evidence.certificate_chain[0]
    leaf_cert = x509.load_pem_x509_certificate(leaf_pem.encode("utf-8"))

    tdx_verifier = IntelTdxVerifier()
    # 1. Verify dual ECDSA signatures (AK over Header+Body, PCK over QE3 report)
    tdx_verifier.verify_token_signature(evidence, leaf_cert)

    # 2. Extract and appraise claims
    claims = tdx_verifier.extract_claims(evidence)
    assert claims["provider"] == "intel-tdx"
    assert claims["debug"] is False
    assert claims["nonce"] == sample_nonce.hex()
    assert claims["ephemeral_pubkey_hash"] == hashlib.sha256(sample_pubkey).hexdigest()
    assert claims["measurements"]["mrtd"] == provider.get_measurements()["mrtd"]


def test_direct_verification_with_attestation_verifier_pipeline(intel_pki, sample_nonce, sample_pubkey) -> None:
    """Full end-to-end verification through AttestationVerifier policy pipeline."""
    provider = IntelTdxProvider(simulate=True)
    evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    policy = VerificationPolicy()
    policy.add_trusted_root_pem(intel_pki.root_cert_pem)
    policy.add_golden_rim("intel-tdx", provider.get_measurements())

    verifier = AttestationVerifier(policy=policy)
    result: VerificationResult = verifier.verify(
        evidence=evidence,
        expected_nonce=sample_nonce,
        expected_client_pubkey=sample_pubkey,
    )

    assert result.verified is True
    assert result.provider == "intel-tdx"
    assert result.chain_length == 3
    assert result.measurements_verified == 5
    assert result.claims["debug"] is False


def test_direct_verification_tampered_signature_rejected(sample_nonce, sample_pubkey) -> None:
    """Verifies that single-bit signature tampering is detected fail-closed."""
    provider = IntelTdxProvider(simulate=True)
    evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    # Mutate byte in quote_signature
    tampered_blob = bytearray(evidence.evidence_blob)
    tampered_blob[640] ^= 0x01
    evidence.evidence_blob = bytes(tampered_blob)
    evidence.token_signature = bytes(tampered_blob[636:700])

    leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))
    tdx_verifier = IntelTdxVerifier()

    with pytest.raises(AttestationSignatureError):
        tdx_verifier.verify_token_signature(evidence, leaf_cert)


def test_direct_verification_tampered_nonce_rejected(intel_pki, sample_nonce, sample_pubkey) -> None:
    """Verifies that challenger nonce mismatch is rejected by the verification pipeline."""
    provider = IntelTdxProvider(simulate=True)
    evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    policy = VerificationPolicy()
    policy.add_trusted_root_pem(intel_pki.root_cert_pem)
    verifier = AttestationVerifier(policy=policy)

    wrong_nonce = secrets.token_bytes(32)
    with pytest.raises(AttestationNonceMismatchError):
        verifier.verify(
            evidence=evidence,
            expected_nonce=wrong_nonce,
            expected_client_pubkey=sample_pubkey,
        )


def test_direct_verification_tampered_pubkey_rejected(intel_pki, sample_nonce, sample_pubkey) -> None:
    """Verifies that client public key substitution is rejected fail-closed."""
    provider = IntelTdxProvider(simulate=True)
    evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    policy = VerificationPolicy()
    policy.add_trusted_root_pem(intel_pki.root_cert_pem)
    verifier = AttestationVerifier(policy=policy)

    wrong_pubkey = secrets.token_bytes(32)
    with pytest.raises(AttestationBindingError):
        verifier.verify(
            evidence=evidence,
            expected_nonce=sample_nonce,
            expected_client_pubkey=wrong_pubkey,
        )


# ==============================================================================
# 7. SERVICER GRPC CONFIDENTIAL HANDSHAKE
# ==============================================================================

def test_servicer_intel_tdx_handshake(sample_nonce, sample_pubkey) -> None:
    """Verifies MatrixComputeServicer executes confidential handshake for required_tee_type='intel-tdx'."""
    provider = IntelTdxProvider(simulate=True)
    servicer = MatrixComputeServicer(attestation_provider=provider)

    req = pb2.TileTaskRequest(
        task_id="task-tdx-001",
        matrix_a_rows=64,
        matrix_a_cols=64,
        matrix_b_cols=64,
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=sample_pubkey,
        attestation_nonce=sample_nonce,
        required_tee_type="intel-tdx",
    )

    ack = servicer.RequestTileTask(req)
    assert ack.is_ready is True
    assert ack.tee_type == "intel-tdx"
    assert len(ack.worker_ephemeral_pubkey) == 32
    assert len(ack.attestation_evidence) > 0

    # Deserialize evidence and check claims
    ev = AttestationEvidence.from_bytes(ack.attestation_evidence)
    assert ev.provider == "intel-tdx"
    assert ev.token_claims["nonce"] == sample_nonce.hex()
    worker_pubkey_hash = hashlib.sha256(ack.worker_ephemeral_pubkey).hexdigest()
    assert ev.token_claims["ephemeral_pubkey_hash"] == worker_pubkey_hash


# ==============================================================================
# 8. NODECONFIG SETTINGS & CLI RESOLUTION
# ==============================================================================

def test_node_config_attestation_provider_intel_tdx() -> None:
    """Verifies NodeConfig supports intel-tdx via kwargs, CLI args, and environment variables."""
    # 1. Constructor
    cfg1 = NodeConfig(attestation_provider="intel-tdx")
    assert cfg1.attestation_provider == "intel-tdx"

    # 2. CLI Argument
    cfg2 = NodeConfig.load(args=["--attestation-provider", "intel-tdx"])
    assert cfg2.attestation_provider == "intel-tdx"

    # 3. Environment Variable (MATRIX_ATTESTATION_PROVIDER)
    cfg3 = NodeConfig.from_env(env={"MATRIX_ATTESTATION_PROVIDER": "intel-tdx"})
    assert cfg3.attestation_provider == "intel-tdx"

    # 4. Environment Variable (ATTESTATION_PROVIDER)
    cfg4 = NodeConfig.from_env(env={"ATTESTATION_PROVIDER": "intel-tdx"})
    assert cfg4.attestation_provider == "intel-tdx"

    # 5. Case-insensitivity normalization
    cfg5 = NodeConfig(attestation_provider="INTEL-TDX")
    assert cfg5.attestation_provider == "intel-tdx"


# ==============================================================================
# 9. AF_VSOCK QGS PROTOCOL & ERROR HANDLING
# ==============================================================================

def test_intel_tdx_vsock_protocol_mock(sample_nonce, sample_pubkey) -> None:
    """Verifies AF_VSOCK protocol framing for QGS communication."""
    from src.attestation.tdx_attest import fetch_quote_vsock

    dummy_report = b"\x42" * 1024
    dummy_quote = b"\x04\x00" + b"\x00" * 700  # >= 636 bytes

    # Success case: response header len = 4 + len(dummy_quote), error_code = 0
    resp_payload = struct.pack(">I", 0) + dummy_quote
    resp_frame = struct.pack(">I", len(resp_payload)) + resp_payload

    mock_sock = MagicMock()
    mock_sock.__enter__.return_value = mock_sock
    mock_sock.recv.side_effect = [
        resp_frame[:4],         # length header
        resp_frame[4:],         # payload
        b"",
    ]

    with patch("socket.socket", return_value=mock_sock):
        with patch("socket.AF_VSOCK", 40, create=True):
            quote = fetch_quote_vsock(dummy_report)
            assert quote == dummy_quote
            assert mock_sock.connect.called
            assert mock_sock.connect.call_args[0][0] == (VSOCK_HOST_CID, VSOCK_QGS_PORT)

    # QGS error case: error_code != 0
    err_payload = struct.pack(">I", 0xE0000001) + b"\x00" * 636
    err_frame = struct.pack(">I", len(err_payload)) + err_payload

    mock_err_sock = MagicMock()
    mock_err_sock.__enter__.return_value = mock_err_sock
    mock_err_sock.recv.side_effect = [
        err_frame[:4],
        err_frame[4:],
        b"",
    ]

    with patch("socket.socket", return_value=mock_err_sock):
        with patch("socket.AF_VSOCK", 40, create=True):
            with pytest.raises(AttestationGenerationError, match="Host QGS daemon returned error status code"):
                fetch_quote_vsock(dummy_report)


# ==============================================================================
# 10. MEMORY ZEROIZATION
# ==============================================================================

def test_intel_tdx_zeroization() -> None:
    """Verifies zeroize_buffer zeroes out ctypes structures, bytearrays, and memoryviews."""
    # 1. tdx_report_req zeroization
    req = tdx_report_req()
    for i in range(64):
        req.reportdata[i] = 0xFF
    for i in range(1024):
        req.tdreport[i] = 0xAA
    assert req.reportdata[0] == 0xFF
    assert req.tdreport[0] == 0xAA

    zeroize_buffer(req)
    assert req.reportdata[0] == 0
    assert req.reportdata[63] == 0
    assert req.tdreport[0] == 0
    assert req.tdreport[1023] == 0

    # 2. tdx_quote_req zeroization
    qreq = tdx_quote_req()
    qreq.buf = 0x12345678
    qreq.len = 8192
    zeroize_buffer(qreq)
    assert qreq.buf == 0
    assert qreq.len == 0

    # 3. bytearray zeroization
    secret = bytearray(b"super_secret_temporary_key_bytes")
    zeroize_buffer(secret)
    assert secret == b"\x00" * len(secret)

    # 4. memoryview zeroization
    raw = bytearray(b"temporary_view_buffer")
    mv = memoryview(raw)
    zeroize_buffer(mv)
    assert raw == b"\x00" * len(raw)

    # 5. None is handled safely
    zeroize_buffer(None)
