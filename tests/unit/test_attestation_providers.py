"""
Unit test suite for matrix-compute-node hardware attestation subsystem.
Covers BaseAttestationProvider, NvidiaCcaProvider, SimulatedAttestationProvider,
registry resolution, parameter validation, fault injection, and servicer integration.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from unittest.mock import MagicMock, mock_open, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from matrix_contracts.schemas.attestation import AttestationEvidence
from matrix_contracts.v1 import pb2
from src.attestation import (
    AMD_SEV_SNP_GOLDEN_RIM,
    AmdSevSnpProvider,
    AttestationError,
    AttestationGenerationError,
    AttestationValidationError,
    BaseAttestationProvider,
    HardwareAttestationUnavailableError,
    INTEL_TDX_GOLDEN_RIM,
    IntelTdxProvider,
    NvidiaCcaProvider,
    SNP_GET_EXT_REPORT,
    SNP_GET_REPORT,
    SNP_NONCE_LEN,
    SNP_PUBKEY_HASH_LEN,
    SNP_REPORT_LEN,
    SNP_SIGNED_PORTION_LEN,
    SNP_USER_DATA_LEN,
    SimulatedAttestationProvider,
    SnpExtReportReq,
    SnpGuestRequest,
    SnpReportReq,
    SyntheticPKICertificateBundle,
    generate_synthetic_pki,
    get_attestation_provider,
    parse_snp_report_binary,
    register_attestation_provider,
    snp_ext_report_req,
    snp_guest_request,
    snp_report_req,
    zeroize_buffer,
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

    p_exp = SimulatedAttestationProvider(expired_cert=True)
    ev_exp = p_exp.generate_evidence(nonce=nonce, public_key=pubkey)
    cert = x509.load_pem_x509_certificate(ev_exp.certificate_chain[0].encode("utf-8"))
    nva = getattr(cert, "not_valid_after_utc", cert.not_valid_after)
    nvb = getattr(cert, "not_valid_before_utc", cert.not_valid_before)
    assert nva < nvb.replace(year=nvb.year + 2)

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


# =============================================================================
# AMD SEV-SNP HARDWARE ATTESTATION PROVIDER TESTS (Task 05 / R1)
# =============================================================================

def test_amd_sev_snp_provider_contract_and_structures() -> None:
    """Verifies AmdSevSnpProvider contracts, structure sizes, and ioctl constants."""
    import ctypes

    # 1. Structure byte size verifications per AMD SEV-SNP Linux ABI (<uapi/linux/sev-guest.h>)
    assert ctypes.sizeof(snp_report_req) == 96
    assert ctypes.sizeof(snp_ext_report_req) == 112
    assert ctypes.sizeof(snp_guest_request) == 32
    assert ctypes.sizeof(SnpReportReq) == 96
    assert ctypes.sizeof(SnpExtReportReq) == 112
    assert ctypes.sizeof(SnpGuestRequest) == 32

    # 2. ioctl numbers
    assert isinstance(SNP_GET_REPORT, int)
    assert isinstance(SNP_GET_EXT_REPORT, int)
    assert SNP_GET_REPORT != 0
    assert SNP_GET_EXT_REPORT != 0

    # 3. Provider instance properties
    provider = AmdSevSnpProvider(device_index=0)
    assert provider.provider_name == "amd-sev-snp"
    assert provider.device_index == 0
    assert provider.vmpl == 0

    measurements = provider.get_measurements()
    assert "measurement" in measurements
    assert "host_data" in measurements

    # 4. Custom RIM and invalid parameters
    custom_rim = {"measurement": "aa" * 48, "host_data": "bb" * 32}
    p_custom = AmdSevSnpProvider(device_index=1, golden_rim=custom_rim, vmpl=1)
    assert p_custom.device_index == 1
    assert p_custom.vmpl == 1
    assert p_custom.get_measurements() == custom_rim

    with pytest.raises(ValueError, match="device_index must be non-negative"):
        AmdSevSnpProvider(device_index=-1)


def test_amd_sev_snp_availability_probing() -> None:
    """Tests platform probing for /dev/sev-guest and CPUID / cpuinfo flags."""
    # 1. Non-SNP platform safely returns False without throwing
    p_hw = AmdSevSnpProvider(simulate=False)
    with patch("os.path.exists", return_value=False):
        assert p_hw.is_available() is False

    # 2. Simulated mode returns True
    p_sim = AmdSevSnpProvider(simulate=True)
    assert p_sim.is_available() is True

    # 3. Mock device node exists and is accessible
    with patch("src.attestation.sev_snp_attest._probe_sev_guest_dev", return_value="/dev/sev-guest"):
        with patch("os.path.exists", return_value=True):
            p_mock = AmdSevSnpProvider(simulate=False)
            assert p_mock.is_available() is True

    # 4. Mock /proc/cpuinfo with sev_snp flag
    cpuinfo_mock = "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext perfctr_core invpcid_single ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves clzero xsaveerptr rdpru wbnoinvd arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi sev sev_es sev_snp\n"
    with patch("src.attestation.sev_snp_attest._probe_sev_guest_dev", return_value=None):
        with patch("src.attestation.sev_snp_attest.os.path.exists", side_effect=lambda p: p in ("/proc/cpuinfo", "/dev/sev-guest")):
            with patch("builtins.open", mock_open(read_data=cpuinfo_mock)):
                p_cpu = AmdSevSnpProvider(simulate=False)
                assert p_cpu.is_available() is True


def test_amd_sev_snp_factory_resolution() -> None:
    """Tests get_attestation_provider factory resolution for AMD SEV-SNP."""
    # 1. Explicit amd-sev-snp with simulate=True
    p1 = get_attestation_provider("amd-sev-snp", simulate=True)
    assert isinstance(p1, AmdSevSnpProvider)
    assert p1.provider_name == "amd-sev-snp"
    assert p1.is_available() is True

    # 2. Aliases amd and sev-snp
    p2 = get_attestation_provider("amd", simulate=True)
    assert isinstance(p2, AmdSevSnpProvider)
    p3 = get_attestation_provider("sev-snp", simulate=True)
    assert isinstance(p3, AmdSevSnpProvider)

    # 3. Fallback behavior when hardware is absent
    with patch.object(AmdSevSnpProvider, "is_available", return_value=False):
        with pytest.raises(HardwareAttestationUnavailableError):
            get_attestation_provider("amd-sev-snp", fallback_to_simulated=False)

        p_fallback = get_attestation_provider("amd-sev-snp", fallback_to_simulated=True)
        assert isinstance(p_fallback, AmdSevSnpProvider)
        assert p_fallback.simulate is True


def test_amd_sev_snp_evidence_generation_valid() -> None:
    """Verifies that AmdSevSnpProvider generates valid 1184-byte report and AttestationEvidence."""
    provider = AmdSevSnpProvider(simulate=True)
    nonce = secrets.token_bytes(32)
    pubkey = secrets.token_bytes(32)

    evidence = provider.generate_evidence(nonce=nonce, public_key=pubkey)

    assert isinstance(evidence, AttestationEvidence)
    assert evidence.provider == "amd-sev-snp"
    assert len(evidence.evidence_blob) == SNP_REPORT_LEN
    assert len(evidence.certificate_chain) == 3

    # Binding assertions
    pubkey_hash = hashlib.sha256(pubkey).digest()
    expected_user_data = nonce + pubkey_hash
    assert evidence.user_data == expected_user_data
    assert evidence.token_claims["nonce"] == nonce.hex()
    assert evidence.token_claims["ephemeral_pubkey_hash"] == pubkey_hash.hex()
    assert evidence.token_claims["user_data"] == expected_user_data.hex()

    # Claims checks
    claims = evidence.token_claims
    assert claims["policy_debug"] is False
    assert (claims["policy"] >> 19) & 0x01 == 0
    assert claims["vmpl"] == 0
    assert "measurement" in claims["measurements"]
    assert "host_data" in claims["measurements"]
    assert len(claims["chip_id"]) == 128
    assert len(evidence.token_signature) == 144


def test_amd_sev_snp_evidence_serialization_roundtrip() -> None:
    """Verifies AttestationEvidence serialization to bytes and deserialization roundtrip."""
    provider = AmdSevSnpProvider(simulate=True)
    nonce = secrets.token_bytes(32)
    pubkey = secrets.token_bytes(32)

    ev1 = provider.generate_evidence(nonce=nonce, public_key=pubkey)
    raw_bytes = ev1.to_bytes()

    ev2 = AttestationEvidence.from_bytes(raw_bytes)
    assert ev2.provider == "amd-sev-snp"
    assert ev2.evidence_blob == ev1.evidence_blob
    assert ev2.user_data == ev1.user_data
    assert ev2.certificate_chain == ev1.certificate_chain
    assert ev2.token_claims == ev1.token_claims
    assert ev2.token_signature == ev1.token_signature
    assert ev2.golden_rim == ev1.golden_rim


def test_amd_sev_snp_cryptographic_signature_verification() -> None:
    """Verifies ECDSA P-384 signature over bytes 0x000..0x29F with little-endian (R, S) conversion."""
    provider = AmdSevSnpProvider(simulate=True)
    nonce = secrets.token_bytes(32)
    pubkey = secrets.token_bytes(32)

    evidence = provider.generate_evidence(nonce=nonce, public_key=pubkey)
    blob = evidence.evidence_blob
    assert blob is not None and len(blob) == 1184

    vcek_pem = evidence.certificate_chain[0]
    vcek_cert = x509.load_pem_x509_certificate(vcek_pem.encode("utf-8"))
    vcek_pubkey = vcek_cert.public_key()
    assert isinstance(vcek_pubkey, ec.EllipticCurvePublicKey)

    raw_r = blob[0x2A0:0x2E8]
    raw_s = blob[0x2E8:0x330]

    r_int = int.from_bytes(raw_r[:48], byteorder="little")
    s_int = int.from_bytes(raw_s[:48], byteorder="little")
    der_sig = encode_dss_signature(r_int, s_int)

    signed_payload = blob[:672]
    vcek_pubkey.verify(der_sig, signed_payload, ec.ECDSA(hashes.SHA384()))

    tampered_payload = bytearray(signed_payload)
    tampered_payload[0x050] ^= 0x01
    with pytest.raises(Exception):
        vcek_pubkey.verify(der_sig, bytes(tampered_payload), ec.ECDSA(hashes.SHA384()))


def test_amd_sev_snp_provider_parameter_validation() -> None:
    """Verifies that invalid nonces and public keys are rejected fail-closed."""
    provider = AmdSevSnpProvider(simulate=True)
    valid_32 = secrets.token_bytes(32)

    # Invalid nonces
    with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
        provider.generate_evidence(nonce=b"short", public_key=valid_32)

    with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
        provider.generate_evidence(nonce=secrets.token_bytes(33), public_key=valid_32)

    with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
        provider.generate_evidence(nonce="not_bytes", public_key=valid_32)  # type: ignore[arg-type]

    # Invalid public keys
    with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
        provider.generate_evidence(nonce=valid_32, public_key=b"bad_len")

    with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
        provider.generate_evidence(nonce=valid_32, public_key=secrets.token_bytes(64))


def test_amd_sev_snp_unavailable_raises_when_not_simulated() -> None:
    """Verifies HardwareAttestationUnavailableError when device is missing and simulate=False."""
    with patch("src.attestation.sev_snp_attest._probe_sev_guest_dev", return_value=None):
        with patch("src.attestation.sev_snp_attest._probe_cpuinfo_sev_snp", return_value=False):
            provider = AmdSevSnpProvider(simulate=False)
            assert provider.is_available() is False
            with pytest.raises(HardwareAttestationUnavailableError, match="AMD SEV-SNP hardware guest device"):
                provider.generate_evidence(nonce=secrets.token_bytes(32), public_key=secrets.token_bytes(32))


def test_amd_sev_snp_zeroization() -> None:
    """Verifies zeroize_buffer zeroes out ctypes structures, arrays, bytearrays, and views."""
    # 1. snp_report_req zeroization
    req = snp_report_req()
    for i in range(64):
        req.user_data[i] = 0xAA
    req.vmpl = 2
    assert req.user_data[0] == 0xAA
    assert req.vmpl == 2
    zeroize_buffer(req)
    assert req.user_data[0] == 0
    assert req.user_data[63] == 0
    assert req.vmpl == 0

    # 2. snp_ext_report_req zeroization
    ext_req = snp_ext_report_req()
    ext_req.data.vmpl = 1
    ext_req.certs_address = 0x12345678
    ext_req.certs_len = 4096
    zeroize_buffer(ext_req)
    assert ext_req.data.vmpl == 0
    assert ext_req.certs_address == 0
    assert ext_req.certs_len == 0

    # 3. bytearray zeroization
    buf = bytearray(b"sensitive_ephemeral_key_secret_material")
    zeroize_buffer(buf)
    assert buf == b"\x00" * len(buf)

    # 4. memoryview zeroization
    raw = bytearray(b"temporary_report_buffer")
    mv = memoryview(raw)
    zeroize_buffer(mv)
    assert raw == b"\x00" * len(raw)

    # 5. None is handled safely
    zeroize_buffer(None)


def test_amd_sev_snp_ioctl_error_handling() -> None:
    """Verifies that ioctl execution errors raise AttestationGenerationError."""
    provider = AmdSevSnpProvider(simulate=False, device_path="/dev/sev-guest")

    with patch("src.attestation.sev_snp_attest.os.path.exists", return_value=True):
        with patch("src.attestation.sev_snp_attest.os.access", return_value=True):
            with patch("src.attestation.sev_snp_attest.fcntl") as mock_fcntl:
                mock_fcntl.ioctl.side_effect = OSError("Driver communication failure")
                mock_file = MagicMock()
                mock_file.fileno.return_value = 42

                with patch("builtins.open", return_value=mock_file):
                    with pytest.raises(AttestationGenerationError, match="Linux ioctl execution failed"):
                        provider._fetch_snp_report_and_certs("/dev/sev-guest", b"\x00" * 64, [])


def test_amd_sev_snp_servicer_handshake() -> None:
    """Verifies MatrixComputeServicer handles AMD SEV-SNP confidential compute handshake."""
    provider = AmdSevSnpProvider(simulate=True)
    servicer = MatrixComputeServicer(attestation_provider=provider)

    nonce = secrets.token_bytes(32)
    client_pubkey = secrets.token_bytes(32)

    req = pb2.TileTaskRequest(
        task_id="task-snp-001",
        matrix_a_rows=64,
        matrix_a_cols=64,
        matrix_b_cols=64,
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=client_pubkey,
        attestation_nonce=nonce,
        required_tee_type="amd-sev-snp",
    )

    ack = servicer.RequestTileTask(req)
    assert ack.is_ready is True
    assert ack.tee_type == "amd-sev-snp"
    assert len(ack.worker_ephemeral_pubkey) == 32
    assert len(ack.attestation_evidence) > 0

    ev = AttestationEvidence.from_bytes(ack.attestation_evidence)
    assert ev.provider == "amd-sev-snp"
    assert ev.token_claims["nonce"] == nonce.hex()
    worker_pubkey_hash = hashlib.sha256(ack.worker_ephemeral_pubkey).hexdigest()
    assert ev.token_claims["ephemeral_pubkey_hash"] == worker_pubkey_hash
    assert len(ev.evidence_blob) == 1184


# =============================================================================
# INTEL TDX HARDWARE ATTESTATION PROVIDER TESTS (Milestone 3)
# =============================================================================

def test_intel_tdx_factory_resolution() -> None:
    """Tests get_attestation_provider factory resolution for Intel TDX."""
    # Canonical
    p1 = get_attestation_provider("intel-tdx", simulate=True)
    assert isinstance(p1, IntelTdxProvider)
    assert p1.provider_name == "intel-tdx"
    assert p1.is_available() is True

    # Aliases
    p2 = get_attestation_provider("tdx", simulate=True)
    assert isinstance(p2, IntelTdxProvider)
    p3 = get_attestation_provider("intel", simulate=True)
    assert isinstance(p3, IntelTdxProvider)

    # Fallback behavior when hardware is absent
    with patch.object(IntelTdxProvider, "is_available", return_value=False):
        with pytest.raises(HardwareAttestationUnavailableError):
            get_attestation_provider("intel-tdx", fallback_to_simulated=False)

        p_fallback = get_attestation_provider("intel-tdx", fallback_to_simulated=True)
        assert isinstance(p_fallback, IntelTdxProvider)
        assert getattr(p_fallback, "simulate", False) is True


def test_intel_tdx_evidence_generation_valid() -> None:
    """Verifies that IntelTdxProvider generates authentic DCAP Quote and AttestationEvidence."""
    provider = IntelTdxProvider(simulate=True)
    nonce = secrets.token_bytes(32)
    pubkey = secrets.token_bytes(32)

    evidence = provider.generate_evidence(nonce=nonce, public_key=pubkey)

    assert isinstance(evidence, AttestationEvidence)
    assert evidence.provider == "intel-tdx"
    assert len(evidence.evidence_blob) >= 636
    assert len(evidence.certificate_chain) == 3

    # Binding assertions
    pubkey_hash = hashlib.sha256(pubkey).digest()
    assert evidence.token_claims["nonce"] == nonce.hex()
    assert evidence.token_claims["ephemeral_pubkey_hash"] == pubkey_hash.hex()
    assert evidence.token_claims["debug"] is False


def test_intel_tdx_servicer_handshake() -> None:
    """Verifies MatrixComputeServicer handles Intel TDX confidential compute handshake."""
    provider = IntelTdxProvider(simulate=True)
    servicer = MatrixComputeServicer(attestation_provider=provider)

    nonce = secrets.token_bytes(32)
    client_pubkey = secrets.token_bytes(32)

    req = pb2.TileTaskRequest(
        task_id="task-tdx-provider-001",
        matrix_a_rows=64,
        matrix_a_cols=64,
        matrix_b_cols=64,
        dtype="fp32",
        require_confidential_compute=True,
        client_ephemeral_pubkey=client_pubkey,
        attestation_nonce=nonce,
        required_tee_type="intel-tdx",
    )

    ack = servicer.RequestTileTask(req)
    assert ack.is_ready is True
    assert ack.tee_type == "intel-tdx"
    assert len(ack.worker_ephemeral_pubkey) == 32
    assert len(ack.attestation_evidence) > 0

    ev = AttestationEvidence.from_bytes(ack.attestation_evidence)
    assert ev.provider == "intel-tdx"
    assert ev.token_claims["nonce"] == nonce.hex()
    worker_pubkey_hash = hashlib.sha256(ack.worker_ephemeral_pubkey).hexdigest()
    assert ev.token_claims["ephemeral_pubkey_hash"] == worker_pubkey_hash


