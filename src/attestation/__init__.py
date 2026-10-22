"""
Hardware Attestation and Confidential Computing (TEE) subsystem for matrix-compute-node.
"""

from src.attestation.base import (
    AttestationError,
    AttestationGenerationError,
    AttestationValidationError,
    BaseAttestationProvider,
    HardwareAttestationUnavailableError,
    get_attestation_provider,
    register_attestation_provider,
)
from src.attestation.nvml_attest import (
    HardwareAttestationEngine,
    NvidiaCcaProvider,
    NVIDIA_HOPPER_GOLDEN_RIM,
    estimate_peak_fp32_tflops,
    verify_attestation_proof,
)
from src.attestation.sev_snp_attest import (
    AMD_SEV_SNP_GOLDEN_RIM,
    AmdSevSnpProvider,
    SNP_GET_EXT_REPORT,
    SNP_GET_REPORT,
    SNP_NONCE_LEN,
    SNP_PUBKEY_HASH_LEN,
    SNP_REPORT_LEN,
    SNP_SIGNED_PORTION_LEN,
    SNP_USER_DATA_LEN,
    SnpExtReportReq,
    SnpGuestRequest,
    SnpReportReq,
    parse_snp_report_binary,
    snp_ext_report_req,
    snp_guest_request,
    snp_report_req,
    zeroize_buffer,
)
from src.attestation.simulated import (
    DEFAULT_SIMULATED_MEASUREMENTS,
    SimulatedAttestationProvider,
    SyntheticPKICertificateBundle,
    generate_synthetic_pki,
)

__all__ = [
    "AMD_SEV_SNP_GOLDEN_RIM",
    "AmdSevSnpProvider",
    "AttestationError",
    "AttestationGenerationError",
    "AttestationValidationError",
    "BaseAttestationProvider",
    "DEFAULT_SIMULATED_MEASUREMENTS",
    "HardwareAttestationEngine",
    "HardwareAttestationUnavailableError",
    "NvidiaCcaProvider",
    "NVIDIA_HOPPER_GOLDEN_RIM",
    "SNP_GET_EXT_REPORT",
    "SNP_GET_REPORT",
    "SNP_NONCE_LEN",
    "SNP_PUBKEY_HASH_LEN",
    "SNP_REPORT_LEN",
    "SNP_SIGNED_PORTION_LEN",
    "SNP_USER_DATA_LEN",
    "SimulatedAttestationProvider",
    "SnpExtReportReq",
    "SnpGuestRequest",
    "SnpReportReq",
    "SyntheticPKICertificateBundle",
    "estimate_peak_fp32_tflops",
    "generate_synthetic_pki",
    "get_attestation_provider",
    "parse_snp_report_binary",
    "register_attestation_provider",
    "snp_ext_report_req",
    "snp_guest_request",
    "snp_report_req",
    "verify_attestation_proof",
    "zeroize_buffer",
]

