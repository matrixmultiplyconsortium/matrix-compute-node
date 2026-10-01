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
from src.attestation.simulated import (
    DEFAULT_SIMULATED_MEASUREMENTS,
    SimulatedAttestationProvider,
    SyntheticPKICertificateBundle,
    generate_synthetic_pki,
)

__all__ = [
    "AttestationError",
    "AttestationGenerationError",
    "AttestationValidationError",
    "BaseAttestationProvider",
    "DEFAULT_SIMULATED_MEASUREMENTS",
    "HardwareAttestationEngine",
    "HardwareAttestationUnavailableError",
    "NvidiaCcaProvider",
    "NVIDIA_HOPPER_GOLDEN_RIM",
    "SimulatedAttestationProvider",
    "SyntheticPKICertificateBundle",
    "estimate_peak_fp32_tflops",
    "generate_synthetic_pki",
    "get_attestation_provider",
    "register_attestation_provider",
    "verify_attestation_proof",
]
