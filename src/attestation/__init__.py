"""
Hardware attestation subsystem for matrix-compute-node daemon.
Exposes HardwareAttestationEngine, estimate_peak_fp32_tflops, and verify_attestation_proof.
"""

from src.attestation.nvml_attest import (
    HardwareAttestationEngine,
    estimate_peak_fp32_tflops,
    verify_attestation_proof,
)

__all__ = [
    "HardwareAttestationEngine",
    "estimate_peak_fp32_tflops",
    "verify_attestation_proof",
]
