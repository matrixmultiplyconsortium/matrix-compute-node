"""Compute engine subsystem."""

from src.engine.compute_core import ComputeEngine
from src.engine.scheduler import PreemptiveComputeScheduler, PriorityTier
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

__all__ = [
    "ComputeEngine",
    "PreemptiveComputeScheduler",
    "PriorityTier",
    "EnclaveComputeCore",
    "EnclaveSession",
    "EnclaveError",
    "EnclaveSessionNotFoundError",
    "EnclaveKeyExchangeError",
    "EnclaveSecurityPolicyError",
    "EnclaveDecryptionError",
    "EnclaveFramingError",
    "EnclaveExecutionError",
    "SessionKeys",
    "aes_gcm_decrypt",
    "aes_gcm_encrypt",
    "canonical_chunk_aad",
    "canonical_result_aad",
    "compute_chunk_aad",
    "compute_result_aad",
    "derive_session_keys",
]
