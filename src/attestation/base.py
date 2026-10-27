"""
Abstract base class and exception hierarchy for pluggable hardware attestation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Callable, Dict, Optional, Type

from matrix_contracts.schemas.attestation import AttestationEvidence

logger = logging.getLogger("matrix_compute_node.attestation")


class AttestationError(Exception):
    """Base exception for all attestation provider errors."""


class HardwareAttestationUnavailableError(AttestationError):
    """Raised when the requested hardware attestation provider is unavailable."""


class AttestationGenerationError(AttestationError):
    """Raised when evidence generation or signing fails."""


class AttestationValidationError(AttestationError):
    """Raised when inputs (nonce, public key) are invalid."""


class BaseAttestationProvider(ABC):
    """
    Abstract Base Class for hardware and simulated TEE attestation providers.
    Defines the standard interface for generating cryptographically signed
    hardware attestation evidence bound to challenger nonces and ephemeral public keys.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Canonical provider identifier (e.g. 'nvidia-cca', 'simulated-cca', 'amd-sev-snp', 'intel-tdx').
        Matches `tee_type` in matrix_contracts.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Probes the platform and driver to determine if this provider is operational.
        Returns True if available, False otherwise. Must not raise unhandled exceptions.
        """
        ...

    @abstractmethod
    def generate_evidence(
        self,
        nonce: bytes,
        public_key: bytes,
    ) -> AttestationEvidence:
        """
        Generates cryptographically signed attestation evidence proving hardware authenticity,
        measurement integrity (PCRs/Golden RIM), and cryptographic binding to the challenger nonce
        and ephemeral public key.

        Args:
            nonce: 32-byte challenger freshness nonce.
            public_key: 32-byte ephemeral public key (e.g. X25519) to bind to the attestation.

        Returns:
            AttestationEvidence: Validated Pydantic container with certificate chain,
            token claims, signature, Golden RIM, and optional evidence blob.

        Raises:
            AttestationValidationError: If nonce or public_key are invalid (not 32 bytes).
            HardwareAttestationUnavailableError: If underlying hardware is unavailable.
            AttestationGenerationError: If evidence generation fails.
        """
        ...

    def get_measurements(self) -> Dict[str, str]:
        """Returns a dictionary of current hardware/enclave measurements or PCR digests."""
        return {}


# Provider Registry
_PROVIDER_REGISTRY: Dict[str, Type[BaseAttestationProvider]] = {}


def register_attestation_provider(
    name: str,
) -> Callable[[Type[BaseAttestationProvider]], Type[BaseAttestationProvider]]:
    """Decorator to register an attestation provider class."""

    def decorator(cls: Type[BaseAttestationProvider]) -> Type[BaseAttestationProvider]:
        _PROVIDER_REGISTRY[name.lower().strip()] = cls
        return cls

    return decorator


def get_attestation_provider(
    provider_type: Optional[str] = "auto",
    device_index: int = 0,
    fallback_to_simulated: bool = True,
    **kwargs: Any,
) -> BaseAttestationProvider:
    """
    Factory function resolving and instantiating the requested attestation provider.

    Args:
        provider_type: 'auto', 'nvidia-cca', 'simulated-cca', or registered provider name.
        device_index: Target GPU index for hardware providers.
        fallback_to_simulated: If True, falls back to SimulatedAttestationProvider when
                              hardware provider is unavailable.
        **kwargs: Provider-specific configuration options.

    Returns:
        BaseAttestationProvider instance.
    """
    norm_type = (provider_type or "auto").strip().lower()

    if norm_type == "auto":
        from src.attestation.nvml_attest import NvidiaCcaProvider

        cca_provider = NvidiaCcaProvider(device_index=device_index, **kwargs)
        if cca_provider.is_available():
            logger.info(
                "Using hardware attestation provider: nvidia-cca (device %d)", device_index
            )
            return cca_provider

        try:
            from src.attestation.sev_snp_attest import AmdSevSnpProvider

            snp_provider = AmdSevSnpProvider(device_index=device_index, **kwargs)
            if snp_provider.is_available():
                logger.info("Using hardware attestation provider: amd-sev-snp")
                return snp_provider
        except Exception as exc:
            logger.debug("Auto probe for amd-sev-snp failed: %s", exc)

        try:
            from src.attestation.tdx_attest import IntelTdxProvider

            tdx_provider = IntelTdxProvider(device_index=device_index, **kwargs)
            if tdx_provider.is_available():
                logger.info("Using hardware attestation provider: intel-tdx")
                return tdx_provider
        except Exception as exc:
            logger.debug("Auto probe for intel-tdx failed: %s", exc)

        if not fallback_to_simulated:
            raise HardwareAttestationUnavailableError(
                "No hardware TEE provider is operational (hardware attestation unavailable and fallback_to_simulated=False)."
            )

        logger.info("Hardware TEE unavailable; falling back to simulated-cca provider.")
        from src.attestation.simulated import SimulatedAttestationProvider

        return SimulatedAttestationProvider(**kwargs)

    if norm_type in ("nvidia-cca", "nvidia"):
        from src.attestation.nvml_attest import NvidiaCcaProvider

        cca_provider = NvidiaCcaProvider(device_index=device_index, **kwargs)
        if cca_provider.is_available():
            logger.info(
                "Using hardware attestation provider: nvidia-cca (device %d)", device_index
            )
            return cca_provider

        if not fallback_to_simulated:
            raise HardwareAttestationUnavailableError(
                f"NVIDIA CCA requested on device {device_index}, but hardware or CC mode is unavailable."
            )

        logger.info("Hardware CCA unavailable; falling back to simulated-cca provider.")
        from src.attestation.simulated import SimulatedAttestationProvider

        return SimulatedAttestationProvider(**kwargs)

    if norm_type in ("amd-sev-snp", "amd", "sev-snp"):
        from src.attestation.sev_snp_attest import AmdSevSnpProvider

        snp_provider = AmdSevSnpProvider(device_index=device_index, **kwargs)
        if snp_provider.is_available():
            logger.info("Using hardware attestation provider: amd-sev-snp")
            return snp_provider

        if not fallback_to_simulated:
            raise HardwareAttestationUnavailableError(
                "AMD SEV-SNP requested, but /dev/sev-guest hardware is unavailable."
            )

        logger.info("Hardware AMD SEV-SNP unavailable; falling back to simulated SEV-SNP provider.")
        return AmdSevSnpProvider(device_index=device_index, simulate=True, **kwargs)

    if norm_type in ("intel-tdx", "tdx", "intel"):
        from src.attestation.tdx_attest import IntelTdxProvider

        tdx_provider = IntelTdxProvider(device_index=device_index, **kwargs)
        if tdx_provider.is_available():
            logger.info("Using hardware attestation provider: intel-tdx")
            return tdx_provider

        if not fallback_to_simulated:
            raise HardwareAttestationUnavailableError(
                "Intel TDX requested, but /dev/tdx_guest hardware is unavailable."
            )

        logger.info("Hardware Intel TDX unavailable; falling back to simulated TDX provider.")
        return IntelTdxProvider(device_index=device_index, simulate=True, **kwargs)

    if norm_type in ("simulated-cca", "simulated", "mock"):
        from src.attestation.simulated import SimulatedAttestationProvider

        return SimulatedAttestationProvider(**kwargs)

    if norm_type in _PROVIDER_REGISTRY:
        cls = _PROVIDER_REGISTRY[norm_type]
        return cls(device_index=device_index, **kwargs)

    raise ValueError(f"Unknown attestation provider type: '{provider_type}'")

