"""
Simulated Attestation Provider and Synthetic PKI Generator.
Produces authentic X.509 certificate chains, nonce/pubkey token claims,
and Golden RIM reference digests for hermetic 100% offline environments.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from matrix_contracts.schemas.attestation import AttestationEvidence
from src.attestation.base import (
    BaseAttestationProvider,
    AttestationValidationError,
    register_attestation_provider,
)

logger = logging.getLogger("matrix_compute_node.attestation.simulated")

DEFAULT_SIMULATED_MEASUREMENTS: Dict[str, str] = {
    "vbios_digest": "4a7d1ed414474e4033ac29ccb8653d9b12c6a0c5c3d2ec3b6a07e2c9497e5d2f",
    "driver_digest": "782356c891048efd429a1b327b87561f95c028a213e8b4e74982a0349ef821cd",
    "runtime_digest": "d348b64e5264b4c798e3b2e95a98762f0123456789abcdef0123456789abcdef",
}


class SyntheticPKICertificateBundle:
    """Holds a generated 3-tier X.509 certificate chain and private keys."""

    def __init__(
        self,
        root_cert_pem: str,
        intermediate_cert_pem: str,
        leaf_cert_pem: str,
        leaf_private_key: ec.EllipticCurvePrivateKey,
    ) -> None:
        self.root_cert_pem = root_cert_pem
        self.intermediate_cert_pem = intermediate_cert_pem
        self.leaf_cert_pem = leaf_cert_pem
        self.leaf_private_key = leaf_private_key

    @property
    def certificate_chain(self) -> List[str]:
        return [self.leaf_cert_pem, self.intermediate_cert_pem, self.root_cert_pem]


def generate_synthetic_pki(
    expired: bool = False,
    untrusted: bool = False,
) -> SyntheticPKICertificateBundle:
    """
    Generates a 3-tier X.509 certificate chain using ECDSA SECP256R1:
    Root CA -> Intermediate CA -> Leaf Attestation Certificate.
    """
    now = datetime.now(timezone.utc)
    if expired:
        valid_from = now - timedelta(days=60)
        valid_to = now - timedelta(days=1)
    else:
        valid_from = now - timedelta(days=1)
        valid_to = now + timedelta(days=365)

    # 1. Root CA
    root_key = ec.generate_private_key(ec.SECP256R1())
    root_org = "Untrusted Rogue CA" if untrusted else "DistributedMatrixMultiply Root Authority"
    root_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, root_org),
        x509.NameAttribute(NameOID.COMMON_NAME, "Simulated Hardware Root CA"),
    ])
    root_cert = (
        x509.CertificateBuilder()
        .subject_name(root_name)
        .issuer_name(root_name)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(valid_from)
        .not_valid_after(valid_to)
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()),
            critical=False,
        )
        .sign(root_key, hashes.SHA256())
    )

    # 2. Intermediate CA
    inter_key = ec.generate_private_key(ec.SECP256R1())
    inter_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "DistributedMatrixMultiply TEE"),
        x509.NameAttribute(NameOID.COMMON_NAME, "Simulated NVIDIA Hopper Attestation Intermediate CA"),
    ])
    inter_cert = (
        x509.CertificateBuilder()
        .subject_name(inter_name)
        .issuer_name(root_name)
        .public_key(inter_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(valid_from)
        .not_valid_after(valid_to)
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(inter_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()),
            critical=False,
        )
        .sign(root_key, hashes.SHA256())
    )

    # 3. Leaf Attestation Certificate
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "DistributedMatrixMultiply Enclave"),
        x509.NameAttribute(NameOID.COMMON_NAME, "Simulated H100 Attestation Leaf"),
        x509.NameAttribute(NameOID.SERIAL_NUMBER, "SIM-H100-ENCLAVE-001"),
    ])
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(inter_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(valid_from)
        .not_valid_after(valid_to)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=True,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(inter_key.public_key()),
            critical=False,
        )
        .sign(inter_key, hashes.SHA256())
    )

    root_pem = root_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    inter_pem = inter_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    leaf_pem = leaf_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")

    return SyntheticPKICertificateBundle(
        root_cert_pem=root_pem,
        intermediate_cert_pem=inter_pem,
        leaf_cert_pem=leaf_pem,
        leaf_private_key=leaf_key,
    )


@register_attestation_provider("simulated-cca")
@register_attestation_provider("simulated")
class SimulatedAttestationProvider(BaseAttestationProvider):
    """
    Simulated Hardware Attestation Provider for offline development and hermetic testing.
    Generates authentic synthetic X.509 cert chains, nonce/pubkey bindings, and Golden RIM.
    Supports fault-injection parameters for testing adversarial scenarios.
    """

    def __init__(
        self,
        golden_rim: Optional[Dict[str, str]] = None,
        measurements: Optional[Dict[str, str]] = None,
        tamper_signature: bool = False,
        tamper_measurements: bool = False,
        tamper_nonce: bool = False,
        expired_cert: bool = False,
        untrusted_ca: bool = False,
        fixed_timestamp: Optional[int] = None,
        pki_bundle: Optional[SyntheticPKICertificateBundle] = None,
        **kwargs: Any,
    ) -> None:
        self._golden_rim = (golden_rim or DEFAULT_SIMULATED_MEASUREMENTS).copy()
        self._measurements = (measurements or self._golden_rim).copy()
        self.tamper_signature = tamper_signature
        self.tamper_measurements = tamper_measurements
        self.tamper_nonce = tamper_nonce
        self.expired_cert = expired_cert
        self.untrusted_ca = untrusted_ca
        self.fixed_timestamp = fixed_timestamp

        # Generate or use injected PKI bundle
        self.pki_bundle = pki_bundle or generate_synthetic_pki(
            expired=self.expired_cert,
            untrusted=self.untrusted_ca,
        )

    @property
    def provider_name(self) -> str:
        return "simulated-cca"

    def is_available(self) -> bool:
        """Simulated provider is always operational offline."""
        return True

    def get_measurements(self) -> Dict[str, str]:
        return self._measurements.copy()

    def generate_evidence(self, nonce: bytes, public_key: bytes) -> AttestationEvidence:
        """
        Generates authentic synthetic attestation evidence.
        Binds challenger nonce and ephemeral public key.
        """
        if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != 32:
            raise AttestationValidationError(
                f"nonce must be 32 bytes, got {len(nonce) if isinstance(nonce, (bytes, bytearray)) else type(nonce)}"
            )

        if not isinstance(public_key, (bytes, bytearray)) or len(public_key) != 32:
            raise AttestationValidationError(
                f"public_key must be 32 bytes, got {len(public_key) if isinstance(public_key, (bytes, bytearray)) else type(public_key)}"
            )

        nonce_bytes = bytes(nonce)
        pubkey_bytes = bytes(public_key)

        # Apply fault injection if requested
        if self.tamper_nonce:
            nonce_bytes = bytes([(b ^ 0xFF) for b in nonce_bytes])

        measurements = self._measurements.copy()
        if self.tamper_measurements:
            measurements["runtime_digest"] = "bad0" * 16

        pubkey_hash = hashlib.sha256(pubkey_bytes).hexdigest()
        user_data = hashlib.sha256(nonce_bytes + pubkey_bytes).digest()

        ts = self.fixed_timestamp if self.fixed_timestamp is not None else int(time.time())

        token_claims = {
            "provider": self.provider_name,
            "nonce": nonce_bytes.hex(),
            "ephemeral_pubkey_hash": pubkey_hash,
            "public_key": pubkey_bytes.hex(),
            "timestamp": ts,
            "security_version": 1,
            "measurements": measurements,
            "gpu_model": "NVIDIA H100 80GB HBM3 (Simulated CC)",
            "cc_mode": True,
        }

        # Canonical claims serialization for signing
        claims_bytes = json.dumps(token_claims, sort_keys=True, separators=(",", ":")).encode("utf-8")

        # Sign using Leaf certificate private key
        sig = self.pki_bundle.leaf_private_key.sign(claims_bytes, ec.ECDSA(hashes.SHA256()))
        if self.tamper_signature:
            sig = bytes([(b ^ 0xAA) for b in sig])

        evidence_blob = b"SIMULATED_NVIDIA_CCA_QUOTE_V1:" + user_data

        return AttestationEvidence(
            provider=self.provider_name,
            certificate_chain=self.pki_bundle.certificate_chain,
            token_claims=token_claims,
            token_signature=sig,
            golden_rim=self._golden_rim.copy(),
            evidence_blob=evidence_blob,
            user_data=user_data,
        )
