"""
Direct Cross-Subsystem Adversarial Verification Test Harness for Intel TDX.
Milestone 3 Challenge 2 (teamwork_preview_challenger).

Directly validates the interface contract between:
- Node Evidence Generation: `IntelTdxProvider(simulate=True)` (Quote v4 & v5)
- SDK Evidence Verification: `IntelTdxVerifier()` & `AttestationVerifier()`

Thoroughly challenges:
1. Cross-subsystem piping across Quote format v4 and modular v5.
2. Serialization roundtrip and deserialization into SDK verifiers.
3. Adversarial signature mutations (AK and PCK dual ECDSA signature tampering, out-of-bounds curve scalars).
4. Container token_signature inconsistency attacks.
5. AK cross-binding corruption in QE3 report body.
6. Challenger nonce mutations and replay attempts across v4 and v5.
7. Ephemeral public key substitutions (MITM key replacement) across v4 and v5.
8. Direct quote body field mutations (tampered REPORTDATA nonce and pubkey hash).
9. Malicious certificate chains (untrusted root, empty chain, broken intermediate link, tampered DER, revoked cert, leaf with CA flag).
10. Hardware security invariants (DEBUG enclave quote rejection, Golden RIM mismatch).
11. Wire buffer mutilations (truncated headers, truncated bodies, invalid versions, wrong TEE type).
12. Node provider parameter boundary validation and memory buffer zeroization.
"""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
import struct
from typing import Any, Dict, List

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from matrix_contracts.schemas.attestation import AttestationEvidence
from matrix_compute.crypto import (
    AttestationVerifier,
    IntelTdxVerifier,
    VerificationPolicy,
    VerificationResult,
    generate_synthetic_intel_pki,
    generate_synthetic_tdx_quote,
    get_default_intel_pki,
    INTEL_TDX_GOLDEN_RIM,
    INTEL_TDX_TEE_TYPE,
)
from matrix_compute.crypto.tdx_verifier import (
    SECP256R1_ORDER,
    SECP384R1_ORDER,
)
from matrix_compute.exceptions import (
    AttestationBindingError,
    AttestationCertificateError,
    AttestationChainInvalidError,
    AttestationError,
    AttestationEvidenceMalformedError,
    AttestationMeasurementMismatchError,
    AttestationNonceMismatchError,
    AttestationRevokedError,
    AttestationSignatureError,
    AttestationUntrustedRootError,
)
from src.attestation import (
    AttestationValidationError,
    IntelTdxProvider,
    zeroize_buffer,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def intel_pki():
    """Session-scoped cached synthetic Intel TDX PKI hierarchy."""
    return get_default_intel_pki()


@pytest.fixture
def default_policy(intel_pki) -> VerificationPolicy:
    """Standard verification policy trusting Intel Root CA and Golden RIM."""
    policy = VerificationPolicy()
    policy.add_trusted_root_pem(intel_pki.root_cert_pem)
    policy.add_golden_rim("intel-tdx", INTEL_TDX_GOLDEN_RIM)
    return policy


@pytest.fixture
def verifier(default_policy) -> AttestationVerifier:
    """Configured SDK AttestationVerifier instance."""
    return AttestationVerifier(policy=default_policy)


@pytest.fixture
def tdx_verifier() -> IntelTdxVerifier:
    """Component-level IntelTdxVerifier instance."""
    return IntelTdxVerifier()


@pytest.fixture
def sample_nonce() -> bytes:
    """32-byte cryptographically secure random nonce."""
    return secrets.token_bytes(32)


@pytest.fixture
def sample_pubkey() -> bytes:
    """32-byte ephemeral public key."""
    return secrets.token_bytes(32)


# ==============================================================================
# 1. DIRECT CROSS-SUBSYSTEM PIPING (QUOTE V4 & V5)
# ==============================================================================

class TestDirectCrossSubsystemPiping:
    """Validates uncorrupted evidence generation and verification across v4 and v5."""

    @pytest.mark.parametrize("quote_version", [4, 5])
    def test_cross_subsystem_evidence_pipeline_honest(
        self, verifier, tdx_verifier, sample_nonce, sample_pubkey, quote_version
    ):
        """
        Pipes evidence from Node IntelTdxProvider(simulate=True) into both
        IntelTdxVerifier and AttestationVerifier across Quote versions 4 and 5.
        """
        provider = IntelTdxProvider(simulate=True, quote_version=quote_version)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        assert isinstance(evidence, AttestationEvidence)
        assert evidence.provider == "intel-tdx"
        assert evidence.evidence_blob is not None
        assert len(evidence.evidence_blob) >= 636
        assert len(evidence.certificate_chain) == 3

        # 1. Component-level verification via IntelTdxVerifier
        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))
        tdx_verifier.verify_token_signature(evidence, leaf_cert)

        claims = tdx_verifier.extract_claims(evidence)
        assert claims["provider"] == "intel-tdx"
        assert claims["quote_version"] == quote_version
        assert claims["debug"] is False
        assert claims["nonce"] == sample_nonce.hex()
        assert claims["ephemeral_pubkey_hash"] == hashlib.sha256(sample_pubkey).hexdigest()

        # 2. Pipeline-level verification via AttestationVerifier
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
        assert result.claims["quote_version"] == quote_version

    def test_cross_subsystem_serialized_json_piping(self, verifier, sample_nonce, sample_pubkey):
        """Pipes serialized JSON evidence across the subsystem boundary."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        json_str = evidence.model_dump_json()
        result = verifier.verify(
            evidence=json_str,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_pubkey,
        )
        assert result.verified is True
        assert result.provider == "intel-tdx"

        # Also verify raw UTF-8 bytes deserialization
        raw_bytes = json_str.encode("utf-8")
        result2 = verifier.verify(
            evidence=raw_bytes,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_pubkey,
        )
        assert result2.verified is True

    @pytest.mark.parametrize(
        "nonce_val,pubkey_val",
        [
            (b"\x00" * 32, b"\xff" * 32),
            (b"\xff" * 32, b"\x00" * 32),
            (bytes(range(32)), bytes(range(31, -1, -1))),
            (secrets.token_bytes(32), secrets.token_bytes(32)),
        ],
    )
    def test_cross_subsystem_diverse_parameter_fixtures(
        self, verifier, nonce_val, pubkey_val
    ):
        """Verifies boundary byte patterns (zeros, ones, sequential) verify accurately."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=nonce_val, public_key=pubkey_val)

        result = verifier.verify(
            evidence=evidence,
            expected_nonce=nonce_val,
            expected_client_pubkey=pubkey_val,
        )
        assert result.verified is True
        assert result.claims["nonce"] == nonce_val.hex()
        assert result.claims["ephemeral_pubkey_hash"] == hashlib.sha256(pubkey_val).hexdigest()


# ==============================================================================
# 2. ADVERSARIAL QUOTE SIGNATURE MUTATIONS (AK & PCK)
# ==============================================================================

class TestAdversarialSignatureMutation:
    """Challenges signature verification with bit flips, corruptions, and forged scalars."""

    @pytest.mark.parametrize("quote_version", [4, 5])
    def test_tampered_ak_signature_fails_closed(
        self, verifier, tdx_verifier, sample_nonce, sample_pubkey, quote_version
    ):
        """Single-bit tampering in AK signature region must raise AttestationSignatureError."""
        provider = IntelTdxProvider(simulate=True, quote_version=quote_version)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        blob = bytearray(evidence.evidence_blob)
        sig_offset = 636 if quote_version == 4 else 642

        # Corrupt single bit in AK signature (r component)
        blob[sig_offset + 5] ^= 0x01
        evidence.evidence_blob = bytes(blob)
        evidence.token_signature = bytes(blob[sig_offset : sig_offset + 64])

        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))

        with pytest.raises(AttestationSignatureError):
            tdx_verifier.verify_token_signature(evidence, leaf_cert)

        with pytest.raises(AttestationSignatureError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    @pytest.mark.parametrize("quote_version", [4, 5])
    def test_tampered_pck_signature_fails_closed(
        self, verifier, tdx_verifier, sample_nonce, sample_pubkey, quote_version
    ):
        """Mutating PCK signature over QE3 report must raise AttestationSignatureError."""
        provider = IntelTdxProvider(simulate=True, quote_version=quote_version)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        blob = bytearray(evidence.evidence_blob)
        pck_sig_offset = (636 + 128 + 384) if quote_version == 4 else (642 + 128 + 384)

        # Corrupt byte in PCK ECDSA signature
        blob[pck_sig_offset + 10] ^= 0x80
        evidence.evidence_blob = bytes(blob)

        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))

        with pytest.raises(AttestationSignatureError):
            tdx_verifier.verify_token_signature(evidence, leaf_cert)

        with pytest.raises(AttestationSignatureError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    def test_mismatched_container_token_signature(self, tdx_verifier, sample_nonce, sample_pubkey):
        """Container token_signature field differing from embedded signature must be rejected."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        # Mutate container signature only
        tampered_sig = bytearray(evidence.token_signature)
        tampered_sig[0] ^= 0xFF
        evidence.token_signature = bytes(tampered_sig)

        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))
        with pytest.raises(AttestationSignatureError, match="does not match the quote_signature"):
            tdx_verifier.verify_token_signature(evidence, leaf_cert)

    @pytest.mark.parametrize("scalar_val", [0, SECP256R1_ORDER, SECP256R1_ORDER + 1])
    def test_out_of_bounds_ecdsa_scalars(
        self, tdx_verifier, sample_nonce, sample_pubkey, scalar_val
    ):
        """Forged ECDSA signature scalars outside (0, curve_order) must be rejected."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        blob = bytearray(evidence.evidence_blob)
        sig_offset = 636
        # Write out-of-bounds scalar for r
        struct.pack_into(">Q", blob, sig_offset + 24, scalar_val & 0xFFFFFFFFFFFFFFFF)
        if scalar_val == 0:
            blob[sig_offset : sig_offset + 32] = b"\x00" * 32
        elif scalar_val >= SECP256R1_ORDER:
            blob[sig_offset : sig_offset + 32] = scalar_val.to_bytes(32, "big")

        evidence.evidence_blob = bytes(blob)
        evidence.token_signature = bytes(blob[sig_offset : sig_offset + 64])

        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))
        with pytest.raises(AttestationSignatureError):
            tdx_verifier.verify_token_signature(evidence, leaf_cert)


# ==============================================================================
# 3. ADVERSARIAL AK-QE CROSS-BINDING
# ==============================================================================

class TestAdversarialQeAkCrossBinding:
    """Challenges the QE3 report data cryptographic hash binding to the Attestation Key."""

    def test_tampered_qe_ak_cross_binding_fails_closed(
        self, verifier, tdx_verifier, sample_nonce, sample_pubkey
    ):
        """
        Tampering with QE3 report_data[:32] (which must bind SHA256(AK || auth_data))
        must fail closed with AttestationBindingError.
        """
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        blob = bytearray(evidence.evidence_blob)
        # QE report offset = 636 + 128 = 764. Report data in QE report is at 764 + 320 = 1084
        qe_report_data_offset = 1084
        blob[qe_report_data_offset + 2] ^= 0x01
        evidence.evidence_blob = bytes(blob)

        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))

        with pytest.raises(AttestationBindingError, match="Quote Enclave report does not bind Attestation Public Key"):
            tdx_verifier.verify_token_signature(evidence, leaf_cert)

        with pytest.raises(AttestationBindingError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )


# ==============================================================================
# 4. ADVERSARIAL NONCE & PUBLIC KEY BINDING MUTATIONS
# ==============================================================================

class TestAdversarialNonceAndPubkeyBinding:
    """Challenges freshness nonce and ephemeral public key binding defenses."""

    @pytest.mark.parametrize("quote_version", [4, 5])
    def test_tampered_challenger_nonce_fails_closed(
        self, verifier, sample_nonce, sample_pubkey, quote_version
    ):
        """Challenger nonce mismatch must raise AttestationNonceMismatchError."""
        provider = IntelTdxProvider(simulate=True, quote_version=quote_version)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        wrong_nonce = secrets.token_bytes(32)
        with pytest.raises(AttestationNonceMismatchError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=wrong_nonce,
                expected_client_pubkey=sample_pubkey,
            )

        # Single-bit flip in expected nonce
        single_bit_tampered = bytearray(sample_nonce)
        single_bit_tampered[0] ^= 0x01
        with pytest.raises(AttestationNonceMismatchError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=bytes(single_bit_tampered),
                expected_client_pubkey=sample_pubkey,
            )

    @pytest.mark.parametrize("quote_version", [4, 5])
    def test_substituted_client_pubkey_fails_closed(
        self, verifier, sample_nonce, sample_pubkey, quote_version
    ):
        """Client public key substitution (MITM attempt) must raise AttestationBindingError."""
        provider = IntelTdxProvider(simulate=True, quote_version=quote_version)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        substitute_pubkey = secrets.token_bytes(32)
        with pytest.raises(AttestationBindingError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=substitute_pubkey,
            )

        # Single-bit flip in public key
        single_bit_pubkey = bytearray(sample_pubkey)
        single_bit_pubkey[15] ^= 0x04
        with pytest.raises(AttestationBindingError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=bytes(single_bit_pubkey),
            )

    @pytest.mark.parametrize("quote_version", [4, 5])
    def test_quote_body_reportdata_tampering_fails_closed(
        self, verifier, tdx_verifier, sample_nonce, sample_pubkey, quote_version
    ):
        """
        Tampering directly with REPORTDATA inside Quote Body breaks the signed body,
        causing AK signature verification to fail immediately.
        """
        provider = IntelTdxProvider(simulate=True, quote_version=quote_version)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        blob = bytearray(evidence.evidence_blob)
        body_offset = 48 if quote_version == 4 else 54
        reportdata_offset = body_offset + 520

        # Corrupt byte in nonce half of REPORTDATA
        blob[reportdata_offset + 4] ^= 0x01
        evidence.evidence_blob = bytes(blob)

        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))

        # Fails closed at signature verification
        with pytest.raises(AttestationSignatureError):
            tdx_verifier.verify_token_signature(evidence, leaf_cert)

        with pytest.raises(AttestationSignatureError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )


# ==============================================================================
# 5. ADVERSARIAL CERTIFICATE CHAIN & PKI ATTACKS
# ==============================================================================

class TestAdversarialCertificateChainMutations:
    """Challenges X.509 PCK certificate chain validation and trust anchoring."""

    def test_untrusted_root_ca_rejected(self, sample_nonce, sample_pubkey):
        """Evidence signed by an untrusted Root CA must raise AttestationUntrustedRootError."""
        foreign_pki = generate_synthetic_intel_pki()
        provider = IntelTdxProvider(
            simulate=True,
            cert_chain=foreign_pki.certificate_chain,
            quote_version=4,
        )
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        # Policy trusts ONLY official default Intel PKI, not foreign_pki
        policy = VerificationPolicy()
        policy.add_trusted_root_pem(get_default_intel_pki().root_cert_pem)
        verifier = AttestationVerifier(policy=policy)

        with pytest.raises(AttestationUntrustedRootError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    def test_empty_certificate_chain_rejected(self, verifier, sample_nonce, sample_pubkey):
        """Empty certificate chain must raise AttestationEvidenceMalformedError."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        evidence.certificate_chain = []
        with pytest.raises(AttestationEvidenceMalformedError, match="Certificate chain is empty"):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    def test_broken_intermediate_chain_rejected(self, verifier, sample_nonce, sample_pubkey):
        """Broken issuer/subject linkage in chain must raise AttestationChainInvalidError."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        foreign_pki = generate_synthetic_intel_pki()
        # Substitute foreign intermediate CA into chain
        broken_chain = [
            evidence.certificate_chain[0],
            foreign_pki.intermediate_cert_pem,
            evidence.certificate_chain[2],
        ]
        evidence.certificate_chain = broken_chain

        with pytest.raises((AttestationChainInvalidError, AttestationSignatureError)):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    def test_tampered_pck_certificate_der_rejected(self, verifier, sample_nonce, sample_pubkey):
        """Corrupting PEM encoding of PCK leaf certificate must raise AttestationCertificateError."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        bad_pem = evidence.certificate_chain[0].replace("A", "B", 10)
        evidence.certificate_chain = [bad_pem, evidence.certificate_chain[1], evidence.certificate_chain[2]]

        with pytest.raises(AttestationCertificateError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    def test_revoked_pck_certificate_rejected(self, default_policy, sample_nonce, sample_pubkey):
        """Revoked PCK leaf certificate must raise AttestationRevokedError."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))
        # Add leaf serial number to revoked set
        default_policy.revoked_serial_numbers.add(leaf_cert.serial_number)
        verifier = AttestationVerifier(policy=default_policy)

        with pytest.raises(AttestationRevokedError, match="revoked"):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    def test_leaf_certificate_with_ca_flag_rejected(
        self, verifier, sample_nonce, sample_pubkey
    ):
        """Leaf certificate with BasicConstraints CA:TRUE must raise AttestationChainInvalidError."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        # Place intermediate CA cert as the leaf (has CA:TRUE)
        ca_as_leaf_chain = [
            evidence.certificate_chain[1],
            evidence.certificate_chain[1],
            evidence.certificate_chain[2],
        ]
        evidence.certificate_chain = ca_as_leaf_chain

        with pytest.raises(AttestationChainInvalidError, match="basic_constraints.ca=True"):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )


# ==============================================================================
# 6. HARDWARE SECURITY INVARIANTS & DEBUG MODE DEFENSE
# ==============================================================================

class TestHardwareSecurityInvariants:
    """Challenges enforcement of production non-debug invariants and Golden RIM."""

    @pytest.mark.parametrize("quote_version", [4, 5])
    def test_debug_enclave_quote_unconditionally_rejected(
        self, verifier, tdx_verifier, intel_pki, sample_nonce, sample_pubkey, quote_version
    ):
        """
        Quotes produced with DEBUG flag set (TDATTRIBUTES & 0x01 != 0) must be
        unconditionally rejected fail-closed with AttestationSignatureError.
        """
        debug_quote = generate_synthetic_tdx_quote(
            pki=intel_pki,
            quote_version=quote_version,
            nonce=sample_nonce,
            worker_ephemeral_pubkey=sample_pubkey,
            debug=True,
        )

        evidence = AttestationEvidence(
            provider="intel-tdx",
            evidence_blob=debug_quote,
            certificate_chain=intel_pki.certificate_chain,
            user_data=sample_nonce + hashlib.sha256(sample_pubkey).digest(),
        )

        with pytest.raises(AttestationSignatureError, match="DEBUG mode enabled"):
            tdx_verifier.extract_claims(evidence)

        with pytest.raises(AttestationSignatureError, match="DEBUG mode enabled"):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    def test_golden_rim_measurement_mismatch_rejected(
        self, default_policy, sample_nonce, sample_pubkey
    ):
        """Deviations in MRTD or RTMR registers must raise AttestationMeasurementMismatchError."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        tampered_rim = INTEL_TDX_GOLDEN_RIM.copy()
        tampered_rim["mrtd"] = "deadbeef" * 12  # Tampered measurement
        default_policy.add_golden_rim("intel-tdx", tampered_rim)
        verifier = AttestationVerifier(policy=default_policy)

        with pytest.raises(AttestationMeasurementMismatchError, match="Golden RIM measurement appraisal failed"):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )


# ==============================================================================
# 7. WIRE BUFFER MUTILATIONS (TRUNCATION, CORRUPT HEADERS)
# ==============================================================================

class TestWireBufferMutilations:
    """Challenges resilience against truncated buffers, illegal headers, and bad wire formats."""

    @pytest.mark.parametrize("trunc_len", [0, 10, 47, 48, 100, 300, 630])
    def test_truncated_quote_v4_fails_closed(
        self, verifier, tdx_verifier, sample_nonce, sample_pubkey, trunc_len
    ):
        """Truncated Quote v4 wire blobs must raise AttestationEvidenceMalformedError."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        evidence.evidence_blob = evidence.evidence_blob[:trunc_len]
        evidence.token_claims = None  # Ensure no fallback to pre-parsed claims
        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))

        with pytest.raises(AttestationEvidenceMalformedError):
            tdx_verifier.extract_claims(evidence)

        with pytest.raises(AttestationEvidenceMalformedError):
            tdx_verifier.verify_token_signature(evidence, leaf_cert)

        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    @pytest.mark.parametrize("trunc_len", [48, 50, 53, 54, 100, 300, 637])
    def test_truncated_quote_v5_fails_closed(
        self, verifier, tdx_verifier, sample_nonce, sample_pubkey, trunc_len
    ):
        """Truncated Quote v5 wire blobs must raise AttestationEvidenceMalformedError."""
        provider = IntelTdxProvider(simulate=True, quote_version=5)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        evidence.evidence_blob = evidence.evidence_blob[:trunc_len]
        evidence.token_claims = None  # Ensure no fallback to pre-parsed claims
        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))

        with pytest.raises(AttestationEvidenceMalformedError):
            tdx_verifier.extract_claims(evidence)

        with pytest.raises(AttestationEvidenceMalformedError):
            tdx_verifier.verify_token_signature(evidence, leaf_cert)

        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    @pytest.mark.parametrize("invalid_version", [0, 1, 2, 3, 6, 7, 255, 65535])
    def test_invalid_quote_version_header_rejected(
        self, verifier, tdx_verifier, sample_nonce, sample_pubkey, invalid_version
    ):
        """Unsupported quote versions in header must raise AttestationEvidenceMalformedError."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        blob = bytearray(evidence.evidence_blob)
        struct.pack_into("<H", blob, 0, invalid_version)
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationEvidenceMalformedError, match="Unsupported Quote version"):
            tdx_verifier.extract_claims(evidence)

        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )

    @pytest.mark.parametrize("invalid_tee", [0x00000000, 0x00000001, 0x00000080, 0x00000082, 0xFFFFFFFF])
    def test_invalid_tee_type_header_rejected(
        self, verifier, tdx_verifier, sample_nonce, sample_pubkey, invalid_tee
    ):
        """Non-TDX TEE types (e.g. 0x00 SGX) must raise AttestationEvidenceMalformedError."""
        provider = IntelTdxProvider(simulate=True, quote_version=4)
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        blob = bytearray(evidence.evidence_blob)
        struct.pack_into("<I", blob, 4, invalid_tee)
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationEvidenceMalformedError, match="Invalid TEE Type"):
            tdx_verifier.extract_claims(evidence)

        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_pubkey,
            )


# ==============================================================================
# 8. PROVIDER PARAMETER BOUNDARY VALIDATION & ZEROIZATION
# ==============================================================================

class TestProviderParameterValidationAndZeroization:
    """Challenges IntelTdxProvider input checks, device indexing, and zeroization."""

    @pytest.mark.parametrize("bad_nonce", [b"", b"123", b"a" * 31, b"a" * 33, b"a" * 64, None, 12345])
    def test_provider_rejects_malformed_nonces(self, sample_pubkey, bad_nonce):
        """Nonces not exactly 32 bytes must raise AttestationValidationError."""
        provider = IntelTdxProvider(simulate=True)
        with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
            provider.generate_evidence(nonce=bad_nonce, public_key=sample_pubkey)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_pubkey", [b"", b"123", b"b" * 31, b"b" * 33, b"b" * 64, None, [1, 2, 3]])
    def test_provider_rejects_malformed_pubkeys(self, sample_nonce, bad_pubkey):
        """Public keys not exactly 32 bytes must raise AttestationValidationError."""
        provider = IntelTdxProvider(simulate=True)
        with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
            provider.generate_evidence(nonce=sample_nonce, public_key=bad_pubkey)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_idx", [-1, -100])
    def test_provider_rejects_negative_device_index(self, bad_idx):
        """Negative device indices must raise ValueError."""
        with pytest.raises(ValueError, match="device_index must be non-negative"):
            IntelTdxProvider(device_index=bad_idx)

    def test_zeroize_buffer_memory_scrubbing(self):
        """zeroize_buffer must overwrite bytearrays and memoryviews cleanly."""
        buf = bytearray(b"sensitive_intel_tdx_ephemeral_key_material")
        assert any(b != 0 for b in buf)

        zeroize_buffer(buf)
        assert all(b == 0 for b in buf)
        assert len(buf) == 42

        # None or non-buffer inputs must be handled safely without error
        zeroize_buffer(None)
        zeroize_buffer("immutable_string")
