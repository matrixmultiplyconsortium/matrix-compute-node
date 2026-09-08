"""
Adversarial Stress Test Suite for Hardware Attestation & HardwareSpecs Packaging.

Milestone 2 Challenger 2 (teamwork_preview_challenger_m2_2).
Target: matrix-compute-node/src/attestation/nvml_attest.py & matrix_contracts/schemas/discovery.py

Tests:
1. Cryptographic verification of attestation_proof under metric tampering:
   - Tampered GPU model in payload vs signed proof.
   - Tampered VRAM in payload vs signed proof.
   - Tampered FP32 TFLOPS in payload vs signed proof.
   - Simultaneous multi-field tampering.
2. Public key verification & key isolation:
   - Verification against wrong / peer public keys.
   - Cross-key rejection matrix across multiple signers.
   - Nostr 32-byte X-only pubkey compatibility.
   - Malformed, invalid, or off-curve public keys.
3. Malformed, corrupted, and boundary attestation proofs:
   - Empty proof string and whitespace strings.
   - Odd-length hex and non-hex characters.
   - Truncated and extended signature byte lengths.
   - Degenerate signatures (all-zero r=0, s=0; all-0xFF).
   - Single-bit flip mutations across signature bytes.
4. Pydantic HardwareSpecs validation against edge values:
   - Missing fields and invalid type casting.
   - Negative VRAM and negative TFLOPS boundary behavior.
   - NaN, Inf, and -Inf float handling.
   - Extreme boundary values (2^63 - 1, 10^12 TFLOPS, large strings).
   - JSON serialization and deserialization roundtrip fidelity.
5. Timestamp tolerance windows & replay resistance:
   - Exact timestamp verification.
   - Out-of-window timestamp rejection.
   - Documenting test fixture timestamp (1788537600) behavior.
6. Engine packaging safety guarantees:
   - detect_hardware() invariants (vram_mb >= 0, peak_fp32_tflops >= 0.5).
   - Graceful handling of None signer.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from matrix_contracts.schemas.discovery import HardwareSpecs
from src.attestation.nvml_attest import (
    HardwareAttestationEngine,
    estimate_peak_fp32_tflops,
    verify_attestation_proof,
)
from src.crypto.signer import WorkerSigner


# ==============================================================================
# TEST FIXTURES & HELPERS
# ==============================================================================

@pytest.fixture
def test_signer():
    """Provides a deterministic WorkerSigner instance."""
    privkey_hex = "1111111111111111111111111111111111111111111111111111111111111111"
    return WorkerSigner(privkey_hex=privkey_hex)


@pytest.fixture
def secondary_signer():
    """Provides a second distinct WorkerSigner instance."""
    privkey_hex = "2222222222222222222222222222222222222222222222222222222222222222"
    return WorkerSigner(privkey_hex=privkey_hex)


@pytest.fixture
def sample_specs(test_signer):
    """Produces a valid signed HardwareSpecs object."""
    engine = HardwareAttestationEngine(signer=test_signer, device_index=0)
    with patch.object(engine, "_query_nvml", return_value=None):
        with patch.object(engine, "_query_nvidia_smi", return_value=None):
            with patch("src.attestation.nvml_attest._get_cpu_model_name", return_value="AMD Ryzen 9 5950X"):
                with patch("src.attestation.nvml_attest._get_system_ram_mb", return_value=(32768, 16384)):
                    with patch("src.attestation.nvml_attest._get_cpu_core_count", return_value=16):
                        return engine.detect_hardware(timestamp=1788537600)


# ==============================================================================
# 1. METRIC TAMPERING TESTS
# ==============================================================================

class TestMetricTamperingRejection:
    """Adversarial challenge: Assert verify_attestation_proof rejects any tampered metrics."""

    def test_tampered_gpu_model_rejection(self, sample_specs, test_signer):
        """Rejects modifications to gpu_model while holding signature constant."""
        pubkey = test_signer.get_pubkey_hex()
        ts = 1788537600

        # Baseline check: authentic specs must verify
        assert verify_attestation_proof(sample_specs, pubkey, timestamp=ts) is True

        # Adversarial mutations of gpu_model
        model_mutations = [
            "NVIDIA GeForce RTX 4090",                    # Complete replacement
            sample_specs.gpu_model.lower(),               # Case mutation
            sample_specs.gpu_model.upper(),               # Case mutation
            f"{sample_specs.gpu_model} ",                 # Trailing space
            f" {sample_specs.gpu_model}",                 # Leading space
            sample_specs.gpu_model + "\x00",              # Null byte injection
            sample_specs.gpu_model + " (Fraudulent)",     # Suffix injection
            "",                                           # Empty model name
            "CPU: Unknown (NumPy Emulated)",              # Slight name change
        ]

        for mutated_model in model_mutations:
            tampered = sample_specs.model_copy(update={"gpu_model": mutated_model})
            assert (
                verify_attestation_proof(tampered, pubkey, timestamp=ts) is False
            ), f"Failed to reject tampered gpu_model: {mutated_model!r}"

    def test_tampered_vram_rejection(self, sample_specs, test_signer):
        """Rejects any modification to attested vram_mb."""
        pubkey = test_signer.get_pubkey_hex()
        ts = 1788537600

        vram_mutations = [
            sample_specs.vram_mb + 1,       # +1 MB
            sample_specs.vram_mb - 1,       # -1 MB
            sample_specs.vram_mb * 2,       # 2x memory
            sample_specs.vram_mb // 2,      # 0.5x memory
            0,                              # 0 MB
            -sample_specs.vram_mb,          # Negative mirror
            -1,                             # -1 MB
            999999999,                      # Absurdly large
        ]

        for mutated_vram in vram_mutations:
            tampered = sample_specs.model_copy(update={"vram_mb": mutated_vram})
            assert (
                verify_attestation_proof(tampered, pubkey, timestamp=ts) is False
            ), f"Failed to reject tampered vram_mb: {mutated_vram}"

    def test_tampered_fp32_tflops_rejection(self, sample_specs, test_signer):
        """Rejects fine-grained and coarse modifications to peak_fp32_tflops."""
        pubkey = test_signer.get_pubkey_hex()
        ts = 1788537600

        tflops_mutations = [
            sample_specs.peak_fp32_tflops + 0.01,   # Fine-grained delta
            sample_specs.peak_fp32_tflops - 0.01,   # Fine-grained delta
            sample_specs.peak_fp32_tflops + 1.0,    # Coarse delta
            sample_specs.peak_fp32_tflops * 10.0,   # 10x multiplier
            0.0,                                    # Zero rating
            -1.0,                                   # Negative rating
            82.6,                                   # Typical GPU rating
            67.0,                                   # H100 rating
        ]

        for mutated_tflops in tflops_mutations:
            tampered = sample_specs.model_copy(update={"peak_fp32_tflops": mutated_tflops})
            assert (
                verify_attestation_proof(tampered, pubkey, timestamp=ts) is False
            ), f"Failed to reject tampered peak_fp32_tflops: {mutated_tflops}"

    def test_simultaneous_multi_field_tampering(self, sample_specs, test_signer):
        """Rejects combinations where multiple fields are forged simultaneously."""
        pubkey = test_signer.get_pubkey_hex()
        ts = 1788537600

        multi_tamperings = [
            {"gpu_model": "NVIDIA H100 80GB", "vram_mb": 81920},
            {"vram_mb": 81920, "peak_fp32_tflops": 67.0},
            {"gpu_model": "NVIDIA H100 80GB", "peak_fp32_tflops": 67.0},
            {"gpu_model": "NVIDIA H100 80GB", "vram_mb": 81920, "peak_fp32_tflops": 67.0},
        ]

        for updates in multi_tamperings:
            tampered = sample_specs.model_copy(update=updates)
            assert (
                verify_attestation_proof(tampered, pubkey, timestamp=ts) is False
            ), f"Failed to reject multi-field tampering: {updates}"


# ==============================================================================
# 2. PUBLIC KEY & CROSS-KEY ISOLATION TESTS
# ==============================================================================

class TestPublicKeyIsolation:
    """Adversarial challenge: Verify strict cryptographic binding to the signing public key."""

    def test_wrong_public_key_rejection(self, sample_specs, secondary_signer):
        """Valid proof signed by Key A must be rejected when verified against Key B."""
        wrong_pubkey = secondary_signer.get_pubkey_hex()
        ts = 1788537600

        assert (
            verify_attestation_proof(sample_specs, wrong_pubkey, timestamp=ts) is False
        ), "Proof unexpectedly verified against a different public key!"

    def test_cross_key_matrix_rejection(self):
        """Cross-verification matrix across 4 signers must yield 0 false positives."""
        signers = [WorkerSigner() for _ in range(4)]
        specs_list = []

        # Generate hardware specs for each signer
        for s in signers:
            engine = HardwareAttestationEngine(signer=s, device_index=0)
            with patch.object(engine, "_query_nvml", return_value=None):
                with patch.object(engine, "_query_nvidia_smi", return_value=None):
                    with patch("src.attestation.nvml_attest._get_cpu_model_name", return_value="AMD Ryzen 9"):
                        with patch("src.attestation.nvml_attest._get_system_ram_mb", return_value=(16384, 8192)):
                            specs = engine.detect_hardware(timestamp=1788537600)
                            specs_list.append(specs)

        # Cross-verify all pairs (i, j)
        for i, specs in enumerate(specs_list):
            for j, signer in enumerate(signers):
                pubkey = signer.get_pubkey_hex()
                is_valid = verify_attestation_proof(specs, pubkey, timestamp=1788537600)
                if i == j:
                    assert is_valid is True, f"Valid key index {i} failed self-verification"
                else:
                    assert is_valid is False, f"Key index {j} falsely verified specs from key {i}"

    def test_nostr_xonly_pubkey_compatibility(self, sample_specs, test_signer):
        """Verifies that 32-byte Nostr X-only public keys are supported by verify_attestation."""
        nostr_pubkey = test_signer.get_nostr_pubkey()  # 64 hex chars (32 bytes)
        assert len(nostr_pubkey) == 64

        assert (
            verify_attestation_proof(sample_specs, nostr_pubkey, timestamp=1788537600) is True
        ), "Nostr 32-byte X-only pubkey failed to verify valid attestation proof"

    def test_tampered_and_malformed_public_keys(self, sample_specs):
        """Public keys with bit flips, bad lengths, and non-hex return False safely."""
        ts = 1788537600
        valid_pubkey = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"

        invalid_pubkeys = [
            "",                                             # Empty
            "   ",                                          # Whitespace
            "not_hex_at_all",                               # Non-hex
            valid_pubkey[:-2],                              # Truncated (32 bytes with 02 prefix = 33 chars?)
            valid_pubkey + "00",                            # Extended (34 bytes)
            "00" * 33,                                      # Invalid prefix 00
            "05" * 33,                                      # Invalid prefix 05
            "04" + valid_pubkey[2:],                        # 04 prefix but only 33 bytes (uncompressed needs 65)
            "02" + "ff" * 32,                               # Off-curve / coordinate exceeding field prime
        ]

        for pub in invalid_pubkeys:
            assert (
                verify_attestation_proof(sample_specs, pub, timestamp=ts) is False
            ), f"Expected verification to return False for invalid pubkey: {pub!r}"


# ==============================================================================
# 3. MALFORMED & BOUNDARY PROOF STRINGS
# ==============================================================================

class TestMalformedProofHandling:
    """Adversarial challenge: Verify graceful rejection of malformed or corrupted proofs."""

    def test_empty_and_whitespace_proofs(self, sample_specs, test_signer):
        """Rejects empty strings, whitespace, and None proofs without exceptions."""
        pubkey = test_signer.get_pubkey_hex()

        empty_proofs = ["", "   ", "\t\n", "None", "null"]
        for proof in empty_proofs:
            tampered = sample_specs.model_copy(update={"attestation_proof": proof})
            assert (
                verify_attestation_proof(tampered, pubkey, timestamp=1788537600) is False
            ), f"Failed to reject empty/whitespace proof: {proof!r}"

    def test_odd_length_and_non_hex_proofs(self, sample_specs, test_signer):
        """Rejects non-hex characters and odd-length hex strings."""
        pubkey = test_signer.get_pubkey_hex()

        malformed_proofs = [
            sample_specs.attestation_proof[:-1],            # 127 hex chars (odd length)
            sample_specs.attestation_proof + "a",           # 129 hex chars (odd length)
            "z" * 128,                                      # Non-hex
            "0x" + sample_specs.attestation_proof,          # Prefixed with 0x (130 chars)
            sample_specs.attestation_proof.replace("a", "g"), # Invalid hex char 'g'
        ]

        for proof in malformed_proofs:
            tampered = sample_specs.model_copy(update={"attestation_proof": proof})
            assert (
                verify_attestation_proof(tampered, pubkey, timestamp=1788537600) is False
            ), f"Failed to reject malformed proof: {proof[:20]}..."

    def test_truncated_and_extended_proof_bytes(self, sample_specs, test_signer):
        """Rejects proofs with invalid byte lengths (anything not 64 bytes compact or valid DER)."""
        pubkey = test_signer.get_pubkey_hex()

        raw_bytes = bytes.fromhex(sample_specs.attestation_proof)
        assert len(raw_bytes) == 64

        test_lengths = [0, 1, 16, 31, 32, 33, 63, 65, 70, 128]
        for length in test_lengths:
            if length == 64:
                continue
            corrupted_hex = os.urandom(length).hex()
            tampered = sample_specs.model_copy(update={"attestation_proof": corrupted_hex})
            assert (
                verify_attestation_proof(tampered, pubkey, timestamp=1788537600) is False
            ), f"Failed to reject proof of invalid byte length {length}"

    def test_degenerate_zero_and_ones_signatures(self, sample_specs, test_signer):
        """Rejects degenerate signatures: (r=0, s=0) and (r=2^256-1, s=2^256-1)."""
        pubkey = test_signer.get_pubkey_hex()

        degenerate_proofs = [
            "00" * 64,                                      # r=0, s=0
            "ff" * 64,                                      # r=all 1s, s=all 1s
            "00" * 32 + sample_specs.attestation_proof[64:], # r=0, valid s
            sample_specs.attestation_proof[:64] + "00" * 32, # valid r, s=0
        ]

        for proof in degenerate_proofs:
            tampered = sample_specs.model_copy(update={"attestation_proof": proof})
            assert (
                verify_attestation_proof(tampered, pubkey, timestamp=1788537600) is False
            ), f"Failed to reject degenerate signature: {proof[:16]}..."

    def test_bit_flip_sweep_on_valid_signature(self, sample_specs, test_signer):
        """Mutating each byte in the signature by 1 bit must invalidate the proof 100% of the time."""
        pubkey = test_signer.get_pubkey_hex()
        raw_sig = bytearray(bytes.fromhex(sample_specs.attestation_proof))
        assert len(raw_sig) == 64

        # Sample across r and s: test 10 evenly spaced byte positions
        test_indices = [0, 1, 15, 31, 32, 33, 47, 62, 63]
        for idx in test_indices:
            mutated = bytearray(raw_sig)
            mutated[idx] ^= 0x01  # Flip 1 bit
            mutated_hex = mutated.hex()

            tampered = sample_specs.model_copy(update={"attestation_proof": mutated_hex})
            assert (
                verify_attestation_proof(tampered, pubkey, timestamp=1788537600) is False
            ), f"Bit flip at byte {idx} was not detected!"


# ==============================================================================
# 4. PYDANTIC HARDWARESPECS VALIDATION TESTS
# ==============================================================================

class TestPydanticHardwareSpecsValidation:
    """Adversarial challenge: Stress-test HardwareSpecs schema boundaries and type validation."""

    def test_missing_required_fields(self):
        """Asserts Pydantic raises ValidationError when required fields are missing."""
        with pytest.raises(ValidationError):
            HardwareSpecs(vram_mb=24576, peak_fp32_tflops=82.6, attestation_proof="a" * 128)

        with pytest.raises(ValidationError):
            HardwareSpecs(gpu_model="RTX 4090", peak_fp32_tflops=82.6, attestation_proof="a" * 128)

        with pytest.raises(ValidationError):
            HardwareSpecs(gpu_model="RTX 4090", vram_mb=24576, attestation_proof="a" * 128)

        with pytest.raises(ValidationError):
            HardwareSpecs(gpu_model="RTX 4090", vram_mb=24576, peak_fp32_tflops=82.6)

    def test_non_convertible_types(self):
        """Asserts Pydantic rejects non-numeric inputs for numeric fields."""
        with pytest.raises(ValidationError):
            HardwareSpecs(
                gpu_model="RTX 4090",
                vram_mb="not_an_int",
                peak_fp32_tflops=82.6,
                attestation_proof="a" * 128,
            )

        with pytest.raises(ValidationError):
            HardwareSpecs(
                gpu_model="RTX 4090",
                vram_mb=24576,
                peak_fp32_tflops="not_a_float",
                attestation_proof="a" * 128,
            )

    def test_type_coercion_fidelity(self):
        """Pydantic coerces numeric strings to correct types."""
        specs = HardwareSpecs(
            gpu_model="NVIDIA RTX 4090",
            vram_mb="24576",  # string int
            peak_fp32_tflops="82.6",  # string float
            attestation_proof="a" * 128,
        )
        assert isinstance(specs.vram_mb, int)
        assert specs.vram_mb == 24576
        assert isinstance(specs.peak_fp32_tflops, float)
        assert specs.peak_fp32_tflops == 82.6

    def test_negative_values_schema_boundary_analysis(self):
        """
        Adversarial Observation:
        In matrix_contracts.schemas.discovery, HardwareSpecs defines:
          vram_mb: int = Field(..., description="Total VRAM in Megabytes")
          peak_fp32_tflops: float = Field(..., description="Peak FP32 TFLOPS rating")
        Because the contract schema lacks gt=0 or ge=0, HardwareSpecs accepts negative numbers.
        This test formally records that behavior.
        """
        # Negative VRAM
        specs_neg_vram = HardwareSpecs(
            gpu_model="RTX 4090",
            vram_mb=-1024,
            peak_fp32_tflops=82.6,
            attestation_proof="a" * 128,
        )
        assert specs_neg_vram.vram_mb == -1024

        # Negative TFLOPS
        specs_neg_tflops = HardwareSpecs(
            gpu_model="RTX 4090",
            vram_mb=24576,
            peak_fp32_tflops=-10.0,
            attestation_proof="a" * 128,
        )
        assert specs_neg_tflops.peak_fp32_tflops == -10.0

    def test_nan_and_infinity_tflops_handling(self):
        """Checks behavior with special float values (NaN, Inf, -Inf)."""
        # Standard Pydantic float accepts NaN / Inf unless allow_inf_nan=False
        specs_nan = HardwareSpecs(
            gpu_model="RTX 4090",
            vram_mb=24576,
            peak_fp32_tflops=float("nan"),
            attestation_proof="a" * 128,
        )
        assert math.isnan(specs_nan.peak_fp32_tflops)

        specs_inf = HardwareSpecs(
            gpu_model="RTX 4090",
            vram_mb=24576,
            peak_fp32_tflops=float("inf"),
            attestation_proof="a" * 128,
        )
        assert math.isinf(specs_inf.peak_fp32_tflops)

    def test_extreme_numeric_values(self):
        """Pydantic handles large integer and float boundaries safely."""
        extreme_vram = 2**63 - 1  # 64-bit max int
        extreme_tflops = 1e12     # 1 Petabyte/Exaflop scale

        specs = HardwareSpecs(
            gpu_model="Supercluster Accelerator",
            vram_mb=extreme_vram,
            peak_fp32_tflops=extreme_tflops,
            attestation_proof="a" * 128,
        )
        assert specs.vram_mb == extreme_vram
        assert specs.peak_fp32_tflops == extreme_tflops

    def test_json_serialization_roundtrip(self, sample_specs, test_signer):
        """HardwareSpecs serialized to JSON and reloaded preserves verifiable proof."""
        pubkey = test_signer.get_pubkey_hex()
        json_str = sample_specs.model_dump_json()

        deserialized = HardwareSpecs.model_validate_json(json_str)
        assert deserialized == sample_specs

        # Re-verify proof after roundtrip
        assert (
            verify_attestation_proof(deserialized, pubkey, timestamp=1788537600) is True
        ), "JSON roundtrip destroyed verification capability!"


# ==============================================================================
# 5. TIMESTAMP TOLERANCE & REPLAY TESTING
# ==============================================================================

class TestTimestampToleranceAndReplay:
    """Adversarial challenge: Verify timestamp window bounds and replay defenses."""

    def test_exact_timestamp_verification(self, sample_specs, test_signer):
        """Exact matching timestamp verifies successfully."""
        pubkey = test_signer.get_pubkey_hex()
        assert verify_attestation_proof(sample_specs, pubkey, timestamp=1788537600) is True

    def test_wrong_timestamp_rejection(self, sample_specs, test_signer):
        """Providing a different timestamp rejects the proof."""
        pubkey = test_signer.get_pubkey_hex()
        # Off by 1 second
        assert verify_attestation_proof(sample_specs, pubkey, timestamp=1788537601) is False
        assert verify_attestation_proof(sample_specs, pubkey, timestamp=1788537599) is False
        # Off by 1 hour
        assert verify_attestation_proof(sample_specs, pubkey, timestamp=1788537600 + 3600) is False

    def test_current_time_tolerance_window(self, test_signer):
        """When timestamp is None, verifies within recent window (+- 5 seconds)."""
        engine = HardwareAttestationEngine(signer=test_signer, device_index=0)
        with patch.object(engine, "_query_nvml", return_value=None):
            with patch.object(engine, "_query_nvidia_smi", return_value=None):
                specs = engine.detect_hardware()  # Uses current time.time()

        pubkey = test_signer.get_pubkey_hex()
        assert verify_attestation_proof(specs, pubkey) is True

    def test_expired_timestamp_outside_tolerance_window(self, test_signer):
        """A proof signed in the distant past (not matching cache or fixture) fails when ts=None."""
        engine = HardwareAttestationEngine(signer=test_signer, device_index=0)
        # Clear cache
        HardwareAttestationEngine._last_attestation_timestamp = None

        # Sign at an arbitrary past timestamp (e.g. 1000000)
        specs = engine.detect_hardware(timestamp=1000000)

        # Clear cache again so it doesn't match via _last_attestation_timestamp
        HardwareAttestationEngine._last_attestation_timestamp = None

        pubkey = test_signer.get_pubkey_hex()
        # When timestamp is None, it should fail because 1000000 is not now (+-5) and not 1788537600
        assert verify_attestation_proof(specs, pubkey, timestamp=None) is False


# ==============================================================================
# 6. ENGINE INVARIANTS & DEFENSIVE PACKAGING
# ==============================================================================

class TestEnginePackagingSafety:
    """Verifies internal safety invariants of HardwareAttestationEngine."""

    def test_detect_hardware_invariants(self, test_signer):
        """detect_hardware() guarantees vram_mb >= 0 and peak_fp32_tflops >= 0.5."""
        engine = HardwareAttestationEngine(signer=test_signer, device_index=0)
        specs = engine.detect_hardware()

        assert specs.vram_mb >= 0, f"detect_hardware() emitted negative VRAM: {specs.vram_mb}"
        assert specs.peak_fp32_tflops >= 0.5, (
            f"detect_hardware() emitted TFLOPS below 0.5 floor: {specs.peak_fp32_tflops}"
        )
        assert len(specs.gpu_model.strip()) > 0
        assert len(specs.attestation_proof) == 128  # Compact Secp256k1 hex

    def test_none_signer_produces_empty_proof(self):
        """When signer is None, attestation_proof is empty string, rejected by verifier."""
        engine = HardwareAttestationEngine(signer=None, device_index=0)
        specs = engine.detect_hardware()

        assert specs.attestation_proof == ""
        # Verifier must reject empty proof
        dummy_pubkey = "02" + "11" * 32
        assert verify_attestation_proof(specs, dummy_pubkey) is False

    def test_negative_device_index_raises_value_error(self, test_signer):
        """HardwareAttestationEngine strictly rejects negative device_index."""
        with pytest.raises(ValueError, match="device_index must be non-negative"):
            HardwareAttestationEngine(signer=test_signer, device_index=-1)

        with pytest.raises(ValueError, match="device_index must be non-negative"):
            HardwareAttestationEngine(signer=test_signer, device_index=-99)
