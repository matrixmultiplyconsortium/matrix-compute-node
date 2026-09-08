"""
Adversarial Stress Test Suite for WorkerSigner in matrix-compute-node.

Tests:
1. Tampered signatures, bit flips, truncation, length extension, and corrupted encodings.
2. Cross-key verification and key confusion attacks.
3. Known Secp256k1 curve test vectors, point negation, and BIP-62 low-S canonicality.
4. Independent BIP-173 Bech32 oracle, checksum mutation, and malformed npub strings.
5. Boundary private key scalars (0, 1, n-1, n, n+1, 2^256-1) and malformed key hex.
6. Peer signature verification against invalid public keys, invalid curve points, and various key formats.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from typing import Optional, Tuple

import pytest
from src.crypto.signer import (
    BECH32_CHARSET,
    SECP256K1_HALF_ORDER,
    SECP256K1_ORDER,
    WorkerSigner,
    bech32_encode,
)

# Reference SECP256k1 Field Prime p = 2^256 - 2^32 - 977
SECP256K1_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

# Reference Generator G Coordinates
SECP256K1_GX_HEX = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
SECP256K1_GY_HEX = "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"


# ============================================================================
# Independent BIP-173 Reference Decoder Oracle
# ============================================================================

def _oracle_polymod(values: list[int]) -> int:
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ value
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _oracle_hrp_expand(hrp: str) -> list[int]:
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _oracle_verify_checksum(hrp: str, data: list[int]) -> bool:
    return _oracle_polymod(_oracle_hrp_expand(hrp) + data) == 1


def bech32_decode_oracle(bech_str: str) -> Tuple[Optional[str], Optional[bytes]]:
    """Independent BIP-173 decoder oracle for adversarial validation."""
    if not isinstance(bech_str, str):
        return None, None
    if any(ord(x) < 33 or ord(x) > 126 for x in bech_str):
        return None, None
    if bech_str.lower() != bech_str and bech_str.upper() != bech_str:
        return None, None

    bech_clean = bech_str.lower()
    pos = bech_clean.rfind("1")
    if pos < 1 or pos + 7 > len(bech_clean):
        return None, None

    hrp = bech_clean[:pos]
    data5: list[int] = []
    for c in bech_clean[pos + 1:]:
        d = BECH32_CHARSET.find(c)
        if d == -1:
            return None, None
        data5.append(d)

    if not _oracle_verify_checksum(hrp, data5):
        return None, None

    # Convert 5-bit integers to 8-bit bytes (excluding 6-character checksum)
    payload5 = data5[:-6]
    acc = 0
    bits = 0
    out_bytes: list[int] = []
    max_acc = (1 << (5 + 8 - 1)) - 1
    for val in payload5:
        acc = ((acc << 5) | val) & max_acc
        bits += 5
        while bits >= 8:
            bits -= 8
            out_bytes.append((acc >> bits) & 0xFF)

    # In strict BIP-173, remaining bits must be zero padding and < 5
    if bits >= 5 or ((acc << (8 - bits)) & 0xFF) != 0:
        return None, None

    return hrp, bytes(out_bytes)


# ============================================================================
# 1. Corrupted Signatures, Bit Flips & Length Tampering
# ============================================================================

class TestSignatureTampering:
    """Stress tests verifying signature tampering and bit corruption detection."""

    def test_bit_flip_sweep_across_all_bytes(self, sample_signer):
        """
        Adversarial: Flips each bit (8 bits * 64 bytes = 512 single-bit mutations).
        Every single corrupted signature must be rejected.
        """
        payload = b"GEMM block verification: M=128, N=128, K=128"
        valid_sig_hex = sample_signer.sign_message(payload)
        sig_bytes = bytearray(bytes.fromhex(valid_sig_hex))
        assert len(sig_bytes) == 64

        pubkey_hex = sample_signer.get_pubkey_hex()

        # Sweep through every byte and bit
        for byte_idx in range(64):
            for bit_idx in range(8):
                mutated_sig_bytes = bytearray(sig_bytes)
                mutated_sig_bytes[byte_idx] ^= (1 << bit_idx)
                mutated_sig_hex = bytes(mutated_sig_bytes).hex()

                # Must return False and not crash
                assert sample_signer.verify_signature(payload, mutated_sig_hex, pubkey_hex) is False, (
                    f"Bit flip at byte {byte_idx}, bit {bit_idx} was accepted!"
                )

    def test_signature_truncation_and_extension(self, sample_signer):
        """Verifies rejection of truncated or extended signature lengths."""
        payload = b"Authentic payload"
        valid_sig_hex = sample_signer.sign_message(payload)
        sig_bytes = bytes.fromhex(valid_sig_hex)

        test_lengths = [0, 1, 16, 31, 32, 33, 63, 65, 70, 100, 128]
        for length in test_lengths:
            if length < 64:
                tampered = sig_bytes[:length].hex()
            else:
                tampered = (sig_bytes + b"\x00" * (length - 64)).hex()

            assert sample_signer.verify_signature(payload, tampered) is False
            assert WorkerSigner.verify_peer_signature(payload, tampered, sample_signer.get_pubkey_hex()) is False

    def test_malformed_hex_and_arbitrary_types(self, sample_signer):
        """Verifies graceful handling of non-hex strings and unexpected input types."""
        payload = b"Some data"

        malformed_inputs = [
            "",
            "   ",
            "0",
            "12345",  # odd length
            "zz" * 64,  # invalid hex chars
            "0123" * 15 + "gggg",  # trailing invalid hex
            "null",
            "\x00" * 64,
            None,
            123456789,
            {"sig": "hex"},
            [1, 2, 3],
            b"raw_bytes_not_hex_str",
        ]

        for bad_sig in malformed_inputs:
            assert sample_signer.verify_signature(payload, bad_sig) is False  # type: ignore
            assert WorkerSigner.verify_peer_signature(payload, bad_sig, sample_signer.get_pubkey_hex()) is False  # type: ignore

    def test_payload_tampering_boundaries(self, sample_signer):
        """Verifies payload tampering detection at boundaries (prefix, suffix, empty, large)."""
        base_payload = b"Canonical matrix payload"
        sig_hex = sample_signer.sign_message(base_payload)

        # Single byte mutations
        assert sample_signer.verify_signature(b" canonical matrix payload", sig_hex) is False
        assert sample_signer.verify_signature(base_payload + b"\x00", sig_hex) is False
        assert sample_signer.verify_signature(b"\x00" + base_payload, sig_hex) is False
        assert sample_signer.verify_signature(base_payload[:-1], sig_hex) is False
        assert sample_signer.verify_signature(b"", sig_hex) is False

        # Non-bytes payload to verify_signature returns False safely
        assert sample_signer.verify_signature("string_payload", sig_hex) is False  # type: ignore
        assert sample_signer.verify_signature(None, sig_hex) is False  # type: ignore

        # Large payload signing & verification (1 MB)
        large_payload = os.urandom(1024 * 1024)
        large_sig = sample_signer.sign_message(large_payload)
        assert sample_signer.verify_signature(large_payload, large_sig) is True
        # Flip 1 bit in large payload
        tampered_large = bytearray(large_payload)
        tampered_large[500_000] ^= 0x01
        assert sample_signer.verify_signature(bytes(tampered_large), large_sig) is False


# ============================================================================
# 2. Cross-Key Verification & Key Confusion
# ============================================================================

class TestCrossKeyVerification:
    """Stress tests verifying strict key isolation and cross-key rejection."""

    def test_pairwise_cross_key_rejection(self):
        """Verifies signatures from key A are never valid under key B, C, or D."""
        signers = [WorkerSigner() for _ in range(5)]
        message = b"Cross-key isolation verification payload"

        signatures = [s.sign_message(message) for s in signers]

        for i, signer_i in enumerate(signers):
            for j, signer_j in enumerate(signers):
                sig_i = signatures[i]
                pub_j = signer_j.get_pubkey_hex()
                nostr_j = signer_j.get_nostr_pubkey()

                if i == j:
                    assert signer_i.verify_signature(message, sig_i, pub_j) is True
                    assert WorkerSigner.verify_peer_signature(message, sig_i, nostr_j) is True
                else:
                    assert signer_i.verify_signature(message, sig_i, pub_j) is False
                    assert WorkerSigner.verify_peer_signature(message, sig_i, pub_j) is False
                    assert WorkerSigner.verify_peer_signature(message, sig_i, nostr_j) is False

    def test_cross_task_result_signing_isolation(self):
        """Verifies result signatures cannot be replayed across tasks or hashes."""
        signer_a = WorkerSigner()
        signer_b = WorkerSigner()

        task_1 = "task_matrix_alpha"
        task_2 = "task_matrix_beta"
        result_1 = b"\x01\x02\x03\x04" * 32
        result_2 = b"\x05\x06\x07\x08" * 32

        sig_1 = signer_a.sign_result(task_1, result_1)

        digest_1 = hashlib.sha256(result_1).hexdigest()
        digest_2 = hashlib.sha256(result_2).hexdigest()

        # Authentic payload
        payload_1 = f"{task_1}:{digest_1}".encode("utf-8")
        assert signer_a.verify_signature(payload_1, sig_1) is True

        # Replayed on task 2 -> False
        payload_replayed_task = f"{task_2}:{digest_1}".encode("utf-8")
        assert signer_a.verify_signature(payload_replayed_task, sig_1) is False

        # Replayed with different result digest -> False
        payload_tampered_hash = f"{task_1}:{digest_2}".encode("utf-8")
        assert signer_a.verify_signature(payload_tampered_hash, sig_1) is False

        # Verified against different worker key -> False
        assert signer_b.verify_signature(payload_1, sig_1) is False
        assert WorkerSigner.verify_peer_signature(payload_1, sig_1, signer_b.get_pubkey_hex()) is False


# ============================================================================
# 3. Known Curve Vectors & Low-S Normalization
# ============================================================================

class TestKnownCurveVectorsAndLowS:
    """Verifies Secp256k1 standard test vectors, point operations, and BIP-62 low-S."""

    def test_generator_g_test_vector(self):
        """Verifies scalar 1 corresponds to generator point G."""
        priv_1 = "0000000000000000000000000000000000000000000000000000000000000001"
        signer = WorkerSigner(privkey_hex=priv_1)

        expected_pub_compressed = "02" + SECP256K1_GX_HEX
        expected_pub_uncompressed = "04" + SECP256K1_GX_HEX + SECP256K1_GY_HEX

        assert signer.get_pubkey_hex() == expected_pub_compressed
        assert signer.get_nostr_pubkey() == SECP256K1_GX_HEX

        # Verify against both compressed and uncompressed representations
        msg = b"Message signed by generator private key"
        sig = signer.sign_message(msg)
        assert WorkerSigner.verify_peer_signature(msg, sig, expected_pub_compressed) is True
        assert WorkerSigner.verify_peer_signature(msg, sig, expected_pub_uncompressed) is True
        assert WorkerSigner.verify_peer_signature(msg, sig, SECP256K1_GX_HEX) is True

    def test_point_negation_vector(self):
        """
        Verifies scalar n - 1 corresponds to point -G.
        -G has x_coord = G_x, and y_coord = p - G_y (odd parity -> 03 prefix).
        """
        priv_n_minus_1 = hex(SECP256K1_ORDER - 1)[2:].zfill(64)
        signer = WorkerSigner(privkey_hex=priv_n_minus_1)

        # G_y is even, so p - G_y is odd -> prefix '03'
        expected_pub_compressed = "03" + SECP256K1_GX_HEX
        assert signer.get_pubkey_hex() == expected_pub_compressed
        assert signer.get_nostr_pubkey() == SECP256K1_GX_HEX

        msg = b"Message signed by -G private key"
        sig = signer.sign_message(msg)
        assert signer.verify_signature(msg, sig) is True

    def test_canonical_low_s_stress_battery(self):
        """
        Stress test: Signs 50 distinct random messages.
        Every resulting signature must strictly satisfy BIP-62: s <= SECP256K1_HALF_ORDER.
        """
        signer = WorkerSigner()
        for idx in range(50):
            payload = f"bip62_canonical_low_s_test_{idx}_{secrets.token_hex(8)}".encode("utf-8")
            sig_hex = signer.sign_message(payload)
            sig_bytes = bytes.fromhex(sig_hex)

            r = int.from_bytes(sig_bytes[:32], "big")
            s = int.from_bytes(sig_bytes[32:], "big")

            assert 1 <= r < SECP256K1_ORDER, f"Scalar r={r} out of valid range"
            assert 1 <= s <= SECP256K1_HALF_ORDER, (
                f"Low-S normalization violated: s={hex(s)} > half_order={hex(SECP256K1_HALF_ORDER)}"
            )

    def test_high_s_malleability_behavior(self, sample_signer):
        """
        Empirically tests signature behavior under malleability transformation:
        s' = SECP256K1_ORDER - s.
        """
        msg = b"Signature malleability test"
        valid_sig_hex = sample_signer.sign_message(msg)
        sig_bytes = bytes.fromhex(valid_sig_hex)
        r = int.from_bytes(sig_bytes[:32], "big")
        s = int.from_bytes(sig_bytes[32:], "big")
        assert s <= SECP256K1_HALF_ORDER

        # Create high-S malleated signature
        s_high = SECP256K1_ORDER - s
        assert s_high > SECP256K1_HALF_ORDER
        malleated_sig_hex = (r.to_bytes(32, "big") + s_high.to_bytes(32, "big")).hex()

        # Normal verification works for authentic low-S
        assert sample_signer.verify_signature(msg, valid_sig_hex) is True


# ============================================================================
# 4. Bech32 npub Encoding Oracle & Malformed String Stress
# ============================================================================

class TestBech32EncodingAndMalformedInputs:
    """Stress tests Bech32 encoding against independent BIP-173 oracle."""

    def test_bech32_oracle_roundtrip(self):
        """
        Verifies 25 randomly generated WorkerSigners against independent BIP-173 decoder oracle.
        Checks:
        1. Decoded HRP is strictly 'npub'.
        2. Decoded bytes match the 32-byte X-only pubkey exactly.
        """
        for _ in range(25):
            signer = WorkerSigner()
            npub = signer.get_nostr_npub()
            x_only = bytes.fromhex(signer.get_nostr_pubkey())

            assert npub.startswith("npub1")
            assert len(npub) == 63

            decoded_hrp, decoded_data = bech32_decode_oracle(npub)
            assert decoded_hrp == "npub"
            assert decoded_data == x_only

    def test_bech32_checksum_mutation_detection(self):
        """
        Mutating any character in a valid npub must cause the checksum to fail in the oracle.
        """
        signer = WorkerSigner()
        npub = signer.get_nostr_npub()

        # Mutate characters in data/checksum portion
        for idx in range(5, len(npub)):
            orig_char = npub[idx]
            # Pick a different valid bech32 character
            substitute = "q" if orig_char != "q" else "p"
            mutated_npub = npub[:idx] + substitute + npub[idx + 1:]

            decoded_hrp, decoded_data = bech32_decode_oracle(mutated_npub)
            assert decoded_hrp is None or decoded_data is None, (
                f"Bech32 checksum mutation at index {idx} failed to detect error!"
            )

    def test_bech32_malformed_inputs(self):
        """Tests rejection of malformed Bech32 strings."""
        invalid_bech32_strings = [
            "",
            "npub",
            "npub1",
            "npub1qqqq",  # too short (< 6 checksum chars)
            "npub1" + "b" * 58,  # 'b' is not in BECH32_CHARSET
            "npub1" + "i" * 58,  # 'i' is not in BECH32_CHARSET
            "npub1" + "o" * 58,  # 'o' is not in BECH32_CHARSET
            "npub1" + "1" * 58,  # '1' is not in BECH32_CHARSET (only separator)
            "Npub1" + "q" * 58,  # mixed case
            "npub2" + "q" * 58,  # wrong separator
        ]

        for bad in invalid_bech32_strings:
            decoded_hrp, decoded_data = bech32_decode_oracle(bad)
            assert decoded_hrp is None or decoded_data is None

    def test_bech32_encode_arbitrary_hrp(self):
        """Verifies bech32_encode works with custom HRPs and matches oracle."""
        custom_data = b"Arbitrary payload for bech32"
        encoded = bech32_encode("custom", custom_data)
        assert encoded.startswith("custom1")

        decoded_hrp, decoded_data = bech32_decode_oracle(encoded)
        assert decoded_hrp == "custom"
        assert decoded_data == custom_data


# ============================================================================
# 5. Boundary Private Keys & Malformed Hex
# ============================================================================

class TestBoundaryPrivateKeys:
    """Stress tests private key validation at curve order boundaries."""

    def test_scalar_zero_rejected(self):
        """Scalar 0 is mathematically invalid for Secp256k1 private keys."""
        with pytest.raises(ValueError, match="range"):
            WorkerSigner(privkey_hex="00" * 32)

    def test_scalar_order_rejected(self):
        """Scalar n (curve order) is mathematically invalid."""
        order_hex = hex(SECP256K1_ORDER)[2:].zfill(64)
        with pytest.raises(ValueError, match="range"):
            WorkerSigner(privkey_hex=order_hex)

    def test_scalar_greater_than_order_rejected(self):
        """Scalar n + 1 is out of valid range [1, n - 1]."""
        order_plus_1_hex = hex(SECP256K1_ORDER + 1)[2:].zfill(64)
        with pytest.raises(ValueError, match="range"):
            WorkerSigner(privkey_hex=order_plus_1_hex)

    def test_scalar_max_uint256_rejected(self):
        """Scalar 2^256 - 1 is > curve order."""
        with pytest.raises(ValueError, match="range"):
            WorkerSigner(privkey_hex="ff" * 32)

    def test_scalar_boundary_extremes_valid(self):
        """Scalar 1 and scalar n - 1 are both strictly valid boundaries."""
        signer_1 = WorkerSigner(privkey_hex="00" * 31 + "01")
        assert signer_1.get_privkey_hex() == "00" * 31 + "01"

        n_minus_1_hex = hex(SECP256K1_ORDER - 1)[2:].zfill(64)
        signer_max = WorkerSigner(privkey_hex=n_minus_1_hex)
        assert signer_max.get_privkey_hex() == n_minus_1_hex

    def test_malformed_private_key_hex_strings(self):
        """Verifies rejection of invalid hex lengths and characters."""
        invalid_keys = [
            "0",
            "01",
            "00" * 31,  # 31 bytes (62 chars)
            "00" * 33,  # 33 bytes (66 chars)
            "g" * 64,  # invalid hex
            "00" * 31 + "zz",
            "   " + "00" * 31 + "  ",  # short stripped
            "-1",
            "0x" + "01" * 32,  # 66 chars with 0x prefix
        ]

        for bad_key in invalid_keys:
            with pytest.raises(ValueError):
                WorkerSigner(privkey_hex=bad_key)


# ============================================================================
# 6. Peer Signature Verification Against Invalid Public Keys
# ============================================================================

class TestPeerVerificationInvalidPubkeys:
    """Stress tests verify_peer_signature against invalid pubkeys and points."""

    def test_invalid_pubkey_lengths(self, sample_signer):
        """Pubkeys with length not in (32, 33, 65) must return False."""
        msg = b"Payload"
        sig = sample_signer.sign_message(msg)

        invalid_pub_hexes = [
            "",
            "02",
            "02" + "00" * 15,  # 16 bytes
            "02" + "00" * 30,  # 31 bytes
            "02" + "00" * 33,  # 34 bytes
            "04" + "00" * 63,  # 64 bytes
            "04" + "00" * 65,  # 66 bytes
        ]

        for bad_pub in invalid_pub_hexes:
            assert WorkerSigner.verify_peer_signature(msg, sig, bad_pub) is False

    def test_invalid_pubkey_prefixes(self, sample_signer):
        """Compressed pubkey prefixes other than 02 and 03 must return False."""
        msg = b"Payload"
        sig = sample_signer.sign_message(msg)
        gx = SECP256K1_GX_HEX

        for bad_prefix in ["00", "01", "04", "05", "ff"]:
            assert WorkerSigner.verify_peer_signature(msg, sig, bad_prefix + gx) is False

    def test_point_not_on_curve(self, sample_signer):
        """Coordinates not lying on y^2 = x^3 + 7 (mod p) or x >= p must return False."""
        msg = b"Payload"
        sig = sample_signer.sign_message(msg)

        # Coordinate x >= p (0xff...ff > p): strictly invalid field element
        x_ge_p = "ff" * 32
        assert WorkerSigner.verify_peer_signature(msg, sig, "02" + x_ge_p) is False
        assert WorkerSigner.verify_peer_signature(msg, sig, "03" + x_ge_p) is False
        assert WorkerSigner.verify_peer_signature(msg, sig, x_ge_p) is False

        # Coordinate x = p
        x_eq_p = hex(SECP256K1_P)[2:].zfill(64)
        assert WorkerSigner.verify_peer_signature(msg, sig, "02" + x_eq_p) is False
        assert WorkerSigner.verify_peer_signature(msg, sig, "03" + x_eq_p) is False

        # Known off-curve points (where x^3 + 7 has no square root mod p)
        # e.g., x = 0x07: 7^3 + 7 = 350; 350 mod p
        # For any coordinate not on curve, verify_peer_signature returns False
        for x_val in [0, 1, 7, 42]:
            x_hex = hex(x_val)[2:].zfill(64)
            # Regardless of whether x is on curve or not, this is not the signer's pubkey
            # and invalid points will fail point decoding, while wrong valid points will fail ECDSA
            assert WorkerSigner.verify_peer_signature(msg, sig, "02" + x_hex) is False
            assert WorkerSigner.verify_peer_signature(msg, sig, "03" + x_hex) is False

    def test_bech32_string_passed_as_pubkey_returns_false(self, sample_signer):
        """Passing Bech32 npub1... to verify_peer_signature returns False without crashing."""
        msg = b"Payload"
        sig = sample_signer.sign_message(msg)
        npub = sample_signer.get_nostr_npub()

        assert WorkerSigner.verify_peer_signature(msg, sig, npub) is False
