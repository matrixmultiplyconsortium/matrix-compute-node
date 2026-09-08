"""Unit tests for WorkerSigner cryptographic operations."""

import hashlib
import pytest

from src.crypto.signer import SECP256K1_ORDER, SECP256K1_HALF_ORDER, WorkerSigner


def test_keypair_generation():
    """TC-CRY-01: Verifies fresh keypair generation and format conformance."""
    signer = WorkerSigner()
    priv = signer.get_privkey_hex()
    pub = signer.get_pubkey_hex()
    npk = signer.get_nostr_pubkey()
    npub = signer.get_nostr_npub()

    assert len(priv) == 64
    assert len(pub) == 66
    assert pub.startswith("02") or pub.startswith("03")
    assert len(npk) == 64
    assert npk == pub[2:]
    assert npub.startswith("npub1")
    assert len(npub) == 63


def test_known_test_vector_generator():
    """TC-CRY-02: Verifies public key derivation against standard SECP256k1 generator point G (scalar 1)."""
    priv_scalar_1 = "0000000000000000000000000000000000000000000000000000000000000001"
    signer = WorkerSigner(privkey_hex=priv_scalar_1)

    expected_pub = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
    expected_nostr = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"

    assert signer.get_pubkey_hex() == expected_pub
    assert signer.get_nostr_pubkey() == expected_nostr


def test_npub_bech32_encoding():
    """TC-CRY-03: Verifies Bech32 npub encoding structure."""
    signer = WorkerSigner()
    npub = signer.get_nostr_npub()
    assert npub.startswith("npub1")
    assert len(npub) == 63
    # Check charset
    from src.crypto.signer import BECH32_CHARSET
    for char in npub[5:]:
        assert char in BECH32_CHARSET


def test_sign_and_verify_valid(sample_signer):
    """TC-CRY-04: Verifies signature creation and positive verification."""
    message = b"Hermetic distributed matrix multiplication GEMM block verification"
    sig_hex = sample_signer.sign_message(message)

    assert len(sig_hex) == 128  # 64-byte compact signature (128 hex chars)
    pubkey = sample_signer.get_pubkey_hex()

    assert sample_signer.verify_signature(message, sig_hex) is True
    assert sample_signer.verify_signature(message, sig_hex, pubkey) is True


def test_verify_signature_tamper_detection(sample_signer):
    """TC-CRY-05: Modifying a single byte of signed payload invalidates signature."""
    message = b"Authentic matrix computation payload"
    sig_hex = sample_signer.sign_message(message)

    tampered_message = b"Tampered matrix computation payload"
    assert sample_signer.verify_signature(tampered_message, sig_hex) is False


def test_verify_wrong_pubkey(sample_signer):
    """TC-CRY-06: Validating signature against a different key returns False."""
    message = b"Signed with signer 1"
    sig_hex = sample_signer.sign_message(message)

    other_signer = WorkerSigner()
    other_pubkey = other_signer.get_pubkey_hex()

    assert sample_signer.verify_signature(message, sig_hex, other_pubkey) is False


def test_corrupted_signature_rejected(sample_signer):
    """TC-CRY-07: Corrupted or invalid hex signatures return False without crashing."""
    message = b"Signed message"
    sig_hex = sample_signer.sign_message(message)

    # Corrupt one byte
    corrupted_sig = "ff" + sig_hex[2:]
    assert sample_signer.verify_signature(message, corrupted_sig) is False

    # Completely invalid hex
    assert sample_signer.verify_signature(message, "invalid_hex_string") is False
    assert sample_signer.verify_signature(message, "") is False


def test_canonical_low_s(sample_signer):
    """TC-CRY-08: Verifies low-S normalization (BIP-62) is strictly satisfied."""
    for i in range(10):
        data = f"canonical_low_s_test_message_{i}".encode("utf-8")
        sig_hex = sample_signer.sign_message(data)
        sig_bytes = bytes.fromhex(sig_hex)
        s = int.from_bytes(sig_bytes[32:], "big")
        assert s <= SECP256K1_HALF_ORDER


def test_sign_result_convenience(sample_signer):
    """TC-CRY-09: Verifies sign_result convenience method over raw bytes and precomputed hashes."""
    task_id = "tile_task_001"
    tensor_bytes = b"\x01\x02\x03\x04" * 16

    # Sign raw bytes
    sig1 = sample_signer.sign_result(task_id, tensor_bytes)
    digest = hashlib.sha256(tensor_bytes).hexdigest()
    expected_payload = f"{task_id}:{digest}".encode("utf-8")
    assert sample_signer.verify_signature(expected_payload, sig1) is True

    # Sign precomputed digest
    sig2 = sample_signer.sign_result(task_id, digest)
    assert sample_signer.verify_signature(expected_payload, sig2) is True


def test_verify_peer_signature_formats(sample_signer):
    """TC-CRY-10: Verifies classmethod verify_peer_signature with 33-byte and 32-byte pubkeys."""
    message = b"Peer to peer handshake"
    sig_hex = sample_signer.sign_message(message)

    compressed_pub = sample_signer.get_pubkey_hex()
    xonly_pub = sample_signer.get_nostr_pubkey()

    assert WorkerSigner.verify_peer_signature(message, sig_hex, compressed_pub) is True
    assert WorkerSigner.verify_peer_signature(message, sig_hex, xonly_pub) is True


def test_invalid_private_key_raises():
    """TC-CRY-11: Verifies invalid private key representations raise ValueError."""
    with pytest.raises(ValueError):
        WorkerSigner(privkey_hex="0123")  # Short

    with pytest.raises(ValueError):
        WorkerSigner(privkey_hex="g" * 64)  # Invalid hex

    with pytest.raises(ValueError):
        # Scalar >= SECP256K1_ORDER
        too_large = hex(SECP256K1_ORDER)[2:].zfill(64)
        WorkerSigner(privkey_hex=too_large)
