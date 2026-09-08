"""Secp256k1 Cryptographic Signer for matrix-compute-node."""

from __future__ import annotations

import hashlib
import secrets
from typing import List, Optional, Tuple, Union

# Pure-Python Bech32 implementation (BIP-173 / Nostr NIP-19)
BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _bech32_polymod(values: List[int]) -> int:
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ value
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str) -> List[int]:
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _convertbits(data: bytes, frombits: int, tobits: int, pad: bool = True) -> Optional[List[int]]:
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    max_acc = (1 << (frombits + tobits - 1)) - 1
    for value in data:
        acc = ((acc << frombits) | value) & max_acc
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret


def bech32_encode(hrp: str, data_bytes: bytes) -> str:
    """Encodes arbitrary bytes into Bech32 format with the specified human-readable part."""
    data5 = _convertbits(data_bytes, 8, 5, True)
    if data5 is None:
        raise ValueError("Failed to convert bits for Bech32 encoding")
    values = _bech32_hrp_expand(hrp) + data5
    polymod = _bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(BECH32_CHARSET[d] for d in (data5 + checksum))


SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP256K1_HALF_ORDER = SECP256K1_ORDER // 2

# Cryptographic backend resolution (cryptography preferred, fallback to pure Python ecdsa)
try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import (
        decode_dss_signature,
        encode_dss_signature,
    )
    from cryptography.hazmat.primitives import hashes, serialization
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False
    try:
        import ecdsa
    except ImportError:
        ecdsa = None  # type: ignore


class WorkerSigner:
    """
    Secp256k1 cryptographic key manager, message signer, and signature verifier.
    Handles worker identity, Nostr npub derivation, and result signing.
    """

    def __init__(self, privkey_hex: Optional[str] = None):
        if privkey_hex is None or not str(privkey_hex).strip():
            self._privkey_int = secrets.randbelow(SECP256K1_ORDER - 1) + 1
        else:
            clean = str(privkey_hex).strip().lower()
            if len(clean) != 64:
                raise ValueError(f"Private key hex must be 64 characters (32 bytes), got {len(clean)}")
            try:
                self._privkey_int = int(clean, 16)
            except ValueError:
                raise ValueError("Private key contains invalid hexadecimal characters")
            if not (1 <= self._privkey_int < SECP256K1_ORDER):
                raise ValueError("Private key scalar out of valid range [1, SECP256K1_ORDER - 1]")

        self._privkey_bytes = self._privkey_int.to_bytes(32, "big")

        if _HAS_CRYPTOGRAPHY:
            self._priv_key_obj = ec.derive_private_key(self._privkey_int, ec.SECP256K1())
            self._pub_key_obj = self._priv_key_obj.public_key()
            self._pubkey_bytes_compressed = self._pub_key_obj.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.CompressedPoint,
            )
        elif ecdsa is not None:
            sk = ecdsa.SigningKey.from_secret_exponent(self._privkey_int, curve=ecdsa.SECP256k1)
            vk = sk.verifying_key
            self._pubkey_bytes_compressed = vk.to_string("compressed")
        else:
            raise RuntimeError("No cryptographic backend available (neither 'cryptography' nor 'ecdsa' found)")

    def get_privkey_hex(self) -> str:
        """Returns 32-byte hex private key (64 hex characters)."""
        return self._privkey_bytes.hex()

    def get_pubkey_hex(self) -> str:
        """Returns 33-byte compressed public key hex (66 hex characters)."""
        return self._pubkey_bytes_compressed.hex()

    def get_nostr_pubkey(self) -> str:
        """Returns 32-byte X-only Nostr public key hex (64 hex characters)."""
        return self._pubkey_bytes_compressed[1:33].hex()

    def get_nostr_npub(self) -> str:
        """Returns Bech32-encoded Nostr public key ('npub1...')."""
        x_only_bytes = self._pubkey_bytes_compressed[1:]
        return bech32_encode("npub", x_only_bytes)

    def get_npub(self) -> str:
        """Alias for get_nostr_npub."""
        return self.get_nostr_npub()

    def sign_message(self, data: bytes) -> str:
        """
        Computes SHA256(data) and generates a deterministic canonical ECDSA signature.
        Returns 64-byte compact hex signature (128 hex characters) with low-S normalization.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Data to sign must be bytes")

        if _HAS_CRYPTOGRAPHY:
            der_sig = self._priv_key_obj.sign(data, ec.ECDSA(hashes.SHA256()))
            r, s = decode_dss_signature(der_sig)
        else:
            digest = hashlib.sha256(data).digest()
            sk = ecdsa.SigningKey.from_secret_exponent(self._privkey_int, curve=ecdsa.SECP256k1)
            sig_bytes = sk.sign_digest_deterministic(digest, hashfunc=hashlib.sha256)
            r = int.from_bytes(sig_bytes[:32], "big")
            s = int.from_bytes(sig_bytes[32:], "big")

        # Enforce canonical low-S normalization (BIP-62 / BIP-141)
        if s > SECP256K1_HALF_ORDER:
            s = SECP256K1_ORDER - s

        r_bytes = r.to_bytes(32, "big")
        s_bytes = s.to_bytes(32, "big")
        return (r_bytes + s_bytes).hex()

    def sign_result(self, task_id: str, result_hash_or_bytes: Union[str, bytes]) -> str:
        """
        Signs (task_id, SHA256(result_bytes)) tuple.
        result_hash_or_bytes can be either raw result bytes or precomputed SHA256 hex string.
        """
        if isinstance(result_hash_or_bytes, (bytes, bytearray)):
            result_hash = hashlib.sha256(result_hash_or_bytes).hexdigest()
        else:
            result_hash = str(result_hash_or_bytes).strip()

        payload = f"{task_id}:{result_hash}".encode("utf-8")
        return self.sign_message(payload)

    def verify_signature(
        self,
        data: bytes,
        signature_hex: str,
        pubkey_hex: Optional[str] = None,
    ) -> bool:
        """
        Verifies signature over data against this signer's or an external public key.
        Supports both 64-byte compact hex (128 chars) and DER hex signatures.
        """
        target_pubkey = pubkey_hex if pubkey_hex is not None else self.get_pubkey_hex()
        return self.verify_peer_signature(data, signature_hex, target_pubkey)

    @classmethod
    def verify_peer_signature(cls, data: bytes, signature_hex: str, pubkey_hex: str) -> bool:
        """
        Verifies signature over data against a given peer's public key.
        Handles both 64-byte compact signatures (128 hex chars) and DER signatures.
        Accepts 33-byte compressed, 65-byte uncompressed, or 32-byte X-only public keys.
        """
        if not isinstance(data, (bytes, bytearray)):
            return False

        try:
            sig_bytes = bytes.fromhex(str(signature_hex).strip())
            pub_str = str(pubkey_hex).strip()
            pub_bytes = bytes.fromhex(pub_str)
        except Exception:
            return False

        # If 32-byte X-only pubkey is provided, try with both 02 and 03 prefixes
        candidate_pubs = []
        if len(pub_bytes) == 32:
            candidate_pubs = [b"\x02" + pub_bytes, b"\x03" + pub_bytes]
        elif len(pub_bytes) in (33, 65):
            candidate_pubs = [pub_bytes]
        else:
            return False

        for candidate_pub in candidate_pubs:
            if cls._verify_single_pubkey(data, sig_bytes, candidate_pub):
                return True

        return False

    @classmethod
    def _verify_single_pubkey(cls, data: bytes, sig_bytes: bytes, pub_bytes: bytes) -> bool:
        """Helper to verify against a single 33-byte or 65-byte public key."""
        if _HAS_CRYPTOGRAPHY:
            try:
                peer_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), pub_bytes)

                if len(sig_bytes) == 64:
                    r = int.from_bytes(sig_bytes[:32], "big")
                    s = int.from_bytes(sig_bytes[32:], "big")
                    s_candidates = [s]
                    if s != 0:
                        s_candidates.append(SECP256K1_ORDER - s)
                    for s_cand in s_candidates:
                        try:
                            der_bytes = encode_dss_signature(r, s_cand)
                            peer_pub.verify(der_bytes, data, ec.ECDSA(hashes.SHA256()))
                            return True
                        except Exception:
                            continue
                    return False
                else:
                    der_bytes = sig_bytes
                    peer_pub.verify(der_bytes, data, ec.ECDSA(hashes.SHA256()))
                    return True
            except Exception:
                return False
        elif ecdsa is not None:
            try:
                vk = ecdsa.VerifyingKey.from_string(pub_bytes, curve=ecdsa.SECP256k1)
                digest = hashlib.sha256(data).digest()
                if len(sig_bytes) == 64:
                    return vk.verify_digest(sig_bytes, digest)
                else:
                    return vk.verify_digest(sig_bytes, digest, sigdecode=ecdsa.util.sigdecode_der)
            except Exception:
                return False
        return False
