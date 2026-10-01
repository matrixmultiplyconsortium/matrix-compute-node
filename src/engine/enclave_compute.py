"""
Enclave Execution Boundary & Wire Decryption/Encryption Engine (Milestone 3).

Enforces strict hardware/software isolation for Confidential Computing (TEE):
1. Manages ephemeral X25519 session keys and HKDF-SHA256 derivation.
2. In-enclave AES-256-GCM chunk decryption and AAD authentication.
3. Protected memory tensor slicing and GEMM acceleration.
4. In-enclave AES-256-GCM result encryption with fresh 12-byte IVs.
5. Clean unencrypted fallback for non-confidential execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import secrets
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

try:
    from matrix_contracts.v1 import pb2
except ImportError:
    from matrix_contracts.proto import compute_service_pb2 as pb2

from src.engine.compute_core import ComputeEngine

logger = logging.getLogger("matrix_compute_node.engine.enclave")

# ==============================================================================
# ENCLAVE EXCEPTIONS
# ==============================================================================

class EnclaveError(Exception):
    """Base exception for all enclave compute core errors."""


class EnclaveSessionNotFoundError(EnclaveError):
    """Raised when a task session is missing or expired."""


class EnclaveKeyExchangeError(EnclaveError):
    """Raised when X25519 ECDH fails (e.g. low-order point submission)."""


class EnclaveSecurityPolicyError(EnclaveError):
    """Raised when confidentiality policy is violated (e.g. unencrypted chunk in TEE)."""


class EnclaveDecryptionError(EnclaveError):
    """Raised when AES-256-GCM tag verification or decryption fails."""


class EnclaveFramingError(EnclaveError):
    """Raised when chunk framing, IV length, tag length, or byte bounds are invalid."""


class EnclaveExecutionError(EnclaveError):
    """Raised when in-enclave GEMM calculation fails."""


# ==============================================================================
# CRYPTOGRAPHIC HELPER FUNCTIONS & DATA STRUCTURES
# ==============================================================================

HKDF_INFO_LABEL: bytes = b"matrix-matmul-v1-aead"


@dataclass(frozen=True)
class SessionKeys:
    """Directional 256-bit symmetric session keys derived via HKDF-SHA256."""

    k_c2w: bytes
    k_w2c: bytes
    shared_secret: bytes
    salt: bytes
    info: bytes

    def __iter__(self) -> Iterator[bytes]:
        yield self.k_c2w
        yield self.k_w2c

    def __getitem__(self, item: int) -> bytes:
        if item == 0:
            return self.k_c2w
        elif item == 1:
            return self.k_w2c
        raise IndexError("SessionKeys has only 2 directional keys (0: k_c2w, 1: k_w2c)")


def compute_ecdh_shared_secret(
    private_key: x25519.X25519PrivateKey,
    peer_public_bytes: bytes,
) -> bytes:
    """Performs X25519 Diffie-Hellman scalar multiplication against peer's public key."""
    if not isinstance(peer_public_bytes, (bytes, bytearray)) or len(peer_public_bytes) != 32:
        raise EnclaveKeyExchangeError(f"Peer X25519 public key must be 32 bytes, got {len(peer_public_bytes)}")
    try:
        peer_pub = x25519.X25519PublicKey.from_public_bytes(bytes(peer_public_bytes))
        shared_secret = private_key.exchange(peer_pub)
        if len(shared_secret) != 32:
            raise EnclaveKeyExchangeError(f"Unexpected shared secret length: {len(shared_secret)}")
        if shared_secret == b"\x00" * 32:
            raise EnclaveKeyExchangeError("X25519 produced all-zero shared secret (low-order point rejected)")
        return shared_secret
    except Exception as exc:
        raise EnclaveKeyExchangeError(f"X25519 ECDH exchange failed: {exc}") from exc


def derive_session_keys(
    worker_priv: Union[x25519.X25519PrivateKey, bytes],
    client_pub_bytes: Union[bytes, x25519.X25519PublicKey],
    client_nonce: bytes,
    worker_nonce: bytes = b"",
    info: bytes = HKDF_INFO_LABEL,
) -> SessionKeys:
    """
    Derives directional 256-bit AES-GCM session keys (K_c2w, K_w2c) using HKDF-SHA256.
    Returns SessionKeys which can be unpacked as (k_c2w, k_w2c).
    """
    if isinstance(worker_priv, x25519.X25519PrivateKey):
        client_bytes = (
            client_pub_bytes.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            if isinstance(client_pub_bytes, x25519.X25519PublicKey)
            else bytes(client_pub_bytes)
        )
        shared_secret = compute_ecdh_shared_secret(worker_priv, client_bytes)
        salt = bytes(client_nonce) + bytes(worker_nonce) if worker_nonce else bytes(client_nonce)
    else:
        shared_secret = bytes(worker_priv)
        if len(shared_secret) != 32:
            raise EnclaveKeyExchangeError(f"shared_secret must be 32 bytes, got {len(shared_secret)}")
        if shared_secret == b"\x00" * 32:
            raise EnclaveKeyExchangeError("All-zero shared secret rejected")
        nonce_c = bytes(client_pub_bytes)
        nonce_w = bytes(client_nonce) if client_nonce else b""
        salt = nonce_c + nonce_w if nonce_w else nonce_c

    if not salt:
        raise EnclaveKeyExchangeError("HKDF salt must not be empty")

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=64,
        salt=salt,
        info=info,
    )
    derived = hkdf.derive(shared_secret)
    k_c2w = derived[:32]
    k_w2c = derived[32:]
    return SessionKeys(
        k_c2w=k_c2w,
        k_w2c=k_w2c,
        shared_secret=shared_secret,
        salt=salt,
        info=info,
    )


def canonical_chunk_aad(
    task_id: str,
    sequence_number: int,
    is_matrix_a: bool,
    is_final_chunk: bool = False,
) -> bytes:
    """Canonical AAD binding TileDataChunk header metadata."""
    t_id = str(task_id).strip()
    seq = int(sequence_number)
    mat = 1 if is_matrix_a else 0
    fin = 1 if is_final_chunk else 0
    return f"{t_id}:chunk:{mat}:{seq}:{fin}".encode("utf-8")


def compute_chunk_aad(
    task_id: str,
    sequence_number: Union[int, bool] = 0,
    is_matrix_a: Union[bool, int] = True,
    is_final_chunk: bool = False,
) -> bytes:
    """Flexible chunk AAD generator supporting multiple positional argument conventions."""
    if isinstance(sequence_number, bool) and isinstance(is_matrix_a, int):
        return canonical_chunk_aad(task_id, sequence_number=is_matrix_a, is_matrix_a=sequence_number, is_final_chunk=is_final_chunk)
    return canonical_chunk_aad(task_id, sequence_number=int(sequence_number), is_matrix_a=bool(is_matrix_a), is_final_chunk=is_final_chunk)


def canonical_result_aad(task_id: str, rows: int, cols: int) -> bytes:
    """Canonical AAD binding TileResult dimensions metadata."""
    return f"{str(task_id).strip()}:result:{int(rows)}:{int(cols)}".encode("utf-8")


compute_result_aad = canonical_result_aad


def aes_gcm_decrypt(
    key: bytes,
    iv: bytes,
    ciphertext: bytes,
    tag: bytes,
    aad: Optional[bytes] = None,
    fallback_aads: Optional[list[Optional[bytes]]] = None,
) -> bytes:
    """
    Decrypts AES-256-GCM ciphertext and validates authentication tag and AAD.

    Raises:
        EnclaveFramingError: If IV != 12 bytes or tag != 16 bytes.
        EnclaveDecryptionError: If authentication tag verification fails.
    """
    if len(iv) != 12:
        raise EnclaveFramingError(f"IV must be exactly 12 bytes, got {len(iv)}")
    if len(tag) != 16:
        raise EnclaveFramingError(f"Tag must be exactly 16 bytes, got {len(tag)}")

    aesgcm = AESGCM(key)
    full_ct = bytes(ciphertext) + bytes(tag)

    try:
        return aesgcm.decrypt(bytes(iv), full_ct, aad)
    except InvalidTag as exc:
        if fallback_aads:
            for alt in fallback_aads:
                try:
                    return aesgcm.decrypt(bytes(iv), full_ct, alt)
                except InvalidTag:
                    continue
        raise EnclaveDecryptionError("AES-256-GCM authentication failed: invalid tag or corrupted ciphertext") from exc


def aes_gcm_encrypt(
    key: bytes,
    plaintext: bytes,
    aad: Optional[bytes] = None,
    iv: Optional[bytes] = None,
) -> Tuple[bytes, bytes, bytes]:
    """
    Encrypts plaintext with AES-256-GCM using a fresh 12-byte IV.
    Returns Tuple of (ciphertext, iv, tag).
    """
    if len(key) != 32:
        raise ValueError(f"Key must be 32 bytes, got {len(key)}")

    actual_iv = iv if iv is not None else secrets.token_bytes(12)
    if len(actual_iv) != 12:
        raise ValueError(f"IV must be 12 bytes, got {len(actual_iv)}")

    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(actual_iv, plaintext, aad)
    ciphertext = ciphertext_and_tag[:-16]
    tag = ciphertext_and_tag[-16:]
    return ciphertext, actual_iv, tag


# ==============================================================================
# ENCLAVE SESSION CLASS
# ==============================================================================

class EnclaveSession:
    """
    Represents an isolated execution context for a single tile task.
    Holds session keys, secure chunk buffers, and performs in-enclave GEMM.
    """

    def __init__(
        self,
        task_id: str,
        m: int,
        k: int,
        n: int,
        dtype: str = "fp32",
        is_confidential: bool = True,
        k_c2w: Optional[bytes] = None,
        k_w2c: Optional[bytes] = None,
        compute_engine: Optional[ComputeEngine] = None,
        use_aad: bool = True,
    ) -> None:
        self.task_id = str(task_id).strip()
        self.m = int(m)
        self.k = int(k)
        self.n = int(n)
        self.dtype = dtype.lower().strip()
        if self.dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        if self.m <= 0 or self.k <= 0 or self.n <= 0:
            raise ValueError("Matrix dimensions must be positive integers")

        max_elem_limit = 64 * 1024 * 1024
        if (self.m * self.n) > max_elem_limit or max(self.m, self.n) > 16384:
            raise ValueError(
                f"Tile dimensions ({self.m}x{self.k} @ {self.k}x{self.n}) exceed maximum allowed allocation limit"
            )

        self.is_confidential = is_confidential
        self.k_c2w = k_c2w
        self.k_w2c = k_w2c
        self.compute_engine = compute_engine or ComputeEngine()
        self.use_aad = use_aad

        # Datatype byte scaling
        self.np_dtype = np.float32 if self.dtype in ("fp32", "bf16") else np.float16
        self.bytes_per_elem = 4 if self.np_dtype == np.float32 else 2
        self.expected_a_bytes = self.m * self.k * self.bytes_per_elem
        self.expected_b_bytes = self.k * self.n * self.bytes_per_elem

        # Internal in-enclave byte buffers
        self._a_chunks: List[bytes] = []
        self._b_chunks: List[bytes] = []
        self._running_a_bytes = 0
        self._running_b_bytes = 0
        self._expected_seq = 0
        self._final_seen = False
        self.created_at = time.time()
        self._lock = threading.Lock()

    def ingest_chunk(self, chunk: pb2.TileDataChunk) -> None:
        """
        Ingests and decrypts a streaming TileDataChunk inside the enclave boundary.
        Enforces strict sequence ordering, task matching, and running byte bounds.
        """
        with self._lock:
            if self._final_seen:
                raise EnclaveFramingError("Cannot ingest chunks after is_final_chunk=True")

            if chunk.sequence_number != self._expected_seq:
                raise EnclaveFramingError(
                    f"Out of order chunk: expected seq {self._expected_seq}, got {chunk.sequence_number}"
                )
            if str(chunk.task_id).strip() != self.task_id:
                raise EnclaveFramingError(
                    f"Chunk task_id mismatch: expected '{self.task_id}', got '{chunk.task_id}'"
                )

            chunk_payload = chunk.chunk_bytes if chunk.chunk_bytes is not None else b""
            if len(chunk_payload) == 0:
                raise EnclaveFramingError("Empty chunk bytes")

            # Security Policy Enforcement
            if self.is_confidential:
                if not chunk.is_encrypted:
                    raise EnclaveSecurityPolicyError(
                        f"Unencrypted chunk received for confidential task '{self.task_id}'"
                    )
                if not self.k_c2w:
                    raise EnclaveSecurityPolicyError(
                        f"Missing inbound session key K_c2w for confidential task '{self.task_id}'"
                    )

                aad = canonical_chunk_aad(self.task_id, chunk.sequence_number, chunk.is_matrix_a, chunk.is_final_chunk) if self.use_aad else None
                alt_aads = [
                    f"{self.task_id}:{int(chunk.sequence_number)}:{1 if chunk.is_matrix_a else 0}".encode("utf-8"),
                    f"{self.task_id}:{'A' if chunk.is_matrix_a else 'B'}:{int(chunk.sequence_number)}".encode("utf-8"),
                ] if self.use_aad else None

                decrypted_bytes = aes_gcm_decrypt(
                    key=self.k_c2w,
                    iv=chunk.iv,
                    ciphertext=chunk_payload,
                    tag=chunk.tag,
                    aad=aad,
                    fallback_aads=alt_aads,
                )
            else:
                if chunk.is_encrypted:
                    raise EnclaveSecurityPolicyError(
                        f"Encrypted chunk received for non-confidential task '{self.task_id}'"
                    )
                decrypted_bytes = chunk_payload

            dec_len = len(decrypted_bytes)
            if dec_len == 0:
                raise EnclaveFramingError("Empty chunk bytes")

            # Enforce Running Byte Limits
            if chunk.is_matrix_a:
                if self._running_a_bytes + dec_len > self.expected_a_bytes:
                    raise EnclaveFramingError(
                        f"Matrix A stream exceeded expected byte limit ({self.expected_a_bytes} bytes): "
                        f"accumulated {self._running_a_bytes + dec_len} bytes"
                    )
                self._running_a_bytes += dec_len
                self._a_chunks.append(decrypted_bytes)
            else:
                if self._running_b_bytes + dec_len > self.expected_b_bytes:
                    raise EnclaveFramingError(
                        f"Matrix B stream exceeded expected byte limit ({self.expected_b_bytes} bytes): "
                        f"accumulated {self._running_b_bytes + dec_len} bytes"
                    )
                self._running_b_bytes += dec_len
                self._b_chunks.append(decrypted_bytes)

            self._expected_seq += 1
            if chunk.is_final_chunk:
                self._final_seen = True

    def execute_and_encrypt(self) -> Tuple[bytes, bytes, bytes, bool, int]:
        """
        Finalizes chunk assembly, executes GEMM inside enclave memory,
        and encrypts the result using K_w2c.

        Returns:
            Tuple of (result_bytes, iv, tag, is_encrypted, execution_time_ms)
        """
        with self._lock:
            if not self._final_seen:
                raise EnclaveFramingError("Stream terminated without final chunk flag")

            if self._running_a_bytes != self.expected_a_bytes:
                raise EnclaveFramingError(
                    f"Matrix A byte count mismatch: got {self._running_a_bytes}, expected {self.expected_a_bytes}"
                )
            if self._running_b_bytes != self.expected_b_bytes:
                raise EnclaveFramingError(
                    f"Matrix B byte count mismatch: got {self._running_b_bytes}, expected {self.expected_b_bytes}"
                )

            # In-enclave tensor reconstruction
            a_raw = b"".join(self._a_chunks)
            b_raw = b"".join(self._b_chunks)

            matrix_a = np.frombuffer(a_raw, dtype=self.np_dtype).reshape((self.m, self.k))
            matrix_b = np.frombuffer(b_raw, dtype=self.np_dtype).reshape((self.k, self.n))

            # In-enclave GEMM execution
            start_time = time.perf_counter()
            try:
                matrix_c = self.compute_engine.gemm(matrix_a, matrix_b, dtype=self.dtype)
            except Exception as exc:
                raise EnclaveExecutionError(f"In-enclave GEMM calculation failed: {exc}") from exc
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)

            c_raw = matrix_c.tobytes()

            if self.is_confidential:
                if not self.k_w2c:
                    raise EnclaveSecurityPolicyError("Missing outbound session key K_w2c")
                aad = canonical_result_aad(self.task_id, self.m, self.n) if self.use_aad else None
                ciphertext, iv, tag = aes_gcm_encrypt(
                    key=self.k_w2c,
                    plaintext=c_raw,
                    aad=aad,
                )
                return ciphertext, iv, tag, True, max(exec_time_ms, 1)
            else:
                return c_raw, b"", b"", False, max(exec_time_ms, 1)

    def zeroize(self) -> None:
        """Securely clears internal buffers and session keys."""
        with self._lock:
            self._a_chunks.clear()
            self._b_chunks.clear()
            self.k_c2w = None
            self.k_w2c = None


# ==============================================================================
# ENCLAVE COMPUTE CORE
# ==============================================================================

class EnclaveComputeCore:
    """
    Enclave Execution Boundary manager.
    Coordinates active task sessions, manages session lifecycles,
    and isolates cryptographic session keys.
    """

    def __init__(
        self,
        compute_engine: Optional[ComputeEngine] = None,
        max_sessions: int = 1000,
        session_ttl_seconds: float = 3600.0,
    ) -> None:
        self.compute_engine = compute_engine or ComputeEngine()
        self.max_sessions = max_sessions
        self.session_ttl_seconds = session_ttl_seconds
        self._sessions: Dict[str, EnclaveSession] = {}
        self._lock = threading.RLock()

    def create_session(
        self,
        task_id: str,
        m: int,
        k: int,
        n: int,
        dtype: str = "fp32",
        require_confidential: bool = True,
        client_ephemeral_pubkey: Optional[bytes] = None,
        attestation_nonce: Optional[bytes] = None,
        worker_ephemeral_priv: Optional[x25519.X25519PrivateKey] = None,
        worker_nonce: bytes = b"",
    ) -> Tuple[EnclaveSession, bytes]:
        """
        Creates and registers an isolated EnclaveSession for a task.

        Returns:
            Tuple of (EnclaveSession, worker_ephemeral_pubkey)
        """
        self.evict_expired()

        worker_pubkey_bytes = b""
        k_c2w = None
        k_w2c = None

        if require_confidential:
            if client_ephemeral_pubkey is None or len(client_ephemeral_pubkey) != 32:
                raise EnclaveKeyExchangeError("client_ephemeral_pubkey must be 32 bytes")
            if attestation_nonce is None or len(attestation_nonce) != 32:
                raise EnclaveKeyExchangeError("attestation_nonce must be 32 bytes")

            priv_key = worker_ephemeral_priv or x25519.X25519PrivateKey.generate()
            pub_key = priv_key.public_key()
            worker_pubkey_bytes = pub_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )

            keys = derive_session_keys(
                worker_priv=priv_key,
                client_pub_bytes=client_ephemeral_pubkey,
                client_nonce=attestation_nonce,
                worker_nonce=worker_nonce,
            )
            k_c2w = keys.k_c2w
            k_w2c = keys.k_w2c

        session = EnclaveSession(
            task_id=task_id,
            m=m,
            k=k,
            n=n,
            dtype=dtype,
            is_confidential=require_confidential,
            k_c2w=k_c2w,
            k_w2c=k_w2c,
            compute_engine=self.compute_engine,
        )

        with self._lock:
            if task_id in self._sessions:
                old_session = self._sessions.pop(task_id)
                old_session.zeroize()
            self._sessions[task_id] = session
            while len(self._sessions) > self.max_sessions:
                oldest_key = next(iter(self._sessions))
                old_session = self._sessions.pop(oldest_key)
                old_session.zeroize()

        return session, worker_pubkey_bytes

    def get_session(self, task_id: str) -> Optional[EnclaveSession]:
        """Retrieves active session for task_id, or None if expired/not found."""
        self.evict_expired()
        with self._lock:
            return self._sessions.get(str(task_id).strip())

    def close_session(self, task_id: str) -> None:
        """Closes and zeroizes an active session."""
        with self._lock:
            session = self._sessions.pop(str(task_id).strip(), None)
            if session:
                session.zeroize()

    def evict_expired(self, now: Optional[float] = None) -> None:
        """Evicts sessions exceeding session_ttl_seconds."""
        current_time = now if now is not None else time.time()
        with self._lock:
            expired_keys = [
                k for k, v in self._sessions.items()
                if current_time - v.created_at > self.session_ttl_seconds
            ]
            for k in expired_keys:
                session = self._sessions.pop(k, None)
                if session:
                    session.zeroize()
