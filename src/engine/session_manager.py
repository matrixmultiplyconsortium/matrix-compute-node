"""
Thread-Safe Pinned Tensor Session & Memory Lifecycle Management for matrix-compute-node.
Supports capacity bounding, memory footprint enforcement, hybrid TTL eviction,
and explicit buffer zeroization.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np

from src.engine.compute_core import ComputeEngine

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class SessionError(Exception):
    """Base exception for stateful session management."""


class SessionNotFoundError(SessionError, KeyError):
    """Raised when the specified lease_id is not found."""


class SessionExpiredError(SessionError, RuntimeError):
    """Raised when an operation is attempted on an expired lease."""


class SessionCapacityError(SessionError, RuntimeError):
    """Raised when maximum concurrent session limit is exceeded."""


class MemoryBoundError(SessionError, RuntimeError):
    """Raised when memory allocation bounds are exceeded."""


class InvalidMatrixDimensionsError(SessionError, ValueError):
    """Raised when matrix dimensions or byte payload sizes are invalid."""


# ==============================================================================
# TIMESTAMP HELPER
# ==============================================================================

class _IntTimestamp(int):
    """
    Subclass of int that allows being called as a function returning itself.
    Handles callers using either session.expires_at_unix_ms or session.expires_at_unix_ms().
    """

    def __call__(self) -> int:
        return int(self)


# ==============================================================================
# PINNED TENSOR SESSION
# ==============================================================================

class PinnedTensorSession:
    """
    Represents an allocated, pinned matrix session in worker memory/VRAM.
    Manages lifetime, access tracking, and zeroization for a single lease.
    """

    def __init__(
        self,
        lease_id: str,
        matrix: Optional[np.ndarray],
        rows: int,
        cols: int,
        dtype: str = "fp32",
        ttl_seconds: float = 3600.0,
        require_confidential_compute: bool = False,
        client_pubkey: Optional[bytes] = None,
        session_keys: Optional[Any] = None,
        worker_ephemeral_pubkey: bytes = b"",
    ) -> None:
        self.lease_id = str(lease_id).strip()
        self.matrix: Optional[np.ndarray] = matrix
        self.rows = int(rows)
        self.cols = int(cols)
        self.dtype = str(dtype).lower().strip()
        self.ttl_seconds = float(ttl_seconds)
        self.created_at = time.time()
        self.expires_at = self.created_at + self.ttl_seconds
        self.last_accessed_at = self.created_at
        self.require_confidential_compute = bool(require_confidential_compute)
        self.client_pubkey = client_pubkey
        self.session_keys = session_keys
        self.worker_ephemeral_pubkey = worker_ephemeral_pubkey
        self.step_count = 0
        if matrix is not None:
            self.memory_bytes = int(matrix.nbytes)
        else:
            elem_size = 4 if self.dtype in ("fp32", "bf16", "float32", "bfloat16") else 2
            self.memory_bytes = int(self.rows * self.cols * elem_size)
        self._zeroized = False
        self._lock = threading.RLock()

    @property
    def is_zeroized(self) -> bool:
        """Returns True if the session buffers have been zeroized."""
        with self._lock:
            return self._zeroized

    @property
    def np_dtype(self) -> np.dtype:
        """Returns corresponding NumPy datatype."""
        if self.dtype in ("fp16", "float16"):
            return np.float16
        return np.float32

    @property
    def is_valid(self) -> bool:
        """Returns True if the session is not zeroized and within TTL."""
        with self._lock:
            return not self._zeroized and not self.is_expired()

    @property
    def expires_at_unix_ms(self) -> _IntTimestamp:
        """Returns expiration time in milliseconds for gRPC response messages."""
        with self._lock:
            return _IntTimestamp(int(self.expires_at * 1000))

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Checks if session has reached its TTL expiration timestamp."""
        with self._lock:
            if self._zeroized:
                return True
            now = current_time if current_time is not None else time.time()
            return now >= self.expires_at

    def touch(self, current_time: Optional[float] = None, extend_ttl: bool = False) -> None:
        """Updates last_accessed_at timestamp and optionally refreshes TTL."""
        with self._lock:
            if self._zeroized:
                raise SessionExpiredError(f"Cannot touch zeroized session '{self.lease_id}'")
            now = current_time if current_time is not None else time.time()
            self.last_accessed_at = now
            if extend_ttl:
                self.expires_at = now + self.ttl_seconds

    def record_step(self) -> int:
        """Records execution of an iterative step, updating step count and access timestamp."""
        with self._lock:
            if self._zeroized or self.is_expired():
                raise SessionExpiredError(f"Session '{self.lease_id}' is expired or zeroized")
            self.step_count += 1
            self.last_accessed_at = time.time()
            return self.step_count

    def zeroize(self) -> int:
        """
        Securely clears pinned memory buffers and cryptographic session keys.
        Overwrites matrix buffer with zeros before releasing reference.
        Returns the number of memory bytes freed.
        """
        with self._lock:
            if self._zeroized:
                return 0
            freed = self.memory_bytes
            if self.matrix is not None:
                try:
                    if isinstance(self.matrix, np.ndarray):
                        if not self.matrix.flags.writeable:
                            self.matrix.setflags(write=True)
                        self.matrix.fill(0)
                except Exception as exc:
                    logger.warning("Error zeroizing matrix for lease '%s': %s", self.lease_id, exc)
                self.matrix = None

            if self.session_keys is not None:
                try:
                    if hasattr(self.session_keys, "zeroize"):
                        self.session_keys.zeroize()
                except Exception as exc:
                    logger.warning("Error zeroizing session keys for lease '%s': %s", self.lease_id, exc)
                self.session_keys = None

            self.client_pubkey = None
            self._zeroized = True
            return freed

    def __iter__(self) -> Iterator[Any]:
        """Allows unpacking (session, worker_ephemeral_pubkey) for tuple-based consumers."""
        yield self
        yield self.worker_ephemeral_pubkey

    def __getitem__(self, item: int) -> Any:
        if item == 0:
            return self
        elif item == 1:
            return self.worker_ephemeral_pubkey
        raise IndexError(f"Index {item} out of bounds for PinnedTensorSession tuple unpacking")


# ==============================================================================
# SESSION MANAGER
# ==============================================================================

class SessionManager:
    """
    Thread-safe in-memory session manager for pinned matrix tensors.
    Enforces maximum session count, memory bounds, and hybrid TTL eviction.
    """

    def __init__(
        self,
        max_sessions: int = 100,
        max_memory_bytes: int = 4 * 1024 * 1024 * 1024,  # 4 GB
        max_matrix_bytes: int = 1024 * 1024 * 1024,      # 1 GB
        default_ttl_seconds: float = 3600.0,              # 1 hour
        max_ttl_seconds: float = 86400.0,                 # 24 hours
        reaper_interval_seconds: float = 10.0,            # 10 seconds
        compute_engine: Optional[ComputeEngine] = None,
        auto_start_reaper: bool = True,
        engine: Optional[ComputeEngine] = None,
        max_active_sessions: Optional[int] = None,
        max_total_memory_bytes: Optional[int] = None,
    ) -> None:
        self.max_sessions = max_active_sessions if max_active_sessions is not None else max_sessions
        self.max_memory_bytes = max_total_memory_bytes if max_total_memory_bytes is not None else max_memory_bytes
        self.max_matrix_bytes = max_matrix_bytes
        self.default_ttl_seconds = default_ttl_seconds
        self.max_ttl_seconds = max_ttl_seconds
        self.reaper_interval_seconds = reaper_interval_seconds
        self.compute_engine = engine or compute_engine or ComputeEngine()

        self._sessions: Dict[str, PinnedTensorSession] = {}
        self._total_memory_bytes: int = 0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._reaper_thread: Optional[threading.Thread] = None

        if auto_start_reaper:
            self._start_reaper()

    def _start_reaper(self) -> None:
        """Starts background daemon reaper thread for proactive TTL eviction."""
        if self._reaper_thread is None or not self._reaper_thread.is_alive():
            self._stop_event.clear()
            self._reaper_thread = threading.Thread(
                target=self._reaper_loop,
                daemon=True,
                name="MatrixSessionReaper",
            )
            self._reaper_thread.start()

    def _reaper_loop(self) -> None:
        """Daemon thread loop waking every reaper_interval_seconds to purge expired leases."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self.reaper_interval_seconds):
                break
            try:
                freed = self.evict_expired()
                if freed > 0:
                    logger.debug("SessionReaper reclaimed %d bytes from expired sessions", freed)
            except Exception as exc:
                logger.warning("Unexpected error in SessionReaper loop: %s", exc)

    def lease_matrix(
        self,
        lease_id: str,
        *args: Any,
        matrix_bytes: Optional[bytes] = None,
        matrix: Optional[np.ndarray] = None,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        dtype: Optional[str] = None,
        ttl_seconds: Optional[Union[int, float]] = None,
        require_confidential_compute: bool = False,
        client_ephemeral_pubkey: Optional[bytes] = None,
        worker_ephemeral_priv: Optional[Any] = None,
        worker_nonce: bytes = b"",
        session_keys: Optional[Any] = None,
        **kwargs: Any,
    ) -> PinnedTensorSession:
        """
        Allocates and pins matrix A for a new stateful session.
        Enforces capacity limits, memory bounds, and parses raw matrix bytes.

        Flexible parameter handling supports:
        - lease_matrix(lease_id, matrix_bytes, rows, cols, dtype, ttl_seconds)
        - lease_matrix(lease_id, rows, cols, dtype, matrix_bytes, ttl_seconds)
        - lease_matrix(lease_id, matrix, rows, cols, dtype, ttl_seconds)
        """
        # Step 1: Lazy TTL eviction before evaluating capacity
        self.evict_expired()

        clean_id = str(lease_id).strip()
        if not clean_id:
            raise ValueError("lease_id must not be empty")

        # Parse positional args if provided
        mat_or_bytes = matrix_bytes if matrix_bytes is not None else matrix
        if len(args) > 0:
            first_arg = args[0]
            if isinstance(first_arg, (bytes, bytearray, np.ndarray)):
                mat_or_bytes = first_arg
                if len(args) > 1 and rows is None:
                    rows = args[1]
                if len(args) > 2 and cols is None:
                    cols = args[2]
                if len(args) > 3 and dtype is None:
                    dtype = args[3]
                if len(args) > 4 and ttl_seconds is None:
                    ttl_seconds = args[4]
            elif isinstance(first_arg, (int, float)):
                if rows is None:
                    rows = int(first_arg)
                if len(args) > 1 and cols is None:
                    cols = int(args[1])
                if len(args) > 2 and dtype is None:
                    dtype = str(args[2])
                if len(args) > 3 and mat_or_bytes is None:
                    mat_or_bytes = args[3]
                if len(args) > 4 and ttl_seconds is None:
                    ttl_seconds = args[4]

        if rows is None or cols is None or rows <= 0 or cols <= 0:
            raise InvalidMatrixDimensionsError(
                f"Matrix dimensions must be positive integers, got {rows}x{cols}"
            )

        norm_dtype = (dtype or "fp32").lower().strip()
        if norm_dtype in ("float32", "fp32"):
            norm_dtype = "fp32"
            elem_size = 4
            np_dtype = np.float32
        elif norm_dtype in ("float16", "fp16"):
            norm_dtype = "fp16"
            elem_size = 2
            np_dtype = np.float16
        elif norm_dtype in ("bfloat16", "bf16"):
            norm_dtype = "bf16"
            np_dtype = np.float32
            # Check length of bytes if provided
            if isinstance(mat_or_bytes, (bytes, bytearray)):
                elem_size = 4 if len(mat_or_bytes) == rows * cols * 4 else 2
            else:
                elem_size = 4
        else:
            raise ValueError(f"Unsupported dtype '{dtype}'. Supported: 'fp32', 'fp16', 'bf16'")

        # Step 2: Handle confidential key exchange if requested and not pre-derived
        worker_pubkey_bytes = b""
        if require_confidential_compute and session_keys is None:
            if client_ephemeral_pubkey is None or len(client_ephemeral_pubkey) != 32:
                raise ValueError("client_ephemeral_pubkey must be exactly 32 bytes for confidential compute")

            from cryptography.hazmat.primitives.asymmetric import x25519
            from cryptography.hazmat.primitives import serialization
            from src.engine.enclave_compute import derive_session_keys

            priv_key = worker_ephemeral_priv or x25519.X25519PrivateKey.generate()
            pub_key = priv_key.public_key()
            worker_pubkey_bytes = pub_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )

            salt = hashlib.sha256(f"iterative:{clean_id}".encode("utf-8")).digest()
            session_keys = derive_session_keys(
                worker_priv=priv_key,
                client_pub_bytes=client_ephemeral_pubkey,
                client_nonce=salt,
                worker_nonce=worker_nonce,
            )

        # Step 3: Reconstruct and pin matrix in memory
        if isinstance(mat_or_bytes, np.ndarray):
            matrix_arr = mat_or_bytes
            if matrix_arr.shape != (rows, cols):
                raise InvalidMatrixDimensionsError(
                    f"Matrix shape mismatch: got {matrix_arr.shape}, expected ({rows}, {cols})"
                )
            if not matrix_arr.flags.writeable:
                matrix_arr = matrix_arr.copy()
            if norm_dtype == "bf16":
                matrix_arr = ComputeEngine.truncate_to_bf16(matrix_arr)
        elif isinstance(mat_or_bytes, (bytes, bytearray)):
            expected_bytes = rows * cols * elem_size
            if len(mat_or_bytes) != expected_bytes:
                # Also check alternative 4-byte or 2-byte representation for bf16
                if norm_dtype == "bf16" and len(mat_or_bytes) in (rows * cols * 2, rows * cols * 4):
                    elem_size = 4 if len(mat_or_bytes) == rows * cols * 4 else 2
                else:
                    raise InvalidMatrixDimensionsError(
                        f"Matrix byte size mismatch: expected {expected_bytes} bytes for "
                        f"{rows}x{cols} ({norm_dtype}), got {len(mat_or_bytes)} bytes"
                    )

            writable_bytes = bytearray(mat_or_bytes)
            if norm_dtype == "bf16" and elem_size == 2:
                u16 = np.frombuffer(writable_bytes, dtype=np.uint16).reshape((rows, cols))
                u32 = u16.astype(np.uint32) << 16
                raw_matrix = u32.view(np.float32)
                matrix_arr = ComputeEngine.truncate_to_bf16(raw_matrix)
            elif norm_dtype == "bf16":
                raw_matrix = np.frombuffer(writable_bytes, dtype=np.float32).reshape((rows, cols))
                matrix_arr = ComputeEngine.truncate_to_bf16(raw_matrix)
            else:
                matrix_arr = np.frombuffer(writable_bytes, dtype=np_dtype).reshape((rows, cols))
        else:
            raise InvalidMatrixDimensionsError("Must provide either matrix_bytes or matrix array")

        pinned_matrix = matrix_arr
        alloc_bytes = int(pinned_matrix.nbytes)

        # Step 4: Validate capacity and bounds under lock
        with self._lock:
            if alloc_bytes > self.max_matrix_bytes:
                raise MemoryBoundError(
                    f"Memory limit exceeded: matrix size ({alloc_bytes} bytes) exceeds maximum "
                    f"single matrix limit ({self.max_matrix_bytes} bytes)"
                )

            if clean_id in self._sessions:
                existing = self._sessions[clean_id]
                if not existing.is_expired():
                    raise ValueError(f"Lease '{clean_id}' already exists")
                # Clean up existing expired session
                self._total_memory_bytes -= existing.zeroize()
                del self._sessions[clean_id]

            if len(self._sessions) >= self.max_sessions:
                raise SessionCapacityError(
                    f"Maximum sessions capacity exceeded: limit {self.max_sessions}"
                )

            if self._total_memory_bytes + alloc_bytes > self.max_memory_bytes:
                raise MemoryBoundError(
                    f"Memory limit exceeded: insufficient memory (requested {alloc_bytes} bytes, "
                    f"currently used {self._total_memory_bytes} bytes, limit {self.max_memory_bytes} bytes)"
                )

            # Resolve TTL
            eff_ttl = float(ttl_seconds if ttl_seconds and ttl_seconds > 0 else self.default_ttl_seconds)
            eff_ttl = min(eff_ttl, self.max_ttl_seconds)

            session = PinnedTensorSession(
                lease_id=clean_id,
                matrix=pinned_matrix,
                rows=rows,
                cols=cols,
                dtype=norm_dtype,
                ttl_seconds=eff_ttl,
                require_confidential_compute=require_confidential_compute,
                client_pubkey=client_ephemeral_pubkey,
                session_keys=session_keys,
                worker_ephemeral_pubkey=worker_pubkey_bytes,
            )

            self._sessions[clean_id] = session
            self._total_memory_bytes += alloc_bytes

        logger.info(
            "Leased matrix for session '%s': %dx%d, %s, %d bytes, TTL=%.1fs (active sessions: %d)",
            clean_id, rows, cols, norm_dtype, alloc_bytes, eff_ttl, len(self._sessions),
        )
        return session

    def get_session(self, lease_id: str, allow_expired: bool = False) -> Optional[PinnedTensorSession]:
        """
        Retrieves active session for lease_id.
        Performs lazy eviction if session has expired, returning None unless allow_expired=True.
        """
        clean_id = str(lease_id).strip()
        with self._lock:
            session = self._sessions.get(clean_id)
            if session is None:
                return None

            if session.is_expired():
                if allow_expired:
                    return session
                logger.info("Lazy eviction: Session '%s' expired, releasing resources", clean_id)
                self._sessions.pop(clean_id, None)
                freed = session.zeroize()
                self._total_memory_bytes = max(0, self._total_memory_bytes - freed)
                return None

            session.touch()
            return session

    def release_lease(self, lease_id: str) -> int:
        """
        Explicitly releases and zeroizes an active lease.
        Returns the number of memory bytes freed.
        """
        clean_id = str(lease_id).strip()
        with self._lock:
            session = self._sessions.pop(clean_id, None)
            if session is None:
                return 0
            freed = session.zeroize()
            self._total_memory_bytes = max(0, self._total_memory_bytes - freed)

        logger.info("Explicit release of lease '%s': reclaimed %d bytes", clean_id, freed)
        return freed

    def evict_expired(self, current_time: Optional[float] = None) -> int:
        """
        Scans all sessions, zeroizes and evicts those that have expired.
        Returns total bytes freed.
        """
        now = current_time if current_time is not None else time.time()
        total_freed = 0
        with self._lock:
            expired_ids = [lid for lid, s in self._sessions.items() if s.is_expired(now)]
            for lid in expired_ids:
                session = self._sessions.pop(lid, None)
                if session is not None:
                    freed = session.zeroize()
                    total_freed += freed
                    self._total_memory_bytes = max(0, self._total_memory_bytes - freed)
                    logger.info("Evicted expired session '%s' (reclaimed %d bytes)", lid, freed)
        return total_freed

    @property
    def active_session_count(self) -> int:
        """Returns current count of active sessions."""
        with self._lock:
            return len(self._sessions)

    @property
    def total_memory_bytes(self) -> int:
        """Returns total bytes currently allocated across all active pinned sessions."""
        with self._lock:
            return self._total_memory_bytes

    @property
    def is_shutdown(self) -> bool:
        """Returns True if the session manager has been shut down."""
        return self._stop_event.is_set()

    def shutdown(self) -> None:
        """Stops background reaper thread and zeroizes all active sessions."""
        self._stop_event.set()
        if self._reaper_thread is not None and self._reaper_thread.is_alive():
            self._reaper_thread.join(timeout=2.0)
            self._reaper_thread = None

        with self._lock:
            for session in list(self._sessions.values()):
                session.zeroize()
            self._sessions.clear()
            self._total_memory_bytes = 0

    def __enter__(self) -> "SessionManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()
