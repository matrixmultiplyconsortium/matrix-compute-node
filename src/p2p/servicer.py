"""P2P gRPC Compute Servicer implementation for matrix-compute-node."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Iterator, Optional

import grpc
import numpy as np

from matrix_contracts.v1 import pb2, pb2_grpc
from src.auth.l402_gate import L402PreAllocationGate
from src.crypto.signer import WorkerSigner
from src.engine.compute_core import ComputeEngine

logger = logging.getLogger(__name__)


class TaskMetadataStore:
    """Thread-safe bounded LRU and TTL store for negotiated task metadata."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def evict_expired(self, now: Optional[float] = None) -> None:
        current_time = now if now is not None else time.time()
        with self._lock:
            expired_keys = [
                k for k, v in self._store.items()
                if current_time - v.get("created_at", 0) > self.ttl_seconds
            ]
            for k in expired_keys:
                self._store.pop(k, None)

    def get(self, task_id: str, default: Any = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            self.evict_expired()
            if task_id in self._store:
                self._store.move_to_end(task_id)
                return self._store[task_id]
            return default

    def set(self, task_id: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self.evict_expired()
            if task_id in self._store:
                self._store.move_to_end(task_id)
            self._store[task_id] = data
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def pop(self, task_id: str, default: Any = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._store.pop(task_id, default)

    def __getitem__(self, task_id: str) -> Dict[str, Any]:
        val = self.get(task_id)
        if val is None:
            raise KeyError(task_id)
        return val

    def __setitem__(self, task_id: str, data: Dict[str, Any]) -> None:
        self.set(task_id, data)

    def __contains__(self, task_id: str) -> bool:
        with self._lock:
            self.evict_expired()
            return task_id in self._store

    def __len__(self) -> int:
        with self._lock:
            self.evict_expired()
            return len(self._store)


class MatrixComputeServicer(pb2_grpc.MatrixComputeServiceServicer):
    """
    gRPC Servicer implementing MatrixComputeService:
    1. RequestTileTask: Handshake, parameter validation, GFLOP quotation, and readiness.
    2. ExecuteTileTask: Strict pre-allocation L402 verification, streaming chunk ingestion,
       GEMM calculation, SHA256 hashing, and worker cryptographic signing.
    """

    def __init__(
        self,
        signer: Optional[WorkerSigner] = None,
        gate: Optional[L402PreAllocationGate] = None,
        engine: Optional[ComputeEngine] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.signer = signer
        self.gate = gate or L402PreAllocationGate()
        self.engine = engine or ComputeEngine()
        self.config = config
        self._lock = threading.Lock()
        self._task_metadata = TaskMetadataStore(max_size=1000, ttl_seconds=3600.0)

    def RequestTileTask(
        self,
        request: pb2.TileTaskRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.TileTaskAck:
        """
        Step 1: Handshake, parameter validation, and theoretical GFLOP estimation.
        """
        if request.matrix_a_rows <= 0 or request.matrix_a_cols <= 0 or request.matrix_b_cols <= 0:
            if context is not None:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Matrix dimensions must be positive integers")
            raise ValueError("Matrix dimensions must be positive integers")

        dtype = (request.dtype or "fp32").lower().strip()
        if dtype not in ("fp32", "fp16", "bf16"):
            if context is not None:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unsupported dtype: {dtype}")
            raise ValueError(f"Unsupported dtype: {dtype}")

        m = int(request.matrix_a_rows)
        k = int(request.matrix_a_cols)
        n = int(request.matrix_b_cols)
        estimated_gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9

        with self._lock:
            self._task_metadata[request.task_id] = {
                "m": m,
                "k": k,
                "n": n,
                "dtype": dtype,
                "gflops": estimated_gflops,
                "created_at": time.time(),
            }

        worker_pubkey = self.signer.get_pubkey_hex() if self.signer is not None else ""

        return pb2.TileTaskAck(
            task_id=request.task_id,
            is_ready=True,
            estimated_gflops=estimated_gflops,
            worker_pubkey=worker_pubkey,
            message="Node ready",
        )

    def _abort(self, context: Optional[grpc.ServicerContext], status_code: grpc.StatusCode, details: str) -> None:
        if context is not None:
            context.abort(status_code, details)
        else:
            if status_code == grpc.StatusCode.UNAUTHENTICATED:
                raise PermissionError(f"UNAUTHENTICATED: {details}")
            elif status_code == grpc.StatusCode.PERMISSION_DENIED:
                raise PermissionError(f"PERMISSION_DENIED: {details}")
            elif status_code == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError(f"FAILED_PRECONDITION: {details}")
            elif status_code == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError(f"INVALID_ARGUMENT: {details}")
            else:
                raise RuntimeError(f"{status_code}: {details}")

    def ExecuteTileTask(
        self,
        request_iterator: Iterator[pb2.TileDataChunk],
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.TileResult:
        """
        Step 2: Strict Pre-Allocation Gate verification, stream ingestion, and GEMM execution.
        """
        # 1. Extract Authorization header and metadata from invocation metadata
        auth_header = ""
        meta_task_id = None
        if context is not None:
            metadata_dict = dict(context.invocation_metadata())
            auth_header = metadata_dict.get("authorization") or metadata_dict.get("authorization-bin") or ""
            meta_task_id = metadata_dict.get("x-task-id")

        # 2. Strict Pre-Allocation Gate:
        # Check authorization and token BEFORE allocating VRAM or processing streams
        if not auth_header:
            self._abort(context, grpc.StatusCode.UNAUTHENTICATED, "Missing Authorization header with L402 token")

        macaroon_b64, payment_hash = self.gate.parse_auth_header(auth_header)
        if not macaroon_b64 or not payment_hash:
            self._abort(context, grpc.StatusCode.UNAUTHENTICATED, "Malformed L402 Authorization header")

        # Extract task_id from macaroon caveats if not in metadata
        task_id = meta_task_id
        if not task_id:
            try:
                import base64
                import json
                raw_bytes = base64.b64decode(macaroon_b64)
                payload = json.loads(raw_bytes.decode("utf-8"))
                task_id = payload.get("caveats", {}).get("task_id")
            except Exception:
                pass

        if not task_id:
            self._abort(context, grpc.StatusCode.UNAUTHENTICATED, "Unable to determine task_id from L402 token")

        try:
            task_info = self._task_metadata.get(task_id)

            m = task_info["m"] if task_info else 1
            k = task_info["k"] if task_info else 1
            n = task_info["n"] if task_info else 1
            dtype = task_info["dtype"] if task_info else "fp32"

            # Execute 5-step gate verification BEFORE allocating memory
            ok, err_msg, err_code = self.gate.verify_token(auth_header, task_id, m=m, k=k, n=n)
            if not ok:
                status = grpc.StatusCode.UNAUTHENTICATED
                if err_code == "PERMISSION_DENIED":
                    status = grpc.StatusCode.PERMISSION_DENIED
                elif err_code == "FAILED_PRECONDITION":
                    status = grpc.StatusCode.FAILED_PRECONDITION
                self._abort(context, status, err_msg or "L402 Pre-allocation check failed")

            if not task_info:
                self._abort(
                    context,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"Task '{task_id}' was not negotiated via RequestTileTask",
                )

            np_dtype = np.float32 if dtype in ("fp32", "bf16") else np.float16
            bytes_per_elem = 4 if np_dtype == np.float32 else 2

            expected_a_bytes = m * k * bytes_per_elem
            expected_b_bytes = k * n * bytes_per_elem

            # 3. Stream Ingestion & Chunk Assembly (with strict running byte bounds)
            a_byte_chunks = []
            b_byte_chunks = []
            running_a_bytes = 0
            running_b_bytes = 0
            expected_seq = 0
            final_seen = False

            for chunk in request_iterator:
                if chunk.sequence_number != expected_seq:
                    self._abort(
                        context,
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Out of order chunk: expected seq {expected_seq}, got {chunk.sequence_number}",
                    )

                if chunk.task_id != task_id:
                    self._abort(
                        context,
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Chunk task_id mismatch: expected '{task_id}', got '{chunk.task_id}'",
                    )

                chunk_bytes = chunk.chunk_bytes if chunk.chunk_bytes is not None else b""
                chunk_len = len(chunk_bytes)

                if chunk.is_matrix_a:
                    if running_a_bytes + chunk_len > expected_a_bytes:
                        self._abort(
                            context,
                            grpc.StatusCode.INVALID_ARGUMENT,
                            f"Matrix A stream exceeded expected byte limit ({expected_a_bytes} bytes): "
                            f"accumulated {running_a_bytes + chunk_len} bytes",
                        )
                    running_a_bytes += chunk_len
                    a_byte_chunks.append(chunk_bytes)
                else:
                    if running_b_bytes + chunk_len > expected_b_bytes:
                        self._abort(
                            context,
                            grpc.StatusCode.INVALID_ARGUMENT,
                            f"Matrix B stream exceeded expected byte limit ({expected_b_bytes} bytes): "
                            f"accumulated {running_b_bytes + chunk_len} bytes",
                        )
                    running_b_bytes += chunk_len
                    b_byte_chunks.append(chunk_bytes)

                expected_seq += 1
                if chunk.is_final_chunk:
                    final_seen = True
                    break

            if not final_seen:
                self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "Stream terminated without final chunk flag")

            # 4. Verify exact byte counts
            if running_a_bytes != expected_a_bytes:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Matrix A byte count mismatch: got {running_a_bytes}, expected {expected_a_bytes}",
                )

            if running_b_bytes != expected_b_bytes:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Matrix B byte count mismatch: got {running_b_bytes}, expected {expected_b_bytes}",
                )

            # Reconstruct Tensors
            a_bytes = b"".join(a_byte_chunks)
            b_bytes = b"".join(b_byte_chunks)

            matrix_a = np.frombuffer(a_bytes, dtype=np_dtype).reshape((m, k))
            matrix_b = np.frombuffer(b_bytes, dtype=np_dtype).reshape((k, n))

            # 5. Execute GEMM
            start_time = time.perf_counter()
            matrix_c = self.engine.gemm(matrix_a, matrix_b, dtype=dtype)
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)

            # 6. Cryptographic digest and signing
            c_bytes = matrix_c.tobytes()
            c_hash = self.engine.compute_sha256(matrix_c)

            sig = ""
            if self.signer is not None:
                payload_to_sign = f"{task_id}:{c_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload_to_sign)

            return pb2.TileResult(
                task_id=task_id,
                rows=m,
                cols=n,
                result_bytes=c_bytes,
                worker_signature=sig,
                execution_time_ms=max(exec_time_ms, 1),
            )
        finally:
            self._task_metadata.pop(task_id, None)
