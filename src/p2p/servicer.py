"""P2P gRPC Compute Servicer implementation for matrix-compute-node."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Iterator, Optional, Set

import grpc
import numpy as np

from matrix_contracts.v1 import pb2, pb2_grpc
from src.auth.l402_gate import L402PreAllocationGate
from src.crypto.signer import WorkerSigner
from src.engine.compute_core import ComputeEngine
from src.engine.enclave_compute import (
    EnclaveComputeCore,
    EnclaveDecryptionError,
    EnclaveExecutionError,
    EnclaveFramingError,
    EnclaveKeyExchangeError,
    EnclaveSecurityPolicyError,
)
from src.attestation.base import BaseAttestationProvider

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
        attestation_provider: Optional[BaseAttestationProvider] = None,
        enclave_core: Optional[EnclaveComputeCore] = None,
    ) -> None:
        self.signer = signer
        self.gate = gate or L402PreAllocationGate()
        self.engine = engine or ComputeEngine()
        self.config = config
        self.enclave_core = enclave_core or EnclaveComputeCore(
            compute_engine=self.engine,
            max_sessions=1000,
            session_ttl_seconds=3600.0,
        )

        if attestation_provider is not None:
            self.attestation_provider = attestation_provider
        elif config is not None and getattr(config, "enable_confidential_compute", False):
            prov_type = getattr(config, "attestation_provider", "auto")
            if prov_type and prov_type.lower() != "none":
                try:
                    from src.attestation.base import get_attestation_provider
                    self.attestation_provider = get_attestation_provider(
                        provider_type=prov_type,
                        device_index=getattr(config, "gpu_device_index", 0),
                        fallback_to_simulated=True,
                    )
                except Exception as exc:
                    logger.warning("Failed initializing attestation provider from config: %s", exc)
                    self.attestation_provider = None
            else:
                self.attestation_provider = None
        else:
            self.attestation_provider = None

        self._lock = threading.Lock()
        self._task_metadata = TaskMetadataStore(max_size=1000, ttl_seconds=3600.0)
        self._active_tasks: Set[str] = set()

    def RequestTileTask(
        self,
        request: pb2.TileTaskRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.TileTaskAck:
        """
        Step 1: Handshake, parameter validation, theoretical GFLOP estimation,
        and Confidential Computing attestation evidence generation.
        """
        with self._lock:
            if request.task_id in self._active_tasks:
                self._abort(
                    context,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"Task '{request.task_id}' is currently in active streaming execution",
                )

        if request.matrix_a_rows <= 0 or request.matrix_a_cols <= 0 or request.matrix_b_cols <= 0:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "Matrix dimensions must be positive integers")

        m = int(request.matrix_a_rows)
        k = int(request.matrix_a_cols)
        n = int(request.matrix_b_cols)

        max_elem_limit = 64 * 1024 * 1024
        if (m * n) > max_elem_limit or max(m, n) > 16384:
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Tile dimensions ({m}x{k} @ {k}x{n}) exceed maximum allowed allocation limits (max_elem={max_elem_limit}, max_dim=16384)",
            )

        dtype = (request.dtype or "fp32").lower().strip()
        if dtype not in ("fp32", "fp16", "bf16"):
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Unsupported dtype: {dtype}")

        estimated_gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9

        worker_ephemeral_pubkey = b""
        attestation_evidence_bytes = b""
        tee_type = ""
        worker_ephemeral_priv = None

        if request.require_confidential_compute:
            if self.attestation_provider is None or not self.attestation_provider.is_available():
                if context is not None:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        "Confidential computing requested but worker has no operational attestation provider",
                    )
                raise RuntimeError(
                    "Confidential computing requested but worker has no operational attestation provider"
                )

            req_tee = request.required_tee_type.strip() if request.required_tee_type else ""
            if req_tee and req_tee not in (self.attestation_provider.provider_name, "any"):
                if context is not None:
                    context.abort(
                        grpc.StatusCode.UNIMPLEMENTED,
                        f"Worker TEE type '{self.attestation_provider.provider_name}' does not match requested '{req_tee}'",
                    )
                raise ValueError(
                    f"Worker TEE type '{self.attestation_provider.provider_name}' does not match requested '{req_tee}'"
                )

            if not request.client_ephemeral_pubkey or len(request.client_ephemeral_pubkey) != 32:
                if context is not None:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "client_ephemeral_pubkey must be exactly 32 bytes")
                raise ValueError("client_ephemeral_pubkey must be exactly 32 bytes")

            if not request.attestation_nonce or len(request.attestation_nonce) != 32:
                if context is not None:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "attestation_nonce must be exactly 32 bytes")
                raise ValueError("attestation_nonce must be exactly 32 bytes")

            # Generate worker ephemeral X25519 keypair
            from cryptography.hazmat.primitives.asymmetric import x25519
            from cryptography.hazmat.primitives import serialization

            worker_ephemeral_priv = x25519.X25519PrivateKey.generate()
            worker_ephemeral_pub = worker_ephemeral_priv.public_key()
            worker_ephemeral_pubkey = worker_ephemeral_pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )

            # Generate hardware attestation evidence bound to challenger nonce and worker ephemeral pubkey
            evidence = self.attestation_provider.generate_evidence(
                nonce=request.attestation_nonce,
                public_key=worker_ephemeral_pubkey,
            )
            attestation_evidence_bytes = evidence.to_bytes()
            tee_type = self.attestation_provider.provider_name

        with self._lock:
            meta: Dict[str, Any] = {
                "m": m,
                "k": k,
                "n": n,
                "dtype": dtype,
                "gflops": estimated_gflops,
                "created_at": time.time(),
                "is_confidential": bool(request.require_confidential_compute),
            }
            if request.require_confidential_compute:
                meta.update({
                    "worker_ephemeral_priv": worker_ephemeral_priv,
                    "worker_ephemeral_pubkey": worker_ephemeral_pubkey,
                    "client_ephemeral_pubkey": request.client_ephemeral_pubkey,
                    "attestation_nonce": request.attestation_nonce,
                })
                try:
                    self.enclave_core.create_session(
                        task_id=request.task_id,
                        m=m,
                        k=k,
                        n=n,
                        dtype=dtype,
                        require_confidential=True,
                        client_ephemeral_pubkey=request.client_ephemeral_pubkey,
                        attestation_nonce=request.attestation_nonce,
                        worker_ephemeral_priv=worker_ephemeral_priv,
                    )
                except EnclaveKeyExchangeError as exc:
                    self._abort(
                        context,
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Enclave key exchange failed: {exc}",
                    )
            else:
                self.enclave_core.create_session(
                    task_id=request.task_id,
                    m=m,
                    k=k,
                    n=n,
                    dtype=dtype,
                    require_confidential=False,
                )
            self._task_metadata[request.task_id] = meta

        worker_pubkey = self.signer.get_pubkey_hex() if self.signer is not None else ""

        return pb2.TileTaskAck(
            task_id=request.task_id,
            is_ready=True,
            estimated_gflops=estimated_gflops,
            worker_pubkey=worker_pubkey,
            message="Node ready",
            worker_ephemeral_pubkey=worker_ephemeral_pubkey,
            attestation_evidence=attestation_evidence_bytes,
            tee_type=tee_type,
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
        # 1. Extract Authorization header and metadata from invocation metadata (case-insensitive)
        auth_header = ""
        meta_task_id = None
        if context is not None:
            metadata_dict = {str(k).lower(): str(v) for k, v in context.invocation_metadata()}
            auth_header = metadata_dict.get("authorization") or metadata_dict.get("authorization-bin") or ""
            meta_task_id = metadata_dict.get("x-task-id")

        # 2. Strict Pre-Allocation Gate:
        # Check authorization and token BEFORE allocating VRAM or processing streams
        if not auth_header:
            self._abort(context, grpc.StatusCode.UNAUTHENTICATED, "Missing Authorization header with L402 token")

        macaroon_b64, payment_hash = self.gate.parse_auth_header(auth_header)
        if not macaroon_b64 or not payment_hash:
            self._abort(context, grpc.StatusCode.UNAUTHENTICATED, "Malformed L402 Authorization header")

        # Extract task_id from metadata or macaroon caveats
        task_id = meta_task_id
        if not task_id:
            try:
                import base64
                import json
                raw_bytes = base64.b64decode(macaroon_b64)
                payload = json.loads(raw_bytes.decode("utf-8"))
                caveats = payload.get("caveats", {})
                task_ids = caveats.get("task_ids") or caveats.get("task_batch") or caveats.get("tasks")
                if task_ids and isinstance(task_ids, list):
                    if len(task_ids) == 1:
                        task_id = str(task_ids[0]).strip()
                elif caveats.get("task_id"):
                    task_id = str(caveats.get("task_id")).strip()
            except Exception:
                pass

        if not task_id:
            self._abort(
                context,
                grpc.StatusCode.UNAUTHENTICATED,
                "Unable to determine task_id: batch macaroon requires explicit 'x-task-id' metadata",
            )

        with self._lock:
            if task_id in self._active_tasks:
                self._abort(
                    context,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"Task '{task_id}' is already actively executing",
                )
            self._active_tasks.add(task_id)

        try:
            task_info = self._task_metadata.get(task_id)

            m = task_info["m"] if task_info else 1
            k = task_info["k"] if task_info else 1
            n = task_info["n"] if task_info else 1
            dtype = task_info["dtype"] if task_info else "fp32"

            # Execute 5-step gate verification BEFORE allocating memory
            ok, err_msg, err_code = self.gate.verify_token(
                auth_header=auth_header,
                task_id=task_id,
                m=m,
                k=k,
                n=n,
                x_task_id=meta_task_id,
            )
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

            session = self.enclave_core.get_session(task_id)
            if session is None:
                session, _ = self.enclave_core.create_session(
                    task_id=task_id,
                    m=m,
                    k=k,
                    n=n,
                    dtype=dtype,
                    require_confidential=task_info.get("is_confidential", False),
                    client_ephemeral_pubkey=task_info.get("client_ephemeral_pubkey"),
                    attestation_nonce=task_info.get("attestation_nonce"),
                    worker_ephemeral_priv=task_info.get("worker_ephemeral_priv"),
                )

            # Ingest and decrypt stream chunks inside enclave execution boundary
            for chunk in request_iterator:
                try:
                    session.ingest_chunk(chunk)
                except EnclaveSecurityPolicyError as exc:
                    self._abort(context, grpc.StatusCode.PERMISSION_DENIED, str(exc))
                except (EnclaveDecryptionError, EnclaveFramingError) as exc:
                    self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))

            # Finalize and execute in-enclave GEMM, encrypt output
            try:
                result_bytes, iv, tag, is_encrypted, exec_time_ms = session.execute_and_encrypt()
            except EnclaveExecutionError as exc:
                self._abort(context, grpc.StatusCode.INTERNAL, str(exc))
            except EnclaveFramingError as exc:
                self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))

            # Cryptographic digest and signing
            c_hash = hashlib.sha256(result_bytes).hexdigest()
            sig = ""
            if self.signer is not None:
                payload_to_sign = f"{task_id}:{c_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload_to_sign)

            return pb2.TileResult(
                task_id=task_id,
                rows=session.m,
                cols=session.n,
                result_bytes=result_bytes,
                worker_signature=sig,
                execution_time_ms=max(exec_time_ms, 1),
                is_encrypted=is_encrypted,
                iv=iv,
                tag=tag,
            )
        finally:
            with self._lock:
                self._active_tasks.discard(task_id)
            self.enclave_core.close_session(task_id)
            self._task_metadata.pop(task_id, None)
