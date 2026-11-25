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
from src.auth.hashcash import HashcashVerifier
from src.auth.l402_gate import L402PreAllocationGate
from src.crypto.signer import WorkerSigner
from src.engine.compute_core import ComputeEngine
from src.engine.scheduler import PreemptiveComputeScheduler, PriorityTier
from src.engine.enclave_compute import (
    EnclaveComputeCore,
    EnclaveDecryptionError,
    EnclaveExecutionError,
    EnclaveFramingError,
    EnclaveKeyExchangeError,
    EnclaveSecurityPolicyError,
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    derive_session_keys,
)
from src.engine.session_manager import SessionExpiredError, SessionManager
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
    3. LeaseMatrix: Pins matrix A in VRAM/memory across iterative steps.
    4. StepIterative: Fast iterative GEMV matrix-vector steps (v_{t+1} = A * v_t).
    5. ReleaseLease: Explicit zeroization and VRAM deallocation.
    """

    def __init__(
        self,
        signer: Optional[WorkerSigner] = None,
        gate: Optional[L402PreAllocationGate] = None,
        engine: Optional[ComputeEngine] = None,
        config: Optional[Any] = None,
        attestation_provider: Optional[BaseAttestationProvider] = None,
        enclave_core: Optional[EnclaveComputeCore] = None,
        session_manager: Optional[SessionManager] = None,
        scheduler: Optional[PreemptiveComputeScheduler] = None,
        hashcash_verifier: Optional[HashcashVerifier] = None,
    ) -> None:
        self.signer = signer
        self.gate = gate
        self.engine = engine or ComputeEngine()
        self.config = config
        self.scheduler = scheduler or PreemptiveComputeScheduler()
        self.hashcash_verifier = hashcash_verifier or HashcashVerifier(
            min_bits=getattr(config, "trial_pow_bits", 12) if config else 12
        )
        self.enclave_core = enclave_core or EnclaveComputeCore(
            compute_engine=self.engine,
            max_sessions=1000,
            session_ttl_seconds=3600.0,
        )
        self.session_manager = session_manager or SessionManager(
            max_sessions=getattr(config, "max_sessions", 100) if config else 100,
            max_memory_bytes=getattr(config, "max_memory_bytes", 4 * 1024 * 1024 * 1024) if config else 4 * 1024 * 1024 * 1024,
            default_ttl_seconds=getattr(config, "lease_default_ttl_seconds", 3600.0) if config else 3600.0,
            compute_engine=self.engine,
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
        hashcash_stamp = None
        if context is not None:
            metadata_dict = {str(k).lower(): str(v) for k, v in context.invocation_metadata()}
            auth_header = metadata_dict.get("authorization") or metadata_dict.get("authorization-bin") or ""
            meta_task_id = metadata_dict.get("x-task-id")
            hashcash_stamp = metadata_dict.get("x-hashcash-stamp") or metadata_dict.get("x-hashcash")

        # 2. Strict Pre-Allocation Gate:
        # Check authorization and token BEFORE allocating VRAM or processing streams
        if not auth_header:
            self._abort(context, grpc.StatusCode.UNAUTHENTICATED, "Missing Authorization header with L402 token")

        macaroon_b64, payment_hash = self.gate.parse_auth_header(auth_header)
        if not macaroon_b64 or not payment_hash:
            self._abort(context, grpc.StatusCode.UNAUTHENTICATED, "Malformed L402 Authorization header")

        is_trial = macaroon_b64.startswith("mc_trial_")

        # Extract task_id from metadata or macaroon caveats
        task_id = meta_task_id
        if not task_id:
            try:
                import base64
                import json
                b64_clean = macaroon_b64[len("mc_trial_") :] if is_trial else macaroon_b64
                rem = len(b64_clean) % 4
                if rem:
                    b64_clean += "=" * (4 - rem)
                raw_bytes = (
                    base64.urlsafe_b64decode(b64_clean.encode("ascii"))
                    if ("-" in b64_clean or "_" in b64_clean)
                    else base64.b64decode(b64_clean)
                )
                payload = json.loads(raw_bytes.decode("utf-8"))
                caveats = payload.get("caveats", payload) if isinstance(payload, dict) else {}
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
            node_pubkey = self.signer.get_pubkey_hex() if (is_trial and self.signer is not None) else None
            ok, err_msg, err_code = self.gate.verify_token(
                auth_header=auth_header,
                task_id=task_id,
                m=m,
                k=k,
                n=n,
                x_task_id=meta_task_id,
                worker_pubkey=node_pubkey,
            )
            if not ok:
                status = grpc.StatusCode.UNAUTHENTICATED
                if err_code == "PERMISSION_DENIED":
                    status = grpc.StatusCode.PERMISSION_DENIED
                elif err_code == "FAILED_PRECONDITION":
                    status = grpc.StatusCode.FAILED_PRECONDITION
                self._abort(context, status, err_msg or "L402 Pre-allocation check failed")

            if is_trial and (hashcash_stamp is not None or bool(getattr(self.config, "require_trial_hashcash", False))):
                if not hashcash_stamp:
                    self._abort(
                        context,
                        grpc.StatusCode.UNAUTHENTICATED,
                        "Missing required x-hashcash-stamp metadata for FREE_TRIAL execution",
                    )
                pow_ok, pow_reason = self.hashcash_verifier.verify_and_consume(
                    stamp=hashcash_stamp,
                    expected_resource=task_id,
                )
                if not pow_ok:
                    pow_status = (
                        grpc.StatusCode.PERMISSION_DENIED
                        if pow_reason == "REPLAYED_HASHCASH_STAMP"
                        else grpc.StatusCode.UNAUTHENTICATED
                    )
                    self._abort(context, pow_status, f"Invalid Hashcash proof-of-work: {pow_reason}")

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

            # Finalize and execute in-enclave GEMM via PreemptiveComputeScheduler, encrypt output
            should_inject_fraud = bool(getattr(self.config, "inject_fraud", False))
            tier = PriorityTier.FREE_TRIAL if is_trial else PriorityTier.PAID_MAINNET
            try:
                result_bytes, iv, tag, is_encrypted, exec_time_ms = self.scheduler.execute(
                    tier,
                    lambda: session.execute_and_encrypt(inject_fraud=should_inject_fraud),
                )
            except EnclaveExecutionError as exc:
                self._abort(context, grpc.StatusCode.INTERNAL, str(exc))
            except EnclaveFramingError as exc:
                self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except RuntimeError as exc:
                if is_trial and "Trial execution throttled" in str(exc):
                    self._abort(context, grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
                raise

            # Cryptographic digest and signing
            c_hash = hashlib.sha256(result_bytes).hexdigest()
            sig = ""
            if self.signer is not None:
                payload_to_sign = f"{task_id}:{c_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload_to_sign)

            locked_sats_display = "trial" if is_trial else "paid"
            if not is_trial:
                try:
                    import base64
                    import json
                    raw_m = base64.b64decode(macaroon_b64)
                    m_obj = json.loads(raw_m.decode("utf-8"))
                    m_cav = m_obj.get("caveats", {}) if isinstance(m_obj, dict) else {}
                    if "locked_sats" in m_cav:
                        locked_sats_display = str(m_cav["locked_sats"])
                except Exception:
                    pass

            logger.info(
                "[TILE COMPLETED] task=%s (%dx%d @ %dx%d %s) in %d ms | batch_escrow=%s sats (hash=%s...)",
                task_id,
                m,
                k,
                k,
                n,
                dtype,
                max(exec_time_ms, 1),
                locked_sats_display,
                payment_hash[:12],
            )

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

    def LeaseMatrix(
        self,
        request: pb2.LeaseMatrixRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.LeaseMatrixResponse:
        """
        Allocates and pins matrix A in memory across multiple vector steps.
        Enforces upfront L402 verification, dimension bounding, and optional X25519 key exchange.
        """
        # 1. Parameter Validation
        lease_id = str(request.lease_id).strip()
        if not lease_id:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "lease_id must not be empty")

        rows = int(request.rows)
        cols = int(request.cols)
        if rows <= 0 or cols <= 0:
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Matrix dimensions must be positive integers, got ({rows}, {cols})",
            )

        max_elem_limit = 64 * 1024 * 1024
        if (rows * cols) > max_elem_limit or max(rows, cols) > 16384:
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Matrix dimensions ({rows}x{cols}) exceed maximum limits (max_elem={max_elem_limit}, max_dim=16384)",
            )

        dtype = (request.dtype or "fp32").lower().strip()
        if dtype not in ("fp32", "fp16", "bf16"):
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Unsupported dtype: {dtype}")

        ttl_seconds = int(request.ttl_seconds) if request.ttl_seconds > 0 else 3600
        ttl_seconds = min(ttl_seconds, 86400)  # Maximum 24h

        # 2. Confidential Computing Handshake (X25519 ECDH)
        item_size = 4 if dtype in ("fp32", "bf16") else 2
        expected_raw_bytes = rows * cols * item_size

        worker_ephemeral_pubkey = b""
        session_keys = None

        if request.require_confidential_compute:
            if not request.client_ephemeral_pubkey or len(request.client_ephemeral_pubkey) != 32:
                self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "client_ephemeral_pubkey must be exactly 32 bytes")

            from cryptography.hazmat.primitives.asymmetric import x25519
            from cryptography.hazmat.primitives import serialization

            try:
                worker_ephemeral_priv = x25519.X25519PrivateKey.generate()
                worker_ephemeral_pub = worker_ephemeral_priv.public_key()
                worker_ephemeral_pubkey = worker_ephemeral_pub.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )

                # Derive directional keys K_c2w, K_w2c using HKDF-SHA256
                salt = hashlib.sha256(f"iterative:{lease_id}".encode("utf-8")).digest()
                session_keys = derive_session_keys(
                    worker_priv=worker_ephemeral_priv,
                    client_pub_bytes=request.client_ephemeral_pubkey,
                    client_nonce=salt,
                )
            except Exception as exc:
                self._abort(context, grpc.StatusCode.INTERNAL, f"ECDH key exchange failed: {exc}")

            if len(request.matrix_bytes) != expected_raw_bytes + 28:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Confidential compute requires 28-byte encrypted framing (IV + ciphertext + tag)",
                )

        # 3. Extract Authorization metadata & Perform Upfront L402 Gate Verification
        auth_header = ""
        meta_lease_id = None
        if context is not None:
            metadata_dict = {str(k).lower(): str(v) for k, v in context.invocation_metadata()}
            auth_header = metadata_dict.get("authorization") or metadata_dict.get("authorization-bin") or ""
            meta_lease_id = metadata_dict.get("x-lease-id") or metadata_dict.get("x-task-id")

            if self.gate is not None:
                if not auth_header:
                    self._abort(context, grpc.StatusCode.UNAUTHENTICATED, "Missing Authorization header with L402 token")

                macaroon_b64, payment_hash = self.gate.parse_auth_header(auth_header)
                if not macaroon_b64 or not payment_hash:
                    self._abort(context, grpc.StatusCode.UNAUTHENTICATED, "Malformed L402 Authorization header")

                # Upfront gate verification: baseline GFLOPs for A (M x K x 1)
                ok, err_msg, err_code = self.gate.verify_token(
                    auth_header=auth_header,
                    task_id=lease_id,
                    m=rows,
                    k=cols,
                    n=1,
                    x_task_id=meta_lease_id or lease_id,
                )
                if not ok:
                    status = grpc.StatusCode.UNAUTHENTICATED
                    if err_code == "PERMISSION_DENIED":
                        status = grpc.StatusCode.PERMISSION_DENIED
                    elif err_code == "FAILED_PRECONDITION":
                        status = grpc.StatusCode.FAILED_PRECONDITION
                    self._abort(context, status, err_msg or "L402 Pre-allocation check failed")

        # 4. Matrix Deserialization & Payload Validation
        item_size = 4 if dtype in ("fp32", "bf16") else 2
        expected_raw_bytes = rows * cols * item_size

        matrix_bytes = request.matrix_bytes
        if request.require_confidential_compute:
            if len(matrix_bytes) != expected_raw_bytes + 28:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Confidential compute requires 28-byte encrypted framing (IV + ciphertext + tag)",
                )
            # Wire encrypted matrix: [IV (12B) || Tag (16B) || Ciphertext]
            iv = matrix_bytes[:12]
            tag = matrix_bytes[12:28]
            ct = matrix_bytes[28:]
            try:
                matrix_bytes = aes_gcm_decrypt(
                    key=session_keys.k_c2w,
                    iv=iv,
                    ciphertext=ct,
                    tag=tag,
                    aad=None,
                    fallback_aads=[f"{lease_id}:matrix".encode("utf-8"), b""],
                )
            except Exception as exc:
                self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Matrix decryption failed: {exc}")

        if len(matrix_bytes) != expected_raw_bytes:
            # Support 16-bit BF16 wire transfer if 2 bytes per element
            if dtype == "bf16" and len(matrix_bytes) == rows * cols * 2:
                raw_u16 = np.frombuffer(matrix_bytes, dtype=np.uint16)
                raw_u32 = raw_u16.astype(np.uint32) << 16
                matrix_arr = raw_u32.view(np.float32).reshape((rows, cols))
            else:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Payload size mismatch: expected {expected_raw_bytes} bytes for {rows}x{cols} {dtype}, got {len(matrix_bytes)} bytes",
                )
        else:
            np_dt = np.float32 if dtype in ("fp32", "bf16") else np.float16
            matrix_arr = np.frombuffer(bytearray(matrix_bytes), dtype=np_dt).reshape((rows, cols))

        if dtype == "bf16":
            matrix_arr = self.engine.truncate_to_bf16(matrix_arr)

        # 5. Pin in SessionManager
        try:
            session = self.session_manager.lease_matrix(
                lease_id=lease_id,
                matrix=matrix_arr,
                rows=rows,
                cols=cols,
                dtype=dtype,
                ttl_seconds=float(ttl_seconds),
                require_confidential_compute=bool(request.require_confidential_compute),
                client_ephemeral_pubkey=request.client_ephemeral_pubkey if request.require_confidential_compute else None,
                session_keys=session_keys,
            )
        except (ValueError, KeyError) as exc:
            self._abort(context, grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except (MemoryError, RuntimeError) as exc:
            self._abort(context, grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        except Exception as exc:
            self._abort(context, grpc.StatusCode.RESOURCE_EXHAUSTED, f"Session allocation failed: {exc}")

        worker_pubkey = self.signer.get_pubkey_hex() if self.signer is not None else ""

        return pb2.LeaseMatrixResponse(
            lease_id=lease_id,
            is_accepted=True,
            expires_at_unix_ms=int(session.expires_at_unix_ms),
            worker_pubkey=worker_pubkey,
            message=f"Lease accepted for {rows}x{cols} ({dtype})",
            worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        )

    def StepIterative(
        self,
        request: pb2.IterativeStepRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.IterativeStepResponse:
        """
        Executes a single matrix-vector multiplication step (v_{t+1} = A * v_t)
        using the pinned matrix tensor. Handles decryption, GEMV math, encryption, and signing.
        """
        lease_id = str(request.lease_id).strip()
        if not lease_id:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "lease_id must not be empty")

        # 1. Session Lookup & Expiration Check
        session = self.session_manager.get_session(lease_id, allow_expired=True)
        if session is None or session.matrix is None:
            self._abort(
                context,
                grpc.StatusCode.NOT_FOUND,
                f"Active lease '{lease_id}' not found or has been evicted",
            )

        if session.is_expired():
            self.session_manager.evict_expired()
            self._abort(
                context,
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Lease '{lease_id}' has expired at {session.expires_at_unix_ms}",
            )

        session.touch()

        # 2. Confidential Decryption
        raw_vector_bytes = request.vector_bytes
        if session.require_confidential_compute:
            if not request.is_encrypted:
                self._abort(
                    context,
                    grpc.StatusCode.PERMISSION_DENIED,
                    "Confidential compute enabled: step request must be encrypted",
                )
            if len(request.iv) != 12 or len(request.tag) != 16:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Invalid AES-GCM framing: IV must be 12 bytes (got {len(request.iv)}), tag must be 16 bytes (got {len(request.tag)})",
                )

            step_aad = f"{lease_id}:step:{request.step_index}".encode("utf-8")
            try:
                raw_vector_bytes = aes_gcm_decrypt(
                    key=session.session_keys.k_c2w,
                    iv=request.iv,
                    ciphertext=request.vector_bytes,
                    tag=request.tag,
                    aad=None,
                    fallback_aads=[step_aad, b""],
                )
            except (EnclaveDecryptionError, EnclaveFramingError, Exception) as exc:
                self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Step vector decryption failed: {exc}")
        else:
            if request.is_encrypted:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Session is unencrypted but request indicated is_encrypted=True",
                )

        # 3. Vector Deserialization & Dimensionality Check
        expected_cols = session.cols
        if request.vector_length and request.vector_length != expected_cols:
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Vector dimension mismatch: matrix inner dimension is {expected_cols}, request vector length is {request.vector_length}",
            )

        item_size = 4 if session.dtype in ("fp32", "bf16") else 2
        expected_len = expected_cols * item_size

        if len(raw_vector_bytes) != expected_len:
            if session.dtype == "bf16" and len(raw_vector_bytes) == expected_cols * 2:
                raw_u16 = np.frombuffer(raw_vector_bytes, dtype=np.uint16)
                raw_u32 = raw_u16.astype(np.uint32) << 16
                vec = raw_u32.view(np.float32)
            else:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Vector byte size mismatch: expected {expected_len} bytes for vector of length {expected_cols} ({session.dtype}), got {len(raw_vector_bytes)} bytes",
                )
        else:
            np_dt = np.float32 if session.dtype in ("fp32", "bf16") else np.float16
            vec = np.frombuffer(raw_vector_bytes, dtype=np_dt)

        if vec.size != expected_cols:
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Vector dimension mismatch: matrix inner dimension is {expected_cols}, vector length is {vec.size}",
            )

        # 4. Fast GEMV Execution
        t_start = time.perf_counter()
        try:
            result_vec = self.engine.gemv(session.matrix, vec, dtype=session.dtype)
        except Exception as exc:
            self._abort(context, grpc.StatusCode.INTERNAL, f"GEMV execution error: {exc}")
        t_elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        try:
            session.record_step()
        except SessionExpiredError as exp_err:
            self._abort(
                context,
                grpc.StatusCode.FAILED_PRECONDITION,
                "Session expired or zeroized during step execution",
            )

        raw_result_bytes = result_vec.tobytes()

        # 5. Confidential Result Encryption
        res_is_encrypted = False
        res_iv = b""
        res_tag = b""
        result_vector_bytes = raw_result_bytes

        if session.require_confidential_compute:
            try:
                ct, iv, tag = aes_gcm_encrypt(
                    key=session.session_keys.k_w2c,
                    plaintext=raw_result_bytes,
                    aad=None,
                )
                result_vector_bytes = ct
                res_iv = iv
                res_tag = tag
                res_is_encrypted = True
            except Exception as exc:
                self._abort(context, grpc.StatusCode.INTERNAL, f"Result encryption failed: {exc}")

        # 6. Secp256k1 Result Signing
        r_hash = hashlib.sha256(result_vector_bytes).hexdigest()
        sig = ""
        if self.signer is not None:
            payload_to_sign = f"{lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
            sig = self.signer.sign_message(payload_to_sign)

        return pb2.IterativeStepResponse(
            lease_id=lease_id,
            step_index=request.step_index,
            result_vector_bytes=result_vector_bytes,
            vector_length=int(session.rows),
            worker_signature=sig,
            execution_time_ms=max(t_elapsed_ms, 1),
            is_encrypted=res_is_encrypted,
            iv=res_iv,
            tag=res_tag,
        )

    def ReleaseLease(
        self,
        request: pb2.ReleaseLeaseRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> pb2.ReleaseLeaseResponse:
        """
        Explicitly releases the leased matrix tensor, executes cryptographic zeroization,
        and reclaims memory in SessionManager.
        """
        lease_id = str(request.lease_id).strip()
        if not lease_id:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "lease_id must not be empty")

        bytes_freed = self.session_manager.release_lease(lease_id)
        is_released = bytes_freed > 0
        if is_released:
            msg = f"Lease '{lease_id}' successfully released; {bytes_freed} bytes zeroized and reclaimed"
        else:
            msg = f"Lease '{lease_id}' was not active or already released"

        return pb2.ReleaseLeaseResponse(
            lease_id=lease_id,
            is_released=is_released,
            bytes_freed=bytes_freed,
            message=msg,
        )
