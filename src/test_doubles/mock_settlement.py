"""In-memory Mock Settlement Service test double for offline L402 testing."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from matrix_contracts.schemas.settlement import (
    CreateHodlInvoiceRequest,
    CreateHodlInvoiceResponse,
    SettlePaymentRequest,
)


class InvoiceStatus(str, Enum):
    UNPAID = "UNPAID"
    HELD = "HELD"
    SETTLED = "SETTLED"
    CANCELLED = "CANCELLED"


class MockInvoiceResponse(CreateHodlInvoiceResponse):
    """
    Response model compatible with both Pydantic attribute access (resp.payment_hash)
    and dict-like subscription (resp["payment_hash"], resp.get("preimage")).
    """
    preimage: Optional[str] = None

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)


@dataclass
class InvoiceRecord:
    task_id: str
    payment_hash: str
    total_compute_sats: int
    platform_fee_percent: float
    platform_fee_sats: int
    worker_payout_sats: int
    worker_lightning_address: str
    payment_request: str
    macaroon: str
    status: InvoiceStatus = InvoiceStatus.UNPAID
    preimage: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    max_gflops: float = 100.0
    task_ids: Optional[List[str]] = None
    job_id: Optional[str] = None


class MockSettlementService:
    """
    Hermetic in-memory Settlement Service and L402 Macaroon issuer.
    Simulates the Lightning HODL invoice lifecycle: UNPAID -> HELD -> SETTLED.
    """

    def __init__(self, verification_key: str = "settlement-secret-verification-key-12345"):
        self.verification_key = verification_key
        self._lock = threading.Lock()
        self._invoices: Dict[str, InvoiceRecord] = {}       # payment_hash -> record
        self._task_index: Dict[str, str] = {}              # task_id -> payment_hash
        self.query_count: int = 0

    def _mint_macaroon(
        self,
        payment_hash: str,
        task_id: str,
        expires_at: int,
        max_gflops: float,
        locked_sats: int = 100,
        task_ids: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Mints an HMAC-SHA256 authenticated L402 Macaroon token."""
        caveats: Dict[str, Any] = {
            "expiry": int(expires_at),
            "locked_sats": locked_sats,
            "max_gflops": float(max_gflops),
            "payment_hash": payment_hash,
            "task_id": task_id,
        }
        if task_ids is not None:
            caveats["task_ids"] = [str(t).strip() for t in task_ids if str(t).strip()]
        if job_id is not None:
            caveats["job_id"] = str(job_id).strip()

        caveat_str = json.dumps(caveats, sort_keys=True)
        sig = hmac.new(
            self.verification_key.encode("utf-8"),
            caveat_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload = {"caveats": caveats, "signature": sig}
        return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

    def verify_macaroon(
        self,
        macaroon_raw: str,
        expected_task_id: str,
        required_gflops: float,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Verifies an L402 Macaroon:
        1. Validates HMAC-SHA256 signature against verification key.
        2. Asserts caveat task containment (expected_task_id in task_ids or == task_id).
        3. Asserts caveat expiry > current_time.
        4. Asserts caveat max_gflops >= required_gflops.
        """
        try:
            raw_bytes = base64.b64decode(macaroon_raw)
            payload = json.loads(raw_bytes.decode("utf-8"))
            if not isinstance(payload, dict):
                return False, "Malformed macaroon token: payload must be a JSON object"
            if "caveats" not in payload or "signature" not in payload:
                return False, "Malformed macaroon token: missing caveats or signature"
            caveats = payload["caveats"]
            signature = payload["signature"]
            if not isinstance(signature, str):
                return False, "Malformed macaroon signature"
            if not isinstance(caveats, dict):
                return False, "Malformed macaroon token: caveats must be a JSON object"

            # 1. Verify HMAC signature
            caveat_str = json.dumps(caveats, sort_keys=True)
            expected_sig = hmac.new(
                self.verification_key.encode("utf-8"),
                caveat_str.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_sig):
                return False, "Invalid macaroon HMAC signature"

            # 2. Check task_id containment
            allowed_tasks: List[str] = []
            caveat_tasks = caveats.get("task_ids") or caveats.get("task_batch") or caveats.get("tasks")
            if caveat_tasks is not None:
                if isinstance(caveat_tasks, list):
                    allowed_tasks.extend(str(t).strip() for t in caveat_tasks if str(t).strip())
                elif isinstance(caveat_tasks, str):
                    allowed_tasks.extend(t.strip() for t in caveat_tasks.split(",") if t.strip())

            caveat_task = caveats.get("task_id")
            if caveat_task is not None:
                clean_single = str(caveat_task).strip()
                if clean_single and clean_single not in allowed_tasks:
                    allowed_tasks.append(clean_single)

            clean_expected = expected_task_id.strip()
            if not any(hmac.compare_digest(clean_expected, allowed) for allowed in allowed_tasks):
                return False, f"Caveat task_id mismatch: task '{expected_task_id}' not in authorized {allowed_tasks}"

            # 3. Check expiry caveat
            now = current_time if current_time is not None else time.time()
            expiry = caveats.get("expiry", 0)
            if not isinstance(expiry, (int, float)) or expiry <= now:
                return False, f"Macaroon expired at {expiry} (current: {int(now)})"

            # 4. Check max_gflops caveat
            max_gflops = caveats.get("max_gflops", 0.0)
            if not isinstance(max_gflops, (int, float)) or max_gflops < required_gflops:
                return False, f"Insufficient GFLOP allowance: {max_gflops} < {required_gflops}"

            return True, "OK"
        except Exception as ex:
            return False, f"Malformed macaroon token: {ex}"

    def create_hodl_invoice(
        self,
        task_id_or_request: Union[str, CreateHodlInvoiceRequest] = "",
        total_compute_sats: Optional[int] = None,
        platform_fee_percent: float = 0.02,
        worker_lightning_address: str = "worker@testnode.local",
        payment_hash: Optional[str] = None,
        expiry_seconds: int = 3600,
        max_gflops: float = 100.0,
        task_ids: Optional[List[str]] = None,
        job_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MockInvoiceResponse:
        """
        Creates a HODL invoice in UNPAID status.
        Supports both Pydantic request object and direct keyword arguments,
        with multi-task batch escrow support.
        """
        resolved_task_ids: Optional[List[str]] = task_ids or kwargs.get("task_ids")
        resolved_job_id: Optional[str] = job_id or kwargs.get("job_id")

        if isinstance(task_id_or_request, CreateHodlInvoiceRequest):
            req = task_id_or_request
            task_id = req.task_id or ""
            if resolved_task_ids is None and hasattr(req, "task_ids"):
                resolved_task_ids = req.task_ids
            if resolved_job_id is None and hasattr(req, "job_id"):
                resolved_job_id = req.job_id
            compute_sats = req.total_compute_sats
            fee_pct = req.platform_fee_percent
            worker_ln = req.worker_lightning_address
            p_hash = req.payment_hash
        else:
            task_id = str(task_id_or_request or kwargs.get("task_id", ""))
            compute_sats = total_compute_sats if total_compute_sats is not None else kwargs.get("total_compute_sats", 100)
            fee_pct = kwargs.get("platform_fee_percent", platform_fee_percent)
            worker_ln = kwargs.get("worker_lightning_address", worker_lightning_address)
            p_hash = payment_hash if payment_hash is not None else kwargs.get("payment_hash")

        if not task_id and resolved_task_ids:
            task_id = resolved_task_ids[0]

        # Resolve payment_hash & preimage
        if p_hash:
            final_hash = p_hash.strip().lower()
            preimage = None
        else:
            preimage_bytes = secrets.token_bytes(32)
            preimage = preimage_bytes.hex()
            final_hash = hashlib.sha256(preimage_bytes).hexdigest()

        with self._lock:
            # Reject duplicate task ID
            if task_id and task_id in self._task_index:
                raise ValueError(f"Task ID already registered: '{task_id}'")
            if resolved_task_ids:
                for tid in resolved_task_ids:
                    if tid in self._task_index:
                        raise ValueError(f"Task ID already registered: '{tid}'")

            # Reject duplicate payment hash
            if final_hash in self._invoices:
                raise ValueError(f"Duplicate payment hash '{final_hash}' already exists")

            platform_fee_sats = int(round(compute_sats * fee_pct))
            worker_payout_sats = compute_sats - platform_fee_sats
            now = time.time()
            expires_at = int(now + expiry_seconds)

            macaroon_str = self._mint_macaroon(
                payment_hash=final_hash,
                task_id=task_id,
                expires_at=expires_at,
                max_gflops=max_gflops,
                locked_sats=compute_sats,
                task_ids=resolved_task_ids,
                job_id=resolved_job_id,
            )

            payment_request = f"lnbc{compute_sats}000n1p{final_hash[:20]}mockhodl"

            record = InvoiceRecord(
                task_id=task_id,
                payment_hash=final_hash,
                total_compute_sats=compute_sats,
                platform_fee_percent=fee_pct,
                platform_fee_sats=platform_fee_sats,
                worker_payout_sats=worker_payout_sats,
                worker_lightning_address=worker_ln,
                payment_request=payment_request,
                macaroon=macaroon_str,
                status=InvoiceStatus.UNPAID,
                preimage=preimage,
                created_at=now,
                expires_at=float(expires_at),
                max_gflops=max_gflops,
                task_ids=resolved_task_ids,
                job_id=resolved_job_id,
            )

            self._invoices[final_hash] = record
            if task_id:
                self._task_index[task_id] = final_hash
            if resolved_task_ids:
                for tid in resolved_task_ids:
                    self._task_index[tid] = final_hash

        return MockInvoiceResponse(
            payment_hash=final_hash,
            payment_request=payment_request,
            platform_fee_sats=platform_fee_sats,
            worker_payout_sats=worker_payout_sats,
            macaroon=macaroon_str,
            preimage=preimage,
        )

    def hold_invoice(self, payment_hash: str) -> bool:
        """
        Transitions invoice status from UNPAID -> HELD.
        Simulates consumer locking funds in Lightning routing.
        """
        clean_hash = str(payment_hash).strip().lower()
        with self._lock:
            if clean_hash not in self._invoices:
                raise KeyError(f"Payment hash '{payment_hash}' not found")
            record = self._invoices[clean_hash]
            if record.status != InvoiceStatus.UNPAID:
                raise ValueError(f"Cannot hold invoice in status '{record.status.value}'; must be UNPAID")
            record.status = InvoiceStatus.HELD
            return True

    def is_held(self, payment_hash: str) -> bool:
        """Returns True if and only if invoice status is HELD."""
        clean_hash = str(payment_hash).strip().lower()
        with self._lock:
            record = self._invoices.get(clean_hash)
            return record is not None and record.status == InvoiceStatus.HELD

    def get_status(self, payment_hash: str) -> Optional[str]:
        """Returns the current invoice status string ('UNPAID', 'HELD', 'SETTLED', 'CANCELLED')."""
        clean_hash = str(payment_hash).strip().lower()
        with self._lock:
            self.query_count += 1
            record = self._invoices.get(clean_hash)
            return record.status.value if record else None

    def get_invoice_status(self, payment_hash: str) -> Optional[str]:
        """Alias for get_status."""
        return self.get_status(payment_hash)

    def settle_invoice(
        self,
        hash_or_request: Union[str, SettlePaymentRequest],
        preimage: Optional[str] = None,
    ) -> bool:
        """
        Settles HODL payment upon revelation of the secret preimage.
        Transitions invoice status from HELD -> SETTLED.
        Supports both settle_invoice(hash, preimage=...) and settle_invoice(SettlePaymentRequest).
        """
        with self._lock:
            if isinstance(hash_or_request, SettlePaymentRequest):
                task_id = hash_or_request.task_id
                clean_hash = self._task_index.get(task_id)
                if not clean_hash or clean_hash not in self._invoices:
                    raise KeyError(f"No invoice found for task ID '{task_id}'")
                clean_preimage = hash_or_request.preimage.strip().lower()
            else:
                first_arg = str(hash_or_request).strip().lower()
                if first_arg in self._invoices:
                    clean_hash = first_arg
                elif first_arg in self._task_index:
                    clean_hash = self._task_index[first_arg]
                else:
                    raise KeyError(f"Payment hash or task ID '{hash_or_request}' not found")

                clean_preimage = str(preimage).strip().lower() if preimage is not None else None

            record = self._invoices[clean_hash]
            if record.status != InvoiceStatus.HELD:
                raise ValueError(f"Cannot settle invoice in status '{record.status.value}'; must be HELD")

            if clean_preimage is None:
                if record.preimage:
                    clean_preimage = record.preimage
                else:
                    raise ValueError("Preimage is required to settle invoice")

            try:
                preimage_bytes = bytes.fromhex(clean_preimage)
            except Exception:
                raise ValueError("Preimage is not a valid hexadecimal string")

            computed_hash = hashlib.sha256(preimage_bytes).hexdigest()
            if computed_hash != record.payment_hash:
                raise ValueError(f"Preimage hash {computed_hash} does not match invoice hash {record.payment_hash}")

            record.preimage = clean_preimage
            record.status = InvoiceStatus.SETTLED
            return True

    def cancel_invoice(self, payment_hash: str) -> bool:
        """Cancels an UNPAID or HELD invoice."""
        clean_hash = str(payment_hash).strip().lower()
        with self._lock:
            if clean_hash not in self._invoices:
                raise KeyError(f"Payment hash '{payment_hash}' not found")
            record = self._invoices[clean_hash]
            if record.status == InvoiceStatus.SETTLED:
                raise ValueError("Cannot cancel an already SETTLED invoice")
            record.status = InvoiceStatus.CANCELLED
            return True

    def reset(self) -> None:
        """Clears all stored invoices for test isolation."""
        with self._lock:
            self._invoices.clear()
            self._task_index.clear()
            self.query_count = 0

    def clear(self) -> None:
        """Alias for reset."""
        self.reset()
