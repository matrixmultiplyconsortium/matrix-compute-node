"""Strict Pre-Allocation L402 Gate & DoS Protection for matrix-compute-node."""

from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import hmac
import json
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class L402PreAllocationGate:
    """
    Strict 5-Step Pre-Allocation Gate enforcing L402 Macaroon validation,
    multi-task caveat authorization, GFLOP bounding, and HODL payment lock status
    BEFORE any GPU VRAM or compute buffer is allocated.

    Features thread-safe in-memory caching of verified HELD payment states to
    eliminate redundant settlement queries across multi-tile batches.
    """

    def __init__(
        self,
        settlement_client_or_url: Optional[Any] = None,
        verification_key: str = "settlement-secret-verification-key-12345",
        min_price_rate: float = 0.0005,
        min_task_sats: int = 10,
        cache_ttl_seconds: float = 300.0,
        max_cache_size: int = 10000,
        cache_ttl: Optional[float] = None,
    ) -> None:
        self.settlement_client = settlement_client_or_url
        self.verification_key = verification_key
        self.min_price_rate = float(min_price_rate)
        self.min_task_sats = int(min_task_sats)
        self.cache_ttl_seconds = float(cache_ttl if cache_ttl is not None else cache_ttl_seconds)
        self.max_cache_size = int(max_cache_size)

        self._lock = threading.RLock()
        # payment_hash -> (status, expires_at)
        self._verified_cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()

        # Telemetry / metrics counters
        self.remote_query_count: int = 0
        self.cache_hit_count: int = 0

    @property
    def _status_cache(self) -> OrderedDict[str, Tuple[str, float]]:
        """Backward-compatible alias for _verified_cache."""
        return self._verified_cache

    @property
    def cache_ttl(self) -> float:
        """Alias for cache_ttl_seconds."""
        return self.cache_ttl_seconds

    @property
    def _cache_hits(self) -> int:
        return self.cache_hit_count

    @_cache_hits.setter
    def _cache_hits(self, val: int) -> None:
        self.cache_hit_count = val

    @property
    def _cache_misses(self) -> int:
        return self.remote_query_count

    @_cache_misses.setter
    def _cache_misses(self, val: int) -> None:
        self.remote_query_count = val

    def get_cached_status(self, payment_hash: str, now: Optional[float] = None) -> Optional[str]:
        """Returns cached status if present and unexpired; otherwise evicts and returns None."""
        clean_hash = str(payment_hash).strip().lower()
        current_time = now if now is not None else time.time()
        with self._lock:
            if clean_hash not in self._verified_cache:
                return None
            status, expires_at = self._verified_cache[clean_hash]
            if current_time >= expires_at:
                self._verified_cache.pop(clean_hash, None)
                return None
            self._verified_cache.move_to_end(clean_hash)
            return status

    def set_cached_status(
        self,
        payment_hash: str,
        status: str,
        expiry: Optional[float] = None,
        now: Optional[float] = None,
    ) -> None:
        """Stores verified status in cache with TTL and capacity bounds."""
        clean_hash = str(payment_hash).strip().lower()
        clean_status = str(status).strip().upper()
        if clean_status != "HELD":
            return
        current_time = now if now is not None else time.time()
        entry_expiry = expiry if expiry is not None else (current_time + self.cache_ttl_seconds)
        entry_expiry = min(entry_expiry, current_time + self.cache_ttl_seconds)
        with self._lock:
            if clean_hash in self._verified_cache:
                self._verified_cache.move_to_end(clean_hash)
            self._verified_cache[clean_hash] = (clean_status, entry_expiry)
            if len(self._verified_cache) > self.max_cache_size:
                self._evict_expired_unlocked(current_time)
                while len(self._verified_cache) > self.max_cache_size:
                    self._verified_cache.popitem(last=False)

    def evict_expired(self, now: Optional[float] = None) -> int:
        """Evicts expired cache entries. Returns count of evicted items."""
        current_time = now if now is not None else time.time()
        with self._lock:
            return self._evict_expired_unlocked(current_time)

    def _evict_expired_unlocked(self, current_time: float) -> int:
        expired_keys = [
            k for k, (_, exp) in self._verified_cache.items()
            if current_time >= exp
        ]
        for k in expired_keys:
            self._verified_cache.pop(k, None)
        return len(expired_keys)

    def invalidate(self, payment_hash: str) -> bool:
        """Removes a specific payment hash from cache. Returns True if removed."""
        clean_hash = str(payment_hash).strip().lower()
        with self._lock:
            return self._verified_cache.pop(clean_hash, None) is not None

    def invalidate_cache(self, payment_hash: Optional[str] = None) -> None:
        """Evicts a single payment hash or purges the entire cache."""
        with self._lock:
            if payment_hash is not None:
                self._verified_cache.pop(str(payment_hash).strip().lower(), None)
            else:
                self._verified_cache.clear()

    def clear_cache(self) -> None:
        """Clears all cached verification states and resets counters."""
        with self._lock:
            self._verified_cache.clear()
            self.cache_hit_count = 0
            self.remote_query_count = 0

    def parse_auth_header(self, auth_header: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses 'Authorization: L402 <macaroon_b64>:<payment_hash>' header.
        Returns (macaroon_b64, payment_hash) or (None, None) if malformed.
        """
        if not auth_header or not isinstance(auth_header, str):
            return None, None

        cleaned = auth_header.strip()
        if cleaned.lower().startswith("authorization:"):
            cleaned = cleaned[14:].strip()
        parts = cleaned.split(" ", 1)
        if len(parts) != 2:
            return None, None

        scheme, credentials = parts[0].upper(), parts[1].strip()
        if scheme not in ("L402", "LSAT"):
            return None, None

        # Credentials expected format: <macaroon_b64>:<payment_hash>
        cred_parts = credentials.split(":", 1)
        if len(cred_parts) != 2:
            return None, None

        macaroon_b64, payment_hash = cred_parts[0].strip(), cred_parts[1].strip()
        if not macaroon_b64 or not payment_hash:
            return None, None

        return macaroon_b64, payment_hash

    def _query_settlement_backend(self, payment_hash: str) -> Tuple[Optional[str], Optional[str]]:
        """Queries settlement backend for invoice status. Returns (status, error_message)."""
        if not self.settlement_client:
            return None, "Settlement verification is unconfigured (fail-closed)"

        status = None
        if isinstance(self.settlement_client, str) and (
            self.settlement_client.startswith("http://") or self.settlement_client.startswith("https://")
        ):
            base_url = self.settlement_client.rstrip("/")
            try:
                import httpx
                with httpx.Client(timeout=5.0) as client:
                    # Allow up to 3 attempts (0.2s backoff) if invoice is still UNPAID due to WAN/Lightning gossip latency
                    for attempt in range(3):
                        resp = client.get(f"{base_url}/invoices/{payment_hash}")
                        if resp.status_code == 200:
                            data = resp.json()
                            if isinstance(data, dict):
                                raw_status = data.get("status") or data.get("invoice_status")
                                status = str(raw_status).strip().upper() if raw_status else None
                            elif isinstance(data, str):
                                status = data.strip().upper()
                            else:
                                status = None
                            if status == "HELD" or status in ("SETTLED", "CANCELLED") or attempt == 2:
                                break
                            time.sleep(0.2)
                        else:
                            status = None
                            break
            except Exception as ex:
                logger.warning("HTTP settlement verification query failed: %s", ex)
                return None, f"Settlement HTTP query failed: {ex}"
        elif hasattr(self.settlement_client, "get_status"):
            raw_status = self.settlement_client.get_status(payment_hash)
            status = str(raw_status).strip().upper() if raw_status else None
        elif callable(self.settlement_client):
            raw_status = self.settlement_client(payment_hash)
            status = str(raw_status).strip().upper() if raw_status else None

        return status, None

    def verify_token(
        self,
        auth_header: str,
        task_id: Optional[str] = None,
        m: int = 1,
        k: int = 1,
        n: int = 1,
        current_time: Optional[float] = None,
        x_task_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Executes the 5-Step Pre-Allocation Validation Gate:
        1. Header Extraction: Parses 'Authorization: L402 <macaroon>:<payment_hash>'.
        2. HMAC Verification: Cryptographically validates Macaroon against verification_key.
        3. Task Caveats: Asserts task_id containment, caveat.payment_hash == payment_hash, and caveat.expiry > current_time.
        4. Workload Bounding: Asserts caveat.max_gflops >= (2*M*N*K)/1e9 and payment sufficiency with min_task_sats floor.
        5. Settlement Lock Check: Asserts payment_hash status is actively HELD (via in-memory cache hit or settlement backend).

        Returns:
            (is_valid, error_message, error_code)
        """
        # Step 1: Header Extraction
        macaroon_b64, payment_hash = self.parse_auth_header(auth_header)
        if not macaroon_b64 or not payment_hash:
            return False, "Missing or malformed L402 Authorization header", "UNAUTHENTICATED"

        clean_payment_hash = payment_hash.strip().lower()

        # Step 2: Decode & HMAC Verification
        try:
            raw_bytes = base64.b64decode(macaroon_b64)
            payload = json.loads(raw_bytes.decode("utf-8"))
        except Exception as ex:
            return False, f"Malformed macaroon base64/JSON: {ex}", "UNAUTHENTICATED"

        if not isinstance(payload, dict) or "caveats" not in payload or "signature" not in payload:
            return False, "Malformed macaroon: missing caveats or signature", "UNAUTHENTICATED"

        caveats = payload.get("caveats")
        signature = payload.get("signature")
        if not isinstance(caveats, dict) or not isinstance(signature, str):
            return False, "Malformed macaroon structure", "UNAUTHENTICATED"

        # Verify HMAC-SHA256
        caveat_str = json.dumps(caveats, sort_keys=True)
        expected_sig = hmac.new(
            self.verification_key.encode("utf-8"),
            caveat_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_sig):
            return False, "Invalid macaroon HMAC signature", "UNAUTHENTICATED"

        # Step 3: Multi-Task Caveat Containment Check (Feature F6)
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

        target_task = str(x_task_id or task_id or "").strip()
        if not target_task:
            if len(allowed_tasks) == 1:
                target_task = allowed_tasks[0]
            else:
                return (
                    False,
                    "Unable to determine task_id from L402 token or metadata",
                    "UNAUTHENTICATED",
                )

        if not allowed_tasks:
            return (
                False,
                f"Macaroon caveats do not authorize requested task '{target_task}'",
                "PERMISSION_DENIED",
            )

        task_matched = any(hmac.compare_digest(target_task, allowed) for allowed in allowed_tasks)
        if not task_matched:
            return (
                False,
                f"Macaroon caveats do not authorize requested task '{target_task}'",
                "PERMISSION_DENIED",
            )

        caveat_payment_hash = str(caveats.get("payment_hash", "")).strip().lower()
        if not caveat_payment_hash or not hmac.compare_digest(caveat_payment_hash, clean_payment_hash):
            return (
                False,
                f"Macaroon caveat payment_hash '{caveat_payment_hash}' does not match Authorization payment_hash '{payment_hash}'",
                "PERMISSION_DENIED",
            )

        now = current_time if current_time is not None else time.time()
        expiry = caveats.get("expiry")
        if expiry is None or not isinstance(expiry, (int, float)) or expiry <= now:
            return False, f"Macaroon expired at {expiry} (current time: {int(now)})", "UNAUTHENTICATED"

        # Step 4: Workload Bounding & Rate Enforcement
        requested_gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9
        max_gflops = caveats.get("max_gflops")
        if max_gflops is None or not isinstance(max_gflops, (int, float)) or max_gflops < requested_gflops:
            return (
                False,
                f"Insufficient GFLOP allowance in macaroon: permitted {max_gflops}, requested {requested_gflops:.6f}",
                "PERMISSION_DENIED",
            )

        # Rate check if locked_sats is specified with min_task_sats floor
        if "locked_sats" in caveats:
            locked_sats = caveats["locked_sats"]
            if not isinstance(locked_sats, (int, float)):
                return False, "Malformed locked_sats caveat: must be numeric", "UNAUTHENTICATED"
            computed_sats = math.ceil(requested_gflops * self.min_price_rate)
            required_sats = max(computed_sats, self.min_task_sats)
            if locked_sats < required_sats:
                return (
                    False,
                    f"Underpayment detected: locked {locked_sats} sats < required {required_sats} sats",
                    "FAILED_PRECONDITION",
                )

        # Step 5: Settlement HODL Lock Verification with Gate Caching (Feature F5)
        # 5a. Check in-memory cache under lock
        with self._lock:
            if clean_payment_hash in self._verified_cache:
                cached_status, expires_at = self._verified_cache[clean_payment_hash]
                if now < expires_at and cached_status == "HELD":
                    self.cache_hit_count += 1
                    self._verified_cache.move_to_end(clean_payment_hash)
                    return True, "Token verified and payment held (cached)", "AUTHORIZED"
                elif now >= expires_at:
                    self._verified_cache.pop(clean_payment_hash, None)

        # 5b. Cache miss: query settlement backend (outside lock to prevent blocking)
        if not self.settlement_client:
            return (
                False,
                "Settlement verification is unconfigured (fail-closed)",
                None,
            )

        self.remote_query_count += 1
        status, query_err = self._query_settlement_backend(clean_payment_hash)
        if query_err:
            return False, query_err, "FAILED_PRECONDITION"

        if status != "HELD":
            return (
                False,
                f"Payment hash '{payment_hash}' is in state '{status}', expected 'HELD'",
                "FAILED_PRECONDITION",
            )

        # 5c. Backend returned HELD: compute expires_at and record in cache under lock
        caveat_expiry = float(expiry) if expiry is not None else float("inf")
        ttl_expiry = now + self.cache_ttl_seconds
        entry_expiry = min(caveat_expiry, ttl_expiry)

        with self._lock:
            if clean_payment_hash in self._verified_cache:
                self._verified_cache.move_to_end(clean_payment_hash)
            self._verified_cache[clean_payment_hash] = ("HELD", entry_expiry)

            if len(self._verified_cache) > self.max_cache_size:
                self._evict_expired_unlocked(now)
                while len(self._verified_cache) > self.max_cache_size:
                    self._verified_cache.popitem(last=False)

        return True, None, None

    # Alias for test compatibility
    verify_request = verify_token
