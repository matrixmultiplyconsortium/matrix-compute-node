"""Strict Pre-Allocation L402 Gate & DoS Protection for matrix-compute-node."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class L402PreAllocationGate:
    """
    Strict 5-Step Pre-Allocation Gate enforcing L402 Macaroon validation,
    task caveat authorization, GFLOP bounding, and HODL payment lock status
    BEFORE any GPU VRAM or compute buffer is allocated.
    """

    def __init__(
        self,
        settlement_client_or_url: Optional[Any] = None,
        verification_key: str = "settlement-secret-verification-key-12345",
        min_price_rate: float = 0.0005,
    ) -> None:
        self.settlement_client = settlement_client_or_url
        self.verification_key = verification_key
        self.min_price_rate = float(min_price_rate)

    def parse_auth_header(self, auth_header: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses 'Authorization: L402 <macaroon_b64>:<payment_hash>' header.
        Returns (macaroon_b64, payment_hash) or (None, None) if malformed.
        """
        if not auth_header or not isinstance(auth_header, str):
            return None, None

        cleaned = auth_header.strip()
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

    def verify_token(
        self,
        auth_header: str,
        task_id: str,
        m: int,
        k: int,
        n: int,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Executes the 5-Step Pre-Allocation Validation Gate:
        1. Header Extraction: Parses 'Authorization: L402 <macaroon>:<payment_hash>'.
        2. HMAC Verification: Cryptographically validates Macaroon against verification_key.
        3. Task Caveats: Asserts caveat.task_id == task_id and caveat.expiry > current_time.
        4. Workload Bounding: Asserts caveat.max_gflops >= (2*M*N*K)/1e9 and payment sufficiency.
        5. Settlement Lock Check: Asserts payment_hash status is actively HELD.

        Returns:
            (is_valid, error_message, error_code)
        """
        # Step 1: Header Extraction
        macaroon_b64, payment_hash = self.parse_auth_header(auth_header)
        if not macaroon_b64 or not payment_hash:
            return False, "Missing or malformed L402 Authorization header", "UNAUTHENTICATED"

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

        # Step 3: Task & Expiry Caveats
        caveat_task_id = caveats.get("task_id")
        if caveat_task_id != task_id:
            return (
                False,
                f"Macaroon caveat task_id '{caveat_task_id}' does not match requested task '{task_id}'",
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

        # Rate check if locked_sats is specified
        locked_sats = caveats.get("locked_sats")
        if locked_sats is not None and isinstance(locked_sats, (int, float)):
            required_sats = math.ceil(requested_gflops * self.min_price_rate)
            if locked_sats < required_sats:
                return (
                    False,
                    f"Underpayment detected: locked {locked_sats} sats < required {required_sats} sats",
                    "FAILED_PRECONDITION",
                )

        # Step 5: Settlement HODL Lock Verification
        if self.settlement_client is not None:
            status = None
            if hasattr(self.settlement_client, "get_status"):
                status = self.settlement_client.get_status(payment_hash)
            elif callable(self.settlement_client):
                status = self.settlement_client(payment_hash)

            if status != "HELD":
                return (
                    False,
                    f"Payment hash '{payment_hash}' is in state '{status}', expected 'HELD'",
                    "FAILED_PRECONDITION",
                )

        return True, None, None
