"""
Adversarial Stress Test Suite for MockSettlementService.

Authored by teamwork_preview_challenger_m1_iter2_1.
Empirical verification targets:
1. Non-string signature types (None, int, float, list, dict, bool) in verify_macaroon.
   Must return clean (False, ...) tuple with zero uncaught exceptions.
2. Duplicate task_id registration across multiple create_hodl_invoice invocations.
   Must raise ValueError and preserve index integrity.
3. Concurrent multithreaded state transitions (50+ parallel worker threads).
   Contending holds, settles, cancellations, and duplicate creations without deadlocks or race conditions.
4. Boundary caveats, custodial vs trustless settle behaviors, and case-insensitivity.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import secrets
import time
from typing import Any, Dict, List
import pytest

from matrix_contracts.schemas.settlement import (
    CreateHodlInvoiceRequest,
    SettlePaymentRequest,
)
from src.test_doubles.mock_settlement import (
    InvoiceStatus,
    MockSettlementService,
)


# ==============================================================================
# SECTION 1: NON-STRING SIGNATURES & MALFORMED MACAROON TESTING
# ==============================================================================

class TestMacaroonNonStringSignatures:
    """Stress testing verify_macaroon against non-string and malformed signatures."""

    @pytest.mark.parametrize("bad_sig,label", [
        (None, "NoneType"),
        (12345, "positive-integer"),
        (-999, "negative-integer"),
        (0, "zero-integer"),
        (3.14159, "float"),
        (float("inf"), "infinity-float"),
        (True, "boolean-True"),
        (False, "boolean-False"),
        ([], "empty-list"),
        (["a", "b", "c"], "string-list"),
        ([1, 2, 3], "int-list"),
        ({}, "empty-dict"),
        ({"alg": "HS256", "sig": "abc"}, "nested-dict"),
    ])
    def test_verify_macaroon_non_string_signature_rejection(self, mock_settlement, bad_sig, label):
        """
        ADV-CHAL-01: verify_macaroon must reject non-string signatures with (False, ...)
        and must NEVER raise TypeError, AttributeError, or other uncaught exceptions.
        """
        payload = {
            "caveats": {
                "task_id": "task_non_string_sig",
                "expiry": int(time.time()) + 3600,
                "max_gflops": 10.0,
                "payment_hash": "a" * 64,
            },
            "signature": bad_sig,
        }
        raw_token = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

        try:
            valid, msg = mock_settlement.verify_macaroon(
                macaroon_raw=raw_token,
                expected_task_id="task_non_string_sig",
                required_gflops=5.0,
            )
        except Exception as exc:
            pytest.fail(f"verify_macaroon raised unexpected exception for signature type {label}: {type(exc).__name__}: {exc}")

        assert valid is False, f"Expected False for {label} signature, got True"
        assert "Malformed macaroon signature" in msg or "Malformed macaroon token" in msg

    @pytest.mark.parametrize("bad_caveats,label", [
        (None, "NoneType"),
        (12345, "integer"),
        (3.14, "float"),
        (True, "boolean"),
        ([], "list"),
        ("string_caveats", "string"),
    ])
    def test_verify_macaroon_non_dict_caveats_rejection(self, mock_settlement, bad_caveats, label):
        """
        ADV-CHAL-02: verify_macaroon must reject non-dict caveats gracefully.
        """
        payload = {
            "caveats": bad_caveats,
            "signature": "abcdef1234567890" * 4,
        }
        raw_token = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

        valid, msg = mock_settlement.verify_macaroon(
            macaroon_raw=raw_token,
            expected_task_id="task_bad_caveats",
            required_gflops=5.0,
        )
        assert valid is False
        assert "caveats must be a JSON object" in msg or "Malformed macaroon token" in msg

    @pytest.mark.parametrize("malformed_input,label", [
        ("", "empty-string"),
        ("   ", "whitespace-only"),
        ("not-valid-base64!@#$%", "invalid-base64"),
        (base64.b64encode(b"not json content").decode("utf-8"), "non-json-base64"),
        (base64.b64encode(b"\x00\x01\x02\x03\xff").decode("utf-8"), "binary-garbage"),
        (base64.b64encode(json.dumps("just a string").encode("utf-8")).decode("utf-8"), "json-string-payload"),
        (base64.b64encode(json.dumps([1, 2, 3]).encode("utf-8")).decode("utf-8"), "json-list-payload"),
        (base64.b64encode(json.dumps({"caveats": {}}).encode("utf-8")).decode("utf-8"), "missing-signature-key"),
        (base64.b64encode(json.dumps({"signature": "abc"}).encode("utf-8")).decode("utf-8"), "missing-caveats-key"),
    ])
    def test_verify_macaroon_malformed_tokens_rejection(self, mock_settlement, malformed_input, label):
        """
        ADV-CHAL-03: verify_macaroon must reject malformed tokens without uncaught exceptions.
        """
        try:
            valid, msg = mock_settlement.verify_macaroon(
                macaroon_raw=malformed_input,
                expected_task_id="task_any",
                required_gflops=1.0,
            )
        except Exception as exc:
            pytest.fail(f"verify_macaroon raised unexpected exception for {label}: {type(exc).__name__}: {exc}")

        assert valid is False
        assert len(msg) > 0


# ==============================================================================
# SECTION 2: DUPLICATE TASK_ID REGISTRATION TESTING
# ==============================================================================

class TestDuplicateTaskIdRejection:
    """Stress testing create_hodl_invoice against duplicate task_id registration."""

    def test_duplicate_task_id_direct_kwargs(self, mock_settlement):
        """
        ADV-CHAL-04: Multiple calls to create_hodl_invoice with the same task_id
        via direct kwargs must raise ValueError.
        """
        task_id = "task_duplicate_kwargs_001"
        inv1 = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
        assert inv1.payment_hash is not None

        with pytest.raises(ValueError, match=f"Task ID already registered: '{task_id}'"):
            mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=200)

    def test_duplicate_task_id_via_request_schema(self, mock_settlement):
        """
        ADV-CHAL-05: Multiple calls to create_hodl_invoice with the same task_id
        via CreateHodlInvoiceRequest must raise ValueError.
        """
        task_id = "task_duplicate_schema_002"
        req1 = CreateHodlInvoiceRequest(
            task_id=task_id,
            total_compute_sats=150,
            platform_fee_percent=0.02,
            worker_lightning_address="worker@node.local",
        )
        inv1 = mock_settlement.create_hodl_invoice(req1)
        assert inv1.payment_hash is not None

        req2 = CreateHodlInvoiceRequest(
            task_id=task_id,
            total_compute_sats=300,
            platform_fee_percent=0.05,
            worker_lightning_address="worker2@node.local",
        )
        with pytest.raises(ValueError, match=f"Task ID already registered: '{task_id}'"):
            mock_settlement.create_hodl_invoice(req2)

    def test_duplicate_task_id_cross_format(self, mock_settlement):
        """
        ADV-CHAL-06: Duplicate task_id across kwargs and Request schema calls must raise ValueError.
        """
        task_id = "task_cross_format_003"
        # First call via kwargs
        mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)

        # Second call via schema
        req = CreateHodlInvoiceRequest(
            task_id=task_id,
            total_compute_sats=200,
            platform_fee_percent=0.02,
            worker_lightning_address="worker@node.local",
        )
        with pytest.raises(ValueError, match=f"Task ID already registered: '{task_id}'"):
            mock_settlement.create_hodl_invoice(req)

    def test_duplicate_task_id_different_payment_hash(self, mock_settlement):
        """
        ADV-CHAL-07: Even if different payment_hashes are provided, duplicate task_id
        must be rejected before inserting.
        """
        task_id = "task_diff_hash_004"
        h1 = "11" * 32
        h2 = "22" * 32

        mock_settlement.create_hodl_invoice(task_id=task_id, payment_hash=h1)
        with pytest.raises(ValueError, match=f"Task ID already registered: '{task_id}'"):
            mock_settlement.create_hodl_invoice(task_id=task_id, payment_hash=h2)

        # The second payment hash must NOT have been registered
        assert mock_settlement.get_status(h2) is None

    def test_task_id_reusable_after_reset(self, mock_settlement):
        """
        ADV-CHAL-08: After reset(), previously used task_id must be cleanly reusable.
        """
        task_id = "task_reusable_005"
        mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)

        with pytest.raises(ValueError, match="Task ID already registered"):
            mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)

        # Reset service
        mock_settlement.reset()

        # Re-registering must succeed
        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
        assert inv.payment_hash is not None
        assert mock_settlement.get_status(inv.payment_hash) == "UNPAID"


# ==============================================================================
# SECTION 3: CONCURRENT MULTITHREADED STRESS TESTING (50+ THREADS)
# ==============================================================================

class TestConcurrentSettlementStress:
    """Stress testing MockSettlementService under heavy multithreaded contention."""

    def test_concurrent_50_threads_holding_same_invoice(self, mock_settlement):
        """
        ADV-CHAL-09: 50 concurrent threads racing to hold the same UNPAID invoice.
        Exactly 1 thread must succeed; 49 threads must raise ValueError('must be UNPAID').
        Zero uncaught exceptions, zero deadlocks, and final state must be HELD.
        """
        inv = mock_settlement.create_hodl_invoice(task_id="task_race_hold", total_compute_sats=100)
        h = inv.payment_hash

        success_count = 0
        error_count = 0
        errors = []

        def attempt_hold():
            nonlocal success_count, error_count
            try:
                ok = mock_settlement.hold_invoice(h)
                if ok:
                    return "SUCCESS"
            except ValueError as ve:
                return f"VALUE_ERROR: {ve}"
            except Exception as ex:
                return f"CRASH: {type(ex).__name__}: {ex}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(attempt_hold) for _ in range(50)]
            results = [f.result() for f in futures]

        successes = [r for r in results if r == "SUCCESS"]
        expected_errors = [r for r in results if r.startswith("VALUE_ERROR: Cannot hold invoice in status 'HELD'")]
        crashes = [r for r in results if r.startswith("CRASH")]

        assert len(crashes) == 0, f"Encountered race crashes: {crashes}"
        assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"
        assert len(expected_errors) == 49, f"Expected 49 ValueErrors, got {len(expected_errors)}"
        assert mock_settlement.get_status(h) == "HELD"
        assert mock_settlement.is_held(h) is True

    def test_concurrent_50_threads_settling_same_invoice(self, mock_settlement):
        """
        ADV-CHAL-10: 50 concurrent threads racing to settle the same HELD invoice.
        Exactly 1 thread must succeed; 49 threads must raise ValueError('must be HELD').
        Zero uncaught exceptions, zero deadlocks, and final state must be SETTLED.
        """
        inv = mock_settlement.create_hodl_invoice(task_id="task_race_settle", total_compute_sats=100)
        h = inv.payment_hash
        mock_settlement.hold_invoice(h)

        def attempt_settle():
            try:
                ok = mock_settlement.settle_invoice(h, preimage=inv.preimage)
                if ok:
                    return "SUCCESS"
            except ValueError as ve:
                return f"VALUE_ERROR: {ve}"
            except Exception as ex:
                return f"CRASH: {type(ex).__name__}: {ex}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(attempt_settle) for _ in range(50)]
            results = [f.result() for f in futures]

        successes = [r for r in results if r == "SUCCESS"]
        expected_errors = [r for r in results if r.startswith("VALUE_ERROR: Cannot settle invoice in status 'SETTLED'")]
        crashes = [r for r in results if r.startswith("CRASH")]

        assert len(crashes) == 0, f"Encountered race crashes: {crashes}"
        assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"
        assert len(expected_errors) == 49, f"Expected 49 ValueErrors, got {len(expected_errors)}"
        assert mock_settlement.get_status(h) == "SETTLED"
        assert mock_settlement.is_held(h) is False

    def test_concurrent_50_threads_duplicate_task_id_creation(self, mock_settlement):
        """
        ADV-CHAL-11: 50 concurrent threads attempting to register the exact same task_id.
        Exactly 1 thread must succeed; 49 threads must raise ValueError('Task ID already registered').
        Internal index must contain exactly 1 entry.
        """
        task_id = "contended_task_id_50_threads"

        def attempt_create(idx: int):
            try:
                inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100 + idx)
                return ("SUCCESS", inv.payment_hash)
            except ValueError as ve:
                return ("VALUE_ERROR", str(ve))
            except Exception as ex:
                return ("CRASH", f"{type(ex).__name__}: {ex}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(attempt_create, i) for i in range(50)]
            results = [f.result() for f in futures]

        successes = [r for r in results if r[0] == "SUCCESS"]
        expected_errors = [r for r in results if r[0] == "VALUE_ERROR" and "Task ID already registered" in r[1]]
        crashes = [r for r in results if r[0] == "CRASH"]

        assert len(crashes) == 0, f"Encountered crashes: {crashes}"
        assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"
        assert len(expected_errors) == 49, f"Expected 49 ValueErrors, got {len(expected_errors)}"

        # Verify internal index integrity
        winning_hash = successes[0][1]
        assert mock_settlement.get_status(winning_hash) == "UNPAID"

    def test_concurrent_50_threads_independent_invoices_full_lifecycle(self, mock_settlement):
        """
        ADV-CHAL-12: 50 concurrent threads, each managing its own independent invoice
        through the complete lifecycle (CREATE -> HOLD -> VERIFY HELD -> SETTLE -> VERIFY SETTLED).
        Zero race conditions, zero deadlocks, and 100% of invoices must reach SETTLED.
        """
        num_workers = 50

        def lifecycle_worker(idx: int):
            task_id = f"independent_task_{idx:03d}"
            # 1. Create
            inv = mock_settlement.create_hodl_invoice(
                task_id=task_id,
                total_compute_sats=100 + idx,
                max_gflops=50.0,
            )
            h = inv.payment_hash
            assert mock_settlement.get_status(h) == "UNPAID"
            assert mock_settlement.is_held(h) is False

            # 2. Hold
            held = mock_settlement.hold_invoice(h)
            assert held is True
            assert mock_settlement.get_status(h) == "HELD"
            assert mock_settlement.is_held(h) is True

            # 3. Verify Macaroon
            valid, msg = mock_settlement.verify_macaroon(
                macaroon_raw=inv.macaroon,
                expected_task_id=task_id,
                required_gflops=25.0,
            )
            assert valid is True, f"Macaroon verification failed: {msg}"

            # 4. Settle
            settled = mock_settlement.settle_invoice(h, preimage=inv.preimage)
            assert settled is True
            assert mock_settlement.get_status(h) == "SETTLED"
            assert mock_settlement.is_held(h) is False

            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(lifecycle_worker, i) for i in range(num_workers)]
            results = [f.result() for f in futures]

        assert all(results)
        assert len(results) == num_workers

    def test_concurrent_interleaved_holds_and_cancels(self, mock_settlement):
        """
        ADV-CHAL-13: 50 concurrent threads racing with conflicting operations on 25 invoices:
        Thread A tries to HOLD, Thread B tries to CANCEL.
        Each invoice must cleanly transition to either HELD or CANCELLED without deadlock or corrupt state.
        """
        invoices = []
        for i in range(25):
            inv = mock_settlement.create_hodl_invoice(task_id=f"contended_interleave_{i}")
            invoices.append(inv)

        def worker_action(inv, action_type: str):
            h = inv.payment_hash
            try:
                if action_type == "HOLD":
                    return ("HOLD", mock_settlement.hold_invoice(h))
                elif action_type == "CANCEL":
                    return ("CANCEL", mock_settlement.cancel_invoice(h))
            except (ValueError, KeyError) as e:
                return ("REJECTED", str(e))

        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            for inv in invoices:
                tasks.append(executor.submit(worker_action, inv, "HOLD"))
                tasks.append(executor.submit(worker_action, inv, "CANCEL"))

            results = [t.result() for t in tasks]

        # For every invoice, the final status must be a valid terminal/intermediate state: HELD or CANCELLED
        for inv in invoices:
            status = mock_settlement.get_status(inv.payment_hash)
            assert status in ("HELD", "CANCELLED"), f"Invalid status {status} for invoice {inv.payment_hash}"
