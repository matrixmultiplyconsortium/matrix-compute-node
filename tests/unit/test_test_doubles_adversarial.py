"""Adversarial stress tests and boundary challenge suite for MockSettlementService and MockNostrRelay.

Authored by teamwork_preview_challenger_m1_2.
Focus areas:
1. Macaroon caveat tampering, HMAC secret key spoofing, non-string signature handling, expired tokens.
2. Settle payment with wrong preimage, non-hex preimage, unheld invoices, duplicate payment hashes, state races.
3. Nostr relay concurrent event publishing, out-of-order timestamps, malformed tags, query filtering edge cases,
   and content-fallback pricing/vram parsing in matchmaking.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Dict, List
import pytest

from matrix_contracts.schemas.discovery import (
    WorkerQueryRequest,
    WorkerQueryResponse,
)
from matrix_contracts.schemas.settlement import (
    CreateHodlInvoiceRequest,
    SettlePaymentRequest,
)
from src.test_doubles.mock_settlement import (
    InvoiceStatus,
    MockSettlementService,
)
from src.test_doubles.mock_nostr_relay import MockNostrRelay


# ==============================================================================
# SECTION 1: MACAROON TAMPERING & AUTHENTICATION ADVERSARIAL TESTS
# ==============================================================================

class TestMacaroonAdversarial:
    """Stress tests for Macaroon cryptographic verification and caveat enforcement."""

    def test_macaroon_caveat_tampering_max_gflops(self, mock_settlement):
        """ADV-MAC-01: Tampering with max_gflops caveat must invalidate HMAC signature."""
        inv = mock_settlement.create_hodl_invoice(
            task_id="task_tamper_gflops",
            total_compute_sats=100,
            max_gflops=10.0,
        )
        raw_token = inv.macaroon
        payload = json.loads(base64.b64decode(raw_token).decode("utf-8"))

        # Attacker inflates max_gflops allowance from 10.0 to 1000.0 without updating HMAC
        payload["caveats"]["max_gflops"] = 1000.0
        tampered_token = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

        valid, msg = mock_settlement.verify_macaroon(
            macaroon_raw=tampered_token,
            expected_task_id="task_tamper_gflops",
            required_gflops=50.0,
        )
        assert valid is False
        assert "Invalid macaroon HMAC signature" in msg

    def test_macaroon_caveat_tampering_expiry(self, mock_settlement):
        """ADV-MAC-02: Tampering with expiry caveat must invalidate HMAC signature."""
        inv = mock_settlement.create_hodl_invoice(
            task_id="task_tamper_expiry",
            expiry_seconds=60,
        )
        raw_token = inv.macaroon
        payload = json.loads(base64.b64decode(raw_token).decode("utf-8"))

        # Attacker extends expiry far into the future
        payload["caveats"]["expiry"] = int(time.time()) + 9999999
        tampered_token = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

        valid, msg = mock_settlement.verify_macaroon(
            macaroon_raw=tampered_token,
            expected_task_id="task_tamper_expiry",
            required_gflops=1.0,
        )
        assert valid is False
        assert "Invalid macaroon HMAC signature" in msg

    def test_macaroon_caveat_tampering_task_id(self, mock_settlement):
        """ADV-MAC-03: Tampering with task_id caveat must invalidate HMAC signature."""
        inv = mock_settlement.create_hodl_invoice(task_id="task_original")
        raw_token = inv.macaroon
        payload = json.loads(base64.b64decode(raw_token).decode("utf-8"))

        # Attacker swaps task_id to hijack payment for another task
        payload["caveats"]["task_id"] = "task_hijacked"
        tampered_token = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

        valid, msg = mock_settlement.verify_macaroon(
            macaroon_raw=tampered_token,
            expected_task_id="task_hijacked",
            required_gflops=1.0,
        )
        assert valid is False
        assert "Invalid macaroon HMAC signature" in msg

    def test_macaroon_hmac_secret_key_spoofing(self, mock_settlement):
        """ADV-MAC-04: Macaroon signed with an attacker-controlled secret key must be rejected."""
        attacker_key = "attacker-forged-secret-key-67890"
        caveats = {
            "expiry": int(time.time()) + 3600,
            "locked_sats": 500,
            "max_gflops": 100.0,
            "payment_hash": "a" * 64,
            "task_id": "task_spoofed",
        }
        caveat_str = json.dumps(caveats, sort_keys=True)
        forged_sig = hmac.new(
            attacker_key.encode("utf-8"),
            caveat_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        forged_payload = {"caveats": caveats, "signature": forged_sig}
        forged_token = base64.b64encode(json.dumps(forged_payload).encode("utf-8")).decode("utf-8")

        valid, msg = mock_settlement.verify_macaroon(
            macaroon_raw=forged_token,
            expected_task_id="task_spoofed",
            required_gflops=10.0,
        )
        assert valid is False
        assert "Invalid macaroon HMAC signature" in msg

    def test_macaroon_non_string_signature_handling(self, mock_settlement):
        """
        ADV-MAC-05: Non-string signature (e.g. integer, None, dict, list) in JSON payload
        must return (False, msg) gracefully and NOT raise an unhandled TypeError.
        """
        for bad_sig in [12345, None, False, [], {"key": "val"}]:
            payload = {
                "caveats": {
                    "task_id": "task_bad_sig",
                    "expiry": int(time.time()) + 3600,
                    "max_gflops": 10.0,
                },
                "signature": bad_sig,
            }
            raw = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
            try:
                valid, msg = mock_settlement.verify_macaroon(
                    macaroon_raw=raw,
                    expected_task_id="task_bad_sig",
                    required_gflops=1.0,
                )
                assert valid is False
            except TypeError as te:
                pytest.fail(f"verify_macaroon crashed with unhandled TypeError on signature={bad_sig}: {te}")

    def test_macaroon_malformed_token_variations(self, mock_settlement):
        """ADV-MAC-06: Validates rejection of non-base64, non-JSON, and missing-field tokens."""
        invalid_inputs = [
            "!!!NOT_BASE64!!!",
            "",
            base64.b64encode(b"This is not JSON text").decode("utf-8"),
            base64.b64encode(json.dumps({"signature": "a" * 64}).encode("utf-8")).decode("utf-8"),  # missing caveats
            base64.b64encode(json.dumps({"caveats": {}}).encode("utf-8")).decode("utf-8"),           # missing signature
            base64.b64encode(json.dumps([1, 2, 3]).encode("utf-8")).decode("utf-8"),                 # JSON array
        ]
        for token in invalid_inputs:
            valid, msg = mock_settlement.verify_macaroon(
                macaroon_raw=token,
                expected_task_id="task_any",
                required_gflops=1.0,
            )
            assert valid is False
            assert len(msg) > 0

    def test_macaroon_exact_boundary_conditions(self, mock_settlement):
        """ADV-MAC-07: Tests exact boundary conditions for expiry and GFLOPs."""
        fixed_now = 1700000000.0
        # Expiry exactly at current timestamp must be rejected (expired)
        inv_expired = mock_settlement.create_hodl_invoice(
            task_id="task_boundary",
            expiry_seconds=0,
            max_gflops=50.0,
        )
        # Manually verify at the exact expiry time
        payload = json.loads(base64.b64decode(inv_expired.macaroon).decode("utf-8"))
        exact_expiry = payload["caveats"]["expiry"]
        valid, msg = mock_settlement.verify_macaroon(
            macaroon_raw=inv_expired.macaroon,
            expected_task_id="task_boundary",
            required_gflops=10.0,
            current_time=float(exact_expiry),
        )
        assert valid is False
        assert "expired" in msg.lower()

        # Required GFLOPs exactly equal to allowance must be accepted
        valid_exact, msg_exact = mock_settlement.verify_macaroon(
            macaroon_raw=inv_expired.macaroon,
            expected_task_id="task_boundary",
            required_gflops=50.0,
            current_time=float(exact_expiry - 10),
        )
        assert valid_exact is True
        assert msg_exact == "OK"


# ==============================================================================
# SECTION 2: SETTLEMENT SERVICE INVOICE STATE MACHINE & RACE CONDITIONS
# ==============================================================================

class TestSettlementAdversarial:
    """Stress tests for invoice state transitions, preimage verification, and race conditions."""

    def test_settle_invoice_wrong_preimage_hash_retains_held_state(self, mock_settlement):
        """ADV-SET-01: Settling with invalid preimage must fail and leave invoice in HELD state."""
        inv = mock_settlement.create_hodl_invoice(task_id="task_wrong_pre", total_compute_sats=100)
        h = inv.payment_hash
        mock_settlement.hold_invoice(h)

        # 32-byte preimage whose hash differs from payment_hash
        wrong_preimage = "bb" * 32
        with pytest.raises(ValueError, match="does not match invoice hash"):
            mock_settlement.settle_invoice(h, preimage=wrong_preimage)

        # Status must remain HELD, not CANCELLED or corrupted
        assert mock_settlement.get_status(h) == "HELD"
        assert mock_settlement.is_held(h) is True

    def test_settle_invoice_non_hex_preimage_rejected(self, mock_settlement):
        """ADV-SET-02: Preimage containing non-hex characters must raise ValueError."""
        inv = mock_settlement.create_hodl_invoice(task_id="task_non_hex", total_compute_sats=100)
        mock_settlement.hold_invoice(inv.payment_hash)

        non_hex_inputs = ["not_a_hex_string_at_all!", "0123456789abcdefg", "zz" * 32]
        for bad_pre in non_hex_inputs:
            with pytest.raises(ValueError, match="valid hexadecimal string"):
                mock_settlement.settle_invoice(inv.payment_hash, preimage=bad_pre)

    def test_settle_unheld_invoices_all_states(self, mock_settlement):
        """ADV-SET-03: Settle must be rejected when invoice is in UNPAID, SETTLED, or CANCELLED."""
        # Case A: UNPAID
        inv_unpaid = mock_settlement.create_hodl_invoice(task_id="task_unpaid_adv")
        with pytest.raises(ValueError, match="must be HELD"):
            mock_settlement.settle_invoice(inv_unpaid.payment_hash, preimage=inv_unpaid.preimage)

        # Case B: Already SETTLED (double-settle)
        mock_settlement.hold_invoice(inv_unpaid.payment_hash)
        mock_settlement.settle_invoice(inv_unpaid.payment_hash, preimage=inv_unpaid.preimage)
        assert mock_settlement.get_status(inv_unpaid.payment_hash) == "SETTLED"
        with pytest.raises(ValueError, match="must be HELD"):
            mock_settlement.settle_invoice(inv_unpaid.payment_hash, preimage=inv_unpaid.preimage)

        # Case C: CANCELLED
        inv_cancelled = mock_settlement.create_hodl_invoice(task_id="task_cancelled_adv")
        mock_settlement.cancel_invoice(inv_cancelled.payment_hash)
        assert mock_settlement.get_status(inv_cancelled.payment_hash) == "CANCELLED"
        with pytest.raises(ValueError, match="must be HELD"):
            mock_settlement.settle_invoice(inv_cancelled.payment_hash, preimage=inv_cancelled.preimage)

    def test_cancel_already_settled_invoice_rejected(self, mock_settlement):
        """ADV-SET-04: Cannot cancel an already SETTLED invoice."""
        inv = mock_settlement.create_hodl_invoice(task_id="task_settled_cancel")
        mock_settlement.hold_invoice(inv.payment_hash)
        mock_settlement.settle_invoice(inv.payment_hash, preimage=inv.preimage)

        with pytest.raises(ValueError, match="Cannot cancel an already SETTLED invoice"):
            mock_settlement.cancel_invoice(inv.payment_hash)

    def test_hold_non_unpaid_invoice_rejected(self, mock_settlement):
        """ADV-SET-05: Cannot hold an already HELD, SETTLED, or CANCELLED invoice."""
        # Hold already HELD
        inv1 = mock_settlement.create_hodl_invoice(task_id="task_hold_twice")
        mock_settlement.hold_invoice(inv1.payment_hash)
        with pytest.raises(ValueError, match="must be UNPAID"):
            mock_settlement.hold_invoice(inv1.payment_hash)

        # Hold CANCELLED
        inv2 = mock_settlement.create_hodl_invoice(task_id="task_hold_cancelled")
        mock_settlement.cancel_invoice(inv2.payment_hash)
        with pytest.raises(ValueError, match="must be UNPAID"):
            mock_settlement.hold_invoice(inv2.payment_hash)

    def test_duplicate_payment_hash_case_insensitivity(self, mock_settlement):
        """ADV-SET-06: Duplicate payment hash check must be case-insensitive."""
        h_lower = "abcdef0123456789" * 4
        h_upper = h_lower.upper()

        mock_settlement.create_hodl_invoice(task_id="task_case_1", payment_hash=h_lower)
        with pytest.raises(ValueError, match="Duplicate payment hash"):
            mock_settlement.create_hodl_invoice(task_id="task_case_2", payment_hash=h_upper)

    def test_concurrent_invoice_hold_and_settle(self, mock_settlement):
        """
        ADV-SET-07: Concurrently creating, holding, and settling 20 distinct invoices
        must execute cleanly without deadlocks or state inconsistencies.
        """
        num_invoices = 20
        created_invoices = []

        for i in range(num_invoices):
            inv = mock_settlement.create_hodl_invoice(task_id=f"concurrent_task_{i}", total_compute_sats=100)
            created_invoices.append(inv)

        def process_invoice(inv):
            h = inv.payment_hash
            mock_settlement.hold_invoice(h)
            return mock_settlement.settle_invoice(h, preimage=inv.preimage)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_invoice, created_invoices))

        assert all(results)
        for inv in created_invoices:
            assert mock_settlement.get_status(inv.payment_hash) == "SETTLED"


# ==============================================================================
# SECTION 3: NOSTR RELAY ADVERSARIAL & QUERY FILTERING TESTS
# ==============================================================================

class TestNostrRelayAdversarial:
    """Stress tests for Nostr relay event deduplication, tags, querying, and concurrency."""

    def test_relay_concurrent_event_publishing(self, mock_relay):
        """
        ADV-REL-01: Multi-threaded publishing of events for the same (kind, pubkey, d_tag)
        must not raise unhandled KeyError during event unindexing.
        """
        pubkey = "concurrent_worker_pubkey"
        num_threads = 15

        def publish_update(idx: int):
            ev = {
                "kind": 30000,
                "pubkey": pubkey,
                "content": f"update_{idx}",
                "tags": [["d", "slot_0"], ["vram_mb", "24576"], ["sats_per_gflop", "0.0005"]],
                "created_at": 1000 + idx,
            }
            return mock_relay.publish(ev)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(publish_update, range(num_threads)))

        assert all(results)
        # NIP-33 ensures exactly one active announcement per (kind, pubkey, d_tag)
        events = mock_relay.get_events()
        matching = [e for e in events if e.get("pubkey") == pubkey]
        assert len(matching) == 1
        # The latest timestamp (1000 + 14 = 1014) must be the active one
        assert matching[0]["content"] == f"update_{num_threads - 1}"

    def test_relay_out_of_order_timestamps(self, mock_relay):
        """ADV-REL-02: Out-of-order event publishing: older timestamp must be discarded."""
        pubkey = "reorder_worker"
        d_tag = "slot_reorder"

        # 1. Publish newer event (ts=500)
        mock_relay.publish({
            "kind": 30000,
            "pubkey": pubkey,
            "content": "newer_event",
            "tags": [["d", d_tag]],
            "created_at": 500,
        })

        # 2. Publish older event (ts=200)
        mock_relay.publish({
            "kind": 30000,
            "pubkey": pubkey,
            "content": "older_event",
            "tags": [["d", d_tag]],
            "created_at": 200,
        })

        active = mock_relay.get_active_announcement(pubkey, d_tag)
        assert active is not None
        assert active["content"] == "newer_event"
        assert active["created_at"] == 500

    def test_relay_malformed_tags_resilience(self, mock_relay):
        """
        ADV-REL-03: Publishing events with short tags (len < 2) or empty tag lists
        must not crash the relay.
        """
        event_with_short_tags = {
            "kind": 30000,
            "pubkey": "malformed_tag_worker",
            "content": "payload",
            "tags": [
                [],                    # empty tag
                ["single_element"],    # short tag
                ["d", "valid_d_tag"],  # valid tag
                ["vram_mb", "16384"],
            ],
            "created_at": 100,
        }
        assert mock_relay.publish(event_with_short_tags) is True
        active = mock_relay.get_active_announcement("malformed_tag_worker", "valid_d_tag")
        assert active is not None

    def test_relay_query_filtering_zero_matches(self, mock_relay):
        """ADV-REL-04: Query with extreme bounds returns empty list without error."""
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "modest_node",
            "tags": [["d", "n1"], ["vram_mb", "8192"], ["sats_per_gflop", "0.0010"]],
        })

        # Impossible VRAM requirement
        assert len(mock_relay.query(min_vram_mb=1000000)) == 0
        # Impossible low price requirement
        assert len(mock_relay.query(max_price=0.0000001)) == 0
        # Mismatched GPU model
        assert len(mock_relay.query(gpu_model="NONEXISTENT_GPU_MODEL_9999")) == 0

    def test_relay_query_workers_content_fallback_defect(self, mock_relay):
        """
        ADV-REL-05: Tests matchmaking behavior when p2p and lightning tags are present,
        but vram_mb and sats_per_gflop are specified in the JSON content payload.

        DEFECT INVESTIGATION: In MockNostrRelay.query_workers:
        `if not p2p or not lightning:` gates parsing price and vram from content.
        If p2p and lightning tags exist, price and vram are NOT extracted from content!
        """
        content_payload = json.dumps({
            "p2p_multiaddr": "/ip4/127.0.0.1/tcp/50051",
            "lightning_routing_pubkey": "02" + "ab" * 32,
            "pricing": {"sats_per_gflop": 0.0005},
            "hardware": {"vram_mb": 24576, "gpu_model": "RTX 4090"},
        })
        event = {
            "kind": 30000,
            "pubkey": "worker_content_specs",
            "tags": [
                ["d", "worker_slot"],
                ["p2p", "/ip4/127.0.0.1/tcp/50051"],
                ["lightning", "02" + "ab" * 32],
                # Notice: vram_mb and sats_per_gflop tags are intentionally omitted from tags!
            ],
            "content": content_payload,
            "created_at": 100,
        }
        mock_relay.publish(event)

        # Query requires min_vram_mb=16000 and max_price=0.0010
        req = WorkerQueryRequest(
            min_vram_mb=16000,
            max_price_sats_per_gflop=0.0010,
            required_slots=1,
        )
        resp = mock_relay.query_workers(req)

        # If content fallback is properly parsed, this worker must match and have price=0.0005
        assert len(resp.matched_workers) == 1, (
            "Worker with specs in content was not matched by query_workers"
        )
        assert resp.matched_workers[0].sats_per_gflop == 0.0005, (
            f"Expected sats_per_gflop=0.0005, got {resp.matched_workers[0].sats_per_gflop} (content fallback skipped)"
        )

    def test_relay_query_workers_tie_breaking(self, mock_relay):
        """
        ADV-REL-06: Matchmaking must sort by lowest price first,
        tie-breaking by largest VRAM when prices are identical.
        """
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "node_small_vram",
            "tags": [
                ["d", "s1"],
                ["p2p", "/ip4/10.0.0.1/tcp/50051"],
                ["lightning", "02ln1"],
                ["vram_mb", "16384"],
                ["sats_per_gflop", "0.0005"],
            ],
            "created_at": 100,
        })
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "node_large_vram",
            "tags": [
                ["d", "s2"],
                ["p2p", "/ip4/10.0.0.2/tcp/50051"],
                ["lightning", "02ln2"],
                ["vram_mb", "49152"],
                ["sats_per_gflop", "0.0005"],
            ],
            "created_at": 100,
        })

        req = WorkerQueryRequest(
            min_vram_mb=8000,
            max_price_sats_per_gflop=0.0010,
            required_slots=2,
        )
        resp = mock_relay.query_workers(req)
        assert len(resp.matched_workers) == 2
        # Both have price 0.0005; the one with larger VRAM (49152) must rank first
        assert resp.matched_workers[0].worker_pubkey == "node_large_vram"
        assert resp.matched_workers[1].worker_pubkey == "node_small_vram"

    def test_relay_concurrent_publishing_50_threads(self, mock_relay):
        """
        ADV-REL-07: Stress-test concurrent publishing and replacing of Kind 30000 events
        across 50 concurrent threads — verify zero KeyError during unindexing, thread safety,
        and NIP-33 parameterized replaceable state consistency.
        """
        pubkey = "concurrent_worker_50"
        num_threads = 50

        def publish_update(idx: int):
            ev = {
                "kind": 30000,
                "pubkey": pubkey,
                "content": f"threaded_payload_{idx}",
                "tags": [
                    ["d", "slot_50"],
                    ["p2p", f"/ip4/10.0.0.{idx}/tcp/50051"],
                    ["lightning", f"02ln{idx:04d}"],
                    ["vram_mb", str(16384 + idx)],
                    ["sats_per_gflop", f"0.{idx:04d}"],
                ],
                "created_at": 2000 + idx,
            }
            return mock_relay.publish(ev)

        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(publish_update, i) for i in range(num_threads)]
            results = [f.result() for f in futures]

        assert all(results), "All concurrent publish calls must succeed"

        # Exactly 1 event should remain in replaceable storage for this (kind, pubkey, d_tag)
        events = mock_relay.get_events()
        matching = [e for e in events if e.get("pubkey") == pubkey]
        assert len(matching) == 1, f"Expected 1 active event for pubkey, got {len(matching)}"

        # The latest timestamp (2000 + 49 = 2049) must be the active one
        active = mock_relay.get_active_announcement(pubkey, "slot_50")
        assert active is not None
        assert active["created_at"] == 2000 + (num_threads - 1)
        assert active["content"] == f"threaded_payload_{num_threads - 1}"

        # Phase 2: Multi-pubkey concurrent publishing across 50 threads
        def publish_distinct(idx: int):
            pk = f"distinct_pubkey_{idx % 10}"
            ev = {
                "kind": 30000,
                "pubkey": pk,
                "content": f"distinct_payload_{idx}",
                "tags": [["d", "default_slot"], ["vram_mb", "24576"], ["sats_per_gflop", "0.0005"]],
                "created_at": 5000 + idx,
            }
            return mock_relay.publish(ev)

        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(publish_distinct, i) for i in range(num_threads)]
            results = [f.result() for f in futures]

        assert all(results)
        # Exactly 10 distinct pubkeys should exist
        distinct_events = [e for e in mock_relay.get_events() if "distinct_pubkey_" in str(e.get("pubkey"))]
        assert len(distinct_events) == 10

    def test_relay_content_fallback_matrix(self, mock_relay):
        """
        ADV-REL-08: Exhaustively verify tag vs. JSON content fallback matrix:
        - Case A: p2p & lightning only in tags, specs in content
        - Case B: p2p, lightning, gpu_model in tags, pricing & vram in content
        - Case C: vram & price in tags, p2p & lightning in content
        - Case D: non-numeric malformed tag values fall back to content
        - Case E: free tier (0.0 sat/gflop, 0 vram) is preserved without fallback corruption
        - Case F: content as pre-deserialized dict
        - Case G: malformed JSON content with valid tags does not crash
        """
        # Case A: p2p/lightning in tags only, vram/price/gpu in content (ADV-REL-05 verification)
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "node_case_a",
            "tags": [["d", "slot_a"], ["p2p", "/ip4/10.0.0.1/tcp/50051"], ["lightning", "02lna"]],
            "content": json.dumps({
                "hardware": {"vram_mb": 24576, "gpu_model": "RTX 4090"},
                "pricing": {"sats_per_gflop": 0.0005},
            }),
            "created_at": 100,
        })

        # Case B: p2p, lightning, gpu_model in tags; pricing and vram in content
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "node_case_b",
            "tags": [
                ["d", "slot_b"],
                ["p2p", "/ip4/10.0.0.2/tcp/50051"],
                ["lightning", "02lnb"],
                ["gpu_model", "A100-SXM4-80GB"],
            ],
            "content": json.dumps({
                "hardware": {"vram_mb": 81920},
                "pricing": {"sats_per_gflop": 0.0003},
            }),
            "created_at": 100,
        })

        # Case C: vram & price in tags; p2p & lightning in content
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "node_case_c",
            "tags": [["d", "slot_c"], ["vram_mb", "49152"], ["sats_per_gflop", "0.0004"]],
            "content": json.dumps({
                "p2p_multiaddr": "/ip4/10.0.0.3/tcp/50051",
                "lightning_routing_pubkey": "02lnc",
                "hardware": {"gpu_model": "H100"},
            }),
            "created_at": 100,
        })

        # Case D: Malformed non-numeric values in tags fall back to content
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "node_case_d",
            "tags": [
                ["d", "slot_d"],
                ["p2p", "/ip4/10.0.0.4/tcp/50051"],
                ["lightning", "02lnd"],
                ["vram_mb", "NOT_AN_INTEGER"],
                ["sats_per_gflop", "NaN_NOT_FLOAT"],
            ],
            "content": json.dumps({
                "hardware": {"vram_mb": 16384, "gpu_model": "RTX 3090"},
                "pricing": {"sats_per_gflop": 0.0006},
            }),
            "created_at": 100,
        })

        # Case E: Free tier (0.0 sat/gflop) and 0 VRAM
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "node_case_e",
            "tags": [
                ["d", "slot_e"],
                ["p2p", "/ip4/10.0.0.5/tcp/50051"],
                ["lightning", "02lne"],
                ["vram_mb", "0"],
                ["sats_per_gflop", "0.0"],
            ],
            "content": json.dumps({
                "pricing": {"sats_per_gflop": 999.0},  # should NOT override 0.0
            }),
            "created_at": 100,
        })

        # Case F: Content as pre-deserialized dict
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "node_case_f",
            "tags": [["d", "slot_f"]],
            "content": {
                "p2p_multiaddr": "/ip4/10.0.0.6/tcp/50051",
                "lightning_routing_pubkey": "02lnf",
                "hardware": {"vram_mb": 32768, "gpu_model": "V100"},
                "pricing": {"sats_per_gflop": 0.0002},
            },
            "created_at": 100,
        })

        # Case G: Malformed JSON content string with valid tags
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "node_case_g",
            "tags": [
                ["d", "slot_g"],
                ["p2p", "/ip4/10.0.0.7/tcp/50051"],
                ["lightning", "02lng"],
                ["vram_mb", "12288"],
                ["sats_per_gflop", "0.0008"],
            ],
            "content": "This is { not valid json ] at all!",
            "created_at": 100,
        })

        # Matchmaking query matching nodes B, C, A, D, F
        req = WorkerQueryRequest(
            min_vram_mb=16000,
            max_price_sats_per_gflop=0.0007,
            required_slots=10,
        )
        resp = mock_relay.query_workers(req)
        assert resp.status == "SUCCESS"
        pubkeys = [w.worker_pubkey for w in resp.matched_workers]

        # All of A, B, C, D, F meet min 16000 MB and max 0.0007 sats/gflop
        assert "node_case_a" in pubkeys
        assert "node_case_b" in pubkeys
        assert "node_case_c" in pubkeys
        assert "node_case_d" in pubkeys
        assert "node_case_f" in pubkeys

        # Verification of Case E free tier
        free_req = WorkerQueryRequest(
            min_vram_mb=0,
            max_price_sats_per_gflop=0.0,
            required_slots=5,
        )
        free_resp = mock_relay.query_workers(free_req)
        free_pubkeys = [w.worker_pubkey for w in free_resp.matched_workers]
        assert "node_case_e" in free_pubkeys
        node_e = next(w for w in free_resp.matched_workers if w.worker_pubkey == "node_case_e")
        assert node_e.sats_per_gflop == 0.0

        # Verification of Case G with malformed JSON
        g_events = mock_relay.query(min_vram_mb=10000, max_price=0.0010, gpu_model=None)
        g_pubkeys = [e.get("pubkey") for e in g_events]
        assert "node_case_g" in g_pubkeys

    def test_relay_query_edge_cases_and_malformed_inputs(self, mock_relay):
        """
        ADV-REL-09: Challenge relay query and query_workers on extreme boundary conditions:
        - Negative min_vram_mb, zero min_vram_mb, huge min_vram_mb
        - Negative max_price, zero max_price, infinite max_price, None max_price
        - required_slots = 0, required_slots > matching count
        - Events with malformed/irregular tags: short tags len<2, empty tags, non-string tags
        - Kind filtering: non-Kind-30000 events ignored by query and matchmaking
        """
        # Publish regular worker
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "worker_standard",
            "tags": [
                ["d", "std_slot"],
                ["p2p", "/ip4/10.0.0.1/tcp/50051"],
                ["lightning", "02lnstd"],
                ["vram_mb", "24576"],
                ["sats_per_gflop", "0.0005"],
                ["gpu_model", "NVIDIA RTX 4090"],
            ],
            "created_at": 100,
        })

        # Publish non-30000 events (e.g. Kind 1 text note, Kind 30001 other service)
        mock_relay.publish({"kind": 1, "pubkey": "p1", "content": "hello", "created_at": 100})
        mock_relay.publish({
            "kind": 30001,
            "pubkey": "p30001",
            "tags": [["vram_mb", "100000"], ["sats_per_gflop", "0.00001"]],
            "created_at": 100,
        })

        # Publish event with bizarre tags
        mock_relay.publish({
            "kind": 30000,
            "pubkey": "worker_bizarre_tags",
            "tags": [
                [],                      # empty tag
                ["singleton"],           # len 1 tag
                ["d", "bizarre_slot"],
                ["custom_tag", "val1", "val2", "val3"], # len 4 tag
                ["vram_mb", "16384"],
                ["sats_per_gflop", "0.0007"],
            ],
            "created_at": 100,
        })

        # 1. Non-30000 events must be ignored by query and query_workers
        all_kind_30000 = mock_relay.query(min_vram_mb=0)
        assert all(int(e.get("kind", 0)) == 30000 for e in all_kind_30000)
        assert len(all_kind_30000) == 2

        # 2. Negative min_vram_mb acts as no minimum
        neg_vram = mock_relay.query(min_vram_mb=-100)
        assert len(neg_vram) == 2

        # 3. Huge min_vram_mb matches nothing
        huge_vram = mock_relay.query(min_vram_mb=2**31 - 1)
        assert len(huge_vram) == 0

        # 4. Negative max_price matches nothing (prices >= 0.0)
        neg_price = mock_relay.query(max_price=-0.0001)
        assert len(neg_price) == 0

        # 5. Infinite max_price matches everything
        inf_price = mock_relay.query(max_price=float("inf"))
        assert len(inf_price) == 2

        # 6. None max_price matches regardless of price
        none_price = mock_relay.query(max_price=None)
        assert len(none_price) == 2

        # 7. GPU model case-insensitive partial matching
        gpu_match = mock_relay.query(gpu_model="rtx 4090")
        assert len(gpu_match) == 1
        assert gpu_match[0]["pubkey"] == "worker_standard"

        gpu_no_match = mock_relay.query(gpu_model="nonexistent_gpu")
        assert len(gpu_no_match) == 0

        # 8. query_workers with required_slots = 0 returns empty list
        resp_0 = mock_relay.query_workers(
            WorkerQueryRequest(min_vram_mb=0, max_price_sats_per_gflop=1.0, required_slots=0)
        )
        assert resp_0.status == "SUCCESS"
        assert len(resp_0.matched_workers) == 0

        # 9. query_workers with required_slots > candidate count returns all available
        resp_many = mock_relay.query_workers(
            WorkerQueryRequest(min_vram_mb=0, max_price_sats_per_gflop=1.0, required_slots=100)
        )
        assert resp_many.status == "SUCCESS"
        assert len(resp_many.matched_workers) == 2

