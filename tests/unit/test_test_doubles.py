"""Unit tests for hermetic test doubles: MockSettlementService and MockNostrRelay."""

import hashlib
import time
import pytest

from matrix_contracts.schemas.discovery import (
    HardwareSpecs,
    PricingSpecs,
    BondStatus,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
    WorkerQueryResponse,
)
from matrix_contracts.schemas.settlement import (
    CreateHodlInvoiceRequest,
    SettlePaymentRequest,
)
from src.test_doubles.mock_settlement import MockSettlementService
from src.test_doubles.mock_nostr_relay import MockNostrRelay


# ==============================================================================
# MockSettlementService Tests
# ==============================================================================

def test_create_hodl_invoice_trustless(mock_settlement):
    """TC-SET-01: Trustless escrow invoice creation with client-provided payment_hash."""
    task_id = "task_trustless_001"
    preimage = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    p_hash = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()

    req = CreateHodlInvoiceRequest(
        task_id=task_id,
        total_compute_sats=1000,
        platform_fee_percent=0.02,
        worker_lightning_address="worker@lightning.local",
        payment_hash=p_hash,
    )
    resp = mock_settlement.create_hodl_invoice(req)

    assert resp.payment_hash == p_hash
    assert resp["payment_hash"] == p_hash
    assert resp.platform_fee_sats == 20
    assert resp.worker_payout_sats == 980
    assert len(resp.macaroon) > 0
    assert mock_settlement.get_status(p_hash) == "UNPAID"
    assert mock_settlement.is_held(p_hash) is False


def test_create_hodl_invoice_custodial(mock_settlement):
    """TC-SET-02: Custodial escrow invoice creation with server-generated preimage."""
    resp = mock_settlement.create_hodl_invoice(task_id="task_custodial_002", total_compute_sats=500)

    assert len(resp.payment_hash) == 64
    assert resp.preimage is not None
    assert len(resp.preimage) == 64
    # Preimage satisfies sha256(preimage) == payment_hash
    computed = hashlib.sha256(bytes.fromhex(resp.preimage)).hexdigest()
    assert computed == resp.payment_hash


def test_duplicate_payment_hash_rejected(mock_settlement):
    """TC-SET-03: Attempting to register duplicate payment_hash raises ValueError."""
    h = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    mock_settlement.create_hodl_invoice(task_id="task_dup_1", total_compute_sats=100, payment_hash=h)

    with pytest.raises(ValueError, match="Duplicate payment hash"):
        mock_settlement.create_hodl_invoice(task_id="task_dup_2", total_compute_sats=200, payment_hash=h)


def test_invoice_state_transitions(mock_settlement):
    """TC-SET-04: Full lifecycle transitions: UNPAID -> HELD -> SETTLED."""
    inv = mock_settlement.create_hodl_invoice(task_id="task_cycle", total_compute_sats=100)
    h = inv.payment_hash

    # 1. UNPAID
    assert mock_settlement.get_status(h) == "UNPAID"
    assert mock_settlement.is_held(h) is False

    # 2. HELD
    assert mock_settlement.hold_invoice(h) is True
    assert mock_settlement.get_status(h) == "HELD"
    assert mock_settlement.is_held(h) is True

    # 3. SETTLED
    settle_ok = mock_settlement.settle_invoice(h, preimage=inv.preimage)
    assert settle_ok is True
    assert mock_settlement.get_status(h) == "SETTLED"
    assert mock_settlement.is_held(h) is False


def test_settle_unpaid_invoice_fails(mock_settlement):
    """TC-SET-05: Cannot settle an UNPAID invoice directly without being HELD."""
    inv = mock_settlement.create_hodl_invoice(task_id="task_unpaid", total_compute_sats=100)
    with pytest.raises(ValueError, match="must be HELD"):
        mock_settlement.settle_invoice(inv.payment_hash, preimage=inv.preimage)


def test_settle_with_invalid_preimage_fails(mock_settlement):
    """TC-SET-06: Settling with invalid preimage raises ValueError and preserves HELD status."""
    inv = mock_settlement.create_hodl_invoice(task_id="task_bad_preimage", total_compute_sats=100)
    mock_settlement.hold_invoice(inv.payment_hash)

    wrong_preimage = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    with pytest.raises(ValueError, match="Preimage hash .* does not match"):
        mock_settlement.settle_invoice(inv.payment_hash, preimage=wrong_preimage)

    # Status remains HELD
    assert mock_settlement.get_status(inv.payment_hash) == "HELD"


def test_settle_via_request_object(mock_settlement):
    """TC-SET-07: Settling using SettlePaymentRequest schema."""
    task_id = "task_req_settle"
    inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=250)
    mock_settlement.hold_invoice(inv.payment_hash)

    req = SettlePaymentRequest(task_id=task_id, preimage=inv.preimage)
    assert mock_settlement.settle_invoice(req) is True
    assert mock_settlement.get_status(inv.payment_hash) == "SETTLED"


def test_cancel_invoice(mock_settlement):
    """TC-SET-08: Cancelling UNPAID and HELD invoices."""
    inv1 = mock_settlement.create_hodl_invoice(task_id="cancel_1", total_compute_sats=100)
    assert mock_settlement.cancel_invoice(inv1.payment_hash) is True
    assert mock_settlement.get_status(inv1.payment_hash) == "CANCELLED"

    # Cancelled invoice cannot be settled
    with pytest.raises(ValueError):
        mock_settlement.settle_invoice(inv1.payment_hash, preimage=inv1.preimage)


def test_macaroon_minting_and_caveat_verification(mock_settlement):
    """TC-SET-09: Validates L402 Macaroon caveat checking (task_id, expiry, max_gflops)."""
    task_id = "task_macaroon_test"
    inv = mock_settlement.create_hodl_invoice(
        task_id=task_id,
        total_compute_sats=100,
        expiry_seconds=3600,
        max_gflops=10.0,
    )

    # Valid check passes
    valid, msg = mock_settlement.verify_macaroon(
        macaroon_raw=inv.macaroon,
        expected_task_id=task_id,
        required_gflops=5.0,
    )
    assert valid is True
    assert msg == "OK"

    # Mismatched task_id fails
    valid, msg = mock_settlement.verify_macaroon(
        macaroon_raw=inv.macaroon,
        expected_task_id="wrong_task",
        required_gflops=5.0,
    )
    assert valid is False
    assert "task_id mismatch" in msg

    # Insufficient GFLOP allowance fails
    valid, msg = mock_settlement.verify_macaroon(
        macaroon_raw=inv.macaroon,
        expected_task_id=task_id,
        required_gflops=20.0,
    )
    assert valid is False
    assert "Insufficient GFLOP" in msg

    # Expired token fails
    valid, msg = mock_settlement.verify_macaroon(
        macaroon_raw=inv.macaroon,
        expected_task_id=task_id,
        required_gflops=5.0,
        current_time=time.time() + 7200,  # 2 hours in the future
    )
    assert valid is False
    assert "expired" in msg.lower()


# ==============================================================================
# MockNostrRelay Tests
# ==============================================================================

def test_publish_and_get_event(mock_relay):
    """TC-REL-01: Verifies raw event publish and retrieval."""
    event = {
        "kind": 30000,
        "pubkey": "worker_pub_01",
        "content": "worker announce payload",
        "tags": [["d", "node1"], ["vram_mb", "24576"], ["sats_per_gflop", "0.0005"]],
        "created_at": 1000,
    }
    assert mock_relay.publish(event) is True

    events = mock_relay.get_events()
    assert len(events) == 1
    assert events[0]["pubkey"] == "worker_pub_01"


def test_publish_worker_announce_payload(mock_relay, sample_announce):
    """TC-REL-02: Verifies publishing structured WorkerAnnouncePayload."""
    event_id = mock_relay.publish_announce(sample_announce)
    assert len(event_id) == 64

    event = mock_relay.get_event(event_id)
    assert event is not None
    assert event["pubkey"] == sample_announce.worker_pubkey
    assert any(t[0] == "vram_mb" and t[1] == "24576" for t in event["tags"])
    assert any(t[0] == "sats_per_gflop" and t[1] == "0.0005" for t in event["tags"])


def test_parameterized_replaceable_overwrite(mock_relay):
    """TC-REL-03: NIP-33 parameterized replaceable event updates and deduplication."""
    ev1 = {
        "kind": 30000,
        "pubkey": "worker_dup",
        "content": "announcement 1",
        "tags": [["d", "worker_slot_1"]],
        "created_at": 100,
    }
    ev2 = {
        "kind": 30000,
        "pubkey": "worker_dup",
        "content": "announcement 2 (updated)",
        "tags": [["d", "worker_slot_1"]],
        "created_at": 200,
    }

    mock_relay.publish(ev1)
    mock_relay.publish(ev2)

    events = mock_relay.get_events()
    assert len(events) == 1
    assert events[0]["content"] == "announcement 2 (updated)"

    # Outdated announcement is ignored
    ev_old = {
        "kind": 30000,
        "pubkey": "worker_dup",
        "content": "outdated announcement",
        "tags": [["d", "worker_slot_1"]],
        "created_at": 50,
    }
    mock_relay.publish(ev_old)
    events_after = mock_relay.get_events()
    assert len(events_after) == 1
    assert events_after[0]["content"] == "announcement 2 (updated)"


def test_query_filtering(mock_relay):
    """TC-REL-04: Verifies querying events by minimum VRAM and maximum pricing."""
    # Worker 1: 8 GB, 0.0008 sats/GFLOP
    mock_relay.publish({
        "kind": 30000,
        "pubkey": "w1",
        "tags": [["d", "w1"], ["vram_mb", "8192"], ["sats_per_gflop", "0.0008"]],
        "created_at": 100,
    })
    # Worker 2: 24 GB, 0.0005 sats/GFLOP
    mock_relay.publish({
        "kind": 30000,
        "pubkey": "w2",
        "tags": [["d", "w2"], ["vram_mb", "24576"], ["sats_per_gflop", "0.0005"]],
        "created_at": 100,
    })
    # Worker 3: 48 GB, 0.0020 sats/GFLOP
    mock_relay.publish({
        "kind": 30000,
        "pubkey": "w3",
        "tags": [["d", "w3"], ["vram_mb", "49152"], ["sats_per_gflop", "0.0020"]],
        "created_at": 100,
    })

    # Query: min 16GB, max price 0.0010 sats/GFLOP
    results = mock_relay.query(min_vram_mb=16000, max_price=0.0010)
    assert len(results) == 1
    assert results[0]["pubkey"] == "w2"


def test_query_workers_matchmaking(mock_relay):
    """TC-REL-05: Verifies WorkerQueryRequest -> WorkerQueryResponse matchmaking."""
    mock_relay.publish({
        "kind": 30000,
        "pubkey": "cheap_node",
        "tags": [
            ["d", "node1"],
            ["p2p", "/ip4/10.0.0.1/tcp/50051"],
            ["lightning", "02pubkey1"],
            ["vram_mb", "24576"],
            ["sats_per_gflop", "0.0004"],
        ],
        "created_at": 100,
    })
    mock_relay.publish({
        "kind": 30000,
        "pubkey": "expensive_node",
        "tags": [
            ["d", "node2"],
            ["p2p", "/ip4/10.0.0.2/tcp/50051"],
            ["lightning", "02pubkey2"],
            ["vram_mb", "24576"],
            ["sats_per_gflop", "0.0009"],
        ],
        "created_at": 100,
    })

    req = WorkerQueryRequest(
        min_vram_mb=16000,
        max_price_sats_per_gflop=0.0010,
        required_slots=2,
    )
    resp = mock_relay.query_workers(req)

    assert isinstance(resp, WorkerQueryResponse)
    assert resp.status == "SUCCESS"
    assert len(resp.matched_workers) == 2
    # Cheapest is sorted first
    assert resp.matched_workers[0].worker_pubkey == "cheap_node"
    assert resp.matched_workers[0].sats_per_gflop == 0.0004


def test_query_workers_no_match(mock_relay):
    """TC-REL-06: Returns empty list when no candidates meet requirements."""
    req = WorkerQueryRequest(
        min_vram_mb=999999,
        max_price_sats_per_gflop=0.00001,
        required_slots=1,
    )
    resp = mock_relay.query_workers(req)
    assert resp.status == "SUCCESS"
    assert len(resp.matched_workers) == 0


def test_relay_clear(mock_relay):
    """TC-REL-07: Verifies relay store reset."""
    mock_relay.publish({"kind": 30000, "pubkey": "p", "tags": [["d", "d1"]]})
    assert len(mock_relay.get_events()) == 1

    mock_relay.clear()
    assert len(mock_relay.get_events()) == 0
