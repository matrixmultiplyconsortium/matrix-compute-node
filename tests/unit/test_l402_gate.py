"""Unit tests for L402PreAllocationGate."""

import base64
import concurrent.futures
import hashlib
import hmac
import json
import time
from typing import List
import pytest

from src.auth.l402_gate import L402PreAllocationGate
from src.test_doubles.mock_settlement import MockSettlementService


def mint_test_macaroon(
    key: str,
    task_id: str,
    expiry: int,
    max_gflops: float,
    locked_sats: int = 100,
    payment_hash: str = "default_test_payment_hash",
) -> str:
    caveats = {
        "task_id": task_id,
        "expiry": expiry,
        "max_gflops": max_gflops,
        "locked_sats": locked_sats,
        "payment_hash": payment_hash,
    }
    caveat_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(key.encode("utf-8"), caveat_str.encode("utf-8"), hashlib.sha256).hexdigest()
    payload = {"caveats": caveats, "signature": sig}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


@pytest.fixture
def vkey():
    return "unit-test-secret-verification-key"


@pytest.fixture
def mock_settlement(vkey):
    return MockSettlementService(verification_key=vkey)


@pytest.fixture
def gate(mock_settlement, vkey):
    return L402PreAllocationGate(
        settlement_client_or_url=mock_settlement,
        verification_key=vkey,
        min_price_rate=0.0005,
    )


def test_parse_auth_header_valid(gate):
    mac, ph = gate.parse_auth_header("L402 macaroon_token:payment_hash_123")
    assert mac == "macaroon_token"
    assert ph == "payment_hash_123"

    mac2, ph2 = gate.parse_auth_header("LSAT token2:hash2")
    assert mac2 == "token2"
    assert ph2 == "hash2"


def test_parse_auth_header_malformed(gate):
    assert gate.parse_auth_header("") == (None, None)
    assert gate.parse_auth_header("Bearer some_jwt") == (None, None)
    assert gate.parse_auth_header("L402 just_token_no_hash") == (None, None)
    assert gate.parse_auth_header("L402 :missing_macaroon") == (None, None)
    assert gate.parse_auth_header("L402 macaroon:") == (None, None)


def test_gate_clears_valid_held_invoice(gate, mock_settlement, vkey):
    task_id = "task_pass_001"
    inv = mock_settlement.create_hodl_invoice(task_id, total_compute_sats=100)
    payment_hash = inv.payment_hash
    mock_settlement.hold_invoice(payment_hash)

    macaroon = mint_test_macaroon(
        vkey,
        task_id=task_id,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        locked_sats=100,
        payment_hash=payment_hash,
    )
    auth_header = f"L402 {macaroon}:{payment_hash}"

    ok, err, code = gate.verify_token(auth_header, task_id, m=64, k=64, n=64)
    assert ok is True
    assert err is None
    assert code is None


def test_gate_rejects_unpaid_invoice(gate, mock_settlement, vkey):
    task_id = "task_unpaid"
    inv = mock_settlement.create_hodl_invoice(task_id, total_compute_sats=100)
    # Status remains UNPAID

    macaroon = mint_test_macaroon(
        vkey,
        task_id=task_id,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        locked_sats=100,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    ok, err, code = gate.verify_token(auth_header, task_id, m=10, k=10, n=10)
    assert ok is False
    assert code == "FAILED_PRECONDITION"
    assert "UNPAID" in err


def test_gate_rejects_invalid_signature(gate, mock_settlement, vkey):
    task_id = "task_bad_sig"
    inv = mock_settlement.create_hodl_invoice(task_id)
    mock_settlement.hold_invoice(inv.payment_hash)

    bad_macaroon = mint_test_macaroon(
        "wrong-key",
        task_id=task_id,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {bad_macaroon}:{inv.payment_hash}"

    ok, err, code = gate.verify_token(auth_header, task_id, m=10, k=10, n=10)
    assert ok is False
    assert code == "UNAUTHENTICATED"


def test_gate_rejects_mismatched_task_id(gate, mock_settlement, vkey):
    inv = mock_settlement.create_hodl_invoice("task_alpha")
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_test_macaroon(
        vkey,
        task_id="task_alpha",
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    ok, err, code = gate.verify_token(auth_header, "task_beta", m=10, k=10, n=10)
    assert ok is False
    assert code == "PERMISSION_DENIED"


def test_gate_rejects_expired_macaroon(gate, mock_settlement, vkey):
    inv = mock_settlement.create_hodl_invoice("task_expired")
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_test_macaroon(
        vkey,
        task_id="task_expired",
        expiry=int(time.time()) - 100,
        max_gflops=10.0,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    ok, err, code = gate.verify_token(auth_header, "task_expired", m=10, k=10, n=10)
    assert ok is False
    assert code == "UNAUTHENTICATED"


def test_gate_rejects_gflop_underpayment(gate, mock_settlement, vkey):
    task_id = "task_gflop_dos"
    inv = mock_settlement.create_hodl_invoice(task_id)
    mock_settlement.hold_invoice(inv.payment_hash)

    # Macaroon only allows 0.001 GFLOPs
    macaroon = mint_test_macaroon(
        vkey,
        task_id=task_id,
        expiry=int(time.time()) + 3600,
        max_gflops=0.001,
        locked_sats=100,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    # Request requires 1024^3 * 2 / 1e9 = 2.147 GFLOPs
    ok, err, code = gate.verify_token(auth_header, task_id, m=1024, k=1024, n=1024)
    assert ok is False
    assert code == "PERMISSION_DENIED"
    assert "Insufficient GFLOP" in err


def mint_batch_macaroon(
    key: str,
    task_ids: List[str],
    expiry: int,
    max_gflops: float,
    locked_sats: int = 100,
    payment_hash: str = "default_batch_ph",
    job_id: str = "batch_job_001",
) -> str:
    caveats = {
        "task_id": task_ids[0] if task_ids else "task_default",
        "task_ids": task_ids,
        "job_id": job_id,
        "expiry": expiry,
        "max_gflops": max_gflops,
        "locked_sats": locked_sats,
        "payment_hash": payment_hash,
    }
    caveat_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(key.encode("utf-8"), caveat_str.encode("utf-8"), hashlib.sha256).hexdigest()
    payload = {"caveats": caveats, "signature": sig}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


def test_gate_clears_multi_task_batch_macaroon(gate, mock_settlement, vkey):
    task_ids = ["batch_task_0", "batch_task_1", "batch_task_2"]
    inv = mock_settlement.create_hodl_invoice(task_ids=task_ids, total_compute_sats=300)
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_batch_macaroon(
        vkey,
        task_ids=task_ids,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        locked_sats=300,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    for tid in task_ids:
        ok, err, code = gate.verify_token(auth_header, tid, m=16, k=16, n=16)
        assert ok is True
        if code is not None:
            assert code == "AUTHORIZED"
            assert "cached" in err


def test_gate_rejects_unauthorized_task_in_batch(gate, mock_settlement, vkey):
    task_ids = ["authorized_0", "authorized_1"]
    inv = mock_settlement.create_hodl_invoice(task_ids=task_ids, total_compute_sats=200)
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_batch_macaroon(
        vkey,
        task_ids=task_ids,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        locked_sats=200,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    # Request an unlisted task
    ok, err, code = gate.verify_token(auth_header, "unauthorized_rogue_task", m=16, k=16, n=16)
    assert ok is False
    assert code == "PERMISSION_DENIED"
    assert "not authorize" in err

    # Verify settlement backend was not queried (aborts at Step 3 before remote query)
    assert mock_settlement.query_count == 0
    assert gate.remote_query_count == 0


def test_gate_caching_eliminates_redundant_settlement_queries(gate, mock_settlement, vkey):
    task_ids = [f"tile_{i}" for i in range(5)]
    inv = mock_settlement.create_hodl_invoice(task_ids=task_ids, total_compute_sats=500)
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_batch_macaroon(
        vkey,
        task_ids=task_ids,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        locked_sats=500,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    # Initial state
    assert mock_settlement.query_count == 0
    assert gate.remote_query_count == 0
    assert gate.cache_hit_count == 0

    # Tile 0: Cache miss, queries settlement
    ok, err, code = gate.verify_token(auth_header, "tile_0", m=16, k=16, n=16)
    assert ok is True
    assert mock_settlement.query_count == 1
    assert gate.remote_query_count == 1
    assert gate.cache_hit_count == 0
    assert inv.payment_hash in gate._status_cache

    # Tiles 1..4: In-memory cache hits, ZERO outbound queries
    for tid in task_ids[1:]:
        ok, msg, code = gate.verify_token(auth_header, tid, m=16, k=16, n=16)
        assert ok is True
        assert code == "AUTHORIZED"
        assert "cached" in msg

    # Final telemetry verification: Exactly 1 remote query, 4 cache hits
    assert mock_settlement.query_count == 1
    assert gate.remote_query_count == 1
    assert gate.cache_hit_count == 4


def test_gate_cache_respects_ttl_expiration(vkey, mock_settlement):
    # Configure gate with 100s TTL
    gate = L402PreAllocationGate(
        settlement_client_or_url=mock_settlement,
        verification_key=vkey,
        cache_ttl_seconds=100.0,
    )

    t0 = 1000000.0
    task_ids = ["task_ttl_0", "task_ttl_1"]
    inv = mock_settlement.create_hodl_invoice(task_ids=task_ids, total_compute_sats=200)
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_batch_macaroon(
        vkey,
        task_ids=task_ids,
        expiry=int(t0) + 3600,
        max_gflops=10.0,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    # Query 1 at t0 -> Cache miss, 1 remote query
    ok, _, _ = gate.verify_token(auth_header, "task_ttl_0", 16, 16, 16, current_time=t0)
    assert ok is True
    assert gate.remote_query_count == 1
    assert gate.cache_hit_count == 0

    # Query 2 at t0 + 50s (within TTL) -> Cache hit, 0 remote queries
    ok, _, code = gate.verify_token(auth_header, "task_ttl_1", 16, 16, 16, current_time=t0 + 50.0)
    assert ok is True
    assert code == "AUTHORIZED"
    assert gate.remote_query_count == 1
    assert gate.cache_hit_count == 1

    # Query 3 at t0 + 150s (past TTL 100s) -> Cache miss, re-queries settlement backend
    ok, _, _ = gate.verify_token(auth_header, "task_ttl_0", 16, 16, 16, current_time=t0 + 150.0)
    assert ok is True
    assert gate.remote_query_count == 2
    assert gate.cache_hit_count == 1


def test_gate_cache_bounded_by_macaroon_caveat_expiry(vkey, mock_settlement):
    # Gate TTL is 300s, but macaroon expires in 30s
    gate = L402PreAllocationGate(
        settlement_client_or_url=mock_settlement,
        verification_key=vkey,
        cache_ttl_seconds=300.0,
    )

    t0 = 2000000.0
    task_id = "task_short_expiry"
    inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_test_macaroon(
        vkey,
        task_id=task_id,
        expiry=int(t0) + 30,
        max_gflops=10.0,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    # Query at t0 -> clears and cached with entry_expiry = t0 + 30
    ok, _, _ = gate.verify_token(auth_header, task_id, 16, 16, 16, current_time=t0)
    assert ok is True
    assert gate.remote_query_count == 1

    # Check cache expires_at timestamp is bounded by macaroon expiry
    status, expires_at = gate._verified_cache[inv.payment_hash]
    assert expires_at == pytest.approx(t0 + 30.0)

    # Query at t0 + 35s -> Macaroon itself is expired, rejected with UNAUTHENTICATED
    ok, err, code = gate.verify_token(auth_header, task_id, 16, 16, 16, current_time=t0 + 35.0)
    assert ok is False
    assert code == "UNAUTHENTICATED"
    assert "expired" in err.lower()


def test_gate_unpaid_invoice_not_cached(gate, mock_settlement, vkey):
    task_id = "task_not_cached"
    inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
    # Invoice remains UNPAID

    macaroon = mint_test_macaroon(
        vkey,
        task_id=task_id,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    ok, err, code = gate.verify_token(auth_header, task_id, 16, 16, 16)
    assert ok is False
    assert code == "FAILED_PRECONDITION"

    # Negative caching policy: Non-HELD status must NOT be cached
    assert inv.payment_hash not in gate._verified_cache
    assert inv.payment_hash not in gate._status_cache
    assert gate.cache_hit_count == 0


def test_gate_cache_invalidation_and_clear(gate, mock_settlement, vkey):
    task_id = "task_invalidation"
    inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_test_macaroon(
        vkey,
        task_id=task_id,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    # Verify and cache
    ok, _, _ = gate.verify_token(auth_header, task_id, 16, 16, 16)
    assert ok is True
    assert inv.payment_hash in gate._status_cache

    # Invalidate specific hash
    removed = gate.invalidate(inv.payment_hash)
    assert removed is True
    assert inv.payment_hash not in gate._status_cache

    # Re-verify -> requires new remote query
    ok, _, _ = gate.verify_token(auth_header, task_id, 16, 16, 16)
    assert ok is True
    assert gate.remote_query_count == 2

    # Clear entire cache
    gate.clear_cache()
    assert len(gate._status_cache) == 0
    assert gate.cache_hit_count == 0
    assert gate.remote_query_count == 0


def test_gate_cache_lru_capacity_eviction(vkey, mock_settlement):
    # Create gate with max_cache_size=3
    gate = L402PreAllocationGate(
        settlement_client_or_url=mock_settlement,
        verification_key=vkey,
        max_cache_size=3,
    )

    hashes = []
    for i in range(4):
        tid = f"cap_task_{i}"
        inv = mock_settlement.create_hodl_invoice(task_id=tid, total_compute_sats=100)
        mock_settlement.hold_invoice(inv.payment_hash)
        hashes.append(inv.payment_hash)

        mac = mint_test_macaroon(vkey, task_id=tid, expiry=int(time.time()) + 3600, max_gflops=10.0, payment_hash=inv.payment_hash)
        ok, _, _ = gate.verify_token(f"L402 {mac}:{inv.payment_hash}", tid, 16, 16, 16)
        assert ok is True

    # Cache should be bounded to 3 entries
    assert len(gate._status_cache) == 3
    # The first entry (hashes[0]) should have been evicted by LRU
    assert hashes[0] not in gate._status_cache
    # The latest three entries should be present
    assert hashes[1] in gate._status_cache
    assert hashes[2] in gate._status_cache
    assert hashes[3] in gate._status_cache


def test_gate_concurrent_threads_stress(vkey, mock_settlement):
    gate = L402PreAllocationGate(
        settlement_client_or_url=mock_settlement,
        verification_key=vkey,
        cache_ttl_seconds=300.0,
    )

    task_ids = [f"conc_task_{i}" for i in range(20)]
    inv = mock_settlement.create_hodl_invoice(task_ids=task_ids, total_compute_sats=2000)
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_batch_macaroon(
        vkey,
        task_ids=task_ids,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        locked_sats=2000,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(gate.verify_token, auth_header, tid, 16, 16, 16)
            for tid in task_ids
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # All 20 threads must succeed with zero deadlocks or race conditions
    assert len(results) == 20
    for ok, err, code in results:
        assert ok is True

    # Backend was queried at least once, rest were cache hits
    assert gate.remote_query_count >= 1
    assert gate.cache_hit_count >= 10
    assert (gate.remote_query_count + gate.cache_hit_count) == 20

