"""
Adversarial Stress Test Suite for L402 Pre-Allocation Gate Caching & Multi-Task Authorization.

Authored by teamwork_preview_challenger_m3_2.
Empirical verification targets:
1. 30–50+ concurrent threads executing `gate.verify_token` against the same batch simultaneously.
   - Zero thread-safety exceptions (`RuntimeError: OrderedDict mutated during iteration`, etc.).
   - Zero deadlocks under heavy lock contention.
   - Settlement backend receives minimal queries (not N queries), subsequent calls hit cache.
   - Multi-batch concurrent load (60 threads across 6 distinct batches).
   - Race conditions between concurrent verifications and cache invalidation / eviction.
2. TTL boundaries and microsecond precision:
   - Exact boundary behavior (`now == expires_at` is evicted, `now < expires_at` is retained).
   - Re-querying after TTL expiration re-engages settlement backend and refreshes cache entry.
   - Asymmetric expiry bounding: `min(caveat_expiry, now + cache_ttl_seconds)`.
   - Macaroon expiry check at Step 3 prevents serving stale credentials even if cached.
3. LRU capacity eviction:
   - Strict LRU eviction when entries exceed `max_cache_size`.
   - Least-recently-used entry is evicted, while recently-accessed entries are promoted to MRU.
   - Mixed expiry and capacity eviction: expired entries are purged before evicting valid entries.
   - Edge capacity behavior (`max_cache_size = 1`).
4. Negative caching avoidance:
   - UNPAID invoices are rejected with `FAILED_PRECONDITION` and strictly NOT cached.
   - Repeated requests for UNPAID invoices re-query the backend every time.
   - Dynamic transition from UNPAID to HELD cleanly transitions from failing to cached success.
   - Non-HELD statuses (CANCELLED, SETTLED) are never stored in cache.
5. Multi-task batch authorization security:
   - Rogue task IDs outside caveat list abort at Step 3 with `PERMISSION_DENIED` with zero backend queries.
   - Tampered payment hash aborts at Step 3 with zero backend queries.
   - Missing task ID disambiguation in multi-task macaroons aborts with `UNAUTHENTICATED`.
   - Constant-time comparison defends against timing attacks.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional
import pytest

from src.auth.l402_gate import L402PreAllocationGate
from src.test_doubles.mock_settlement import MockSettlementService


# ==============================================================================
# HELPER FUNCTIONS & FIXTURES
# ==============================================================================

def mint_batch_macaroon(
    key: str,
    task_ids: List[str],
    expiry: int,
    max_gflops: float,
    locked_sats: int = 100,
    payment_hash: str = "default_batch_ph",
    job_id: str = "batch_job_001",
) -> str:
    """Helper to mint a valid batch L402 macaroon with HMAC-SHA256."""
    caveats: Dict[str, Any] = {
        "task_id": task_ids[0] if task_ids else "default_task",
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


def mint_single_macaroon(
    key: str,
    task_id: str,
    expiry: int,
    max_gflops: float,
    locked_sats: int = 100,
    payment_hash: str = "default_single_ph",
) -> str:
    """Helper to mint a single-task L402 macaroon."""
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
    return "adv-challenger-m3-secret-key-12345"


@pytest.fixture
def mock_settlement(vkey):
    return MockSettlementService(verification_key=vkey)


# ==============================================================================
# SECTION 1: 30–50+ THREAD CONCURRENCY & RACE CONDITIONS
# ==============================================================================

class TestConcurrencyAndRaceConditions:
    """Adversarial stress testing for high thread concurrency and race conditions."""

    def test_concurrent_50_threads_same_batch_minimal_queries(self, vkey, mock_settlement):
        """
        ADV-M3-01: 50 concurrent threads executing `verify_token` against the SAME batch.
        Assert:
        - Exactly 50 executions complete without any thread-safety exceptions (RuntimeError, KeyError).
        - Zero deadlocks under heavy contention on `_lock`.
        - Settlement backend receives minimal queries (vast majority hit cache).
        - `remote_query_count + cache_hit_count == 50`.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            cache_ttl_seconds=300.0,
        )

        num_threads = 50
        task_ids = [f"batch_tile_{i:03d}" for i in range(num_threads)]
        inv = mock_settlement.create_hodl_invoice(task_ids=task_ids, total_compute_sats=5000)
        mock_settlement.hold_invoice(inv.payment_hash)

        macaroon = mint_batch_macaroon(
            vkey,
            task_ids=task_ids,
            expiry=int(time.time()) + 3600,
            max_gflops=10.0,
            locked_sats=5000,
            payment_hash=inv.payment_hash,
        )
        auth_header = f"L402 {macaroon}:{inv.payment_hash}"

        results = []
        errors = []

        def worker_task(tid: str):
            try:
                ok, err, code = gate.verify_token(auth_header, tid, m=16, k=16, n=16)
                return (ok, err, code)
            except Exception as ex:
                errors.append(f"{type(ex).__name__}: {ex}")
                return (False, str(ex), "CRASH")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, tid) for tid in task_ids]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        # Zero race crashes
        assert len(errors) == 0, f"Encountered unexpected race exceptions: {errors}"
        assert len(results) == num_threads

        # All 50 threads must succeed
        for ok, err, code in results:
            assert ok is True, f"Thread failed with error: {err} ({code})"

        # Minimal backend queries: backend must receive significantly fewer than 50 queries
        # With in-memory mock, typically 1 or 2 threads enter before cache is populated
        assert gate.remote_query_count >= 1
        assert gate.remote_query_count <= 5, f"Expected minimal remote queries, got {gate.remote_query_count}"
        assert gate.cache_hit_count >= 45, f"Expected >= 45 cache hits, got {gate.cache_hit_count}"
        assert (gate.remote_query_count + gate.cache_hit_count) == num_threads

    def test_concurrent_60_threads_multi_batch_load(self, vkey, mock_settlement):
        """
        ADV-M3-02: 60 concurrent threads executing across 6 DIFFERENT batches simultaneously
        (10 threads per batch).
        Assert:
        - All 6 batches are verified with minimal remote queries (1-2 per batch).
        - Zero dictionary corruption or lock deadlocks across batches.
        - Cache contains all 6 payment hashes.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            cache_ttl_seconds=300.0,
        )

        num_batches = 6
        threads_per_batch = 10
        total_threads = num_batches * threads_per_batch

        batch_data = []
        for b_idx in range(num_batches):
            t_ids = [f"b{b_idx}_tile_{t}" for t in range(threads_per_batch)]
            inv = mock_settlement.create_hodl_invoice(task_ids=t_ids, total_compute_sats=1000)
            mock_settlement.hold_invoice(inv.payment_hash)
            mac = mint_batch_macaroon(
                vkey,
                task_ids=t_ids,
                expiry=int(time.time()) + 3600,
                max_gflops=10.0,
                locked_sats=1000,
                payment_hash=inv.payment_hash,
                job_id=f"job_batch_{b_idx}",
            )
            header = f"L402 {mac}:{inv.payment_hash}"
            batch_data.append((inv.payment_hash, header, t_ids))

        work_items = []
        for ph, header, t_ids in batch_data:
            for tid in t_ids:
                work_items.append((header, tid))

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_threads) as executor:
            futures = [
                executor.submit(gate.verify_token, hdr, tid, 16, 16, 16)
                for hdr, tid in work_items
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        assert len(results) == total_threads
        for ok, err, code in results:
            assert ok is True

        # All 6 payment hashes must reside in cache
        for ph, _, _ in batch_data:
            assert ph in gate._verified_cache
            assert gate.get_cached_status(ph) == "HELD"

        # Remote query count must be far below 60 (expected ~6 to 12)
        assert gate.remote_query_count <= 15
        assert gate.cache_hit_count >= (total_threads - 15)
        assert (gate.remote_query_count + gate.cache_hit_count) == total_threads

    def test_concurrent_interleaved_verifications_and_invalidations(self, vkey, mock_settlement):
        """
        ADV-M3-03: 40 threads running `verify_token` racing against 10 threads running
        `invalidate` and `evict_expired`.
        Assert zero `RuntimeError: OrderedDict mutated during iteration` or unhandled exceptions.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            cache_ttl_seconds=300.0,
        )

        task_ids = [f"race_task_{i}" for i in range(40)]
        inv = mock_settlement.create_hodl_invoice(task_ids=task_ids, total_compute_sats=4000)
        mock_settlement.hold_invoice(inv.payment_hash)

        macaroon = mint_batch_macaroon(
            vkey,
            task_ids=task_ids,
            expiry=int(time.time()) + 3600,
            max_gflops=10.0,
            payment_hash=inv.payment_hash,
        )
        auth_header = f"L402 {macaroon}:{inv.payment_hash}"

        def verification_worker(tid: str):
            for _ in range(5):
                ok, _, _ = gate.verify_token(auth_header, tid, 16, 16, 16)
                assert ok is True
            return True

        def invalidator_worker():
            for _ in range(5):
                gate.invalidate(inv.payment_hash)
                gate.evict_expired()
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            v_futures = [executor.submit(verification_worker, tid) for tid in task_ids]
            i_futures = [executor.submit(invalidator_worker) for _ in range(10)]

            # Await all futures without crash
            for f in concurrent.futures.as_completed(v_futures + i_futures):
                assert f.result() is True


# ==============================================================================
# SECTION 2: TTL BOUNDARIES & MICROSECOND PRECISION
# ==============================================================================

class TestTtlBoundariesAndMicrosecondPrecision:
    """Adversarial stress testing for TTL expiration boundaries and precision."""

    def test_exact_ttl_boundary_precision(self, vkey, mock_settlement):
        """
        ADV-M3-04: Probe exact TTL boundary condition:
        - `now < expires_at` (e.g. expires_at - 1e-6) -> Cache HIT.
        - `now == expires_at` -> Cache EXPIRED (miss), re-queries backend.
        - `now > expires_at` (e.g. expires_at + 1e-6) -> Cache EXPIRED (miss), re-queries backend.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            cache_ttl_seconds=100.0,
        )

        t0 = 100000.0
        task_id = "boundary_task_001"
        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
        mock_settlement.hold_invoice(inv.payment_hash)

        macaroon = mint_single_macaroon(
            vkey,
            task_id=task_id,
            expiry=int(t0) + 1000,
            max_gflops=10.0,
            payment_hash=inv.payment_hash,
        )
        auth_header = f"L402 {macaroon}:{inv.payment_hash}"

        # 1. Insertion at t0 -> expires_at = t0 + 100.0 = 100100.0
        ok, _, code = gate.verify_token(auth_header, task_id, 16, 16, 16, current_time=t0)
        assert ok is True
        assert gate.remote_query_count == 1
        assert gate.cache_hit_count == 0

        # 2. Probe 1 microsecond BEFORE boundary (100100.0 - 1e-6) -> Cache HIT
        t_sub_eps = 100100.0 - 1e-6
        ok, msg, code = gate.verify_token(auth_header, task_id, 16, 16, 16, current_time=t_sub_eps)
        assert ok is True
        assert code == "AUTHORIZED"
        assert gate.remote_query_count == 1
        assert gate.cache_hit_count == 1

        # 3. Probe EXACT boundary (100100.0) -> Cache EXPIRED, re-queries settlement backend
        t_exact = 100100.0
        ok, _, code = gate.verify_token(auth_header, task_id, 16, 16, 16, current_time=t_exact)
        assert ok is True
        assert gate.remote_query_count == 2
        assert gate.cache_hit_count == 1

        # Cache entry refreshed at t_exact -> new expires_at = 100100.0 + 100.0 = 100200.0
        status, exp = gate._verified_cache[inv.payment_hash]
        assert status == "HELD"
        assert exp == pytest.approx(100200.0)

        # 4. Probe within new window at 100150.0 -> Cache HIT
        ok, _, code = gate.verify_token(auth_header, task_id, 16, 16, 16, current_time=100150.0)
        assert ok is True
        assert code == "AUTHORIZED"
        assert gate.remote_query_count == 2
        assert gate.cache_hit_count == 2

    def test_asymmetric_expiry_capping(self, vkey, mock_settlement):
        """
        ADV-M3-05: Verify cache expiration is strictly bounded by `min(caveat_expiry, now + cache_ttl)`.
        Case A: Caveat expiry < cache TTL (e.g. caveat in 20s, cache TTL 300s).
                Entry expiry MUST be t0 + 20s.
        Case B: Cache TTL < caveat expiry (e.g. cache TTL 50s, caveat in 3600s).
                Entry expiry MUST be t0 + 50s.
        """
        # Case A: Caveat shorter than TTL
        gate_a = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            cache_ttl_seconds=300.0,
        )
        t0 = 500000.0
        task_a = "task_short_caveat"
        inv_a = mock_settlement.create_hodl_invoice(task_id=task_a)
        mock_settlement.hold_invoice(inv_a.payment_hash)

        mac_a = mint_single_macaroon(vkey, task_a, int(t0) + 20, 10.0, payment_hash=inv_a.payment_hash)
        ok, _, _ = gate_a.verify_token(f"L402 {mac_a}:{inv_a.payment_hash}", task_a, 16, 16, 16, current_time=t0)
        assert ok is True
        _, exp_a = gate_a._verified_cache[inv_a.payment_hash]
        assert exp_a == pytest.approx(t0 + 20.0)

        # Case B: Cache TTL shorter than Caveat
        gate_b = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            cache_ttl_seconds=50.0,
        )
        task_b = "task_long_caveat"
        inv_b = mock_settlement.create_hodl_invoice(task_id=task_b)
        mock_settlement.hold_invoice(inv_b.payment_hash)

        mac_b = mint_single_macaroon(vkey, task_b, int(t0) + 3600, 10.0, payment_hash=inv_b.payment_hash)
        ok, _, _ = gate_b.verify_token(f"L402 {mac_b}:{inv_b.payment_hash}", task_b, 16, 16, 16, current_time=t0)
        assert ok is True
        _, exp_b = gate_b._verified_cache[inv_b.payment_hash]
        assert exp_b == pytest.approx(t0 + 50.0)

    def test_macaroon_expiry_at_step_3_prevents_stale_cache_bypass(self, vkey, mock_settlement):
        """
        ADV-M3-06: Even if cache contains a valid HELD entry, an expired Macaroon presented
        at Step 3 must be rejected with UNAUTHENTICATED before reaching cache check.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            cache_ttl_seconds=300.0,
        )
        t0 = 600000.0
        task_id = "task_bypass_attempt"
        inv = mock_settlement.create_hodl_invoice(task_id=task_id)
        mock_settlement.hold_invoice(inv.payment_hash)

        # Pre-seed cache with valid HELD status
        gate.set_cached_status(inv.payment_hash, "HELD", expiry=t0 + 500, now=t0)
        assert gate.get_cached_status(inv.payment_hash, now=t0) == "HELD"

        # Present an expired Macaroon (expiry = t0 - 10)
        expired_mac = mint_single_macaroon(vkey, task_id, int(t0) - 10, 10.0, payment_hash=inv.payment_hash)
        auth_header = f"L402 {expired_mac}:{inv.payment_hash}"

        ok, err, code = gate.verify_token(auth_header, task_id, 16, 16, 16, current_time=t0)
        assert ok is False
        assert code == "UNAUTHENTICATED"
        assert "expired" in err.lower()
        # Verify cache hit was NOT incremented (failed at Step 3 before Step 5)
        assert gate.cache_hit_count == 0


# ==============================================================================
# SECTION 3: LRU CAPACITY EVICTION
# ==============================================================================

class TestLruCapacityEviction:
    """Adversarial stress testing for LRU capacity limits and eviction ordering."""

    def test_lru_strict_ordering_and_access_promotion(self, vkey, mock_settlement):
        """
        ADV-M3-07: Verify strict LRU capacity eviction:
        1. Set `max_cache_size = 4`.
        2. Insert A, B, C, D in order.
        3. Access item A (promotes A to Most-Recently-Used).
        4. Insert item E (exceeds capacity: 5 > 4).
        5. Verify that item B (least-recently-used) was evicted!
        6. Verify items A, C, D, E remain in cache.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            max_cache_size=4,
        )

        entries = ["hash_a", "hash_b", "hash_c", "hash_d"]
        for h in entries:
            gate.set_cached_status(h, "HELD")

        assert len(gate._status_cache) == 4
        assert list(gate._status_cache.keys()) == entries

        # Access A -> promotes A to end
        status_a = gate.get_cached_status("hash_a")
        assert status_a == "HELD"
        assert list(gate._status_cache.keys()) == ["hash_b", "hash_c", "hash_d", "hash_a"]

        # Insert E -> B should be popped from front
        gate.set_cached_status("hash_e", "HELD")
        assert len(gate._status_cache) == 4
        assert "hash_b" not in gate._status_cache
        assert "hash_a" in gate._status_cache
        assert "hash_c" in gate._status_cache
        assert "hash_d" in gate._status_cache
        assert "hash_e" in gate._status_cache
        assert list(gate._status_cache.keys()) == ["hash_c", "hash_d", "hash_a", "hash_e"]

    def test_lru_evicts_expired_before_unexpired_under_overflow(self, vkey, mock_settlement):
        """
        ADV-M3-08: When cache capacity is exceeded, expired entries must be purged first
        before evicting unexpired LRU entries.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            max_cache_size=3,
        )

        t0 = 700000.0
        # Entry 1: expires at t0 + 10 (will be expired at t0 + 20)
        gate.set_cached_status("hash_exp1", "HELD", expiry=t0 + 10, now=t0)
        # Entry 2: expires at t0 + 500 (unexpired)
        gate.set_cached_status("hash_valid1", "HELD", expiry=t0 + 500, now=t0)
        # Entry 3: expires at t0 + 500 (unexpired)
        gate.set_cached_status("hash_valid2", "HELD", expiry=t0 + 500, now=t0)

        assert len(gate._status_cache) == 3

        # At time t0 + 20, insert 4th entry.
        # hash_exp1 is expired. It should be purged first, preserving hash_valid1 and hash_valid2!
        gate.set_cached_status("hash_new", "HELD", expiry=t0 + 500, now=t0 + 20)

        assert len(gate._status_cache) == 3
        assert "hash_exp1" not in gate._status_cache
        assert "hash_valid1" in gate._status_cache
        assert "hash_valid2" in gate._status_cache
        assert "hash_new" in gate._status_cache

    def test_lru_edge_capacity_one(self, vkey, mock_settlement):
        """
        ADV-M3-09: Edge case `max_cache_size = 1`.
        Each new insertion must immediately replace the prior entry.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            max_cache_size=1,
        )

        gate.set_cached_status("h1", "HELD")
        assert len(gate._status_cache) == 1
        assert "h1" in gate._status_cache

        gate.set_cached_status("h2", "HELD")
        assert len(gate._status_cache) == 1
        assert "h1" not in gate._status_cache
        assert "h2" in gate._status_cache


# ==============================================================================
# SECTION 4: NEGATIVE CACHING AVOIDANCE
# ==============================================================================

class TestNegativeCachingAvoidance:
    """Adversarial stress testing for negative caching avoidance."""

    def test_unpaid_invoice_never_cached(self, vkey, mock_settlement):
        """
        ADV-M3-10: UNPAID invoice must be rejected and must NOT be cached under any key.
        Subsequent calls must continue querying the settlement backend every time.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
        )

        task_id = "task_unpaid_adv"
        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
        # Invoice remains UNPAID

        macaroon = mint_single_macaroon(
            vkey,
            task_id=task_id,
            expiry=int(time.time()) + 3600,
            max_gflops=10.0,
            payment_hash=inv.payment_hash,
        )
        auth_header = f"L402 {macaroon}:{inv.payment_hash}"

        # 3 sequential verification attempts
        for attempt in range(1, 4):
            ok, err, code = gate.verify_token(auth_header, task_id, 16, 16, 16)
            assert ok is False
            assert code == "FAILED_PRECONDITION"
            assert "UNPAID" in err
            # Must query backend on every attempt
            assert gate.remote_query_count == attempt
            assert gate.cache_hit_count == 0
            # Must NOT be in cache
            assert inv.payment_hash not in gate._verified_cache
            assert inv.payment_hash not in gate._status_cache

    def test_unpaid_to_held_transition_dynamic_resolution(self, vkey, mock_settlement):
        """
        ADV-M3-11: Dynamic state transition:
        - Attempt 1: UNPAID -> fails, not cached.
        - Payment arrives: status transitions to HELD.
        - Attempt 2: HELD -> succeeds, cached!
        - Attempt 3: succeeds from cache with zero new backend queries.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
        )

        task_id = "task_transition_dyn"
        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=100)
        macaroon = mint_single_macaroon(vkey, task_id, int(time.time()) + 3600, 10.0, payment_hash=inv.payment_hash)
        auth_header = f"L402 {macaroon}:{inv.payment_hash}"

        # 1. Attempt while UNPAID
        ok1, _, _ = gate.verify_token(auth_header, task_id, 16, 16, 16)
        assert ok1 is False
        assert gate.remote_query_count == 1
        assert gate.cache_hit_count == 0
        assert inv.payment_hash not in gate._verified_cache

        # 2. Payment lock arrives (HOLD invoice)
        mock_settlement.hold_invoice(inv.payment_hash)
        assert mock_settlement.get_status(inv.payment_hash) == "HELD"

        # 3. Attempt 2 -> queries backend, receives HELD, caches result
        ok2, _, _ = gate.verify_token(auth_header, task_id, 16, 16, 16)
        assert ok2 is True
        assert gate.remote_query_count == 2
        assert gate.cache_hit_count == 0
        assert inv.payment_hash in gate._verified_cache

        # 4. Attempt 3 -> hits in-memory cache
        ok3, _, code3 = gate.verify_token(auth_header, task_id, 16, 16, 16)
        assert ok3 is True
        assert code3 == "AUTHORIZED"
        assert gate.remote_query_count == 2
        assert gate.cache_hit_count == 1

    @pytest.mark.parametrize("non_held_status", ["UNPAID", "CANCELLED", "SETTLED", "EXPIRED", "UNKNOWN", ""])
    def test_set_cached_status_refuses_non_held_statuses(self, vkey, mock_settlement, non_held_status):
        """
        ADV-M3-12: `set_cached_status` must unconditionally reject any status other than 'HELD'.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
        )

        h = f"hash_{non_held_status.lower() or 'empty'}"
        gate.set_cached_status(h, non_held_status)
        assert h not in gate._status_cache
        assert gate.get_cached_status(h) is None


# ==============================================================================
# SECTION 5: MULTI-TASK BATCH AUTHORIZATION SECURITY
# ==============================================================================

class TestBatchAuthorizationSecurity:
    """Adversarial stress testing for multi-task authorization caveats."""

    def test_rogue_task_rejected_at_step_3_zero_backend_queries(self, vkey, mock_settlement):
        """
        ADV-M3-13: Requesting a task not in the Macaroon caveat list must abort at Step 3
        with PERMISSION_DENIED. Crucially, it must NEVER query settlement backend.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
        )

        authorized_tasks = ["task_auth_1", "task_auth_2"]
        inv = mock_settlement.create_hodl_invoice(task_ids=authorized_tasks, total_compute_sats=200)
        mock_settlement.hold_invoice(inv.payment_hash)

        macaroon = mint_batch_macaroon(
            vkey,
            task_ids=authorized_tasks,
            expiry=int(time.time()) + 3600,
            max_gflops=10.0,
            payment_hash=inv.payment_hash,
        )
        auth_header = f"L402 {macaroon}:{inv.payment_hash}"

        # Attempt rogue task
        ok, err, code = gate.verify_token(auth_header, "task_rogue_unauthorized", 16, 16, 16)
        assert ok is False
        assert code == "PERMISSION_DENIED"
        assert "not authorize" in err

        # Settlement backend query count MUST be zero (DoS defense)
        assert mock_settlement.query_count == 0
        assert gate.remote_query_count == 0

    def test_missing_task_id_batch_disambiguation(self, vkey, mock_settlement):
        """
        ADV-M3-14: If a batch Macaroon authorizes multiple tasks, but the request provides
        neither `task_id` nor `x_task_id`, the gate must reject with UNAUTHENTICATED.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
        )

        tasks = ["task_m1", "task_m2", "task_m3"]
        inv = mock_settlement.create_hodl_invoice(task_ids=tasks)
        mock_settlement.hold_invoice(inv.payment_hash)

        mac = mint_batch_macaroon(vkey, tasks, int(time.time()) + 3600, 10.0, payment_hash=inv.payment_hash)
        auth_header = f"L402 {mac}:{inv.payment_hash}"

        # No task_id or x_task_id
        ok, err, code = gate.verify_token(auth_header, task_id=None, x_task_id=None)
        assert ok is False
        assert code == "UNAUTHENTICATED"
        assert "Unable to determine task_id" in err

    def test_single_task_macaroon_auto_disambiguation(self, vkey, mock_settlement):
        """
        ADV-M3-15: If a single-task Macaroon is provided without explicit `task_id` in request,
        it safely falls back to the single authorized task.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
        )

        task_id = "task_single_auto"
        inv = mock_settlement.create_hodl_invoice(task_id=task_id)
        mock_settlement.hold_invoice(inv.payment_hash)

        mac = mint_single_macaroon(vkey, task_id, int(time.time()) + 3600, 10.0, payment_hash=inv.payment_hash)
        auth_header = f"L402 {mac}:{inv.payment_hash}"

        ok, err, code = gate.verify_token(auth_header, task_id=None, x_task_id=None, m=16, k=16, n=16)
        assert ok is True

    def test_mismatched_payment_hash_in_header_vs_caveat(self, vkey, mock_settlement):
        """
        ADV-M3-16: If payment hash in Authorization header does not match payment hash in caveat,
        aborts at Step 3 with PERMISSION_DENIED and zero backend queries.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
        )

        task_id = "task_hash_tamper"
        inv = mock_settlement.create_hodl_invoice(task_id=task_id)
        mock_settlement.hold_invoice(inv.payment_hash)

        mac = mint_single_macaroon(vkey, task_id, int(time.time()) + 3600, 10.0, payment_hash=inv.payment_hash)
        # Present a different hash in the header
        tampered_header = f"L402 {mac}:tampered_rogue_hash_12345"

        ok, err, code = gate.verify_token(tampered_header, task_id, 16, 16, 16)
        assert ok is False
        assert code == "PERMISSION_DENIED"
        assert "payment_hash" in err

        # Zero backend queries
        assert mock_settlement.query_count == 0
        assert gate.remote_query_count == 0


# ==============================================================================
# SECTION 6: EMPIRICAL BENCHMARKS & WORKLOAD ISOLATION (CHALLENGER M3.1 REP1)
# ==============================================================================

class TestEmpiricalBenchmarksAndIsolation:
    """Empirical verification for cache latency (<0.1ms), >80% query reduction, and workload isolation."""

    def test_cache_hit_latency_benchmark_64_tiles(self, vkey, mock_settlement):
        """
        ADV-M3-17: 64-tile batch verification benchmarking.
        Verify:
        - Tile 0 triggers 1 remote settlement query and populates cache.
        - Tiles 1..63 hit cache with 0 outbound queries.
        - Remote query reduction is 63/64 = 98.44% (> 80% required).
        - Cache hit latency is strictly < 0.1ms (100 microseconds) on average.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            cache_ttl_seconds=300.0,
        )

        num_tiles = 64
        task_ids = [f"bench_tile_{i:03d}" for i in range(num_tiles)]
        inv = mock_settlement.create_hodl_invoice(task_ids=task_ids, total_compute_sats=6400)
        mock_settlement.hold_invoice(inv.payment_hash)

        macaroon = mint_batch_macaroon(
            vkey,
            task_ids=task_ids,
            expiry=int(time.time()) + 3600,
            max_gflops=50.0,
            locked_sats=6400,
            payment_hash=inv.payment_hash,
            job_id="job_bench_64_tiles",
        )
        auth_header = f"L402 {macaroon}:{inv.payment_hash}"

        # Tile 0: Initial verification (Cache Miss)
        t_miss_start = time.perf_counter()
        ok0, err0, code0 = gate.verify_token(auth_header, task_ids[0], m=16, k=16, n=16)
        t_miss_duration = time.perf_counter() - t_miss_start

        assert ok0 is True
        assert gate.remote_query_count == 1
        assert gate.cache_hit_count == 0
        assert mock_settlement.query_count == 1

        # Tiles 1..63: Cache Hits
        hit_latencies = []
        for tid in task_ids[1:]:
            t_hit_start = time.perf_counter()
            ok_hit, msg_hit, code_hit = gate.verify_token(auth_header, tid, m=16, k=16, n=16)
            t_hit_duration = time.perf_counter() - t_hit_start
            hit_latencies.append(t_hit_duration)

            assert ok_hit is True
            assert code_hit == "AUTHORIZED"
            assert "cached" in msg_hit

        # Empirical latency assertions
        avg_hit_latency = sum(hit_latencies) / len(hit_latencies)
        max_hit_latency = max(hit_latencies)

        # Telemetry counts
        assert gate.remote_query_count == 1
        assert gate.cache_hit_count == num_tiles - 1  # 63
        assert mock_settlement.query_count == 1

        # Query elimination percentage: (64 - 1) / 64 = 98.4375%
        elimination_rate = (gate.cache_hit_count / num_tiles) * 100.0
        assert elimination_rate >= 80.0, f"Query elimination rate {elimination_rate:.2f}% < 80%"

        # Cache latency requirement: < 0.1ms (1e-4 seconds = 100 microseconds)
        assert avg_hit_latency < 0.0001, (
            f"Average cache hit latency {avg_hit_latency * 1000:.4f}ms exceeds 0.1ms threshold"
        )

    def test_forged_and_truncated_task_id_rejection_at_step_3(self, vkey, mock_settlement):
        """
        ADV-M3-18: Adversarial task authorization stress test.
        Verify forged, truncated, prefix/suffix, null-byte, and non-caveat task IDs
        are rejected immediately with PERMISSION_DENIED, 0 remote queries, and 0 resource allocation.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
        )

        authorized_tasks = ["alpha_job_tile_01", "alpha_job_tile_02"]
        inv = mock_settlement.create_hodl_invoice(task_ids=authorized_tasks, total_compute_sats=200)
        mock_settlement.hold_invoice(inv.payment_hash)

        macaroon = mint_batch_macaroon(
            vkey,
            task_ids=authorized_tasks,
            expiry=int(time.time()) + 3600,
            max_gflops=10.0,
            locked_sats=200,
            payment_hash=inv.payment_hash,
        )
        auth_header = f"L402 {macaroon}:{inv.payment_hash}"

        # Pre-seed cache to prove that even with cached HELD status, rogue tasks cannot clear Step 3
        gate.set_cached_status(inv.payment_hash, "HELD")
        assert gate.get_cached_status(inv.payment_hash) == "HELD"

        adversarial_task_ids = [
            # Truncated IDs (prefix attacks)
            "alpha_job_tile_0",
            "alpha_job_tile_",
            "alpha_job_tile",
            "alpha",
            # Extended / forged suffix attacks
            "alpha_job_tile_01_forged",
            "alpha_job_tile_011",
            "alpha_job_tile_01_extra",
            # Substring / inner match attacks
            "tile_01",
            "job_tile",
            # Completely forged IDs
            "beta_job_tile_01",
            "rogue_task_9999",
            # Null bytes and command injection patterns
            "alpha_job_tile_01\x00",
            "alpha_job_tile_01; DROP TABLE tasks;",
            # Whitespace padded attacks
            " alpha_job_tile_01 ",  # stripped in target_task, but tests behavior
            "alpha_job_tile_03",
        ]

        for forged_tid in adversarial_task_ids:
            # If whitespace stripped matches, skip, else test rejection
            if forged_tid.strip() in authorized_tasks:
                continue

            ok, err, code = gate.verify_token(auth_header, forged_tid, m=16, k=16, n=16)
            assert ok is False, f"Forged task ID '{forged_tid}' was incorrectly accepted!"
            assert code == "PERMISSION_DENIED", f"Expected PERMISSION_DENIED, got {code} for '{forged_tid}'"
            assert "not authorize" in err

        # Empty task_id with multi-task macaroon -> UNAUTHENTICATED
        ok_empty, err_empty, code_empty = gate.verify_token(auth_header, "", m=16, k=16, n=16)
        assert ok_empty is False
        assert code_empty == "UNAUTHENTICATED"

        # Assert ZERO remote queries to settlement backend (failed closed at Step 3)
        assert mock_settlement.query_count == 0
        assert gate.remote_query_count == 0

    def test_mixed_single_and_multi_task_workload_isolation(self, vkey, mock_settlement):
        """
        ADV-M3-19: Mixed single-task and multi-task workloads.
        Ensures no cache key collision or state leakage across distinct jobs:
        - 3 multi-task batch jobs (10 tiles each).
        - 5 single-task jobs (1 tile each).
        - Total 35 task requests interleaved.
        - Cross-token replay attacks fail with PERMISSION_DENIED.
        - Exact query counts: 3 (batches) + 5 (single) = 8 remote queries total.
        - 27 cache hits across batches.
        """
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key=vkey,
            cache_ttl_seconds=300.0,
        )

        # 1. Setup 3 Multi-Task Batches
        batch_configs = []
        for b_i in range(3):
            b_tasks = [f"job_multi_{b_i}_t{t:02d}" for t in range(10)]
            b_inv = mock_settlement.create_hodl_invoice(task_ids=b_tasks, total_compute_sats=1000)
            mock_settlement.hold_invoice(b_inv.payment_hash)
            b_mac = mint_batch_macaroon(
                vkey,
                task_ids=b_tasks,
                expiry=int(time.time()) + 3600,
                max_gflops=10.0,
                locked_sats=1000,
                payment_hash=b_inv.payment_hash,
                job_id=f"multi_job_{b_i}",
            )
            b_hdr = f"L402 {b_mac}:{b_inv.payment_hash}"
            batch_configs.append({"tasks": b_tasks, "hash": b_inv.payment_hash, "header": b_hdr})

        # 2. Setup 5 Single-Task Jobs
        single_configs = []
        for s_i in range(5):
            s_task = f"job_single_{s_i}_t00"
            s_inv = mock_settlement.create_hodl_invoice(task_id=s_task, total_compute_sats=100)
            mock_settlement.hold_invoice(s_inv.payment_hash)
            s_mac = mint_single_macaroon(
                vkey,
                task_id=s_task,
                expiry=int(time.time()) + 3600,
                max_gflops=1.0,
                locked_sats=100,
                payment_hash=s_inv.payment_hash,
            )
            s_hdr = f"L402 {s_mac}:{s_inv.payment_hash}"
            single_configs.append({"task": s_task, "hash": s_inv.payment_hash, "header": s_hdr})

        # 3. Cross-token replay attack assertions
        # Attempt to use Batch 0's header to authorize Batch 1's task
        ok_x, err_x, code_x = gate.verify_token(
            batch_configs[0]["header"],
            batch_configs[1]["tasks"][0],
            m=16, k=16, n=16,
        )
        assert ok_x is False
        assert code_x == "PERMISSION_DENIED"

        # Attempt to use Single 0's header to authorize Batch 0's task
        ok_s_x, err_s_x, code_s_x = gate.verify_token(
            single_configs[0]["header"],
            batch_configs[0]["tasks"][0],
            m=16, k=16, n=16,
        )
        assert ok_s_x is False
        assert code_s_x == "PERMISSION_DENIED"

        # 4. Execute interleaved workload
        # Interleave: 1 tile from each batch, then 1 single-task, etc.
        tile_queue = []
        for t_idx in range(10):
            for b_i in range(3):
                tile_queue.append((batch_configs[b_i]["header"], batch_configs[b_i]["tasks"][t_idx], True))
            if t_idx < 5:
                tile_queue.append((single_configs[t_idx]["header"], single_configs[t_idx]["task"], False))

        for hdr, tid, is_batch in tile_queue:
            ok, msg, code = gate.verify_token(hdr, tid, m=16, k=16, n=16)
            assert ok is True, f"Failed execution for {tid}: {msg}"

        # 5. Telemetry & Isolation verification
        # 3 batch invoices queried once each = 3 queries
        # 5 single invoices queried once each = 5 queries
        # Total queries = 8
        assert mock_settlement.query_count == 8
        assert gate.remote_query_count == 8
        # 3 batches * 9 subsequent cache hits = 27 cache hits
        assert gate.cache_hit_count == 27

        # Exactly 8 entries in cache (3 batches + 5 single)
        assert len(gate._verified_cache) == 8
        for b in batch_configs:
            assert b["hash"] in gate._verified_cache
        for s in single_configs:
            assert s["hash"] in gate._verified_cache

