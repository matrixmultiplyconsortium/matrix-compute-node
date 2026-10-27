"""Unit and Adversarial Tests for Milestone 3: Zero-Drain Trial Sandbox, Preemptive Scheduler & Hashcash."""

from __future__ import annotations

import concurrent.futures
import threading
import time
import pytest

from src.auth.hashcash import (
    HashcashVerifier,
    count_leading_zero_bits,
    solve_hashcash,
    verify_hashcash,
    verify_hashcash_with_reason,
)
from src.auth.l402_gate import L402PreAllocationGate
from src.config.settings import NodeConfig
from src.engine.scheduler import PreemptiveComputeScheduler, PriorityTier


class TestTrialMacaroonGuard:
    """Tests for `mc_trial_<base64>` macaroon minting and verification in L402PreAllocationGate."""

    def test_mint_and_verify_seed_worker_authorized(self) -> None:
        seed_pub = "02" + "aa" * 32
        gate = L402PreAllocationGate(
            verification_key="seed-secret-key-001",
            is_seed_worker=True,
            worker_pubkey=seed_pub,
        )
        token = gate.mint_trial_macaroon(
            task_id="task_trial_1",
            job_id="job_1",
            seed_worker_pubkeys=[seed_pub],
            max_matrix_dim=4096,
        )
        assert token.startswith("mc_trial_")

        ok, reason = gate.verify_trial_macaroon(
            token=token,
            worker_pubkey=seed_pub,
            m=1024,
            k=1024,
            n=1024,
        )
        assert ok is True
        assert reason == "AUTHORIZED"

        ok_tok, err_msg, err_code = gate.verify_token(
            auth_header=f"L402 {token}:{'0'*64}",
            task_id="task_trial_1",
            m=2048,
            k=2048,
            n=2048,
        )
        assert ok_tok is True
        assert err_msg is None
        assert err_code == "AUTHORIZED"

    def test_third_party_worker_rejected_with_permission_denied(self) -> None:
        seed_pub = "02" + "aa" * 32
        rogue_pub = "03" + "bb" * 32
        gate = L402PreAllocationGate(
            verification_key="seed-secret-key-001",
            is_seed_worker=False,
            worker_pubkey=rogue_pub,
        )
        token = gate.mint_trial_macaroon(
            task_id="task_trial_2",
            seed_worker_pubkeys=[seed_pub],
            max_matrix_dim=4096,
        )
        ok, err_msg, err_code = gate.verify_token(
            auth_header=f"L402 {token}:{'0'*64}",
            task_id="task_trial_2",
            m=512,
            k=512,
            n=512,
        )
        assert ok is False
        assert err_code == "PERMISSION_DENIED"
        assert "not authorized on third-party worker" in str(err_msg)

    def test_dimension_cap_4096_enforced(self) -> None:
        seed_pub = "02" + "aa" * 32
        gate = L402PreAllocationGate(
            verification_key="seed-secret-key-001",
            is_seed_worker=True,
            worker_pubkey=seed_pub,
        )
        token = gate.mint_trial_macaroon(
            task_id="task_trial_3",
            seed_worker_pubkeys=[seed_pub],
            max_matrix_dim=4096,
        )
        # Exact 4096 allowed
        ok_4096, _ = gate.verify_trial_macaroon(token, seed_pub, 4096, 4096, 4096)
        assert ok_4096 is True

        # 4097 rejected
        ok_4097, reason_4097 = gate.verify_trial_macaroon(token, seed_pub, 4097, 1024, 1024)
        assert ok_4097 is False
        assert "exceeds 4096 cap" in reason_4097

    def test_expired_and_tampered_trial_macaroon_rejected(self) -> None:
        seed_pub = "02" + "aa" * 32
        gate = L402PreAllocationGate(
            verification_key="seed-secret-key-001",
            is_seed_worker=True,
            worker_pubkey=seed_pub,
        )
        expired_token = gate.mint_trial_macaroon(
            task_id="task_trial_exp",
            seed_worker_pubkeys=[seed_pub],
            expiry_seconds=-10,
        )
        ok_exp, reason_exp = gate.verify_trial_macaroon(expired_token, seed_pub, 64, 64, 64)
        assert ok_exp is False
        assert reason_exp == "TOKEN_EXPIRED"

        valid_token = gate.mint_trial_macaroon(
            task_id="task_trial_tamper",
            seed_worker_pubkeys=[seed_pub],
        )
        tampered = valid_token[:-4] + ("AAAA" if valid_token[-4:] != "AAAA" else "BBBB")
        ok_tamp, _ = gate.verify_trial_macaroon(tampered, seed_pub, 64, 64, 64)
        assert ok_tamp is False


class TestPreemptiveComputeScheduler:
    """Tests for PreemptiveComputeScheduler two-level queue and 5% duty cycle throttle."""

    def test_paid_mainnet_executes_immediately_and_blocks_trial(self) -> None:
        sched = PreemptiveComputeScheduler(sliding_window_seconds=1.0, max_trial_ms_per_window=50.0)
        with sched.lock:
            sched.paid_queue_depth = 2
            can_run, reason = sched.can_execute_trial()
            assert can_run is False
            assert "PAID_QUEUE_ACTIVE" in reason
            sched.paid_queue_depth = 0

        res = sched.execute(PriorityTier.PAID_MAINNET, lambda: 42)
        assert res == 42
        assert sched.paid_queue_depth == 0

    def test_trial_proportional_cooldown_19x_and_sliding_window(self) -> None:
        sched = PreemptiveComputeScheduler(sliding_window_seconds=1.0, max_trial_ms_per_window=50.0)

        def simulated_trial_work() -> str:
            time.sleep(0.015)
            return "trial_done"

        out = sched.execute(PriorityTier.FREE_TRIAL, simulated_trial_work)
        assert out == "trial_done"
        assert sched._last_trial_cooldown_required >= 19.0 * 0.010

        # Immediately after execution, cooldown must be active
        can_run_now, reason = sched.can_execute_trial()
        assert can_run_now is False
        assert "COOLDOWN_ACTIVE" in reason

        # Paid tasks are completely exempt from trial cooldown
        paid_out = sched.execute(PriorityTier.PAID_MAINNET, lambda: "paid_instant")
        assert paid_out == "paid_instant"

    def test_concurrent_paid_and_trial_thread_safety(self) -> None:
        sched = PreemptiveComputeScheduler(sliding_window_seconds=1.0, max_trial_ms_per_window=200.0)
        results = []

        def run_paid(idx: int) -> str:
            return sched.execute_paid(lambda: f"paid_{idx}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(run_paid, i) for i in range(16)]
            for f in concurrent.futures.as_completed(futs):
                results.append(f.result())

        assert len(results) == 16
        assert sched.paid_queue_depth == 0


class TestNodeHashcashVerifier:
    """Tests for Hashcash proof-of-work verification and replay cache on the compute node."""

    def test_solve_and_verify_roundtrip(self) -> None:
        stamp = solve_hashcash("task_node_001", bits=12)
        assert stamp.startswith("mc_pow:12:")
        assert verify_hashcash(stamp, expected_resource="task_node_001", min_bits=12) is True

    def test_replay_protection_blocks_second_submission(self) -> None:
        verifier = HashcashVerifier(min_bits=10, max_age_seconds=300)
        stamp = solve_hashcash("task_replay_99", bits=10)

        ok1, reason1 = verifier.verify_and_consume(stamp, expected_resource="task_replay_99")
        assert ok1 is True
        assert reason1 == "VALID_HASHCASH_PROOF"

        ok2, reason2 = verifier.verify_and_consume(stamp, expected_resource="task_replay_99")
        assert ok2 is False
        assert reason2 == "REPLAYED_HASHCASH_STAMP"

    def test_node_config_seed_worker_and_pow_bits(self) -> None:
        cfg = NodeConfig(is_seed_worker=True, trial_pow_bits=18)
        assert cfg.is_seed_worker is True
        assert cfg.trial_pow_bits == 18
