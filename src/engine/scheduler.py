"""Preemptive Two-Level Priority Compute Scheduler for matrix-compute-node."""

from __future__ import annotations

from collections import deque
from enum import Enum
import logging
import threading
import time
from typing import Any, Callable, Deque, Optional, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PriorityTier(str, Enum):
    """Execution priority tiers for compute tasks."""

    PAID_MAINNET = "PAID_MAINNET"
    FREE_TRIAL = "FREE_TRIAL"


class PreemptiveComputeScheduler:
    """
    Two-level preemptive compute scheduler enforcing:
    - Priority 1 (`PAID_MAINNET`): Executes immediately with 100% GPU allocation.
      Wakes any sleeping condition immediately upon arrival.
    - Priority 2 (`FREE_TRIAL`): Executes strictly when `paid_queue_depth == 0`.
      Enforces a <= 5% GPU duty cycle via:
        1. Sliding window cap (`<= 50ms` per `1.0s` rolling window).
        2. Proportional cooldown (`T_cooldown >= 19 * T_exec` for 5% duty cycle).
    """

    def __init__(
        self,
        sliding_window_seconds: float = 1.0,
        max_trial_ms_per_window: float = 50.0,
        max_trial_duty_cycle: float = 0.05,
        max_trial_ms_per_second: Optional[float] = None,
        window_seconds: Optional[float] = None,
        max_wait_seconds: float = 5.0,
    ) -> None:
        win_sec = float(window_seconds if window_seconds is not None else sliding_window_seconds)
        max_ms = float(
            max_trial_ms_per_second if max_trial_ms_per_second is not None else max_trial_ms_per_window
        )
        duty = float(max_trial_duty_cycle)
        if win_sec <= 0:
            raise ValueError(f"sliding_window_seconds must be positive, got {win_sec}")
        if max_ms <= 0:
            raise ValueError(f"max_trial_ms_per_window must be positive, got {max_ms}")
        if not (0.0 < duty <= 1.0):
            raise ValueError(f"max_trial_duty_cycle must be in (0.0, 1.0], got {duty}")

        self.sliding_window_seconds: float = win_sec
        self.window_seconds: float = win_sec
        self.max_trial_ms_per_window: float = max_ms
        self.max_trial_ms_per_second: float = max_ms
        self.max_trial_duty_cycle: float = duty
        self.max_wait_seconds: float = float(max_wait_seconds)

        self.lock = threading.RLock()
        self._condition = threading.Condition(self.lock)

        self.paid_queue_depth: int = 0
        self.trial_queue_depth: int = 0
        self._active_paid: int = 0
        self._active_trial: int = 0

        # Sliding window history of (finish_timestamp_sec, duration_ms)
        self._trial_execution_history: Deque[Tuple[float, float]] = deque()
        self._last_trial_finish_time: float = 0.0
        self._last_trial_cooldown_required: float = 0.0

    def get_window_trial_usage_ms(self, now: Optional[float] = None) -> float:
        """Calculates trial compute time (ms) used within the sliding window."""
        current_time = float(now if now is not None else time.time())
        with self.lock:
            cutoff = current_time - self.sliding_window_seconds
            while self._trial_execution_history and self._trial_execution_history[0][0] < cutoff:
                self._trial_execution_history.popleft()
            return sum(duration_ms for _, duration_ms in self._trial_execution_history)

    def can_execute_trial(
        self,
        estimated_exec_ms: float = 1.0,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Determines whether a FREE_TRIAL task can execute right now.
        Must satisfy:
        1. `paid_queue_depth == 0`
        2. Cooldown from previous trial execution has expired (`T_cooldown >= 19 * T_exec`)
        3. Sliding window usage + `estimated_exec_ms <= max_trial_ms_per_window` (50ms)
        """
        with self.lock:
            if self.paid_queue_depth > 0 or self._active_paid > 0:
                return False, f"PAID_QUEUE_ACTIVE: {self.paid_queue_depth} paid tasks waiting"

            current_time = float(now if now is not None else time.time())
            time_since_last = current_time - self._last_trial_finish_time
            if time_since_last < self._last_trial_cooldown_required:
                rem_cooldown = self._last_trial_cooldown_required - time_since_last
                return False, f"COOLDOWN_ACTIVE: {rem_cooldown:.4f}s cooldown remaining"

            used_ms = self.get_window_trial_usage_ms(current_time)
            if used_ms + float(estimated_exec_ms) > self.max_trial_ms_per_window:
                return (
                    False,
                    f"WINDOW_QUOTA_EXCEEDED: {used_ms:.2f}ms used, quota {self.max_trial_ms_per_window}ms",
                )

            return True, "OK"

    def execute_paid(self, compute_fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Executes a `PAID_MAINNET` task immediately with Priority 1.
        Preempts waiting trial tasks and wakes condition variables immediately.
        """
        if not callable(compute_fn):
            raise TypeError("compute_fn must be callable")

        with self.lock:
            self.paid_queue_depth += 1
            self._active_paid += 1
            self._condition.notify_all()

        try:
            return compute_fn(*args, **kwargs)
        finally:
            with self.lock:
                self._active_paid = max(0, self._active_paid - 1)
                self.paid_queue_depth = max(0, self.paid_queue_depth - 1)
                self._condition.notify_all()

    def execute_trial(
        self,
        compute_fn: Callable[..., T],
        *args: Any,
        max_wait_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """
        Executes a `FREE_TRIAL` task with Priority 2 and <= 5% duty cycle throttle.
        """
        if not callable(compute_fn):
            raise TypeError("compute_fn must be callable")

        wait_budget = float(max_wait_seconds if max_wait_seconds is not None else self.max_wait_seconds)

        with self.lock:
            self.trial_queue_depth += 1

        try:
            start_wait = time.time()
            can_run = False
            reason = ""

            while True:
                with self.lock:
                    can_run, reason = self.can_execute_trial(estimated_exec_ms=1.0)
                    if can_run and self._active_trial == 0:
                        self._active_trial += 1
                        break

                    elapsed = time.time() - start_wait
                    if elapsed >= wait_budget:
                        break

                    rem_wait = min(0.005, max(0.001, wait_budget - elapsed))
                    self._condition.wait(timeout=rem_wait)

            if not can_run:
                raise RuntimeError(f"Trial execution throttled or preempted: {reason}")

            t0 = time.time()
            try:
                result = compute_fn(*args, **kwargs)
            finally:
                t1 = time.time()
                exec_duration = max(0.0001, t1 - t0)
                with self.lock:
                    self._active_trial = max(0, self._active_trial - 1)
                    exec_ms = exec_duration * 1000.0
                    self._trial_execution_history.append((t1, exec_ms))
                    self._last_trial_finish_time = t1
                    cooldown_mult = (1.0 - self.max_trial_duty_cycle) / self.max_trial_duty_cycle
                    self._last_trial_cooldown_required = cooldown_mult * exec_duration
                    self._condition.notify_all()

            return result
        finally:
            with self.lock:
                self.trial_queue_depth = max(0, self.trial_queue_depth - 1)

    def execute(
        self,
        tier: Union[PriorityTier, str],
        compute_fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Schedules and executes `compute_fn` according to priority tier (`PAID_MAINNET` vs `FREE_TRIAL`).
        """
        tier_str = tier.value if isinstance(tier, PriorityTier) else str(tier).strip().upper()
        if tier_str == PriorityTier.PAID_MAINNET.value:
            return self.execute_paid(compute_fn, *args, **kwargs)
        elif tier_str == PriorityTier.FREE_TRIAL.value:
            return self.execute_trial(compute_fn, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported PriorityTier: {tier}")
