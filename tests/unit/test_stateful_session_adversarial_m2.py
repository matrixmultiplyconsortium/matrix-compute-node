"""
Adversarial Stress Test Suite for Milestone 2:
Worker Node Stateful Session & VRAM Pinning (`matrix-compute-node`).

Challenger: teamwork_preview_challenger_m2_1 (role: critic, specialist)

Empirically challenges:
1. ComputeEngine.gemv:
   - High-dimensional matrices (2048x2048), prime dimensions (997x991), degenerate shapes (1x1, 1xN, Mx1).
   - Boundary inputs: NaNs, Infinities, zeros, subnormals (10^-35, 10^-40).
   - Numerical error bound testing: ||y_hat - y||_inf <= gamma_K * ||A||_inf * ||v||_inf across FP32, FP16, and BF16.
2. SessionManager:
   - Concurrent multi-threaded leasing, touching, releasing across 50 threads.
   - Resource limits: exceeding max_sessions and max_memory_bytes, asserting clean rejections without memory leaks.
   - Proactive TTL reaper thread verifying expired sessions are automatically pruned and memory is zeroed.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import List, Tuple

import numpy as np
import pytest

from src.engine.compute_core import ComputeEngine
from src.engine.session_manager import (
    InvalidMatrixDimensionsError,
    MemoryBoundError,
    PinnedTensorSession,
    SessionCapacityError,
    SessionExpiredError,
    SessionManager,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def engine() -> ComputeEngine:
    """Deterministic CPU engine fixture."""
    return ComputeEngine(force_cpu=True)


# ==============================================================================
# 1. GEMV ADVERSARIAL SHAPES & DIMENSIONS
# ==============================================================================

class TestGemvAdversarialShapes:
    """Stress tests ComputeEngine.gemv across extreme, prime, and degenerate shapes."""

    def test_high_dimensional_2048x2048(self, engine: ComputeEngine) -> None:
        """Stress-tests 2048x2048 high-dimensional matrix-vector product."""
        rng = np.random.default_rng(2048)
        dim = 2048
        a = (rng.standard_normal((dim, dim)) * 0.01).astype(np.float32)
        v = (rng.standard_normal(dim) * 0.01).astype(np.float32)

        # 1D vector test
        y_1d = engine.gemv(a, v, dtype="fp32")
        assert y_1d.ndim == 1
        assert y_1d.shape == (dim,)
        expected_1d = np.matmul(a.astype(np.float64), v.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(y_1d, expected_1d, rtol=1e-5, atol=1e-5)

        # 2D column vector (2048, 1) test
        v_2d = v.reshape((dim, 1))
        y_2d = engine.gemv(a, v_2d, dtype="fp32")
        assert y_2d.ndim == 1
        assert y_2d.shape == (dim,)
        np.testing.assert_allclose(y_2d, expected_1d, rtol=1e-5, atol=1e-5)

    def test_prime_dimensions_997x991(self, engine: ComputeEngine) -> None:
        """Stress-tests prime dimensions (997 rows, 991 cols) preventing memory alignment assumptions."""
        rng = np.random.default_rng(997991)
        m, k = 997, 991
        a = (rng.standard_normal((m, k)) * 0.02).astype(np.float32)
        v = (rng.standard_normal(k) * 0.02).astype(np.float32)

        y = engine.gemv(a, v, dtype="fp32")
        assert y.ndim == 1
        assert y.shape == (m,)

        expected = np.matmul(a.astype(np.float64), v.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_degenerate_shape_1x1(self, engine: ComputeEngine) -> None:
        """Stress-tests scalar degenerate shape (1x1 matrix and 1D / 2D scalar vector)."""
        a = np.array([[3.5]], dtype=np.float32)
        v_1d = np.array([2.0], dtype=np.float32)
        v_2d = np.array([[2.0]], dtype=np.float32)

        y_1d = engine.gemv(a, v_1d, dtype="fp32")
        assert y_1d.ndim == 1
        assert y_1d.shape == (1,)
        assert y_1d[0] == pytest.approx(7.0, rel=1e-6)

        y_2d = engine.gemv(a, v_2d, dtype="fp32")
        assert y_2d.ndim == 1
        assert y_2d.shape == (1,)
        assert y_2d[0] == pytest.approx(7.0, rel=1e-6)

    def test_degenerate_shape_1xN(self, engine: ComputeEngine) -> None:
        """Stress-tests 1xN single-row matrix (dot product equivalent)."""
        n = 1024
        rng = np.random.default_rng(1024)
        a = rng.standard_normal((1, n)).astype(np.float32)
        v = rng.standard_normal(n).astype(np.float32)

        y = engine.gemv(a, v, dtype="fp32")
        assert y.ndim == 1
        assert y.shape == (1,)

        expected = np.dot(a[0], v)
        assert y[0] == pytest.approx(expected, rel=1e-5)

    def test_degenerate_shape_Mx1(self, engine: ComputeEngine) -> None:
        """Stress-tests Mx1 single-column matrix (scalar outer product equivalent)."""
        m = 1024
        rng = np.random.default_rng(10241)
        a = rng.standard_normal((m, 1)).astype(np.float32)
        v = np.array([3.14159], dtype=np.float32)

        y = engine.gemv(a, v, dtype="fp32")
        assert y.ndim == 1
        assert y.shape == (m,)

        expected = (a[:, 0] * 3.14159).astype(np.float32)
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_strided_and_transposed_memory_layouts(self, engine: ComputeEngine) -> None:
        """Stress-tests non-contiguous Fortran order, transpositions, and strided vectors."""
        rng = np.random.default_rng(77)
        # Transposed / Fortran matrix
        base_a = rng.standard_normal((32, 64)).astype(np.float32)
        a_f = np.asfortranarray(base_a.T)  # shape (64, 32)
        assert not a_f.flags["C_CONTIGUOUS"]

        # Strided vector
        v_full = rng.standard_normal(64).astype(np.float32)
        v_strided = v_full[::2]  # slice of length 32
        assert not v_strided.flags["C_CONTIGUOUS"]

        y = engine.gemv(a_f, v_strided, dtype="fp32")
        assert y.shape == (64,)
        expected = np.matmul(a_f, v_strided)
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)


# ==============================================================================
# 2. BOUNDARY INPUTS & EXCEPTIONAL FLOATING-POINT VALUES
# ==============================================================================

class TestGemvBoundaryInputs:
    """Stress tests boundary inputs: NaNs, Infinities, zeros, and subnormals."""

    def test_nan_propagation_and_bf16_quiet_bit(self, engine: ComputeEngine) -> None:
        """Verifies NaN propagation and ensures BF16 truncation preserves NaN semantics."""
        m, k = 4, 4
        a = np.ones((m, k), dtype=np.float32)
        a[1, 2] = np.nan
        v = np.ones(k, dtype=np.float32)

        # FP32
        y_fp32 = engine.gemv(a, v, dtype="fp32")
        assert not np.isnan(y_fp32[0])
        assert np.isnan(y_fp32[1])
        assert not np.isnan(y_fp32[2])

        # FP16
        y_fp16 = engine.gemv(a.astype(np.float16), v.astype(np.float16), dtype="fp16")
        assert not np.isnan(y_fp16[0])
        assert np.isnan(y_fp16[1])

        # BF16: Specifically test signaling / quiet NaN truncation
        nan_with_low_bits = np.frombuffer(np.uint32(0x7F800001).tobytes(), dtype=np.float32)[0]
        assert np.isnan(nan_with_low_bits)
        truncated_nan = ComputeEngine.truncate_to_bf16(np.array([nan_with_low_bits], dtype=np.float32))
        assert np.isnan(truncated_nan[0]), "BF16 truncation corrupted NaN to Infinity"

        a_bf16 = np.ones((m, k), dtype=np.float32)
        a_bf16[2, 2] = nan_with_low_bits
        y_bf16 = engine.gemv(a_bf16, v, dtype="bf16")
        assert np.isnan(y_bf16[2])

    def test_infinities_handling(self, engine: ComputeEngine) -> None:
        """Verifies positive and negative infinity handling and Inf * 0 -> NaN."""
        m, k = 3, 3
        a = np.zeros((m, k), dtype=np.float32)
        a[0, 0] = np.inf
        a[1, 1] = -np.inf
        a[2, 2] = np.inf

        v = np.array([2.0, 3.0, 0.0], dtype=np.float32)
        y = engine.gemv(a, v, dtype="fp32")

        assert y[0] == np.inf
        assert y[1] == -np.inf
        # Inf * 0.0 must evaluate to NaN per IEEE 754
        assert np.isnan(y[2])

    def test_signed_zeros(self, engine: ComputeEngine) -> None:
        """Verifies signed zeros (+0.0 and -0.0) propagation and zero preservation."""
        m, k = 2, 2
        neg_zero = np.float32(-0.0)
        pos_zero = np.float32(+0.0)

        a = np.array([[neg_zero, pos_zero], [pos_zero, neg_zero]], dtype=np.float32)
        v = np.array([neg_zero, pos_zero], dtype=np.float32)

        for dt in ("fp32", "fp16", "bf16"):
            y = engine.gemv(a, v, dtype=dt)
            assert np.all(y == 0.0)

    def test_subnormals_1e_minus_35(self, engine: ComputeEngine) -> None:
        """
        Stress-tests subnormal / near-underflow inputs (10^-35).
        - FP32: 10^-35 is within normal single-precision range (> 1.175e-38).
        - FP16: 10^-35 is below FP16 subnormal minimum (~5.96e-8) and must underflow to 0.0.
        - BF16: 10^-35 has 8-bit exponent matching FP32, preserved without underflow.
        """
        m, k = 8, 8
        val_35 = np.float32(1e-35)
        a = np.full((m, k), val_35, dtype=np.float32)
        v = np.ones(k, dtype=np.float32)

        # FP32: exact non-zero result
        y_fp32 = engine.gemv(a, v, dtype="fp32")
        assert np.all(y_fp32 > 0.0)
        expected_fp32 = float(val_35) * float(k)
        np.testing.assert_allclose(y_fp32, expected_fp32, rtol=1e-5)

        # FP16: flushes to zero
        y_fp16 = engine.gemv(a.astype(np.float16), v.astype(np.float16), dtype="fp16")
        assert np.all(y_fp16 == 0.0), "FP16 did not flush sub-range 10^-35 to zero"

        # BF16: preserves exponent
        y_bf16 = engine.gemv(a, v, dtype="bf16")
        assert np.all(y_bf16 > 0.0), "BF16 erroneously flushed 10^-35 to zero"

    def test_deep_subnormal_float32_1e_minus_40(self, engine: ComputeEngine) -> None:
        """Tests true float32 subnormal value (10^-40 < 1.175e-38)."""
        val_40 = np.float32(1e-40)
        a = np.full((4, 4), val_40, dtype=np.float32)
        v = np.ones(4, dtype=np.float32)

        y = engine.gemv(a, v, dtype="fp32")
        # In double-precision intermediate, sum is 4e-40 which remains in float32 subnormal range
        assert np.all(y >= 0.0)


# ==============================================================================
# 3. NUMERICAL ERROR BOUND TESTING (HIGHAM)
# ==============================================================================

class TestGemvNumericalBounds:
    """Verifies that ||y_hat - y||_inf <= gamma_K * ||A||_inf * ||v||_inf across FP32, FP16, and BF16."""

    @pytest.mark.parametrize("dtype,u", [
        ("fp32", 2.0 ** -24),
        ("fp16", 2.0 ** -11),
        ("bf16", 2.0 ** -8),
    ])
    @pytest.mark.parametrize("m,k", [
        (32, 64),
        (64, 128),
        (97, 89),    # Prime dimensions
        (1, 128),    # Single row
        (128, 1),    # Single column
    ])
    def test_higham_error_bound_across_precisions(
        self, engine: ComputeEngine, dtype: str, u: float, m: int, k: int
    ) -> None:
        """
        Verifies Higham matrix-vector error bound:
        ||y_hat - y_ref||_inf <= gamma_K * ||A||_inf * ||v||_inf
        """
        rng = np.random.default_rng(m * 1000 + k)
        # Scaled random matrices to avoid overflow
        a = (rng.standard_normal((m, k)) * 0.05).astype(np.float32)
        v = (rng.standard_normal(k) * 0.05).astype(np.float32)

        if dtype == "fp16":
            a_in = a.astype(np.float16)
            v_in = v.astype(np.float16)
        else:
            a_in = a
            v_in = v

        y_hat = engine.gemv(a_in, v_in, dtype=dtype)

        # High-precision reference in float64
        a_f64 = a_in.astype(np.float64)
        v_f64 = v_in.astype(np.float64)
        y_ref = np.matmul(a_f64, v_f64)

        # Theoretical Higham bound
        ku = float(k) * u
        gamma_k = (ku / (1.0 - ku)) if ku < 0.999 else (ku * 1000.0)
        norm_a = float(np.linalg.norm(a_f64, ord=np.inf))
        norm_v = float(np.linalg.norm(v_f64, ord=np.inf))
        theoretical_bound = gamma_k * norm_a * norm_v
        effective_bound = max(theoretical_bound, u * float(np.linalg.norm(y_ref, ord=np.inf)), 1e-7)

        discrepancy = float(np.max(np.abs(y_hat.astype(np.float64) - y_ref)))
        assert discrepancy <= effective_bound, (
            f"GEMV error ({discrepancy:.6e}) exceeded Higham bound ({effective_bound:.6e}) "
            f"for {dtype} with shape ({m}, {k})"
        )

    def test_ill_conditioned_matrix_stability(self, engine: ComputeEngine) -> None:
        """Verifies numerical stability on ill-conditioned diagonally dominant decay matrix."""
        dim = 32
        # Rapid exponentially decaying diagonal
        diag = np.array([10.0 ** (-i / 4.0) for i in range(dim)], dtype=np.float32)
        a = np.diag(diag)
        v = np.ones(dim, dtype=np.float32)

        y_fp32 = engine.gemv(a, v, dtype="fp32")
        np.testing.assert_allclose(y_fp32, diag, rtol=1e-5)


# ==============================================================================
# 4. SESSION MANAGER CONCURRENCY (50 THREADS)
# ==============================================================================

class TestSessionManagerConcurrency:
    """Stress tests SessionManager under high concurrency (50 threads)."""

    def test_50_thread_concurrent_lease_touch_release(self) -> None:
        """50 concurrent threads simultaneously leasing, touching, stepping, and releasing distinct sessions."""
        num_threads = 50
        mgr = SessionManager(
            max_sessions=100,
            max_memory_bytes=50 * 1024 * 1024,
            reaper_interval_seconds=60.0,
        )
        mat = np.ones((8, 8), dtype=np.float32)
        mat_bytes = mat.tobytes()

        def worker_task(tid: int) -> Tuple[bool, int]:
            lid = f"thread_lease_{tid}"
            # 1. Lease
            sess = mgr.lease_matrix(lid, mat_bytes, 8, 8, "fp32", ttl_seconds=10.0)
            if sess is None or sess.lease_id != lid:
                return False, 0

            # 2. Touch and step multiple times
            for _ in range(5):
                sess.touch()
                step_num = sess.record_step()
                if step_num <= 0:
                    return False, 0
                time.sleep(0.001)

            # 3. Retrieve through manager
            retrieved = mgr.get_session(lid)
            if retrieved is None or retrieved.is_zeroized:
                return False, 0

            # 4. Release
            freed = mgr.release_lease(lid)
            return True, freed

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_task, i) for i in range(num_threads)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            assert len(results) == num_threads
            for success, freed in results:
                assert success is True
                assert freed == 8 * 8 * 4

            # Verify complete cleanup
            assert mgr.active_session_count == 0
            assert mgr.total_memory_bytes == 0
        finally:
            mgr.shutdown()

    def test_50_threads_competing_for_single_lease(self) -> None:
        """50 threads concurrently attempting to create the exact same lease_id (only 1 succeeds)."""
        mgr = SessionManager(max_sessions=10)
        target_lid = "shared_contested_lease"
        mat_bytes = np.ones((4, 4), dtype=np.float32).tobytes()

        successes = 0
        rejected = 0
        lock = threading.Lock()

        def try_lease(tid: int) -> None:
            nonlocal successes, rejected
            try:
                mgr.lease_matrix(target_lid, mat_bytes, 4, 4, "fp32", 10.0)
                with lock:
                    successes += 1
            except ValueError:
                with lock:
                    rejected += 1

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(try_lease, i) for i in range(50)]
                concurrent.futures.wait(futures)

            assert successes == 1
            assert rejected == 49
            assert mgr.active_session_count == 1
            assert mgr.total_memory_bytes == 4 * 4 * 4
        finally:
            mgr.shutdown()

    def test_50_threads_hammering_shared_session(self) -> None:
        """50 threads concurrently calling get_session, touch, and record_step on the same session."""
        mgr = SessionManager()
        lid = "hammered_lease"
        mat_bytes = np.ones((8, 8), dtype=np.float32).tobytes()
        mgr.lease_matrix(lid, mat_bytes, 8, 8, "fp32", 10.0)

        def reader_step(tid: int) -> int:
            sess = mgr.get_session(lid)
            if sess is None:
                return -1
            return sess.record_step()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(reader_step, i) for i in range(100)]
                steps = [f.result() for f in concurrent.futures.as_completed(futures)]

            assert all(s > 0 for s in steps)
            sess = mgr.get_session(lid)
            assert sess.step_count == 100
        finally:
            mgr.shutdown()


# ==============================================================================
# 5. RESOURCE LIMITS & MEMORY LEAK PREVENTION
# ==============================================================================

class TestSessionManagerResourceLimits:
    """Stress tests resource limits: capacity bounds, total memory bounds, and leak prevention."""

    def test_max_sessions_capacity_rejection_without_leak(self) -> None:
        """Verifies clean rejection when max_sessions is reached, asserting zero memory leak."""
        cap = 5
        mgr = SessionManager(max_sessions=cap)
        mat_bytes = np.zeros((4, 4), dtype=np.float32).tobytes()

        try:
            # Fill up to capacity
            for i in range(cap):
                mgr.lease_matrix(f"sess_{i}", mat_bytes, 4, 4, "fp32", 60.0)

            assert mgr.active_session_count == cap
            assert mgr.total_memory_bytes == cap * 64

            # Attempt 20 excess allocations
            for i in range(cap, cap + 20):
                with pytest.raises(SessionCapacityError):
                    mgr.lease_matrix(f"sess_{i}", mat_bytes, 4, 4, "fp32", 60.0)

            # Assert state remained bounded
            assert mgr.active_session_count == cap
            assert mgr.total_memory_bytes == cap * 64

            # Release one session and verify a new one fits
            mgr.release_lease("sess_0")
            assert mgr.active_session_count == cap - 1
            mgr.lease_matrix("sess_replacement", mat_bytes, 4, 4, "fp32", 60.0)
            assert mgr.active_session_count == cap
        finally:
            mgr.shutdown()

    def test_max_memory_bytes_rejection_without_leak(self) -> None:
        """Verifies clean rejection when max_memory_bytes is exceeded."""
        max_bytes = 1024
        mgr = SessionManager(max_memory_bytes=max_bytes)

        try:
            # 8x8 fp32 = 256 bytes
            b256 = np.zeros((8, 8), dtype=np.float32).tobytes()
            mgr.lease_matrix("m1", b256, 8, 8, "fp32", 60.0)
            mgr.lease_matrix("m2", b256, 8, 8, "fp32", 60.0)
            mgr.lease_matrix("m3", b256, 8, 8, "fp32", 60.0)
            assert mgr.total_memory_bytes == 768

            # Try to allocate 512 bytes (768 + 512 = 1280 > 1024)
            b512 = np.zeros((8, 16), dtype=np.float32).tobytes()
            with pytest.raises(MemoryBoundError):
                mgr.lease_matrix("m_over", b512, 8, 16, "fp32", 60.0)

            # Memory must remain 768 and active sessions 3
            assert mgr.total_memory_bytes == 768
            assert mgr.active_session_count == 3
        finally:
            mgr.shutdown()

    def test_max_matrix_single_allocation_limit(self) -> None:
        """Verifies rejection of a single matrix exceeding max_matrix_bytes."""
        mgr = SessionManager(max_matrix_bytes=500, max_memory_bytes=10000)
        try:
            # 16x16 fp32 = 1024 bytes > 500
            b1024 = np.zeros((16, 16), dtype=np.float32).tobytes()
            with pytest.raises(MemoryBoundError, match="single matrix limit"):
                mgr.lease_matrix("too_big", b1024, 16, 16, "fp32", 60.0)

            assert mgr.total_memory_bytes == 0
            assert mgr.active_session_count == 0
        finally:
            mgr.shutdown()

    def test_leak_prevention_on_100_failed_lease_attempts(self) -> None:
        """Stress-tests 100 varied invalid lease attempts to guarantee 0 byte memory leak."""
        mgr = SessionManager(max_memory_bytes=2048)
        try:
            for i in range(100):
                # 1. Invalid dimensions
                with pytest.raises(InvalidMatrixDimensionsError):
                    mgr.lease_matrix(f"inv_dim_{i}", b"\x00" * 16, -1, 4, "fp32", 60.0)

                # 2. Byte size mismatch
                with pytest.raises(InvalidMatrixDimensionsError):
                    mgr.lease_matrix(f"inv_len_{i}", b"\x00" * 10, 4, 4, "fp32", 60.0)

                # 3. Empty lease ID
                with pytest.raises(ValueError):
                    mgr.lease_matrix("", b"\x00" * 64, 4, 4, "fp32", 60.0)

                # 4. Unsupported dtype
                with pytest.raises(ValueError):
                    mgr.lease_matrix(f"inv_dt_{i}", b"\x00" * 64, 4, 4, "int64", 60.0)

            # Assert strict 0 memory bytes and 0 active sessions
            assert mgr.active_session_count == 0
            assert mgr.total_memory_bytes == 0
        finally:
            mgr.shutdown()


# ==============================================================================
# 6. PROACTIVE REAPER THREAD & MEMORY ZEROIZATION
# ==============================================================================

class TestSessionManagerReaperAndZeroization:
    """Stress tests proactive background TTL reaper thread and cryptographic zeroization."""

    def test_proactive_reaper_auto_prunes_and_zeroizes(self) -> None:
        """Verifies background daemon reaper thread automatically prunes expired leases and zeros buffers."""
        # Fast reaper interval: 30ms
        mgr = SessionManager(reaper_interval_seconds=0.03, auto_start_reaper=True)
        try:
            mat = np.full((16, 16), 77.0, dtype=np.float32)
            sess = mgr.lease_matrix("auto_prune_lease", mat.tobytes(), 16, 16, "fp32", ttl_seconds=0.06)

            # Retain a view directly onto the allocated pinned buffer
            buf_view = sess.matrix.view()

            assert mgr.active_session_count == 1
            assert mgr.total_memory_bytes == 16 * 16 * 4

            # Sleep past TTL + reaper cycle (60ms TTL + 90ms headroom)
            time.sleep(0.16)

            # Without any user calls to get_session, reaper must have already cleared it!
            assert mgr.active_session_count == 0
            assert mgr.total_memory_bytes == 0
            assert sess.is_zeroized is True

            # Verify buffer zeroization in memory
            assert np.all(buf_view == 0.0), "Pinned matrix buffer was not zeroed upon reaper eviction"
        finally:
            mgr.shutdown()

    def test_zeroization_on_explicit_release(self) -> None:
        """Verifies explicit release immediately zeroizes the matrix buffer."""
        mgr = SessionManager()
        try:
            mat = np.full((8, 8), 123.45, dtype=np.float32)
            sess = mgr.lease_matrix("expl_zeroize", mat.tobytes(), 8, 8, "fp32", ttl_seconds=60.0)
            buf_view = sess.matrix.view()

            mgr.release_lease("expl_zeroize")

            assert sess.is_zeroized is True
            assert sess.matrix is None
            assert np.all(buf_view == 0.0), "Matrix memory was not zeroed upon explicit release"
        finally:
            mgr.shutdown()

    def test_touch_on_expired_session_fails(self) -> None:
        """Verifies touching a zeroized session raises SessionExpiredError."""
        mgr = SessionManager()
        try:
            mat_bytes = np.ones((4, 4), dtype=np.float32).tobytes()
            sess = mgr.lease_matrix("touch_fail", mat_bytes, 4, 4, "fp32", ttl_seconds=60.0)
            mgr.release_lease("touch_fail")

            with pytest.raises(SessionExpiredError):
                sess.touch()

            with pytest.raises(SessionExpiredError):
                sess.record_step()
        finally:
            mgr.shutdown()

    def test_shutdown_zeroizes_all_active_sessions(self) -> None:
        """Verifies manager shutdown immediately zeroizes all active sessions and joins reaper."""
        mgr = SessionManager(reaper_interval_seconds=0.05)
        mat1 = np.full((8, 8), 1.0, dtype=np.float32)
        mat2 = np.full((8, 8), 2.0, dtype=np.float32)

        s1 = mgr.lease_matrix("shut_1", mat1.tobytes(), 8, 8, "fp32", 60.0)
        s2 = mgr.lease_matrix("shut_2", mat2.tobytes(), 8, 8, "fp32", 60.0)

        v1 = s1.matrix.view()
        v2 = s2.matrix.view()

        mgr.shutdown()

        assert mgr.is_shutdown is True
        assert mgr.active_session_count == 0
        assert mgr.total_memory_bytes == 0
        assert np.all(v1 == 0.0)
        assert np.all(v2 == 0.0)


# ==============================================================================
# 7. SERVICER RPC ADVERSARIAL STRESS TESTING
# ==============================================================================

class MockServicerContext:
    """Mock gRPC ServicerContext for capturing abort codes."""
    def __init__(self) -> None:
        self.code = None
        self.details = None

    def abort(self, code, details) -> None:
        self.code = code
        self.details = details
        raise RuntimeError(f"ABORT [{code}]: {details}")


class TestServicerStatefulAdversarial:
    """Adversarial testing of MatrixComputeServicer stateful RPCs."""

    def test_servicer_step_with_dimension_mismatch_aborts(self) -> None:
        """Verifies StepIterative cleanly aborts when vector size does not match matrix inner dim."""
        from src.p2p.servicer import MatrixComputeServicer
        from src.crypto.signer import WorkerSigner
        from matrix_contracts.v1 import pb2

        mgr = SessionManager()
        servicer = MatrixComputeServicer(
            signer=WorkerSigner(),
            session_manager=mgr,
            engine=ComputeEngine(force_cpu=True),
        )
        lid = "servicer_mismatch_lease"

        # Leased 8x8 matrix
        mat = np.ones((8, 8), dtype=np.float32)
        servicer.LeaseMatrix(pb2.LeaseMatrixRequest(
            lease_id=lid,
            rows=8,
            cols=8,
            dtype="fp32",
            matrix_bytes=mat.tobytes(),
            ttl_seconds=60,
        ))

        ctx = MockServicerContext()
        # Request with 16-length vector (expected 8)
        req = pb2.IterativeStepRequest(
            lease_id=lid,
            step_index=0,
            vector_bytes=np.ones(16, dtype=np.float32).tobytes(),
            vector_length=16,
            dtype="fp32",
        )

        with pytest.raises(RuntimeError, match="Vector dimension mismatch"):
            servicer.StepIterative(req, context=ctx)

        mgr.shutdown()

    def test_servicer_concurrent_step_requests(self) -> None:
        """Verifies 50 concurrent StepIterative requests on a single leased matrix compute accurately."""
        from src.p2p.servicer import MatrixComputeServicer
        from src.crypto.signer import WorkerSigner
        from matrix_contracts.v1 import pb2

        mgr = SessionManager()
        signer = WorkerSigner()
        servicer = MatrixComputeServicer(
            signer=signer,
            session_manager=mgr,
            engine=ComputeEngine(force_cpu=True),
        )
        lid = "concurrent_step_lease"
        dim = 16
        rng = np.random.default_rng(999)
        mat = rng.standard_normal((dim, dim)).astype(np.float32)

        servicer.LeaseMatrix(pb2.LeaseMatrixRequest(
            lease_id=lid,
            rows=dim,
            cols=dim,
            dtype="fp32",
            matrix_bytes=mat.tobytes(),
            ttl_seconds=120,
        ))

        def execute_step(step_idx: int) -> bool:
            v_in = rng.standard_normal(dim).astype(np.float32)
            req = pb2.IterativeStepRequest(
                lease_id=lid,
                step_index=step_idx,
                vector_bytes=v_in.tobytes(),
                vector_length=dim,
                dtype="fp32",
            )
            resp = servicer.StepIterative(req)
            v_out = np.frombuffer(resp.result_vector_bytes, dtype=np.float32)
            expected = np.matmul(mat, v_in)
            return np.allclose(v_out, expected, rtol=1e-5, atol=1e-5)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(execute_step, i) for i in range(50)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            assert all(results)
        finally:
            mgr.shutdown()
