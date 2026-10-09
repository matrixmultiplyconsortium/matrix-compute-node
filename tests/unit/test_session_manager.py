"""
Unit tests for PinnedTensorSession and SessionManager.
Verifies session allocation, tensor pinning, activity touching,
explicit release, memory zeroization, capacity bounds, and hybrid TTL eviction.
"""

import concurrent.futures
import time
import numpy as np
import pytest

from src.engine.session_manager import PinnedTensorSession, SessionManager


# ==============================================================================
# 1. PinnedTensorSession Lifecycle & Zeroization Tests
# ==============================================================================

def test_pinned_tensor_session_init_and_touch():
    """Verifies PinnedTensorSession initialization, expiration check, and touch."""
    arr = np.ones((8, 8), dtype=np.float32)
    session = PinnedTensorSession(
        lease_id="test_lease_01",
        matrix=arr,
        rows=8,
        cols=8,
        dtype="fp32",
        ttl_seconds=10.0,
    )

    assert session.lease_id == "test_lease_01"
    assert session.rows == 8
    assert session.cols == 8
    assert session.memory_bytes == 8 * 8 * 4
    assert session.is_expired() is False

    old_access = session.last_accessed_at
    time.sleep(0.01)
    session.touch()
    assert session.last_accessed_at > old_access


def test_pinned_tensor_session_zeroize():
    """Verifies that session.zeroize() securely overwrites matrix buffer with zeros."""
    arr = np.full((16, 16), 42.0, dtype=np.float32)
    session = PinnedTensorSession(
        lease_id="zeroize_lease",
        matrix=arr,
        rows=16,
        cols=16,
        dtype="fp32",
        ttl_seconds=60.0,
    )

    # Retain view to the underlying buffer
    buf_view = arr.view()
    session.zeroize()

    assert np.all(buf_view == 0.0), "Buffer was not zeroized"
    assert session.is_zeroized is True


# ==============================================================================
# 2. SessionManager Leasing & Retrieval
# ==============================================================================

def test_session_manager_lease_and_get():
    """Verifies allocating and retrieving a pinned tensor session."""
    mgr = SessionManager(max_sessions=10, max_memory_bytes=1024 * 1024)
    try:
        mat = np.random.randn(8, 8).astype(np.float32)
        mat_bytes = mat.tobytes()

        session = mgr.lease_matrix(
            lease_id="lease_1",
            matrix_bytes=mat_bytes,
            rows=8,
            cols=8,
            dtype="fp32",
            ttl_seconds=300.0,
        )

        assert session is not None
        assert session.lease_id == "lease_1"
        assert mgr.active_session_count == 1
        assert mgr.total_memory_bytes == 8 * 8 * 4

        # Retrieve session
        retrieved = mgr.get_session("lease_1")
        assert retrieved is not None
        assert retrieved.lease_id == "lease_1"
        np.testing.assert_allclose(retrieved.matrix, mat, rtol=1e-6)
    finally:
        mgr.shutdown()


def test_session_manager_duplicate_lease_rejected():
    """Verifies that duplicate lease_id registration raises ValueError."""
    mgr = SessionManager(max_sessions=5)
    try:
        mat_bytes = np.zeros((4, 4), dtype=np.float32).tobytes()
        mgr.lease_matrix("dup_lease", mat_bytes, 4, 4, "fp32", 60.0)

        with pytest.raises(ValueError, match="already exists|Duplicate lease"):
            mgr.lease_matrix("dup_lease", mat_bytes, 4, 4, "fp32", 60.0)
    finally:
        mgr.shutdown()


def test_session_manager_byte_size_mismatch_rejected():
    """Verifies rejection when matrix_bytes length does not match rows * cols * itemsize."""
    mgr = SessionManager()
    try:
        # 4x4 fp32 expects 64 bytes; provide 32 bytes
        corrupted_bytes = b"\x00" * 32
        with pytest.raises(ValueError, match="size mismatch|Byte size"):
            mgr.lease_matrix("corrupt_lease", corrupted_bytes, 4, 4, "fp32", 60.0)
    finally:
        mgr.shutdown()


# ==============================================================================
# 3. Capacity & Memory Footprint Bounding
# ==============================================================================

def test_session_manager_capacity_limit_enforced():
    """Verifies that exceeding max_sessions raises an exception and does not leak."""
    mgr = SessionManager(max_sessions=2)
    try:
        mat_bytes = np.zeros((4, 4), dtype=np.float32).tobytes()
        mgr.lease_matrix("s1", mat_bytes, 4, 4, "fp32", 60.0)
        mgr.lease_matrix("s2", mat_bytes, 4, 4, "fp32", 60.0)

        assert mgr.active_session_count == 2

        with pytest.raises(RuntimeError, match="capacity exceeded|Maximum sessions"):
            mgr.lease_matrix("s3", mat_bytes, 4, 4, "fp32", 60.0)

        assert mgr.active_session_count == 2
    finally:
        mgr.shutdown()


def test_session_manager_memory_limit_enforced():
    """Verifies that exceeding max_memory_bytes rejects new lease."""
    # Cap total memory at 1024 bytes
    mgr = SessionManager(max_memory_bytes=1024)
    try:
        # Allocate 8x8 fp32 = 256 bytes (fits)
        b1 = np.zeros((8, 8), dtype=np.float32).tobytes()
        mgr.lease_matrix("s_mem1", b1, 8, 8, "fp32", 60.0)
        assert mgr.total_memory_bytes == 256

        # Attempt to allocate 16x16 fp32 = 1024 bytes (256 + 1024 = 1280 > 1024)
        b2 = np.zeros((16, 16), dtype=np.float32).tobytes()
        with pytest.raises(RuntimeError, match="Memory limit exceeded|insufficient memory"):
            mgr.lease_matrix("s_mem2", b2, 16, 16, "fp32", 60.0)

        # Confirm memory accounting remains intact
        assert mgr.total_memory_bytes == 256
        assert mgr.active_session_count == 1
    finally:
        mgr.shutdown()


# ==============================================================================
# 4. Explicit Release & Memory Reclamation
# ==============================================================================

def test_session_manager_explicit_release():
    """Verifies explicit release frees memory and marks lease as released."""
    mgr = SessionManager()
    try:
        mat = np.full((8, 8), 99.0, dtype=np.float32)
        session = mgr.lease_matrix("rel_lease", mat.tobytes(), 8, 8, "fp32", 60.0)

        # Keep buffer view to check zeroization
        buf_ref = session.matrix.view()

        bytes_freed = mgr.release_lease("rel_lease")
        assert bytes_freed == 8 * 8 * 4
        assert mgr.active_session_count == 0
        assert mgr.total_memory_bytes == 0

        # Verify zeroization occurred
        assert np.all(buf_ref == 0.0)

        # Second release should be a no-op
        assert mgr.release_lease("rel_lease") == 0
    finally:
        mgr.shutdown()


# ==============================================================================
# 5. Hybrid TTL Eviction (Lazy + Proactive Reaper)
# ==============================================================================

def test_session_manager_lazy_ttl_eviction():
    """Verifies that expired sessions are evicted on get_session access."""
    mgr = SessionManager()
    try:
        mat_bytes = np.zeros((4, 4), dtype=np.float32).tobytes()
        # Set short TTL
        mgr.lease_matrix("lazy_exp", mat_bytes, 4, 4, "fp32", ttl_seconds=0.05)

        # Wait past TTL
        time.sleep(0.08)

        # Access expired lease
        session = mgr.get_session("lazy_exp")
        assert session is None
        assert mgr.active_session_count == 0
        assert mgr.total_memory_bytes == 0
    finally:
        mgr.shutdown()


def test_session_manager_proactive_reaper_eviction():
    """Verifies background daemon reaper thread automatically cleans expired sessions."""
    # Fast reaper cycle: 20ms
    mgr = SessionManager(reaper_interval_seconds=0.02)
    try:
        mat_bytes = np.zeros((4, 4), dtype=np.float32).tobytes()
        mgr.lease_matrix("reaper_exp", mat_bytes, 4, 4, "fp32", ttl_seconds=0.05)

        assert mgr.active_session_count == 1
        assert mgr.total_memory_bytes == 64

        # Wait for reaper to cycle past expiration (50ms TTL + 70ms reaper headroom)
        time.sleep(0.12)

        # Without calling get_session, reaper must have already cleared it
        assert mgr.active_session_count == 0
        assert mgr.total_memory_bytes == 0
    finally:
        mgr.shutdown()


# ==============================================================================
# 6. Concurrency & Clean Shutdown
# ==============================================================================

def test_session_manager_concurrent_operations():
    """Verifies thread safety across concurrent leasing, touching, and releasing."""
    mgr = SessionManager(max_sessions=50, max_memory_bytes=1024 * 1024)
    mat_bytes = np.ones((4, 4), dtype=np.float32).tobytes()

    def worker_task(idx):
        lid = f"concurrent_lease_{idx}"
        mgr.lease_matrix(lid, mat_bytes, 4, 4, "fp32", ttl_seconds=2.0)
        sess = mgr.get_session(lid)
        if sess:
            sess.touch()
        time.sleep(0.01)
        mgr.release_lease(lid)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker_task, i) for i in range(20)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert mgr.active_session_count == 0
        assert mgr.total_memory_bytes == 0
    finally:
        mgr.shutdown()


def test_session_manager_shutdown_reclaims_all():
    """Verifies that shutdown stops the reaper thread and wipes all active sessions."""
    mgr = SessionManager(reaper_interval_seconds=0.05)
    mat_bytes = np.ones((4, 4), dtype=np.float32).tobytes()
    mgr.lease_matrix("s1", mat_bytes, 4, 4, "fp32", 60.0)
    mgr.lease_matrix("s2", mat_bytes, 4, 4, "fp32", 60.0)

    mgr.shutdown()

    assert mgr.active_session_count == 0
    assert mgr.total_memory_bytes == 0
    assert mgr.is_shutdown is True
