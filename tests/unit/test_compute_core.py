"""Unit tests for ComputeEngine."""

import hashlib
import numpy as np
import pytest

from src.engine.compute_core import ComputeEngine


@pytest.fixture
def engine():
    return ComputeEngine()


def test_compute_engine_init(engine):
    assert engine.backend in ("torch_cuda", "cupy", "torch_cpu", "numpy")


def test_gemm_fp32(engine):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    c = engine.gemm(a, b, dtype="fp32")

    expected = np.matmul(a, b)
    np.testing.assert_allclose(c, expected, rtol=1e-5)
    assert c.dtype == np.float32


def test_gemm_fp16(engine):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float16)
    c = engine.gemm(a, b, dtype="fp16")

    expected = np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(np.float16)
    np.testing.assert_allclose(c, expected, rtol=1e-3)
    assert c.dtype == np.float16


def test_gemm_bf16(engine):
    a = np.ones((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    c = engine.gemm(a, b, dtype="bf16")

    assert c.shape == (4, 4)
    assert np.all(c == 4.0)


def test_gemm_dimension_mismatch(engine):
    a = np.ones((3, 4), dtype=np.float32)
    b = np.ones((5, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Inner dimension mismatch"):
        engine.gemm(a, b, dtype="fp32")


def test_gemm_invalid_ndim(engine):
    a = np.ones((3, 4, 2), dtype=np.float32)
    b = np.ones((4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="2-dimensional"):
        engine.gemm(a, b, dtype="fp32")


def test_gemm_unsupported_dtype(engine):
    a = np.ones((2, 2), dtype=np.float32)
    b = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported dtype"):
        engine.gemm(a, b, dtype="int32")


def test_compute_sha256(engine):
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    digest = engine.compute_sha256(tensor)
    expected = hashlib.sha256(tensor.tobytes()).hexdigest()
    assert digest == expected


def test_verify_freivalds_accurate(engine):
    np.random.seed(42)
    a = np.random.randn(16, 16).astype(np.float32)
    b = np.random.randn(16, 16).astype(np.float32)
    c = engine.gemm(a, b, dtype="fp32")

    passed, disc = engine.verify_freivalds(a, b, c, epsilon=1e-4)
    assert passed is True
    assert disc < 1e-4


def test_verify_freivalds_fraud(engine):
    np.random.seed(42)
    a = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8, 8).astype(np.float32)
    c_fraud = np.random.randn(8, 8).astype(np.float32)

    passed, disc = engine.verify_freivalds(a, b, c_fraud, epsilon=1e-4)
    assert passed is False
    assert disc > 1e-4
