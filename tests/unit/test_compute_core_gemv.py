"""
Unit tests for ComputeEngine.gemv (Matrix-Vector Multiplication).
Verifies dimensional handling, precision modes (FP32, FP16, BF16),
and Higham numerical floating-point error bounds.
"""

import numpy as np
import pytest

from src.engine.compute_core import ComputeEngine


@pytest.fixture
def engine():
    """Provides CPU-deterministic ComputeEngine instance for unit testing."""
    return ComputeEngine(force_cpu=True)


# ==============================================================================
# 1. Basic Dimensionality & Vector Shape Tests
# ==============================================================================

def test_gemv_1d_vector_fp32(engine):
    """Verifies GEMV with 1D vector (K,) produces 1D result (M,)."""
    np.random.seed(42)
    m, k = 8, 12
    a = np.random.randn(m, k).astype(np.float32)
    v = np.random.randn(k).astype(np.float32)

    res = engine.gemv(a, v, dtype="fp32")

    assert res.ndim == 1
    assert res.shape == (m,)
    assert res.dtype == np.float32

    expected = np.matmul(a, v)
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-6)


def test_gemv_2d_column_vector_fp32(engine):
    """Verifies GEMV accepts 2D column vector (K, 1) and flattens return to 1D (M,)."""
    np.random.seed(43)
    m, k = 16, 8
    a = np.random.randn(m, k).astype(np.float32)
    v = np.random.randn(k, 1).astype(np.float32)

    res = engine.gemv(a, v, dtype="fp32")

    assert res.ndim == 1
    assert res.shape == (m,)
    assert res.dtype == np.float32

    expected = np.matmul(a, v.squeeze())
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-6)


# ==============================================================================
# 2. Precision & Emulation Tests (FP32, FP16, BF16)
# ==============================================================================

def test_gemv_precision_fp16(engine):
    """Verifies GEMV execution in FP16 precision."""
    np.random.seed(44)
    m, k = 10, 10
    a = np.random.randn(m, k).astype(np.float16)
    v = np.random.randn(k).astype(np.float16)

    res = engine.gemv(a, v, dtype="fp16")

    assert res.shape == (m,)
    assert res.dtype == np.float16

    expected = np.matmul(a.astype(np.float64), v.astype(np.float64)).astype(np.float16)
    np.testing.assert_allclose(res, expected, rtol=1e-3, atol=1e-3)


def test_gemv_precision_bf16_masking(engine):
    """Verifies BF16 emulation: mantissa truncation to 7 bits (0xFFFF0000 mask)."""
    np.random.seed(45)
    m, k = 8, 8
    a = np.random.randn(m, k).astype(np.float32)
    v = np.random.randn(k).astype(np.float32)

    res = engine.gemv(a, v, dtype="bf16")

    assert res.shape == (m,)
    assert res.dtype == np.float32

    # Verify that the lower 16 bits of each float32 element are exactly zero
    u32_view = res.view(np.uint32)
    lower_16_bits = u32_view & np.uint32(0x0000FFFF)
    assert np.all(lower_16_bits == 0), "BF16 result contains unmasked lower mantissa bits"


# ==============================================================================
# 3. Higham Error Bound Equivalence Tests
# ==============================================================================

@pytest.mark.parametrize("dtype,u", [
    ("fp32", 2.0 ** -24),
    ("fp16", 2.0 ** -11),
    ("bf16", 2.0 ** -8),
])
def test_gemv_higham_error_bounds(engine, dtype, u):
    """
    Verifies that GEMV numerical error strictly adheres to Higham's bound:
    ||y_hat - y_ref||_inf <= gamma_K * ||A||_inf * ||v||_inf
    where gamma_K = (K * u) / (1 - K * u).
    """
    np.random.seed(100)
    m, k = 64, 64

    # Generate random matrix and vector scaled to avoid floating-point overflow
    a = (np.random.randn(m, k) * 0.1).astype(np.float32)
    v = (np.random.randn(k) * 0.1).astype(np.float32)

    if dtype == "fp16":
        a = a.astype(np.float16)
        v = v.astype(np.float16)

    # Compute engine result
    y_hat = engine.gemv(a, v, dtype=dtype)

    # Compute high-precision FP64 reference
    a_f64 = a.astype(np.float64)
    v_f64 = v.astype(np.float64)
    y_ref = np.matmul(a_f64, v_f64)

    # Calculate Higham bound
    ku = float(k) * u
    gamma_k = (ku / (1.0 - ku)) if ku < 0.999 else (ku * 1000.0)
    norm_a = float(np.linalg.norm(a_f64, ord=np.inf))
    norm_v = float(np.linalg.norm(v_f64, ord=np.inf))
    theoretical_bound = gamma_k * norm_a * norm_v
    effective_bound = max(theoretical_bound, u * float(np.linalg.norm(y_ref, ord=np.inf)), 1e-7)

    # Evaluate discrepancy
    discrepancy = float(np.max(np.abs(y_hat.astype(np.float64) - y_ref)))
    assert discrepancy <= effective_bound, (
        f"GEMV error ({discrepancy:.6e}) exceeded Higham bound ({effective_bound:.6e}) for {dtype}"
    )


def test_higham_vector_bound_method(engine):
    """Verifies ComputeEngine.higham_vector_bound calculation."""
    bound = engine.higham_vector_bound(64, 2.5, 1.2, dtype="fp32")
    expected = ComputeEngine.higham_matrix_bound(64, 2.5, 1.2, dtype="fp32")
    assert bound == expected


# ==============================================================================
# 4. Rectangular, Odd, Prime, and Scalar Dimensions
# ==============================================================================

@pytest.mark.parametrize("m,k", [
    (1, 1),       # 1x1 scalar
    (1, 32),      # Single row
    (32, 1),      # Single column
    (64, 16),     # Tall rectangular
    (16, 64),     # Wide rectangular
    (7, 13),      # Small prime / odd
    (17, 11),     # Asymmetric prime
    (31, 29),     # Medium prime
    (127, 63),    # Odd boundary
])
def test_gemv_varied_shapes(engine, m, k):
    """Verifies GEMV across rectangular, prime, odd, and scalar dimensions."""
    np.random.seed(m * 100 + k)
    a = np.random.randn(m, k).astype(np.float32)
    v = np.random.randn(k).astype(np.float32)

    res = engine.gemv(a, v, dtype="fp32")

    assert res.shape == (m,)
    expected = np.matmul(a, v)
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-6)


# ==============================================================================
# 5. Numerical Edge Cases & Invariant Properties
# ==============================================================================

def test_gemv_identity_matrix(engine):
    """Verifies I * v == v."""
    k = 16
    eye = np.eye(k, dtype=np.float32)
    v = np.random.randn(k).astype(np.float32)

    res = engine.gemv(eye, v, dtype="fp32")
    np.testing.assert_allclose(res, v, rtol=1e-7, atol=1e-7)


def test_gemv_zero_matrix_and_vector(engine):
    """Verifies 0 * v == 0 and A * 0 == 0."""
    m, k = 8, 8
    zeros_a = np.zeros((m, k), dtype=np.float32)
    v = np.random.randn(k).astype(np.float32)
    res_a = engine.gemv(zeros_a, v, dtype="fp32")
    assert np.all(res_a == 0.0)

    a = np.random.randn(m, k).astype(np.float32)
    zeros_v = np.zeros(k, dtype=np.float32)
    res_v = engine.gemv(a, zeros_v, dtype="fp32")
    assert np.all(res_v == 0.0)


def test_gemv_non_contiguous_input(engine):
    """Verifies GEMV handles Fortran-order and strided slices seamlessly."""
    a_f = np.asfortranarray(np.random.randn(8, 8).astype(np.float32))
    v_full = np.random.randn(16).astype(np.float32)
    v_strided = v_full[::2]  # strided 1D slice of length 8

    res = engine.gemv(a_f, v_strided, dtype="fp32")
    assert res.shape == (8,)
    expected = np.matmul(a_f, v_strided)
    np.testing.assert_allclose(res, expected, rtol=1e-5)


# ==============================================================================
# 6. Error Conditions & Invalid Argument Rejections
# ==============================================================================

def test_gemv_inner_dimension_mismatch(engine):
    """Verifies ValueError when A cols != v length."""
    a = np.ones((4, 6), dtype=np.float32)
    v = np.ones(5, dtype=np.float32)

    with pytest.raises(ValueError, match="dimension mismatch|Inner dimension"):
        engine.gemv(a, v, dtype="fp32")


def test_gemv_invalid_matrix_ndim(engine):
    """Verifies ValueError when matrix A is not 2-dimensional."""
    # 1D matrix
    with pytest.raises(ValueError, match="2-dimensional"):
        engine.gemv(np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))

    # 3D tensor
    with pytest.raises(ValueError, match="2-dimensional"):
        engine.gemv(np.ones((2, 4, 4), dtype=np.float32), np.ones(4, dtype=np.float32))


def test_gemv_invalid_vector_ndim(engine):
    """Verifies ValueError when vector v is multi-column or 3D."""
    a = np.ones((4, 4), dtype=np.float32)

    # 2D matrix with cols != 1
    with pytest.raises(ValueError, match="1D or 2D column vector"):
        engine.gemv(a, np.ones((4, 2), dtype=np.float32))

    # 3D tensor
    with pytest.raises(ValueError, match="1D or 2D column vector"):
        engine.gemv(a, np.ones((4, 1, 1), dtype=np.float32))


def test_gemv_unsupported_dtype(engine):
    """Verifies ValueError on unsupported data types."""
    a = np.ones((4, 4), dtype=np.float32)
    v = np.ones(4, dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported dtype"):
        engine.gemv(a, v, dtype="int64")
