"""Math Acceleration Compute Engine for matrix-compute-node."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try importing hardware acceleration backends
_TORCH_AVAILABLE = False
_CUDA_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

_CUPY_AVAILABLE = False
try:
    import cupy
    _CUPY_AVAILABLE = True
except ImportError:
    pass


class ComputeEngine:
    """
    High-performance block matrix multiplication (GEMM) acceleration engine.
    Supports PyTorch (CUDA / CPU), CuPy (CUDA), and NumPy fallback.
    """

    def __init__(self, prefer_gpu: bool = True, force_cpu: bool = False) -> None:
        self.prefer_gpu = prefer_gpu and not force_cpu
        self.backend = self._determine_backend()

    def _determine_backend(self) -> str:
        if self.prefer_gpu and _TORCH_AVAILABLE and _CUDA_AVAILABLE:
            return "torch_cuda"
        if self.prefer_gpu and _CUPY_AVAILABLE:
            return "cupy"
        if _TORCH_AVAILABLE and not self.prefer_gpu:
            return "torch_cpu"
        return "numpy"

    def gemm(self, a: np.ndarray, b: np.ndarray, dtype: str = "fp32") -> np.ndarray:
        """
        Calculates C = A x B for the specified precision datatype.
        Supported dtypes: 'fp32', 'fp16', 'bf16'.
        """
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Matrices must be 2-dimensional. Got shapes {a.shape} and {b.shape}")

        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Inner dimension mismatch: A columns ({a.shape[1]}) != B rows ({b.shape[0]})"
            )

        norm_dtype = dtype.lower().strip()
        if norm_dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"Unsupported dtype '{dtype}'. Supported: 'fp32', 'fp16', 'bf16'")

        # Execute using determined backend
        if self.backend == "torch_cuda" and _TORCH_AVAILABLE:
            try:
                torch_dtype = torch.float32
                if norm_dtype == "fp16":
                    torch_dtype = torch.float16
                elif norm_dtype == "bf16":
                    torch_dtype = torch.bfloat16

                t_a = torch.from_numpy(np.ascontiguousarray(a)).to(device="cuda", dtype=torch_dtype)
                t_b = torch.from_numpy(np.ascontiguousarray(b)).to(device="cuda", dtype=torch_dtype)
                t_c = torch.matmul(t_a, t_b)
                if norm_dtype == "bf16":
                    return ComputeEngine.truncate_to_bf16(t_c.to(torch.float32).cpu().numpy())
                c = t_c.cpu().numpy()
                return c
            except Exception as ex:
                logger.warning(f"CUDA PyTorch GEMM failed, falling back to NumPy: {ex}")

        if self.backend == "cupy" and _CUPY_AVAILABLE:
            try:
                cp_a = cupy.asarray(a)
                cp_b = cupy.asarray(b)
                if norm_dtype == "bf16":
                    cp_a = cupy.asarray(ComputeEngine.truncate_to_bf16(a), dtype=cupy.float32)
                    cp_b = cupy.asarray(ComputeEngine.truncate_to_bf16(b), dtype=cupy.float32)
                    cp_c = cupy.matmul(cp_a, cp_b)
                    return ComputeEngine.truncate_to_bf16(cupy.asnumpy(cp_c))
                elif norm_dtype == "fp16":
                    cp_c = cupy.matmul(cp_a.astype(cupy.float16), cp_b.astype(cupy.float16))
                    return cupy.asnumpy(cp_c)
                else:
                    cp_c = cupy.matmul(cp_a.astype(cupy.float32), cp_b.astype(cupy.float32))
                    return cupy.asnumpy(cp_c)
            except Exception as ex:
                logger.warning(f"CuPy GEMM failed, falling back to NumPy: {ex}")

        # NumPy CPU Fallback
        if norm_dtype == "bf16":
            # True BF16 emulation via 16-bit float32 mantissa masking
            a_bf16 = ComputeEngine.truncate_to_bf16(a)
            b_bf16 = ComputeEngine.truncate_to_bf16(b)
            c = np.matmul(a_bf16.astype(np.float64), b_bf16.astype(np.float64)).astype(np.float32)
            return ComputeEngine.truncate_to_bf16(c)

        a_f64 = a.astype(np.float64)
        b_f64 = b.astype(np.float64)
        c_f64 = np.matmul(a_f64, b_f64)

        if norm_dtype == "fp32":
            return c_f64.astype(np.float32)
        elif norm_dtype == "fp16":
            return c_f64.astype(np.float16)

        return c_f64.astype(np.float32)

    @staticmethod
    def truncate_to_bf16(tensor: np.ndarray) -> np.ndarray:
        """
        Truncates / masks float32 mantissa to 7 bits (IEEE bfloat16 emulation).
        Clears the lowest 16 bits of 32-bit float representation.
        Guarantees contiguous memory layout, preserves NaN/Inf semantics, signed zeros,
        and subnormal underflow characteristics.
        """
        f32 = np.ascontiguousarray(tensor, dtype=np.float32)
        u32 = f32.view(np.uint32)
        masked_u32 = u32 & np.uint32(0xFFFF0000)

        # Preserve NaN semantics: if input was NaN and lower-16-bit truncation cleared
        # the entire mantissa to 0 (which would otherwise falsely convert NaN to Infinity),
        # restore the quiet NaN bit (bit 22, 0x00400000).
        is_nan = np.isnan(f32)
        if np.any(is_nan):
            lost_mantissa = is_nan & ((masked_u32 & np.uint32(0x007F0000)) == 0)
            if np.any(lost_mantissa):
                masked_u32 = masked_u32.copy()
                masked_u32[lost_mantissa] |= np.uint32(0x00400000)

        return masked_u32.view(np.float32)

    @staticmethod
    def get_unit_roundoff(dtype: str = "fp32") -> float:
        """Returns IEEE 754 unit roundoff u for given precision."""
        dt = str(dtype).lower().strip()
        if dt in ("fp16", "float16"):
            return 2.0 ** -11  # ~4.8828e-4
        elif dt in ("bf16", "bfloat16"):
            return 2.0 ** -8   # ~3.90625e-3
        elif dt in ("fp64", "float64"):
            return 2.0 ** -53  # ~1.1102e-16
        else:
            return 2.0 ** -24  # ~5.9605e-8 (fp32)

    @staticmethod
    def higham_matrix_bound(k: int, norm_a: float, norm_b: float, dtype: str = "fp32") -> float:
        """
        Higham's matrix multiplication rounding bound:
        gamma_K = (K * u) / (1 - K * u)
        bound = gamma_K * ||A||_inf * ||B||_inf
        """
        u = ComputeEngine.get_unit_roundoff(dtype)
        ku = float(k) * u
        gamma_k = (ku / (1.0 - ku)) if ku < 0.999 else (ku * 1000.0)
        return gamma_k * float(norm_a) * float(norm_b)

    @staticmethod
    def compute_sha256(tensor: np.ndarray) -> str:
        """Computes SHA256 hex digest over raw contiguous bytes of tensor."""
        if not tensor.flags["C_CONTIGUOUS"]:
            tensor = np.ascontiguousarray(tensor)
        return hashlib.sha256(tensor.tobytes()).hexdigest()

    @staticmethod
    def verify_freivalds(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        epsilon: Optional[float] = None,
        num_trials: int = 5,
        dtype: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """
        Randomized Freivalds algorithm verification with Higham's dynamic scale-aware bound:
        Computes gamma_K = (K * u) / (1 - K * u) scaled by ||A||_inf * ||B||_inf.
        Evaluates both absolute discrepancy and relative discrepancy to defend against
        false positives and zero-work fraud on small-magnitude inputs.
        """
        # Validate matrix shapes
        if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
            return False, float("inf")
        if a.shape[1] != b.shape[0] or c.shape != (a.shape[0], b.shape[1]):
            return False, float("inf")

        # Guard against NaN/Inf in inputs or outputs (fraudulent or corrupted computation)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
            return False, float("inf")

        n = b.shape[1]
        k = a.shape[1]

        # Determine precision datatype
        if dtype is None:
            if a.dtype == np.float16 or c.dtype == np.float16:
                dtype = "fp16"
            elif a.dtype == np.float64 or c.dtype == np.float64:
                dtype = "fp64"
            else:
                dtype = "fp32"

        u = ComputeEngine.get_unit_roundoff(dtype)
        ku = float(k) * u
        gamma_k = (ku / (1.0 - ku)) if ku < 0.999 else (ku * 1000.0)

        # Calculate input infinity norms (maximum absolute row sum)
        norm_a = float(np.linalg.norm(a, ord=np.inf))
        norm_b = float(np.linalg.norm(b, ord=np.inf))
        scale = norm_a * norm_b

        # Dynamic scale-aware error bound
        dynamic_bound = gamma_k * scale

        max_discrepancy = 0.0

        for _ in range(num_trials):
            r = np.random.randint(0, 2, size=(n, 1)).astype(np.float64)
            if not np.any(r):
                r[np.random.randint(0, n), 0] = 1.0
            br = np.matmul(b.astype(np.float64), r)
            abr = np.matmul(a.astype(np.float64), br)
            cr = np.matmul(c.astype(np.float64), r)
            diff = float(np.max(np.abs(abr - cr)))
            if np.isnan(diff) or np.isinf(diff):
                return False, float("inf")
            if diff > max_discrepancy:
                max_discrepancy = diff

            # Check relative discrepancy against gamma_k to catch zero-work fraud on small inputs
            abr_norm = float(np.max(np.abs(abr)))
            cr_norm = float(np.max(np.abs(cr)))
            denom = max(scale, abr_norm, cr_norm)

            # If inputs are non-zero:
            if denom > 0.0:
                rel_diff = diff / denom
                # Relative discrepancy check prevents zero-work fraud
                allowed_rel = max(gamma_k, epsilon if epsilon is not None else 0.0, 1e-7)
                if rel_diff > allowed_rel:
                    return False, max_discrepancy

                # Absolute bound check scaled by input norms (never tighter than physical rounding bound)
                effective_bound = dynamic_bound
                if epsilon is not None and scale >= 1.0:
                    effective_bound = max(dynamic_bound, epsilon)
                elif epsilon is not None:
                    effective_bound = max(dynamic_bound, epsilon * scale)

                if diff > effective_bound:
                    return False, max_discrepancy
            else:
                # Scale is zero (A=0 or B=0) -> A*B is 0. C must be near 0.
                if diff > 1e-7:
                    return False, max_discrepancy

        return True, max_discrepancy
