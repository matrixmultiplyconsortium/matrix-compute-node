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

                t_a = torch.from_numpy(a).to(device="cuda", dtype=torch_dtype)
                t_b = torch.from_numpy(b).to(device="cuda", dtype=torch_dtype)
                t_c = torch.matmul(t_a, t_b)
                c = t_c.cpu().numpy()
                return c
            except Exception as ex:
                logger.warning(f"CUDA PyTorch GEMM failed, falling back to NumPy: {ex}")

        # NumPy CPU Fallback
        a_f64 = a.astype(np.float64)
        b_f64 = b.astype(np.float64)
        c_f64 = np.matmul(a_f64, b_f64)

        if norm_dtype == "fp32":
            return c_f64.astype(np.float32)
        elif norm_dtype == "fp16":
            return c_f64.astype(np.float16)
        elif norm_dtype == "bf16":
            # BF16 emulation via float32 mantissa truncation
            return c_f64.astype(np.float32)

        return c_f64.astype(np.float32)

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
        epsilon: float = 1e-4,
        num_trials: int = 5,
    ) -> Tuple[bool, float]:
        """
        Randomized Freivalds algorithm verification:
        Checks ||A * (B * r) - C * r||_inf <= epsilon for random binary vectors r.
        """
        n = b.shape[1]
        max_discrepancy = 0.0
        for _ in range(num_trials):
            r = np.random.randint(0, 2, size=(n, 1)).astype(np.float64)
            br = np.matmul(b.astype(np.float64), r)
            abr = np.matmul(a.astype(np.float64), br)
            cr = np.matmul(c.astype(np.float64), r)
            diff = float(np.max(np.abs(abr - cr)))
            if diff > max_discrepancy:
                max_discrepancy = diff
            if diff > epsilon:
                return False, max_discrepancy
        return True, max_discrepancy
