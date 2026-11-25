"""
matrix-compute-node: Decentralized worker daemon for Distributed Matrix Multiplication.
Handles hardware attestation, Nostr discovery, L402 payment gate, and distributed GEMM compute.
"""

import sys
from pathlib import Path


def _ensure_contracts_linked() -> None:
    """
    Ensures matrix_contracts is importable.
    If not installed in current environment, dynamically searches sibling workspace directories
    and adds the matrix-contracts root to sys.path.
    """
    try:
        import matrix_contracts  # type: ignore # noqa: F401
    except ImportError:
        this_file = Path(__file__).resolve()
        # Candidate 1: Direct parent sibling (../matrix-contracts from matrix-compute-node)
        candidate_parent = this_file.parent.parent.parent / "matrix-contracts"
        # Candidate 2: Two levels up (workspace root)
        candidate_sibling = this_file.parent.parent / "matrix-contracts"

        resolved = False
        for candidate in (candidate_sibling, candidate_parent):
            if candidate.exists() and (candidate / "matrix_contracts").is_dir():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                proto_dir = candidate / "matrix_contracts" / "proto"
                if proto_dir.exists() and str(proto_dir) not in sys.path:
                    sys.path.insert(1, str(proto_dir))
                resolved = True
                break

        if not resolved:
            raise ImportError(
                "Unable to locate 'matrix_contracts'. Ensure 'matrix-contracts' is installed "
                "via 'pip install -e ../matrix-contracts' or located as a sibling directory in the workspace."
            )
        import matrix_contracts  # type: ignore # noqa: F401


# Link matrix_contracts immediately upon package initialization
_ensure_contracts_linked()

__version__ = "0.1.3"

__all__ = [
    "__version__",
]
