# matrix-compute-node

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matrix-compute/matrix-compute-colab/blob/main/01_quickstart_distributed_matmul.ipynb)

Decentralized worker daemon for the Distributed Matrix Multiplication network.

## Overview
`matrix-compute-node` is a high-performance, zero-trust compute daemon that:
- Discovers work and announces availability via Nostr relays (Kind 30000).
- Performs hardware attestation across NVML, `nvidia-smi`, and CPU fallbacks.
- Enforces a strict 5-step L402 pre-allocation payment gate to prevent DoS attacks.
- Executes distributed block GEMM matrix multiplications ($C = A \times B$) with hardware acceleration (CuPy / PyTorch / NumPy).
- Computes SHA256 result digests and cryptographically signs results using Secp256k1.
- Services tasks streamed directly via gRPC.

## Architecture
- `src/config/`: Central configuration management (`NodeConfig`) supporting CLI flags, environment variables (`MATRIX_*`), and defaults.
- `src/crypto/`: Secp256k1 key management, signing, and verification (`WorkerSigner`).
- `src/attestation/`: Hardware attestation engine (NVML, `nvidia-smi`, CPU).
- `src/relay/`: Nostr discovery relay client (Kind 30000).
- `src/auth/`: Strict pre-allocation L402 payment gate.
- `src/engine/`: Multi-backend GEMM acceleration core.
- `src/p2p/`: P2P gRPC compute service (`MatrixComputeService`).
- `src/test_doubles/`: Hermetic in-memory test doubles (`MockSettlementService`, `MockNostrRelay`).

## Quickstart

```bash
# Install package in editable mode
pip install -e .

# Run daemon
matrix-compute-node --p2p-port 50051

# Run unit tests
pytest tests/unit
```
