"""
Pytest configuration and shared fixtures for matrix-compute-node test suite.
Configures sys.path to ensure matrix-contracts and matrix-compute-node/src are resolvable.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure matrix-contracts, matrix-compute-sdk, and matrix-compute-node are on sys.path
TESTS_DIR = Path(__file__).resolve().parent
NODE_DIR = TESTS_DIR.parent
ROOT_DIR = NODE_DIR.parent
CONTRACTS_DIR = ROOT_DIR / "matrix-contracts"
RELAY_DIR = ROOT_DIR / "matrix-compute-relay" / "src"
SDK_DIR = ROOT_DIR / "matrix-compute-sdk" / "src"
SRC_DIR = NODE_DIR / "src"

for p in [str(CONTRACTS_DIR), str(RELAY_DIR), str(SDK_DIR), str(NODE_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import contracts and internal modules
import src  # noqa: F401
from src.config.settings import NodeConfig
from src.crypto.signer import WorkerSigner
from src.test_doubles.mock_settlement import MockSettlementService
from src.test_doubles.mock_nostr_relay import MockNostrRelay
from matrix_contracts.schemas.discovery import (
    HardwareSpecs,
    PricingSpecs,
    BondStatus,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
    WorkerQueryResponse,
    MatchedWorker,
)
from matrix_contracts.schemas.settlement import (
    CreateHodlInvoiceRequest,
    CreateHodlInvoiceResponse,
    SettlePaymentRequest,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Isolates environment variables from the host system during configuration tests."""
    env_keys_to_clean = [
        "MATRIX_WORKER_PRIVKEY",
        "MATRIX_WORKER_PRIVKEY_HEX",
        "WORKER_PRIVKEY",
        "MATRIX_P2P_HOST",
        "P2P_HOST",
        "MATRIX_P2P_PORT",
        "P2P_PORT",
        "MATRIX_P2P_MULTIADDR",
        "P2P_MULTIADDR",
        "MATRIX_LIGHTNING_ROUTING_PUBKEY",
        "LIGHTNING_ROUTING_PUBKEY",
        "MATRIX_NOSTR_RELAYS",
        "NOSTR_RELAYS",
        "MATRIX_SETTLEMENT_API_URL",
        "MATRIX_SETTLEMENT_URL",
        "SETTLEMENT_API_URL",
        "SETTLEMENT_URL",
        "MATRIX_SETTLEMENT_VERIFICATION_KEY",
        "SETTLEMENT_VERIFICATION_KEY",
        "MATRIX_SATS_PER_GFLOP",
        "SATS_PER_GFLOP",
        "MATRIX_MIN_TASK_SATS",
        "MIN_TASK_SATS",
        "MATRIX_GPU_DEVICE_INDEX",
        "GPU_DEVICE_INDEX",
        "MATRIX_DTYPE_PREFERENCE",
        "DTYPE_PREFERENCE",
        "MATRIX_LOG_LEVEL",
        "LOG_LEVEL",
    ]
    for key in env_keys_to_clean:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_settlement():
    """Provides a fresh, isolated MockSettlementService instance per test."""
    service = MockSettlementService(verification_key="test-settlement-secret-key-12345")
    yield service
    service.reset()


@pytest.fixture
def mock_relay():
    """Provides a fresh, isolated MockNostrRelay instance per test."""
    relay = MockNostrRelay()
    yield relay
    relay.clear()


@pytest.fixture
def sample_signer():
    """Deterministic WorkerSigner fixture with a known private key."""
    privkey_hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    return WorkerSigner(privkey_hex=privkey_hex)


@pytest.fixture
def sample_hardware():
    return HardwareSpecs(
        gpu_model="NVIDIA RTX 4090",
        vram_mb=24576,
        peak_fp32_tflops=82.6,
        attestation_proof="mock_attestation_proof_blob_secp256k1",
    )


@pytest.fixture
def sample_pricing():
    return PricingSpecs(sats_per_gflop=0.0005, min_task_sats=10)


@pytest.fixture
def sample_bond():
    return BondStatus(staked_amount_sats=100000, bond_txid="mock_txid_e3b0c44298fc")


@pytest.fixture
def sample_announce(sample_signer, sample_hardware, sample_pricing, sample_bond):
    return WorkerAnnouncePayload(
        event="WORKER_ANNOUNCE",
        worker_pubkey=sample_signer.get_nostr_pubkey(),
        p2p_multiaddr="/ip4/127.0.0.1/tcp/50051",
        lightning_routing_pubkey=sample_signer.get_pubkey_hex(),
        hardware=sample_hardware,
        pricing=sample_pricing,
        bond_status=sample_bond,
        timestamp=1788537600,
    )


@pytest.fixture(scope="session")
def sample_tensors():
    """Generates deterministic sample float32 matrices A (4x4) and B (4x4) with reference product C."""
    np.random.seed(42)
    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0]
    ], dtype=np.float32)

    b = np.array([
        [7.0, 6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0]
    ], dtype=np.float32)

    c_expected = np.matmul(a, b)
    return {
        "m": 4, "k": 4, "n": 4,
        "a": a, "b": b,
        "c_expected": c_expected,
        "a_bytes": a.tobytes(),
        "b_bytes": b.tobytes(),
        "c_bytes_expected": c_expected.tobytes()
    }
