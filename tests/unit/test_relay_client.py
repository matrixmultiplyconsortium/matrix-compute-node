"""Unit tests for NostrRelayClient."""

import time
import pytest
from matrix_contracts.schemas.discovery import (
    BondStatus,
    HardwareSpecs,
    PricingSpecs,
    WorkerQueryRequest,
)
from src.crypto.signer import WorkerSigner
from src.relay.nostr_client import NostrRelayClient, RelayConnectionError
from src.test_doubles.mock_nostr_relay import MockNostrRelay


@pytest.fixture
def signer():
    return WorkerSigner()


@pytest.fixture
def mock_relay():
    return MockNostrRelay()


@pytest.fixture
def hardware_specs():
    return HardwareSpecs(
        gpu_model="NVIDIA RTX 4090",
        vram_mb=24576,
        peak_fp32_tflops=82.6,
        attestation_proof="mock_attestation_proof_hex",
    )


@pytest.fixture
def pricing_specs():
    return PricingSpecs(sats_per_gflop=0.0005, min_task_sats=10)


@pytest.fixture
def bond_status():
    return BondStatus(staked_amount_sats=100000, bond_txid="mock_bond_txid_123")


def test_nostr_client_init_defaults():
    client = NostrRelayClient()
    assert client.ANNOUNCE_KIND == 30000
    assert len(client.relays) > 0
    assert client.is_connected is False


def test_nostr_client_connect_mock_relay(mock_relay):
    client = NostrRelayClient(relays=[mock_relay])
    assert client.connect() is True
    assert client.is_connected is True
    client.disconnect()
    assert client.is_connected is False


def test_nostr_client_connect_unreachable_fails():
    client = NostrRelayClient(relays=["ws://127.0.0.1:1"])
    with pytest.raises(RelayConnectionError):
        client.connect()


def test_nostr_client_connect_invalid_scheme():
    client = NostrRelayClient(relays=["ftp://example.com/relay"])
    with pytest.raises(RelayConnectionError):
        client.connect()


def test_build_announce_event(signer, hardware_specs, pricing_specs, bond_status):
    client = NostrRelayClient(signer=signer)
    event = client.build_announce_event(
        signer=signer,
        hardware_specs=hardware_specs,
        pricing_specs=pricing_specs,
        bond_status=bond_status,
        p2p_multiaddr="/ip4/127.0.0.1/tcp/50051",
        lightning_pubkey="02e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        d_tag="matrix-worker-unit-test",
    )

    assert event["kind"] == 30000
    assert event["pubkey"] == signer.get_pubkey_hex()
    assert "id" in event and len(event["id"]) == 64
    assert "sig" in event and len(event["sig"]) > 0

    tag_map = {t[0]: t[1] for t in event["tags"]}
    assert tag_map["d"] == "matrix-worker-unit-test"
    assert tag_map["p2p"] == "/ip4/127.0.0.1/tcp/50051"
    assert tag_map["vram_mb"] == "24576"
    assert float(tag_map["sats_per_gflop"]) == 0.0005
    assert tag_map["gpu_model"] == "NVIDIA RTX 4090"


def test_publish_and_query_with_mock_relay(signer, mock_relay, hardware_specs, pricing_specs, bond_status):
    client = NostrRelayClient(relays=[mock_relay], signer=signer)
    client.connect()

    event = client.build_announce_event(
        signer=signer,
        hardware_specs=hardware_specs,
        pricing_specs=pricing_specs,
        bond_status=bond_status,
        p2p_multiaddr="/ip4/127.0.0.1/tcp/50051",
        lightning_pubkey="02e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    published = client.publish_announce(event)
    assert published is True

    # Query matching workers
    query = WorkerQueryRequest(min_vram_mb=16384, max_price_sats_per_gflop=0.001)
    response = client.query_workers(query)
    assert response.status == "SUCCESS"
    assert len(response.matched_workers) == 1
    assert response.matched_workers[0].worker_pubkey == signer.get_pubkey_hex()
