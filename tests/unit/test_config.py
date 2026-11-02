"""Unit tests for NodeConfig settings management."""

import pytest
from pydantic import ValidationError

from src.config.settings import NodeConfig


def test_default_config(clean_env):
    """TC-CFG-01: Verifies standard defaults match architectural specifications."""
    cfg = NodeConfig()
    assert cfg.p2p_host == "0.0.0.0"
    assert cfg.p2p_port == 50051
    assert cfg.sats_per_gflop == 0.0005
    assert cfg.min_task_sats == 10
    assert cfg.dtype_preference == "fp32"
    assert cfg.settlement_api_url == "https://settlement.matrix-compute.net"
    assert cfg.settlement_verification_key == "settlement_secret_key_dev"
    assert len(cfg.worker_privkey_hex) == 64
    assert cfg.nostr_relays == ["wss://relay.matrix-compute.net"]


def test_effective_p2p_multiaddr_default(clean_env):
    """TC-CFG-02: Verifies fallback multiaddr derivation when p2p_multiaddr is None."""
    cfg = NodeConfig(p2p_host="192.168.1.50", p2p_port=50052)
    assert cfg.effective_p2p_multiaddr == "/ip4/192.168.1.50/tcp/50052"


def test_effective_p2p_multiaddr_explicit(clean_env):
    """TC-CFG-03: Verifies explicit p2p_multiaddr takes precedence over host/port derivation."""
    custom_addr = "/dns4/worker.compute.net/tcp/9000/p2p/12D3KooW..."
    cfg = NodeConfig(p2p_multiaddr=custom_addr)
    assert cfg.effective_p2p_multiaddr == custom_addr


def test_effective_lightning_pubkey(clean_env):
    """TC-CFG-04: Verifies lightning pubkey resolution with fallback to worker pubkey."""
    worker_pubkey = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
    cfg = NodeConfig()
    assert cfg.effective_lightning_pubkey(worker_pubkey) == worker_pubkey

    explicit_ln = "03abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    cfg_explicit = NodeConfig(lightning_routing_pubkey=explicit_ln)
    assert cfg_explicit.effective_lightning_pubkey(worker_pubkey) == explicit_ln


def test_env_variable_parsing(monkeypatch, clean_env):
    """TC-CFG-05: Verifies MATRIX_* environment variables override defaults."""
    monkeypatch.setenv("MATRIX_P2P_PORT", "50099")
    monkeypatch.setenv("MATRIX_SATS_PER_GFLOP", "0.0015")
    monkeypatch.setenv("MATRIX_NOSTR_RELAYS", "ws://relay1.local:8008,ws://relay2.local:8008")
    monkeypatch.setenv("MATRIX_DTYPE_PREFERENCE", "fp16")

    cfg = NodeConfig()
    assert cfg.p2p_port == 50099
    assert cfg.sats_per_gflop == 0.0015
    assert cfg.nostr_relays == ["ws://relay1.local:8008", "ws://relay2.local:8008"]
    assert cfg.dtype_preference == "fp16"


def test_unprefixed_env_variable_parsing(monkeypatch, clean_env):
    """TC-CFG-06: Verifies unprefixed environment variables (P2P_PORT, SATS_PER_GFLOP)."""
    monkeypatch.setenv("P2P_PORT", "50088")
    monkeypatch.setenv("SATS_PER_GFLOP", "0.002")

    cfg = NodeConfig()
    assert cfg.p2p_port == 50088
    assert cfg.sats_per_gflop == 0.002


def test_cli_overrides_env_and_defaults(monkeypatch, clean_env):
    """TC-CFG-07: Verifies CLI arguments override environment variables and defaults."""
    monkeypatch.setenv("MATRIX_P2P_PORT", "50055")
    monkeypatch.setenv("MATRIX_SATS_PER_GFLOP", "0.001")

    cfg = NodeConfig.load(args=["--p2p-port", "50060", "--sats-per-gflop", "0.005"])
    assert cfg.p2p_port == 50060
    assert cfg.sats_per_gflop == 0.005


def test_port_validation_range(clean_env):
    """TC-CFG-08: Verifies port bounds validation (1 <= port <= 65535)."""
    with pytest.raises(ValidationError):
        NodeConfig(p2p_port=0)

    with pytest.raises(ValidationError):
        NodeConfig(p2p_port=-1)

    with pytest.raises(ValidationError):
        NodeConfig(p2p_port=70000)

    valid_cfg = NodeConfig(p2p_port=8080)
    assert valid_cfg.p2p_port == 8080


def test_dtype_validation(clean_env):
    """TC-CFG-09: Verifies compute precision format validation."""
    for dt in ["fp32", "fp16", "bf16", "FP32", "FP16", "BF16"]:
        cfg = NodeConfig(dtype_preference=dt)
        assert cfg.dtype_preference == dt.lower()

    with pytest.raises(ValidationError):
        NodeConfig(dtype_preference="float64")

    with pytest.raises(ValidationError):
        NodeConfig(dtype_preference="int32")


def test_invalid_privkey_length(clean_env):
    """TC-CFG-10: Verifies 64-hex character validation for private key."""
    with pytest.raises(ValidationError):
        NodeConfig(worker_privkey_hex="0123456789abcdef")  # Too short

    with pytest.raises(ValidationError):
        NodeConfig(worker_privkey_hex="z" * 64)  # Invalid hex


def test_non_negative_pricing(clean_env):
    """TC-CFG-11: Verifies non-negative pricing validation."""
    with pytest.raises(ValidationError):
        NodeConfig(sats_per_gflop=-0.0001)

    with pytest.raises(ValidationError):
        NodeConfig(min_task_sats=-5)

    zero_cfg = NodeConfig(sats_per_gflop=0.0, min_task_sats=0)
    assert zero_cfg.sats_per_gflop == 0.0
    assert zero_cfg.min_task_sats == 0


def test_malformed_settlement_api_url(clean_env):
    """TC-CFG-12: Verifies settlement API URL validation."""
    with pytest.raises(ValidationError):
        NodeConfig(settlement_api_url="not_a_valid_url")

    with pytest.raises(ValidationError):
        NodeConfig(settlement_api_url="ftp://invalid.scheme.com")

    valid_cfg = NodeConfig(settlement_api_url="https://settlement.matrix.net:8443")
    assert valid_cfg.settlement_api_url == "https://settlement.matrix.net:8443"


def test_nostr_relays_json_format(clean_env):
    """TC-CFG-13: Verifies JSON array parsing for nostr_relays."""
    cfg = NodeConfig.load(args=["--nostr-relays", "wss://r1.example.com,wss://r2.example.com"])
    assert cfg.nostr_relays == ["wss://r1.example.com", "wss://r2.example.com"]
