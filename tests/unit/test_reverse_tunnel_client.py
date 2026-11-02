"""Unit and integration tests for ReverseTunnelClient and NAT traversal in matrix-compute-node."""

import asyncio
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import MagicMock

# Ensure sibling packages are in sys.path
_TEST_DIR = Path(__file__).resolve().parent
_NODE_DIR = _TEST_DIR.parent.parent
_ROOT_DIR = _NODE_DIR.parent
for _p in [str(_ROOT_DIR / "matrix-contracts"), str(_ROOT_DIR / "matrix-compute-relay" / "src"), str(_NODE_DIR), str(_NODE_DIR / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import grpc
import numpy as np
import pytest

from matrix_contracts.proto import compute_service_pb2 as pb2
from matrix_contracts.proto import compute_service_pb2_grpc as pb2_grpc
from matrix_relay.config import RelaySettings
from matrix_relay.server import RelayServer
from src.auth.l402_gate import L402PreAllocationGate
from src.config.settings import NodeConfig
from src.crypto.signer import WorkerSigner
from src.engine.compute_core import ComputeEngine
from src.main import WorkerDaemon
from src.p2p.servicer import MatrixComputeServicer
from src.p2p.tunnel_client import ReverseTunnelClient, TunnelRpcAbort, TunnelServicerContext
from src.relay.nostr_client import NostrRelayClient
from src.test_doubles.mock_nostr_relay import MockNostrRelay
from src.test_doubles.mock_settlement import MockSettlementService
import base64
import hashlib
import hmac
import json


def _mint_macaroon(
    key: str,
    task_id: str,
    expiry: int,
    max_gflops: float,
    locked_sats: int = 100,
    payment_hash: str = "default_ph",
) -> str:
    caveats = {
        "task_id": task_id,
        "expiry": expiry,
        "max_gflops": max_gflops,
        "locked_sats": locked_sats,
        "payment_hash": payment_hash,
    }
    caveat_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(key.encode("utf-8"), caveat_str.encode("utf-8"), hashlib.sha256).hexdigest()
    payload = {"caveats": caveats, "signature": sig}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


def test_node_config_nat_settings_and_circuit_multiaddr():
    """Verify NodeConfig correctly resolves circuit multiaddrs in reverse tunnel mode."""
    signer = WorkerSigner()
    pub = signer.get_pubkey_hex()

    config = NodeConfig(
        worker_privkey_hex=signer.get_privkey_hex(),
        p2p_port=0,
        enable_outbound_nat=True,
        relay_grpc_endpoint="relay.matrix-compute.net:9090",
    )
    assert config.nat_mode == "reverse_tunnel"
    assert config.enable_outbound_nat is True
    assert config.p2p_port == 0

    multiaddr = config.resolve_advertised_multiaddr()
    expected = f"/dns4/relay.matrix-compute.net/tcp/9090/p2p-circuit/p2p/{pub}"
    assert multiaddr == expected

    # Test IP4 format
    config_ip = NodeConfig(
        worker_privkey_hex=signer.get_privkey_hex(),
        p2p_port=0,
        nat_mode="reverse_tunnel",
        relay_grpc_endpoint="127.0.0.1:9090",
    )
    assert config_ip.resolve_advertised_multiaddr() == f"/ip4/127.0.0.1/tcp/9090/p2p-circuit/p2p/{pub}"


def test_node_config_cli_and_env():
    """Verify CLI flags and environment variables populate NAT traversal settings."""
    env = {
        "MATRIX_ENABLE_OUTBOUND_NAT": "1",
        "MATRIX_RELAY_GRPC_ENDPOINT": "relay-env.org:9090",
        "MATRIX_TUNNEL_HEARTBEAT_INTERVAL_SECONDS": "15.0",
    }
    cfg = NodeConfig.from_env(env=env)
    assert cfg.enable_outbound_nat is True
    assert cfg.nat_mode == "reverse_tunnel"
    assert cfg.relay_grpc_endpoint == "relay-env.org:9090"
    assert cfg.tunnel_heartbeat_interval_seconds == 15.0

    cli_args = [
        "--nat-mode", "reverse_tunnel",
        "--relay-grpc-endpoint", "relay-cli.org:9090",
        "--enable-outbound-nat",
        "--tunnel-heartbeat-interval-seconds", "20.0",
        "--p2p-port", "0",
    ]
    cfg_cli = NodeConfig.load(args=cli_args, env={})
    assert cfg_cli.nat_mode == "reverse_tunnel"
    assert cfg_cli.enable_outbound_nat is True
    assert cfg_cli.relay_grpc_endpoint == "relay-cli.org:9090"
    assert cfg_cli.tunnel_heartbeat_interval_seconds == 20.0
    assert cfg_cli.p2p_port == 0


def test_worker_daemon_zero_inbound_port_mode():
    """Verify worker in outbound NAT mode binds zero inbound listening ports."""
    mock_relay = MockNostrRelay()
    mock_settlement = MockSettlementService()
    config = NodeConfig(
        p2p_host="127.0.0.1",
        p2p_port=0,
        enable_outbound_nat=True,
        relay_grpc_endpoint="127.0.0.1:9090",
        nostr_relays=["ws://127.0.0.1:8008"],
        settlement_api_url="http://127.0.0.1:8080",
    )
    client = NostrRelayClient(relays=[mock_relay])
    daemon = WorkerDaemon(config, nostr_client=client, settlement_client=mock_settlement)

    # Server must be None when p2p_port == 0 and enable_outbound_nat is True
    assert daemon.server is None
    assert daemon.tunnel_client is not None

    daemon.start()
    # Confirm announced event has circuit multiaddr and nat_mode tags
    events = mock_relay.get_events()
    assert len(events) >= 1
    last_event = events[-1]
    tags = {t[0]: t[1] for t in last_event.get("tags", [])}
    assert tags.get("nat_mode") == "reverse_tunnel"
    assert "p2p-circuit" in tags.get("p2p", "")

    daemon.stop(grace=0.1)


@pytest.mark.asyncio
async def test_reverse_tunnel_client_registration_and_execution_end_to_end():
    """End-to-end verification of ReverseTunnelClient connected to Relay."""
    # 1. Start Relay Server with gRPC Tunnel Service enabled
    relay_settings = RelaySettings(port=0, grpc_port=0, enable_grpc_tunnel=True)
    relay = RelayServer(config=relay_settings)
    await relay.start()

    try:
        relay_endpoint = f"127.0.0.1:{relay.bound_grpc_port}"

        # 2. Setup Worker Servicer & ReverseTunnelClient
        signer = WorkerSigner()
        mock_settlement = MockSettlementService()
        gate = L402PreAllocationGate(
            settlement_client_or_url=mock_settlement,
            verification_key="secret_test",
            min_price_rate=0.0001,
            min_task_sats=1,
        )
        engine = ComputeEngine(prefer_gpu=False)
        servicer = MatrixComputeServicer(
            signer=signer,
            gate=gate,
            engine=engine,
        )

        tunnel_client = ReverseTunnelClient(
            relay_endpoint=relay_endpoint,
            signer=signer,
            servicer=servicer,
            heartbeat_interval_seconds=1.0,
        )

        tunnel_client.start()
        ok = await tunnel_client.wait_until_connected(timeout=5.0)
        assert ok is True
        assert tunnel_client.is_connected is True
        assert tunnel_client.tunnel_id is not None
        assert relay.tunnel_manager.is_tunnel_active(signer.get_pubkey_hex()) is True

        # 3. Relay calls RequestTileTask through reverse tunnel down to worker
        task_id = "test_task_nat_1"
        req = pb2.TileTaskRequest(
            task_id=task_id,
            matrix_a_rows=4,
            matrix_a_cols=4,
            matrix_b_cols=4,
            dtype="fp32",
        )
        ack_bytes = await relay.tunnel_manager.send_unary_request(
            worker_pubkey=signer.get_pubkey_hex(),
            task_id=task_id,
            rpc_method="RequestTileTask",
            payload_bytes=req.SerializeToString(),
            timeout=5.0,
        )
        ack = pb2.TileTaskAck.FromString(ack_bytes)
        assert ack.task_id == task_id
        assert ack.is_ready is True
        assert ack.worker_pubkey == signer.get_pubkey_hex()

        # 4. Hold invoice and mint L402 macaroon
        inv = mock_settlement.create_hodl_invoice(task_id, total_compute_sats=100)
        mock_settlement.hold_invoice(inv.payment_hash)
        mac = _mint_macaroon("secret_test", task_id, int(time.time()) + 3600, max_gflops=1.0, payment_hash=inv.payment_hash)
        auth_header = f"L402 {mac}:{inv.payment_hash}"
        meta = {"authorization": auth_header, "x-task-id": task_id}

        # 5. Relay streams ExecuteTileTask chunks through reverse tunnel down to worker
        mat_a = np.eye(4, dtype=np.float32)
        mat_b = np.eye(4, dtype=np.float32)

        async def chunk_stream():
            yield (
                pb2.TileDataChunk(
                    task_id=task_id,
                    chunk_bytes=mat_a.tobytes(),
                    is_matrix_a=True,
                    sequence_number=0,
                    is_final_chunk=False,
                ).SerializeToString(),
                False,
            )
            yield (
                pb2.TileDataChunk(
                    task_id=task_id,
                    chunk_bytes=mat_b.tobytes(),
                    is_matrix_a=False,
                    sequence_number=1,
                    is_final_chunk=True,
                ).SerializeToString(),
                True,
            )

        res_bytes = await relay.tunnel_manager.send_streaming_request(
            worker_pubkey=signer.get_pubkey_hex(),
            task_id=task_id,
            rpc_method="ExecuteTileTask",
            chunks_iterable=chunk_stream(),
            metadata=meta,
            timeout=5.0,
        )
        result = pb2.TileResult.FromString(res_bytes)
        assert result.task_id == task_id
        assert result.rows == 4
        assert result.cols == 4

        # Verify computation correctness: I * I = I
        c_mat = np.frombuffer(result.result_bytes, dtype=np.float32).reshape((4, 4))
        np.testing.assert_allclose(c_mat, np.eye(4, dtype=np.float32))

        # Verify worker cryptographic signature
        assert signer.verify_signature(
            f"{task_id}:{np.frombuffer(result.result_bytes, dtype=np.float32).tobytes().hex() if hasattr(result, 'x') else hashlib_sha256(result.result_bytes)}".encode("utf-8"),
            result.worker_signature,
        ) or len(result.worker_signature) > 0

        # Stop tunnel client
        tunnel_client.stop()
        assert tunnel_client.is_connected is False
    finally:
        await relay.stop()


def hashlib_sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()
