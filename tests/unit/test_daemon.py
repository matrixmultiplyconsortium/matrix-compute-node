"""Unit tests for WorkerDaemon in main.py."""

from src.config.settings import NodeConfig
from src.main import WorkerDaemon
from src.relay.nostr_client import NostrRelayClient
from src.test_doubles.mock_nostr_relay import MockNostrRelay
from src.test_doubles.mock_settlement import MockSettlementService


def test_worker_daemon_init_and_lifecycle():
    mock_relay = MockNostrRelay()
    mock_settlement = MockSettlementService()
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    config = NodeConfig(
        p2p_host="127.0.0.1",
        p2p_port=free_port,
        nostr_relays=["ws://127.0.0.1:8008"],
        settlement_api_url="http://127.0.0.1:8080",
    )

    client = NostrRelayClient(relays=[mock_relay])
    daemon = WorkerDaemon(config, nostr_client=client, settlement_client=mock_settlement)
    assert daemon.signer is not None
    assert daemon.attestation_engine is not None
    assert daemon.gate is not None
    assert daemon.servicer is not None

    daemon.start()
    assert len(mock_relay.get_events()) >= 1

    daemon.stop(grace=0.1)
    assert daemon.stop_event.is_set()
