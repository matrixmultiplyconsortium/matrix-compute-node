"""Daemon runner and entrypoint for matrix-compute-node."""

from __future__ import annotations

from concurrent import futures
import logging
import signal
import sys
import threading
import time
from typing import Any, Optional, Sequence

import grpc

from matrix_contracts.schemas.discovery import BondStatus, PricingSpecs
from matrix_contracts.v1 import pb2_grpc
from src.attestation.nvml_attest import HardwareAttestationEngine
from src.auth.l402_gate import L402PreAllocationGate
from src.config.settings import NodeConfig
from src.crypto.signer import WorkerSigner
from src.engine.compute_core import ComputeEngine
from src.p2p.servicer import MatrixComputeServicer
from src.relay.nostr_client import NostrRelayClient

logger = logging.getLogger("matrix_compute_node")


class WorkerDaemon:
    """
    Worker daemon orchestrating hardware attestation, Nostr relay announcement,
    and P2P gRPC compute servicing with strict L402 pre-allocation protection.
    """

    def __init__(
        self,
        config: NodeConfig,
        nostr_client: Optional[NostrRelayClient] = None,
        settlement_client: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.stop_event = threading.Event()

        # 1. Initialize Cryptographic Signer
        self.signer = WorkerSigner(privkey_hex=config.worker_privkey_hex)

        # 2. Hardware Attestation Engine
        self.attestation_engine = HardwareAttestationEngine(
            signer=self.signer,
            device_index=config.gpu_device_index,
        )

        # 3. L402 Pre-Allocation Security Gate
        self.gate = L402PreAllocationGate(
            settlement_client_or_url=settlement_client or config.settlement_api_url,
            verification_key=config.settlement_verification_key,
            min_price_rate=config.sats_per_gflop,
            min_task_sats=config.min_task_sats,
            cache_ttl_seconds=getattr(config, "l402_cache_ttl_seconds", 300.0),
            max_cache_size=getattr(config, "l402_max_cache_size", 10000),
        )

        # 4. Compute Engine
        self.engine = ComputeEngine(prefer_gpu=True)

        # 5. P2P gRPC Servicer
        self.servicer = MatrixComputeServicer(
            signer=self.signer,
            gate=self.gate,
            engine=self.engine,
            config=self.config,
        )

        # 6. Reverse Tunnel Client (for Outbound NAT Traversal)
        self.tunnel_client = None
        is_nat_mode = bool(config.enable_outbound_nat or config.nat_mode == "reverse_tunnel")
        if is_nat_mode and config.relay_grpc_endpoint:
            from src.p2p.tunnel_client import ReverseTunnelClient
            self.tunnel_client = ReverseTunnelClient(
                relay_endpoint=config.relay_grpc_endpoint,
                signer=self.signer,
                servicer=self.servicer,
                heartbeat_interval_seconds=getattr(config, "tunnel_heartbeat_interval_seconds", 30.0),
            )

        # 7. gRPC Inbound Server (only bound if p2p_port > 0 or not outbound NAT)
        self.server = None
        self.bound_port = 0
        if not (is_nat_mode and config.p2p_port == 0):
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=config.grpc_max_workers),
                options=[
                    ("grpc.max_receive_message_length", config.grpc_max_message_bytes),
                    ("grpc.max_send_message_length", config.grpc_max_message_bytes),
                    ("grpc.keepalive_permit_without_calls", 1),
                    ("grpc.http2.min_ping_interval_without_data_ms", 5000),
                    ("grpc.http2.max_ping_strikes", 0),
                ],
            )
            pb2_grpc.add_MatrixComputeServiceServicer_to_server(self.servicer, self.server)
            bind_address = f"{config.p2p_host}:{config.p2p_port}"
            self.bound_port = self.server.add_insecure_port(bind_address)

        # 8. Nostr Relay Client
        self.nostr_client = nostr_client or NostrRelayClient(
            relays=config.nostr_relays,
            signer=self.signer,
        )

    def start(self) -> None:
        """Starts daemon services and registers announcement."""
        logger.info("Starting matrix-compute-node daemon...")
        logger.info(f"Worker Public Key (npub): {self.signer.get_npub()}")
        logger.info(f"Worker Public Key (hex):  {self.signer.get_pubkey_hex()}")

        # 1. Attest hardware
        self.hardware_specs = self.attestation_engine.detect_hardware()
        logger.info(
            f"Detected Hardware: {self.hardware_specs.gpu_model} "
            f"({self.hardware_specs.vram_mb} MB VRAM, {self.hardware_specs.peak_fp32_tflops} TFLOPS)"
        )

        # 2. Start gRPC server and/or reverse tunnel
        if self.server is not None:
            self.server.start()
            logger.info(f"P2P gRPC server listening on {self.config.p2p_host}:{self.bound_port or self.config.p2p_port}")

        if self.tunnel_client is not None:
            self.tunnel_client.start()
            logger.info(f"Reverse tunnel client started towards {self.config.relay_grpc_endpoint}")

        # 3. Announce to Nostr relays
        self._announce_once()

    def _announce_once(self) -> bool:
        """Builds and publishes a fresh Kind 30000 announcement event."""
        try:
            self.nostr_client.connect()
            pricing_specs = PricingSpecs(
                sats_per_gflop=self.config.sats_per_gflop,
                min_task_sats=self.config.min_task_sats,
            )
            bond_status = BondStatus(
                staked_amount_sats=100000,
                bond_txid="genesis_bond_txid_001",
            )
            multiaddr = self.config.resolve_advertised_multiaddr()
            lightning_pubkey = self.config.lightning_routing_pubkey or self.signer.get_pubkey_hex()

            event = self.nostr_client.build_announce_event(
                signer=self.signer,
                hardware_specs=self.hardware_specs,
                pricing_specs=pricing_specs,
                bond_status=bond_status,
                p2p_multiaddr=multiaddr,
                lightning_pubkey=lightning_pubkey,
                d_tag=self.config.d_tag,
                nat_mode=self.config.nat_mode,
                relay_endpoint=self.config.relay_grpc_endpoint,
            )
            ok = self.nostr_client.publish_announce(event)
            if ok:
                logger.info(
                    f"Worker availability announced on Nostr relays (Kind 30000) "
                    f"with multiaddr={multiaddr}, lightning_pubkey={lightning_pubkey[:16]}..."
                )
            return ok
        except Exception as ex:
            logger.warning(f"Nostr announcement skipped/deferred: {ex}")
            return False

    def stop(self, grace: float = 2.0) -> None:
        """Stops the daemon cleanly."""
        logger.info("Shutting down matrix-compute-node...")
        self.stop_event.set()
        self.nostr_client.disconnect()
        if self.tunnel_client is not None:
            self.tunnel_client.stop()
        if self.server is not None:
            self.server.stop(grace=grace)
        logger.info("Daemon stopped successfully")

    def run_forever(self) -> None:
        """Runs the daemon blocking until SIGINT/SIGTERM."""
        self.start()
        last_announce = time.time()
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
                if time.time() - last_announce >= 30.0:
                    self._announce_once()
                    last_announce = time.time()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    config = NodeConfig.load(args)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    daemon = WorkerDaemon(config)

    def _signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Signal {signum} received, stopping...")
        daemon.stop()
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, AttributeError):
        pass

    daemon.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
