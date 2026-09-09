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

        # 6. gRPC Server
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=config.grpc_max_workers),
            options=[
                ("grpc.max_receive_message_length", config.grpc_max_message_bytes),
                ("grpc.max_send_message_length", config.grpc_max_message_bytes),
            ],
        )
        pb2_grpc.add_MatrixComputeServiceServicer_to_server(self.servicer, self.server)
        bind_address = f"{config.p2p_host}:{config.p2p_port}"
        self.server.add_insecure_port(bind_address)

        # 7. Nostr Relay Client
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
        hardware_specs = self.attestation_engine.detect_hardware()
        logger.info(
            f"Detected Hardware: {hardware_specs.gpu_model} "
            f"({hardware_specs.vram_mb} MB VRAM, {hardware_specs.peak_fp32_tflops} TFLOPS)"
        )

        # 2. Start gRPC server
        self.server.start()
        logger.info(f"P2P gRPC server listening on {self.config.p2p_host}:{self.config.p2p_port}")

        # 3. Announce to Nostr relays
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
            multiaddr = self.config.p2p_multiaddr or f"/ip4/127.0.0.1/tcp/{self.config.p2p_port}"
            lightning_pubkey = self.config.lightning_routing_pubkey or self.signer.get_pubkey_hex()

            event = self.nostr_client.build_announce_event(
                signer=self.signer,
                hardware_specs=hardware_specs,
                pricing_specs=pricing_specs,
                bond_status=bond_status,
                p2p_multiaddr=multiaddr,
                lightning_pubkey=lightning_pubkey,
            )
            self.nostr_client.publish_announce(event)
            logger.info("Worker availability announced on Nostr relays (Kind 30000)")
        except Exception as ex:
            logger.warning(f"Nostr announcement skipped/deferred: {ex}")

    def stop(self, grace: float = 2.0) -> None:
        """Stops the daemon cleanly."""
        logger.info("Shutting down matrix-compute-node...")
        self.stop_event.set()
        self.nostr_client.disconnect()
        self.server.stop(grace=grace)
        logger.info("Daemon stopped successfully")

    def run_forever(self) -> None:
        """Runs the daemon blocking until SIGINT/SIGTERM."""
        self.start()
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
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
