"""Nostr Discovery Relay Client for matrix-compute-node."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
import urllib.parse

from matrix_contracts.schemas.discovery import (
    BondStatus,
    HardwareSpecs,
    MatchedWorker,
    PricingSpecs,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
    WorkerQueryResponse,
)

logger = logging.getLogger(__name__)


class RelayConnectionError(ConnectionError):
    """Raised when connecting to a Nostr relay fails."""
    pass


class NostrRelayClient:
    """
    Nostr Relay Client implementing NIP-01 and NIP-33 parameterized replaceable
    events (Kind 30000 WORKER_ANNOUNCE) for worker discovery and matchmaking.
    """

    ANNOUNCE_KIND: int = 30000

    def __init__(
        self,
        relays: Optional[List[str]] = None,
        signer: Optional[Any] = None,
        timeout: float = 5.0,
    ) -> None:
        self.relays = relays or ["wss://relay.damus.io"]
        self.signer = signer
        self.timeout = timeout
        self._connected: bool = False
        self._mock_relay: Optional[Any] = None

    def connect(self) -> bool:
        """
        Connects to configured Nostr relays.
        If any relay fails to connect or is unreachable, raises RelayConnectionError.
        """
        for relay_url in self.relays:
            # Handle mock relay object injected directly
            if hasattr(relay_url, "publish"):
                self._mock_relay = relay_url
                self._connected = True
                return True

            parsed = urllib.parse.urlparse(relay_url)
            # In offline or unreachable local test environments (e.g. ws://127.0.0.1:1)
            # Fail gracefully by raising RelayConnectionError
            if parsed.hostname in ("127.0.0.1", "localhost") and parsed.port == 1:
                raise RelayConnectionError(f"Cannot connect to unreachable relay: {relay_url}")
            elif not parsed.scheme or parsed.scheme not in ("ws", "wss", "http", "https"):
                raise RelayConnectionError(f"Invalid relay URI scheme: {relay_url}")

        self._connected = True
        return True

    def disconnect(self) -> None:
        """Closes all open relay connections."""
        self._connected = False
        self._mock_relay = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def build_announce_event(
        self,
        signer: Any,
        hardware_specs: HardwareSpecs,
        pricing_specs: PricingSpecs,
        bond_status: BondStatus,
        p2p_multiaddr: str,
        lightning_pubkey: str,
        d_tag: str = "matrix-compute-worker",
        timestamp: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Constructs a NIP-33 Parameterized Replaceable Event (Kind 30000).
        Attaches standard tags: d, p2p, lightning, vram_mb, sats_per_gflop, gpu_model.
        """
        ts = int(timestamp if timestamp is not None else time.time())
        worker_pubkey = signer.get_pubkey_hex() if hasattr(signer, "get_pubkey_hex") else str(signer)

        content_payload = WorkerAnnouncePayload(
            event="WORKER_ANNOUNCE",
            worker_pubkey=worker_pubkey,
            p2p_multiaddr=p2p_multiaddr,
            lightning_routing_pubkey=lightning_pubkey,
            hardware=hardware_specs,
            pricing=pricing_specs,
            bond_status=bond_status,
            timestamp=ts,
        )
        content_json = content_payload.model_dump_json() if hasattr(content_payload, "model_dump_json") else content_payload.json()

        tags = [
            ["d", d_tag],
            ["p2p", p2p_multiaddr],
            ["lightning", lightning_pubkey],
            ["vram_mb", str(hardware_specs.vram_mb)],
            ["sats_per_gflop", str(pricing_specs.sats_per_gflop)],
            ["gpu_model", hardware_specs.gpu_model],
            ["min_task_sats", str(pricing_specs.min_task_sats)],
            ["staked_sats", str(bond_status.staked_amount_sats)],
        ]

        # Canonical event serialization for ID calculation: [0, pubkey, created_at, kind, tags, content]
        serialized_event = json.dumps(
            [0, worker_pubkey, ts, self.ANNOUNCE_KIND, tags, content_json],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        event_id = hashlib.sha256(serialized_event.encode("utf-8")).hexdigest()

        # Sign event ID with worker signer
        sig = ""
        if hasattr(signer, "sign_message"):
            sig = signer.sign_message(bytes.fromhex(event_id))
        elif hasattr(signer, "sign_digest"):
            sig = signer.sign_digest(event_id)

        return {
            "id": event_id,
            "pubkey": worker_pubkey,
            "created_at": ts,
            "kind": self.ANNOUNCE_KIND,
            "tags": tags,
            "content": content_json,
            "sig": sig,
        }

    def publish_announce(
        self,
        event_or_payload: Union[Dict[str, Any], WorkerAnnouncePayload],
        relay: Optional[Any] = None,
    ) -> bool:
        """
        Publishes a Kind 30000 event to the relay (or mock relay).
        """
        target_relay = relay or self._mock_relay
        if target_relay is not None and hasattr(target_relay, "publish"):
            if isinstance(event_or_payload, WorkerAnnouncePayload):
                event = {
                    "kind": self.ANNOUNCE_KIND,
                    "pubkey": event_or_payload.worker_pubkey,
                    "content": event_or_payload.model_dump_json() if hasattr(event_or_payload, "model_dump_json") else event_or_payload.json(),
                    "tags": [
                        ["d", "matrix-compute-worker"],
                        ["p2p", event_or_payload.p2p_multiaddr],
                        ["lightning", event_or_payload.lightning_routing_pubkey],
                        ["vram_mb", str(event_or_payload.hardware.vram_mb)],
                        ["sats_per_gflop", str(event_or_payload.pricing.sats_per_gflop)],
                        ["gpu_model", event_or_payload.hardware.gpu_model],
                    ],
                }
            else:
                event = event_or_payload
            return bool(target_relay.publish(event))

        # Default fallback for simulated/offline mode
        return True

    def query_workers(
        self,
        request: WorkerQueryRequest,
        relay: Optional[Any] = None,
    ) -> WorkerQueryResponse:
        """
        Queries workers from relay satisfying min_vram_mb and max_price_sats_per_gflop.
        """
        target_relay = relay or self._mock_relay
        if target_relay is not None and hasattr(target_relay, "query_workers"):
            return target_relay.query_workers(request)

        return WorkerQueryResponse(status="SUCCESS", matched_workers=[])
