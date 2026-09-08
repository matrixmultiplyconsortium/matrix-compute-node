"""In-memory Mock Nostr Relay test double for offline discovery testing."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from matrix_contracts.schemas.discovery import (
    MatchedWorker,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
    WorkerQueryResponse,
)


class MockNostrRelay:
    """
    Hermetic in-memory Nostr Discovery Relay test double.
    Implements NIP-01 and NIP-33 parameterized replaceable events (Kind 30000).
    Thread-safe across all publish, query, and administrative operations.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # Key: (kind, pubkey, d_tag) -> event dict
        self._replaceable_events: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
        # Key: event_id -> event dict
        self._events_by_id: Dict[str, Dict[str, Any]] = {}
        # Tag index: tag_name -> {tag_value -> set(event_id)}
        self._tag_index: Dict[str, Dict[str, Set[str]]] = {}

    def _extract_d_tag(self, tags: List[List[str]]) -> str:
        """Extracts the 'd' identifier tag value, defaulting to 'matrix-worker'."""
        for t in tags:
            if len(t) >= 2 and t[0] == "d":
                return str(t[1])
        return "matrix-worker"

    def _extract_timestamp(self, event: Dict[str, Any]) -> int:
        """Extracts event timestamp, checking 'created_at', 'timestamp', and 'ts'."""
        if "created_at" in event and event["created_at"] is not None:
            return int(event["created_at"])
        if "timestamp" in event and event["timestamp"] is not None:
            return int(event["timestamp"])
        if "ts" in event and event["ts"] is not None:
            return int(event["ts"])
        return int(time.time())

    @staticmethod
    def _parse_event_specs(event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses hardware, pricing, and connectivity specifications from event tags
        with automatic fallback to JSON 'content' payload if tags omit them.
        """
        tag_map = {t[0]: t[1] for t in event.get("tags", []) if len(t) >= 2}
        p2p = tag_map.get("p2p", "")
        lightning = tag_map.get("lightning", "")
        gpu_model = tag_map.get("gpu_model", "")

        price: Optional[float] = None
        if "sats_per_gflop" in tag_map:
            try:
                price = float(tag_map["sats_per_gflop"])
            except (ValueError, TypeError):
                price = None

        vram_mb: Optional[int] = None
        if "vram_mb" in tag_map:
            try:
                vram_mb = int(tag_map["vram_mb"])
            except (ValueError, TypeError):
                vram_mb = None

        # Fallback to content if any specification is missing
        if not p2p or not lightning or not gpu_model or price is None or vram_mb is None:
            raw_content = event.get("content", "{}")
            data: Dict[str, Any] = {}
            if isinstance(raw_content, dict):
                data = raw_content
            elif isinstance(raw_content, str):
                try:
                    parsed = json.loads(raw_content)
                    if isinstance(parsed, dict):
                        data = parsed
                except Exception:
                    data = {}

            if data:
                if not p2p:
                    p2p = str(data.get("p2p_multiaddr", ""))
                if not lightning:
                    lightning = str(data.get("lightning_routing_pubkey", ""))
                if not gpu_model:
                    hw = data.get("hardware", {})
                    if isinstance(hw, dict):
                        gpu_model = str(hw.get("gpu_model", ""))
                if price is None:
                    pricing = data.get("pricing", {})
                    if isinstance(pricing, dict) and "sats_per_gflop" in pricing:
                        try:
                            price = float(pricing["sats_per_gflop"])
                        except (ValueError, TypeError):
                            pass
                if vram_mb is None:
                    hw = data.get("hardware", {})
                    if isinstance(hw, dict) and "vram_mb" in hw:
                        try:
                            vram_mb = int(hw["vram_mb"])
                        except (ValueError, TypeError):
                            pass

        return {
            "p2p": p2p,
            "lightning": lightning,
            "gpu_model": gpu_model,
            "price": price if price is not None else 0.0,
            "vram_mb": vram_mb if vram_mb is not None else 0,
        }

    def publish(self, event: Dict[str, Any]) -> bool:
        """
        Publishes a Nostr event to the in-memory store according to NIP-33 rules.
        Thread-safe against concurrent updates to identical (kind, pubkey, d_tag).
        Returns True on successful ingestion.
        """
        kind = int(event.get("kind", 1))
        pubkey = str(event.get("pubkey", ""))
        tags = event.get("tags", [])
        content = event.get("content", "")
        ts = self._extract_timestamp(event)

        d_tag = self._extract_d_tag(tags)
        key = (kind, pubkey, d_tag)

        # Generate event id if absent
        event_id = event.get("id")
        if not event_id:
            serialized = json.dumps(
                [0, pubkey, ts, kind, tags, str(content)],
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            event_id = hashlib.sha256(serialized).hexdigest()
            event["id"] = event_id

        with self._lock:
            # NIP-33 Parameterized Replaceable Event resolution
            if key in self._replaceable_events:
                existing = self._replaceable_events[key]
                existing_ts = self._extract_timestamp(existing)
                if ts < existing_ts:
                    # Reject/ignore older announcement
                    return True
                # Remove old event from indexes safely
                old_id = existing.get("id")
                if old_id:
                    self._events_by_id.pop(old_id, None)
                    self._unindex_tags(old_id, existing.get("tags", []))

            # Store new event
            stored_event = dict(event)
            stored_event["id"] = event_id
            stored_event["created_at"] = ts
            self._replaceable_events[key] = stored_event
            self._events_by_id[event_id] = stored_event
            self._index_tags(event_id, tags)

            return True

    def publish_event(self, event: Dict[str, Any]) -> str:
        """Publishes event and returns the event ID string."""
        self.publish(event)
        return str(event.get("id", ""))

    def publish_announce(self, payload: WorkerAnnouncePayload) -> str:
        """
        Serializes a WorkerAnnouncePayload into a standard Nostr Kind 30000 event
        and publishes it to the relay.
        """
        content_json = payload.model_dump_json()
        tags = [
            ["d", "matrix-worker"],
            ["p2p", payload.p2p_multiaddr],
            ["lightning", payload.lightning_routing_pubkey],
            ["vram_mb", str(payload.hardware.vram_mb)],
            ["sats_per_gflop", str(payload.pricing.sats_per_gflop)],
            ["gpu_model", payload.hardware.gpu_model],
            ["peak_fp32_tflops", str(payload.hardware.peak_fp32_tflops)],
        ]
        created_at = payload.timestamp or int(time.time())

        serialized = json.dumps(
            [0, payload.worker_pubkey, created_at, 30000, tags, content_json],
            separators=(",", ":"),
        ).encode("utf-8")
        event_id = hashlib.sha256(serialized).hexdigest()

        event = {
            "id": event_id,
            "pubkey": payload.worker_pubkey,
            "created_at": created_at,
            "kind": 30000,
            "tags": tags,
            "content": content_json,
            "sig": "mock_schnorr_sig_" + event_id[:16],
        }
        self.publish(event)
        return event_id

    def query(
        self,
        min_vram_mb: int = 0,
        max_price: Optional[float] = None,
        gpu_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries stored events matching hardware and pricing criteria.
        Returns a list of matching event dictionaries.
        """
        with self._lock:
            events_snapshot = list(self._replaceable_events.values())

        results: List[Dict[str, Any]] = []

        for event in events_snapshot:
            if int(event.get("kind", 0)) != 30000:
                continue

            specs = self._parse_event_specs(event)
            vram_mb = specs["vram_mb"]
            price = specs["price"]
            event_gpu = specs["gpu_model"]

            # Check criteria
            if vram_mb < min_vram_mb:
                continue
            if max_price is not None and price > max_price:
                continue
            if gpu_model is not None:
                if gpu_model.lower() not in event_gpu.lower():
                    continue

            results.append(event)

        return results

    def query_workers(self, request: WorkerQueryRequest) -> WorkerQueryResponse:
        """
        Matchmaking engine:
        Filters active Kind 30000 worker announcements by minimum VRAM and maximum price,
        sorts candidates by price ascending (cheapest first), and returns up to required_slots.
        """
        matching_events = self.query(
            min_vram_mb=request.min_vram_mb,
            max_price=request.max_price_sats_per_gflop,
        )

        candidates: List[Tuple[float, int, MatchedWorker]] = []

        for event in matching_events:
            specs = self._parse_event_specs(event)
            pubkey = str(event.get("pubkey", ""))
            price = specs["price"]
            vram_mb = specs["vram_mb"]

            worker = MatchedWorker(
                worker_pubkey=pubkey,
                p2p_multiaddr=specs["p2p"],
                lightning_routing_pubkey=specs["lightning"],
                reputation_score=1.0,
                sats_per_gflop=price,
            )
            candidates.append((price, -vram_mb, worker))

        # Sort: lowest price first, tie-break by largest VRAM
        candidates.sort(key=lambda x: (x[0], x[1]))
        matched = [item[2] for item in candidates[:request.required_slots]]

        return WorkerQueryResponse(
            status="SUCCESS",
            matched_workers=matched,
        )

    def get_events(self) -> List[Dict[str, Any]]:
        """Returns all active stored events."""
        with self._lock:
            return list(self._replaceable_events.values())

    def get_events_by_d_tag(self, d_tag: str) -> List[Dict[str, Any]]:
        """Returns all active events matching the given d-tag."""
        with self._lock:
            return [
                ev for (k, pk, d), ev in self._replaceable_events.items()
                if d == d_tag
            ]

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Returns event by ID."""
        with self._lock:
            return self._events_by_id.get(event_id)

    def get_active_announcement(self, pubkey: str, d_tag: str = "matrix-worker") -> Optional[Dict[str, Any]]:
        """Returns the active Kind 30000 event for the specified pubkey and d-tag."""
        with self._lock:
            return self._replaceable_events.get((30000, pubkey, d_tag))

    def _index_tags(self, event_id: str, tags: List[List[str]]) -> None:
        for t in tags:
            if len(t) >= 2:
                name, val = t[0], t[1]
                if name not in self._tag_index:
                    self._tag_index[name] = {}
                if val not in self._tag_index[name]:
                    self._tag_index[name][val] = set()
                self._tag_index[name][val].add(event_id)

    def _unindex_tags(self, event_id: str, tags: List[List[str]]) -> None:
        for t in tags:
            if len(t) >= 2:
                name, val = t[0], t[1]
                if name in self._tag_index and val in self._tag_index[name]:
                    self._tag_index[name][val].discard(event_id)

    def clear(self) -> None:
        """Clears all stored events and indexes."""
        with self._lock:
            self._replaceable_events.clear()
            self._events_by_id.clear()
            self._tag_index.clear()
