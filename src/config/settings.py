"""Configuration manager for matrix-compute-node daemon."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


class NodeConfig(BaseModel):
    """
    Worker daemon configuration model.
    Resolves settings with priority: CLI Arguments > Environment Variables > Defaults.
    """

    worker_privkey_hex: str = Field(
        default_factory=lambda: hex(secrets.randbelow(SECP256K1_ORDER - 1) + 1)[2:].zfill(64),
        description="32-byte Secp256k1 private key hex (64 hex characters)",
    )
    p2p_host: str = Field("0.0.0.0", description="P2P gRPC server binding host")
    p2p_port: int = Field(50051, description="P2P gRPC server binding port")
    p2p_multiaddr: Optional[str] = Field(None, description="Explicit libp2p multiaddr")
    lightning_routing_pubkey: Optional[str] = Field(
        None, description="Lightning node pubkey for routing payments"
    )
    nostr_relays: List[str] = Field(
        default_factory=lambda: ["ws://127.0.0.1:8008"],
        description="Nostr discovery relay WebSocket URLs",
    )
    settlement_api_url: str = Field(
        "http://127.0.0.1:8080", description="Settlement layer base HTTP URL"
    )
    settlement_verification_key: str = Field(
        "settlement_secret_key_dev", description="L402 Macaroon HMAC verification key"
    )
    sats_per_gflop: float = Field(0.0005, description="Price per GFLOP in Satoshis")
    min_task_sats: int = Field(10, description="Minimum price floor per task in Satoshis")
    gpu_device_index: int = 0
    dtype_preference: str = Field("fp32", description="Compute precision preference")
    log_level: str = "INFO"
    grpc_max_workers: int = 10
    grpc_max_message_bytes: int = 100 * 1024 * 1024  # 100 MB
    l402_cache_ttl_seconds: float = Field(
        300.0, description="TTL in seconds for cached HELD invoice verifications in L402 gate"
    )
    l402_max_cache_size: int = Field(
        10000, description="Maximum number of verified payment hashes to cache in L402 gate"
    )
    enable_confidential_compute: bool = Field(
        True, description="Enable TEE Confidential Computing"
    )
    attestation_provider: str = Field(
        "auto",
        description="Attestation provider: 'auto', 'nvidia-cca', 'amd-sev-snp', 'simulated-cca', 'none'",
    )
    inject_fraud: bool = Field(
        False,
        description="Stage 03 fault-injection flag: intentionally perturbs GEMM output to test Freivalds & HODL cancellation",
    )
    nat_mode: str = Field("direct", description="NAT traversal mode: 'direct' or 'reverse_tunnel'")
    relay_grpc_endpoint: Optional[str] = Field(
        None, description="gRPC endpoint of relay for reverse tunnel (host:port)"
    )
    enable_outbound_nat: bool = Field(
        False, description="Enable outbound-only reverse tunnel to relay"
    )
    tunnel_heartbeat_interval_seconds: float = Field(
        30.0, description="Heartbeat interval in seconds for reverse tunnel"
    )

    @model_validator(mode="before")
    @classmethod
    def populate_from_environment(cls, data: Any) -> Any:
        """
        If field values are not explicitly supplied in data dictionary,
        check os.environ for corresponding MATRIX_* and unprefixed env variables.
        """
        if not isinstance(data, dict):
            return data

        env_mappings = {
            "worker_privkey_hex": ["MATRIX_WORKER_PRIVKEY", "MATRIX_WORKER_PRIVKEY_HEX", "WORKER_PRIVKEY", "WORKER_PRIVKEY_HEX"],
            "p2p_host": ["MATRIX_P2P_HOST", "MATRIX_WORKER_P2P_HOST", "P2P_HOST"],
            "p2p_port": ["MATRIX_P2P_PORT", "MATRIX_WORKER_P2P_PORT", "P2P_PORT"],
            "p2p_multiaddr": ["MATRIX_P2P_MULTIADDR", "MATRIX_WORKER_P2P_MULTIADDR", "P2P_MULTIADDR"],
            "lightning_routing_pubkey": [
                "MATRIX_LIGHTNING_ROUTING_PUBKEY",
                "MATRIX_WORKER_LIGHTNING_ROUTING_PUBKEY",
                "LIGHTNING_ROUTING_PUBKEY",
                "WORKER_LIGHTNING_PUBKEY",
            ],
            "nostr_relays": ["MATRIX_NOSTR_RELAYS", "MATRIX_WORKER_NOSTR_RELAYS", "NOSTR_RELAYS"],
            "settlement_api_url": [
                "MATRIX_SETTLEMENT_API_URL",
                "MATRIX_WORKER_SETTLEMENT_API_URL",
                "MATRIX_SETTLEMENT_URL",
                "SETTLEMENT_API_URL",
                "SETTLEMENT_URL",
            ],
            "settlement_verification_key": [
                "MATRIX_SETTLEMENT_VERIFICATION_KEY",
                "MATRIX_WORKER_SETTLEMENT_VERIFICATION_KEY",
                "SETTLEMENT_VERIFICATION_KEY",
            ],
            "sats_per_gflop": ["MATRIX_SATS_PER_GFLOP", "MATRIX_WORKER_SATS_PER_GFLOP", "SATS_PER_GFLOP"],
            "min_task_sats": ["MATRIX_MIN_TASK_SATS", "MATRIX_WORKER_MIN_TASK_SATS", "MIN_TASK_SATS"],
            "gpu_device_index": ["MATRIX_GPU_DEVICE_INDEX", "MATRIX_WORKER_GPU_DEVICE_INDEX", "GPU_DEVICE_INDEX"],
            "dtype_preference": ["MATRIX_DTYPE_PREFERENCE", "MATRIX_WORKER_DTYPE_PREFERENCE", "DTYPE_PREFERENCE"],
            "log_level": ["MATRIX_LOG_LEVEL", "MATRIX_WORKER_LOG_LEVEL", "LOG_LEVEL"],
            "grpc_max_workers": ["MATRIX_GRPC_MAX_WORKERS", "MATRIX_WORKER_GRPC_MAX_WORKERS", "GRPC_MAX_WORKERS"],
            "grpc_max_message_bytes": ["MATRIX_GRPC_MAX_MESSAGE_BYTES", "GRPC_MAX_MESSAGE_BYTES"],
            "l402_cache_ttl_seconds": ["MATRIX_L402_CACHE_TTL_SECONDS", "L402_CACHE_TTL_SECONDS"],
            "l402_max_cache_size": ["MATRIX_L402_MAX_CACHE_SIZE", "L402_MAX_CACHE_SIZE"],
            "enable_confidential_compute": [
                "MATRIX_ENABLE_CONFIDENTIAL_COMPUTE",
                "MATRIX_WORKER_ENABLE_CONFIDENTIAL_COMPUTE",
                "ENABLE_CONFIDENTIAL_COMPUTE",
            ],
            "attestation_provider": [
                "MATRIX_ATTESTATION_PROVIDER",
                "MATRIX_WORKER_ATTESTATION_PROVIDER",
                "ATTESTATION_PROVIDER",
            ],
            "inject_fraud": [
                "MATRIX_WORKER_INJECT_FRAUD",
                "MATRIX_INJECT_FRAUD",
                "INJECT_FRAUD",
            ],
            "nat_mode": ["MATRIX_NAT_MODE", "NAT_MODE"],
            "relay_grpc_endpoint": [
                "MATRIX_RELAY_GRPC_ENDPOINT",
                "RELAY_GRPC_ENDPOINT",
                "MATRIX_RELAY_ENDPOINT",
                "RELAY_ENDPOINT",
            ],
            "enable_outbound_nat": [
                "MATRIX_ENABLE_OUTBOUND_NAT",
                "ENABLE_OUTBOUND_NAT",
            ],
            "tunnel_heartbeat_interval_seconds": [
                "MATRIX_TUNNEL_HEARTBEAT_INTERVAL_SECONDS",
                "TUNNEL_HEARTBEAT_INTERVAL_SECONDS",
            ],
        }

        enriched = dict(data)
        for field_name, env_vars in env_mappings.items():
            if field_name not in enriched or enriched[field_name] is None:
                for env_var in env_vars:
                    if env_var in os.environ and os.environ[env_var] != "":
                        val = os.environ[env_var]
                        enriched[field_name] = val
                        break

        # Convert nostr_relays from string if needed
        if "nostr_relays" in enriched and isinstance(enriched["nostr_relays"], str):
            raw = enriched["nostr_relays"].strip()
            if raw.startswith("[") and raw.endswith("]"):
                try:
                    enriched["nostr_relays"] = json.loads(raw)
                except Exception:
                    enriched["nostr_relays"] = [r.strip() for r in raw[1:-1].split(",") if r.strip()]
            else:
                enriched["nostr_relays"] = [r.strip() for r in raw.split(",") if r.strip()]

        if "enable_confidential_compute" in enriched and isinstance(enriched["enable_confidential_compute"], str):
            enriched["enable_confidential_compute"] = enriched["enable_confidential_compute"].strip().lower() in ("1", "true", "yes")

        if "inject_fraud" in enriched and isinstance(enriched["inject_fraud"], str):
            enriched["inject_fraud"] = enriched["inject_fraud"].strip().lower() in ("1", "true", "yes", "on")

        if "enable_outbound_nat" in enriched and isinstance(enriched["enable_outbound_nat"], str):
            enriched["enable_outbound_nat"] = enriched["enable_outbound_nat"].strip().lower() in ("1", "true", "yes", "on")

        if "tunnel_heartbeat_interval_seconds" in enriched and enriched["tunnel_heartbeat_interval_seconds"] is not None:
            enriched["tunnel_heartbeat_interval_seconds"] = float(enriched["tunnel_heartbeat_interval_seconds"])

        if enriched.get("enable_outbound_nat") and enriched.get("nat_mode") in (None, "direct", ""):
            enriched["nat_mode"] = "reverse_tunnel"

        return enriched

    @field_validator("p2p_port")
    @classmethod
    def validate_port(cls, v: Any) -> int:
        port = int(v)
        if not (0 <= port <= 65535):
            raise ValueError(f"p2p_port must be between 0 and 65535, got {port}")
        return port

    @model_validator(mode="after")
    def validate_port_mode(self) -> "NodeConfig":
        is_nat = bool(self.enable_outbound_nat or self.nat_mode == "reverse_tunnel")
        if not is_nat and self.p2p_port == 0:
            raise ValueError("p2p_port must be between 1 and 65535 when nat_mode is direct, got 0")
        return self

    @field_validator("dtype_preference")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"fp32", "fp16", "bf16"}
        cleaned = str(v).lower().strip()
        if cleaned not in allowed:
            raise ValueError(f"dtype_preference must be one of {allowed}, got {v}")
        return cleaned

    @field_validator("sats_per_gflop")
    @classmethod
    def validate_sats_rate(cls, v: Any) -> float:
        rate = float(v)
        if rate < 0:
            raise ValueError(f"sats_per_gflop must be >= 0, got {rate}")
        return rate

    @field_validator("min_task_sats")
    @classmethod
    def validate_min_sats(cls, v: Any) -> int:
        min_sats = int(v)
        if min_sats < 0:
            raise ValueError(f"min_task_sats must be >= 0, got {min_sats}")
        return min_sats

    @field_validator("settlement_api_url")
    @classmethod
    def validate_settlement_url(cls, v: str) -> str:
        cleaned = str(v).strip()
        parsed = urlparse(cleaned)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"settlement_api_url must be a valid http or https URL, got '{v}'")
        return cleaned

    @field_validator("worker_privkey_hex")
    @classmethod
    def validate_privkey(cls, v: str) -> str:
        clean = str(v).strip().lower()
        if not clean:
            return secrets.token_hex(32)
        if len(clean) != 64:
            raise ValueError(f"worker_privkey_hex must be 64 hex characters (32 bytes), got length {len(clean)}")
        try:
            scalar = int(clean, 16)
        except ValueError:
            raise ValueError("worker_privkey_hex contains invalid hexadecimal characters")
        if not (1 <= scalar < SECP256K1_ORDER):
            raise ValueError("worker_privkey_hex scalar out of valid Secp256k1 range")
        return clean

    @property
    def worker_pubkey_hex(self) -> str:
        """Derives 33-byte compressed Secp256k1 public key from worker_privkey_hex."""
        try:
            from src.crypto.signer import WorkerSigner
            return WorkerSigner(self.worker_privkey_hex).get_pubkey_hex()
        except Exception:
            return ""

    @property
    def effective_p2p_multiaddr(self) -> str:
        """Returns explicit multiaddr or derives /ip4/{host}/tcp/{port}."""
        if self.p2p_multiaddr:
            return self.p2p_multiaddr
        return f"/ip4/{self.p2p_host}/tcp/{self.p2p_port}"

    def resolve_advertised_multiaddr(self, env: Optional[Dict[str, str]] = None) -> str:
        """
        Resolves a routable multiaddr for Nostr Kind 30000 announcements.
        1. Uses explicit p2p_multiaddr if configured.
        2. Outbound NAT / Reverse Tunnel mode: constructs circuit multiaddr via relay.
        3. Auto-detects cloud GPU port-mapping env vars (Vast.ai / RunPod):
           - Vast.ai: PUBLIC_IPADDR + VAST_TCP_PORT_{p2p_port}
           - RunPod:  RUNPOD_PUBLIC_IP + RUNPOD_TCP_PORT_{p2p_port}
        4. Uses /ip4/{p2p_host}/tcp/{p2p_port} if p2p_host is not a wildcard (0.0.0.0 / ::).
        5. Otherwise probes outbound interface towards settlement_api_url or Nostr relay,
           falling back to 127.0.0.1.
        """
        if self.p2p_multiaddr:
            return self.p2p_multiaddr

        # Outbound NAT / Reverse Tunnel mode
        if self.enable_outbound_nat or self.nat_mode == "reverse_tunnel":
            if self.relay_grpc_endpoint:
                clean_relay = self.relay_grpc_endpoint.strip()
                pubkey = self.worker_pubkey_hex
                if clean_relay.startswith("/"):
                    if "p2p-circuit" in clean_relay:
                        return clean_relay
                    return f"{clean_relay}/p2p-circuit/p2p/{pubkey}" if pubkey else f"{clean_relay}/p2p-circuit"
                else:
                    if ":" in clean_relay:
                        host, port_str = clean_relay.split(":", 1)
                    else:
                        host, port_str = clean_relay, "9090"
                    proto = "ip4" if re.match(r"^\d+\.\d+\.\d+\.\d+$", host) else "dns4"
                    return f"/{proto}/{host}/tcp/{port_str}/p2p-circuit/p2p/{pubkey}" if pubkey else f"/{proto}/{host}/tcp/{port_str}/p2p-circuit"

        environ = os.environ if env is None else env
        # Vast.ai external TCP port mapping detection
        vast_ip = environ.get("PUBLIC_IPADDR", "").strip()
        vast_port = environ.get(f"VAST_TCP_PORT_{self.p2p_port}", "").strip()
        if vast_ip and vast_port.isdigit():
            return f"/ip4/{vast_ip}/tcp/{int(vast_port)}"

        # RunPod external TCP port mapping detection
        runpod_ip = environ.get("RUNPOD_PUBLIC_IP", "").strip()
        runpod_port = environ.get(f"RUNPOD_TCP_PORT_{self.p2p_port}", "").strip()
        if runpod_ip and runpod_port.isdigit():
            return f"/ip4/{runpod_ip}/tcp/{int(runpod_port)}"

        if self.p2p_host and self.p2p_host not in ("0.0.0.0", "::", ""):
            return f"/ip4/{self.p2p_host}/tcp/{self.p2p_port}"

        import socket
        probe_targets: List[Tuple[str, int]] = []
        try:
            parsed_settlement = urlparse(self.settlement_api_url)
            if parsed_settlement.hostname and parsed_settlement.hostname not in ("127.0.0.1", "localhost", "0.0.0.0"):
                probe_targets.append((parsed_settlement.hostname, parsed_settlement.port or 8080))
        except Exception:
            pass

        for relay in self.nostr_relays:
            try:
                parsed_relay = urlparse(relay)
                if parsed_relay.hostname and parsed_relay.hostname not in ("127.0.0.1", "localhost", "0.0.0.0"):
                    probe_targets.append((parsed_relay.hostname, parsed_relay.port or 8008))
            except Exception:
                pass

        for host, port in probe_targets:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect((host, port))
                    local_ip = s.getsockname()[0]
                    if local_ip and not local_ip.startswith("127.") and local_ip != "0.0.0.0":
                        return f"/ip4/{local_ip}/tcp/{self.p2p_port}"
            except Exception:
                continue

        return f"/ip4/127.0.0.1/tcp/{self.p2p_port}"

    def effective_lightning_pubkey(self, default_worker_pubkey: str) -> str:
        """Returns lightning routing pubkey or defaults to worker pubkey."""
        return self.lightning_routing_pubkey or default_worker_pubkey

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "NodeConfig":
        """Constructs NodeConfig from explicit environment mapping."""
        environ = os.environ if env is None else env
        data: Dict[str, Any] = {}

        for field_name, env_vars in {
            "worker_privkey_hex": ["MATRIX_WORKER_PRIVKEY", "MATRIX_WORKER_PRIVKEY_HEX", "WORKER_PRIVKEY", "WORKER_PRIVKEY_HEX"],
            "p2p_host": ["MATRIX_P2P_HOST", "MATRIX_WORKER_P2P_HOST", "P2P_HOST"],
            "p2p_port": ["MATRIX_P2P_PORT", "MATRIX_WORKER_P2P_PORT", "P2P_PORT"],
            "p2p_multiaddr": ["MATRIX_P2P_MULTIADDR", "MATRIX_WORKER_P2P_MULTIADDR", "P2P_MULTIADDR"],
            "lightning_routing_pubkey": [
                "MATRIX_LIGHTNING_ROUTING_PUBKEY",
                "MATRIX_WORKER_LIGHTNING_ROUTING_PUBKEY",
                "LIGHTNING_ROUTING_PUBKEY",
                "WORKER_LIGHTNING_PUBKEY",
            ],
            "nostr_relays": ["MATRIX_NOSTR_RELAYS", "MATRIX_WORKER_NOSTR_RELAYS", "NOSTR_RELAYS"],
            "settlement_api_url": [
                "MATRIX_SETTLEMENT_API_URL",
                "MATRIX_WORKER_SETTLEMENT_API_URL",
                "MATRIX_SETTLEMENT_URL",
                "SETTLEMENT_API_URL",
                "SETTLEMENT_URL",
            ],
            "settlement_verification_key": [
                "MATRIX_SETTLEMENT_VERIFICATION_KEY",
                "MATRIX_WORKER_SETTLEMENT_VERIFICATION_KEY",
                "SETTLEMENT_VERIFICATION_KEY",
            ],
            "sats_per_gflop": ["MATRIX_SATS_PER_GFLOP", "MATRIX_WORKER_SATS_PER_GFLOP", "SATS_PER_GFLOP"],
            "min_task_sats": ["MATRIX_MIN_TASK_SATS", "MATRIX_WORKER_MIN_TASK_SATS", "MIN_TASK_SATS"],
            "gpu_device_index": ["MATRIX_GPU_DEVICE_INDEX", "MATRIX_WORKER_GPU_DEVICE_INDEX", "GPU_DEVICE_INDEX"],
            "dtype_preference": ["MATRIX_DTYPE_PREFERENCE", "MATRIX_WORKER_DTYPE_PREFERENCE", "DTYPE_PREFERENCE"],
            "log_level": ["MATRIX_LOG_LEVEL", "MATRIX_WORKER_LOG_LEVEL", "LOG_LEVEL"],
            "l402_cache_ttl_seconds": ["MATRIX_L402_CACHE_TTL_SECONDS", "L402_CACHE_TTL_SECONDS"],
            "l402_max_cache_size": ["MATRIX_L402_MAX_CACHE_SIZE", "L402_MAX_CACHE_SIZE"],
            "enable_confidential_compute": [
                "MATRIX_ENABLE_CONFIDENTIAL_COMPUTE",
                "MATRIX_WORKER_ENABLE_CONFIDENTIAL_COMPUTE",
                "ENABLE_CONFIDENTIAL_COMPUTE",
            ],
            "attestation_provider": [
                "MATRIX_ATTESTATION_PROVIDER",
                "MATRIX_WORKER_ATTESTATION_PROVIDER",
                "ATTESTATION_PROVIDER",
            ],
            "inject_fraud": [
                "MATRIX_WORKER_INJECT_FRAUD",
                "MATRIX_INJECT_FRAUD",
                "INJECT_FRAUD",
            ],
            "nat_mode": ["MATRIX_NAT_MODE", "NAT_MODE"],
            "relay_grpc_endpoint": [
                "MATRIX_RELAY_GRPC_ENDPOINT",
                "RELAY_GRPC_ENDPOINT",
                "MATRIX_RELAY_ENDPOINT",
                "RELAY_ENDPOINT",
            ],
            "enable_outbound_nat": [
                "MATRIX_ENABLE_OUTBOUND_NAT",
                "ENABLE_OUTBOUND_NAT",
            ],
            "tunnel_heartbeat_interval_seconds": [
                "MATRIX_TUNNEL_HEARTBEAT_INTERVAL_SECONDS",
                "TUNNEL_HEARTBEAT_INTERVAL_SECONDS",
            ],
        }.items():
            for env_var in env_vars:
                if env_var in environ and environ[env_var] != "":
                    data[field_name] = environ[env_var]
                    break

        if "p2p_port" in data:
            data["p2p_port"] = int(data["p2p_port"])
        if "sats_per_gflop" in data:
            data["sats_per_gflop"] = float(data["sats_per_gflop"])
        if "min_task_sats" in data:
            data["min_task_sats"] = int(data["min_task_sats"])
        if "gpu_device_index" in data:
            data["gpu_device_index"] = int(data["gpu_device_index"])
        if "l402_cache_ttl_seconds" in data:
            data["l402_cache_ttl_seconds"] = float(data["l402_cache_ttl_seconds"])
        if "l402_max_cache_size" in data:
            data["l402_max_cache_size"] = int(data["l402_max_cache_size"])
        if "nostr_relays" in data and isinstance(data["nostr_relays"], str):
            val = data["nostr_relays"].strip()
            if val.startswith("[") and val.endswith("]"):
                data["nostr_relays"] = json.loads(val)
            else:
                data["nostr_relays"] = [r.strip() for r in val.split(",") if r.strip()]
        if "enable_confidential_compute" in data and isinstance(data["enable_confidential_compute"], str):
            data["enable_confidential_compute"] = data["enable_confidential_compute"].strip().lower() in ("1", "true", "yes")
        if "inject_fraud" in data and isinstance(data["inject_fraud"], str):
            data["inject_fraud"] = data["inject_fraud"].strip().lower() in ("1", "true", "yes", "on")
        if "enable_outbound_nat" in data and isinstance(data["enable_outbound_nat"], str):
            data["enable_outbound_nat"] = data["enable_outbound_nat"].strip().lower() in ("1", "true", "yes", "on")
        if "tunnel_heartbeat_interval_seconds" in data and data["tunnel_heartbeat_interval_seconds"] is not None:
            data["tunnel_heartbeat_interval_seconds"] = float(data["tunnel_heartbeat_interval_seconds"])
        if data.get("enable_outbound_nat") and data.get("nat_mode") in (None, "direct", ""):
            data["nat_mode"] = "reverse_tunnel"

        return cls(**data)

    @classmethod
    def create_arg_parser(cls, add_help: bool = True) -> argparse.ArgumentParser:
        """Builds CLI argument parser."""
        parser = argparse.ArgumentParser(
            description="matrix-compute-node daemon CLI options",
            add_help=add_help,
        )
        parser.add_argument(
            "--worker-privkey",
            "--worker-privkey-hex",
            dest="worker_privkey_hex",
            type=str,
            default=None,
            help="32-byte Secp256k1 private key in hexadecimal",
        )
        parser.add_argument(
            "--p2p-host",
            dest="p2p_host",
            type=str,
            default=None,
            help="gRPC server binding host (e.g. 0.0.0.0)",
        )
        parser.add_argument(
            "--p2p-port",
            dest="p2p_port",
            type=int,
            default=None,
            help="gRPC server binding port (e.g. 50051)",
        )
        parser.add_argument(
            "--p2p-multiaddr",
            dest="p2p_multiaddr",
            type=str,
            default=None,
            help="Explicit libp2p/gRPC multiaddr",
        )
        parser.add_argument(
            "--lightning-routing-pubkey",
            dest="lightning_routing_pubkey",
            type=str,
            default=None,
            help="Lightning node public key for HODL invoices",
        )
        parser.add_argument(
            "--nostr-relays",
            "--nostr-relay",
            dest="nostr_relays",
            type=str,
            default=None,
            help="Comma-separated Nostr relay WebSocket URLs",
        )
        parser.add_argument(
            "--settlement-api-url",
            "--settlement-url",
            dest="settlement_api_url",
            type=str,
            default=None,
            help="Settlement layer HTTP URL",
        )
        parser.add_argument(
            "--settlement-verification-key",
            dest="settlement_verification_key",
            type=str,
            default=None,
            help="L402 Macaroon verification secret key",
        )
        parser.add_argument(
            "--sats-per-gflop",
            dest="sats_per_gflop",
            type=float,
            default=None,
            help="Price in Satoshis per computed GFLOP",
        )
        parser.add_argument(
            "--min-task-sats",
            dest="min_task_sats",
            type=int,
            default=None,
            help="Minimum price floor per task in Satoshis",
        )
        parser.add_argument(
            "--gpu-device-index",
            dest="gpu_device_index",
            type=int,
            default=None,
            help="Target GPU device index",
        )
        parser.add_argument(
            "--dtype-preference",
            dest="dtype_preference",
            type=str,
            default=None,
            help="Compute precision format (fp32, fp16, bf16)",
        )
        parser.add_argument(
            "--log-level",
            dest="log_level",
            type=str,
            default=None,
            help="Logging verbosity level",
        )
        parser.add_argument(
            "--l402-cache-ttl-seconds",
            dest="l402_cache_ttl_seconds",
            type=float,
            default=None,
            help="TTL in seconds for cached HELD invoice verifications in L402 gate",
        )
        parser.add_argument(
            "--l402-max-cache-size",
            dest="l402_max_cache_size",
            type=int,
            default=None,
            help="Maximum number of verified payment hashes to cache in L402 gate",
        )
        parser.add_argument(
            "--enable-confidential-compute",
            dest="enable_confidential_compute",
            action=argparse.BooleanOptionalAction if hasattr(argparse, "BooleanOptionalAction") else "store_true",
            default=None,
            help="Enable TEE Confidential Computing",
        )
        parser.add_argument(
            "--attestation-provider",
            dest="attestation_provider",
            type=str,
            default=None,
            help="Attestation provider ('auto', 'nvidia-cca', 'amd-sev-snp', 'simulated-cca', 'none')",
        )
        parser.add_argument(
            "--inject-fraud",
            dest="inject_fraud",
            action=argparse.BooleanOptionalAction if hasattr(argparse, "BooleanOptionalAction") else "store_true",
            default=None,
            help="Intentionally perturb GEMM output for Scenario 3 invalidation drills",
        )
        parser.add_argument(
            "--nat-mode",
            dest="nat_mode",
            type=str,
            default=None,
            help="NAT traversal mode ('direct', 'reverse_tunnel')",
        )
        parser.add_argument(
            "--relay-grpc-endpoint",
            "--relay-endpoint",
            dest="relay_grpc_endpoint",
            type=str,
            default=None,
            help="Relay gRPC endpoint for reverse tunnel (host:port)",
        )
        parser.add_argument(
            "--enable-outbound-nat",
            dest="enable_outbound_nat",
            action=argparse.BooleanOptionalAction if hasattr(argparse, "BooleanOptionalAction") else "store_true",
            default=None,
            help="Enable outbound-only reverse tunnel to relay",
        )
        parser.add_argument(
            "--tunnel-heartbeat-interval-seconds",
            dest="tunnel_heartbeat_interval_seconds",
            type=float,
            default=None,
            help="Heartbeat interval in seconds for reverse tunnel",
        )
        return parser

    @classmethod
    def load(
        cls,
        args: Optional[Sequence[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> "NodeConfig":
        """
        Loads configuration merging Defaults < Environment Variables < CLI Arguments.
        """
        base_config = cls.from_env(env=env).model_dump()

        parser = cls.create_arg_parser()
        parsed_args, _ = parser.parse_known_args(args)
        cli_dict = {k: v for k, v in vars(parsed_args).items() if v is not None}

        if "nostr_relays" in cli_dict and isinstance(cli_dict["nostr_relays"], str):
            raw = cli_dict["nostr_relays"].strip()
            if raw.startswith("[") and raw.endswith("]"):
                try:
                    cli_dict["nostr_relays"] = json.loads(raw)
                except Exception:
                    cli_dict["nostr_relays"] = [r.strip() for r in raw[1:-1].split(",") if r.strip()]
            else:
                cli_dict["nostr_relays"] = [r.strip() for r in raw.split(",") if r.strip()]

        merged = {**base_config, **cli_dict}
        if merged.get("enable_outbound_nat") and merged.get("nat_mode") in (None, "direct", ""):
            merged["nat_mode"] = "reverse_tunnel"
        return cls(**merged)

    # Alias for backwards-compatibility and CLI invocations
    from_cli_and_env = load

