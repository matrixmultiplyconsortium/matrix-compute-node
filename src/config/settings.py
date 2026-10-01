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
        description="Attestation provider: 'auto', 'nvidia-cca', 'simulated-cca', 'none'",
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
            "p2p_host": ["MATRIX_P2P_HOST", "P2P_HOST"],
            "p2p_port": ["MATRIX_P2P_PORT", "P2P_PORT"],
            "p2p_multiaddr": ["MATRIX_P2P_MULTIADDR", "P2P_MULTIADDR"],
            "lightning_routing_pubkey": ["MATRIX_LIGHTNING_ROUTING_PUBKEY", "LIGHTNING_ROUTING_PUBKEY"],
            "nostr_relays": ["MATRIX_NOSTR_RELAYS", "NOSTR_RELAYS"],
            "settlement_api_url": ["MATRIX_SETTLEMENT_API_URL", "MATRIX_SETTLEMENT_URL", "SETTLEMENT_API_URL", "SETTLEMENT_URL"],
            "settlement_verification_key": ["MATRIX_SETTLEMENT_VERIFICATION_KEY", "SETTLEMENT_VERIFICATION_KEY"],
            "sats_per_gflop": ["MATRIX_SATS_PER_GFLOP", "SATS_PER_GFLOP"],
            "min_task_sats": ["MATRIX_MIN_TASK_SATS", "MIN_TASK_SATS"],
            "gpu_device_index": ["MATRIX_GPU_DEVICE_INDEX", "GPU_DEVICE_INDEX"],
            "dtype_preference": ["MATRIX_DTYPE_PREFERENCE", "DTYPE_PREFERENCE"],
            "log_level": ["MATRIX_LOG_LEVEL", "LOG_LEVEL"],
            "grpc_max_workers": ["MATRIX_GRPC_MAX_WORKERS", "GRPC_MAX_WORKERS"],
            "grpc_max_message_bytes": ["MATRIX_GRPC_MAX_MESSAGE_BYTES", "GRPC_MAX_MESSAGE_BYTES"],
            "l402_cache_ttl_seconds": ["MATRIX_L402_CACHE_TTL_SECONDS", "L402_CACHE_TTL_SECONDS"],
            "l402_max_cache_size": ["MATRIX_L402_MAX_CACHE_SIZE", "L402_MAX_CACHE_SIZE"],
            "enable_confidential_compute": ["MATRIX_ENABLE_CONFIDENTIAL_COMPUTE", "ENABLE_CONFIDENTIAL_COMPUTE"],
            "attestation_provider": ["MATRIX_ATTESTATION_PROVIDER", "ATTESTATION_PROVIDER"],
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

        return enriched

    @field_validator("p2p_port")
    @classmethod
    def validate_port(cls, v: Any) -> int:
        port = int(v)
        if not (1 <= port <= 65535):
            raise ValueError(f"p2p_port must be between 1 and 65535, got {port}")
        return port

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
    def effective_p2p_multiaddr(self) -> str:
        """Returns explicit multiaddr or derives /ip4/{host}/tcp/{port}."""
        if self.p2p_multiaddr:
            return self.p2p_multiaddr
        return f"/ip4/{self.p2p_host}/tcp/{self.p2p_port}"

    def effective_lightning_pubkey(self, default_worker_pubkey: str) -> str:
        """Returns lightning routing pubkey or defaults to worker pubkey."""
        return self.lightning_routing_pubkey or default_worker_pubkey

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "NodeConfig":
        """Constructs NodeConfig from explicit environment mapping."""
        environ = os.environ if env is None else env
        data: Dict[str, Any] = {}

        if "MATRIX_WORKER_PRIVKEY" in environ:
            data["worker_privkey_hex"] = environ["MATRIX_WORKER_PRIVKEY"]
        elif "MATRIX_WORKER_PRIVKEY_HEX" in environ:
            data["worker_privkey_hex"] = environ["MATRIX_WORKER_PRIVKEY_HEX"]
        elif "WORKER_PRIVKEY" in environ:
            data["worker_privkey_hex"] = environ["WORKER_PRIVKEY"]

        if "MATRIX_P2P_HOST" in environ:
            data["p2p_host"] = environ["MATRIX_P2P_HOST"]
        elif "P2P_HOST" in environ:
            data["p2p_host"] = environ["P2P_HOST"]

        if "MATRIX_P2P_PORT" in environ:
            data["p2p_port"] = int(environ["MATRIX_P2P_PORT"])
        elif "P2P_PORT" in environ:
            data["p2p_port"] = int(environ["P2P_PORT"])

        if "MATRIX_P2P_MULTIADDR" in environ:
            data["p2p_multiaddr"] = environ["MATRIX_P2P_MULTIADDR"]
        elif "P2P_MULTIADDR" in environ:
            data["p2p_multiaddr"] = environ["P2P_MULTIADDR"]

        if "MATRIX_LIGHTNING_ROUTING_PUBKEY" in environ:
            data["lightning_routing_pubkey"] = environ["MATRIX_LIGHTNING_ROUTING_PUBKEY"]
        elif "LIGHTNING_ROUTING_PUBKEY" in environ:
            data["lightning_routing_pubkey"] = environ["LIGHTNING_ROUTING_PUBKEY"]

        if "MATRIX_NOSTR_RELAYS" in environ:
            val = environ["MATRIX_NOSTR_RELAYS"].strip()
            if val.startswith("[") and val.endswith("]"):
                data["nostr_relays"] = json.loads(val)
            else:
                data["nostr_relays"] = [r.strip() for r in val.split(",") if r.strip()]
        elif "NOSTR_RELAYS" in environ:
            val = environ["NOSTR_RELAYS"].strip()
            if val.startswith("[") and val.endswith("]"):
                data["nostr_relays"] = json.loads(val)
            else:
                data["nostr_relays"] = [r.strip() for r in val.split(",") if r.strip()]

        if "MATRIX_SETTLEMENT_API_URL" in environ:
            data["settlement_api_url"] = environ["MATRIX_SETTLEMENT_API_URL"]
        elif "MATRIX_SETTLEMENT_URL" in environ:
            data["settlement_api_url"] = environ["MATRIX_SETTLEMENT_URL"]
        elif "SETTLEMENT_API_URL" in environ:
            data["settlement_api_url"] = environ["SETTLEMENT_API_URL"]
        elif "SETTLEMENT_URL" in environ:
            data["settlement_api_url"] = environ["SETTLEMENT_URL"]

        if "MATRIX_SETTLEMENT_VERIFICATION_KEY" in environ:
            data["settlement_verification_key"] = environ["MATRIX_SETTLEMENT_VERIFICATION_KEY"]
        elif "SETTLEMENT_VERIFICATION_KEY" in environ:
            data["settlement_verification_key"] = environ["SETTLEMENT_VERIFICATION_KEY"]

        if "MATRIX_SATS_PER_GFLOP" in environ:
            data["sats_per_gflop"] = float(environ["MATRIX_SATS_PER_GFLOP"])
        elif "SATS_PER_GFLOP" in environ:
            data["sats_per_gflop"] = float(environ["SATS_PER_GFLOP"])

        if "MATRIX_MIN_TASK_SATS" in environ:
            data["min_task_sats"] = int(environ["MATRIX_MIN_TASK_SATS"])
        elif "MIN_TASK_SATS" in environ:
            data["min_task_sats"] = int(environ["MIN_TASK_SATS"])

        if "MATRIX_GPU_DEVICE_INDEX" in environ:
            data["gpu_device_index"] = int(environ["MATRIX_GPU_DEVICE_INDEX"])
        elif "GPU_DEVICE_INDEX" in environ:
            data["gpu_device_index"] = int(environ["GPU_DEVICE_INDEX"])

        if "MATRIX_DTYPE_PREFERENCE" in environ:
            data["dtype_preference"] = environ["MATRIX_DTYPE_PREFERENCE"]
        elif "DTYPE_PREFERENCE" in environ:
            data["dtype_preference"] = environ["DTYPE_PREFERENCE"]

        if "MATRIX_LOG_LEVEL" in environ:
            data["log_level"] = environ["MATRIX_LOG_LEVEL"]
        elif "LOG_LEVEL" in environ:
            data["log_level"] = environ["LOG_LEVEL"]

        if "MATRIX_L402_CACHE_TTL_SECONDS" in environ:
            data["l402_cache_ttl_seconds"] = float(environ["MATRIX_L402_CACHE_TTL_SECONDS"])
        elif "L402_CACHE_TTL_SECONDS" in environ:
            data["l402_cache_ttl_seconds"] = float(environ["L402_CACHE_TTL_SECONDS"])

        if "MATRIX_L402_MAX_CACHE_SIZE" in environ:
            data["l402_max_cache_size"] = int(environ["MATRIX_L402_MAX_CACHE_SIZE"])
        elif "L402_MAX_CACHE_SIZE" in environ:
            data["l402_max_cache_size"] = int(environ["L402_MAX_CACHE_SIZE"])

        if "MATRIX_ENABLE_CONFIDENTIAL_COMPUTE" in environ:
            data["enable_confidential_compute"] = environ["MATRIX_ENABLE_CONFIDENTIAL_COMPUTE"].strip().lower() in ("1", "true", "yes")
        elif "ENABLE_CONFIDENTIAL_COMPUTE" in environ:
            data["enable_confidential_compute"] = environ["ENABLE_CONFIDENTIAL_COMPUTE"].strip().lower() in ("1", "true", "yes")

        if "MATRIX_ATTESTATION_PROVIDER" in environ:
            data["attestation_provider"] = environ["MATRIX_ATTESTATION_PROVIDER"].strip()
        elif "ATTESTATION_PROVIDER" in environ:
            data["attestation_provider"] = environ["ATTESTATION_PROVIDER"].strip()

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
            help="Attestation provider ('auto', 'nvidia-cca', 'simulated-cca', 'none')",
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
        return cls(**merged)

    # Alias for backwards-compatibility and CLI invocations
    from_cli_and_env = load

