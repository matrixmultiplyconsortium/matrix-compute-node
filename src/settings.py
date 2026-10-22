"""
Compatibility re-export module for matrix-compute-node daemon settings.
Exposes NodeConfig from src.config.settings.
"""

from src.config.settings import NodeConfig, SECP256K1_ORDER

__all__ = ["NodeConfig", "SECP256K1_ORDER"]
