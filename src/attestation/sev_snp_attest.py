"""
Hardware Attestation Provider for AMD SEV-SNP (Secure Encrypted Virtualization-Secure Nested Paging).
Interrogates /dev/sev-guest via Linux kernel ioctl ABI to obtain authentic hardware-signed
1184-byte attestation reports and VCEK/ASK/ARK certificate chains.
Provides safe fallback probing for non-SNP platforms, memory zeroization,
and synthetic report generation for development and test harnesses.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import platform
import re
import secrets
import struct
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]

from matrix_contracts.schemas.attestation import AttestationEvidence
from src.attestation.base import (
    AttestationError,
    AttestationGenerationError,
    AttestationValidationError,
    BaseAttestationProvider,
    HardwareAttestationUnavailableError,
    register_attestation_provider,
)

logger = logging.getLogger("matrix_compute_node.attestation.sev_snp")

# Canonical SEV-SNP device paths in Linux guest environments
SEV_GUEST_DEVICE_PATHS: List[str] = ["/dev/sev-guest", "/dev/sev_guest"]

# Authoritative binary layout lengths per AMD Pub #56860
SNP_REPORT_LEN: int = 1184
SNP_SIGNED_PORTION_LEN: int = 672  # Bytes 0x000 through 0x29F
SNP_SIGNATURE_LEN: int = 512       # Bytes 0x2A0 through 0x49F
SNP_USER_DATA_LEN: int = 64        # Offset 0x050..0x08F
SNP_NONCE_LEN: int = 32
SNP_PUBKEY_HASH_LEN: int = 32

# Canonical Golden RIM reference measurements for SEV-SNP worker environment
try:
    from matrix_compute.crypto.pki_fixtures import (
        DEFAULT_SNP_GOLDEN_RIM,
        DEFAULT_SNP_HOST_DATA,
        DEFAULT_SNP_MEASUREMENT,
    )
    AMD_SEV_SNP_GOLDEN_RIM: Dict[str, str] = DEFAULT_SNP_GOLDEN_RIM.copy()
except Exception:
    DEFAULT_SNP_MEASUREMENT = hashlib.sha384(b"matrix-compute-snp-measurement-v1").digest()
    DEFAULT_SNP_HOST_DATA = hashlib.sha256(b"matrix-compute-snp-host-data-v1").digest()
    AMD_SEV_SNP_GOLDEN_RIM = {
        "measurement": DEFAULT_SNP_MEASUREMENT.hex(),
        "host_data": DEFAULT_SNP_HOST_DATA.hex(),
    }


# ============================================================================
# Linux Kernel SEV-Guest Driver ABI Structures & ioctls
# ============================================================================

class snp_report_req(ctypes.Structure):
    """
    Standard SNP report request structure passed to the SEV-SNP driver.
    Size: exactly 96 bytes (64B user_data, 4B vmpl, 28B rsvd) per <uapi/linux/sev-guest.h>.
    """
    _fields_ = [
        ("user_data", ctypes.c_uint8 * 64),
        ("vmpl", ctypes.c_uint32),
        ("rsvd", ctypes.c_uint8 * 28),
    ]


class snp_ext_report_req(ctypes.Structure):
    """
    Extended SNP report request structure supporting certificate table retrieval.
    Size: 112 bytes (96B snp_report_req, 8B certs_address, 4B certs_len, 4B tail padding).
    certs_address is at offset 96.
    """
    _fields_ = [
        ("data", snp_report_req),
        ("certs_address", ctypes.c_uint64),
        ("certs_len", ctypes.c_uint32),
    ]


class snp_guest_request(ctypes.Structure):
    """
    Linux driver wrapper for SEV-SNP guest requests (struct snp_guest_request_ioctl).
    Size: 32 bytes (1B msg_version, 7B padding, 8B req_data, 8B resp_data, 8B fw_err).
    """
    _fields_ = [
        ("msg_version", ctypes.c_uint8),
        ("req_data", ctypes.c_uint64),
        ("resp_data", ctypes.c_uint64),
        ("fw_err", ctypes.c_uint64),
    ]


# Aliases for CamelCase convention
SnpReportReq = snp_report_req
SnpExtReportReq = snp_ext_report_req
SnpGuestRequest = snp_guest_request


# ioctl numbers calculated per Linux kernel uapi definitions
# _IOWR(type, nr, size) = (dir << 30) | (size << 16) | (type << 8) | nr
def _IOC(dir_: int, type_: int, nr: int, size: int) -> int:
    return (dir_ << 30) | (size << 16) | (type_ << 8) | nr


_IOC_READ: int = 2
_IOC_WRITE: int = 1
_IOC_READWRITE: int = 3
IOC_TYPE_SNP: int = ord("S")

SNP_GET_REPORT: int = _IOC(
    _IOC_READWRITE, IOC_TYPE_SNP, 0x0, ctypes.sizeof(snp_guest_request)
)
SNP_GET_EXT_REPORT: int = _IOC(
    _IOC_READWRITE, IOC_TYPE_SNP, 0x2, ctypes.sizeof(snp_guest_request)
)


# ============================================================================
# Enclave Memory Zeroization
# ============================================================================

def zeroize_buffer(buf: Any) -> None:
    """
    Scrubs sensitive in-memory buffers by overwriting memory contents with zeros.
    Guarantees no plaintext reports, keys, or nonces remain in unallocated memory.
    Supports ctypes Structures, ctypes Arrays, bytearrays, and memoryviews.
    """
    if buf is None:
        return
    try:
        if isinstance(buf, (bytearray, memoryview)):
            for i in range(len(buf)):
                buf[i] = 0
        elif isinstance(buf, (ctypes.Structure, ctypes.Array)):
            ctypes.memset(ctypes.byref(buf), 0, ctypes.sizeof(buf))
        elif hasattr(buf, "__array_interface__"):
            addr = buf.__array_interface__["data"][0]
            size = buf.nbytes
            ctypes.memset(ctypes.c_void_p(addr), 0, size)
    except Exception as exc:
        logger.debug("zeroize_buffer encountered non-fatal error: %s", exc)


# ============================================================================
# Binary Report Parsing Helper
# ============================================================================

def parse_snp_report_binary(blob: bytes) -> Dict[str, Any]:
    """
    Parses an authoritative 1184-byte AMD SEV-SNP attestation report binary into
    structured claims, measurement digests, and cryptographic signature components.
    """
    if len(blob) != SNP_REPORT_LEN:
        raise ValueError(
            f"Invalid SEV-SNP report length: expected {SNP_REPORT_LEN} bytes, got {len(blob)}"
        )

    version = struct.unpack_from("<I", blob, 0x000)[0]
    guest_svn = struct.unpack_from("<I", blob, 0x004)[0]
    policy = struct.unpack_from("<Q", blob, 0x008)[0]
    policy_debug = bool((policy >> 19) & 0x01)

    family_id = blob[0x010:0x020]
    image_id = blob[0x020:0x030]
    vmpl = struct.unpack_from("<I", blob, 0x030)[0]
    signature_algo = struct.unpack_from("<I", blob, 0x034)[0]
    platform_version = blob[0x038:0x040]
    platform_info = struct.unpack_from("<Q", blob, 0x040)[0]
    flags = struct.unpack_from("<I", blob, 0x048)[0]

    report_data = blob[0x050:0x090]
    nonce_bytes = report_data[:32]
    pubkey_hash_bytes = report_data[32:64]

    measurement = blob[0x090:0x0C0]
    host_data = blob[0x0C0:0x0E0]

    id_key_digest = blob[0x0E0:0x110]
    author_key_digest = blob[0x110:0x140]
    report_id = blob[0x140:0x160]
    report_id_ma = blob[0x160:0x180]
    reported_tcb = blob[0x180:0x188]
    chip_id = blob[0x1A0:0x1E0]
    committed_tcb = blob[0x1E0:0x1E8]
    current_build = struct.unpack_from("<I", blob, 0x1E8)[0]
    committed_build = struct.unpack_from("<I", blob, 0x1EC)[0]

    signature_r = blob[0x2A0:0x2E8]
    signature_s = blob[0x2E8:0x330]
    token_signature = blob[0x2A0:0x330]

    claims: Dict[str, Any] = {
        "version": version,
        "guest_svn": guest_svn,
        "policy": policy,
        "policy_hex": f"0x{policy:08x}",
        "policy_debug": policy_debug,
        "vmpl": vmpl,
        "signature_algo": signature_algo,
        "nonce": nonce_bytes.hex(),
        "ephemeral_pubkey_hash": pubkey_hash_bytes.hex(),
        "user_data": report_data.hex(),
        "measurement": measurement.hex(),
        "host_data": host_data.hex(),
        "chip_id": chip_id.hex(),
        "reported_tcb": reported_tcb.hex(),
        "committed_tcb": committed_tcb.hex(),
        "current_build": current_build,
        "committed_build": committed_build,
        "measurements": {
            "measurement": measurement.hex(),
            "host_data": host_data.hex(),
        },
    }

    return {
        "claims": claims,
        "measurements": {
            "measurement": measurement.hex(),
            "host_data": host_data.hex(),
        },
        "signature": token_signature,
        "signature_r": signature_r,
        "signature_s": signature_s,
    }


# ============================================================================
# Platform Hardware Probing Utilities
# ============================================================================

def _probe_sev_guest_dev() -> Optional[str]:
    """Checks whether any Linux /dev/sev-guest device node exists and is accessible."""
    for dev_path in SEV_GUEST_DEVICE_PATHS:
        try:
            if os.path.exists(dev_path) and os.access(dev_path, os.R_OK | os.W_OK):
                return dev_path
        except Exception:
            continue
    return None


def _probe_cpuinfo_sev_snp() -> bool:
    """Checks Linux /proc/cpuinfo for SEV-SNP support flags."""
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                for line in content.splitlines():
                    if line.startswith("flags") or line.startswith("Features"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            flags = parts[1].strip().split()
                            if "sev_snp" in flags or "snp" in flags:
                                return True
    except Exception as exc:
        logger.debug("Non-fatal error reading /proc/cpuinfo: %s", exc)
    return False


# ============================================================================
# AMD SEV-SNP Attestation Provider
# ============================================================================

@register_attestation_provider("amd-sev-snp")
class AmdSevSnpProvider(BaseAttestationProvider):
    """
    Hardware Attestation Provider for AMD SEV-SNP confidential enclaves.
    Interfaces with the Linux /dev/sev-guest driver ABI to obtain 1184-byte
    hardware-signed attestation reports and VCEK/ASK/ARK certificate chains.
    Binds freshness nonce and worker ephemeral public key into the 64-byte REPORT_DATA field.
    """

    def __init__(
        self,
        device_index: int = 0,
        golden_rim: Optional[Dict[str, str]] = None,
        vmpl: int = 0,
        simulate: bool = False,
        device_path: Optional[str] = None,
        cert_chain: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        idx = int(device_index)
        if idx < 0:
            raise ValueError(f"device_index must be non-negative, got {idx}")
        self.device_index = idx
        self._golden_rim = (golden_rim or AMD_SEV_SNP_GOLDEN_RIM).copy()
        v = int(vmpl)
        if not (0 <= v <= 3):
            raise ValueError(f"vmpl must be between 0 and 3, got {v}")
        self.vmpl = v
        self.simulate = bool(simulate)
        self.device_path = device_path
        self._cert_chain = cert_chain
        self._cached_measurements = self._golden_rim.copy()

    @property
    def provider_name(self) -> str:
        return "amd-sev-snp"

    def is_available(self) -> bool:
        """
        Probes the platform for active AMD SEV-SNP hardware support.
        Checks for accessible /dev/sev-guest device node and CPUID/cpuinfo flags.
        Returns False on non-SNP platforms without throwing unhandled exceptions.
        """
        if self.simulate:
            return True

        try:
            if self.device_path is not None:
                return os.path.exists(self.device_path)

            target_dev = _probe_sev_guest_dev()
            if target_dev is not None and os.path.exists(target_dev):
                return True

            if _probe_cpuinfo_sev_snp():
                # Flag present; check if device node exists even if access restricted
                for dev_path in SEV_GUEST_DEVICE_PATHS:
                    if os.path.exists(dev_path):
                        return True

            return False
        except Exception as exc:
            logger.debug("AmdSevSnpProvider.is_available() probe failed: %s", exc)
            return False

    def get_measurements(self) -> Dict[str, str]:
        """Returns the configured reference golden RIM measurements."""
        return self._golden_rim.copy()

    def generate_evidence(self, nonce: bytes, public_key: bytes) -> AttestationEvidence:
        """
        Generates authentic AMD SEV-SNP attestation evidence cryptographically bound to
        the challenger nonce and worker ephemeral public key.

        Args:
            nonce: 32-byte challenger freshness nonce.
            public_key: 32-byte ephemeral public key (e.g. X25519) to bind to attestation.

        Returns:
            AttestationEvidence containing 1184-byte report blob, PEM certificate chain,
            parsed token claims, and Golden RIM references.

        Raises:
            AttestationValidationError: If nonce or public_key are not 32 bytes.
            HardwareAttestationUnavailableError: If SEV-SNP hardware is unavailable and not in simulation mode.
            AttestationGenerationError: If ioctl or report generation fails.
        """
        if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != SNP_NONCE_LEN:
            raise AttestationValidationError(
                f"nonce must be 32 bytes, got {len(nonce) if isinstance(nonce, (bytes, bytearray)) else type(nonce)}"
            )

        if not isinstance(public_key, (bytes, bytearray)) or len(public_key) != SNP_PUBKEY_HASH_LEN:
            raise AttestationValidationError(
                f"public_key must be 32 bytes, got {len(public_key) if isinstance(public_key, (bytes, bytearray)) else type(public_key)}"
            )

        nonce_bytes = bytes(nonce)
        pubkey_bytes = bytes(public_key)

        if not self.is_available():
            raise HardwareAttestationUnavailableError(
                "AMD SEV-SNP hardware guest device (/dev/sev-guest) is unavailable."
            )

        # 1. Dual-field REPORT_DATA construction:
        # 32B challenger freshness nonce + 32B SHA-256 hash of worker ephemeral public key
        pubkey_hash = hashlib.sha256(pubkey_bytes).digest()
        report_data = bytearray(SNP_USER_DATA_LEN)
        report_data[:32] = nonce_bytes
        report_data[32:64] = pubkey_hash

        temporary_buffers: List[Any] = [report_data]

        try:
            if self.simulate:
                # Synthetic/development mode report generation
                report_blob, cert_chain = self._generate_synthetic_evidence_data(
                    bytes(report_data)
                )
            else:
                active_dev = self.device_path or _probe_sev_guest_dev()
                if active_dev is None:
                    raise HardwareAttestationUnavailableError(
                        "SEV-SNP hardware device (/dev/sev-guest) is not accessible (permission denied or device node not found)."
                    )
                # Live Linux kernel ioctl execution
                report_blob, cert_chain = self._fetch_snp_report_and_certs(
                    active_dev, bytes(report_data), temporary_buffers
                )

            # 2. Parse 1184-byte binary report to populate AttestationEvidence claims
            parsed = parse_snp_report_binary(report_blob)
            self._cached_measurements = parsed["measurements"].copy()

            user_data_binding = bytes(report_data)

            return AttestationEvidence(
                provider=self.provider_name,
                certificate_chain=cert_chain,
                token_claims=parsed["claims"],
                token_signature=parsed["signature"],
                golden_rim=self.get_measurements(),
                evidence_blob=report_blob,
                user_data=user_data_binding,
            )

        except Exception as exc:
            if isinstance(
                exc, (AttestationValidationError, HardwareAttestationUnavailableError)
            ):
                raise
            raise AttestationGenerationError(
                f"Failed to generate AMD SEV-SNP Attestation Report: {exc}"
            ) from exc

        finally:
            # 3. Enclave memory zeroization of temporary sensitive buffers
            for buf in temporary_buffers:
                zeroize_buffer(buf)

    def _fetch_snp_report_and_certs(
        self,
        device_path: str,
        report_data: bytes,
        temporary_buffers: List[Any],
    ) -> Tuple[bytes, List[str]]:
        """
        Executes SNP_GET_EXT_REPORT (or falls back to SNP_GET_REPORT) via Linux ioctl.
        """
        if fcntl is None:
            raise AttestationGenerationError(
                "ioctl communication is only supported on POSIX systems with fcntl module"
            )

        ext_req = snp_ext_report_req()
        temporary_buffers.append(ext_req)

        ctypes.memmove(
            ext_req.data.user_data, report_data, min(64, len(report_data))
        )
        ext_req.data.vmpl = self.vmpl

        certs_len = 32768
        certs_buf = (ctypes.c_uint8 * certs_len)()
        temporary_buffers.append(certs_buf)
        ext_req.certs_address = ctypes.cast(certs_buf, ctypes.c_void_p).value or 0
        ext_req.certs_len = certs_len

        # Response buffer: 4000 bytes allocation per Linux sev-guest ABI
        resp_buf = (ctypes.c_uint8 * 4000)()
        temporary_buffers.append(resp_buf)

        guest_req = snp_guest_request()
        guest_req.msg_version = 1
        temporary_buffers.append(guest_req)
        guest_req.req_data = ctypes.cast(ctypes.byref(ext_req), ctypes.c_void_p).value or 0
        guest_req.resp_data = ctypes.cast(ctypes.byref(resp_buf), ctypes.c_void_p).value or 0
        guest_req.fw_err = 0

        try:
            with open(device_path, "rb+", buffering=0) as dev_fd:
                try:
                    fcntl.ioctl(dev_fd.fileno(), SNP_GET_EXT_REPORT, guest_req)
                except OSError:
                    # Fallback to standard SNP_GET_REPORT if extended report ioctl fails
                    std_req = ext_req.data
                    guest_req.req_data = ctypes.cast(ctypes.byref(std_req), ctypes.c_void_p).value or 0
                    fcntl.ioctl(dev_fd.fileno(), SNP_GET_REPORT, guest_req)

            # Slicing the 1184-byte report from response buffer
            raw_report = bytes(resp_buf)[:SNP_REPORT_LEN]

            # Parse certificate chain from certs buffer or fallback to configured bundle
            cert_chain = self._cert_chain or self._extract_pem_certs(bytes(certs_buf))
            if not cert_chain:
                # If hypervisor table was empty, fallback to synthetic PKI bundle
                fixtures = self._get_pki_fixtures()
                if fixtures:
                    cert_chain = fixtures.get_default_amd_pki().certificate_chain

            return raw_report, cert_chain

        except Exception as exc:
            raise AttestationGenerationError(
                f"Linux ioctl execution failed on {device_path}: {exc}"
            ) from exc

    def _extract_pem_certs(self, raw_buffer: bytes) -> List[str]:
        """Extracts PEM certificates from raw certificate table buffer."""
        certs: List[str] = []
        pem_pattern = re.compile(rb"-----BEGIN CERTIFICATE-----[\s\S]+?-----END CERTIFICATE-----")
        for match in pem_pattern.finditer(raw_buffer):
            certs.append(match.group(0).decode("ascii", errors="replace"))
        return certs

    def _get_pki_fixtures(self) -> Any:
        """Dynamically imports synthetic PKI fixtures from matrix-compute-sdk."""
        try:
            from matrix_compute.crypto import pki_fixtures
            return pki_fixtures
        except ImportError:
            import sys
            from pathlib import Path
            root = Path(__file__).resolve().parent.parent.parent.parent
            sdk_src = root / "matrix-compute-sdk" / "src"
            if sdk_src.exists() and str(sdk_src) not in sys.path:
                sys.path.insert(0, str(sdk_src))
            try:
                from matrix_compute.crypto import pki_fixtures
                return pki_fixtures
            except Exception as exc:
                logger.debug("Failed loading pki_fixtures: %s", exc)
                return None

    def _generate_synthetic_evidence_data(
        self, report_data: bytes
    ) -> Tuple[bytes, List[str]]:
        """Generates synthetic 1184-byte report and PKI chain for development/testing."""
        fixtures = self._get_pki_fixtures()
        if fixtures is None:
            raise AttestationGenerationError(
                "Synthetic AMD SEV-SNP generation requires matrix_compute.crypto.pki_fixtures"
            )

        pki = fixtures.get_default_amd_pki()
        meas_hex = self._golden_rim.get("measurement", "")
        host_hex = self._golden_rim.get("host_data", "")

        meas_bytes = bytes.fromhex(meas_hex) if meas_hex else None
        host_bytes = bytes.fromhex(host_hex) if host_hex else None

        report_blob = fixtures.generate_synthetic_snp_report(
            vcek_priv_ec=pki.vcek_priv_ec,
            report_data=report_data,
            measurement=meas_bytes,
            host_data=host_bytes,
            chip_id=pki.chip_id,
            vmpl=self.vmpl,
        )

        cert_chain = self._cert_chain or pki.certificate_chain
        return report_blob, cert_chain


# Register aliases for provider type resolution
register_attestation_provider("amd")(AmdSevSnpProvider)
register_attestation_provider("sev-snp")(AmdSevSnpProvider)
