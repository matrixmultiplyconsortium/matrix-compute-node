"""
Hardware Attestation Provider for Intel Trust Domain Extensions (TDX).
Interfaces with the Linux /dev/tdx_guest driver ABI and host QGS daemon via AF_VSOCK
to obtain authentic, hardware-signed Intel DCAP Quotes (Format v4/v5).
Provides safe fallback probing for non-TDX platforms, memory zeroization,
and synthetic quote generation for offline hermetic testing.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import secrets
import socket
import struct
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

logger = logging.getLogger("matrix_compute_node.attestation.tdx")

# Canonical device nodes and networking constants
TDX_GUEST_DEV_NODES: List[str] = ["/dev/tdx_guest", "/dev/tdx-guest"]
TDX_REPORTDATA_LEN: int = 64
TDREPORT_STRUCT_LEN: int = 1024
VSOCK_HOST_CID: int = 2
VSOCK_QGS_PORT: int = 4050
INTEL_TDX_TEE_TYPE: int = 0x00000081

# Linux Kernel TDX guest driver ioctl numbers
# TDX_CMD_GET_REPORT0 = _IOWR('T', 1, struct tdx_report_req) = 0xC4405401
TDX_CMD_GET_REPORT0: int = 0xC4405401
# TDX_CMD_GET_QUOTE = _IOWR('T', 2, struct tdx_quote_req) = 0xC0105402
TDX_CMD_GET_QUOTE: int = 0xC0105402

# Default Golden RIM measurements for Intel TDX worker environment
DEFAULT_TDX_GOLDEN_RIM: Dict[str, str] = {
    "mrtd": hashlib.sha384(b"matrix-compute-tdx-mrtd-synthetic-v1").hexdigest(),
    "rtmr0": "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
    "rtmr1": "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899",
    "rtmr2": "99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa",
    "rtmr3": "f1e2d3c4b5a60718293a4b5c6d7e8f90f1e2d3c4b5a60718293a4b5c6d7e8f90f1e2d3c4b5a60718293a4b5c6d7e8f90",
}

try:
    from matrix_compute.crypto.pki_fixtures import DEFAULT_TDX_GOLDEN_RIM as _SDK_RIM
    DEFAULT_TDX_GOLDEN_RIM = _SDK_RIM.copy()
except Exception:
    pass

INTEL_TDX_GOLDEN_RIM: Dict[str, str] = DEFAULT_TDX_GOLDEN_RIM.copy()


# ============================================================================
# Linux Kernel TDX Guest Driver ABI Structures
# ============================================================================

class tdx_report_req(ctypes.Structure):
    """
    Standard TDX report request structure per <uapi/linux/tdx-guest.h>.
    Size: exactly 1088 bytes (64 bytes reportdata, 1024 bytes tdreport).
    """
    _fields_ = [
        ("reportdata", ctypes.c_uint8 * 64),
        ("tdreport", ctypes.c_uint8 * 1024),
    ]


class tdx_quote_req(ctypes.Structure):
    """
    Standard TDX quote request structure.
    Size: exactly 16 bytes (8 bytes user buffer pointer, 8 bytes buffer length).
    """
    _fields_ = [
        ("buf", ctypes.c_uint64),
        ("len", ctypes.c_uint64),
    ]


TdxReportReq = tdx_report_req
TdxQuoteReq = tdx_quote_req


# ============================================================================
# Memory Zeroization
# ============================================================================

def zeroize_buffer(buf: Any) -> None:
    """Scrubs sensitive in-memory buffers by overwriting memory contents with zeros."""
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
# Hardware Probing Utilities
# ============================================================================

def _probe_tdx_guest_dev() -> Optional[str]:
    """Checks whether any Linux /dev/tdx_guest device node exists and is accessible."""
    for dev_path in TDX_GUEST_DEV_NODES:
        try:
            if os.path.exists(dev_path) and os.access(dev_path, os.R_OK | os.W_OK):
                return dev_path
        except Exception:
            continue
    return None


def _probe_cpuinfo_tdx() -> bool:
    """Checks Linux /proc/cpuinfo for Intel TDX guest support flags."""
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("flags") or line.startswith("Features"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            flags = parts[1].strip().split()
                            if any(flag in flags for flag in ("tdx_guest", "intel_tdx", "tdx")):
                                return True
    except Exception as exc:
        logger.debug("Non-fatal error reading /proc/cpuinfo: %s", exc)
    return False


def _probe_vsock_qgs(timeout: float = 0.2) -> bool:
    """Probes host Quote Generation Service (QGS) via AF_VSOCK on Host CID 2, Port 4050."""
    if not hasattr(socket, "AF_VSOCK"):
        return False
    try:
        with socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((VSOCK_HOST_CID, VSOCK_QGS_PORT))
            return True
    except Exception:
        return False


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Reads exactly num_bytes from stream socket, guarding against packet fragmentation."""
    buf = bytearray()
    while len(buf) < num_bytes:
        chunk = sock.recv(min(4096, num_bytes - len(buf)))
        if not chunk:
            raise AttestationGenerationError(
                f"VSOCK connection closed prematurely by QGS: received {len(buf)} of {num_bytes} expected bytes"
            )
        buf.extend(chunk)
    return bytes(buf)


def fetch_quote_vsock(td_report: bytes, timeout: float = 10.0) -> bytes:
    """
    Transmits TDREPORT to host Quote Generation Service (QGS) daemon via AF_VSOCK.
    Returns the raw binary DCAP Quote.
    """
    if not hasattr(socket, "AF_VSOCK"):
        raise HardwareAttestationUnavailableError("AF_VSOCK is not supported on this platform")

    if len(td_report) != TDREPORT_STRUCT_LEN:
        raise AttestationValidationError(
            f"td_report must be exactly {TDREPORT_STRUCT_LEN} bytes, got {len(td_report)}"
        )

    req_type = 1  # GET_QUOTE_REQ
    req_payload = struct.pack(">I", req_type) + td_report
    req_frame = struct.pack(">I", len(req_payload)) + req_payload

    try:
        with socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((VSOCK_HOST_CID, VSOCK_QGS_PORT))
            sock.sendall(req_frame)

            hdr_bytes = _recv_exact(sock, 4)
            resp_len = struct.unpack(">I", hdr_bytes)[0]
            if resp_len < 4:
                raise AttestationGenerationError(f"Malformed QGS response length header: {resp_len}")

            body_bytes = _recv_exact(sock, resp_len)
            error_code = struct.unpack(">I", body_bytes[:4])[0]
            if error_code != 0:
                raise AttestationGenerationError(
                    f"Host QGS daemon returned error status code: 0x{error_code:08X}"
                )

            quote_bytes = body_bytes[4:]
            if len(quote_bytes) < 636:
                raise AttestationGenerationError(
                    f"QGS returned truncated Quote binary ({len(quote_bytes)} bytes; expected >= 636)"
                )
            return quote_bytes

    except (socket.error, OSError) as exc:
        raise AttestationGenerationError(f"AF_VSOCK communication with host QGS failed: {exc}") from exc


def parse_tdx_quote_binary(quote_blob: bytes) -> Dict[str, Any]:
    """Parses raw binary DCAP Quote to extract claims, measurements, and PCK certificate chain."""
    if len(quote_blob) < 636:
        raise AttestationValidationError(
            f"Quote binary too short ({len(quote_blob)} bytes; expected >= 636 bytes)"
        )

    version, att_key_type, tee_type = struct.unpack_from("<HHI", quote_blob, 0)
    if version not in (4, 5):
        raise AttestationValidationError(f"Unsupported Quote version: {version}")

    if version == 4:
        body_offset = 48
        body_size = 584
    else:
        if len(quote_blob) < 54:
            raise AttestationValidationError("Truncated Quote v5 body descriptor")
        body_type, body_size = struct.unpack_from("<HI", quote_blob, 48)
        body_offset = 54

    if len(quote_blob) < body_offset + 584:
        raise AttestationValidationError("Truncated TD Report Body in quote")

    tdattributes = struct.unpack_from("<Q", quote_blob, body_offset + 120)[0]
    mrtd = quote_blob[body_offset + 136 : body_offset + 184].hex()
    rtmr0 = quote_blob[body_offset + 328 : body_offset + 376].hex()
    rtmr1 = quote_blob[body_offset + 376 : body_offset + 424].hex()
    rtmr2 = quote_blob[body_offset + 424 : body_offset + 472].hex()
    rtmr3 = quote_blob[body_offset + 472 : body_offset + 520].hex()
    reportdata = quote_blob[body_offset + 520 : body_offset + 584]

    signed_body_len = body_offset + body_size
    sig_offset = signed_body_len + 4
    key_len = 64 if att_key_type == 2 else 96
    quote_sig = quote_blob[sig_offset : sig_offset + key_len]

    # Extract PEM Certificate Chain from Quote Certification Data
    cert_chain: List[str] = []
    pem_pattern = b"-----BEGIN CERTIFICATE-----"
    if pem_pattern in quote_blob:
        try:
            raw_text = quote_blob.decode("latin1", errors="ignore")
            parts = raw_text.split("-----BEGIN CERTIFICATE-----")
            for p in parts[1:]:
                clean = p.split("-----END CERTIFICATE-----")[0].strip()
                if clean:
                    cert_chain.append(f"-----BEGIN CERTIFICATE-----\n{clean}\n-----END CERTIFICATE-----")
        except Exception:
            pass

    measurements = {
        "mrtd": mrtd,
        "rtmr0": rtmr0,
        "rtmr1": rtmr1,
        "rtmr2": rtmr2,
        "rtmr3": rtmr3,
    }

    claims = {
        "provider": "intel-tdx",
        "quote_version": version,
        "tee_type": hex(tee_type),
        "nonce": reportdata[:32].hex(),
        "ephemeral_pubkey_hash": reportdata[32:64].hex(),
        "client_pubkey_hash": reportdata[32:64].hex(),
        "user_data": reportdata.hex(),
        "measurements": measurements,
        "tdattributes": tdattributes,
        "debug": bool((tdattributes & 0x01) != 0),
    }

    return {
        "claims": claims,
        "measurements": measurements,
        "quote_signature": quote_sig,
        "certificate_chain": cert_chain,
    }


# ============================================================================
# Intel TDX Attestation Provider
# ============================================================================

@register_attestation_provider("intel-tdx")
class IntelTdxProvider(BaseAttestationProvider):
    """
    Hardware Attestation Provider for Intel Trust Domain Extensions (TDX).
    Interfaces with /dev/tdx_guest and host QGS to produce authentic DCAP Quotes.
    Binds challenger freshness nonce and worker ephemeral public key into the 64-byte REPORTDATA.
    """

    def __init__(
        self,
        device_index: int = 0,
        golden_rim: Optional[Dict[str, str]] = None,
        simulate: bool = False,
        device_path: Optional[str] = None,
        cert_chain: Optional[List[str]] = None,
        quote_strategy: str = "auto",  # 'auto', 'ioctl', 'vsock'
        quote_version: int = 4,
        binding_mode: str = "dual",    # 'dual' or 'sha256_dual'
        **kwargs: Any,
    ) -> None:
        idx = int(device_index)
        if idx < 0:
            raise ValueError(f"device_index must be non-negative, got {idx}")
        self.device_index = idx
        self._golden_rim = (golden_rim or INTEL_TDX_GOLDEN_RIM).copy()
        self.simulate = bool(simulate)
        self.device_path = device_path
        self._cert_chain = cert_chain
        self.quote_strategy = quote_strategy.lower().strip()
        self.quote_version = quote_version
        self.binding_mode = binding_mode.lower().strip()
        self._cached_measurements = self._golden_rim.copy()

    @property
    def provider_name(self) -> str:
        return "intel-tdx"

    def is_available(self) -> bool:
        """
        Probes platform for active Intel TDX guest environment.
        Safe on non-TDX hosts (returns False without raising exceptions).
        """
        if self.simulate:
            return True

        try:
            if self.device_path is not None:
                return os.path.exists(self.device_path)

            target_dev = _probe_tdx_guest_dev()
            if target_dev is not None and os.path.exists(target_dev):
                return True

            if _probe_cpuinfo_tdx():
                for dev_path in TDX_GUEST_DEV_NODES:
                    if os.path.exists(dev_path):
                        return True

            if _probe_vsock_qgs(timeout=0.2):
                return True

            return False
        except Exception as exc:
            logger.debug("IntelTdxProvider.is_available() probe failed: %s", exc)
            return False

    def get_measurements(self) -> Dict[str, str]:
        """Returns current MRTD and RTMR0-3 measurements."""
        if self._cached_measurements:
            return self._cached_measurements.copy()
        return self._golden_rim.copy()

    def generate_evidence(self, nonce: bytes, public_key: bytes) -> AttestationEvidence:
        """
        Generates authentic Intel TDX Quote bound to challenger nonce and ephemeral public key.

        Args:
            nonce: 32-byte challenger freshness nonce.
            public_key: 32-byte ephemeral public key (e.g. X25519) to bind to attestation.

        Returns:
            AttestationEvidence containing DCAP Quote binary, PCK PEM chain, claims, and measurements.

        Raises:
            AttestationValidationError: If nonce or public_key are not 32 bytes.
            HardwareAttestationUnavailableError: If TDX hardware is unavailable and not in simulation mode.
            AttestationGenerationError: If quote generation fails.
        """
        if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != 32:
            raise AttestationValidationError(
                f"nonce must be 32 bytes, got {len(nonce) if isinstance(nonce, (bytes, bytearray)) else type(nonce)}"
            )

        if not isinstance(public_key, (bytes, bytearray)) or len(public_key) != 32:
            raise AttestationValidationError(
                f"public_key must be 32 bytes, got {len(public_key) if isinstance(public_key, (bytes, bytearray)) else type(public_key)}"
            )

        nonce_bytes = bytes(nonce)
        pubkey_bytes = bytes(public_key)

        if not self.is_available():
            raise HardwareAttestationUnavailableError(
                "Intel TDX hardware guest environment is unavailable."
            )

        # 1. 64-byte REPORTDATA construction: 32B nonce + 32B SHA-256(public_key)
        pubkey_hash = hashlib.sha256(pubkey_bytes).digest()
        report_data = bytearray(TDX_REPORTDATA_LEN)
        if self.binding_mode == "sha256_dual":
            report_data[:32] = hashlib.sha256(nonce_bytes).digest()
        else:
            report_data[:32] = nonce_bytes
        report_data[32:64] = pubkey_hash

        temporary_buffers: List[Any] = [report_data]

        try:
            if self.simulate:
                quote_blob, cert_chain = self._generate_synthetic_evidence_data(
                    bytes(report_data), nonce_bytes, pubkey_bytes
                )
            else:
                quote_blob, cert_chain = self._fetch_td_quote(
                    bytes(report_data), temporary_buffers
                )

            # 2. Parse binary Quote to extract claims and measurements
            parsed = parse_tdx_quote_binary(quote_blob)
            self._cached_measurements = parsed["measurements"].copy()
            if cert_chain:
                parsed["certificate_chain"] = cert_chain

            # Ensure client/worker pubkey hash alignment in claims
            parsed["claims"]["client_pubkey_hash"] = pubkey_hash.hex()
            parsed["claims"]["ephemeral_pubkey_hash"] = pubkey_hash.hex()
            parsed["claims"]["nonce"] = nonce_bytes.hex()

            return AttestationEvidence(
                provider=self.provider_name,
                certificate_chain=parsed["certificate_chain"],
                token_claims=parsed["claims"],
                token_signature=parsed["quote_signature"],
                golden_rim=self.get_measurements(),
                evidence_blob=quote_blob,
                user_data=bytes(report_data),
            )

        except Exception as exc:
            if isinstance(exc, (AttestationValidationError, HardwareAttestationUnavailableError)):
                raise
            raise AttestationGenerationError(f"Failed to generate Intel TDX Quote: {exc}") from exc

        finally:
            for buf in temporary_buffers:
                zeroize_buffer(buf)

    def _fetch_td_quote(self, report_data: bytes, temporary_buffers: List[Any]) -> Tuple[bytes, List[str]]:
        """Dispatches quote generation to ioctl or AF_VSOCK according to quote_strategy."""
        active_dev = self.device_path or _probe_tdx_guest_dev()
        if active_dev is None and self.quote_strategy in ("auto", "ioctl"):
            raise HardwareAttestationUnavailableError(
                f"No accessible TDX guest device node found among {TDX_GUEST_DEV_NODES}"
            )

        # Strategy: ioctl
        if self.quote_strategy in ("auto", "ioctl") and active_dev is not None:
            try:
                quote_blob = self._get_quote_ioctl(active_dev, report_data, temporary_buffers)
                return quote_blob, self._cert_chain or []
            except Exception as exc:
                if self.quote_strategy == "ioctl":
                    raise AttestationGenerationError(f"ioctl quote generation failed: {exc}") from exc
                logger.debug("TDX_CMD_GET_QUOTE failed, falling back to AF_VSOCK QGS: %s", exc)

        # Strategy: vsock
        if self.quote_strategy in ("auto", "vsock"):
            if active_dev is None:
                raise HardwareAttestationUnavailableError("TDX device node missing for TDREPORT0 generation")
            td_report = self._get_td_report0_ioctl(active_dev, report_data, temporary_buffers)
            quote_blob = self._get_quote_vsock(td_report)
            return quote_blob, self._cert_chain or []

        raise AttestationGenerationError(f"Unsupported quote strategy: {self.quote_strategy}")

    def _get_td_report0_ioctl(self, dev_path: str, report_data: bytes, temporary_buffers: List[Any]) -> bytes:
        """Invokes TDX_CMD_GET_REPORT0 ioctl on dev_path to retrieve the 1024-byte TDREPORT0."""
        if fcntl is None:
            raise AttestationGenerationError("fcntl module is only available on POSIX systems")

        req = tdx_report_req()
        temporary_buffers.append(req)
        ctypes.memmove(req.reportdata, report_data, min(64, len(report_data)))

        try:
            with open(dev_path, "r+b", buffering=0) as fd:
                fcntl.ioctl(fd.fileno(), TDX_CMD_GET_REPORT0, req)
        except OSError as exc:
            raise AttestationGenerationError(f"TDX_CMD_GET_REPORT0 ioctl failed on {dev_path}: {exc}") from exc

        return bytes(req.tdreport)

    def _get_quote_ioctl(self, dev_path: str, report_data: bytes, temporary_buffers: List[Any]) -> bytes:
        """Invokes TDX_CMD_GET_QUOTE ioctl when supported by driver."""
        if fcntl is None:
            raise AttestationGenerationError("fcntl module is only available on POSIX systems")

        td_report = self._get_td_report0_ioctl(dev_path, report_data, temporary_buffers)

        quote_buf_len = 8192
        out_buf = (ctypes.c_uint8 * quote_buf_len)()
        temporary_buffers.append(out_buf)

        ctypes.memmove(out_buf, td_report, len(td_report))

        req = tdx_quote_req()
        temporary_buffers.append(req)
        req.buf = ctypes.cast(out_buf, ctypes.c_void_p).value or 0
        req.len = quote_buf_len

        try:
            with open(dev_path, "r+b", buffering=0) as fd:
                fcntl.ioctl(fd.fileno(), TDX_CMD_GET_QUOTE, req)
            return bytes(out_buf)[:req.len]
        except OSError as exc:
            raise AttestationGenerationError(f"TDX_CMD_GET_QUOTE ioctl failed: {exc}") from exc

    def _get_quote_vsock(self, td_report: bytes) -> bytes:
        """Transmits TDREPORT to host Quote Generation Service (QGS) daemon via AF_VSOCK."""
        return fetch_quote_vsock(td_report)

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
        self, report_data: bytes, nonce: bytes, public_key: bytes
    ) -> Tuple[bytes, List[str]]:
        """Generates authentic synthetic Quote and PKI chain for offline hermetic testing."""
        fixtures = self._get_pki_fixtures()
        if fixtures is None:
            raise AttestationGenerationError(
                "Synthetic Intel TDX generation requires matrix_compute.crypto.pki_fixtures"
            )

        pki = fixtures.get_default_intel_pki()

        mrtd_bytes = bytes.fromhex(self._golden_rim.get("mrtd", "")) if "mrtd" in self._golden_rim else None
        rtmr0_bytes = bytes.fromhex(self._golden_rim.get("rtmr0", "")) if "rtmr0" in self._golden_rim else None
        rtmr1_bytes = bytes.fromhex(self._golden_rim.get("rtmr1", "")) if "rtmr1" in self._golden_rim else None
        rtmr2_bytes = bytes.fromhex(self._golden_rim.get("rtmr2", "")) if "rtmr2" in self._golden_rim else None
        rtmr3_bytes = bytes.fromhex(self._golden_rim.get("rtmr3", "")) if "rtmr3" in self._golden_rim else None

        quote_blob = fixtures.generate_synthetic_tdx_quote(
            pki=pki,
            quote_version=self.quote_version,
            nonce=nonce,
            worker_ephemeral_pubkey=public_key,
            report_data=report_data,
            debug=False,
            mrtd=mrtd_bytes,
            rtmr0=rtmr0_bytes,
            rtmr1=rtmr1_bytes,
            rtmr2=rtmr2_bytes,
            rtmr3=rtmr3_bytes,
        )

        cert_chain = self._cert_chain or pki.certificate_chain
        return quote_blob, cert_chain


# Register aliases for provider type resolution
register_attestation_provider("tdx")(IntelTdxProvider)
register_attestation_provider("intel")(IntelTdxProvider)
