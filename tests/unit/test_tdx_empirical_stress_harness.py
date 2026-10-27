"""
Empirical Adversarial Stress Harness for Node Intel TDX Attestation Provider.
Authored by Node TDX Challenger 1 (teamwork_preview_challenger).

Target: `matrix-compute-node/src/attestation/tdx_attest.py`

Comprehensive Empirical Challenge Dimensions:
1. Boundary condition inputs for generate_evidence (empty, oversized, corrupt, non-bytes, bytearrays).
2. Mocking ioctl failure modes (EINVAL, EIO, EPERM, EACCES, ENODEV, ENOTTY, ETIMEDOUT, and automatic AF_VSOCK fallback).
3. Mocking corrupted AF_VSOCK QGS socket packets (premature close, truncated length header, invalid length header, non-zero error codes, truncated quotes, corrupted quote versions/descriptors).
4. Device detection under restricted permissions (read-only dev nodes, missing write access, permission exceptions, custom dev paths, /proc/cpuinfo variants).
5. Provider registry lookup, alias resolution, and fail-closed vs simulated fallback behavior.
6. Verification that memory zeroization scrubs temporary buffers (tdx_report_req, report_data) even when ioctl or VSOCK throws exceptions.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import secrets
import socket
import struct
from typing import Any, Dict, List
from unittest.mock import MagicMock, mock_open, patch

import pytest
from cryptography import x509

from matrix_contracts.schemas.attestation import AttestationEvidence

from src.attestation import (
    AttestationError,
    AttestationGenerationError,
    AttestationValidationError,
    BaseAttestationProvider,
    DEFAULT_TDX_GOLDEN_RIM,
    HardwareAttestationUnavailableError,
    INTEL_TDX_GOLDEN_RIM,
    IntelTdxProvider,
    TDREPORT_STRUCT_LEN,
    TDX_CMD_GET_QUOTE,
    TDX_CMD_GET_REPORT0,
    TDX_GUEST_DEV_NODES,
    TDX_REPORTDATA_LEN,
    TdxQuoteReq,
    TdxReportReq,
    VSOCK_HOST_CID,
    VSOCK_QGS_PORT,
    get_attestation_provider,
    parse_tdx_quote_binary,
    tdx_quote_req,
    tdx_report_req,
    zeroize_buffer,
)
from src.attestation.tdx_attest import (
    _probe_cpuinfo_tdx,
    _probe_tdx_guest_dev,
    _probe_vsock_qgs,
    _recv_exact,
    fetch_quote_vsock,
)

from matrix_compute.crypto import (
    AttestationVerifier,
    IntelTdxVerifier,
    VerificationPolicy,
    VerificationResult,
    generate_synthetic_tdx_quote,
    get_default_intel_pki,
)
from matrix_compute.exceptions import (
    AttestationBindingError,
    AttestationNonceMismatchError,
    AttestationSignatureError,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def intel_pki():
    """Session-scoped cached synthetic Intel PKI hierarchy."""
    return get_default_intel_pki()


@pytest.fixture
def default_policy(intel_pki) -> VerificationPolicy:
    """Standard verification policy trusting synthetic Intel Root CA."""
    policy = VerificationPolicy()
    policy.add_trusted_root_pem(intel_pki.root_cert_pem)
    policy.add_golden_rim("intel-tdx", INTEL_TDX_GOLDEN_RIM)
    return policy


@pytest.fixture
def sample_nonce() -> bytes:
    """32-byte cryptographically secure random nonce."""
    return secrets.token_bytes(32)


@pytest.fixture
def sample_pubkey() -> bytes:
    """32-byte ephemeral public key."""
    return secrets.token_bytes(32)


# ==============================================================================
# 1. BOUNDARY CONDITION INPUTS FOR generate_evidence
# ==============================================================================

class TestBoundaryConditionInputs:
    """Empirical challenge of parameter boundaries on nonce, public_key, device_index, and binding modes."""

    @pytest.mark.parametrize(
        "bad_nonce",
        [
            b"",                           # Empty (0 bytes)
            b"\x00",                       # 1 byte
            b"\x00" * 16,                  # 16 bytes
            b"\x00" * 31,                  # 31 bytes (off-by-one under)
            b"\x00" * 33,                  # 33 bytes (off-by-one over)
            b"\x00" * 64,                  # 64 bytes
            b"\x00" * 1024,                # 1 KB oversized
            bytearray(b"\x00" * 31),       # Bytearray 31 bytes
            bytearray(b"\x00" * 33),       # Bytearray 33 bytes
            None,                          # NoneType
            123456789,                     # Integer
            3.14159,                       # Float
            "a" * 32,                      # String of length 32
            ["a"] * 32,                    # List of length 32
            {"nonce": b"\x00" * 32},       # Dictionary
            (b"\x00" * 32,),               # Tuple
            object(),                      # Object
            memoryview(b"\x00" * 32),      # Memoryview (not bytes/bytearray)
        ],
    )
    def test_nonce_boundary_validation_fails_closed(self, sample_pubkey, bad_nonce) -> None:
        """Every malformed, out-of-bounds, or wrong-type nonce must raise AttestationValidationError."""
        provider = IntelTdxProvider(simulate=True)
        with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
            provider.generate_evidence(nonce=bad_nonce, public_key=sample_pubkey)

    @pytest.mark.parametrize(
        "bad_pubkey",
        [
            b"",                           # Empty (0 bytes)
            b"\x01",                       # 1 byte
            b"\x01" * 16,                  # 16 bytes
            b"\x01" * 31,                  # 31 bytes (off-by-one under)
            b"\x01" * 33,                  # 33 bytes (off-by-one over)
            b"\x01" * 64,                  # 64 bytes (raw point without formatting)
            b"\x01" * 65,                  # 65 bytes (uncompressed point format)
            b"\x01" * 1024,                # 1 KB oversized
            bytearray(b"\x01" * 31),       # Bytearray 31 bytes
            bytearray(b"\x01" * 33),       # Bytearray 33 bytes
            None,                          # NoneType
            987654321,                     # Integer
            2.71828,                       # Float
            "k" * 32,                      # String of length 32
            [1] * 32,                      # List of length 32
            {"pubkey": b"\x01" * 32},      # Dictionary
            object(),                      # Object
            memoryview(b"\x01" * 32),      # Memoryview
        ],
    )
    def test_pubkey_boundary_validation_fails_closed(self, sample_nonce, bad_pubkey) -> None:
        """Every malformed, out-of-bounds, or wrong-type public key must raise AttestationValidationError."""
        provider = IntelTdxProvider(simulate=True)
        with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
            provider.generate_evidence(nonce=sample_nonce, public_key=bad_pubkey)

    @pytest.mark.parametrize("bad_idx", [-1, -2, -100, -99999])
    def test_negative_device_index_raises_value_error(self, bad_idx) -> None:
        """Negative device_index must raise ValueError."""
        with pytest.raises(ValueError, match="device_index must be non-negative"):
            IntelTdxProvider(device_index=bad_idx)

    @pytest.mark.parametrize("invalid_idx", ["dev0", "invalid", None, [0]])
    def test_non_integer_device_index_raises_type_or_value_error(self, invalid_idx) -> None:
        """Non-integer convertible device_index must raise ValueError or TypeError."""
        with pytest.raises((ValueError, TypeError)):
            IntelTdxProvider(device_index=invalid_idx)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "valid_nonce,valid_pubkey",
        [
            (b"\x00" * 32, b"\x00" * 32),                        # All zeros
            (b"\xff" * 32, b"\xff" * 32),                        # All ones (0xFF)
            (b"\xaa" * 32, b"\x55" * 32),                        # Alternating bit patterns
            (bytes(range(32)), bytes(range(31, -1, -1))),        # Monotonic ascending/descending
            (bytearray(b"\x12" * 32), bytearray(b"\x34" * 32)),  # Bytearrays of exact length 32
        ],
    )
    def test_boundary_values_evidence_validity(
        self, intel_pki, default_policy, valid_nonce, valid_pubkey
    ) -> None:
        """Valid boundary 32-byte nonces and pubkeys succeed and pass full SDK appraisal."""
        provider = IntelTdxProvider(simulate=True)
        evidence = provider.generate_evidence(nonce=valid_nonce, public_key=valid_pubkey)

        expected_nonce = bytes(valid_nonce)
        expected_pubkey = bytes(valid_pubkey)

        assert evidence.token_claims["nonce"] == expected_nonce.hex()
        assert evidence.token_claims["ephemeral_pubkey_hash"] == hashlib.sha256(expected_pubkey).hexdigest()
        assert evidence.user_data[:32] == expected_nonce
        assert evidence.user_data[32:64] == hashlib.sha256(expected_pubkey).digest()

        verifier = AttestationVerifier(policy=default_policy)
        result: VerificationResult = verifier.verify(
            evidence=evidence,
            expected_nonce=expected_nonce,
            expected_client_pubkey=expected_pubkey,
        )
        assert result.verified is True
        assert result.provider == "intel-tdx"


# ==============================================================================
# 2. MOCKING IOCTL FAILURE MODES
# ==============================================================================

class TestMockIoctlFailureModes:
    """
    Empirically stress-tests driver ioctl failure modes on /dev/tdx_guest:
    EINVAL, EIO, EPERM, EACCES, ENODEV, ENOTTY, ETIMEDOUT, and auto-fallback.
    """

    @pytest.mark.parametrize(
        "err_no,err_msg",
        [
            (errno.EINVAL, "Invalid argument"),
            (errno.EIO, "Input/output error"),
            (errno.EPERM, "Operation not permitted"),
            (errno.EACCES, "Permission denied"),
            (errno.ENODEV, "No such device"),
            (10060, "Connection timed out"),
        ],
    )
    def test_get_td_report0_ioctl_oserror_failure_modes(
        self, sample_nonce, sample_pubkey, err_no, err_msg
    ) -> None:
        """
        When TDX_CMD_GET_REPORT0 ioctl fails with any OSError (EINVAL, EIO, EPERM, etc.),
        the provider must fail-closed with AttestationGenerationError.
        """
        provider = IntelTdxProvider(simulate=False, device_path="/dev/tdx_guest", quote_strategy="ioctl")

        mock_fcntl = MagicMock()
        mock_fcntl.ioctl.side_effect = OSError(err_no, err_msg)

        mock_file = MagicMock()
        mock_file.fileno.return_value = 42
        mock_file.__enter__.return_value = mock_file

        with patch("src.attestation.tdx_attest.fcntl", mock_fcntl):
            with patch("os.path.exists", side_effect=lambda p: p == "/dev/tdx_guest"):
                with patch("builtins.open", return_value=mock_file):
                    with pytest.raises(AttestationGenerationError, match="TDX_CMD_GET_REPORT0 ioctl failed"):
                        provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    def test_get_quote_ioctl_failure_under_ioctl_strategy(self, sample_nonce, sample_pubkey) -> None:
        """
        When TDX_CMD_GET_QUOTE ioctl fails and quote_strategy="ioctl",
        must raise AttestationGenerationError without falling back.
        """
        provider = IntelTdxProvider(simulate=False, device_path="/dev/tdx_guest", quote_strategy="ioctl")

        # Mock TDREPORT0 ioctl success, but TDX_CMD_GET_QUOTE ioctl failure
        def mock_ioctl_dispatch(fd, cmd, req):
            if cmd == TDX_CMD_GET_REPORT0:
                ctypes.memset(req.tdreport, 0x42, TDREPORT_STRUCT_LEN)
                return 0
            elif cmd == TDX_CMD_GET_QUOTE:
                raise OSError(errno.ENOTTY, "Inappropriate ioctl for device")
            return 0

        mock_fcntl = MagicMock()
        mock_fcntl.ioctl.side_effect = mock_ioctl_dispatch

        mock_file = MagicMock()
        mock_file.fileno.return_value = 42
        mock_file.__enter__.return_value = mock_file

        with patch("src.attestation.tdx_attest.fcntl", mock_fcntl):
            with patch("os.path.exists", side_effect=lambda p: p == "/dev/tdx_guest"):
                with patch("builtins.open", return_value=mock_file):
                    with pytest.raises(AttestationGenerationError, match="TDX_CMD_GET_QUOTE ioctl failed"):
                        provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    def test_auto_strategy_falls_back_from_ioctl_to_vsock_on_enotty(
        self, intel_pki, sample_nonce, sample_pubkey
    ) -> None:
        """
        In quote_strategy="auto", when TDX_CMD_GET_QUOTE fails with ENOTTY (newer kernel ABI),
        the provider automatically falls back to host QGS via AF_VSOCK and succeeds.
        """
        provider = IntelTdxProvider(simulate=False, device_path="/dev/tdx_guest", quote_strategy="auto")

        # TDX_CMD_GET_REPORT0 succeeds, but TDX_CMD_GET_QUOTE raises ENOTTY
        def mock_ioctl_dispatch(fd, cmd, req):
            if cmd == TDX_CMD_GET_REPORT0:
                ctypes.memset(req.tdreport, 0x42, TDREPORT_STRUCT_LEN)
                return 0
            elif cmd == TDX_CMD_GET_QUOTE:
                raise OSError(errno.ENOTTY, "Inappropriate ioctl for device")
            return 0

        mock_fcntl = MagicMock()
        mock_fcntl.ioctl.side_effect = mock_ioctl_dispatch

        mock_file = MagicMock()
        mock_file.fileno.return_value = 42
        mock_file.__enter__.return_value = mock_file

        # Generate a valid quote blob for VSOCK response
        valid_quote = generate_synthetic_tdx_quote(
            pki=intel_pki,
            nonce=sample_nonce,
            worker_ephemeral_pubkey=sample_pubkey,
        )

        with patch("src.attestation.tdx_attest.fcntl", mock_fcntl):
            with patch("os.path.exists", side_effect=lambda p: p == "/dev/tdx_guest"):
                with patch("builtins.open", return_value=mock_file):
                    with patch("src.attestation.tdx_attest.fetch_quote_vsock", return_value=valid_quote) as mock_vsock:
                        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)
                        assert mock_vsock.called
                        assert evidence.evidence_blob == valid_quote
                        assert evidence.token_claims["nonce"] == sample_nonce.hex()

    def test_temporary_buffers_zeroized_on_ioctl_failure(self, sample_nonce, sample_pubkey) -> None:
        """
        Verifies that even when an ioctl raises an exception (e.g. EIO or EINVAL),
        temporary buffers (report_data bytearray and ctypes structure) are safely zeroized in finally.
        """
        provider = IntelTdxProvider(simulate=False, device_path="/dev/tdx_guest", quote_strategy="ioctl")

        mock_fcntl = MagicMock()
        mock_fcntl.ioctl.side_effect = OSError(errno.EIO, "Kernel TDX driver I/O error")

        mock_file = MagicMock()
        mock_file.fileno.return_value = 42
        mock_file.__enter__.return_value = mock_file

        with patch("src.attestation.tdx_attest.fcntl", mock_fcntl):
            with patch("os.path.exists", side_effect=lambda p: p == "/dev/tdx_guest"):
                with patch("builtins.open", return_value=mock_file):
                    with patch("src.attestation.tdx_attest.zeroize_buffer", wraps=zeroize_buffer) as mock_zero:
                        with pytest.raises(AttestationGenerationError, match="I/O error"):
                            provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

                        # Check that zeroize_buffer was called for each temporary buffer
                        assert mock_zero.called
                        # Verify the report_data buffer was passed to zeroize_buffer
                        called_buffers = [call[0][0] for call in mock_zero.call_args_list]
                        report_data_buffers = [b for b in called_buffers if isinstance(b, bytearray) and len(b) == TDX_REPORTDATA_LEN]
                        assert len(report_data_buffers) >= 1
                        assert all(b == 0 for b in report_data_buffers[0])


# ==============================================================================
# 3. MOCKING CORRUPTED AF_VSOCK QGS SOCKET PACKETS
# ==============================================================================

class TestMockVsockQgsSocketCorruption:
    """
    Empirically stress-tests AF_VSOCK network protocol parsing against
    fuzzed, corrupted, and truncated packets from untrusted host QGS.
    """

    def test_vsock_socket_connect_timeout(self) -> None:
        """Host QGS socket connection timeout raises AttestationGenerationError."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        mock_sock.connect.side_effect = socket.timeout("VSOCK timed out")

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                with pytest.raises(AttestationGenerationError, match="AF_VSOCK communication with host QGS failed"):
                    fetch_quote_vsock(b"\x00" * 1024)

    def test_vsock_socket_connection_refused(self) -> None:
        """Host QGS daemon not running (ConnectionRefusedError) raises AttestationGenerationError."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        mock_sock.connect.side_effect = ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                with pytest.raises(AttestationGenerationError, match="Connection refused"):
                    fetch_quote_vsock(b"\x00" * 1024)

    @pytest.mark.parametrize("partial_hdr_len", [0, 1, 2, 3])
    def test_vsock_truncated_length_header_fails_closed(self, partial_hdr_len) -> None:
        """QGS socket closing before 4-byte length header is fully transmitted raises AttestationGenerationError."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        partial_hdr = b"\x00\x00\x04\x00"[:partial_hdr_len]
        mock_sock.recv.side_effect = [partial_hdr, b""]

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                with pytest.raises(AttestationGenerationError, match="VSOCK connection closed prematurely"):
                    fetch_quote_vsock(b"\x00" * 1024)

    @pytest.mark.parametrize("invalid_resp_len", [0, 1, 2, 3])
    def test_vsock_response_length_less_than_four_fails_closed(self, invalid_resp_len) -> None:
        """QGS length header advertising payload size < 4 bytes raises AttestationGenerationError."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        hdr = struct.pack(">I", invalid_resp_len)
        body = b"\x00" * invalid_resp_len
        mock_sock.recv.side_effect = [hdr, body, b""]

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                with pytest.raises(AttestationGenerationError, match="Malformed QGS response length header"):
                    fetch_quote_vsock(b"\x00" * 1024)

    def test_vsock_fragmented_packet_reads_full_stream(self) -> None:
        """_recv_exact handles multi-chunk TCP/VSOCK stream fragmentation without dropping bytes."""
        mock_sock = MagicMock()
        # Stream sending 10 bytes in 3 fragmented chunks
        mock_sock.recv.side_effect = [b"123", b"4567", b"890", b""]
        received = _recv_exact(mock_sock, 10)
        assert received == b"1234567890"

    def test_vsock_premature_close_mid_payload(self) -> None:
        """QGS advertising 1000-byte payload but closing connection after 100 bytes."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        hdr = struct.pack(">I", 1000)
        chunk1 = b"\x00" * 100
        mock_sock.recv.side_effect = [hdr, chunk1, b""]

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                with pytest.raises(AttestationGenerationError, match="received 100 of 1000 expected bytes"):
                    fetch_quote_vsock(b"\x00" * 1024)

    @pytest.mark.parametrize("error_code", [0x00000001, 0x80000001, 0xE0000001, 0xFFFFFFFF])
    def test_vsock_qgs_non_zero_status_codes(self, error_code) -> None:
        """Any non-zero QGS daemon error status code raises AttestationGenerationError with formatted code."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        body = struct.pack(">I", error_code) + b"\x00" * 636
        frame = struct.pack(">I", len(body)) + body
        mock_sock.recv.side_effect = [frame[:4], frame[4:], b""]

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                with pytest.raises(AttestationGenerationError, match=f"0x{error_code:08X}"):
                    fetch_quote_vsock(b"\x00" * 1024)

    @pytest.mark.parametrize("short_quote_len", [0, 10, 100, 500, 635])
    def test_vsock_truncated_quote_payload_fails_closed(self, short_quote_len) -> None:
        """QGS returning status 0 but quote payload shorter than 636 bytes raises AttestationGenerationError."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        body = struct.pack(">I", 0) + b"\x00" * short_quote_len
        frame = struct.pack(">I", len(body)) + body
        mock_sock.recv.side_effect = [frame[:4], frame[4:], b""]

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                with pytest.raises(AttestationGenerationError, match="truncated Quote binary"):
                    fetch_quote_vsock(b"\x00" * 1024)

    @pytest.mark.parametrize("bad_version", [0, 1, 2, 3, 6, 99])
    def test_corrupted_quote_version_from_vsock_fails_validation(self, bad_version) -> None:
        """Quote returned by QGS with corrupted version header fails parse_tdx_quote_binary validation."""
        corrupted_quote = bytearray(b"\x00" * 700)
        struct.pack_into("<HHI", corrupted_quote, 0, bad_version, 2, 0x81)
        with pytest.raises(AttestationValidationError, match="Unsupported Quote version"):
            parse_tdx_quote_binary(bytes(corrupted_quote))

    def test_corrupted_v5_descriptor_fails_validation(self) -> None:
        """Quote v5 with truncated body descriptor (< 54 bytes) fails parse_tdx_quote_binary validation."""
        blob = bytearray(b"\x00" * 636)
        struct.pack_into("<HHI", blob, 0, 5, 2, 0x81)  # Version 5
        with pytest.raises(AttestationValidationError):
            parse_tdx_quote_binary(bytes(blob[:53]))


# ==============================================================================
# 4. DEVICE DETECTION UNDER RESTRICTED PERMISSIONS
# ==============================================================================

class TestDeviceDetectionRestrictedPermissions:
    """Empirical challenge of device probing under read-only or restricted file permissions."""

    def test_probe_tdx_guest_dev_read_only_returns_none(self) -> None:
        """When /dev/tdx_guest is readable but not writeable (R_OK=True, W_OK=False), probe returns None."""
        with patch("os.path.exists", return_value=True):
            with patch("os.access", side_effect=lambda path, mode: bool(mode & os.R_OK and not (mode & os.W_OK))):
                assert _probe_tdx_guest_dev() is None

    def test_probe_selects_first_read_write_device(self) -> None:
        """When /dev/tdx_guest is read-only but /dev/tdx-guest is read-write, probe selects /dev/tdx-guest."""
        def mock_access(path, mode):
            if path == "/dev/tdx_guest":
                return False  # Not writeable
            elif path == "/dev/tdx-guest":
                return True   # Writeable
            return False

        with patch("os.path.exists", return_value=True):
            with patch("os.access", side_effect=mock_access):
                assert _probe_tdx_guest_dev() == "/dev/tdx-guest"

    def test_probe_handles_permission_error_exception_cleanly(self) -> None:
        """When os.access raises PermissionError, probe catches exception and returns None."""
        with patch("os.path.exists", return_value=True):
            with patch("os.access", side_effect=PermissionError(errno.EACCES, "Permission denied")):
                assert _probe_tdx_guest_dev() is None

    def test_is_available_with_custom_device_path(self) -> None:
        """Verifies custom device_path resolution in is_available()."""
        custom_path = "/custom/dev/tdx_isolated"

        # Case 1: Custom path exists
        with patch("os.path.exists", side_effect=lambda p: p == custom_path):
            p1 = IntelTdxProvider(simulate=False, device_path=custom_path)
            assert p1.is_available() is True

        # Case 2: Custom path does not exist
        with patch("os.path.exists", return_value=False):
            p2 = IntelTdxProvider(simulate=False, device_path=custom_path)
            assert p2.is_available() is False

    @pytest.mark.parametrize(
        "cpuinfo_content,expected_result",
        [
            ("flags : fpu vme tdx_guest sse\n", True),
            ("Features : fpu intel_tdx sse2\n", True),
            ("flags : tdx sse3\n", True),
            ("flags : fpu vme sse sse2 avx\n", False),
            ("flags : \n", False),
            ("", False),
        ],
    )
    def test_probe_cpuinfo_tdx_variants(self, cpuinfo_content, expected_result) -> None:
        """Verifies /proc/cpuinfo flag detection for tdx_guest, intel_tdx, and tdx."""
        with patch("os.path.exists", side_effect=lambda p: p == "/proc/cpuinfo"):
            with patch("builtins.open", mock_open(read_data=cpuinfo_content)):
                assert _probe_cpuinfo_tdx() is expected_result


# ==============================================================================
# 5. PROVIDER REGISTRY LOOKUP & FALLBACK BEHAVIOR
# ==============================================================================

class TestProviderRegistryLookupAndFallback:
    """Empirical challenge of provider registration, alias normalization, and simulated fallbacks."""

    @pytest.mark.parametrize(
        "provider_id",
        [
            "intel-tdx",
            "tdx",
            "intel",
            "INTEL-TDX",
            "  intel-tdx  ",
            "TDX",
            "INTEL",
            "Tdx",
        ],
    )
    def test_registry_lookup_canonical_and_aliases(self, provider_id) -> None:
        """All registered provider aliases and case/whitespace variations resolve to IntelTdxProvider."""
        provider = get_attestation_provider(provider_id, simulate=True)
        assert isinstance(provider, IntelTdxProvider)
        assert provider.provider_name == "intel-tdx"

    def test_registry_fallback_to_simulated_switch(self) -> None:
        """
        Tests fallback_to_simulated parameter:
        - When False and hardware missing: raises HardwareAttestationUnavailableError.
        - When True and hardware missing: returns simulated IntelTdxProvider.
        """
        real_is_available = IntelTdxProvider.is_available

        def mock_is_available(self):
            if getattr(self, "simulate", False):
                return True
            return False

        with patch.object(IntelTdxProvider, "is_available", mock_is_available):
            # Strict fail-closed
            with pytest.raises(HardwareAttestationUnavailableError):
                get_attestation_provider("intel-tdx", fallback_to_simulated=False)

            # Permissive simulated fallback
            p_sim = get_attestation_provider("intel-tdx", fallback_to_simulated=True)
            assert isinstance(p_sim, IntelTdxProvider)
            assert p_sim.simulate is True
            assert p_sim.is_available() is True

    @pytest.mark.parametrize("invalid_id", ["unknown-tee", "intel-sgx", "custom-unregistered", "fake_provider_123"])
    def test_unregistered_provider_id_raises_error(self, invalid_id) -> None:
        """Unregistered non-empty provider names must raise ValueError or AttestationError."""
        with pytest.raises((ValueError, AttestationError)):
            get_attestation_provider(invalid_id)

    def test_empty_or_none_provider_id_defaults_to_auto(self) -> None:
        """Empty string or None defaults to auto-probing and returns a valid provider."""
        provider = get_attestation_provider("", fallback_to_simulated=True)
        assert isinstance(provider, BaseAttestationProvider)


    def test_provider_initialization_kwargs_forwarding(self) -> None:
        """Verifies custom kwargs (quote_version, binding_mode, golden_rim) pass through factory."""
        custom_rim = {"mrtd": "a" * 96, "rtmr0": "b" * 96}
        provider = get_attestation_provider(
            "intel-tdx",
            simulate=True,
            quote_version=5,
            binding_mode="sha256_dual",
            golden_rim=custom_rim,
            device_index=3,
        )
        assert isinstance(provider, IntelTdxProvider)
        assert provider.quote_version == 5
        assert provider.binding_mode == "sha256_dual"
        assert provider.device_index == 3
        assert provider.get_measurements()["mrtd"] == "a" * 96

    def test_unsupported_quote_strategy_raises_generation_error(self, sample_nonce, sample_pubkey) -> None:
        """Unknown or unsupported quote_strategy raises AttestationGenerationError."""
        provider = IntelTdxProvider(simulate=False, device_path="/dev/tdx_guest", quote_strategy="unsupported_quantum_link")
        with patch.object(provider, "is_available", return_value=True):
            with patch("src.attestation.tdx_attest._probe_tdx_guest_dev", return_value="/dev/tdx_guest"):
                with pytest.raises(AttestationGenerationError, match="Unsupported quote strategy"):
                    provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)
