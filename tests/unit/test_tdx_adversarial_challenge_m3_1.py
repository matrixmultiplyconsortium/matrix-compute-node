"""
Adversarial and Stress Test Harness for Intel TDX Attestation Provider (Milestone 3 Challenge).
Authored by Milestone 3 Challenger 1 (teamwork_preview_challenger).

Empirically challenges:
1. Non-Hardware Probe Safety under Multiple Mock Filesystem and Network States:
   - Windows OS state (sys.platform == 'win32', no fcntl, no /proc/cpuinfo, no /dev/tdx*, AF_VSOCK absent/unreachable).
   - macOS (Darwin) state (no /proc/cpuinfo, no /dev/tdx*, AF_VSOCK absent).
   - Non-TDX Linux state (/proc/cpuinfo present without TDX flags, no /dev/tdx*, AF_VSOCK unreachable).
   - Linux with TDX CPU flags in /proc/cpuinfo but missing /dev/tdx* and unreachable QGS.
   - Linux with /dev/tdx_guest present but inaccessible due to permission restrictions (os.access returns False).
   - Mock filesystem I/O and permission errors (PermissionError, UnicodeDecodeError, OSError) during probing.
   - AF_VSOCK network probe failures (connection timeout, connection refused, host unreachable, permission denied, address family not supported).
   - Custom device_path probe failures and fail-closed behavior.
   - Factory fallback to simulated provider when hardware is missing.

2. Nonce and Pubkey Validation Boundaries:
   - Non-bytes types (int, str, None, list, dict, float, tuple, object, memoryview) for nonce and public key.
   - Boundary lengths for nonce and pubkey (0, 1, 16, 31, 33, 48, 64, 1024 bytes).
   - Extreme boundary byte values (all-0x00, all-0xFF, alternating 0xAA/0x55, high-entropy tokens).
   - Negative and non-integer device_index values (-1, -999, invalid types).
   - Dual vs SHA-256 dual binding mode 64-byte REPORTDATA wire formatting and cryptographic appraisal.

3. Concurrency and Thread Safety during Evidence Generation:
   - Multi-threaded stress testing: 25 concurrent threads calling generate_evidence on a shared IntelTdxProvider.
   - Verification of 100 generated evidence blobs for strict thread isolation (matching nonce, matching pubkey hash).
   - Dual ECDSA signature and claims verification on all concurrently generated evidence.
   - Concurrent MatrixComputeServicer RequestTileTask handling with required_tee_type="intel-tdx".
   - Concurrency safety during simultaneous get_measurements reads and evidence generation.

4. Memory Zeroization Edge Cases:
   - zeroize_buffer behavior on None, empty bytearrays, large bytearrays (100KB), and writeable memoryviews.
   - Read-only memoryviews and immutable types (bytes, str, int, list, dict) handled safely without crashing.
   - Ctypes structures (tdx_report_req, tdx_quote_req, custom arrays) memory scrubbing via ctypes.memset.
   - Objects exposing __array_interface__ memory scrubbing.
   - Fault injection during zeroize_buffer handling exceptions safely.
   - Verification of internal temporary buffer zeroization in generate_evidence under success and exception.

5. AF_VSOCK Transport & Binary Quote Parser Robustness:
   - Premature EOF, truncated frames, malformed length headers, and QGS error status codes in fetch_quote_vsock.
   - parse_tdx_quote_binary boundary fuzzing: truncated blobs, unsupported versions, invalid descriptors, and corrupt PEM blocks.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import hashlib
import os
import secrets
import socket
import struct
import sys
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, mock_open, patch

import pytest
from cryptography import x509

from matrix_contracts.schemas.attestation import AttestationEvidence
from matrix_contracts.v1 import pb2

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
from src.p2p.servicer import MatrixComputeServicer

from matrix_compute.crypto import (
    AttestationVerifier,
    IntelTdxVerifier,
    VerificationPolicy,
    VerificationResult,
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
    """Cached singleton synthetic Intel PKI hierarchy."""
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
# 1. NON-HARDWARE PROBE SAFETY UNDER MULTIPLE MOCK PLATFORMS & STATES
# ==============================================================================

class TestNonHardwareProbeSafetyAdversarial:
    """
    Stress-tests platform detection and environment probing across simulated Windows,
    macOS, and non-TDX Linux states, verifying zero unhandled exceptions and fail-closed
    behavior on missing hardware.
    """

    def test_mock_windows_environment(self, sample_nonce, sample_pubkey) -> None:
        """Verifies clean, exception-free False return on simulated Windows platform."""
        with patch("sys.platform", "win32"):
            with patch("os.path.exists", return_value=False):
                with patch("src.attestation.tdx_attest.fcntl", None):
                    # Ensure hasattr(socket, 'AF_VSOCK') is False or socket connect fails
                    with patch("socket.socket", side_effect=OSError(10047, "Address family not supported")):
                        provider = IntelTdxProvider(simulate=False)
                        assert provider.is_available() is False
                        with pytest.raises(HardwareAttestationUnavailableError, match="unavailable"):
                            provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    def test_mock_macos_environment(self, sample_nonce, sample_pubkey) -> None:
        """Verifies clean, exception-free False return on simulated macOS (Darwin) platform."""
        with patch("sys.platform", "darwin"):
            with patch("os.path.exists", return_value=False):
                with patch("src.attestation.tdx_attest._probe_vsock_qgs", return_value=False):
                    provider = IntelTdxProvider(simulate=False)
                    assert provider.is_available() is False
                    with pytest.raises(HardwareAttestationUnavailableError):
                        provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    def test_mock_non_tdx_linux_environment(self, sample_nonce, sample_pubkey) -> None:
        """Verifies clean rejection on standard Linux host with standard CPU features (no TDX)."""
        non_tdx_cpuinfo = (
            "processor\t: 0\n"
            "vendor_id\t: GenuineIntel\n"
            "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov "
            "pat pse36 clflush mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp "
            "lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq "
            "pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 "
            "xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes "
            "xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch\n"
        )
        with patch("sys.platform", "linux"):
            with patch("os.path.exists", side_effect=lambda p: p == "/proc/cpuinfo"):
                with patch("builtins.open", mock_open(read_data=non_tdx_cpuinfo)):
                    with patch("src.attestation.tdx_attest._probe_vsock_qgs", return_value=False):
                        provider = IntelTdxProvider(simulate=False)
                        assert provider.is_available() is False
                        with pytest.raises(HardwareAttestationUnavailableError):
                            provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    def test_mock_linux_with_tdx_flag_but_missing_dev_nodes(self, sample_nonce, sample_pubkey) -> None:
        """CPU advertises TDX, but kernel driver /dev/tdx_guest is missing and QGS is down."""
        tdx_cpuinfo = "flags\t\t: fpu vme tdx_guest sse sse2\n"
        with patch("os.path.exists", side_effect=lambda p: p == "/proc/cpuinfo"):
            with patch("builtins.open", mock_open(read_data=tdx_cpuinfo)):
                with patch("src.attestation.tdx_attest._probe_vsock_qgs", return_value=False):
                    provider = IntelTdxProvider(simulate=False)
                    assert provider.is_available() is False
                    with pytest.raises(HardwareAttestationUnavailableError):
                        provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    def test_mock_linux_with_dev_node_permission_denied(self) -> None:
        """Device node exists in filesystem, but current user lacks read/write permissions."""
        with patch("os.path.exists", side_effect=lambda p: p == "/dev/tdx_guest"):
            with patch("os.access", return_value=False):  # EACCES / Permission denied
                with patch("src.attestation.tdx_attest._probe_cpuinfo_tdx", return_value=False):
                    with patch("src.attestation.tdx_attest._probe_vsock_qgs", return_value=False):
                        assert _probe_tdx_guest_dev() is None
                        provider = IntelTdxProvider(simulate=False)
                        assert provider.is_available() is False

    @pytest.mark.parametrize(
        "fs_exception",
        [
            PermissionError(13, "Permission denied: '/proc/cpuinfo'"),
            OSError(5, "Input/output error"),
            UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, "invalid start byte"),
            FileNotFoundError(2, "No such file or directory"),
        ],
    )
    def test_mock_filesystem_probe_exceptions_handled_cleanly(self, fs_exception) -> None:
        """Filesystem probe errors while reading /proc/cpuinfo must be handled without unhandled crashes."""
        with patch("os.path.exists", side_effect=lambda p: p == "/proc/cpuinfo"):
            with patch("builtins.open", side_effect=fs_exception):
                assert _probe_cpuinfo_tdx() is False

    @pytest.mark.parametrize(
        "net_exception",
        [
            socket.timeout("timed out"),
            ConnectionRefusedError(111, "Connection refused"),
            OSError(113, "No route to host"),
            OSError(97, "Address family not supported by protocol"),
            PermissionError(13, "Permission denied"),
        ],
    )
    def test_mock_vsock_network_probe_exceptions_handled_cleanly(self, net_exception) -> None:
        """Network exceptions during host QGS probing must be handled cleanly, returning False."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        mock_sock.connect.side_effect = net_exception

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                assert _probe_vsock_qgs(timeout=0.1) is False

    def test_mock_custom_device_path_nonexistent(self, sample_nonce, sample_pubkey) -> None:
        """User specifies a custom device_path that does not exist."""
        with patch("os.path.exists", return_value=False):
            provider = IntelTdxProvider(simulate=False, device_path="/dev/custom_tdx_device")
            assert provider.is_available() is False
            with pytest.raises(HardwareAttestationUnavailableError):
                provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

    def test_factory_fallback_and_fail_closed_switch(self) -> None:
        """Verifies get_attestation_provider enforces strict fail-closed vs fallback."""
        with patch("src.attestation.tdx_attest._probe_tdx_guest_dev", return_value=None):
            with patch("src.attestation.tdx_attest._probe_cpuinfo_tdx", return_value=False):
                with patch("src.attestation.tdx_attest._probe_vsock_qgs", return_value=False):
                    with patch("os.path.exists", return_value=False):
                        # Strict mode: raise error
                        with pytest.raises(HardwareAttestationUnavailableError):
                            get_attestation_provider("intel-tdx", fallback_to_simulated=False)

                        # Fallback mode: gracefully fall back to simulated provider
                        sim_p = get_attestation_provider("intel-tdx", fallback_to_simulated=True)
                        assert isinstance(sim_p, IntelTdxProvider)
                        assert sim_p.simulate is True
                        assert sim_p.is_available() is True


# ==============================================================================
# 2. NONCE AND PUBKEY VALIDATION BOUNDARIES
# ==============================================================================

class TestAdversarialNonceAndPubkeyBoundaries:
    """
    Stress-tests parameter boundary checks on challenger nonces, public keys,
    device indices, and cryptographic binding modes.
    """

    @pytest.mark.parametrize(
        "bad_nonce",
        [
            None,
            12345,
            3.14159,
            "nonce_as_string_of_32_bytes___!",
            [0] * 32,
            {"nonce": b"\x00" * 32},
            (b"\x00" * 32,),
            object(),
            memoryview(b"\x00" * 32),  # Memoryview is not bytes/bytearray instance
        ],
    )
    def test_non_bytes_nonce_types_rejected(self, sample_pubkey, bad_nonce) -> None:
        """Non-bytes/bytearray nonce types must raise AttestationValidationError."""
        provider = IntelTdxProvider(simulate=True)
        with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
            provider.generate_evidence(nonce=bad_nonce, public_key=sample_pubkey)

    @pytest.mark.parametrize(
        "bad_len",
        [0, 1, 15, 16, 31, 33, 48, 64, 128, 1024],
    )
    def test_invalid_length_nonces_rejected(self, sample_pubkey, bad_len) -> None:
        """Nonces with length != 32 bytes must raise AttestationValidationError."""
        provider = IntelTdxProvider(simulate=True)
        bad_nonce = b"\x55" * bad_len
        with pytest.raises(AttestationValidationError, match="nonce must be 32 bytes"):
            provider.generate_evidence(nonce=bad_nonce, public_key=sample_pubkey)

    @pytest.mark.parametrize(
        "bad_pubkey",
        [
            None,
            99999,
            2.71828,
            "pubkey_as_string_of_32_bytes__!",
            [1] * 32,
            {"pubkey": b"\x01" * 32},
            (b"\x01" * 32,),
            object(),
            memoryview(b"\x01" * 32),
        ],
    )
    def test_non_bytes_pubkey_types_rejected(self, sample_nonce, bad_pubkey) -> None:
        """Non-bytes/bytearray pubkey types must raise AttestationValidationError."""
        provider = IntelTdxProvider(simulate=True)
        with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
            provider.generate_evidence(nonce=sample_nonce, public_key=bad_pubkey)

    @pytest.mark.parametrize(
        "bad_len",
        [0, 1, 15, 16, 31, 33, 48, 64, 128, 1024],
    )
    def test_invalid_length_pubkeys_rejected(self, sample_nonce, bad_len) -> None:
        """Public keys with length != 32 bytes must raise AttestationValidationError."""
        provider = IntelTdxProvider(simulate=True)
        bad_pubkey = b"\xAA" * bad_len
        with pytest.raises(AttestationValidationError, match="public_key must be 32 bytes"):
            provider.generate_evidence(nonce=sample_nonce, public_key=bad_pubkey)

    @pytest.mark.parametrize("invalid_idx", [-1, -2, -100, -99999])
    def test_negative_device_index_rejected(self, invalid_idx) -> None:
        """Negative device_index must raise ValueError."""
        with pytest.raises(ValueError, match="device_index must be non-negative"):
            IntelTdxProvider(device_index=invalid_idx)

    @pytest.mark.parametrize("invalid_idx", ["gpu0", "invalid", None, [0]])
    def test_non_integer_device_index_rejected(self, invalid_idx) -> None:
        """Non-integer device_index must raise ValueError or TypeError."""
        with pytest.raises((ValueError, TypeError)):
            IntelTdxProvider(device_index=invalid_idx)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "boundary_nonce,boundary_pubkey",
        [
            (b"\x00" * 32, b"\x00" * 32),                     # All-zeros
            (b"\xFF" * 32, b"\xFF" * 32),                     # All-ones (0xFF)
            (b"\xAA" * 32, b"\x55" * 32),                     # Alternating bits
            (b"\x01" + b"\x00" * 31, b"\x80" + b"\x00" * 31), # Extreme bit positions
            (bytearray(b"\x12" * 32), bytearray(b"\x34" * 32)), # Mutable bytearrays
        ],
    )
    def test_boundary_byte_values_valid_and_verified(
        self, intel_pki, boundary_nonce, boundary_pubkey
    ) -> None:
        """Boundary 32-byte nonces and public keys format correctly and pass verification."""
        provider = IntelTdxProvider(simulate=True)
        evidence = provider.generate_evidence(nonce=boundary_nonce, public_key=boundary_pubkey)

        expected_nonce_bytes = bytes(boundary_nonce)
        expected_pubkey_bytes = bytes(boundary_pubkey)

        # Check evidence claims
        assert evidence.token_claims["nonce"] == expected_nonce_bytes.hex()
        assert evidence.token_claims["ephemeral_pubkey_hash"] == hashlib.sha256(expected_pubkey_bytes).hexdigest()

        # Check raw report_data in user_data
        assert evidence.user_data[:32] == expected_nonce_bytes
        assert evidence.user_data[32:64] == hashlib.sha256(expected_pubkey_bytes).digest()

        # Direct verification with IntelTdxVerifier
        leaf_cert = x509.load_pem_x509_certificate(evidence.certificate_chain[0].encode("utf-8"))
        verifier = IntelTdxVerifier()
        verifier.verify_token_signature(evidence, leaf_cert)
        claims = verifier.extract_claims(evidence)
        assert claims["nonce"] == expected_nonce_bytes.hex()
        assert claims["ephemeral_pubkey_hash"] == hashlib.sha256(expected_pubkey_bytes).hexdigest()

    def test_sha256_dual_binding_mode(self, intel_pki, sample_nonce, sample_pubkey) -> None:
        """Verifies sha256_dual binding mode embeds sha256(nonce) + sha256(pubkey)."""
        provider = IntelTdxProvider(simulate=True, binding_mode="sha256_dual")
        evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

        expected_nonce_hash = hashlib.sha256(sample_nonce).digest()
        expected_pubkey_hash = hashlib.sha256(sample_pubkey).digest()

        assert evidence.user_data[:32] == expected_nonce_hash
        assert evidence.user_data[32:64] == expected_pubkey_hash


# ==============================================================================
# 3. THREAD SAFETY AND CONCURRENCY DURING EVIDENCE GENERATION
# ==============================================================================

class TestAdversarialConcurrencyAndThreadSafety:
    """
    Stress-tests thread safety, memory isolation, and absence of cross-talk
    under heavy multi-threaded concurrent execution.
    """

    def test_concurrent_evidence_generation_thread_safety(self, intel_pki) -> None:
        """
        Launches 25 concurrent threads calling generate_evidence on the same IntelTdxProvider.
        Verifies zero race conditions, zero cross-thread pollution, and 100% cryptographic validity.
        """
        provider = IntelTdxProvider(simulate=True)
        num_threads = 25
        iterations_per_thread = 4
        barrier = threading.Barrier(num_threads)

        results: List[Dict[str, Any]] = []
        lock = threading.Lock()
        leaf_cert = x509.load_pem_x509_certificate(intel_pki.pck_cert_pem.encode("utf-8"))
        tdx_verifier = IntelTdxVerifier()

        def worker_task(thread_id: int) -> None:
            # Synchronize start to maximize contention
            barrier.wait()
            for i in range(iterations_per_thread):
                nonce = secrets.token_bytes(32)
                pubkey = secrets.token_bytes(32)
                evidence = provider.generate_evidence(nonce=nonce, public_key=pubkey)

                # Verify immediately
                tdx_verifier.verify_token_signature(evidence, leaf_cert)
                claims = tdx_verifier.extract_claims(evidence)

                assert claims["nonce"] == nonce.hex()
                assert claims["ephemeral_pubkey_hash"] == hashlib.sha256(pubkey).hexdigest()
                assert evidence.user_data[:32] == nonce
                assert evidence.user_data[32:64] == hashlib.sha256(pubkey).digest()

                with lock:
                    results.append({
                        "thread_id": thread_id,
                        "nonce": nonce,
                        "pubkey": pubkey,
                        "evidence": evidence,
                    })

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, tid) for tid in range(num_threads)]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Raises if any assertion failed inside a thread

        assert len(results) == num_threads * iterations_per_thread

        # Cross-check that every result has unique nonces and unique public keys
        all_nonces = [r["nonce"] for r in results]
        all_pubkeys = [r["pubkey"] for r in results]
        assert len(set(all_nonces)) == len(results)
        assert len(set(all_pubkeys)) == len(results)

    def test_concurrent_servicer_handshakes(self) -> None:
        """
        Launches 20 concurrent threads requesting tile task handshakes through MatrixComputeServicer.
        Verifies each handshake returns distinct worker pubkeys, matching nonces, and uncorrupted evidence.
        """
        provider = IntelTdxProvider(simulate=True)
        servicer = MatrixComputeServicer(attestation_provider=provider)

        num_threads = 20
        barrier = threading.Barrier(num_threads)
        acks: List[pb2.TileTaskAck] = []
        lock = threading.Lock()

        def servicer_worker(thread_id: int) -> None:
            barrier.wait()
            nonce = secrets.token_bytes(32)
            pubkey = secrets.token_bytes(32)
            req = pb2.TileTaskRequest(
                task_id=f"task-concurrent-{thread_id:04d}",
                matrix_a_rows=32,
                matrix_a_cols=32,
                matrix_b_cols=32,
                dtype="fp32",
                require_confidential_compute=True,
                client_ephemeral_pubkey=pubkey,
                attestation_nonce=nonce,
                required_tee_type="intel-tdx",
            )
            ack = servicer.RequestTileTask(req)
            assert ack.is_ready is True
            assert ack.tee_type == "intel-tdx"
            assert len(ack.worker_ephemeral_pubkey) == 32
            assert len(ack.attestation_evidence) > 0

            # Unpack and verify evidence claims
            ev = AttestationEvidence.from_bytes(ack.attestation_evidence)
            assert ev.token_claims["nonce"] == nonce.hex()
            worker_hash = hashlib.sha256(ack.worker_ephemeral_pubkey).hexdigest()
            assert ev.token_claims["ephemeral_pubkey_hash"] == worker_hash

            with lock:
                acks.append(ack)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(servicer_worker, tid) for tid in range(num_threads)]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(acks) == num_threads
        worker_pubkeys = [a.worker_ephemeral_pubkey for a in acks]
        # Each servicer request generates a fresh worker ephemeral pubkey
        assert len(set(worker_pubkeys)) == num_threads

    def test_concurrent_measurement_reads_during_generation(self) -> None:
        """Verifies get_measurements returns thread-safe isolated copies during active generation."""
        provider = IntelTdxProvider(simulate=True)
        stop_event = threading.Event()

        def reader():
            while not stop_event.is_set():
                m = provider.get_measurements()
                assert "mrtd" in m
                assert len(m["mrtd"]) == 96
                # Verify mutation of returned dict does not affect internal cache
                m["mrtd"] = "corrupted"

        def writer():
            for _ in range(20):
                nonce = secrets.token_bytes(32)
                pubkey = secrets.token_bytes(32)
                provider.generate_evidence(nonce=nonce, public_key=pubkey)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        writer_threads = [threading.Thread(target=writer) for _ in range(5)]

        for t in threads + writer_threads:
            t.start()

        for wt in writer_threads:
            wt.join()
        stop_event.set()

        for rt in threads:
            rt.join()

        # Cache remains uncorrupted
        final_m = provider.get_measurements()
        assert final_m["mrtd"] != "corrupted"


# ==============================================================================
# 4. MEMORY ZEROIZATION EDGE CASES
# ==============================================================================

class TestAdversarialZeroizationEdgeCases:
    """
    Stress-tests zeroize_buffer across standard, non-standard, read-only,
    and adversarial memory buffers.
    """

    def test_zeroize_none(self) -> None:
        """zeroize_buffer(None) must return cleanly without error."""
        zeroize_buffer(None)

    def test_zeroize_empty_bytearray(self) -> None:
        """zeroize_buffer on empty bytearray must be a safe no-op."""
        buf = bytearray()
        zeroize_buffer(buf)
        assert len(buf) == 0

    @pytest.mark.parametrize("size", [1, 16, 32, 64, 1024, 65536, 131072])
    def test_zeroize_bytearray_various_sizes(self, size) -> None:
        """Verifies bytearrays of all sizes are completely overwritten with zeros."""
        buf = bytearray(secrets.token_bytes(size))
        # Ensure at least some non-zero bytes exist
        buf[0] = 0xAA
        buf[-1] = 0x55
        zeroize_buffer(buf)
        assert buf == bytearray(size)
        assert all(b == 0 for b in buf)

    def test_zeroize_writeable_memoryview(self) -> None:
        """Verifies memoryview over bytearray zeroes the underlying buffer."""
        raw = bytearray(b"highly_sensitive_worker_signing_key_secret_material!")
        mv = memoryview(raw)
        zeroize_buffer(mv)
        assert raw == bytearray(len(raw))

    def test_zeroize_readonly_memoryview_safe_non_fatal(self) -> None:
        """Read-only memoryview (e.g. over bytes) must not raise unhandled exceptions."""
        immutable = b"immutable_bytes_cannot_be_modified"
        mv = memoryview(immutable)
        # In Python, mv[i] = 0 raises TypeError. zeroize_buffer catches and logs.
        zeroize_buffer(mv)

    @pytest.mark.parametrize(
        "immutable_obj",
        [
            b"immutable_bytes_data",
            "string_object",
            123456789,
            3.14159,
            (1, 2, 3),
            {"key": "value"},
            [1, 2, 3],
        ],
    )
    def test_zeroize_immutable_types_safe_non_fatal(self, immutable_obj) -> None:
        """Immutable or unsupported Python objects must not crash zeroize_buffer."""
        zeroize_buffer(immutable_obj)

    def test_zeroize_ctypes_tdx_structures(self) -> None:
        """Verifies ctypes structures (tdx_report_req, tdx_quote_req) are thoroughly wiped."""
        # 1. tdx_report_req
        req = tdx_report_req()
        for i in range(64):
            req.reportdata[i] = 0xDE
        for i in range(1024):
            req.tdreport[i] = 0xAD
        assert req.reportdata[0] == 0xDE
        assert req.tdreport[1023] == 0xAD

        zeroize_buffer(req)
        for i in range(64):
            assert req.reportdata[i] == 0
        for i in range(1024):
            assert req.tdreport[i] == 0

        # 2. tdx_quote_req
        qreq = tdx_quote_req()
        qreq.buf = 0xCAFEBABE
        qreq.len = 99999
        zeroize_buffer(qreq)
        assert qreq.buf == 0
        assert qreq.len == 0

        # 3. Custom ctypes array
        arr = (ctypes.c_uint8 * 256)()
        for i in range(256):
            arr[i] = 0xFF
        zeroize_buffer(arr)
        assert all(arr[i] == 0 for i in range(256))

    def test_zeroize_array_interface_object(self) -> None:
        """Verifies objects implementing __array_interface__ are wiped via ctypes.memset."""
        raw_arr = (ctypes.c_uint8 * 128)()
        ctypes.memset(ctypes.byref(raw_arr), 0x5A, 128)
        assert raw_arr[0] == 0x5A

        class MockArrayInterfaceObj:
            def __init__(self, c_buf):
                self._buf = c_buf
                self.nbytes = ctypes.sizeof(c_buf)
                self.__array_interface__ = {
                    "data": (ctypes.addressof(c_buf), False),
                    "shape": (128,),
                    "typestr": "|u1",
                }

        mock_obj = MockArrayInterfaceObj(raw_arr)
        zeroize_buffer(mock_obj)
        assert all(raw_arr[i] == 0 for i in range(128))

    def test_generate_evidence_zeroizes_temporary_buffers_on_success(
        self, sample_nonce, sample_pubkey
    ) -> None:
        """Verifies internal report_data bytearray is scrubbed to zeros upon method completion."""
        provider = IntelTdxProvider(simulate=True)
        with patch("src.attestation.tdx_attest.zeroize_buffer", wraps=zeroize_buffer) as mock_zeroize:
            evidence = provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

            assert mock_zeroize.called
            scrubbed_buffer = mock_zeroize.call_args[0][0]
            assert isinstance(scrubbed_buffer, bytearray)
            assert len(scrubbed_buffer) == TDX_REPORTDATA_LEN
            assert all(b == 0 for b in scrubbed_buffer)
            assert evidence.user_data[:32] == sample_nonce

    def test_generate_evidence_zeroizes_temporary_buffers_on_exception(
        self, sample_nonce, sample_pubkey
    ) -> None:
        """Verifies temporary buffers are zeroized in the finally block even when an exception occurs."""
        provider = IntelTdxProvider(simulate=True)

        def failing_synthetic(report_data, nonce, pubkey):
            raise RuntimeError("Synthetic failure during quote creation")

        with patch.object(provider, "_generate_synthetic_evidence_data", side_effect=failing_synthetic):
            with patch("src.attestation.tdx_attest.zeroize_buffer", wraps=zeroize_buffer) as mock_zeroize:
                with pytest.raises(AttestationGenerationError, match="Synthetic failure"):
                    provider.generate_evidence(nonce=sample_nonce, public_key=sample_pubkey)

                assert mock_zeroize.called
                scrubbed_buffer = mock_zeroize.call_args[0][0]
                assert isinstance(scrubbed_buffer, bytearray)
                assert len(scrubbed_buffer) == TDX_REPORTDATA_LEN
                assert all(b == 0 for b in scrubbed_buffer)


# ==============================================================================
# 5. AF_VSOCK TRANSPORT & QUOTE PARSER ROBUSTNESS
# ==============================================================================

class TestAdversarialVsockAndQuoteParsing:
    """
    Stress-tests socket communication and binary DCAP Quote parsing against
    fuzzed, corrupted, and truncated network frames.
    """

    def test_recv_exact_premature_eof_raises_generation_error(self) -> None:
        """_recv_exact raising when socket closes with 0 bytes."""
        mock_sock = MagicMock()
        mock_sock.recv.return_value = b""
        with pytest.raises(AttestationGenerationError, match="VSOCK connection closed prematurely"):
            _recv_exact(mock_sock, 4)

    def test_recv_exact_partial_eof_raises_generation_error(self) -> None:
        """_recv_exact raising when socket closes after partial payload."""
        mock_sock = MagicMock()
        mock_sock.recv.side_effect = [b"\x01\x02", b""]  # 2 of 4 bytes then EOF
        with pytest.raises(AttestationGenerationError, match="received 2 of 4 expected bytes"):
            _recv_exact(mock_sock, 4)

    def test_vsock_invalid_td_report_length_rejected(self) -> None:
        """fetch_quote_vsock rejects td_report != 1024 bytes."""
        with patch.object(socket, "AF_VSOCK", 40, create=True):
            with pytest.raises(AttestationValidationError, match="td_report must be exactly 1024 bytes"):
                fetch_quote_vsock(b"\x00" * 1023)

            with pytest.raises(AttestationValidationError, match="td_report must be exactly 1024 bytes"):
                fetch_quote_vsock(b"\x00" * 1025)

    def test_vsock_malformed_response_length_header(self) -> None:
        """QGS daemon sending header claiming length < 4 bytes."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        # Header claiming resp_len = 2 (< 4)
        mock_sock.recv.side_effect = [struct.pack(">I", 2), b"\x00\x00", b""]

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                with pytest.raises(AttestationGenerationError, match="Malformed QGS response length header"):
                    fetch_quote_vsock(b"\x00" * 1024)

    @pytest.mark.parametrize("error_code", [0x00000001, 0x80000001, 0xE0000005, 0xFFFFFFFF])
    def test_vsock_qgs_daemon_error_status_codes(self, error_code) -> None:
        """QGS returning non-zero error status code triggers AttestationGenerationError."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        payload = struct.pack(">I", error_code) + b"\x00" * 636
        frame = struct.pack(">I", len(payload)) + payload

        mock_sock.recv.side_effect = [frame[:4], frame[4:], b""]

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                with pytest.raises(AttestationGenerationError, match=f"0x{error_code:08X}"):
                    fetch_quote_vsock(b"\x00" * 1024)

    def test_vsock_truncated_quote_binary(self) -> None:
        """QGS returning status 0 but truncated Quote (< 636 bytes)."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        payload = struct.pack(">I", 0) + b"\x00" * 300  # < 636 bytes
        frame = struct.pack(">I", len(payload)) + payload

        mock_sock.recv.side_effect = [frame[:4], frame[4:], b""]

        with patch("socket.socket", return_value=mock_sock):
            with patch.object(socket, "AF_VSOCK", 40, create=True):
                with pytest.raises(AttestationGenerationError, match="truncated Quote binary"):
                    fetch_quote_vsock(b"\x00" * 1024)

    @pytest.mark.parametrize("short_len", [0, 10, 48, 54, 100, 500, 635])
    def test_parse_tdx_quote_binary_short_blobs(self, short_len) -> None:
        """parse_tdx_quote_binary rejects any blob shorter than 636 bytes."""
        with pytest.raises(AttestationValidationError, match="Quote binary too short"):
            parse_tdx_quote_binary(b"\x00" * short_len)

    @pytest.mark.parametrize("unsupported_version", [0, 1, 2, 3, 6, 7, 255])
    def test_parse_tdx_quote_binary_unsupported_versions(self, unsupported_version) -> None:
        """parse_tdx_quote_binary rejects unsupported Quote versions."""
        blob = bytearray(b"\x00" * 700)
        struct.pack_into("<HHI", blob, 0, unsupported_version, 2, 0x81)
        with pytest.raises(AttestationValidationError, match="Unsupported Quote version"):
            parse_tdx_quote_binary(bytes(blob))

    def test_parse_tdx_quote_binary_extracts_pem_certificates(self) -> None:
        """parse_tdx_quote_binary extracts embedded PEM certificates cleanly."""
        blob = bytearray(b"\x00" * 800)
        struct.pack_into("<HHI", blob, 0, 4, 2, 0x81)  # v4, ECDSA-256, TDX
        # Append simulated PEM chain
        pem1 = "-----BEGIN CERTIFICATE-----\nMIIB_CERT1_DATA\n-----END CERTIFICATE-----"
        pem2 = "-----BEGIN CERTIFICATE-----\nMIIB_CERT2_DATA\n-----END CERTIFICATE-----"
        blob.extend((pem1 + "\n" + pem2).encode("latin1"))

        parsed = parse_tdx_quote_binary(bytes(blob))
        assert len(parsed["certificate_chain"]) == 2
        assert "MIIB_CERT1_DATA" in parsed["certificate_chain"][0]
        assert "MIIB_CERT2_DATA" in parsed["certificate_chain"][1]
