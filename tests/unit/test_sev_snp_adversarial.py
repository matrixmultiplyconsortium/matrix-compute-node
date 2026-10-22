"""
Adversarial Coverage Hardening and Penetration Edge Cases (Tier 5)
for matrix-compute-node AMD SEV-SNP Attestation Provider.

Empirically challenges:
1. Linux Kernel <uapi/linux/sev-guest.h> ABI compliance (structure sizes, offsets, ioctl numbers).
2. Silent fallback to synthetic evidence when hardware device permissions are restricted.
3. Silent fallback to synthetic PKI certificate chain when certificate table is empty or binary DER.
4. VMPL range bounds and wrapping (negative, > 3, float, non-integer).
5. get_attestation_provider('auto', fallback_to_simulated=False) fail-closed behavior.
6. Reference Golden RIM state pollution / cache mutation after generating evidence.
7. Device path probing priority and nonexistent custom device paths.
8. Memory zeroization across all buffer types, read-only buffers, and exception paths.
9. Extreme nonce and public key inputs (zero length, oversized, non-bytes types).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import secrets
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure matrix-compute-sdk is importable
sdk_src = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../matrix-compute-sdk/src")
)
if os.path.exists(sdk_src) and sdk_src not in sys.path:
    sys.path.insert(0, sdk_src)

from src.attestation import (
    AmdSevSnpProvider,
    AttestationError,
    AttestationGenerationError,
    AttestationValidationError,
    HardwareAttestationUnavailableError,
    get_attestation_provider,
    parse_snp_report_binary,
    snp_ext_report_req,
    snp_guest_request,
    snp_report_req,
    zeroize_buffer,
)
from src.attestation.base import _PROVIDER_REGISTRY
from src.attestation.nvml_attest import NvidiaCcaProvider
from src.attestation.sev_snp_attest import (
    IOC_TYPE_SNP,
    SNP_GET_EXT_REPORT,
    SNP_GET_REPORT,
    SNP_NONCE_LEN,
    SNP_PUBKEY_HASH_LEN,
    SNP_REPORT_LEN,
    _IOC,
    _IOC_READWRITE,
)


# ==============================================================================
# SECTION 1: LINUX KERNEL <uapi/linux/sev-guest.h> ABI COMPLIANCE
# ==============================================================================

class TestLinuxKernelAbiCompliance:
    """
    Verifies ctypes structures against the official Linux Kernel UAPI
    header (<uapi/linux/sev-guest.h>) and AMD SEV-SNP driver ABI.
    """

    def test_snp_guest_request_ioctl_structure_and_offsets(self):
        """
        Linux UAPI struct snp_guest_request_ioctl is:
            struct snp_guest_request_ioctl {
                __u8 msg_version;
                __u64 req_data;
                __u64 resp_data;
                __u64 fw_err;
            };
        Size must be 32 bytes (with alignment padding).
        msg_version is at offset 0, req_data at offset 8.
        """
        class LinuxSnpGuestRequestIoctl(ctypes.Structure):
            _fields_ = [
                ("msg_version", ctypes.c_uint8),
                ("req_data", ctypes.c_uint64),
                ("resp_data", ctypes.c_uint64),
                ("fw_err", ctypes.c_uint64),
            ]

        linux_size = ctypes.sizeof(LinuxSnpGuestRequestIoctl)
        current_size = ctypes.sizeof(snp_guest_request)
        
        # Check whether current implementation matches the Linux kernel UAPI
        assert current_size == linux_size, (
            f"snp_guest_request ABI mismatch: current size is {current_size} bytes, "
            f"Linux UAPI struct snp_guest_request_ioctl is {linux_size} bytes (missing msg_version)"
        )

    def test_snp_report_req_size_and_fields(self):
        """
        Linux UAPI struct snp_report_req is:
            struct snp_report_req {
                __u8 user_data[64];
                __u32 vmpl;
                __u8 rsvd[28];
            };
        Total size is 96 bytes.
        """
        class LinuxSnpReportReq(ctypes.Structure):
            _fields_ = [
                ("user_data", ctypes.c_uint8 * 64),
                ("vmpl", ctypes.c_uint32),
                ("rsvd", ctypes.c_uint8 * 28),
            ]

        linux_size = ctypes.sizeof(LinuxSnpReportReq)
        current_size = ctypes.sizeof(snp_report_req)

        assert current_size == linux_size, (
            f"snp_report_req ABI mismatch: current size is {current_size} bytes, "
            f"Linux UAPI struct snp_report_req is {linux_size} bytes (reserved is 20B instead of 28B)"
        )

    def test_snp_ext_report_req_certs_address_offset(self):
        """
        In Linux UAPI struct snp_ext_report_req:
            struct snp_ext_report_req {
                struct snp_report_req data; // 96 bytes
                __u64 certs_address;        // offset 96
                __u32 certs_len;            // offset 104
            };
        certs_address MUST be at offset 96, and total size is 112 bytes.
        """
        class LinuxSnpReportReq(ctypes.Structure):
            _fields_ = [
                ("user_data", ctypes.c_uint8 * 64),
                ("vmpl", ctypes.c_uint32),
                ("rsvd", ctypes.c_uint8 * 28),
            ]

        class LinuxSnpExtReportReq(ctypes.Structure):
            _fields_ = [
                ("data", LinuxSnpReportReq),
                ("certs_address", ctypes.c_uint64),
                ("certs_len", ctypes.c_uint32),
            ]

        linux_offset = LinuxSnpExtReportReq.certs_address.offset
        current_offset = snp_ext_report_req.certs_address.offset

        assert current_offset == linux_offset, (
            f"snp_ext_report_req ABI mismatch: certs_address is at offset {current_offset}, "
            f"Linux kernel expects offset {linux_offset}"
        )

    def test_snp_get_ext_report_ioctl_command_number(self):
        """
        Linux UAPI defines:
            #define SNP_GET_REPORT _IOWR('S', 0x0, struct snp_guest_request_ioctl)
            #define SNP_GET_DERIVED_KEY _IOWR('S', 0x1, struct snp_guest_request_ioctl)
            #define SNP_GET_EXT_REPORT _IOWR('S', 0x2, struct snp_guest_request_ioctl)
        SNP_GET_EXT_REPORT must have command number (nr) 0x2, not 0x1!
        """
        # Command number is extracted from bits 0..7 of ioctl value
        nr = SNP_GET_EXT_REPORT & 0xFF
        assert nr == 0x2, (
            f"SNP_GET_EXT_REPORT ioctl command number nr is {hex(nr)}; "
            f"Linux kernel defines nr=0x2 (nr=0x1 is SNP_GET_DERIVED_KEY)"
        )


# ==============================================================================
# SECTION 2: FAIL-CLOSED PERMISSION & SILENT FALLBACK BEHAVIOR
# ==============================================================================

class TestFailClosedPermissionAndFallback:
    """
    Tests behavior when hardware devices exist but access is restricted.
    """

    def test_simulate_false_restricted_device_does_not_silently_generate_synthetic(self):
        """
        If simulate=False and /dev/sev-guest exists on disk but is not readable/writable
        (e.g. running unprivileged without sev group), generate_evidence() MUST NOT
        silently return synthetic test fixtures. It must raise HardwareAttestationUnavailableError
        or AttestationGenerationError.
        """
        with patch("src.attestation.sev_snp_attest._probe_sev_guest_dev", return_value=None):
            with patch("src.attestation.sev_snp_attest._probe_cpuinfo_sev_snp", return_value=True):
                with patch("src.attestation.sev_snp_attest.os.path.exists", return_value=True):
                    provider = AmdSevSnpProvider(simulate=False)
                    # If is_available() is True, generate_evidence must not produce synthetic data!
                    if provider.is_available():
                        with pytest.raises(
                            (HardwareAttestationUnavailableError, AttestationGenerationError),
                            match="hardware.*unavailable|permission|ioctl",
                        ):
                            provider.generate_evidence(
                                nonce=secrets.token_bytes(32),
                                public_key=secrets.token_bytes(32),
                            )

    def test_auto_provider_respects_fallback_to_simulated_false(self):
        """
        get_attestation_provider('auto', fallback_to_simulated=False) MUST raise
        HardwareAttestationUnavailableError when no hardware TEE is available.
        """
        with patch.object(NvidiaCcaProvider, "is_available", return_value=False):
            with patch.object(AmdSevSnpProvider, "is_available", return_value=False):
                with pytest.raises(
                    HardwareAttestationUnavailableError,
                    match="hardware.*unavailable",
                ):
                    get_attestation_provider("auto", fallback_to_simulated=False)


# ==============================================================================
# SECTION 3: VMPL PRIVILEGE LEVEL BOUNDS
# ==============================================================================

class TestVmplPrivilegeLevelBounds:
    """
    AMD SEV-SNP defines VMPL levels 0 to 3.
    Values < 0 or > 3 are invalid and must be rejected.
    """

    @pytest.mark.parametrize("invalid_vmpl", [-1, -99, 4, 100, 4294967295])
    def test_invalid_vmpl_rejected_at_construction(self, invalid_vmpl):
        """Constructor should reject invalid VMPL privilege levels."""
        with pytest.raises((ValueError, AttestationValidationError)):
            AmdSevSnpProvider(simulate=True, vmpl=invalid_vmpl)


# ==============================================================================
# SECTION 4: STATE POLLUTION & GOLDEN RIM MUTATION
# ==============================================================================

class TestGoldenRimStateIntegrity:
    """
    Verifies that provider reference Golden RIM is immutable and not corrupted
    by generated reports.
    """

    def test_configured_golden_rim_not_overwritten_by_untrusted_report(self):
        """
        If a provider is configured with a reference Golden RIM, calling
        get_measurements() after generate_evidence() must still return the
        configured reference Golden RIM, NOT whatever raw measurement was
        inside the parsed report.
        """
        reference_rim = {
            "measurement": "aa" * 48,
            "host_data": "bb" * 32,
        }
        provider = AmdSevSnpProvider(simulate=True, golden_rim=reference_rim)

        # Generate evidence with synthetic generator
        evidence = provider.generate_evidence(
            nonce=secrets.token_bytes(32),
            public_key=secrets.token_bytes(32),
        )

        # Measurements returned by provider must still match the configured reference RIM
        assert provider.get_measurements() == reference_rim

    def test_golden_rim_not_polluted_when_report_measurement_differs(self):
        """
        When a hardware report contains an untrusted/different measurement,
        calling generate_evidence() must NOT overwrite self.get_measurements()
        with the untrusted measurement.
        """
        reference_rim = {
            "measurement": "11" * 48,
            "host_data": "22" * 32,
        }
        provider = AmdSevSnpProvider(simulate=True, golden_rim=reference_rim)

        # Mock an attestation report with a different measurement
        fake_report = bytearray(b"\x00" * 1184)
        fake_report[0x090:0x0C0] = b"\x99" * 48  # Different measurement
        fake_report[0x0C0:0x0E0] = b"\x88" * 32  # Different host_data

        with patch.object(provider, "_generate_synthetic_evidence_data", return_value=(bytes(fake_report), [])):
            provider.generate_evidence(nonce=b"a" * 32, public_key=b"b" * 32)

        # The reference golden RIM returned by get_measurements() must remain unchanged!
        assert provider.get_measurements() == reference_rim, (
            f"State pollution: get_measurements() was overwritten with report measurement "
            f"{provider.get_measurements()['measurement'][:16]}... instead of configured reference RIM"
        )


# ==============================================================================
# SECTION 5: CERTIFICATE TABLE FORMAT & PROBING EDGE CASES
# ==============================================================================

class TestCertTableAndDeviceProbing:
    """
    Tests edge cases in certificate extraction and device path handling.
    """

    def test_binary_der_or_ghcb_cert_table_falls_back_to_synthetic(self):
        """
        GHCB certificate tables or raw DER certificate blobs do not contain ASCII
        '-----BEGIN CERTIFICATE-----'. In such cases, _fetch_snp_report_and_certs()
        silently falls back to returning the synthetic PKI chain rather than
        either properly parsing GHCB GUID cert table entries or raising an error.
        """
        provider = AmdSevSnpProvider(simulate=False, device_path="/dev/sev-guest")
        mock_file = MagicMock()
        mock_file.fileno.return_value = 3

        with patch("src.attestation.sev_snp_attest.fcntl") as mock_fcntl:
            mock_fcntl.ioctl.return_value = 0
            with patch("builtins.open", return_value=mock_file):
                report_blob, cert_chain = provider._fetch_snp_report_and_certs(
                    "/dev/sev-guest", b"\x00" * 64, []
                )
                from matrix_compute.crypto.pki_fixtures import get_default_amd_pki
                # If certs_buf did not have PEM headers, it silently fell back to synthetic PKI!
                assert cert_chain == get_default_amd_pki().certificate_chain

    def test_nonexistent_custom_device_path_with_cpuinfo_flag(self):
        """
        If a user specifies a non-existent device_path='/dev/nonexistent-snp'
        on a host with sev_snp flag in /proc/cpuinfo, is_available() should
        not report True when the specified device does not exist.
        """
        with patch("src.attestation.sev_snp_attest._probe_cpuinfo_sev_snp", return_value=True):
            with patch("src.attestation.sev_snp_attest.os.path.exists", side_effect=lambda p: p in ("/proc/cpuinfo", "/dev/sev-guest")):
                provider = AmdSevSnpProvider(simulate=False, device_path="/dev/nonexistent-snp")
                # When explicit device_path is provided, availability should track that path
                # rather than falling through to default /dev/sev-guest
                is_avail = provider.is_available()
                assert is_avail is False, (
                    "Provider reported available even though explicitly configured device_path does not exist"
                )

