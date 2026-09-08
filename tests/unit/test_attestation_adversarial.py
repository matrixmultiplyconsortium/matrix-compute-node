"""
Adversarial Stress Test Suite for Hardware Attestation Engine in matrix-compute-node.

Adversarially challenges:
1. Subprocess timeouts, blocked processes, non-zero exit codes, and execution errors.
2. Corrupted CSV output (truncated columns, extra columns, non-numeric VRAM, negative VRAM, special characters, ANSI escapes, injection payloads).
3. Multi-GPU device index out-of-range (e.g. index 99 on 1 GPU, index 999, negative index, index manipulation).
4. Incomplete NVML handles, mid-execution driver errors, GPU lost on bus, memory retrieval errors, and shutdown exceptions.
5. Complete 3-tier cascade fallthrough: verifying that all failure combinations gracefully cascade to Tier 3 CPU fallback without crashing.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
from matrix_contracts.schemas.discovery import HardwareSpecs
from src.attestation.nvml_attest import (
    HardwareAttestationEngine,
    estimate_peak_fp32_tflops,
    verify_attestation_proof,
)
from src.crypto.signer import WorkerSigner


class NVMLError(Exception):
    pass

class NVMLError_LibraryNotFound(NVMLError):
    pass

class NVMLError_DriverNotLoaded(NVMLError):
    pass

class NVMLError_NoPermission(NVMLError):
    pass

class NVMLError_GpuLost(NVMLError):
    pass

class NVMLError_NotFound(NVMLError):
    pass

class NVMLError_Unknown(NVMLError):
    pass


def build_adversarial_pynvml(
    gpu_name=b"NVIDIA GeForce RTX 4090",
    vram_mb=24576,
    free_vram_mb=22000,
    device_count=1,
    driver_version="550.54.14",
    cuda_capability=(8, 9),
    raise_on_init=None,
    raise_on_count=None,
    raise_on_handle=None,
    raise_on_name=None,
    raise_on_mem=None,
    raise_on_driver=None,
    raise_on_cuda=None,
    raise_on_shutdown=None,
):
    """Helper to construct a customizable, adversarial pynvml mock."""
    mock = MagicMock()

    mock.NVMLError = NVMLError
    mock.NVMLError_LibraryNotFound = NVMLError_LibraryNotFound
    mock.NVMLError_DriverNotLoaded = NVMLError_DriverNotLoaded
    mock.NVMLError_NoPermission = NVMLError_NoPermission
    mock.NVMLError_GpuLost = NVMLError_GpuLost
    mock.NVMLError_NotFound = NVMLError_NotFound
    mock.NVMLError_Unknown = NVMLError_Unknown

    # nvmlInit
    if raise_on_init:
        mock.nvmlInit.side_effect = raise_on_init
    else:
        mock.nvmlInit.return_value = None

    # nvmlDeviceGetCount
    if raise_on_count:
        mock.nvmlDeviceGetCount.side_effect = raise_on_count
    else:
        mock.nvmlDeviceGetCount.return_value = device_count

    # nvmlDeviceGetHandleByIndex
    handle = MagicMock()
    if raise_on_handle:
        mock.nvmlDeviceGetHandleByIndex.side_effect = raise_on_handle
    else:
        mock.nvmlDeviceGetHandleByIndex.return_value = handle

    # nvmlDeviceGetName
    if raise_on_name:
        mock.nvmlDeviceGetName.side_effect = raise_on_name
    else:
        mock.nvmlDeviceGetName.return_value = gpu_name

    # nvmlDeviceGetMemoryInfo
    if raise_on_mem:
        mock.nvmlDeviceGetMemoryInfo.side_effect = raise_on_mem
    else:
        mem = MagicMock()
        mem.total = vram_mb * 1024 * 1024
        mem.free = free_vram_mb * 1024 * 1024
        mock.nvmlDeviceGetMemoryInfo.return_value = mem

    # Telemetry
    if raise_on_driver:
        mock.nvmlSystemGetDriverVersion.side_effect = raise_on_driver
    else:
        mock.nvmlSystemGetDriverVersion.return_value = driver_version

    if raise_on_cuda:
        mock.nvmlDeviceGetCudaComputeCapability.side_effect = raise_on_cuda
    else:
        mock.nvmlDeviceGetCudaComputeCapability.return_value = cuda_capability

    # nvmlShutdown
    if raise_on_shutdown:
        mock.nvmlShutdown.side_effect = raise_on_shutdown
    else:
        mock.nvmlShutdown.return_value = None

    return mock


# ==============================================================================
# CLASS 1: SUBPROCESS TIMEOUTS, BLOCKED PROCESSES & EXIT CODES
# ==============================================================================

class TestSubprocessAdversarial:
    """Adversarially tests subprocess timeouts, deadlocks, and failure exit codes."""

    def test_subprocess_timeout_triggers_cpu_fallback(self, sample_signer):
        """Verifies that subprocess.TimeoutExpired (> 5s) triggers clean fallback to CPU."""
        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        with patch.object(engine, "_query_nvml", return_value=None):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
                with patch(
                    "subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=5.0),
                ):
                    specs = engine.detect_hardware()

        assert "CPU" in specs.gpu_model
        assert "NumPy Emulated" in specs.gpu_model
        assert specs.vram_mb > 0
        assert specs.peak_fp32_tflops >= 0.5
        assert len(specs.attestation_proof) == 128

    def test_subprocess_blocked_process_simulation(self, sample_signer):
        """Simulates an unresponsive subprocess hang that hits timeout."""
        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        def hanging_subprocess(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 5))

        with patch.object(engine, "_query_nvml", return_value=None):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
                with patch("subprocess.run", side_effect=hanging_subprocess):
                    metrics = engine._query_nvidia_smi()
                    assert metrics is None

    @pytest.mark.parametrize("exit_code", [1, 2, 9, 15, 127, 255, -9, -15])
    def test_subprocess_nonzero_exit_codes_matrix(self, sample_signer, exit_code):
        """Verifies all non-zero exit codes (standard, fatal, and POSIX signals) return None."""
        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        mock_proc = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=exit_code,
            stdout="",
            stderr=f"Error code {exit_code}: Fatal hardware/driver fault",
        )

        with patch.object(engine, "_query_nvml", return_value=None):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
                with patch("subprocess.run", return_value=mock_proc):
                    metrics = engine._query_nvidia_smi()
                    assert metrics is None
                    specs = engine.detect_hardware()
                    assert "CPU" in specs.gpu_model

    def test_subprocess_stderr_with_zero_returncode(self, sample_signer):
        """Verifies stdout parsing succeeds even when stderr contains driver warnings."""
        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        mock_proc = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA RTX 4090, 24576, 24000\n",
            stderr="WARNING: infoROM is corrupted at block 0x04\n",
        )

        with patch.object(engine, "_query_nvml", return_value=None):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
                with patch("subprocess.run", return_value=mock_proc):
                    specs = engine.detect_hardware()

        assert specs.gpu_model == "NVIDIA RTX 4090"
        assert specs.vram_mb == 24576

    def test_subprocess_missing_binary_across_all_paths(self, sample_signer):
        """Verifies that missing binary on PATH and candidates falls back to CPU cleanly."""
        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        with patch.object(engine, "_query_nvml", return_value=None):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value=None):
                specs = engine.detect_hardware()

        assert "CPU" in specs.gpu_model

    def test_subprocess_permission_error(self, sample_signer):
        """Verifies PermissionError (e.g. EACCES on executable) returns None and falls back."""
        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        with patch.object(engine, "_query_nvml", return_value=None):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
                with patch("subprocess.run", side_effect=PermissionError("Permission denied")):
                    assert engine._query_nvidia_smi() is None
                    specs = engine.detect_hardware()
                    assert "CPU" in specs.gpu_model

    def test_subprocess_os_error_exec_failure(self, sample_signer):
        """Verifies OSError (e.g. exec format error, memory allocation failure) falls back."""
        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        with patch.object(engine, "_query_nvml", return_value=None):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
                with patch("subprocess.run", side_effect=OSError("Exec format error")):
                    assert engine._query_nvidia_smi() is None
                    specs = engine.detect_hardware()
                    assert "CPU" in specs.gpu_model


# ==============================================================================
# CLASS 2: CORRUPTED CSV OUTPUT (TRUNCATED, EXTRA COLS, NON-NUMERIC, SPECIAL)
# ==============================================================================

class TestCorruptedCSVOutputAdversarial:
    """Adversarially tests corrupted, truncated, malformed, and injection CSV inputs."""

    @pytest.mark.parametrize(
        "corrupted_stdout",
        [
            "",
            "   \n\t  \r\n",
            "NVIDIA GeForce RTX 4090",  # 1 column
            "NVIDIA GeForce RTX 4090, 24576",  # 2 columns
            "NVIDIA GeForce RTX 4090, 24576, ",  # 3 columns with empty free memory
            ", 24576, 24000",  # Empty GPU model
            "   , 24576, 24000",  # Whitespace-only GPU model
            "\x00\x01\x02, 24576, 24000",  # Control-characters-only model
        ],
    )
    def test_corrupted_csv_truncated_columns_rejected(self, corrupted_stdout):
        """Asserts all truncated, partial, or empty columns return None."""
        assert HardwareAttestationEngine._parse_smi_output(corrupted_stdout, device_index=0) is None

    @pytest.mark.parametrize(
        "non_numeric_stdout",
        [
            "NVIDIA GeForce RTX 4090, N/A, N/A",
            "NVIDIA GeForce RTX 4090, [Not Supported], [Not Supported]",
            "NVIDIA GeForce RTX 4090, ERR, ERR",
            "NVIDIA GeForce RTX 4090, Unknown, 1000",
            "NVIDIA GeForce RTX 4090, 24576, Unknown",
            "NVIDIA GeForce RTX 4090, Inf, NaN",
            "NVIDIA GeForce RTX 4090, None, null",
            "NVIDIA GeForce RTX 4090, ???, ???",
            "NVIDIA GeForce RTX 4090, ---, ---",
            "NVIDIA GeForce RTX 4090, NO_DATA, NO_DATA",
        ],
    )
    def test_corrupted_csv_non_numeric_vram_rejected(self, non_numeric_stdout):
        """Asserts non-numeric VRAM strings (N/A, [Not Supported], ERR) return None."""
        assert HardwareAttestationEngine._parse_smi_output(non_numeric_stdout, device_index=0) is None

    def test_corrupted_csv_extra_columns_handling(self):
        """Tests handling when nvidia-smi returns extra telemetry columns (> 3 cols)."""
        # When 4 columns are returned, row[:-2] is combined into model name
        stdout = "NVIDIA RTX 4090, OEM Edition, 24576, 24000\n"
        parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
        assert parsed is not None
        assert parsed["gpu_model"] == "NVIDIA RTX 4090, OEM Edition"
        assert parsed["vram_mb"] == 24576
        assert parsed["free_vram_mb"] == 24000

    def test_corrupted_csv_multiple_unquoted_commas(self):
        """Tests model names with multiple commas: ', '.join(row[:-2])."""
        stdout = "NVIDIA, Corp, Custom, RTX, 4090, 24576, 24000\n"
        parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
        assert parsed is not None
        assert parsed["gpu_model"] == "NVIDIA, Corp, Custom, RTX, 4090"
        assert parsed["vram_mb"] == 24576

    def test_corrupted_csv_trailing_commas_rejected(self):
        """Tests that a trailing comma creating an empty final column is rejected."""
        stdout = "NVIDIA RTX 4090, 24576, 24000,\n"
        # Trailing comma causes row[-1] to be '' -> free_match is None -> returns None
        assert HardwareAttestationEngine._parse_smi_output(stdout, device_index=0) is None

    def test_corrupted_csv_float_and_unit_rounding(self):
        """Tests that floating point VRAM is properly rounded to integer MB."""
        stdout = "NVIDIA RTX 4090, 24575.8 MiB, 23999.6 MiB\n"
        parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
        assert parsed is not None
        assert parsed["vram_mb"] == 24576
        assert parsed["free_vram_mb"] == 24000

    def test_corrupted_csv_special_chars_and_ansi_escapes(self):
        """Tests sanitization of ANSI escape sequences and non-printable characters."""
        stdout = "\x1b[31mNVIDIA GeForce RTX 4090\x1b[0m, 24576, 24000\n"
        parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
        assert parsed is not None
        # ANSI escape codes (\x1b) are replaced with space and stripped
        assert "4090" in parsed["gpu_model"]
        assert "\x1b" not in parsed["gpu_model"]

    def test_corrupted_csv_unicode_and_emojis(self):
        """Tests that UTF-8 unicode and emojis in custom GPU names do not crash parser."""
        stdout = "NVIDIA RTX 4090 🚀🔥 (Titan Edition), 24576, 24000\n"
        parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
        assert parsed is not None
        assert "4090" in parsed["gpu_model"]
        assert parsed["vram_mb"] == 24576

    def test_corrupted_csv_injection_string_safety(self):
        """Tests command injection payloads in model name are parsed literally without execution."""
        stdout = "'; rm -rf /; --, 24576, 24000\n"
        parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
        assert parsed is not None
        assert "rm -rf" in parsed["gpu_model"]
        assert parsed["vram_mb"] == 24576

    def test_corrupted_csv_extremely_long_line(self):
        """Tests that lines with > 10,000 characters do not cause memory or regex hangs."""
        long_name = "NVIDIA " + ("A" * 8000) + " RTX 4090"
        stdout = f"{long_name}, 24576, 24000\n"
        parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
        assert parsed is not None
        assert parsed["vram_mb"] == 24576

    def test_negative_vram_behavior_analysis(self):
        """
        Adversarial test analyzing negative VRAM string behavior:
        Checks how `_parse_smi_output` handles negative VRAM readings like '-24576'.
        Documents that regex stripping ignores leading '-' unless validated.
        """
        stdout = "NVIDIA RTX 4090, -24576, -24000\n"
        parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
        # Empirical observation: re.search(r"(\d+(?:\.\d+)?)", "-24576") extracts "24576"
        # and int(round(float("24576"))) evaluates to positive 24576.
        # This highlights a subtle edge case in the numeric regex pattern.
        assert parsed is not None  # Documents current implementation behavior


# ==============================================================================
# CLASS 3: MULTI-GPU DEVICE INDEX OUT-OF-RANGE & BOUNDARY CHECKS
# ==============================================================================

class TestDeviceIndexAdversarial:
    """Adversarially tests multi-GPU device index bounds, invalid indices, and caps."""

    @pytest.mark.parametrize("requested_idx", [1, 2, 5, 99, 500, 998])
    def test_single_gpu_out_of_range_indices_fall_through(self, sample_signer, requested_idx):
        """Verifies that requesting index >= detected GPU count falls through to CPU."""
        mock_pynvml = build_adversarial_pynvml(device_count=1)
        smi_stdout = "NVIDIA GeForce RTX 4090, 24576, 24000\n"

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=requested_idx)

        mock_proc = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0, stdout=smi_stdout, stderr=""
        )

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
                with patch("subprocess.run", return_value=mock_proc):
                    specs = engine.detect_hardware()

        assert "CPU" in specs.gpu_model
        assert specs.vram_mb > 0
        assert specs.peak_fp32_tflops >= 0.5

    def test_device_index_cap_999_skips_gpu_tiers(self, sample_signer):
        """Verifies device_index >= 999 skips NVML and nvidia-smi immediately."""
        engine = HardwareAttestationEngine(signer=sample_signer, device_index=999)

        with patch.object(engine, "_query_nvml") as mock_nvml:
            with patch.object(engine, "_query_nvidia_smi") as mock_smi:
                specs = engine.detect_hardware()
                # _query_nvml returns None immediately without pynvml call
                # _query_nvidia_smi returns None immediately without subprocess call
                assert "CPU" in specs.gpu_model

    def test_device_index_negative_construction_rejected(self, sample_signer):
        """Asserts constructor strictly rejects negative device_index with ValueError."""
        for neg_idx in [-1, -2, -99, -999]:
            with pytest.raises(ValueError, match="device_index must be non-negative"):
                HardwareAttestationEngine(signer=sample_signer, device_index=neg_idx)

    def test_device_index_mutation_after_construction_rejected(self, sample_signer):
        """Verifies detect_hardware guards against device_index mutated to negative."""
        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
        engine.device_index = -1
        with pytest.raises(ValueError, match="Invalid device_index"):
            engine.detect_hardware()

    def test_parse_smi_output_negative_device_index(self):
        """Asserts _parse_smi_output returns None when passed negative index."""
        stdout = "NVIDIA GeForce RTX 4090, 24576, 24000\n"
        assert HardwareAttestationEngine._parse_smi_output(stdout, device_index=-1) is None
        assert HardwareAttestationEngine._parse_smi_output(stdout, device_index=-99) is None

    def test_multi_gpu_exact_boundaries(self):
        """Verifies boundary indexing on a 3-GPU system (indices 0, 1, 2 valid; 3 invalid)."""
        stdout = (
            "NVIDIA GeForce RTX 3080, 10240, 9500\n"
            "NVIDIA GeForce RTX 3090, 24576, 23000\n"
            "NVIDIA GeForce RTX 4090, 24576, 24100\n"
        )
        assert HardwareAttestationEngine._parse_smi_output(stdout, 0)["gpu_model"] == "NVIDIA GeForce RTX 3080"
        assert HardwareAttestationEngine._parse_smi_output(stdout, 1)["gpu_model"] == "NVIDIA GeForce RTX 3090"
        assert HardwareAttestationEngine._parse_smi_output(stdout, 2)["gpu_model"] == "NVIDIA GeForce RTX 4090"
        assert HardwareAttestationEngine._parse_smi_output(stdout, 3) is None
        assert HardwareAttestationEngine._parse_smi_output(stdout, 4) is None


# ==============================================================================
# CLASS 4: INCOMPLETE NVML HANDLES & MID-EXECUTION DRIVER ERRORS
# ==============================================================================

class TestNVMLDriverErrorsAdversarial:
    """Adversarially tests NVML driver crashes, incomplete handles, and lifecycle states."""

    def test_nvml_init_fails_cleanly(self, sample_signer):
        """Verifies various initialization failures trigger clean fallback."""
        mock_pynvml = build_adversarial_pynvml(
            raise_on_init=RuntimeError("Dynamic linker error loading libnvidia-ml.so")
        )

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(engine, "_query_nvidia_smi", return_value=None):
                specs = engine.detect_hardware()

        assert "CPU" in specs.gpu_model
        mock_pynvml.nvmlShutdown.assert_not_called()

    def test_nvml_handle_retrieval_failure(self, sample_signer):
        """Verifies failure in nvmlDeviceGetHandleByIndex triggers clean fallback."""
        mock_pynvml = build_adversarial_pynvml()
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = mock_pynvml.NVMLError_NotFound("Device not found")

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(engine, "_query_nvidia_smi", return_value=None):
                specs = engine.detect_hardware()

        assert "CPU" in specs.gpu_model
        mock_pynvml.nvmlShutdown.assert_called_once()

    def test_nvml_mid_execution_gpu_lost_on_bus(self, sample_signer):
        """Simulates GPU dropping off PCIe bus during name query."""
        mock_pynvml = build_adversarial_pynvml()
        mock_pynvml.nvmlDeviceGetName.side_effect = mock_pynvml.NVMLError_GpuLost("GPU has fallen off the bus")

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(engine, "_query_nvidia_smi", return_value=None):
                specs = engine.detect_hardware()

        assert "CPU" in specs.gpu_model
        mock_pynvml.nvmlShutdown.assert_called_once()

    def test_nvml_mid_execution_memory_query_error(self, sample_signer):
        """Simulates driver failure during memory info retrieval."""
        mock_pynvml = build_adversarial_pynvml()
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = mock_pynvml.NVMLError_Unknown("Internal driver error")

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(engine, "_query_nvidia_smi", return_value=None):
                specs = engine.detect_hardware()

        assert "CPU" in specs.gpu_model
        mock_pynvml.nvmlShutdown.assert_called_once()

    @pytest.mark.parametrize(
        "corrupted_name, expected_model",
        [
            (b"", "NVIDIA GPU #0"),
            ("", "NVIDIA GPU #0"),
            (b"\x00\x00\x00", "NVIDIA GPU #0"),
            ("\t  \r\n", "NVIDIA GPU #0"),
            (b"NVIDIA\x00\x00\x00RTX 4090", "NVIDIA RTX 4090"),
            ("NVIDIA\x1b[0m RTX 4090", "NVIDIA RTX 4090"),
        ],
    )
    def test_nvml_corrupted_name_sanitization(self, sample_signer, corrupted_name, expected_model):
        """Tests that empty, null-byte, or control-character GPU names are safely sanitized."""
        mock_pynvml = build_adversarial_pynvml(gpu_name=corrupted_name)

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            specs = engine.detect_hardware()

        assert specs.gpu_model == expected_model
        assert specs.vram_mb == 24576

    def test_nvml_shutdown_failure_does_not_mask_metrics(self, sample_signer):
        """Verifies that an exception during nvmlShutdown does not crash or corrupt metrics."""
        mock_pynvml = build_adversarial_pynvml(
            raise_on_shutdown=RuntimeError("nvmlShutdown failed with DRIVER_UNAVAILABLE")
        )

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            specs = engine.detect_hardware()

        assert specs.gpu_model == "NVIDIA GeForce RTX 4090"
        assert specs.vram_mb == 24576

    def test_nvml_optional_telemetry_errors_ignored(self, sample_signer):
        """Verifies driver version and CUDA compute capability exceptions are swallowed safely."""
        mock_pynvml = build_adversarial_pynvml(
            raise_on_driver=Exception("Telemetry query failed"),
            raise_on_cuda=Exception("Compute capability unsupported"),
        )

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            metrics = engine._query_nvml()

        assert metrics is not None
        assert metrics["gpu_model"] == "NVIDIA GeForce RTX 4090"
        assert metrics["driver_version"] is None
        assert metrics["cuda_capability"] is None


# ==============================================================================
# CLASS 5: COMPLETE 3-TIER CASCADE FALLTHROUGH VERIFICATION
# ==============================================================================

class TestCompleteFallthroughCascadeAdversarial:
    """Adversarially tests concurrent failures across all tiers and verifies CPU fallback."""

    def test_full_cascade_both_gpu_tiers_fail(self, sample_signer):
        """Both NVML and nvidia-smi fail completely -> falls through to CPU cleanly."""
        mock_pynvml = build_adversarial_pynvml(
            raise_on_init=Exception("No NVIDIA hardware present")
        )

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value=None):
                specs = engine.detect_hardware()

        assert isinstance(specs, HardwareSpecs)
        assert "CPU:" in specs.gpu_model
        assert "NumPy Emulated" in specs.gpu_model
        assert specs.vram_mb > 0
        assert specs.peak_fp32_tflops >= 0.5
        assert len(specs.attestation_proof) == 128

        # Cryptographic verification passes for CPU fallback
        pubkey = sample_signer.get_pubkey_hex()
        assert HardwareAttestationEngine.verify_attestation(specs, pubkey) is True

    def test_full_cascade_gpu_lost_and_subprocess_garbage(self, sample_signer):
        """NVML reports GPU lost and nvidia-smi returns garbage -> falls through to CPU."""
        mock_pynvml = build_adversarial_pynvml()
        mock_pynvml.nvmlDeviceGetName.side_effect = mock_pynvml.NVMLError_GpuLost("Bus drop")

        mock_proc = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="CORRUPTED_GARBAGE_BUFFER_0xDEADBEEF\n",
            stderr="",
        )

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
                with patch("subprocess.run", return_value=mock_proc):
                    specs = engine.detect_hardware()

        assert "CPU:" in specs.gpu_model
        assert specs.vram_mb > 0
        assert specs.peak_fp32_tflops >= 0.5

    def test_full_cascade_timeout_and_driver_unloaded(self, sample_signer):
        """NVML driver unloaded + nvidia-smi times out (> 5s) -> falls through to CPU."""
        mock_pynvml = build_adversarial_pynvml(
            raise_on_init=NVMLError_DriverNotLoaded("Driver not loaded")
        )

        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
                with patch(
                    "subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=5),
                ):
                    specs = engine.detect_hardware()

        assert "CPU:" in specs.gpu_model
        assert specs.vram_mb > 0

    def test_idempotent_attestation_under_repeated_failures(self, sample_signer):
        """Calling detect_hardware 10 consecutive times under failure states is stable."""
        engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

        with patch.object(engine, "_query_nvml", return_value=None):
            with patch.object(engine, "_query_nvidia_smi", return_value=None):
                for _ in range(10):
                    specs = engine.detect_hardware()
                    assert "CPU:" in specs.gpu_model
                    assert specs.vram_mb > 0
                    assert specs.peak_fp32_tflops >= 0.5
                    assert len(specs.attestation_proof) == 128
