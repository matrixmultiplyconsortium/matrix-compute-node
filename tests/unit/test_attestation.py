"""
Unit tests for HardwareAttestationEngine and peak FP32 TFLOPS calculation.
Verifies NVML interrogation, nvidia-smi CLI fallback, CPU/NumPy fallback,
lookup table matching, HardwareSpecs packaging, and Secp256k1 signed attestation.
"""

from __future__ import annotations

import os
import platform
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from matrix_contracts.schemas.discovery import HardwareSpecs
from src.attestation.nvml_attest import (
    HardwareAttestationEngine,
    estimate_peak_fp32_tflops,
    verify_attestation_proof,
    _get_cpu_model_name,
    _get_system_ram_mb,
    _get_cpu_core_count,
)
from src.crypto.signer import WorkerSigner


def build_mock_pynvml(
    gpu_name=b"NVIDIA GeForce RTX 4090",
    vram_mb=24576,
    free_vram_mb=22000,
    device_count=1,
    driver_version="535.129.03",
    raise_on_init=None,
    raise_on_query=None,
):
    """Helper to construct a fully configured pynvml mock."""
    mock = MagicMock()

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

    mock.NVMLError = NVMLError
    mock.NVMLError_LibraryNotFound = NVMLError_LibraryNotFound
    mock.NVMLError_DriverNotLoaded = NVMLError_DriverNotLoaded
    mock.NVMLError_NoPermission = NVMLError_NoPermission
    mock.NVMLError_GpuLost = NVMLError_GpuLost

    if raise_on_init:
        mock.nvmlInit.side_effect = raise_on_init
    else:
        mock.nvmlInit.return_value = None

    mock.nvmlDeviceGetCount.return_value = device_count
    handle = MagicMock()
    mock.nvmlDeviceGetHandleByIndex.return_value = handle

    if raise_on_query:
        mock.nvmlDeviceGetName.side_effect = raise_on_query
    else:
        mock.nvmlDeviceGetName.return_value = gpu_name

    mem = MagicMock()
    mem.total = vram_mb * 1024 * 1024
    mem.free = free_vram_mb * 1024 * 1024
    mock.nvmlDeviceGetMemoryInfo.return_value = mem
    mock.nvmlSystemGetDriverVersion.return_value = driver_version
    mock.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)
    return mock


# ==============================================================================
# TIER 1: NVML INTERROGATION TESTS
# ==============================================================================

def test_nvml_detection_success(sample_signer):
    """TC-ATT-01: Verifies primary NVML detection path when pynvml succeeds."""
    mock_pynvml = build_mock_pynvml(
        gpu_name=b"NVIDIA GeForce RTX 4090",
        vram_mb=24576,
        free_vram_mb=22000,
        device_count=1,
    )

    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        specs = engine.detect_hardware(timestamp=1788537600)

    assert isinstance(specs, HardwareSpecs)
    assert specs.gpu_model == "NVIDIA GeForce RTX 4090"
    assert specs.vram_mb == 24576
    assert specs.peak_fp32_tflops == 82.6
    assert len(specs.attestation_proof) == 128

    mock_pynvml.nvmlInit.assert_called_once()
    mock_pynvml.nvmlShutdown.assert_called_once()

    # Verify signature
    msg = "NVIDIA GeForce RTX 4090:24576:82.6:1788537600".encode("utf-8")
    assert sample_signer.verify_signature(msg, specs.attestation_proof) is True


def test_nvml_device_name_byte_and_string_handling(sample_signer):
    """TC-ATT-02: Verifies handling of both str and bytes for nvmlDeviceGetName."""
    mock_pynvml = build_mock_pynvml(
        gpu_name="NVIDIA RTX 4080\x00",
        vram_mb=16384,
        free_vram_mb=15000,
        device_count=1,
    )

    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        specs = engine.detect_hardware()

    assert specs.gpu_model == "NVIDIA RTX 4080"
    assert specs.vram_mb == 16384
    assert specs.peak_fp32_tflops == 48.7


def test_nvml_exceptions_cascade_to_nvidia_smi(sample_signer):
    """TC-ATT-03: NVML exceptions cleanly cascade to Tier 2 (nvidia-smi CLI)."""
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.side_effect = RuntimeError("NVML Shared Library Not Found")

    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

    smi_stdout = "NVIDIA A100-SXM4-80GB, 81920, 80000\n"
    mock_proc = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout=smi_stdout, stderr=""
    )

    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_proc):
                specs = engine.detect_hardware()

    assert specs.gpu_model == "NVIDIA A100-SXM4-80GB"
    assert specs.vram_mb == 81920
    assert specs.peak_fp32_tflops == 19.5


def test_nvml_library_not_found_cascade(sample_signer):
    """Verifies NVMLError_LibraryNotFound triggers clean fallback to Tier 2/3."""
    mock_pynvml = build_mock_pynvml()
    mock_pynvml.nvmlInit.side_effect = mock_pynvml.NVMLError_LibraryNotFound("nvml.dll not found")

    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        with patch.object(engine, "_query_nvidia_smi", return_value=None):
            specs = engine.detect_hardware()

    assert "CPU" in specs.gpu_model
    mock_pynvml.nvmlShutdown.assert_not_called()


def test_nvml_driver_not_loaded_cascade(sample_signer):
    """Verifies NVMLError_DriverNotLoaded triggers clean fallback."""
    mock_pynvml = build_mock_pynvml()
    mock_pynvml.nvmlInit.side_effect = mock_pynvml.NVMLError_DriverNotLoaded("driver not loaded")

    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        with patch.object(engine, "_query_nvidia_smi", return_value=None):
            specs = engine.detect_hardware()

    assert "CPU" in specs.gpu_model


def test_nvml_no_permission_cascade(sample_signer):
    """Verifies NVMLError_NoPermission triggers clean fallback."""
    mock_pynvml = build_mock_pynvml()
    mock_pynvml.nvmlInit.side_effect = mock_pynvml.NVMLError_NoPermission("permission denied")

    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        with patch.object(engine, "_query_nvidia_smi", return_value=None):
            specs = engine.detect_hardware()

    assert "CPU" in specs.gpu_model


def test_nvml_gpu_lost_cascade(sample_signer):
    """Verifies NVMLError_GpuLost calls shutdown in finally and falls back."""
    mock_pynvml = build_mock_pynvml()
    mock_pynvml.nvmlDeviceGetName.side_effect = mock_pynvml.NVMLError_GpuLost("GPU fell off bus")

    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        with patch.object(engine, "_query_nvidia_smi", return_value=None):
            specs = engine.detect_hardware()

    assert "CPU" in specs.gpu_model
    mock_pynvml.nvmlShutdown.assert_called_once()


def test_nvml_zero_device_count(sample_signer):
    """Verifies NVML reporting 0 devices returns None and falls back."""
    mock_pynvml = build_mock_pynvml(device_count=0)

    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        with patch.object(engine, "_query_nvidia_smi", return_value=None):
            specs = engine.detect_hardware()

    assert "CPU" in specs.gpu_model


def test_nvml_shutdown_exception_non_fatal(sample_signer):
    """Verifies error during nvmlShutdown does not corrupt returned metrics."""
    mock_pynvml = build_mock_pynvml(
        gpu_name=b"NVIDIA GeForce RTX 4090",
        vram_mb=24576,
    )
    mock_pynvml.nvmlShutdown.side_effect = Exception("Shutdown warning")

    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        specs = engine.detect_hardware()

    assert specs.gpu_model == "NVIDIA GeForce RTX 4090"
    assert specs.vram_mb == 24576


def test_nvml_control_character_sanitization(sample_signer):
    """Verifies null bytes and non-printable control chars are sanitized."""
    mock_pynvml = build_mock_pynvml(
        gpu_name=b"NVIDIA\x00 GeForce\x1b RTX \t 4090\x00",
        vram_mb=24576,
    )

    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        specs = engine.detect_hardware()

    assert specs.gpu_model == "NVIDIA GeForce RTX 4090"


# ==============================================================================
# TIER 2: NVIDIA-SMI SUBPROCESS CLI TESTS
# ==============================================================================

def test_nvidia_smi_cli_success(sample_signer):
    """TC-ATT-04: Verifies Tier 2 nvidia-smi CLI execution and parsing."""
    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

    smi_stdout = "NVIDIA GeForce RTX 3090, 24576, 23500\n"
    mock_proc = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout=smi_stdout, stderr=""
    )

    with patch.object(engine, "_query_nvml", return_value=None):
        with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_proc):
                specs = engine.detect_hardware()

    assert specs.gpu_model == "NVIDIA GeForce RTX 3090"
    assert specs.vram_mb == 24576
    assert specs.peak_fp32_tflops == 35.6


def test_nvidia_smi_multi_gpu_device_index(sample_signer):
    """TC-ATT-05: Verifies selecting correct GPU line based on device_index."""
    engine = HardwareAttestationEngine(signer=sample_signer, device_index=1)

    smi_stdout = "NVIDIA GeForce RTX 3080, 10240, 9500\nNVIDIA GeForce RTX 4090, 24576, 24000\n"
    mock_proc = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout=smi_stdout, stderr=""
    )

    with patch.object(engine, "_query_nvml", return_value=None):
        with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_proc):
                specs = engine.detect_hardware()

    assert specs.gpu_model == "NVIDIA GeForce RTX 4090"
    assert specs.vram_mb == 24576
    assert specs.peak_fp32_tflops == 82.6


def test_nvidia_smi_failures_cascade_to_cpu(sample_signer):
    """TC-ATT-06: nvidia-smi CLI failures (FileNotFoundError, Timeout) cascade to CPU fallback."""
    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

    with patch.object(engine, "_query_nvml", return_value=None):
        with patch.object(engine, "_find_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
                specs = engine.detect_hardware()

    assert "CPU" in specs.gpu_model
    assert "NumPy Emulated" in specs.gpu_model
    assert specs.vram_mb > 0
    assert specs.peak_fp32_tflops >= 0.5


# Binary Discovery Unit Tests
def test_find_binary_via_path(monkeypatch):
    """Resolves nvidia-smi when located directly in system PATH."""
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr("os.path.isfile", lambda path: path == "/usr/bin/nvidia-smi")
    binary = HardwareAttestationEngine._find_nvidia_smi_binary()
    assert binary is not None
    assert "nvidia-smi" in binary


def test_find_binary_windows_fallback_system32(monkeypatch):
    """Resolves Windows System32 nvidia-smi.exe when PATH lookup returns None."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("sys.platform", "win32")
    expected = os.path.abspath(r"C:\Windows\System32\nvidia-smi.exe")
    monkeypatch.setattr("os.path.isfile", lambda path: os.path.abspath(path) == expected)
    binary = HardwareAttestationEngine._find_nvidia_smi_binary()
    assert binary == expected


def test_find_binary_windows_fallback_nvsmi(monkeypatch):
    """Resolves Windows Program Files NVSMI path."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("sys.platform", "win32")
    expected = os.path.abspath(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    monkeypatch.setattr("os.path.isfile", lambda path: os.path.abspath(path) == expected)
    binary = HardwareAttestationEngine._find_nvidia_smi_binary()
    assert binary == expected


def test_find_binary_linux_fallback(monkeypatch):
    """Resolves Linux /usr/local/cuda/bin/nvidia-smi when PATH is missing."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("sys.platform", "linux")
    expected = "/usr/local/cuda/bin/nvidia-smi"
    monkeypatch.setattr("os.path.isfile", lambda path: path == expected)
    monkeypatch.setattr("os.access", lambda path, mode: True)
    binary = HardwareAttestationEngine._find_nvidia_smi_binary()
    assert binary == expected


def test_find_binary_not_found(monkeypatch):
    """Returns None gracefully when binary does not exist anywhere."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("os.path.isfile", lambda path: False)
    assert HardwareAttestationEngine._find_nvidia_smi_binary() is None


# CSV Output Parsing Unit Tests
def test_parse_single_gpu():
    """Correctly parses standard single-GPU CSV output."""
    stdout = "NVIDIA GeForce RTX 4090, 24576, 24100\n"
    parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
    assert parsed is not None
    assert parsed["gpu_model"] == "NVIDIA GeForce RTX 4090"
    assert parsed["vram_mb"] == 24576
    assert parsed["free_vram_mb"] == 24100
    assert parsed["source"] == "nvidia-smi"
    assert parsed["device_index"] == 0


def test_parse_multi_gpu():
    """Correctly extracts the second GPU when device_index=1."""
    stdout = (
        "NVIDIA GeForce RTX 4090, 24576, 24100\r\n"
        "NVIDIA RTX A6000, 49152, 48000\r\n"
    )
    parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=1)
    assert parsed is not None
    assert parsed["gpu_model"] == "NVIDIA RTX A6000"
    assert parsed["vram_mb"] == 49152
    assert parsed["free_vram_mb"] == 48000
    assert parsed["device_index"] == 1


def test_parse_multi_gpu_out_of_bounds():
    """Returns None when requested device_index exceeds detected GPU count."""
    stdout = "NVIDIA GeForce RTX 4090, 24576, 24100\n"
    assert HardwareAttestationEngine._parse_smi_output(stdout, device_index=1) is None
    assert HardwareAttestationEngine._parse_smi_output(stdout, device_index=999) is None
    assert HardwareAttestationEngine._parse_smi_output(stdout, device_index=-1) is None


def test_parse_corrupted_stdout():
    """Returns None on garbage output matching E2E test requirement."""
    assert HardwareAttestationEngine._parse_smi_output("GARBAGE_STDOUT_ERROR_CODE_404") is None
    assert HardwareAttestationEngine._parse_smi_output("") is None
    assert HardwareAttestationEngine._parse_smi_output("   \n\n  ") is None


def test_parse_residual_units():
    """Gracefully strips residual unit strings (e.g. '24576 MiB')."""
    stdout = "NVIDIA GeForce RTX 3090, 24576 MiB, 22000 MiB\n"
    parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
    assert parsed is not None
    assert parsed["vram_mb"] == 24576
    assert parsed["free_vram_mb"] == 22000


def test_parse_comma_in_gpu_name():
    """Correctly handles GPU model name containing an unquoted comma."""
    stdout = "NVIDIA RTX 4090, Special Edition, 24576, 24100\n"
    parsed = HardwareAttestationEngine._parse_smi_output(stdout, device_index=0)
    assert parsed is not None
    assert parsed["gpu_model"] == "NVIDIA RTX 4090, Special Edition"
    assert parsed["vram_mb"] == 24576
    assert parsed["free_vram_mb"] == 24100


# Subprocess Execution & Error Handling Tests
def test_query_nvidia_smi_success(monkeypatch):
    """Verifies successful subprocess invocation and parameter passing."""
    engine = HardwareAttestationEngine(device_index=0)
    monkeypatch.setattr(engine, "_find_nvidia_smi_binary", lambda: "/usr/bin/nvidia-smi")

    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = "NVIDIA GeForce RTX 4090, 24576, 24100\n"
    mock_res.stderr = ""

    with patch("subprocess.run", return_value=mock_res) as mock_run:
        metrics = engine._query_nvidia_smi()
        assert metrics is not None
        assert metrics["vram_mb"] == 24576
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0][0] == "/usr/bin/nvidia-smi"
        assert "--query-gpu=name,memory.total,memory.free" in args[0]
        assert "--format=csv,noheader,nounits" in args[0]
        assert kwargs["timeout"] == 5


def test_query_nvidia_smi_timeout(monkeypatch):
    """Subprocess TimeoutExpired triggers clean warning and returns None."""
    engine = HardwareAttestationEngine()
    monkeypatch.setattr(engine, "_find_nvidia_smi_binary", lambda: "/usr/bin/nvidia-smi")

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)):
        assert engine._query_nvidia_smi() is None


def test_query_nvidia_smi_nonzero_returncode(monkeypatch):
    """Non-zero return code (e.g. driver mismatch) returns None."""
    engine = HardwareAttestationEngine()
    monkeypatch.setattr(engine, "_find_nvidia_smi_binary", lambda: "/usr/bin/nvidia-smi")

    mock_res = MagicMock()
    mock_res.returncode = 9
    mock_res.stdout = ""
    mock_res.stderr = "NVIDIA-SMI failed to communicate with driver."

    with patch("subprocess.run", return_value=mock_res):
        assert engine._query_nvidia_smi() is None


def test_query_nvidia_smi_file_not_found(monkeypatch):
    """FileNotFoundError during execution returns None."""
    engine = HardwareAttestationEngine()
    monkeypatch.setattr(engine, "_find_nvidia_smi_binary", lambda: "/usr/bin/nvidia-smi")

    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert engine._query_nvidia_smi() is None


# ==============================================================================
# TIER 3: CPU / NUMPY FALLBACK TESTS
# ==============================================================================

def test_cpu_numpy_fallback_full_flow(sample_signer):
    """TC-ATT-07: Verifies CPU/NumPy fallback metadata extraction and TFLOPS calculation."""
    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)

    with patch.object(engine, "_query_nvml", return_value=None):
        with patch.object(engine, "_query_nvidia_smi", return_value=None):
            with patch("src.attestation.nvml_attest._get_cpu_model_name", return_value="AMD Ryzen 9 5900X 12-Core Processor"):
                with patch("src.attestation.nvml_attest._get_system_ram_mb", return_value=(32768, 16384)):
                    with patch("src.attestation.nvml_attest._get_cpu_core_count", return_value=12):
                        specs = engine.detect_hardware(timestamp=1788537600)

    assert specs.gpu_model == "CPU: AMD Ryzen 9 5900X 12-Core Processor (NumPy Emulated)"
    assert specs.vram_mb == 32768
    # 12 cores * 0.05 = 0.60 TFLOPS
    assert specs.peak_fp32_tflops == 0.6
    assert len(specs.attestation_proof) == 128


def test_cpu_fallback_without_psutil():
    """TC-ATT-08: Verifies fallback to 16384 MB default when psutil is not available."""
    with patch.dict("sys.modules", {"psutil": None}):
        with patch("os.path.exists", return_value=False):
            total, free = _get_system_ram_mb()
            assert total == 16384
            assert free == 8192


def test_cpu_peak_tflops_minimum_floor():
    """TC-ATT-09: Asserts that low core counts still enforce 0.5 TFLOPS minimum."""
    # 2 cores * 0.05 = 0.10 -> floor to 0.5
    assert estimate_peak_fp32_tflops("CPU: Dual Core (NumPy Emulated)", cpu_cores=2) == 0.5
    assert estimate_peak_fp32_tflops("CPU: Single Core (NumPy Emulated)", cpu_cores=1) == 0.5
    # 16 cores * 0.05 = 0.80
    assert estimate_peak_fp32_tflops("CPU: 16 Core (NumPy Emulated)", cpu_cores=16) == 0.8


# ==============================================================================
# PEAK FP32 TFLOPS & LOOKUP TABLE TESTS
# ==============================================================================

@pytest.mark.parametrize(
    "model_str, expected_tflops",
    [
        ("NVIDIA GeForce RTX 4090", 82.6),
        ("NVIDIA GeForce RTX 4080", 48.7),
        ("NVIDIA GeForce RTX 4070 Ti", 40.1),
        ("NVIDIA GeForce RTX 4070", 29.0),
        ("NVIDIA GeForce RTX 3090 Ti", 40.0),
        ("NVIDIA GeForce RTX 3090", 35.6),
        ("NVIDIA GeForce RTX 3080", 29.8),
        ("NVIDIA GeForce RTX 3070", 20.3),
        ("NVIDIA GeForce RTX 3060", 12.7),
        ("NVIDIA A100-SXM4-80GB", 19.5),
        ("NVIDIA H100 80GB HBM3", 67.0),
        ("Tesla T4", 8.1),
        ("Tesla V100-PCIE-16GB", 15.7),
        ("NVIDIA L4", 30.3),
        ("NVIDIA L40", 90.5),
        ("NVIDIA RTX 6000 Ada Generation", 91.1),
        ("Apple M1", 2.6),
        ("Apple M2 Max", 13.6),
        ("Apple M3", 4.1),
        ("Apple M1 Ultra", 20.8),
    ],
)
def test_curated_gpu_lookup_table(model_str, expected_tflops):
    """TC-ATT-10: Verifies exact TFLOPS ratings for curated prominent GPU models."""
    assert estimate_peak_fp32_tflops(model_str) == expected_tflops


def test_tflops_fuzzy_and_case_insensitive_matching():
    """TC-ATT-11: Verifies case-insensitivity and regex fuzzy matching."""
    assert estimate_peak_fp32_tflops("geforce rtx 4090") == 82.6
    assert estimate_peak_fp32_tflops("RTX 3080 (Founders Edition)") == 29.8
    assert estimate_peak_fp32_tflops("h100-sxm5") == 67.0
    assert estimate_peak_fp32_tflops("a100") == 19.5


def test_unknown_gpu_fallback_tflops():
    """TC-ATT-12: Verifies uncatalogued GPU model defaults safely to 10.0 TFLOPS."""
    assert estimate_peak_fp32_tflops("NVIDIA Unknown Experimental Accelerator") == 10.0


# ==============================================================================
# HARDWARESPECS PACKAGING & CRYPTOGRAPHIC ATTESTATION TESTS
# ==============================================================================

def test_hardware_specs_pydantic_validation(sample_signer):
    """TC-ATT-13: Attestation output conforms strictly to matrix_contracts HardwareSpecs schema."""
    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
    specs = engine.detect_hardware()

    assert isinstance(specs, HardwareSpecs)
    dumped = specs.model_dump()
    assert "gpu_model" in dumped
    assert "vram_mb" in dumped
    assert "peak_fp32_tflops" in dumped
    assert "attestation_proof" in dumped


def test_attestation_proof_signature_verification(sample_signer):
    """TC-ATT-14: Verifies cryptographic attestation proof is valid Secp256k1 signature."""
    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
    specs = engine.detect_hardware(timestamp=1788537600)

    pubkey = sample_signer.get_pubkey_hex()
    is_valid = HardwareAttestationEngine.verify_attestation(
        specs=specs,
        pubkey_hex=pubkey,
        timestamp=1788537600,
    )
    assert is_valid is True

    # Also test standalone function
    assert verify_attestation_proof(specs=specs, signer_pubkey=pubkey, timestamp=1788537600) is True
    # Without timestamp argument (falls back to cache/defaults)
    assert verify_attestation_proof(specs=specs, signer_pubkey=pubkey) is True


def test_attestation_proof_tamper_detection(sample_signer):
    """TC-ATT-15: Modifying attested metrics causes signature verification to fail."""
    engine = HardwareAttestationEngine(signer=sample_signer, device_index=0)
    specs = engine.detect_hardware(timestamp=1788537600)
    pubkey = sample_signer.get_pubkey_hex()

    # Tamper with VRAM
    tampered_vram = specs.model_copy(update={"vram_mb": 999999})
    assert HardwareAttestationEngine.verify_attestation(tampered_vram, pubkey, 1788537600) is False

    # Tamper with GPU Model
    tampered_model = specs.model_copy(update={"gpu_model": "NVIDIA Fraudulent H100"})
    assert HardwareAttestationEngine.verify_attestation(tampered_model, pubkey, 1788537600) is False

    # Tamper with TFLOPS
    tampered_tflops = specs.model_copy(update={"peak_fp32_tflops": 500.0})
    assert HardwareAttestationEngine.verify_attestation(tampered_tflops, pubkey, 1788537600) is False


def test_attestation_engine_with_none_signer():
    """TC-ATT-16: When signer is None, attestation_proof is empty string without error."""
    engine = HardwareAttestationEngine(signer=None, device_index=0)
    specs = engine.detect_hardware()

    assert specs.attestation_proof == ""
    assert isinstance(specs, HardwareSpecs)


def test_device_index_validation_and_bounds(sample_signer):
    """TC-ATT-17: Verifies negative device index raises ValueError and index 999 cascades safely."""
    with pytest.raises(ValueError, match="device_index must be non-negative"):
        HardwareAttestationEngine(signer=sample_signer, device_index=-1)

    # Index 999 simulates missing GPU device -> cascades safely to CPU fallback
    engine = HardwareAttestationEngine(signer=sample_signer, device_index=999)
    specs = engine.detect_hardware()
    assert "CPU" in specs.gpu_model
    assert specs.vram_mb > 0
