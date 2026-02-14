"""
GPU detection and compute configuration for Ollama inference.

Probes for available GPU hardware (NVIDIA CUDA, AMD ROCm, Apple Metal)
and configures Ollama options accordingly. Falls back to CPU with
optimized thread settings when no GPU is found.
"""

import os
import platform
import logging
import subprocess
import multiprocessing
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    """Detected GPU hardware information."""
    has_gpu: bool = False
    backend: str = "cpu"           # "cuda", "rocm", "metal", "cpu"
    device_name: str = ""
    vram_mb: int = 0
    driver_version: str = ""
    cpu_threads: int = 1

    @property
    def summary(self) -> str:
        if self.has_gpu:
            vram = f"{self.vram_mb}MB VRAM" if self.vram_mb else "unknown VRAM"
            return f"{self.backend.upper()}: {self.device_name} ({vram})"
        return f"CPU only ({self.cpu_threads} threads)"


def probe_gpu() -> GpuInfo:
    """
    Detect available GPU hardware.

    Checks in order: NVIDIA (nvidia-smi), AMD (rocm-smi), Apple Metal.
    Returns GpuInfo with detection results.
    """
    cpu_threads = multiprocessing.cpu_count() or 4
    info = GpuInfo(cpu_threads=cpu_threads)

    # 1. NVIDIA CUDA
    nvidia = _probe_nvidia()
    if nvidia:
        info.has_gpu = True
        info.backend = "cuda"
        info.device_name = nvidia["name"]
        info.vram_mb = nvidia["vram_mb"]
        info.driver_version = nvidia["driver"]
        logger.info(f"GPU detected: {info.summary}")
        return info

    # 2. AMD ROCm
    rocm = _probe_rocm()
    if rocm:
        info.has_gpu = True
        info.backend = "rocm"
        info.device_name = rocm["name"]
        info.vram_mb = rocm["vram_mb"]
        logger.info(f"GPU detected: {info.summary}")
        return info

    # 3. Apple Metal (macOS)
    metal = _probe_metal()
    if metal:
        info.has_gpu = True
        info.backend = "metal"
        info.device_name = metal["name"]
        info.vram_mb = metal["vram_mb"]
        logger.info(f"GPU detected: {info.summary}")
        return info

    logger.warning(f"No GPU detected. Falling back to CPU inference ({cpu_threads} threads)")
    return info


def get_ollama_options(gpu_info: GpuInfo) -> dict:
    """
    Build Ollama runtime options based on detected hardware.

    GPU present:  let Ollama use all GPU layers (default behavior)
    CPU only:     set num_gpu=0, optimize num_thread for available cores
    """
    if gpu_info.has_gpu:
        return {
            # Let Ollama auto-detect layer count for the GPU
            # num_gpu: -1 means "use all layers on GPU" (Ollama default)
        }
    else:
        return {
            "num_gpu": 0,
            "num_thread": gpu_info.cpu_threads,
        }


def _probe_nvidia() -> dict | None:
    """Probe for NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            return {
                "name": parts[0],
                "vram_mb": int(float(parts[1])),
                "driver": parts[2],
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _probe_rocm() -> dict | None:
    """Probe for AMD GPU via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        name = "AMD GPU"
        vram_mb = 0

        # Parse product name from rocm-smi output
        for line in result.stdout.strip().split("\n"):
            if "card" in line.lower() or "gpu" in line.lower():
                # Try to extract the device name
                parts = line.split(",")
                if len(parts) >= 2:
                    name = parts[1].strip() or name

        # Try to get VRAM separately
        mem_result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10,
        )
        if mem_result.returncode == 0:
            for line in mem_result.stdout.split("\n"):
                if "total" in line.lower():
                    # Extract number (typically in bytes or MB)
                    import re
                    nums = re.findall(r"(\d+)", line)
                    if nums:
                        val = int(nums[-1])
                        # If value > 100000, assume bytes and convert
                        vram_mb = val // (1024 * 1024) if val > 100000 else val

        return {"name": name, "vram_mb": vram_mb}

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _probe_metal() -> dict | None:
    """Probe for Apple Metal GPU on macOS."""
    if platform.system() != "Darwin":
        return None

    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        name = ""
        vram_mb = 0
        for line in result.stdout.split("\n"):
            line = line.strip()
            if "Chipset Model:" in line or "Chip:" in line:
                name = line.split(":", 1)[1].strip()
            elif "VRAM" in line or "Memory" in line:
                import re
                nums = re.findall(r"(\d+)", line)
                if nums:
                    val = int(nums[0])
                    # Apple reports in MB or GB
                    vram_mb = val * 1024 if val < 128 else val

        if not name:
            # Check for Apple Silicon unified memory
            chip_result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if chip_result.returncode == 0 and "Apple" in chip_result.stdout:
                name = chip_result.stdout.strip()
                # Unified memory — get total system RAM as approximation
                mem_result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5,
                )
                if mem_result.returncode == 0:
                    vram_mb = int(mem_result.stdout.strip()) // (1024 * 1024)

        if name:
            return {"name": name, "vram_mb": vram_mb}

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None
