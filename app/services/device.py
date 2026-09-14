"""
Single source of truth for which device models run on.

The project runs models on GPU. The one exception is the pytest suite, which
must stay runnable on GitHub Actions runners that have no GPU, so it sets
ALLOW_CPU=1 explicitly. Nothing in the application sets it.

This replaces scattered `torch.cuda.is_available()` guessing - notably
`use_cpu=(device == "cpu")` in the old train_transformer.py - which silently
produced CPU execution whenever a CUDA build was missing.
"""

import functools
import os

import torch

ALLOW_CPU_ENV = "ALLOW_CPU"


def cpu_allowed() -> bool:
    return os.getenv(ALLOW_CPU_ENV, "").strip().lower() in ("1", "true", "yes")


def resolve_device(purpose: str) -> torch.device:
    """
    Returns the device `purpose` must run on, or raises if CUDA is unavailable.

    `purpose` is quoted back in the error so the failure names what was being
    attempted ("training", "inference") rather than surfacing a bare assertion.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if cpu_allowed():
        return torch.device("cpu")

    raise RuntimeError(
        f"{purpose} requires CUDA, but torch.cuda.is_available() is False.\n"
        f"Installed torch: {torch.__version__}\n"
        f"If that version string ends in '+cpu', the CPU-only build is installed; "
        f"reinstall from the CUDA wheel index (see pyproject.toml "
        f"[[tool.uv.index]] pytorch-cu130) with: uv sync\n"
        f"Set {ALLOW_CPU_ENV}=1 only to run the test suite without a GPU."
    )


@functools.cache
def describe_device() -> dict:
    """Device facts for /health. Cached: these do not change within a process."""
    available = torch.cuda.is_available()
    info = {
        "cuda_available": available,
        "torch_version": torch.__version__,
        "cpu_allowed": cpu_allowed(),
        "device": "cuda" if available else ("cpu" if cpu_allowed() else "none"),
    }
    if available:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
    return info
