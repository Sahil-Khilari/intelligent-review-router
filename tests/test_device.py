import os

import pytest
import torch

from app.services import device as device_mod


def test_cpu_allowed_reads_env(monkeypatch):
    monkeypatch.setenv("ALLOW_CPU", "1")
    assert device_mod.cpu_allowed() is True
    monkeypatch.setenv("ALLOW_CPU", "0")
    assert device_mod.cpu_allowed() is False
    monkeypatch.delenv("ALLOW_CPU", raising=False)
    assert device_mod.cpu_allowed() is False


def test_resolve_device_returns_cuda_when_available():
    dev = device_mod.resolve_device("test")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.type == expected


def test_resolve_device_raises_without_cuda_or_escape(monkeypatch):
    """The whole point: no silent CPU execution in the application."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("ALLOW_CPU", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        device_mod.resolve_device("Training")
    message = str(excinfo.value)
    assert "Training requires CUDA" in message
    assert "ALLOW_CPU" in message


def test_resolve_device_allows_cpu_with_escape(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("ALLOW_CPU", "1")
    assert device_mod.resolve_device("Tests").type == "cpu"


def test_describe_device_reports_shape():
    device_mod.describe_device.cache_clear()
    info = device_mod.describe_device()
    assert set(["cuda_available", "torch_version", "cpu_allowed", "device"]) <= set(info)
    assert info["device"] in ("cuda", "cpu", "none")
