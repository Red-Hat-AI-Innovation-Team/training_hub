"""Tests for Intel XPU device handling.

The mocked tests run on any host. Tests decorated with ``requires_xpu`` carry
the ``xpu`` marker, exercise a real XPU, and are skipped when ``torch.xpu`` is
unavailable. Select them with ``pytest -m xpu``; ``-m "not xpu"`` runs the rest.
"""

import pytest
import torch

from training_hub import checkpoint_manager as cm
from training_hub.utils import get_torchrun_params

_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()



def requires_xpu(obj):
    """Mark a test as needing a real Intel XPU: ``xpu`` marker + skip when absent."""
    return pytest.mark.xpu(pytest.mark.skipif(not _HAS_XPU, reason="Intel XPU not available")(obj))


@pytest.fixture(autouse=True)
def _clear_torchrun_env(monkeypatch):
    for var in ("PET_NPROC_PER_NODE", "NPROC_PER_NODE"):
        monkeypatch.delenv(var, raising=False)


def _fake_devices(monkeypatch, *, cuda: bool, xpu: bool):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu)


# ---------------------------------------------------------------------------
# nproc_per_node resolution (mocked devices)
# ---------------------------------------------------------------------------


class TestNprocPerNode:
    def test_auto_picks_xpu_when_only_xpu_present(self, monkeypatch):
        _fake_devices(monkeypatch, cuda=False, xpu=True)
        assert get_torchrun_params({"nproc_per_node": "auto"})["nproc_per_node"] == "xpu"

    def test_auto_prefers_cuda_when_both_present(self, monkeypatch):
        _fake_devices(monkeypatch, cuda=True, xpu=True)
        assert get_torchrun_params({"nproc_per_node": "auto"})["nproc_per_node"] == "gpu"

    def test_explicit_xpu_accepted(self, monkeypatch):
        _fake_devices(monkeypatch, cuda=False, xpu=True)
        assert get_torchrun_params({"nproc_per_node": "XPU"})["nproc_per_node"] == "xpu"

    def test_gpu_still_means_cuda(self, monkeypatch):
        # torchrun's 'gpu' counts CUDA devices only, so it must not silently
        # resolve on an XPU-only host.
        _fake_devices(monkeypatch, cuda=False, xpu=True)
        with pytest.raises(ValueError, match="requires CUDA GPUs"):
            get_torchrun_params({"nproc_per_node": "gpu"})

    def test_xpu_without_device_raises(self, monkeypatch):
        _fake_devices(monkeypatch, cuda=False, xpu=False)
        with pytest.raises(ValueError, match="requires Intel XPUs"):
            get_torchrun_params({"nproc_per_node": "xpu"})

    def test_auto_without_any_accelerator_raises(self, monkeypatch):
        _fake_devices(monkeypatch, cuda=False, xpu=False)
        with pytest.raises(ValueError, match="CUDA GPUs or Intel XPUs"):
            get_torchrun_params({"nproc_per_node": "auto"})


# ---------------------------------------------------------------------------
# any_rank reduction device (mocked process group)
# ---------------------------------------------------------------------------


def _fake_dist(monkeypatch, backend: str, seen: list):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: backend)
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda t, op=None: seen.append(t.device.type)
    )


class TestAnyRankDevice:
    def test_xccl_reduces_on_xpu(self, monkeypatch):
        seen = []
        _fake_dist(monkeypatch, "xccl", seen)
        _fake_devices(monkeypatch, cuda=False, xpu=True)
        monkeypatch.setattr(torch.xpu, "current_device", lambda: 0)
        monkeypatch.setattr(torch, "tensor", _DeviceProbe)
        assert cm.any_rank(True) is True
        assert seen == ["xpu"]

    def test_gloo_reduces_on_cpu_even_with_xpu(self, monkeypatch):
        seen = []
        _fake_dist(monkeypatch, "gloo", seen)
        _fake_devices(monkeypatch, cuda=False, xpu=True)
        monkeypatch.setattr(torch, "tensor", _DeviceProbe)
        assert cm.any_rank(False) is False
        assert seen == ["cpu"]


class _DeviceProbe:
    """Stands in for a tensor so mocked tests never allocate on a real device."""

    def __init__(self, data, device=None):
        self.device = device
        self._value = data[0]

    def item(self):
        return self._value


# ---------------------------------------------------------------------------
# Real XPU
# ---------------------------------------------------------------------------


@requires_xpu
class TestRealXpu:
    def test_auto_resolves_to_xpu(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA present; 'auto' resolves to CUDA first")
        assert get_torchrun_params({"nproc_per_node": "auto"})["nproc_per_node"] == "xpu"

    def test_tensor_roundtrip_and_bf16(self):
        x = torch.arange(8, dtype=torch.float32, device="xpu")
        y = (x * 2).to(torch.bfloat16)
        assert y.device.type == "xpu"
        assert torch.xpu.is_bf16_supported()
        assert y.float().cpu().tolist() == [float(i * 2) for i in range(8)]

    def test_any_rank_without_process_group_is_local(self):
        # No process group: any_rank must not touch the device and return as-is.
        assert cm.any_rank(True) is True
        assert cm.any_rank(False) is False


@requires_xpu
@pytest.mark.skipif(
    not (hasattr(torch.distributed, "is_xccl_available") and torch.distributed.is_xccl_available()),
    reason="XCCL backend not available",
)
def test_collectives_on_real_xccl_group(monkeypatch, caplog):
    import logging

    import torch.distributed as dist

    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29571")
    torch.xpu.set_device(0)
    dist.init_process_group("xccl", rank=0, world_size=1)
    try:
        assert dist.get_backend() == "xccl"
        cm.wait_for_all_ranks()
        with caplog.at_level(logging.ERROR, logger=cm.logger.name):
            assert cm.any_rank(True) is True
            assert cm.any_rank(False) is False
        # any_rank swallows errors and returns the local flag, so the asserts
        # above alone can't tell a real all_reduce from the fallback.
        assert "Rank sync failed" not in caplog.text
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# get_accelerator_module (mocked devices)
# ---------------------------------------------------------------------------


class TestAcceleratorModule:
    @pytest.mark.parametrize(
        "cuda,xpu,expected",
        [(True, False, "cuda"), (True, True, "cuda"), (False, True, "xpu"), (False, False, None)],
    )
    def test_selection(self, monkeypatch, cuda, xpu, expected):
        from training_hub.utils import get_accelerator_module

        _fake_devices(monkeypatch, cuda=cuda, xpu=xpu)
        module = get_accelerator_module()
        assert module is (getattr(torch, expected) if expected else None)


@requires_xpu
def test_real_accelerator_module_is_xpu():
    from training_hub.utils import get_accelerator_module

    if torch.cuda.is_available():
        pytest.skip("CUDA present; CUDA is preferred")
    module = get_accelerator_module()
    assert module is torch.xpu
    assert module.device_count() >= 1
