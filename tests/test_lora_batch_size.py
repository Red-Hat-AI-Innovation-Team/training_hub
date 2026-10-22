"""Tests for LoRA effective batch-size handling."""

import sys
import types
from pathlib import Path

import pytest

from training_hub.algorithms.lora import UnslothLoRABackend


def _install_fake_trl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal SFTConfig replacement for configuration-only tests."""
    module = types.ModuleType("trl")

    class SFTConfig:
        def __init__(self, **kwargs: object) -> None:
            """Expose configuration values as attributes like the real class."""
            self.__dict__.update(kwargs)

    module.SFTConfig = SFTConfig
    monkeypatch.setitem(sys.modules, "trl", module)


def test_non_divisible_effective_batch_size_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject an effective batch size that cannot be represented exactly."""
    _install_fake_trl(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(
        ValueError,
        match=(
            r"effective_batch_size \(10\).*gradient_accumulation_steps \(3\) "
            r"\* num_gpus \(2\).*Try effective_batch_size=6 or 12"
        ),
    ):
        UnslothLoRABackend()._build_training_args(
            {
                "ckpt_output_dir": str(tmp_path),
                "effective_batch_size": 10,
                "gradient_accumulation_steps": 3,
            }
        )


def test_divisible_effective_batch_size_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Derive the expected per-device batch size for a divisible value."""
    _install_fake_trl(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")

    config = UnslothLoRABackend()._build_training_args(
        {
            "ckpt_output_dir": str(tmp_path),
            "effective_batch_size": 12,
            "gradient_accumulation_steps": 3,
        }
    )

    assert config.per_device_train_batch_size == 2


def test_explicit_micro_batch_size_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preserve an explicit micro-batch size when no effective size is given."""
    _install_fake_trl(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")

    config = UnslothLoRABackend()._build_training_args(
        {
            "ckpt_output_dir": str(tmp_path),
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 3,
        }
    )

    assert config.per_device_train_batch_size == 4
