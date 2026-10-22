"""Tests for LoRA effective batch-size handling."""

import sys
import types

import pytest

from training_hub.algorithms.lora import UnslothLoRABackend


def _install_fake_trl(monkeypatch):
    module = types.ModuleType("trl")

    class SFTConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module.SFTConfig = SFTConfig
    monkeypatch.setitem(sys.modules, "trl", module)


def test_non_divisible_effective_batch_size_is_rejected(monkeypatch, tmp_path):
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


def test_divisible_effective_batch_size_is_preserved(monkeypatch, tmp_path):
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


def test_explicit_micro_batch_size_is_preserved(monkeypatch, tmp_path):
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
