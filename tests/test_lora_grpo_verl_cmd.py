"""Tests for the verl command VeRLLoRAGRPOBackend builds (captured at Popen, verl never runs)."""

import pandas as pd
import pytest

from training_hub.algorithms import lora_grpo_verl
from training_hub.algorithms.lora_grpo_verl import VeRLLoRAGRPOBackend

SAVE_LORA_ONLY = "+actor_rollout_ref.actor.checkpoint.save_lora_only=True"


class _CapturedCmd(Exception):
    def __init__(self, cmd):
        super().__init__("verl launch intercepted")
        self.cmd = cmd


def _verl_cmd(tmp_path, monkeypatch, **params) -> list[str]:
    """Run execute_training up to the verl launch and return the command it would run."""
    train_path = tmp_path / "train.parquet"
    pd.DataFrame({"prompt": ["q"] * 8}).to_parquet(train_path)

    monkeypatch.setattr(
        lora_grpo_verl, "_prepare_verl_data",
        lambda **_: (str(train_path), str(train_path)),
    )
    monkeypatch.setattr(lora_grpo_verl, "_write_reward_function", lambda d: f"{d}/reward.py")
    monkeypatch.setattr(lora_grpo_verl, "_write_agent_loop", lambda d: f"{d}/agent_loop.py")

    def _intercept(cmd, **_):
        raise _CapturedCmd(cmd)

    monkeypatch.setattr(lora_grpo_verl.subprocess, "Popen", _intercept)
    for var in ("WANDB_PROJECT", "MLFLOW_TRACKING_URI"):
        monkeypatch.delenv(var, raising=False)

    algorithm_params = {
        "ckpt_output_dir": str(tmp_path / "out"),
        "data_path": "unused.jsonl",
        "model_path": "Qwen/Qwen3-4B",
        "prompt_batch_size": 4,
        "group_size": 2,
        **params,
    }
    with pytest.raises(_CapturedCmd) as captured:
        VeRLLoRAGRPOBackend().execute_training(algorithm_params)
    return captured.value.cmd


def test_lora_run_checkpoints_lora_weights_only(tmp_path, monkeypatch):
    cmd = _verl_cmd(tmp_path, monkeypatch, lora_r=32, lora_alpha=32)

    assert "actor_rollout_ref.model.lora_rank=32" in cmd
    assert SAVE_LORA_ONLY in cmd


def test_full_finetune_keeps_full_checkpoints(tmp_path, monkeypatch):
    cmd = _verl_cmd(tmp_path, monkeypatch, lora_r=0)

    assert not any("lora_rank" in arg for arg in cmd)
    assert SAVE_LORA_ONLY not in cmd
