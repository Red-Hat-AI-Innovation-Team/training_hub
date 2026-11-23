"""Tests for exporting PEFT adapters from verl FSDP checkpoints (_normalize_verl_checkpoints)."""

import json
import os

import pytest
import torch

pytest.importorskip("verl.model_merger.fsdp_model_merger")
peft = pytest.importorskip("peft")
from transformers import Qwen3Config, Qwen3ForCausalLM

from training_hub.algorithms.lora_grpo_verl import _normalize_verl_checkpoints

TARGET_MODULES = ["q_proj", "v_proj"]


def _tiny_base():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    )
    return Qwen3ForCausalLM(config)


def _trained_peft_model():
    model = peft.get_peft_model(
        _tiny_base(),
        peft.LoraConfig(r=4, lora_alpha=8, target_modules=TARGET_MODULES, task_type="CAUSAL_LM"),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn_like(param))
    return model


def _write_verl_checkpoint(actor_dir, model, lora_only: bool, lora: bool = True):
    """Mimic verl 0.9.1 FSDPCheckpointManager.save_checkpoint output for world_size=1."""
    os.makedirs(os.path.join(actor_dir, "huggingface"))
    state_dict = model.state_dict()
    if lora_only:
        state_dict = {k: v for k, v in state_dict.items() if "lora_" in k}
    torch.save(state_dict, os.path.join(actor_dir, "model_world_size_1_rank_0.pt"))
    with open(os.path.join(actor_dir, "fsdp_config.json"), "w") as f:
        json.dump({"FSDP_version": 1, "world_size": 1}, f)
    if lora:
        with open(os.path.join(actor_dir, "lora_train_meta.json"), "w") as f:
            json.dump({"r": 4, "lora_alpha": 8, "task_type": "CAUSAL_LM"}, f)
    base_config = model.config if not lora else model.base_model.model.config
    base_config.save_pretrained(os.path.join(actor_dir, "huggingface"))
    with open(os.path.join(actor_dir, "huggingface", "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "Qwen2Tokenizer"}, f)


@pytest.mark.parametrize("lora_only", [True, False], ids=["save_lora_only", "full_shards"])
def test_exports_loadable_peft_adapter(tmp_path, lora_only):
    trained = _trained_peft_model()
    actor_dir = tmp_path / "checkpoints" / "global_step_3" / "actor"
    _write_verl_checkpoint(str(actor_dir), trained, lora_only=lora_only)

    adapters = _normalize_verl_checkpoints(str(tmp_path / "checkpoints"))

    lora_dir = actor_dir / "lora_adapter"
    assert adapters == [str(lora_dir)]
    files = set(os.listdir(lora_dir))
    assert {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"} <= files

    adapter_config = json.loads((lora_dir / "adapter_config.json").read_text())
    assert (adapter_config["r"], adapter_config["lora_alpha"]) == (4, 8)
    assert sorted(adapter_config["target_modules"]) == TARGET_MODULES

    loaded = peft.PeftModel.from_pretrained(_tiny_base(), str(lora_dir))
    expected = {k: v for k, v in trained.state_dict().items() if "lora_" in k}
    got = {k: v for k, v in loaded.state_dict().items() if "lora_" in k}
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        # the verl merger exports in bf16
        torch.testing.assert_close(got[key], value.to(torch.bfloat16).to(value.dtype))


def test_skips_full_finetune_checkpoints(tmp_path):
    actor_dir = tmp_path / "checkpoints" / "global_step_3" / "actor"
    _write_verl_checkpoint(str(actor_dir), _tiny_base(), lora_only=False, lora=False)

    assert _normalize_verl_checkpoints(str(tmp_path / "checkpoints")) == []
    assert not (actor_dir / "lora_adapter").exists()
