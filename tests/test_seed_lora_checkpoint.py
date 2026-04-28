"""Unit tests for _create_seed_lora_checkpoint (vLLM V1 compatibility)."""

import json
import os

import pytest
import torch
from safetensors.torch import load_file

from training_hub.algorithms.lora_grpo import _create_seed_lora_checkpoint


class TestCreateSeedLoraCheckpoint:
    """Tests for the seed LoRA checkpoint creation."""

    def test_creates_required_files(self, model_path, tmp_output_dir):
        """Seed checkpoint must produce adapter_config.json and adapter_model.safetensors."""
        ckpt_path = os.path.join(tmp_output_dir, "seed_ckpt")

        _create_seed_lora_checkpoint(
            model_path=model_path,
            ckpt_path=ckpt_path,
            lora_r=16,
            lora_alpha=8,
        )

        assert os.path.isfile(os.path.join(ckpt_path, "adapter_config.json"))
        assert os.path.isfile(os.path.join(ckpt_path, "adapter_model.safetensors"))

    def test_config_fields(self, model_path, tmp_output_dir):
        """adapter_config.json must contain correct LoRA parameters."""
        ckpt_path = os.path.join(tmp_output_dir, "seed_ckpt")

        _create_seed_lora_checkpoint(
            model_path=model_path,
            ckpt_path=ckpt_path,
            lora_r=32,
            lora_alpha=16,
        )

        with open(os.path.join(ckpt_path, "adapter_config.json")) as f:
            config = json.load(f)

        assert config["r"] == 32
        assert config["lora_alpha"] == 16
        assert config["peft_type"] == "LORA"
        assert config["task_type"] == "CAUSAL_LM"
        assert config["base_model_name_or_path"] == model_path
        # target_modules should be resolved to a list (not the string "all-linear")
        assert isinstance(config["target_modules"], list)
        assert len(config["target_modules"]) > 0

    def test_weights_are_zero(self, model_path, tmp_output_dir):
        """All LoRA weights in the seed checkpoint must be zero (identity transform)."""
        ckpt_path = os.path.join(tmp_output_dir, "seed_ckpt")

        _create_seed_lora_checkpoint(
            model_path=model_path,
            ckpt_path=ckpt_path,
            lora_r=8,
            lora_alpha=4,
        )

        state_dict = load_file(os.path.join(ckpt_path, "adapter_model.safetensors"))
        assert len(state_dict) > 0, "State dict should contain LoRA tensors"

        for name, tensor in state_dict.items():
            assert torch.all(tensor == 0), f"Tensor {name} should be all zeros"

    def test_weight_shapes_match_lora_rank(self, model_path, tmp_output_dir):
        """LoRA A/B weight dimensions must match the requested rank."""
        lora_r = 32
        ckpt_path = os.path.join(tmp_output_dir, "seed_ckpt")

        _create_seed_lora_checkpoint(
            model_path=model_path,
            ckpt_path=ckpt_path,
            lora_r=lora_r,
            lora_alpha=16,
        )

        state_dict = load_file(os.path.join(ckpt_path, "adapter_model.safetensors"))
        for name, tensor in state_dict.items():
            if "lora_A" in name:
                assert tensor.shape[0] == lora_r, (
                    f"{name}: lora_A first dim should be {lora_r}, got {tensor.shape[0]}"
                )
            elif "lora_B" in name:
                assert tensor.shape[1] == lora_r, (
                    f"{name}: lora_B second dim should be {lora_r}, got {tensor.shape[1]}"
                )

    def test_skips_when_exists(self, model_path, tmp_output_dir):
        """Must not overwrite an existing checkpoint (resume case)."""
        ckpt_path = os.path.join(tmp_output_dir, "seed_ckpt")
        os.makedirs(ckpt_path, exist_ok=True)

        # Write a sentinel file
        config_path = os.path.join(ckpt_path, "adapter_config.json")
        sentinel = {"sentinel": True}
        with open(config_path, "w") as f:
            json.dump(sentinel, f)

        _create_seed_lora_checkpoint(
            model_path=model_path,
            ckpt_path=ckpt_path,
            lora_r=16,
            lora_alpha=8,
        )

        # Should still be the sentinel, not overwritten
        with open(config_path) as f:
            assert json.load(f) == sentinel

    def test_custom_target_modules(self, model_path, tmp_output_dir):
        """Custom target_modules should appear in the saved config."""
        ckpt_path = os.path.join(tmp_output_dir, "seed_ckpt")
        custom_modules = ["q_proj", "v_proj"]

        _create_seed_lora_checkpoint(
            model_path=model_path,
            ckpt_path=ckpt_path,
            lora_r=16,
            lora_alpha=8,
            target_modules=custom_modules,
        )

        with open(os.path.join(ckpt_path, "adapter_config.json")) as f:
            config = json.load(f)

        assert set(config["target_modules"]) == set(custom_modules)

    def test_different_ranks_produce_different_shapes(self, model_path, tmp_output_dir):
        """Two seed checkpoints with different ranks must have different weight shapes."""
        ckpt_8 = os.path.join(tmp_output_dir, "r8")
        ckpt_64 = os.path.join(tmp_output_dir, "r64")

        _create_seed_lora_checkpoint(
            model_path=model_path, ckpt_path=ckpt_8, lora_r=8, lora_alpha=4,
        )
        _create_seed_lora_checkpoint(
            model_path=model_path, ckpt_path=ckpt_64, lora_r=64, lora_alpha=32,
        )

        sd_8 = load_file(os.path.join(ckpt_8, "adapter_model.safetensors"))
        sd_64 = load_file(os.path.join(ckpt_64, "adapter_model.safetensors"))

        # Same number of keys (same architecture)
        assert set(sd_8.keys()) == set(sd_64.keys())

        # But different shapes
        for key in sd_8:
            assert sd_8[key].shape != sd_64[key].shape, (
                f"{key}: shapes should differ between r=8 and r=64"
            )
