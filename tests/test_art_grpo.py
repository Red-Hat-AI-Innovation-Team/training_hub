"""Functional tests for the ART LoRA GRPO backend.

These tests run real (tiny) training on a GPU to verify the end-to-end
ART GRPO pipeline works, including the vLLM V1 seed-checkpoint fix.

Requires:
  - CUDA GPU available
  - openpipe-art installed
  - A locally cached model (Qwen2.5-7B-Instruct, Qwen3-4B, etc.)

Run:
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_art_grpo.py -v -s
"""

import json
import os

import pytest
import torch


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_skip_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required"
)

try:
    import art  # noqa: F401
    _has_art = True
except ImportError:
    _has_art = False

_skip_no_art = pytest.mark.skipif(not _has_art, reason="openpipe-art not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generic_data_file(tmp_path):
    """Create a small generic Q&A dataset in JSONL format."""
    questions = [
        {"question": "What is 2 + 2?", "ground_truth": "4"},
        {"question": "What is the capital of France?", "ground_truth": "Paris"},
        {"question": "What color is the sky on a clear day?", "ground_truth": "blue"},
        {"question": "What is 10 * 5?", "ground_truth": "50"},
        {"question": "How many days are in a week?", "ground_truth": "7"},
        {"question": "What is the boiling point of water in Celsius?", "ground_truth": "100"},
        {"question": "What planet is closest to the Sun?", "ground_truth": "Mercury"},
        {"question": "What is the chemical symbol for water?", "ground_truth": "H2O"},
        {"question": "How many continents are there?", "ground_truth": "7"},
        {"question": "What is the square root of 144?", "ground_truth": "12"},
    ]
    data_path = str(tmp_path / "generic_data.jsonl")
    with open(data_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    return data_path


# Module-level reward function so it's picklable by multiprocessing.spawn.
def keyword_reward(*, data_source, solution_str, ground_truth):
    """Simple reward: 1.0 if the ground-truth string appears in the response."""
    if ground_truth.lower() in solution_str.lower():
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_skip_no_gpu
@_skip_no_art
@pytest.mark.gpu
@pytest.mark.slow
class TestARTGRPOGenericData:
    """End-to-end ART GRPO with generic data and custom reward function."""

    def test_single_iteration(self, model_path, generic_data_file, tmp_path):
        """Run 1 GRPO iteration and verify results structure and outputs."""
        from training_hub import lora_grpo

        ckpt_dir = str(tmp_path / "grpo_output")

        result = lora_grpo(
            model_path=model_path,
            ckpt_output_dir=ckpt_dir,
            data_path=generic_data_file,
            reward_fn=keyword_reward,
            backend="art",
            # Minimal config for speed
            num_iterations=1,
            group_size=2,
            prompt_batch_size=5,
            learning_rate=1e-5,
            lora_r=8,
            lora_alpha=4,
            max_tokens=64,
            concurrency=4,
            gpu_memory_utilization=0.3,
        )

        # -- Result structure --
        assert isinstance(result, dict)
        assert result.get("status") == "success", f"Training failed: {result}"
        assert "reward_history" in result
        assert "checkpoint_path" in result
        assert "total_time_seconds" in result
        assert "total_rollouts" in result

        # -- Reward history --
        assert len(result["reward_history"]) == 1
        reward = result["reward_history"][0]
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0

        # -- Rollout count --
        expected_rollouts = 5 * 2  # prompt_batch_size * group_size
        assert result["total_rollouts"] == expected_rollouts

        # -- Training metrics file --
        metrics_path = os.path.join(ckpt_dir, "training_metrics.jsonl")
        assert os.path.isfile(metrics_path)
        with open(metrics_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        # Should have at least a rollout entry
        assert len(entries) >= 1
        phases = {e["phase"] for e in entries}
        assert "rollout" in phases

        # -- Results JSON persisted to disk --
        results_json = os.path.join(ckpt_dir, "training_results.json")
        assert os.path.isfile(results_json)
        with open(results_json) as f:
            saved = json.load(f)
        assert saved["status"] == "success"
        assert len(saved["reward_history"]) == 1

    def test_seed_checkpoint_created(self, model_path, generic_data_file, tmp_path):
        """Verify the seed LoRA checkpoint is created in the ART directory."""
        from training_hub import lora_grpo

        ckpt_dir = str(tmp_path / "grpo_output")

        result = lora_grpo(
            model_path=model_path,
            ckpt_output_dir=ckpt_dir,
            data_path=generic_data_file,
            reward_fn=keyword_reward,
            backend="art",
            num_iterations=1,
            group_size=2,
            prompt_batch_size=5,
            lora_r=8,
            lora_alpha=4,
            max_tokens=64,
            concurrency=4,
            gpu_memory_utilization=0.3,
        )

        assert result["status"] == "success"

        # The seed checkpoint should exist at the ART path
        art_path = os.path.join(ckpt_dir, ".art")
        assert os.path.isdir(art_path), f"ART path not found: {art_path}"

        # Verify adapter_config.json exists somewhere in the checkpoints tree
        found_adapter_config = False
        for root, _dirs, files in os.walk(art_path):
            if "adapter_config.json" in files:
                found_adapter_config = True
                break
        assert found_adapter_config, (
            "adapter_config.json not found in ART checkpoint directory"
        )
