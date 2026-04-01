"""verl backend for LoRA + GRPO training.

Provides multi-GPU distributed GRPO training using the verl framework
(Volcano Engine Reinforcement Learning for LLMs). Uses FSDP for distributed
LoRA training and vLLM for parallel rollout generation.

This backend supports scaling to large models (70B+) across multiple GPUs,
unlike the ART backend which is limited to single-GPU time-sharing.

Usage:
    from training_hub import lora_grpo

    # Single-turn tool-call training on 8 GPUs
    result = lora_grpo(
        model_path="Qwen/Qwen3-4B",
        data_path="Agent-Ark/Toucan-1.5M",
        ckpt_output_dir="./grpo_output",
        backend="verl",
        num_iterations=15,
        group_size=8,
        n_gpus=8,
    )
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import Backend, AlgorithmRegistry

logger = logging.getLogger(__name__)


def _prepare_verl_data(
    data_path: str,
    output_dir: str,
    n_train: int = 5000,
    n_val: int = 500,
    data_config: str = "Qwen3",
) -> tuple[str, str]:
    """Convert training data to verl's expected Parquet format.

    verl expects a Parquet file with at minimum a 'prompt' column containing
    chat messages (list of dicts with role/content). Additional columns like
    'ground_truth' are passed through to the reward function.

    For tool-call data, each sample becomes a row with:
    - prompt: [system_msg, user_msg, ...context_msgs] (messages up to the turn)
    - ground_truth: {"tool_name": ..., "tool_args": ...} (expected tool call)
    - tools: list of tool definitions (for the tools= API parameter)

    Returns:
        (train_parquet_path, val_parquet_path)
    """
    import pandas as pd

    # Use our existing data loading + decomposition
    from .lora_grpo import _load_local_dataset, _load_tool_call_dataset

    if data_path.endswith((".json", ".jsonl")):
        train_data, val_data = _load_local_dataset(data_path, n_train, n_val)
    else:
        train_data, val_data = _load_tool_call_dataset(
            data_path, n_train=n_train, n_val=n_val, data_config=data_config,
        )

    def samples_to_parquet(samples, path):
        rows = []
        for s in samples:
            # Build the prompt messages (everything the model sees before generating)
            prompt = [{"role": "system", "content": s.get("system_prompt", "")}]
            prompt.append({"role": "user", "content": s["question"]})
            # Add GT context for multi-turn samples
            for ctx_msg in s.get("context_messages", []):
                prompt.append(ctx_msg)

            row = {
                "prompt": prompt,
                "reward_model": {
                    "ground_truth": json.dumps({
                        "tool_name": s["target_tool_name"],
                        "tool_args": s["target_arguments"],
                    }),
                },
                "tools": json.dumps(s.get("tools", [])),
                "data_source": "tool_call",
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        logger.info("Saved %d samples to %s", len(rows), path)
        return path

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")

    samples_to_parquet(train_data, train_path)
    if val_data:
        samples_to_parquet(val_data, val_path)
    else:
        # verl requires a val file, create a small one from train
        samples_to_parquet(train_data[:min(50, len(train_data))], val_path)

    return train_path, val_path


def _write_reward_function(output_dir: str) -> str:
    """Write the tool-call reward function for verl to load.

    verl loads custom reward functions from a file path + function name.
    This writes a standalone reward module that verl can import.

    Returns:
        Path to the reward function file.
    """
    reward_code = '''
"""Tool-call reward function for verl GRPO training."""
import json


def tool_call_compute_score(data_source, solution_str, ground_truth, **kwargs):
    """Compute reward for tool-call verification.

    Called by verl's reward manager for each generated response.

    Args:
        data_source: Dataset identifier (unused, always "tool_call")
        solution_str: The model's generated response text
        ground_truth: JSON string with {"tool_name": ..., "tool_args": ...}

    Returns:
        float: 1.0 (correct name + args), 0.5 (correct name), 0.0 (wrong/missing)
    """
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    expected_name = gt["tool_name"]
    expected_args = gt.get("tool_args", {})

    # Extract tool call from the model's response
    # verl passes the raw generated text — we need to parse tool calls from it
    predicted_name, predicted_args = _extract_tool_call_from_text(solution_str)

    if predicted_name is None:
        return 0.0

    if not _names_match(predicted_name, expected_name):
        return 0.0

    if _args_match(predicted_args, expected_args):
        return 1.0

    return 0.5


def _extract_tool_call_from_text(text):
    """Extract tool call from model-generated text.

    Handles multiple formats:
    - JSON function call: {"name": "...", "arguments": {...}}
    - Tool call tags: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    if not text:
        return None, {}

    import re

    # Try to find JSON with name/arguments pattern
    patterns = [
        r'<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>',
        r'\\{"name":\\s*"([^"]+)".*?"arguments":\\s*(\\{.*?\\})\\s*\\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                if match.lastindex == 1:
                    obj = json.loads(match.group(1))
                    return obj.get("name"), obj.get("arguments", {})
                else:
                    name = match.group(1)
                    args = json.loads(match.group(2))
                    return name, args
            except json.JSONDecodeError:
                continue

    # Try parsing the entire text as JSON
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "name" in obj:
            return obj["name"], obj.get("arguments", {})
    except json.JSONDecodeError:
        pass

    return None, {}


def _names_match(predicted, expected):
    if predicted == expected:
        return True
    if predicted.endswith(expected) or expected.endswith(predicted):
        return True
    p = predicted.lower().replace("-", "_").replace(".", "_")
    e = expected.lower().replace("-", "_").replace(".", "_")
    return p == e or p.endswith(e) or e.endswith(p)


def _args_match(predicted, expected):
    if isinstance(predicted, str):
        try:
            predicted = json.loads(predicted)
        except json.JSONDecodeError:
            return False
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError:
            return False
    if not predicted and not expected:
        return True
    if not predicted or not expected:
        return False
    try:
        return json.dumps(_normalize(predicted), sort_keys=True) == json.dumps(_normalize(expected), sort_keys=True)
    except (TypeError, ValueError):
        return False


def _normalize(args):
    normalized = {}
    for k, v in args.items():
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v = v.strip()
        normalized[k] = v
    return normalized
'''
    reward_path = os.path.join(output_dir, "verl_reward.py")
    with open(reward_path, "w") as f:
        f.write(reward_code)
    return reward_path


class VeRLLoRAGRPOBackend(Backend):
    """verl backend for distributed multi-GPU LoRA + GRPO training.

    Uses verl's FSDP for distributed LoRA training and vLLM for parallel
    rollout generation. Supports scaling to 70B+ models across multiple GPUs.
    """

    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute LoRA GRPO training using verl."""
        ckpt_output_dir = algorithm_params["ckpt_output_dir"]
        os.makedirs(ckpt_output_dir, exist_ok=True)

        # Prepare data in verl's Parquet format
        data_path = algorithm_params.get("data_path")
        if data_path is None:
            raise ValueError("data_path is required for verl backend")

        data_dir = os.path.join(ckpt_output_dir, "verl_data")
        train_path, val_path = _prepare_verl_data(
            data_path=data_path,
            output_dir=data_dir,
            n_train=algorithm_params.get("n_train", 5000),
            n_val=algorithm_params.get("n_val", 500),
            data_config=algorithm_params.get("data_config", "Qwen3"),
        )

        # Write reward function
        reward_path = _write_reward_function(ckpt_output_dir)

        # Extract parameters
        model_path = algorithm_params["model_path"]
        num_iterations = algorithm_params.get("num_iterations", 15)
        group_size = algorithm_params.get("group_size", 8)
        tasks_per_iteration = algorithm_params.get("tasks_per_iteration", 100)
        learning_rate = algorithm_params.get("learning_rate", 1e-5)
        lora_r = algorithm_params.get("lora_r", 16)
        lora_alpha = algorithm_params.get("lora_alpha", 8)
        max_tokens = algorithm_params.get("max_tokens", 512)
        temperature = algorithm_params.get("temperature", 0.7)
        gpu_memory_utilization = algorithm_params.get("gpu_memory_utilization", 0.35)
        n_gpus = algorithm_params.get("n_gpus", 1)
        tp_size = algorithm_params.get("tensor_parallel_size", 1)

        # Effective batch size: tasks_per_iteration * group_size
        train_batch_size = tasks_per_iteration * group_size

        # Build verl command
        cmd = [
            sys.executable, "-m", "verl.trainer.main_ppo",
            # Algorithm
            "algorithm.adv_estimator=grpo",
            "algorithm.use_kl_in_reward=False",
            # Data
            f"data.train_files={train_path}",
            f"data.val_files={val_path}",
            f"data.train_batch_size={train_batch_size}",
            f"data.max_prompt_length=4096",
            f"data.max_response_length={max_tokens}",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            # Model + LoRA
            f"actor_rollout_ref.model.path={model_path}",
            f"actor_rollout_ref.model.lora_rank={lora_r}",
            f"actor_rollout_ref.model.lora_alpha={lora_alpha}",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            "actor_rollout_ref.model.use_remove_padding=True",
            # Actor (training)
            f"actor_rollout_ref.actor.optim.lr={learning_rate}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={min(train_batch_size, 256)}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={min(train_batch_size // max(n_gpus, 1), 4)}",
            "actor_rollout_ref.actor.use_kl_loss=False",
            "actor_rollout_ref.actor.entropy_coeff=0",
            "actor_rollout_ref.actor.fsdp_config.param_offload=False",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
            # Rollout (vLLM)
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={tp_size}",
            f"actor_rollout_ref.rollout.n={group_size}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}",
            "actor_rollout_ref.rollout.load_format=safetensors",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={min(train_batch_size // max(n_gpus, 1), 4)}",
            # Reference model
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={min(train_batch_size // max(n_gpus, 1), 4)}",
            # Reward
            f"reward.custom_reward_function.path={reward_path}",
            "reward.custom_reward_function.name=tool_call_compute_score",
            # Trainer
            "trainer.critic_warmup=0",
            f"trainer.n_gpus_per_node={n_gpus}",
            "trainer.nnodes=1",
            f"trainer.total_epochs={num_iterations}",
            "trainer.save_freq=5",
            "trainer.test_freq=-1",
            "trainer.val_before_train=False",
            'trainer.logger=["console"]',
            "trainer.project_name=training-hub-grpo",
            f"trainer.experiment_name=lora_grpo",
            f"trainer.default_local_dir={ckpt_output_dir}/checkpoints",
            "trainer.resume_mode=auto",
        ]

        # Add wandb/mlflow logging if configured
        if algorithm_params.get("wandb_project"):
            cmd.append(f'trainer.logger=["console","wandb"]')
            cmd.append(f"trainer.project_name={algorithm_params['wandb_project']}")

        logger.info(
            "Starting verl GRPO training: model=%s, n_gpus=%d, "
            "lora_r=%d, lr=%s, epochs=%d, batch=%d, group=%d",
            model_path, n_gpus, lora_r, learning_rate,
            num_iterations, train_batch_size, group_size,
        )

        # Run verl as a subprocess
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "true"
        env["NCCL_DEBUG"] = "WARN"
        env["VLLM_LOGGING_LEVEL"] = "WARN"
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"

        result = subprocess.run(
            cmd,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"verl training failed with exit code {result.returncode}"
            )

        # Collect results
        return {
            "status": "success",
            "checkpoint_path": os.path.join(ckpt_output_dir, "checkpoints"),
            "backend": "verl",
            "model_path": model_path,
            "n_gpus": n_gpus,
            "lora_r": lora_r,
            "num_iterations": num_iterations,
        }


# Register the verl backend for the lora_grpo algorithm
# (algorithm is already registered by lora_grpo.py, we just add a new backend)
try:
    AlgorithmRegistry.register_backend("lora_grpo", "verl", VeRLLoRAGRPOBackend)
except ValueError:
    pass  # Algorithm not registered yet (import order), will be registered on first use
