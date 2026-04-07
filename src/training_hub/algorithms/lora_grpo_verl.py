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

            tools = s.get("tools", [])

            # Embed tools in the system prompt so the model sees tool definitions
            # during generation. This is simpler than configuring verl's tool agent
            # loop and works with the default SingleTurnAgentLoop.
            if tools:
                tools_text = json.dumps(tools, indent=2)
                tools_instruction = (
                    "\n\nYou have access to the following tools. To call a tool, "
                    "respond with a JSON object in this format: "
                    '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n\n'
                    f"Tools:\n{tools_text}"
                )
                if prompt and prompt[0]["role"] == "system":
                    prompt[0] = {
                        "role": "system",
                        "content": prompt[0]["content"] + tools_instruction,
                    }
                else:
                    prompt.insert(0, {"role": "system", "content": tools_instruction})

            row = {
                "prompt": prompt,
                "reward_model": {
                    "ground_truth": json.dumps({
                        "tool_name": s["target_tool_name"],
                        "tool_args": s["target_arguments"],
                    }),
                },
                "extra_info": {
                    "index": len(rows),
                },
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

    Handles Qwen3 format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    Also handles thinking blocks and various JSON formats.
    """
    if not text:
        return None, {}

    import re

    # Strip thinking blocks first
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Try <tool_call> tags (Qwen3 format) - use greedy match for nested braces
    tc_match = re.search(r'<tool_call>\\s*(.+?)\\s*</tool_call>', text, re.DOTALL)
    if tc_match:
        try:
            obj = json.loads(tc_match.group(1))
            if isinstance(obj, dict):
                return obj.get("name"), obj.get("arguments", {})
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object with name+arguments
    # Find all { } balanced blocks
    for match in re.finditer(r'\\{', text):
        start = match.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i+1])
                        if isinstance(obj, dict) and "name" in obj:
                            return obj["name"], obj.get("arguments", {})
                    except json.JSONDecodeError:
                        pass
                    break

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


def _write_agent_loop(output_dir: str) -> tuple[str, str]:
    """Write a single-turn tool-calling agent loop for verl.

    verl's default SingleTurnAgentLoop doesn't pass tools to the chat template.
    This custom agent loop passes tools from the dataset's tools_kwargs field,
    enabling models like Qwen3 to generate structured tool calls.

    Returns:
        (agent_module_path, agent_config_path)
    """
    import shutil

    # Copy the agent loop module to the output dir
    src = os.path.join(os.path.dirname(__file__), "verl_tool_agent.py")
    dst = os.path.join(output_dir, "verl_tool_agent.py")
    shutil.copy2(src, dst)

    # Also install it into verl's agent_loop directory so Ray workers can import it
    import verl.experimental.agent_loop as agent_loop_pkg
    agent_loop_dir = os.path.dirname(agent_loop_pkg.__file__)
    installed_path = os.path.join(agent_loop_dir, "verl_tool_agent.py")
    shutil.copy2(src, installed_path)

    # Patch the __init__.py to import it
    init_path = os.path.join(agent_loop_dir, "__init__.py")
    with open(init_path, "r") as f:
        init_content = f.read()
    if "verl_tool_agent" not in init_content:
        with open(init_path, "a") as f:
            f.write("\nfrom .verl_tool_agent import SingleTurnToolAgentLoop\n")
            f.write("_ = [*_, SingleTurnToolAgentLoop]\n")

    return dst, installed_path


def _normalize_verl_checkpoints(checkpoint_dir: str) -> list[str]:
    """Merge tokenizer files into verl's lora_adapter dirs for direct vLLM use.

    verl saves adapter weights and tokenizer files in separate subdirs:
        global_step_N/actor/lora_adapter/{adapter_config.json, adapter_model.safetensors}
        global_step_N/actor/huggingface/{tokenizer files, config.json}

    This copies the huggingface files into lora_adapter/ so that directory
    is self-contained and directly loadable by vLLM as a LoRA adapter path.
    The full checkpoint (FSDP state, optimizer, etc.) is preserved for resume.

    Returns:
        List of paths to the usable lora_adapter checkpoint directories.
    """
    import shutil

    adapter_paths = []
    if not os.path.exists(checkpoint_dir):
        return adapter_paths

    for entry in sorted(os.listdir(checkpoint_dir)):
        if not entry.startswith("global_step_"):
            continue

        raw_dir = os.path.join(checkpoint_dir, entry)
        lora_dir = os.path.join(raw_dir, "actor", "lora_adapter")
        hf_dir = os.path.join(raw_dir, "actor", "huggingface")

        if not os.path.isdir(lora_dir):
            continue

        # Copy tokenizer/config files into lora_adapter dir
        if os.path.isdir(hf_dir):
            for f in os.listdir(hf_dir):
                dst = os.path.join(lora_dir, f)
                if not os.path.exists(dst):  # don't overwrite adapter files
                    shutil.copy2(os.path.join(hf_dir, f), dst)

        adapter_paths.append(lora_dir)
        logger.info("Normalized checkpoint: %s/actor/lora_adapter/", entry)

    return adapter_paths


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

        # Write reward function and install custom agent loop
        reward_path = _write_reward_function(ckpt_output_dir)
        _write_agent_loop(ckpt_output_dir)

        # Extract parameters
        model_path = algorithm_params["model_path"]
        num_iterations = algorithm_params.get("num_iterations", 15)
        group_size = algorithm_params.get("group_size", 8)
        tasks_per_iteration = algorithm_params.get("tasks_per_iteration", 100)
        learning_rate = algorithm_params.get("learning_rate", 1e-5)
        lora_r = algorithm_params.get("lora_r", 16)
        lora_alpha = algorithm_params.get("lora_alpha", 8)
        max_tokens = algorithm_params.get("max_tokens", 512)
        max_prompt_length = algorithm_params.get("max_prompt_length", 16384)
        temperature = algorithm_params.get("temperature", 0.7)
        gpu_memory_utilization = algorithm_params.get("gpu_memory_utilization", 0.35)
        n_gpus = algorithm_params.get("n_gpus", 1)
        tp_size = algorithm_params.get("tensor_parallel_size", 1)

        # train_batch_size = number of prompts per batch. verl's rollout.n
        # (set from group_size) controls how many rollouts per prompt, so
        # batch size should NOT be multiplied by group_size.
        train_batch_size = tasks_per_iteration

        # Calculate steps per epoch for checkpoint frequency.
        # Save once per epoch by default so each epoch has a usable checkpoint.
        import pandas as pd
        train_dataset_size = len(pd.read_parquet(train_path))
        steps_per_epoch = max(1, (train_dataset_size + train_batch_size - 1) // train_batch_size)
        save_freq = steps_per_epoch

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
            f"data.max_prompt_length={max_prompt_length}",
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
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={min(train_batch_size // max(n_gpus, 1), 2)}",
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
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={min(train_batch_size // max(n_gpus, 1), 2)}",
            # Reference model
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={min(train_batch_size // max(n_gpus, 1), 2)}",
            # Reward
            f"reward.custom_reward_function.path={reward_path}",
            "reward.custom_reward_function.name=tool_call_compute_score",
            # Trainer
            "trainer.critic_warmup=0",
            f"trainer.n_gpus_per_node={n_gpus}",
            "trainer.nnodes=1",
            f"trainer.total_epochs={num_iterations}",
            f"trainer.save_freq={save_freq}",
            "trainer.test_freq=-1",
            "trainer.val_before_train=False",
            "trainer.project_name=training-hub-grpo",
            "trainer.experiment_name=lora_grpo",
            f"trainer.default_local_dir={ckpt_output_dir}/checkpoints",
            "trainer.resume_mode=auto",
        ]

        # Experiment tracking (with env var fallbacks, matching other algorithms)
        loggers = ["console"]
        wandb_project = algorithm_params.get("wandb_project") or os.environ.get("WANDB_PROJECT")
        mlflow_tracking_uri = algorithm_params.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")

        if wandb_project:
            loggers.append("wandb")
            cmd.append(f"trainer.project_name={wandb_project}")
        if mlflow_tracking_uri:
            loggers.append("mlflow")
            mlflow_experiment_name = (
                algorithm_params.get("mlflow_experiment_name")
                or os.environ.get("MLFLOW_EXPERIMENT_NAME")
            )
            if mlflow_experiment_name and not wandb_project:
                # verl maps project_name → MLflow experiment
                cmd.append(f"trainer.project_name={mlflow_experiment_name}")

        wandb_run_name = algorithm_params.get("wandb_run_name") or os.environ.get("WANDB_RUN_NAME")
        mlflow_run_name = algorithm_params.get("mlflow_run_name") or os.environ.get("MLFLOW_RUN_NAME")
        if wandb_run_name or mlflow_run_name:
            experiment_name = wandb_run_name or mlflow_run_name
            cmd.append(f"trainer.experiment_name={experiment_name}")

        cmd.append(f'trainer.logger={json.dumps(loggers)}')

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

        # Pass tracking env vars to subprocess
        wandb_entity = algorithm_params.get("wandb_entity") or os.environ.get("WANDB_ENTITY")
        if wandb_entity:
            env["WANDB_ENTITY"] = wandb_entity
        if mlflow_tracking_uri:
            env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        # Add output dir to PYTHONPATH so verl can import the custom agent loop
        env["PYTHONPATH"] = ckpt_output_dir + ":" + env.get("PYTHONPATH", "")

        # Stream output to stdout, log file, and parse metrics live
        import re
        log_path = os.path.join(ckpt_output_dir, "verl_training.log")
        metrics_path = os.path.join(ckpt_output_dir, "training_metrics.jsonl")
        reward_history = []
        step_pattern = re.compile(r'step:(\d+)\s*-\s*(.*)')

        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                # Stream to stdout live
                sys.stdout.write(line)
                sys.stdout.flush()
                # Save to log file
                log_file.write(line)
                log_file.flush()

                # Parse step metrics and write training_metrics.jsonl live
                match = step_pattern.search(line)
                if match:
                    step = int(match.group(1))
                    metrics_str = match.group(2)
                    raw = {}
                    for kv in metrics_str.split(" - "):
                        kv = kv.strip()
                        if ":" not in kv:
                            continue
                        key, val = kv.split(":", 1)
                        try:
                            raw[key.strip()] = float(val.strip())
                        except ValueError:
                            pass

                    entry = {
                        "step": step,
                        "epoch": raw.get("training/epoch", 0),
                        "mean_reward": raw.get("critic/score/mean", 0),
                        "loss": raw.get("actor/pg_loss"),
                        "grad_norm": raw.get("actor/grad_norm"),
                        "learning_rate": raw.get("actor/lr"),
                        "entropy": raw.get("actor/entropy"),
                        "response_length_mean": raw.get("response_length/mean"),
                        "prompt_length_mean": raw.get("prompt_length/mean"),
                        "wall_time_s": raw.get("timing_s/step"),
                    }
                    entry = {k: v for k, v in entry.items() if v is not None}
                    reward_history.append(entry.get("mean_reward", 0))

                    with open(metrics_path, "a") as mf:
                        mf.write(json.dumps(entry) + "\n")
                        mf.flush()

            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"verl training failed with exit code {proc.returncode}"
            )

        # Post-training: normalize checkpoints into flat adapter dirs
        # (matching ART's format for uniform vLLM loading)
        checkpoint_dir = os.path.join(ckpt_output_dir, "checkpoints")
        adapter_checkpoints = _normalize_verl_checkpoints(checkpoint_dir)

        return {
            "status": "success",
            "checkpoint_path": checkpoint_dir,
            "adapter_checkpoints": adapter_checkpoints,
            "reward_history": reward_history,
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
