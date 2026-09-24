"""Training Hub CLI.

Provides the ``thub`` command for running training algorithms from the
command line with YAML config files or direct argument passing.

Usage::

    thub sft --model-path ./my-model --data-path ./data.jsonl --ckpt-output-dir ./out
    thub osft --config my_config.yaml
    thub lora-sft --config base.yaml --learning-rate 1e-3  # CLI overrides config
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from typing import Any, Callable, NoReturn, Optional

_ALGO_PARAM_DEFS: dict[str, dict] = {}


def _fail(message: str) -> NoReturn:
    """Print a clean CLI error to stderr and exit with status 1."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def _get_version() -> str:
    """Return a human-readable version string for ``--version``."""
    try:
        from training_hub._version import version
        return f"training-hub {version}"
    except ImportError:
        return "training-hub (version unknown — not installed from a tagged release)"


def _define_params() -> None:
    """Define parameter specs for each algorithm subcommand.

    Each entry maps a CLI flag name to a dict with:
      - type: the Python type for argparse
      - python_name: the kwarg name passed to the convenience function
      - help: help text
      - default: default value (None means optional, other values are explicit defaults)
      - callable: if True, the value is a dotted import path resolved at runtime
      - json: if True, the value is parsed as JSON
      - nargs: argparse nargs (for list params)
    """
    # Common logging params shared across most algorithms
    logging_params = {
        "--wandb-project": {"type": str, "help": "Weights & Biases project name"},
        "--wandb-entity": {"type": str, "help": "Weights & Biases team/entity name"},
        "--wandb-run-name": {"type": str, "help": "Weights & Biases run name"},
        "--tensorboard-log-dir": {"type": str, "help": "Directory for TensorBoard logs"},
        "--mlflow-tracking-uri": {"type": str, "help": "MLflow tracking server URI"},
        "--mlflow-experiment-name": {"type": str, "help": "MLflow experiment name"},
        "--mlflow-run-name": {"type": str, "help": "MLflow run name"},
    }

    torchrun_params = {
        "--nproc-per-node": {"type": str, "help": "Number of processes (GPUs) per node"},
        "--nnodes": {"type": int, "help": "Total number of nodes"},
        "--node-rank": {"type": int, "help": "Rank of this node (0 to nnodes-1)"},
        "--rdzv-id": {"type": str, "help": "Unique job ID for rendezvous"},
        "--rdzv-endpoint": {"type": str, "help": "Master node endpoint for multi-node training"},
        "--master-addr": {"type": str, "help": "Master node address for distributed training"},
        "--master-port": {"type": int, "help": "Master node port for distributed training"},
    }

    adamw_params = {
        "--beta1": {"type": float, "help": "AdamW beta1 coefficient"},
        "--beta2": {"type": float, "help": "AdamW beta2 coefficient"},
        "--eps": {"type": float, "help": "AdamW epsilon for numerical stability"},
        "--weight-decay": {"type": float, "help": "AdamW weight decay coefficient"},
    }

    # --- SFT ---
    _ALGO_PARAM_DEFS["sft"] = {
        "--model-path": {"type": str, "required": True, "help": "Path to the model to fine-tune"},
        "--data-path": {"type": str, "required": True, "help": "Path to the training data"},
        "--ckpt-output-dir": {"type": str, "required": True, "help": "Directory to save checkpoints"},
        "--backend": {"type": str, "default": "instructlab-training", "help": "Backend implementation (default: instructlab-training)"},
        "--num-epochs": {"type": int, "help": "Number of training epochs"},
        "--effective-batch-size": {"type": int, "help": "Effective batch size for training"},
        "--learning-rate": {"type": float, "help": "Learning rate for training"},
        "--max-seq-len": {"type": int, "help": "Maximum sequence length"},
        "--max-tokens-per-gpu": {"type": int, "help": "Maximum tokens per GPU in a mini-batch"},
        "--data-output-dir": {"type": str, "help": "Directory to save processed data"},
        "--save-samples": {"type": int, "help": "Number of samples to save after training"},
        "--warmup-steps": {"type": int, "help": "Number of warmup steps"},
        "--accelerate-full-state-at-epoch": {"type": bool, "help": "Save full state at epoch"},
        "--checkpoint-at-epoch": {"type": bool, "help": "Checkpoint at each epoch"},
        "--is-pretraining": {"type": bool, "help": "Enable document-style continual pretraining mode"},
        "--block-size": {"type": int, "help": "Token length of each document block (required when is_pretraining=True)"},
        "--document-column-name": {"type": str, "help": "Column name containing raw documents for pretraining"},
        **adamw_params,
        **torchrun_params,
        **logging_params,
    }

    # --- OSFT ---
    _ALGO_PARAM_DEFS["osft"] = {
        "--model-path": {"type": str, "required": True, "help": "Path to the model to fine-tune"},
        "--data-path": {"type": str, "required": True, "help": "Path to the training data"},
        "--unfreeze-rank-ratio": {"type": float, "required": True, "help": "Fraction of each matrix to unfreeze (0.0-1.0)"},
        "--effective-batch-size": {"type": int, "required": True, "help": "Effective batch size for training"},
        "--max-tokens-per-gpu": {"type": int, "required": True, "help": "Maximum tokens per GPU"},
        "--max-seq-len": {"type": int, "required": True, "help": "Maximum sequence length"},
        "--learning-rate": {"type": float, "required": True, "help": "Learning rate"},
        "--ckpt-output-dir": {"type": str, "required": True, "help": "Directory to save checkpoints"},
        "--data-output-dir": {"type": str, "help": "Directory to save processed data"},
        "--backend": {"type": str, "default": "mini-trainer", "help": "Backend implementation (default: mini-trainer)"},
        "--target-patterns": {"type": str, "nargs": "+", "help": "Patterns for selecting OSFT modules"},
        "--seed": {"type": int, "help": "Random seed"},
        "--use-liger": {"type": bool, "help": "Use Liger kernels"},
        "--use-processed-dataset": {"type": bool, "help": "Use pre-processed dataset"},
        "--unmask-messages": {"type": bool, "help": "Unmask messages during training"},
        "--is-pretraining": {"type": bool, "help": "Enable pretraining mode"},
        "--block-size": {"type": int, "help": "Token length of each document block"},
        "--document-column-name": {"type": str, "help": "Column name for pretraining documents"},
        "--lr-scheduler": {"type": str, "help": "Learning rate scheduler type"},
        "--warmup-steps": {"type": int, "help": "Number of warmup steps"},
        "--lr-scheduler-kwargs": {"type": str, "json": True, "help": "LR scheduler kwargs as JSON object"},
        "--checkpoint-at-epoch": {"type": bool, "help": "Checkpoint at each epoch"},
        "--save-final-checkpoint": {"type": bool, "help": "Save final checkpoint"},
        "--num-epochs": {"type": int, "help": "Number of training epochs"},
        "--trust-remote-code": {"type": bool, "help": "Trust remote code when loading models (executes arbitrary code from the model repo; enable only for trusted models)"},
        **adamw_params,
        **torchrun_params,
        **logging_params,
    }

    # --- LoRA SFT ---
    _ALGO_PARAM_DEFS["lora-sft"] = {
        "--model-path": {"type": str, "required": True, "help": "Path to the model to fine-tune"},
        "--data-path": {"type": str, "required": True, "help": "Path to the training data"},
        "--ckpt-output-dir": {"type": str, "required": True, "help": "Directory to save checkpoints"},
        "--backend": {"type": str, "default": "unsloth", "help": "Backend implementation (default: unsloth)"},
        # LoRA params
        "--lora-r": {"type": int, "help": "LoRA rank (default: 16)"},
        "--lora-alpha": {"type": int, "help": "LoRA alpha parameter (default: 32)"},
        "--lora-dropout": {"type": float, "help": "LoRA dropout rate (default: 0.0)"},
        "--target-modules": {"type": str, "nargs": "+", "help": "Modules to apply LoRA to"},
        # Training
        "--num-epochs": {"type": int, "help": "Number of training epochs"},
        "--effective-batch-size": {"type": int, "help": "Effective batch size"},
        "--micro-batch-size": {"type": int, "help": "Batch size per GPU (default: 2)"},
        "--gradient-accumulation-steps": {"type": int, "help": "Gradient accumulation steps"},
        "--learning-rate": {"type": float, "help": "Learning rate (default: 2e-4)"},
        "--max-seq-len": {"type": int, "help": "Maximum sequence length (default: 2048)"},
        "--lr-scheduler": {"type": str, "help": "LR scheduler type (default: linear)"},
        "--warmup-steps": {"type": int, "help": "Number of warmup steps (default: 10)"},
        # Quantization
        "--load-in-4bit": {"type": bool, "help": "Use 4-bit quantization (QLoRA)"},
        "--load-in-8bit": {"type": bool, "help": "Use 8-bit quantization"},
        "--bnb-4bit-quant-type": {"type": str, "help": "4-bit quantization type (default: nf4)"},
        "--bnb-4bit-compute-dtype": {"type": str, "help": "Compute dtype for 4-bit (default: bfloat16)"},
        "--bnb-4bit-use-double-quant": {"type": bool, "help": "Use double quantization"},
        # Optimization
        "--flash-attention": {"type": bool, "help": "Use Flash Attention"},
        "--sample-packing": {"type": bool, "help": "Pack multiple samples per sequence"},
        "--bf16": {"type": bool, "help": "Use bfloat16 precision"},
        "--fp16": {"type": bool, "help": "Use float16 precision"},
        "--tf32": {"type": bool, "help": "Use TensorFloat-32"},
        # Saving
        "--save-steps": {"type": int, "help": "Steps between checkpoints (default: 500)"},
        "--eval-steps": {"type": int, "help": "Steps between evaluations (default: 500)"},
        "--logging-steps": {"type": int, "help": "Steps between log outputs (default: 1)"},
        "--save-total-limit": {"type": int, "help": "Max checkpoints to keep (default: 3)"},
        # Dataset format
        "--dataset-type": {"type": str, "help": "Dataset type: chat_template, alpaca, passthrough"},
        "--field-messages": {"type": str, "help": "Messages field name (default: messages)"},
        "--field-instruction": {"type": str, "help": "Instruction field name (alpaca format)"},
        "--field-input": {"type": str, "help": "Input field name (alpaca format)"},
        "--field-output": {"type": str, "help": "Output field name (alpaca format)"},
        # Multi-GPU
        "--enable-model-splitting": {"type": bool, "help": "Enable device_map=balanced for large models"},
        # VLM
        "--finetune-vision-layers": {"type": bool, "help": "Fine-tune vision layers (VLM)"},
        "--finetune-language-layers": {"type": bool, "help": "Fine-tune language layers (VLM)"},
        # Model loading
        "--trust-remote-code": {"type": bool, "help": "Trust remote code when loading models (executes arbitrary code from the model repo; enable only for trusted models)"},
        **torchrun_params,
        **logging_params,
    }

    # --- LoRA GRPO ---
    _ALGO_PARAM_DEFS["lora-grpo"] = {
        "--model-path": {"type": str, "required": True, "help": "HuggingFace model ID or local path"},
        "--ckpt-output-dir": {"type": str, "required": True, "help": "Directory to save checkpoints"},
        # Data source
        "--data-path": {"type": str, "help": "Dataset path (HuggingFace ID or local JSON/JSONL)"},
        "--data-config": {"type": str, "default": "Qwen3", "help": "HuggingFace dataset config name (default: Qwen3)"},
        "--n-train": {"type": int, "default": 5000, "help": "Number of training samples (default: 5000)"},
        "--n-val": {"type": int, "default": 500, "help": "Number of validation samples (default: 500)"},
        # Custom rollout
        "--rollout-fn": {"type": str, "callable": True, "help": "Dotted import path to async rollout function"},
        "--tasks": {"type": str, "json": True, "help": "Tasks as a JSON array (e.g. '[{\"prompt\": ...}]')"},
        "--reward-fn": {"type": str, "callable": True, "help": "Dotted import path to reward function"},
        # GRPO hyperparameters
        "--num-iterations": {"type": int, "default": 15, "help": "GRPO training iterations (default: 15)"},
        "--group-size": {"type": int, "default": 8, "help": "Rollouts per task (default: 8)"},
        "--prompt-batch-size": {"type": int, "default": 100, "help": "Unique prompts per step (default: 100)"},
        "--learning-rate": {"type": float, "default": 1e-5, "help": "Learning rate (default: 1e-5)"},
        "--temperature": {"type": float, "default": 0.7, "help": "Sampling temperature (default: 0.7)"},
        "--max-tokens": {"type": int, "default": 512, "help": "Max response tokens (default: 512)"},
        "--max-prompt-length": {"type": int, "default": 16384, "help": "Max prompt length in tokens (default: 16384)"},
        "--concurrency": {"type": int, "default": 32, "help": "Max concurrent rollouts (default: 32)"},
        # LoRA
        "--lora-r": {"type": int, "default": 16, "help": "LoRA rank (default: 16)"},
        "--lora-alpha": {"type": int, "default": 8, "help": "LoRA alpha (default: 8)"},
        "--target-modules": {"type": str, "nargs": "+", "help": "Modules to apply LoRA to"},
        "--max-grad-norm": {"type": float, "default": 0.1, "help": "Gradient clipping norm (default: 0.1)"},
        # vLLM
        "--gpu-memory-utilization": {"type": float, "default": 0.45, "help": "vLLM GPU memory fraction (default: 0.45)"},
        "--max-lora-rank": {"type": int, "help": "Max LoRA rank for vLLM engine"},
        "--enforce-eager": {"type": bool, "default": False, "help": "Disable CUDA graph for vLLM"},
        # Multi-GPU
        "--n-gpus": {"type": int, "default": 1, "help": "Number of GPUs (default: 1)"},
        "--nnodes": {"type": int, "default": 1, "help": "Number of nodes (default: 1)"},
        "--tensor-parallel-size": {"type": int, "default": 1, "help": "vLLM tensor parallelism (default: 1)"},
        # Algorithm variant
        "--use-dr-grpo": {"type": bool, "default": True, "help": "Use Dr. GRPO variant (default: True)"},
        # Backend
        "--backend": {"type": str, "default": "verl", "help": "Backend: verl or art (default: verl)"},
        # Callbacks
        "--iteration-callback": {"type": str, "callable": True, "help": "Dotted import path to iteration callback"},
        # Logging (subset - no tensorboard for GRPO)
        "--wandb-project": {"type": str, "help": "Weights & Biases project name"},
        "--wandb-entity": {"type": str, "help": "Weights & Biases team/entity name"},
        "--wandb-run-name": {"type": str, "help": "Weights & Biases run name"},
        "--mlflow-tracking-uri": {"type": str, "help": "MLflow tracking server URI"},
        "--mlflow-experiment-name": {"type": str, "help": "MLflow experiment name"},
        "--mlflow-run-name": {"type": str, "help": "MLflow run name"},
    }

    # --- GRPO (full fine-tuning) ---
    _ALGO_PARAM_DEFS["grpo"] = {
        "--model-path": {"type": str, "required": True, "help": "HuggingFace model ID or local path"},
        "--ckpt-output-dir": {"type": str, "required": True, "help": "Directory to save checkpoints"},
        # Data source
        "--data-path": {"type": str, "help": "Dataset path (HuggingFace ID or local JSON/JSONL)"},
        "--data-config": {"type": str, "default": "Qwen3", "help": "HuggingFace dataset config name (default: Qwen3)"},
        "--n-train": {"type": int, "default": 5000, "help": "Number of training samples (default: 5000)"},
        "--n-val": {"type": int, "default": 500, "help": "Number of validation samples (default: 500)"},
        # Custom reward
        "--reward-fn": {"type": str, "callable": True, "help": "Dotted import path to reward function"},
        # GRPO hyperparameters
        "--num-iterations": {"type": int, "default": 15, "help": "GRPO training iterations (default: 15)"},
        "--group-size": {"type": int, "default": 8, "help": "Rollouts per prompt (default: 8)"},
        "--prompt-batch-size": {"type": int, "default": 100, "help": "Unique prompts per step (default: 100)"},
        "--learning-rate": {"type": float, "default": 1e-5, "help": "Learning rate (default: 1e-5)"},
        "--temperature": {"type": float, "default": 0.7, "help": "Sampling temperature (default: 0.7)"},
        "--max-tokens": {"type": int, "default": 512, "help": "Max response tokens (default: 512)"},
        "--max-prompt-length": {"type": int, "default": 16384, "help": "Max prompt length in tokens (default: 16384)"},
        # vLLM
        "--gpu-memory-utilization": {"type": float, "default": 0.45, "help": "vLLM GPU memory fraction (default: 0.45)"},
        # Multi-GPU
        "--n-gpus": {"type": int, "default": 1, "help": "Number of GPUs (default: 1)"},
        "--nnodes": {"type": int, "default": 1, "help": "Number of nodes (default: 1)"},
        "--tensor-parallel-size": {"type": int, "default": 1, "help": "vLLM tensor parallelism (default: 1)"},
        # Algorithm variant
        "--use-dr-grpo": {"type": bool, "default": True, "help": "Use Dr. GRPO variant (default: True)"},
        # Callbacks
        "--iteration-callback": {"type": str, "callable": True, "help": "Dotted import path to iteration callback"},
        # Logging
        "--wandb-project": {"type": str, "help": "Weights & Biases project name"},
        "--wandb-entity": {"type": str, "help": "Weights & Biases team/entity name"},
        "--wandb-run-name": {"type": str, "help": "Weights & Biases run name"},
        "--mlflow-tracking-uri": {"type": str, "help": "MLflow tracking server URI"},
        "--mlflow-experiment-name": {"type": str, "help": "MLflow experiment name"},
        "--mlflow-run-name": {"type": str, "help": "MLflow run name"},
    }

    # --- GEPA ---
    _ALGO_PARAM_DEFS["gepa"] = {
        "--seed-candidate": {"type": str, "json": True, "required": True, "help": 'Initial prompt as JSON, e.g. \'{"system_prompt": "You are..."}\''},
        "--task-lm": {"type": str, "required": True, "help": "Model string (e.g. openai/gpt-4o-mini)"},
        "--data-path": {"type": str, "help": "Path to JSONL training data"},
        "--trainset": {"type": str, "json": True, "help": "Training examples as JSON array"},
        "--valset": {"type": str, "json": True, "help": "Validation examples as JSON array"},
        "--output-dir": {"type": str, "help": "Directory to save optimized prompt and results"},
        "--backend": {"type": str, "default": "gepa", "help": "Backend: gepa or mlflow (default: gepa)"},
        # Model config
        "--evaluator": {"type": str, "callable": True, "help": "Dotted import path to evaluator function"},
        "--reflection-lm": {"type": str, "help": "Model for reflection/mutation"},
        "--api-base": {"type": str, "help": "Base URL for the LLM API endpoint"},
        # Optimization
        "--max-metric-calls": {"type": int, "help": "Maximum number of evaluation calls"},
        "--candidate-selection-strategy": {"type": str, "help": "Selection strategy: pareto, current_best, epsilon_greedy, top_k_pareto"},
        "--frontier-type": {"type": str, "help": "Pareto frontier type: instance, objective, hybrid, cartesian"},
        "--skip-perfect-score": {"type": bool, "help": "Skip perfect-scoring candidates"},
        "--perfect-score": {"type": float, "help": "Score value considered perfect (default 1.0)"},
        "--reflection-minibatch-size": {"type": int, "help": "Examples per reflection batch"},
        "--seed": {"type": int, "help": "Random seed"},
        # Adapter
        "--adapter": {"type": str, "callable": True, "help": "Dotted import path to GEPAAdapter"},
        # Logging
        "--run-dir": {"type": str, "help": "Directory for GEPA run logs"},
        "--use-wandb": {"type": bool, "help": "Enable Weights & Biases logging"},
        "--wandb-api-key": {"type": str, "help": "W&B API key (visible in process listings/shell history; prefer the WANDB_API_KEY environment variable)"},
        "--wandb-init-kwargs": {"type": str, "json": True, "help": "Additional W&B init kwargs as JSON"},
        "--use-mlflow": {"type": bool, "help": "Enable MLflow logging"},
        "--mlflow-tracking-uri": {"type": str, "help": "MLflow tracking server URI"},
        "--mlflow-experiment-name": {"type": str, "help": "MLflow experiment name"},
        # Advanced
        "--batch-sampler": {"type": str, "callable": True, "help": "Dotted import path to batch sampler"},
        "--reflection-prompt-template": {"type": str, "help": "Custom reflection prompt template (string or JSON dict)"},
        "--custom-candidate-proposer": {"type": str, "callable": True, "help": "Dotted import path to custom proposer"},
        "--module-selector": {"type": str, "help": "Module selection strategy"},
        "--use-merge": {"type": bool, "help": "Enable candidate merging"},
        "--stop-callbacks": {"type": str, "callable": True, "help": "Dotted import path to stop callbacks"},
        "--callbacks": {"type": str, "json": True, "help": "Callbacks as JSON or dotted import path"},
        "--display-progress-bar": {"type": bool, "help": "Show progress bar"},
        "--cache-evaluation": {"type": bool, "help": "Cache evaluation results"},
        "--raise-on-exception": {"type": bool, "help": "Raise exceptions instead of logging"},
        # MLflow backend
        "--predict-fn": {"type": str, "callable": True, "help": "Dotted import path to predict function (MLflow backend)"},
        "--prompt-uris": {"type": str, "nargs": "+", "help": "MLflow prompt URIs"},
        "--scorers": {"type": str, "json": True, "help": "MLflow Scorer instances as JSON or dotted import path"},
        "--aggregation": {"type": str, "callable": True, "help": "Dotted import path to aggregation function"},
        "--enable-tracking": {"type": bool, "help": "Log optimization progress to MLflow"},
        "--gepa-kwargs": {"type": str, "json": True, "help": "Additional kwargs for gepa.optimize() as JSON"},
    }


# Maps CLI subcommand name to the Python convenience function name
_SUBCOMMAND_TO_FUNC = {
    "sft": "sft",
    "osft": "osft",
    "lora-sft": "lora_sft",
    "lora-grpo": "lora_grpo",
    "grpo": "grpo",
    "gepa": "gepa",
}

# Maps the convenience function name to its (module, attribute). Imports are done
# lazily for the selected subcommand only — algorithm modules transitively pull in
# heavy deps (torch, transformers, vLLM, ...), so importing all of them up front is
# slow and lets one algorithm's missing dependency break every other subcommand.
_FUNC_IMPORTS: dict[str, tuple[str, str]] = {
    "sft": ("training_hub.algorithms.sft", "sft"),
    "osft": ("training_hub.algorithms.osft", "osft"),
    "lora_sft": ("training_hub.algorithms.lora", "lora_sft"),
    "lora_grpo": ("training_hub.algorithms.lora_grpo", "lora_grpo"),
    "grpo": ("training_hub.algorithms.lora_grpo", "grpo"),
    "gepa": ("training_hub.algorithms.gepa", "gepa"),
}


def _flag_to_python_name(flag: str) -> str:
    """Convert a CLI flag like ``--learning-rate`` to a Python kwarg ``learning_rate``."""
    return flag.lstrip("-").replace("-", "_")


def _resolve_dotted_path(path: str) -> Any:
    """Import and return the object at a dotted path like ``my_module.my_func``.

    Supports nested attributes: ``pkg.module.Class.method``.

    Security note: this executes ``importlib.import_module()`` on the caller's
    string with no allowlist, and callable parameters (``--reward-fn``,
    ``--rollout-fn``, etc.) resolve to arbitrary imported objects. A shared YAML
    config file is therefore a **trust boundary** — a config authored by an
    untrusted party can execute arbitrary code (e.g. ``reward_fn: os.system``).
    Only run configs you trust.
    """
    if "." not in path:
        raise ValueError(
            f"Cannot resolve '{path}': expected a dotted path like 'module.function'"
        )

    # Try progressively shorter module prefixes (longest first) so nested
    # attributes like ``pkg.module.Class.method`` resolve correctly.
    parts = path.split(".")
    last_missing: Optional[ModuleNotFoundError] = None
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            # Only treat "this prefix isn't a module" as a signal to try a
            # shorter prefix. A *missing dependency* of a module that does
            # exist is a real error — surface it instead of masking it.
            if e.name and (e.name == module_path or module_path.startswith(f"{e.name}.")):
                last_missing = e
                continue
            raise
        # The module imported cleanly; resolving the remaining attributes must
        # succeed or fail loudly (don't fall through and hide a real typo/bug).
        obj = module
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError as e:
            raise ImportError(
                f"Cannot resolve dotted path '{path}': imported module "
                f"'{module_path}' but attribute lookup failed ({e})."
            ) from e
        return obj

    hint = f" (last import error: {last_missing})" if last_missing else ""
    raise ImportError(f"Cannot resolve dotted path '{path}': no importable module prefix found{hint}.")


def _load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML config file and return as a flat dict with Python-style keys."""
    try:
        import yaml
    except ImportError:
        print(
            "error: PyYAML is required for config file support. "
            "Install it with: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        _fail(f"config file not found: {path}")
    except OSError as e:
        _fail(f"could not read config file '{path}': {e}")
    except yaml.YAMLError as e:
        _fail(f"could not parse YAML config '{path}': {e}")

    if not isinstance(data, dict):
        _fail(f"config file must contain a YAML mapping, got {type(data).__name__}")

    return data


def _parse_bool(value: str) -> bool:
    """Parse a boolean from a string, supporting common representations."""
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    if value.lower() in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse '{value}' as boolean. Use true/false, yes/no, 1/0.")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands for each algorithm."""
    _define_params()

    parser = argparse.ArgumentParser(
        prog="thub",
        description="Training Hub CLI — run LLM training algorithms from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  thub sft --model-path ./model --data-path ./data.jsonl --ckpt-output-dir ./out\n"
            "  thub osft --config my_config.yaml\n"
            "  thub lora-grpo --config base.yaml --learning-rate 1e-3\n"
            "  thub grpo --config grpo.yaml --reward-fn my_rewards.compute_reward\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=_get_version(),
        help="Show training-hub version and exit",
    )

    subparsers = parser.add_subparsers(dest="algorithm", help="Training algorithm to run")

    descriptions = {
        "sft": "Supervised Fine-Tuning",
        "osft": "Orthogonal Subspace Fine-Tuning",
        "lora-sft": "LoRA + Supervised Fine-Tuning",
        "lora-grpo": "LoRA + Group Relative Policy Optimization",
        "grpo": "Full-parameter Group Relative Policy Optimization",
        "gepa": "Genetic-Pareto prompt optimization (gradient-free)",
    }

    for subcmd, param_defs in _ALGO_PARAM_DEFS.items():
        sub = subparsers.add_parser(
            subcmd,
            help=descriptions[subcmd],
            description=descriptions[subcmd],
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        sub.add_argument(
            "--config", "-c", type=str, metavar="FILE",
            help="YAML config file (CLI arguments override its values). Only use "
                 "configs you trust — callable params import arbitrary modules.",
        )
        sub.add_argument(
            "--traceback", action="store_true",
            help="Print the full Python traceback on training failure.",
        )
        sub.add_argument(
            "--version", action="version", version=_get_version(),
            help="Show training-hub version and exit",
        )

        for flag, spec in param_defs.items():
            kwargs: dict[str, Any] = {"help": spec.get("help", "")}

            if spec.get("callable") or spec.get("json"):
                kwargs["type"] = str
            elif spec["type"] is bool:
                kwargs["type"] = _parse_bool
                kwargs["metavar"] = "BOOL"
            else:
                kwargs["type"] = spec["type"]

            if "nargs" in spec:
                kwargs["nargs"] = spec["nargs"]

            # Set dest explicitly for every param (not just required ones) so the
            # kwarg name is unambiguous and independent of argparse's auto-derivation.
            kwargs["dest"] = _flag_to_python_name(flag)

            sub.add_argument(flag, **kwargs)

    return parser


def _coerce_value(value: Any, spec: dict) -> Any:
    """Coerce a value to the type expected by the backend.

    Handles both YAML config values (which ``yaml.safe_load`` may parse into
    native ints/floats/bools/lists/dicts) and already-typed CLI values. Raises
    ``ValueError`` with a clear message on type mismatches rather than silently
    passing the wrong type through to the backend.
    """
    if value is None:
        return None

    # Callable params: must be a dotted-path string that resolves to a callable.
    if spec.get("callable"):
        if not isinstance(value, str):
            raise ValueError("expected a dotted-path string (e.g. 'my_module.my_fn')")
        obj = _resolve_dotted_path(value)
        if not callable(obj):
            raise ValueError(f"'{value}' resolved to a non-callable {type(obj).__name__}")
        return obj

    # JSON params: parse a string; accept a YAML-native list/dict as-is.
    if spec.get("json"):
        if isinstance(value, str):
            return json.loads(value)
        return value

    if spec["type"] is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return _parse_bool(value)
        if isinstance(value, int):  # YAML 0/1 for a boolean flag
            if value in (0, 1):
                return bool(value)
            raise ValueError(f"expected a boolean, got {value!r}")
        raise ValueError(f"expected a boolean, got {type(value).__name__}")

    if "nargs" in spec:
        # Accept a single scalar as a one-element list (YAML: `x: a` == `x: [a]`).
        items = value if isinstance(value, list) else [value]
        return [spec["type"](v) for v in items]

    if spec["type"] is int:
        # bool is a subclass of int; reject it so `num_epochs: true` isn't read as 1.
        if isinstance(value, bool):
            raise ValueError(f"expected an integer, got boolean {value!r}")
        if isinstance(value, float):
            # Don't silently truncate `max_seq_len: 2048.5` to 2048.
            if not value.is_integer():
                raise ValueError(f"expected an integer, got {value!r}")
            return int(value)
        return value if isinstance(value, int) else int(value)

    if spec["type"] is float:
        # bool is a subclass of int/float-castable; reject `learning_rate: true` → 1.0.
        if isinstance(value, bool):
            raise ValueError(f"expected a number, got boolean {value!r}")
        result = value if isinstance(value, float) else float(value)
        if math.isnan(result) or math.isinf(result):
            raise ValueError(f"expected a finite number, got {value!r}")
        return result

    if spec["type"] is str and not isinstance(value, str):
        # YAML may parse an intended string as int/bool (e.g. `nproc_per_node: 8`).
        return str(value)

    if not isinstance(value, spec["type"]):
        return spec["type"](value)

    return value


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.algorithm:
        parser.print_help()
        sys.exit(1)

    subcmd = args.algorithm
    param_defs = _ALGO_PARAM_DEFS[subcmd]

    # Step 1: Load config file if provided
    config_values: dict[str, Any] = {}
    if args.config:
        raw_config = _load_yaml_config(args.config)
        # Config keys use python_style (underscores), map them
        for key, value in raw_config.items():
            python_key = key.replace("-", "_")
            # Find the matching spec
            cli_flag = "--" + key.replace("_", "-")
            spec = param_defs.get(cli_flag)
            if spec is None:
                # Pass through, but warn — this is usually a typo (e.g.
                # `lerning_rate`) that would otherwise fail obscurely at call time.
                print(
                    f"warning: config key '{key}' is not a recognized '{subcmd}' "
                    "parameter; passing it through unchanged",
                    file=sys.stderr,
                )
                config_values[python_key] = value
            else:
                try:
                    config_values[python_key] = _coerce_value(value, spec)
                except (ValueError, TypeError, ImportError, argparse.ArgumentTypeError) as e:
                    _fail(f"invalid value for '{key}' in config '{args.config}': {e}")

    # Step 2: Overlay CLI args (they take precedence over config). Coerce through
    # the same path as config values so callable/JSON/type validation is uniform.
    cli_dict = vars(args)
    for flag, spec in param_defs.items():
        python_name = _flag_to_python_name(flag)
        cli_value = cli_dict.get(python_name)

        if cli_value is not None:
            try:
                config_values[python_name] = _coerce_value(cli_value, spec)
            except (ValueError, TypeError, ImportError, argparse.ArgumentTypeError) as e:
                _fail(f"invalid value for '{flag}': {e}")

    # Step 2.5: Apply spec-defined defaults for anything still unset. Precedence is
    # CLI > config > spec default. (argparse defaults can't be used here — they'd
    # be indistinguishable from user input and would override config values.)
    for flag, spec in param_defs.items():
        if "default" in spec:
            python_name = _flag_to_python_name(flag)
            if config_values.get(python_name) is None:
                config_values[python_name] = spec["default"]

    # Step 3: Check for missing required params
    missing = []
    for flag, spec in param_defs.items():
        if spec.get("required"):
            python_name = _flag_to_python_name(flag)
            if python_name not in config_values or config_values[python_name] is None:
                missing.append(flag)

    if missing:
        print(f"error: the following arguments are required: {', '.join(missing)}", file=sys.stderr)
        print("hint: provide them via CLI flags or in a YAML config file with --config", file=sys.stderr)
        sys.exit(1)

    # Step 4: Remove internal keys
    for internal in ("config", "algorithm", "version", "traceback"):
        config_values.pop(internal, None)

    # Step 5: Import only the selected algorithm's backend and call it
    func_name = _SUBCOMMAND_TO_FUNC[subcmd]
    module_name, attr_name = _FUNC_IMPORTS[func_name]
    try:
        module = importlib.import_module(module_name)
        func: Callable = getattr(module, attr_name)
    except (ImportError, AttributeError) as e:
        _fail(f"could not import the backend for '{subcmd}' ({module_name}): {e}")

    # Remove None values so the function uses its own defaults
    final_kwargs = {k: v for k, v in config_values.items() if v is not None}

    try:
        result = func(**final_kwargs)
    except KeyboardInterrupt:
        print("\nTraining interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        if getattr(args, "traceback", False):
            import traceback
            traceback.print_exc()
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
