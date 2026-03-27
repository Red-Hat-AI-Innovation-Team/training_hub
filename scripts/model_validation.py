"""
Model Validation Script for Training Hub

This script validates that various model architectures can be trained successfully
using SFT, OSFT, and LoRA algorithms. The goal is to overfit on a single sample
(replicated 1000 times) to achieve NLL approaching 0.

Usage:
    python model_validation.py --models llama --mode sft
    python model_validation.py --models llama --mode osft --liger off
    python model_validation.py --models llama --mode lora
    python model_validation.py --models llama --mode lora --qlora on
    python model_validation.py --run-all --mode all --liger both --qlora both
    python model_validation.py --run-all

Configuration:
    - GPUs: 8
    - Effective batch size: 32
    - Learning rate: 1e-5
    - Tokens per GPU: 8192 (8k)
    - Epochs: 1
    - OSFT unfreeze rank ratio: 0.5
"""

import argparse
import glob
import json
import os
import shutil
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal



def _get_free_port() -> int:
    """Get a free TCP port by binding to port 0 and immediately releasing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Rich console for formatted output
console = Console()


@contextmanager
def trust_remote_code_context(enabled: bool):
    """
    Context manager to temporarily enable/disable trust_remote_code via environment variable.

    When enabled=True, sets HF_HUB_TRUST_REMOTE_CODE=1 which tells Hugging Face
    to trust remote code without requiring the explicit parameter.
    """
    env_var = "HF_HUB_TRUST_REMOTE_CODE"
    old_value = os.environ.get(env_var)

    if enabled:
        os.environ[env_var] = "1"
        console.print(f"[dim]🔓 Enabling trust_remote_code via {env_var}[/dim]")
    elif env_var in os.environ:
        del os.environ[env_var]

    try:
        yield
    finally:
        # Restore original state
        if old_value is not None:
            os.environ[env_var] = old_value
        elif env_var in os.environ:
            del os.environ[env_var]

# ============================================================================
# CONFIGURATION - Edit these settings as needed
# ============================================================================

# Base paths - MODIFY THESE
BASE_OUTPUT_DIR = "./tmp-outputs"  # TODO: Set your output directory
DATASET_OUTPUT_DIR = "./tmp-datasets"  # TODO: Set your dataset directory

# Training parameters
NUM_GPUS = 8
EFFECTIVE_BATCH_SIZE = 32
LEARNING_RATE = 1e-5
MAX_TOKENS_PER_GPU = 8192  # 8k tokens
NUM_EPOCHS = 1
OSFT_UNFREEZE_RANK_RATIO = 0.5
MAX_SEQ_LEN = 4096  # Should be enough for our single sample

# Dataset size (copies of single sample)
NUM_SAMPLES = 1000

# Convergence threshold - loss below this indicates successful overfitting (for SFT/OSFT)
CONVERGENCE_THRESHOLD = 0.1

# Simple mode parameters - tiny random models, smoke test only
SIMPLE_NUM_SAMPLES = 10
SIMPLE_EFFECTIVE_BATCH_SIZE = 4
SIMPLE_MAX_TOKENS_PER_GPU = 512
SIMPLE_MAX_SEQ_LEN = 256
SIMPLE_NUM_GPUS = 2

# ============================================================================
# MODEL REGISTRY
# ============================================================================


@dataclass
class ModelConfig:
    """Configuration for a model to validate."""

    model_id: str
    architecture: str
    notes: str = ""
    # Override defaults if needed for specific models
    max_tokens_per_gpu: int | None = None
    max_seq_len: int | None = None
    # Flags for expected behavior
    is_vision_model: bool = False  # Vision models expected to fail with text-only training
    requires_trust_remote_code: bool = False  # Models with custom code
    requires_dev_transformers: bool = False  # Requires transformers from main branch


# Models to validate - organized by architecture class
MODELS = {
    # GptOssForCausalLM
    "gpt-oss": ModelConfig(
        model_id="openai/gpt-oss-20b",
        architecture="GptOssForCausalLM",
        notes="OpenAI GPT-OSS 20B",
    ),
    # NemotronHForCausalLM
    "nemotron": ModelConfig(
        model_id="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        architecture="NemotronHForCausalLM",
        notes="NVIDIA Nemotron Nano 9B v2",
        requires_trust_remote_code=True,
    ),
    # Qwen3ForCausalLM
    "qwen3": ModelConfig(
        model_id="qwen/Qwen3-4B-Instruct-2507",
        architecture="Qwen3ForCausalLM",
        notes="Qwen3 4B Instruct (reasonably sized)",
    ),
    # Qwen2ForCausalLM
    "qwen2": ModelConfig(
        model_id="qwen/Qwen2.5-1.5B-Instruct",
        architecture="Qwen2ForCausalLM",
        notes="Qwen2.5 1.5B Instruct",
    ),
    # LlamaForCausalLM
    "llama": ModelConfig(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        architecture="LlamaForCausalLM",
        notes="Llama 3.2 1B Instruct",
    ),
    # GraniteForCausalLM (classic)
    "granite": ModelConfig(
        model_id="ibm-granite/granite-3.1-8b-instruct",
        architecture="GraniteForCausalLM",
        notes="Granite 3.1 8B Instruct (classic)",
    ),
    # GraniteMoeHybridForCausalLM
    "granite-moe": ModelConfig(
        model_id="ibm-granite/granite-4.0-h-tiny",
        architecture="GraniteMoeHybridForCausalLM",
        notes="Granite 4.0 Hybrid Tiny (MoE) - LoRA requires causal_conv1d_cuda (Unsloth limitation)",
    ),
    # Phi3ForCausalLM
    "phi4": ModelConfig(
        model_id="microsoft/Phi-4-mini-instruct",
        architecture="Phi3ForCausalLM",
        notes="Phi-4 Mini Instruct",
    ),
    # Gemma3ForConditionalGeneration
    "gemma3": ModelConfig(
        model_id="google/gemma-3-4b-it",
        architecture="Gemma3ForConditionalGeneration",
        notes="Gemma 3 4B IT (VLM, text-only LoRA training)",
    ),
    # Gemma3nForConditionalGeneration
    "gemma3n": ModelConfig(
        model_id="google/gemma-3n-E4B-it",
        architecture="Gemma3nForConditionalGeneration",
        notes="Gemma 3n E4B IT (VLM, loaded as CausalLM for text-only training)",
    ),
    # MistralForCausalLM
    "mistral": ModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        architecture="MistralForCausalLM",
        notes="Mistral 7B Instruct v0.3",
    ),
    # Ministral3ForCausalLM (extracted from Mistral3ForConditionalGeneration VLM wrapper)
    "ministral": ModelConfig(
        model_id="mistralai/Ministral-3-3B-Instruct-2512",
        architecture="Ministral3ForCausalLM",
        notes="Ministral 3 3B Instruct (VLM wrapper, CausalLM text backbone extracted)",
    ),
    # Mistral3ForConditionalGeneration (full VLM architecture)
    "mistral3-vlm": ModelConfig(
        model_id="mistralai/Ministral-3-3B-Reasoning-2512",
        architecture="Mistral3ForConditionalGeneration",
        notes="Ministral 3 3B Reasoning (full Mistral3 VLM architecture, text-only training)",
    ),
    # Qwen3VLForConditionalGeneration
    "qwen3-vl": ModelConfig(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        architecture="Qwen3VLForConditionalGeneration",
        notes="Qwen3 VL 2B Instruct (VLM loaded directly for text-only training)",
    ),
    # Qwen3_5ForCausalLM (Gated DeltaNet + MoE hybrid)
    "qwen3.5": ModelConfig(
        model_id="Qwen/Qwen3.5-4B",
        architecture="Qwen3_5ForCausalLM",
        notes="Qwen3.5 4B (Gated DeltaNet hybrid, multi-modal origin but loaded as CausalLM)",
    ),
}

# ============================================================================
# SAMPLE DATA FOR OVERFITTING
# ============================================================================

# Single sample to replicate - designed to be simple enough to overfit on
OVERFIT_SAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant that provides accurate information."},
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": "The capital of France is Paris. Paris is located in northern France on the Seine River and is known for iconic landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        },
    ]
}


# ============================================================================
# LOSS EXTRACTION
# ============================================================================


def get_final_sft_loss(output_dir: str) -> float | None:
    """
    Get final loss from SFT training metrics.

    SFT (instructlab-training) logs to: training_params_and_metrics_global{rank}.jsonl
    Loss field: "avg_loss"

    Args:
        output_dir: The checkpoint output directory used for training

    Returns:
        Final average loss value, or None if not found
    """
    metrics_file = os.path.join(output_dir, "training_params_and_metrics_global0.jsonl")
    if not os.path.exists(metrics_file):
        console.print(f"[dim]Loss file not found: {metrics_file}[/dim]")
        return None

    last_entry = None
    with open(metrics_file) as f:
        for line in f:
            if line.strip():
                try:
                    last_entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

    if last_entry is None:
        return None

    return last_entry.get("avg_loss")


def get_final_osft_loss(output_dir: str) -> float | None:
    """
    Get final loss from OSFT training metrics.

    OSFT (mini-trainer) logs to: training_metrics_{node_rank}.jsonl
    Loss field: "loss"

    Args:
        output_dir: The checkpoint output directory used for training

    Returns:
        Final loss value, or None if not found
    """
    metrics_file = os.path.join(output_dir, "training_metrics_0.jsonl")
    if not os.path.exists(metrics_file):
        console.print(f"[dim]Loss file not found: {metrics_file}[/dim]")
        return None

    last_entry = None
    with open(metrics_file) as f:
        for line in f:
            if line.strip():
                try:
                    last_entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

    if last_entry is None:
        return None

    return last_entry.get("loss")


def get_lora_loss_trajectory(output_dir: str) -> tuple[float | None, float | None]:
    """
    Get initial and final loss from LoRA training metrics.

    LoRA (Unsloth/TRL) logs to: checkpoint-*/trainer_state.json
    Loss field: log_history[]["loss"]

    Args:
        output_dir: The checkpoint output directory used for training

    Returns:
        Tuple of (initial_loss, final_loss), either may be None if not found
    """
    import glob as glob_module

    # Find checkpoint directories
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob_module.glob(checkpoint_pattern)

    if not checkpoint_dirs:
        console.print(f"[dim]No checkpoint directories found in: {output_dir}[/dim]")
        return None, None

    # Sort by checkpoint number to get the most recent
    def get_checkpoint_num(path):
        try:
            return int(os.path.basename(path).split("-")[1])
        except (IndexError, ValueError):
            return 0

    checkpoint_dirs.sort(key=get_checkpoint_num, reverse=True)
    latest_checkpoint = checkpoint_dirs[0]

    # Read trainer_state.json
    trainer_state_file = os.path.join(latest_checkpoint, "trainer_state.json")
    if not os.path.exists(trainer_state_file):
        console.print(f"[dim]trainer_state.json not found in: {latest_checkpoint}[/dim]")
        return None, None

    try:
        with open(trainer_state_file) as f:
            trainer_state = json.load(f)
    except json.JSONDecodeError:
        console.print("[dim]Failed to parse trainer_state.json[/dim]")
        return None, None

    # Extract losses from log_history
    log_history = trainer_state.get("log_history", [])
    if not log_history:
        return None, None

    # Find first and last entries with loss values
    initial_loss = None
    final_loss = None

    for entry in log_history:
        if "loss" in entry:
            initial_loss = entry["loss"]
            break

    for entry in reversed(log_history):
        if "loss" in entry:
            final_loss = entry["loss"]
            break

    return initial_loss, final_loss


def get_final_lora_loss(output_dir: str) -> float | None:
    """
    Get final loss from LoRA training metrics.

    Args:
        output_dir: The checkpoint output directory used for training

    Returns:
        Final loss value, or None if not found
    """
    _, final_loss = get_lora_loss_trajectory(output_dir)
    return final_loss


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_models_table():
    """Print available models in a rich table."""
    table = Table(
        title="Available Models",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Key", style="green", no_wrap=True)
    table.add_column("Architecture", style="yellow", overflow="fold")
    table.add_column("Model ID", style="blue", overflow="fold")
    table.add_column("Flags", style="magenta")

    for key, config in MODELS.items():
        flags = []
        if config.is_vision_model:
            flags.append("[yellow]vision[/yellow]")
        if config.requires_trust_remote_code:
            flags.append("[cyan]remote-code[/cyan]")
        if config.requires_dev_transformers:
            flags.append("[red]dev-xformers[/red]")
        flags_str = ", ".join(flags) if flags else "[dim]-[/dim]"

        table.add_row(key, config.architecture, config.model_id, flags_str)

    console.print(table)


def print_validation_summary(results: list[dict], results_file: str):
    """Print validation results in a rich table."""
    # Summary stats
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") in ["failed", "error"])
    expected_fail_count = sum(
        1 for r in results
        if r.get("status") in ["failed", "error"]
        and (r.get("is_vision_model") or r.get("requires_dev_transformers"))
    )
    unexpected_fail_count = failed_count - expected_fail_count
    converged_count = sum(1 for r in results if r.get("converged", False))

    # Create summary panel
    summary_text = Text()
    summary_text.append(f"Total: {len(results)}  ", style="bold")
    summary_text.append(f"Success: {success_count}  ", style="bold green")
    summary_text.append(f"Converged: {converged_count}  ", style="bold cyan")
    if unexpected_fail_count > 0:
        summary_text.append(f"Failed: {unexpected_fail_count}  ", style="bold red")
    if expected_fail_count > 0:
        summary_text.append(f"Expected Failures: {expected_fail_count}", style="bold yellow")

    console.print()
    console.print(Panel(summary_text, title="Validation Summary", border_style="blue"))

    # Results table
    table = Table(
        title="Validation Results",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Model", style="blue", no_wrap=True)
    table.add_column("Mode", style="cyan", no_wrap=True)
    table.add_column("Variant", style="magenta", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Final Loss", no_wrap=True)
    table.add_column("Duration", style="dim", no_wrap=True)
    table.add_column("Notes", overflow="fold")

    for r in results:
        model_id = r.get("model_id", r.get("model_key", "unknown"))
        # Shorten model_id for display
        model_short = model_id.split("/")[-1] if "/" in model_id else model_id

        mode = r.get("mode", "?").upper()

        # Show variant based on mode: Liger for SFT/OSFT, QLoRA for LoRA
        if r.get("mode") == "lora":
            variant = "QLoRA" if r.get("use_qlora") else "LoRA"
        else:
            variant = "Liger" if r.get("use_liger") else "No Liger"

        status = r.get("status", "unknown")
        if status == "success":
            status_text = Text("PASS", style="bold green")
        elif status == "skipped":
            status_text = Text("SKIP", style="bold yellow")
        elif status in ["failed", "error"]:
            if r.get("is_vision_model") or r.get("requires_dev_transformers"):
                status_text = Text("EXPECTED", style="bold yellow")
            else:
                status_text = Text("FAIL", style="bold red")
        else:
            status_text = Text(status, style="dim")

        duration = format_duration(r.get("duration_seconds", 0))

        # Format final loss with convergence indicator
        final_loss = r.get("final_loss")
        if final_loss is not None:
            converged = r.get("converged", False)
            if converged:
                loss_text = Text(f"{final_loss:.4f}", style="bold green")
            else:
                loss_text = Text(f"{final_loss:.4f}", style="yellow")
        else:
            loss_text = Text("-", style="dim")

        # Notes/error
        notes = ""
        if r.get("is_vision_model"):
            notes = "Vision model"
        elif r.get("requires_dev_transformers"):
            notes = "Needs dev transformers"
        elif r.get("error"):
            error_str = str(r.get("error", ""))
            # Truncate long errors
            if len(error_str) > 47:
                notes = error_str[:47] + "..."
            else:
                notes = error_str

        table.add_row(model_short, mode, variant, status_text, loss_text, duration, notes)

    console.print(table)

    # Print file location
    console.print(f"\n[dim]Full results saved to:[/dim] [cyan]{results_file}[/cyan]")


def print_test_header(
    model_config: "ModelConfig",
    mode: str,
    use_liger: bool,
    use_qlora: bool,
    output_dir: str,
):
    """Print test header with rich formatting."""
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Field", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Architecture", model_config.architecture)
    table.add_row("Model", model_config.model_id)
    table.add_row("Mode", mode.upper())
    if mode == "lora":
        table.add_row("QLoRA", "Enabled (4-bit)" if use_qlora else "Disabled")
    else:
        table.add_row("Liger", "Enabled" if use_liger else "Disabled")
    table.add_row("Output", output_dir)

    title = f"Validation: {model_config.architecture}"
    if model_config.is_vision_model:
        title += " [yellow](vision model - expected to fail)[/yellow]"

    console.print(Panel(table, title=title, border_style="blue"))


def create_overfit_dataset(output_path: str, num_samples: int = NUM_SAMPLES) -> str:
    """
    Create a dataset with multiple copies of a single sample for overfitting validation.

    Args:
        output_path: Directory to save the dataset
        num_samples: Number of copies of the sample to create

    Returns:
        Path to the created dataset file
    """
    os.makedirs(output_path, exist_ok=True)
    dataset_file = os.path.join(output_path, "overfit_dataset.jsonl")

    with open(dataset_file, "w") as f:
        for _ in range(num_samples):
            f.write(json.dumps(OVERFIT_SAMPLE) + "\n")

    print(f"Created overfit dataset with {num_samples} samples at: {dataset_file}")
    return dataset_file


def create_tiny_model(model_config: ModelConfig, output_dir: str) -> str:
    """
    Create a tiny randomly-initialized version of a model architecture for smoke testing.

    Loads the real config from HuggingFace, overrides dimensions to be minimal
    (2 layers, 64 hidden, 2 heads), creates the model from config (random weights),
    and saves it locally with the tokenizer.

    Args:
        model_config: Model configuration from the MODELS registry
        output_dir: Base directory to save tiny models

    Returns:
        Path to the saved tiny model directory
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    model_key = model_config.model_id.split("/")[-1]
    tiny_dir = os.path.join(output_dir, f"tiny_{model_key}")

    ready_marker = os.path.join(tiny_dir, ".ready")

    # Return cached version if fully created
    if os.path.exists(ready_marker):
        console.print(f"[dim]Using cached tiny model: {tiny_dir}[/dim]")
        return tiny_dir

    console.print(f"[dim]Creating tiny model for {model_config.model_id}...[/dim]")

    trust_remote_code = model_config.requires_trust_remote_code
    config = AutoConfig.from_pretrained(
        model_config.model_id, trust_remote_code=trust_remote_code
    )

    # Determine which config to shrink (top-level or text_config for VLMs)
    text_cfg = config
    if hasattr(config, "text_config"):
        text_cfg = config.text_config

    # Shrink text model dimensions
    text_cfg.num_hidden_layers = 2
    text_cfg.hidden_size = 64
    # intermediate_size can be a list (per-layer) in some models like Gemma3n
    if isinstance(getattr(text_cfg, "intermediate_size", None), list):
        text_cfg.intermediate_size = [128] * text_cfg.num_hidden_layers
    else:
        text_cfg.intermediate_size = 128
    text_cfg.num_attention_heads = 2
    text_cfg.num_key_value_heads = 2
    if hasattr(text_cfg, "head_dim"):
        text_cfg.head_dim = 32

    # Handle MoE models
    if hasattr(text_cfg, "num_local_experts"):
        text_cfg.num_local_experts = 2
    if hasattr(text_cfg, "num_experts_per_tok"):
        text_cfg.num_experts_per_tok = min(text_cfg.num_experts_per_tok, 2)

    # Handle Mamba/SSM hybrid models — ensure dimension divisibility
    if hasattr(text_cfg, "mamba_n_heads"):
        # mamba_n_heads must divide mamba_expand * hidden_size
        mamba_expand = getattr(text_cfg, "mamba_expand", 2)
        # Set hidden_size to be divisible by mamba_n_heads
        text_cfg.mamba_n_heads = 2
        text_cfg.hidden_size = max(64, text_cfg.mamba_n_heads * mamba_expand * 16)
        text_cfg.intermediate_size = text_cfg.hidden_size * 2
    if hasattr(text_cfg, "mamba_d_state"):
        text_cfg.mamba_d_state = min(text_cfg.mamba_d_state, 16)

    # Disable KV sharing for tiny models (avoids index errors with truncated layer lists)
    if hasattr(text_cfg, "num_kv_shared_layers"):
        text_cfg.num_kv_shared_layers = 0

    # Truncate rope_scaling factors to match new head_dim
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)
    rope_scaling = getattr(text_cfg, "rope_scaling", None)
    if rope_scaling and isinstance(rope_scaling, dict):
        partial = rope_scaling.get("partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * partial) // 2
        for key in ("long_factor", "short_factor"):
            if key in rope_scaling and isinstance(rope_scaling[key], list):
                if len(rope_scaling[key]) != rotary_dim:
                    rope_scaling[key] = [1.0] * rotary_dim

    # Truncate ALL per-layer config lists to match num_hidden_layers.
    # Some models (Gemma3n, Qwen3.5) have multiple list-valued config fields
    # sized per layer (layer_types, activation_sparsity_pattern, etc.).
    n_layers = text_cfg.num_hidden_layers
    for attr in dir(text_cfg):
        if attr.startswith("_"):
            continue
        val = getattr(text_cfg, attr, None)
        if isinstance(val, list) and len(val) == getattr(text_cfg, "_original_num_layers", 999):
            # This looks like a per-layer list that needs truncating
            pass  # handled below
    # Explicit list of known per-layer attributes
    per_layer_attrs = [
        "layer_types", "sliding_window_pattern", "activation_sparsity_pattern",
    ]
    # Also detect any list attribute whose length matches the ORIGINAL num_hidden_layers
    original_n_layers = len(getattr(text_cfg, "layer_types", []) or [])
    if original_n_layers == 0:
        original_n_layers = 999  # no layer_types, skip auto-detection
    for attr in list(vars(text_cfg).keys()):
        val = getattr(text_cfg, attr, None)
        if isinstance(val, list) and len(val) == original_n_layers and attr not in per_layer_attrs:
            per_layer_attrs.append(attr)

    for attr in per_layer_attrs:
        if hasattr(text_cfg, attr):
            val = getattr(text_cfg, attr)
            if isinstance(val, (list, tuple)) and len(val) > n_layers:
                unique_types = list(dict.fromkeys(val))  # preserve order, dedupe
                if attr == "layer_types" and len(unique_types) > 1:
                    # Ensure all unique layer types are represented
                    text_cfg.num_hidden_layers = max(n_layers, len(unique_types))
                    n_layers = text_cfg.num_hidden_layers
                    # Also re-truncate intermediate_size if it's a list
                    if isinstance(text_cfg.intermediate_size, list):
                        text_cfg.intermediate_size = [128] * n_layers
                    new_val = (unique_types * ((n_layers // len(unique_types)) + 1))[:n_layers]
                    setattr(text_cfg, attr, new_val)
                else:
                    setattr(text_cfg, attr, val[:n_layers])

    # Handle hybrid_override_pattern (NemotronH) — truncate to match num_hidden_layers
    # The remote config needs this for the layers_block_type property
    hop = getattr(text_cfg, "hybrid_override_pattern", None)
    if hop and isinstance(hop, str) and len(hop) != n_layers:
        text_cfg.hybrid_override_pattern = hop[:n_layers]

    # Handle vision config for VLMs - shrink or disable
    if hasattr(config, "vision_config") and config.vision_config is not None:
        vc = config.vision_config
        if hasattr(vc, "num_hidden_layers"):
            vc.num_hidden_layers = 1
        if hasattr(vc, "hidden_size"):
            vc.hidden_size = 32
        if hasattr(vc, "intermediate_size"):
            vc.intermediate_size = 64
        if hasattr(vc, "num_attention_heads"):
            vc.num_attention_heads = 1

    # Handle audio config for multimodal models
    if hasattr(config, "audio_config") and config.audio_config is not None:
        ac = config.audio_config
        if hasattr(ac, "num_hidden_layers"):
            ac.num_hidden_layers = 1
        if hasattr(ac, "hidden_size"):
            ac.hidden_size = 32

    # Create model from config (random weights, no download)
    # Try multiple strategies: with auto_map, without it, and VLM fallback
    import copy
    model = None
    for attempt in range(3):
        try:
            if attempt == 0:
                # First try: as-is (works for remote code models like Nemotron)
                model = AutoModelForCausalLM.from_config(
                    config, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
                )
            elif attempt == 1:
                # Second try: remove auto_map (works for Phi4 whose remote code
                # conflicts with the standard transformers class)
                config_copy = copy.deepcopy(config)
                if hasattr(config_copy, "auto_map"):
                    del config_copy.auto_map
                tc = getattr(config_copy, "text_config", None)
                if tc and hasattr(tc, "auto_map"):
                    del tc.auto_map
                model = AutoModelForCausalLM.from_config(
                    config_copy, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
                )
            else:
                # Third try: VLM loader
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_config(
                    config, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
                )
            break
        except Exception:
            if attempt == 2:
                raise
            continue

    # Save model — some custom models have tied weights bugs, work around them
    os.makedirs(tiny_dir, exist_ok=True)
    try:
        model.save_pretrained(tiny_dir)
    except (AttributeError, TypeError):
        # Fallback: save state dict directly + config
        import safetensors.torch
        safetensors.torch.save_file(model.state_dict(), os.path.join(tiny_dir, "model.safetensors"))
        model.config.save_pretrained(tiny_dir)
    del model

    # Save tokenizer (download from original)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(tiny_dir)

    # Note: We intentionally do NOT copy remote code .py files to tiny models.
    # Tiny models should use built-in transformers classes for simplicity.

    # Clean up the saved config.json - remove auto_map and hybrid_override_pattern
    # to avoid issues when loading with built-in transformers classes
    config_path = os.path.join(tiny_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)

        # Remove auto_map and hybrid_override_pattern so tiny models use
        # built-in transformers classes without issues
        saved_config.pop("auto_map", None)
        saved_config.pop("hybrid_override_pattern", None)

        # Remove quantization_config (e.g., FP8 in Ministral-3) to avoid issues
        # on unsupported hardware (FP8 requires L4+ GPUs, not A100)
        saved_config.pop("quantization_config", None)

        if "text_config" in saved_config and isinstance(saved_config["text_config"], dict):
            saved_config["text_config"].pop("auto_map", None)
            saved_config["text_config"].pop("hybrid_override_pattern", None)
            saved_config["text_config"].pop("quantization_config", None)

        with open(config_path, 'w') as f:
            json.dump(saved_config, f, indent=2)

    # Mark as fully created so partial writes aren't treated as cache hits
    open(ready_marker, "w").close()

    console.print(f"[dim]Created tiny model at: {tiny_dir}[/dim]")
    return tiny_dir


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def run_sft_validation(
    model_config: ModelConfig,
    data_path: str,
    output_dir: str,
    use_liger: bool = False,
    nproc_per_node: int = NUM_GPUS,
    simple: bool = False,
) -> dict:
    """
    Run SFT validation for a model.

    Args:
        model_config: Model configuration
        data_path: Path to training data
        output_dir: Directory for checkpoints and outputs
        use_liger: Whether to enable Liger kernels

    Returns:
        Dictionary with validation results
    """
    from training_hub import sft

    # Use simple mode overrides if applicable
    if simple:
        max_tokens = SIMPLE_MAX_TOKENS_PER_GPU
        max_seq = SIMPLE_MAX_SEQ_LEN
        batch_size = SIMPLE_EFFECTIVE_BATCH_SIZE
        nproc_per_node = SIMPLE_NUM_GPUS
    else:
        max_tokens = model_config.max_tokens_per_gpu or MAX_TOKENS_PER_GPU
        max_seq = model_config.max_seq_len or MAX_SEQ_LEN
        batch_size = EFFECTIVE_BATCH_SIZE

    start_time = time.time()
    result = {
        "model_id": model_config.model_id,
        "architecture": model_config.architecture,
        "mode": "sft",
        "use_liger": use_liger,
        "simple": simple,
        "status": "unknown",
        "error": None,
        "duration_seconds": 0,
        "is_vision_model": model_config.is_vision_model,
        "requires_dev_transformers": model_config.requires_dev_transformers,
    }

    try:
        with trust_remote_code_context(model_config.requires_trust_remote_code):
            sft(
                model_path=model_config.model_id,
                data_path=data_path,
                ckpt_output_dir=output_dir,
                # Training parameters
                num_epochs=NUM_EPOCHS,
                effective_batch_size=batch_size,
                learning_rate=LEARNING_RATE,
                max_seq_len=max_seq,
                max_tokens_per_gpu=max_tokens,
                # Data processing
                data_output_dir=os.path.join(output_dir, "_data_processing"),
                warmup_steps=0,
                save_samples=0,  # Disable sample-based checkpointing
                # Checkpointing
                checkpoint_at_epoch=False,
                accelerate_full_state_at_epoch=False,
                # Multi-GPU setup
                nproc_per_node=nproc_per_node,
                nnodes=1,
                node_rank=0,
                rdzv_id=f"validation-sft-{int(time.time())}",
                rdzv_endpoint=f"127.0.0.1:{_get_free_port()}",
                # Optimization - passed through kwargs to TrainingArgs
                use_liger=use_liger,
            )

        result["status"] = "success"

        # Extract final loss
        final_loss = get_final_sft_loss(output_dir)
        result["final_loss"] = final_loss
        if final_loss is not None and not simple:
            result["converged"] = final_loss < CONVERGENCE_THRESHOLD

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    result["duration_seconds"] = time.time() - start_time
    return result


def run_osft_validation(
    model_config: ModelConfig,
    data_path: str,
    output_dir: str,
    use_liger: bool = True,
    nproc_per_node: int = NUM_GPUS,
    simple: bool = False,
) -> dict:
    """
    Run OSFT validation for a model.

    Args:
        model_config: Model configuration
        data_path: Path to training data
        output_dir: Directory for checkpoints and outputs
        use_liger: Whether to enable Liger kernels

    Returns:
        Dictionary with validation results
    """
    from training_hub import osft

    # Use simple mode overrides if applicable
    if simple:
        max_tokens = SIMPLE_MAX_TOKENS_PER_GPU
        max_seq = SIMPLE_MAX_SEQ_LEN
        batch_size = SIMPLE_EFFECTIVE_BATCH_SIZE
        nproc_per_node = SIMPLE_NUM_GPUS
    else:
        max_tokens = model_config.max_tokens_per_gpu or MAX_TOKENS_PER_GPU
        max_seq = model_config.max_seq_len or MAX_SEQ_LEN
        batch_size = EFFECTIVE_BATCH_SIZE

    start_time = time.time()
    result = {
        "model_id": model_config.model_id,
        "architecture": model_config.architecture,
        "mode": "osft",
        "use_liger": use_liger,
        "simple": simple,
        "status": "unknown",
        "error": None,
        "duration_seconds": 0,
        "is_vision_model": model_config.is_vision_model,
        "requires_dev_transformers": model_config.requires_dev_transformers,
    }

    try:
        with trust_remote_code_context(model_config.requires_trust_remote_code):
            osft(
                model_path=model_config.model_id,
                data_path=data_path,
                ckpt_output_dir=output_dir,
                # OSFT-specific parameters
                unfreeze_rank_ratio=OSFT_UNFREEZE_RANK_RATIO,
                # Training parameters
                num_epochs=NUM_EPOCHS,
                effective_batch_size=batch_size,
                learning_rate=LEARNING_RATE,
                max_seq_len=max_seq,
                max_tokens_per_gpu=max_tokens,
                # Data processing
                data_output_dir=os.path.join(output_dir, "_data_processing"),
                warmup_steps=0,
                # Optimization
                use_liger=use_liger,
                seed=42,
                lr_scheduler="cosine",
                # Checkpointing
                checkpoint_at_epoch=True,
                save_final_checkpoint=True,
                # Multi-GPU setup
                nproc_per_node=nproc_per_node,
                nnodes=1,
                node_rank=0,
                rdzv_id=f"validation-osft-{int(time.time())}",
                rdzv_endpoint=f"127.0.0.1:{_get_free_port()}",
            )

        result["status"] = "success"

        # Extract final loss
        final_loss = get_final_osft_loss(output_dir)
        result["final_loss"] = final_loss
        if final_loss is not None and not simple:
            result["converged"] = final_loss < CONVERGENCE_THRESHOLD

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    result["duration_seconds"] = time.time() - start_time
    return result


def run_lora_validation(
    model_config: ModelConfig,
    data_path: str,
    output_dir: str,
    use_qlora: bool = False,
    nproc_per_node: int = NUM_GPUS,
    simple: bool = False,
) -> dict:
    """
    Run LoRA validation for a model.

    Args:
        model_config: Model configuration
        data_path: Path to training data
        output_dir: Directory for checkpoints and outputs
        use_qlora: Whether to enable QLoRA (4-bit quantization)

    Returns:
        Dictionary with validation results
    """
    from training_hub import lora_sft

    # Use model-specific overrides or defaults
    max_seq = model_config.max_seq_len or MAX_SEQ_LEN

    start_time = time.time()
    result = {
        "model_id": model_config.model_id,
        "architecture": model_config.architecture,
        "mode": "lora",
        "use_qlora": use_qlora,
        "status": "unknown",
        "error": None,
        "duration_seconds": 0,
        "is_vision_model": model_config.is_vision_model,
        "requires_dev_transformers": model_config.requires_dev_transformers,
    }

    # Disable flex_attention only for models with non-zero attention_dropout.
    # Unsloth's compiled cache dispatches through transformers' flex_attention_forward
    # which rejects dropout > 0 during training (e.g. Granite 3.1 has dropout=0.1).
    # Models with dropout=0 benefit from flex_attention so we leave it enabled.
    _flex_env = "UNSLOTH_ENABLE_FLEX_ATTENTION"
    _old_flex = os.environ.get(_flex_env)
    try:
        from transformers import AutoConfig as _AutoConfig
        _cfg = _AutoConfig.from_pretrained(
            model_config.model_id,
            trust_remote_code=model_config.requires_trust_remote_code,
        )
        _dropout = getattr(_cfg, "attention_dropout", 0.0) or 0.0
        if _dropout > 0:
            os.environ[_flex_env] = "0"
            console.print(f"[dim]Disabling flex_attention (attention_dropout={_dropout})[/dim]")
    except (OSError, ValueError):
        pass

    try:
        with trust_remote_code_context(model_config.requires_trust_remote_code):
            lora_sft(
                model_path=model_config.model_id,
                data_path=data_path,
                ckpt_output_dir=output_dir,
                # QLoRA - enable 4-bit quantization
                load_in_4bit=use_qlora,
                # Training parameters
                num_epochs=NUM_EPOCHS,
                effective_batch_size=EFFECTIVE_BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                max_seq_len=max_seq,
                warmup_steps=0,
                # Model loading - pass trust_remote_code explicitly since the
                # HF_HUB_TRUST_REMOTE_CODE env var is not respected by
                # Unsloth's FastModel.from_pretrained
                trust_remote_code=model_config.requires_trust_remote_code,
                # Multi-GPU setup
                nproc_per_node=nproc_per_node,
                nnodes=1,
                node_rank=0,
                rdzv_id=f"validation-lora-{int(time.time())}",
                rdzv_endpoint=f"127.0.0.1:{_get_free_port()}",
            )

        result["status"] = "success"

        # Extract initial and final loss for LoRA
        # LoRA convergence = any decrease in loss (not absolute threshold)
        initial_loss, final_loss = get_lora_loss_trajectory(output_dir)
        result["initial_loss"] = initial_loss
        result["final_loss"] = final_loss
        if initial_loss is not None and final_loss is not None:
            result["converged"] = final_loss < initial_loss

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
    finally:
        # Restore flex_attention env var so subsequent models aren't affected
        if _old_flex is not None:
            os.environ[_flex_env] = _old_flex
        elif _flex_env in os.environ:
            del os.environ[_flex_env]

    result["duration_seconds"] = time.time() - start_time
    return result


# ============================================================================
# VALIDATION ORCHESTRATION
# ============================================================================


def run_single_validation(
    model_key: str,
    mode: Literal["sft", "osft", "lora"],
    use_liger: bool = True,
    use_qlora: bool = False,
    base_output_dir: str = BASE_OUTPUT_DIR,
    dataset_dir: str = DATASET_OUTPUT_DIR,
    nproc_per_node: int = NUM_GPUS,
    simple: bool = False,
) -> dict:
    """
    Run a single validation test.

    Args:
        model_key: Key from MODELS dictionary
        mode: Training mode ("sft", "osft", or "lora")
        use_liger: Whether to use Liger kernels (applies to SFT and OSFT)
        use_qlora: Whether to use QLoRA (applies to LoRA mode)
        base_output_dir: Base directory for outputs
        dataset_dir: Directory for dataset

    Returns:
        Validation result dictionary
    """
    # Reset torch._dynamo compilation caches between model runs to prevent
    # stale shape guards from prior models contaminating the current model's
    # @torch.compile-decorated functions (e.g. Ministral3's _get_llama_4_attn_scale
    # failing with "too many indices" after Mistral 7B populates the cache).
    import torch._dynamo
    torch._dynamo.reset()

    # Free memory from the previous model — gc first to release Python
    # references to CUDA tensors, then empty_cache to reclaim GPU memory
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODELS.keys())}")

    model_config = MODELS[model_key]

    # Create output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode == "lora":
        variant_suffix = "_qlora" if use_qlora else "_lora"
    else:
        variant_suffix = "_liger" if use_liger else "_noliger"
    run_name = f"{model_key}_{mode}{variant_suffix}_{timestamp}"
    output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create or use existing dataset
    num_samples = SIMPLE_NUM_SAMPLES if simple else NUM_SAMPLES
    data_path = create_overfit_dataset(dataset_dir, num_samples=num_samples)

    # In simple mode, create a tiny model and override model_id
    if simple:
        tiny_models_dir = os.path.join(base_output_dir, "_tiny_models")
        try:
            tiny_model_path = create_tiny_model(model_config, tiny_models_dir)
        except Exception as e:
            console.print(f"[yellow]Skipping {model_key} in simple mode: could not create tiny model ({e})[/yellow]")
            return {
                "model_id": model_config.model_id,
                "architecture": model_config.architecture,
                "mode": mode,
                "simple": True,
                "status": "skipped",
                "error": f"Cannot create tiny model: {e}",
                "duration_seconds": 0,
            }
        # Override model_id for training but keep original for reporting
        # Models requiring trust_remote_code need it preserved for the local copy
        model_config_for_training = ModelConfig(
            model_id=tiny_model_path,
            architecture=model_config.architecture,
            notes=model_config.notes,
            max_tokens_per_gpu=SIMPLE_MAX_TOKENS_PER_GPU,
            max_seq_len=SIMPLE_MAX_SEQ_LEN,
            is_vision_model=model_config.is_vision_model,
            requires_trust_remote_code=model_config.requires_trust_remote_code,
            requires_dev_transformers=False,
        )
    else:
        model_config_for_training = model_config

    # Print header
    print_test_header(model_config, mode, use_liger, use_qlora, output_dir)

    if mode == "sft":
        result = run_sft_validation(model_config_for_training, data_path, output_dir, use_liger, nproc_per_node=nproc_per_node, simple=simple)
    elif mode == "osft":
        result = run_osft_validation(model_config_for_training, data_path, output_dir, use_liger, nproc_per_node=nproc_per_node, simple=simple)
    else:  # lora
        result = run_lora_validation(model_config_for_training, data_path, output_dir, use_qlora, nproc_per_node=nproc_per_node, simple=simple)

    # Restore original model_id for reporting
    result["model_id"] = model_config.model_id

    # Print result summary
    if result["status"] == "success":
        duration_str = format_duration(result['duration_seconds'])
        final_loss = result.get("final_loss")
        if final_loss is not None:
            converged = result.get("converged", False)
            loss_style = "bold green" if converged else "yellow"
            converge_status = "converged" if converged else "not converged"
            console.print(
                f"[bold green]SUCCESS[/bold green] - "
                f"Final loss: [{loss_style}]{final_loss:.4f}[/{loss_style}] ({converge_status}) - "
                f"Completed in {duration_str}"
            )
        else:
            console.print(f"[bold green]SUCCESS[/bold green] - Completed in {duration_str}")
    else:
        console.print(f"[bold red]FAILED[/bold red] - {result['error']}")

    # Save result to file
    result_file = os.path.join(output_dir, "validation_result.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_all_validations(
    mode: Literal["sft", "osft", "lora", "all"] = "all",
    liger_modes: list[bool] | None = None,
    qlora_modes: list[bool] | None = None,
    base_output_dir: str = BASE_OUTPUT_DIR,
    dataset_dir: str = DATASET_OUTPUT_DIR,
    model_keys: list[str] | None = None,
    nproc_per_node: int = NUM_GPUS,
    simple: bool = False,
) -> list[dict]:
    """
    Run validation tests for all models.

    Args:
        mode: Training mode(s) to test ("sft", "osft", "lora", or "all")
        liger_modes: Liger configurations to test (applies to SFT and OSFT)
        qlora_modes: QLoRA configurations to test (applies to LoRA)
        base_output_dir: Base directory for outputs
        dataset_dir: Directory for dataset
        model_keys: Optional list of specific model keys to test

    Returns:
        List of validation results
    """
    if liger_modes is None:
        liger_modes = [True, False]
    if qlora_modes is None:
        qlora_modes = [True, False]
    results = []
    models_to_test = model_keys or list(MODELS.keys())
    modes_to_test = ["sft", "osft", "lora"] if mode == "all" else [mode]

    # Calculate total tests: SFT/OSFT use liger_modes, LoRA uses qlora_modes
    total_tests = 0
    for test_mode in modes_to_test:
        if test_mode == "lora":
            total_tests += len(models_to_test) * len(qlora_modes)
        else:
            total_tests += len(models_to_test) * len(liger_modes)

    # Print run configuration
    config_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    config_table.add_column("", style="dim")
    config_table.add_column("", style="bold")
    config_table.add_row("Total Tests", str(total_tests))
    config_table.add_row("Models", str(len(models_to_test)))
    config_table.add_row("Modes", ", ".join(modes_to_test))
    if "lora" in modes_to_test:
        config_table.add_row("QLoRA Variants", str(qlora_modes))
    if any(m in modes_to_test for m in ["sft", "osft"]):
        config_table.add_row("Liger Variants", str(liger_modes))

    console.print()
    console.print(Panel(config_table, title="Validation Run Configuration", border_style="cyan"))
    console.print()

    test_num = 0
    for model_key in models_to_test:
        for test_mode in modes_to_test:
            # Choose variants based on mode
            if test_mode == "lora":
                variants = [(False, use_qlora) for use_qlora in qlora_modes]
            else:
                variants = [(use_liger, False) for use_liger in liger_modes]

            for use_liger, use_qlora in variants:
                test_num += 1
                if test_mode == "lora":
                    variant_str = "qlora" if use_qlora else "lora"
                else:
                    variant_str = "liger" if use_liger else "no-liger"
                console.print(f"\n[bold cyan][{test_num}/{total_tests}][/bold cyan] Testing [green]{model_key}[/green] - {test_mode} ({variant_str})")
                try:
                    result = run_single_validation(
                        model_key=model_key,
                        mode=test_mode,
                        use_liger=use_liger,
                        use_qlora=use_qlora,
                        base_output_dir=base_output_dir,
                        dataset_dir=dataset_dir,
                        nproc_per_node=nproc_per_node,
                        simple=simple,
                    )
                    results.append(result)
                except Exception as e:
                    console.print(f"  [bold red]ERROR:[/bold red] {e}")
                    results.append(
                        {
                            "model_key": model_key,
                            "mode": test_mode,
                            "use_liger": use_liger,
                            "use_qlora": use_qlora,
                            "status": "error",
                            "error": str(e),
                        }
                    )

    # Save full results
    results_file = os.path.join(base_output_dir, f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(base_output_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print_validation_summary(results, results_file)

    return results


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Model Validation Script for Training Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run validation for specific model(s)
    python model_validation.py --models llama --mode sft
    python model_validation.py --models llama granite --mode osft
    python model_validation.py --models llama --mode lora
    python model_validation.py --models llama --mode lora --qlora on

    # Run all models for a specific mode
    python model_validation.py --run-all --mode sft
    python model_validation.py --run-all --mode osft
    python model_validation.py --run-all --mode lora

    # Control Liger kernels (applies to SFT/OSFT)
    python model_validation.py --models llama --mode osft --liger off
    python model_validation.py --run-all --mode sft --liger both

    # Control QLoRA (applies to LoRA)
    python model_validation.py --models llama --mode lora --qlora on
    python model_validation.py --run-all --mode lora --qlora both

    # Run all combinations (all models, all modes, all variants)
    python model_validation.py --run-all --mode all --liger both --qlora both

    # List available models
    python model_validation.py --list-models

Available model keys:
    """
        + ", ".join(MODELS.keys()),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="Model(s) to validate (see --list-models for options)",
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "osft", "lora", "all"],
        default="sft",
        help="Training mode (default: sft)",
    )

    # Liger kernels (applies to SFT and OSFT)
    parser.add_argument(
        "--liger", choices=["on", "off", "both"], default="on",
        help="Liger kernel mode for SFT/OSFT: on, off, or both (default: on)"
    )

    # QLoRA (applies to LoRA)
    parser.add_argument(
        "--qlora", choices=["on", "off", "both"], default="off",
        help="QLoRA (4-bit quantization) for LoRA: on, off, or both (default: off)"
    )

    parser.add_argument("--run-all", action="store_true", help="Run validation for all models")
    parser.add_argument(
        "--output-dir", default=BASE_OUTPUT_DIR, help=f"Base output directory (default: {BASE_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--dataset-dir", default=DATASET_OUTPUT_DIR, help=f"Dataset directory (default: {DATASET_OUTPUT_DIR})"
    )
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    def _positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
        return parsed

    parser.add_argument(
        "--nproc-per-node", type=_positive_int, default=NUM_GPUS,
        help=f"Number of GPUs per node (default: {NUM_GPUS})"
    )
    parser.add_argument(
        "--simple", action="store_true",
        help="Smoke test mode: create tiny random models (2 layers, 64 hidden), "
             "train briefly on 2 GPUs, check only that training runs without errors"
    )

    args = parser.parse_args()

    if args.list_models:
        print_models_table()
        return

    # Map choice strings to mode lists
    choice_map = {"on": [True], "off": [False], "both": [True, False]}

    if args.run_all or args.models:
        # Determine which models to run
        model_keys = args.models if args.models else None  # None means all models

        liger_modes = choice_map[args.liger]
        qlora_modes = choice_map[args.qlora]

        run_all_validations(
            mode=args.mode,
            liger_modes=liger_modes,
            qlora_modes=qlora_modes,
            base_output_dir=args.output_dir,
            dataset_dir=args.dataset_dir,
            model_keys=model_keys,
            nproc_per_node=args.nproc_per_node,
            simple=args.simple,
        )
    else:
        parser.print_help()
        print("\nError: Either --models or --run-all is required")
        sys.exit(1)


if __name__ == "__main__":
    main()
