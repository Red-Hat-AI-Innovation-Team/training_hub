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
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

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
        notes="Granite 4.0 Hybrid Tiny (MoE)",
    ),
    # Phi3ForCausalLM
    "phi4": ModelConfig(
        model_id="microsoft/Phi-4-mini-instruct",
        architecture="Phi3ForCausalLM",
        notes="Phi-4 Mini Instruct",
    ),
    # Gemma3nForConditionalGeneration
    "gemma3n": ModelConfig(
        model_id="google/gemma-3n-E4B-it",
        architecture="Gemma3nForConditionalGeneration",
        notes="Gemma 3n E4B IT (vision-language, expected to fail)",
        is_vision_model=True,
    ),
    # MistralForCausalLM
    "mistral": ModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        architecture="MistralForCausalLM",
        notes="Mistral 7B Instruct v0.3",
    ),
    # Mistral3ForConditionalGeneration
    "ministral": ModelConfig(
        model_id="mistralai/Ministral-3-3B-Instruct-2512",
        architecture="Mistral3ForConditionalGeneration",
        notes="Ministral 3 3B Instruct (requires transformers main branch)",
        requires_dev_transformers=True,
        requires_trust_remote_code=True,  # Model not natively supported in 4.57.6
    ),
    # Qwen3VLForConditionalGeneration
    "qwen3-vl": ModelConfig(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        architecture="Qwen3VLForConditionalGeneration",
        notes="Qwen3 VL 2B Instruct (vision-language, expected to fail)",
        is_vision_model=True,
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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def run_sft_validation(
    model_config: ModelConfig,
    data_path: str,
    output_dir: str,
    use_liger: bool = False,
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

    # Use model-specific overrides or defaults
    max_tokens = model_config.max_tokens_per_gpu or MAX_TOKENS_PER_GPU
    max_seq = model_config.max_seq_len or MAX_SEQ_LEN

    start_time = time.time()
    result = {
        "model_id": model_config.model_id,
        "architecture": model_config.architecture,
        "mode": "sft",
        "use_liger": use_liger,
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
                effective_batch_size=EFFECTIVE_BATCH_SIZE,
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
                nproc_per_node=NUM_GPUS,
                nnodes=1,
                node_rank=0,
                rdzv_id=f"validation-sft-{int(time.time())}",
                rdzv_endpoint="127.0.0.1:42067",
                # Optimization - passed through kwargs to TrainingArgs
                use_liger=use_liger,
            )

        result["status"] = "success"

        # Extract final loss
        final_loss = get_final_sft_loss(output_dir)
        result["final_loss"] = final_loss
        if final_loss is not None:
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

    # Use model-specific overrides or defaults
    max_tokens = model_config.max_tokens_per_gpu or MAX_TOKENS_PER_GPU
    max_seq = model_config.max_seq_len or MAX_SEQ_LEN

    start_time = time.time()
    result = {
        "model_id": model_config.model_id,
        "architecture": model_config.architecture,
        "mode": "osft",
        "use_liger": use_liger,
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
                effective_batch_size=EFFECTIVE_BATCH_SIZE,
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
                nproc_per_node=NUM_GPUS,
                nnodes=1,
                node_rank=0,
                rdzv_id=f"validation-osft-{int(time.time())}",
                rdzv_endpoint="127.0.0.1:29500",
            )

        result["status"] = "success"

        # Extract final loss
        final_loss = get_final_osft_loss(output_dir)
        result["final_loss"] = final_loss
        if final_loss is not None:
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
                # Multi-GPU setup
                nproc_per_node=NUM_GPUS,
                nnodes=1,
                node_rank=0,
                rdzv_id=f"validation-lora-{int(time.time())}",
                rdzv_endpoint="127.0.0.1:29501",
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
    data_path = create_overfit_dataset(dataset_dir)

    # Print header
    print_test_header(model_config, mode, use_liger, use_qlora, output_dir)

    if mode == "sft":
        result = run_sft_validation(model_config, data_path, output_dir, use_liger)
    elif mode == "osft":
        result = run_osft_validation(model_config, data_path, output_dir, use_liger)
    else:  # lora
        result = run_lora_validation(model_config, data_path, output_dir, use_qlora)

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
        )
    else:
        parser.print_help()
        print("\nError: Either --models or --run-all is required")
        sys.exit(1)


if __name__ == "__main__":
    main()
