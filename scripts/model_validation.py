#!/usr/bin/env python3
"""
Model Validation Script for Training Hub

This script validates that various model architectures can be trained successfully
using both SFT and OSFT algorithms. The goal is to overfit on a single sample
(replicated 1000 times) to achieve NLL approaching 0.

Usage:
    python model_validation.py --model-key llama --mode sft
    python model_validation.py --model-key llama --mode osft --use-liger
    python model_validation.py --run-all
    python model_validation.py --run-all --mode sft

Configuration:
    - GPUs: 8
    - Effective batch size: 32
    - Learning rate: 1e-5
    - Tokens per GPU: 8192 (8k)
    - Epochs: 5
    - OSFT unfreeze rank ratio: 0.5
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Rich console for formatted output
console = Console()

# CRITICAL: Set this BEFORE any transformers imports to prevent worker processes
# from using incompatible cached remote code
os.environ["HF_HUB_DISABLE_REMOTE_CODE_EXECUTION"] = "1"

# Models that require trust_remote_code - populated from MODELS registry after it's defined
_MODELS_REQUIRING_TRUST_REMOTE_CODE: set[str] = set()


def _patch_trust_remote_code():
    """
    Monkey-patch transformers and instructlab to selectively trust remote code.

    Only enables trust_remote_code for models explicitly marked with
    requires_trust_remote_code=True in the MODELS registry. This prevents
    using potentially incompatible remote code when transformers has native support.
    """
    from transformers import dynamic_module_utils

    original_resolve = dynamic_module_utils.resolve_trust_remote_code

    def patched_resolve_trust_remote_code(
        trust_remote_code, model_name, has_local_code, has_remote_code,
        error_message=None, upstream_repo=None
    ):
        # If trust_remote_code was explicitly set, honor it
        if trust_remote_code is not None:
            return original_resolve(
                trust_remote_code, model_name, has_local_code, has_remote_code,
                error_message, upstream_repo
            )

        # Only trust remote code for models explicitly configured to require it
        if has_remote_code:
            # Check if this model is in our allow-list
            for allowed_model in _MODELS_REQUIRING_TRUST_REMOTE_CODE:
                if allowed_model in model_name:
                    print(f"🔓 Trusting remote code for: {model_name}")
                    return True
            # NOT in allow-list - use local/native implementation
            if has_local_code:
                print(f"🔒 Using native transformers implementation for: {model_name}")
                return False

        # No remote code or no local code - use original logic
        return original_resolve(
            trust_remote_code, model_name, has_local_code, has_remote_code,
            error_message, upstream_repo
        )

    dynamic_module_utils.resolve_trust_remote_code = patched_resolve_trust_remote_code

    # Also patch instructlab's Model class to inject trust_remote_code into base_model_args
    try:
        from instructlab.training import model as ilab_model

        original_model_init = ilab_model.Model.__init__

        def patched_model_init(self, *args, **kwargs):
            original_model_init(self, *args, **kwargs)
            # Inject trust_remote_code based on allow-list
            model_path = self.base_model_args.get("pretrained_model_name_or_path", "")
            should_trust = any(allowed in model_path for allowed in _MODELS_REQUIRING_TRUST_REMOTE_CODE)
            self.base_model_args["trust_remote_code"] = should_trust
            if should_trust:
                print(f"  [Worker] Trusting remote code: {model_path}")
            else:
                print(f"  [Worker] Using native implementation: {model_path}")

        ilab_model.Model.__init__ = patched_model_init
        print("✅ Patched instructlab Model class for worker processes")
    except ImportError:
        pass  # instructlab not loaded yet

    print("✅ Patched transformers for selective trust_remote_code")


# Apply the patch immediately
_patch_trust_remote_code()

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

# Populate the trust_remote_code allow-list from MODELS registry
for _key, _config in MODELS.items():
    if _config.requires_trust_remote_code:
        _MODELS_REQUIRING_TRUST_REMOTE_CODE.add(_config.model_id)
        print(f"  → {_config.model_id} requires trust_remote_code")

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

    # Create summary panel
    summary_text = Text()
    summary_text.append(f"Total: {len(results)}  ", style="bold")
    summary_text.append(f"Success: {success_count}  ", style="bold green")
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
    table.add_column("Liger", style="magenta", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Duration", style="dim", no_wrap=True)
    table.add_column("Notes", overflow="fold")

    for r in results:
        model_id = r.get("model_id", r.get("model_key", "unknown"))
        # Shorten model_id for display
        model_short = model_id.split("/")[-1] if "/" in model_id else model_id

        mode = r.get("mode", "?").upper()
        liger = "Yes" if r.get("use_liger") else "No"

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

        table.add_row(model_short, mode, liger, status_text, duration, notes)

    console.print(table)

    # Print file location
    console.print(f"\n[dim]Full results saved to:[/dim] [cyan]{results_file}[/cyan]")


def print_test_header(model_config: "ModelConfig", mode: str, use_liger: bool, output_dir: str):
    """Print test header with rich formatting."""
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Field", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Architecture", model_config.architecture)
    table.add_row("Model", model_config.model_id)
    table.add_row("Mode", mode.upper())
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
    mode: Literal["sft", "osft"],
    use_liger: bool = True,
    base_output_dir: str = BASE_OUTPUT_DIR,
    dataset_dir: str = DATASET_OUTPUT_DIR,
) -> dict:
    """
    Run a single validation test.

    Args:
        model_key: Key from MODELS dictionary
        mode: Training mode ("sft" or "osft")
        use_liger: Whether to use Liger kernels (OSFT only)
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
    liger_suffix = "_liger" if use_liger else "_noliger"
    run_name = f"{model_key}_{mode}{liger_suffix}_{timestamp}"
    output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create or use existing dataset
    data_path = create_overfit_dataset(dataset_dir)

    # Print header
    print_test_header(model_config, mode, use_liger, output_dir)

    if mode == "sft":
        result = run_sft_validation(model_config, data_path, output_dir, use_liger)
    else:
        result = run_osft_validation(model_config, data_path, output_dir, use_liger)

    # Print result summary
    if result["status"] == "success":
        console.print(f"[bold green]SUCCESS[/bold green] - Completed in {format_duration(result['duration_seconds'])}")
    else:
        console.print(f"[bold red]FAILED[/bold red] - {result['error']}")

    # Save result to file
    result_file = os.path.join(output_dir, "validation_result.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_all_validations(
    mode: Literal["sft", "osft", "both"] = "both",
    liger_modes: list[bool] = [True, False],
    base_output_dir: str = BASE_OUTPUT_DIR,
    dataset_dir: str = DATASET_OUTPUT_DIR,
    model_keys: list[str] | None = None,
) -> list[dict]:
    """
    Run validation tests for all models.

    Args:
        mode: Training mode(s) to test
        liger_modes: Liger configurations to test (OSFT only)
        base_output_dir: Base directory for outputs
        dataset_dir: Directory for dataset
        model_keys: Optional list of specific model keys to test

    Returns:
        List of validation results
    """
    results = []
    models_to_test = model_keys or list(MODELS.keys())
    modes_to_test = ["sft", "osft"] if mode == "both" else [mode]

    # Total tests = models * modes * liger_variants
    total_tests = len(models_to_test) * len(modes_to_test) * len(liger_modes)

    # Print run configuration
    config_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    config_table.add_column("", style="dim")
    config_table.add_column("", style="bold")
    config_table.add_row("Total Tests", str(total_tests))
    config_table.add_row("Models", str(len(models_to_test)))
    config_table.add_row("Modes", ", ".join(modes_to_test))
    config_table.add_row("Liger Variants", str(liger_modes))

    console.print()
    console.print(Panel(config_table, title="Validation Run Configuration", border_style="cyan"))
    console.print()

    test_num = 0
    for model_key in models_to_test:
        for test_mode in modes_to_test:
            for use_liger in liger_modes:
                test_num += 1
                liger_str = "liger" if use_liger else "no-liger"
                console.print(f"\n[bold cyan][{test_num}/{total_tests}][/bold cyan] Testing [green]{model_key}[/green] - {test_mode} ({liger_str})")
                try:
                    result = run_single_validation(
                        model_key=model_key,
                        mode=test_mode,
                        use_liger=use_liger,
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
    # Run single model validation
    python model_validation.py --model-key llama --mode sft
    python model_validation.py --model-key llama --mode osft --use-liger
    python model_validation.py --model-key llama --mode osft --no-liger

    # Run all models for a specific mode
    python model_validation.py --run-all --mode sft
    python model_validation.py --run-all --mode osft

    # Run all combinations
    python model_validation.py --run-all --mode both

    # List available models
    python model_validation.py --list-models

Available model keys:
    """
        + ", ".join(MODELS.keys()),
    )

    parser.add_argument(
        "--model-key", choices=list(MODELS.keys()), help="Model key to validate (see --list-models for options)"
    )
    parser.add_argument("--mode", choices=["sft", "osft", "both"], default="sft", help="Training mode (default: sft)")
    parser.add_argument(
        "--use-liger", action="store_true", default=True, help="Enable Liger kernels for OSFT (default: True)"
    )
    parser.add_argument("--no-liger", action="store_true", help="Disable Liger kernels for OSFT")
    parser.add_argument("--run-all", action="store_true", help="Run validation for all models")
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS.keys()), help="Specific models to test (with --run-all)"
    )
    parser.add_argument(
        "--output-dir", default=BASE_OUTPUT_DIR, help=f"Base output directory (default: {BASE_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--dataset-dir", default=DATASET_OUTPUT_DIR, help=f"Dataset directory (default: {DATASET_OUTPUT_DIR})"
    )
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")

    args = parser.parse_args()

    # Handle liger flag
    use_liger = not args.no_liger if args.no_liger else args.use_liger

    if args.list_models:
        print_models_table()
        return

    if args.run_all:
        # Determine liger modes to test
        if args.mode == "osft":
            liger_modes = [True, False] if not args.no_liger else [False]
        else:
            liger_modes = [True, False]

        run_all_validations(
            mode=args.mode,
            liger_modes=liger_modes,
            base_output_dir=args.output_dir,
            dataset_dir=args.dataset_dir,
            model_keys=args.models,
        )
    elif args.model_key:
        run_single_validation(
            model_key=args.model_key,
            mode=args.mode if args.mode != "both" else "sft",
            use_liger=use_liger,
            base_output_dir=args.output_dir,
            dataset_dir=args.dataset_dir,
        )
    else:
        parser.print_help()
        print("\nError: Either --model-key or --run-all is required")
        sys.exit(1)


if __name__ == "__main__":
    main()
