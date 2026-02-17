#!/usr/bin/env python3
"""
LoRA Training Example: Interactive Script

This script demonstrates LoRA + SFT training with various options including
QLoRA (quantization), single-GPU, and multi-GPU configurations.

Example usage:
    # Basic LoRA training
    python lora_example.py \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    # QLoRA with 4-bit quantization
    python lora_example.py \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints \\
        --qlora

    # Multi-GPU training
    torchrun --nproc-per-node=4 lora_example.py \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

from training_hub import lora_sft


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

default_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
default_lora_r = 32
default_lora_alpha = 64
default_learning_rate = 1e-4
default_micro_batch_size = 1
default_max_seq_len = 2048
default_nproc_per_node = 1


def create_sample_data(output_path: str = "./sample_data.jsonl", format_type: str = "messages"):
    """Create sample training data in the specified format."""

    if format_type == "messages":
        data = [
            {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Explain what LoRA is."},
                    {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method for large language models."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."},
                    {"role": "assistant", "content": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)"}
                ]
            }
        ]
    elif format_type == "alpaca":
        data = [
            {
                "instruction": "What is the capital of France?",
                "input": "",
                "output": "The capital of France is Paris."
            },
            {
                "instruction": "Explain what LoRA is.",
                "input": "",
                "output": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method for large language models."
            },
            {
                "instruction": "Write a Python function to calculate fibonacci numbers.",
                "input": "",
                "output": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)"
            }
        ]
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write data
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\\n")

    print(f"📝 Created sample data: {output_path} ({format_type} format, {len(data)} samples)")
    return output_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA + SFT Training with Unsloth Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python lora_example.py --data-path data.jsonl --ckpt-output-dir ./outputs

  # QLoRA with 4-bit quantization
  python lora_example.py --data-path data.jsonl --ckpt-output-dir ./outputs --qlora

  # Multi-GPU training (requires torchrun)
  torchrun --nproc-per-node=4 lora_example.py --data-path data.jsonl --ckpt-output-dir ./outputs

  # Different model
  python lora_example.py --data-path data.jsonl --ckpt-output-dir ./outputs --model-path ibm-granite/granite-3.3-8b-instruct
        """
    )

    # Required arguments
    parser.add_argument('--data-path',
                       help='Path to training data (JSONL format). Use --create-sample-data to generate test data.')
    parser.add_argument('--ckpt-output-dir', required=True,
                       help='Directory to save checkpoints and outputs')

    # Model configuration
    parser.add_argument('--model-path', default=default_model_path,
                       help=f'Model path or HuggingFace name (default: {default_model_path})')

    # LoRA configuration
    parser.add_argument('--lora-r', type=int, default=default_lora_r,
                       help=f'LoRA rank (default: {default_lora_r})')
    parser.add_argument('--lora-alpha', type=int, default=default_lora_alpha,
                       help=f'LoRA alpha parameter (default: {default_lora_alpha})')
    parser.add_argument('--lora-dropout', type=float, default=0.0,
                       help='LoRA dropout (default: 0.0, optimized for Unsloth)')

    # Training configuration
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--learning-rate', type=float, default=default_learning_rate,
                       help=f'Learning rate (default: {default_learning_rate})')
    parser.add_argument('--micro-batch-size', type=int, default=default_micro_batch_size,
                       help=f'Batch size per GPU (default: {default_micro_batch_size})')
    parser.add_argument('--effective-batch-size', type=int,
                       help='Effective batch size across all GPUs')
    parser.add_argument('--max-seq-len', type=int, default=default_max_seq_len,
                       help=f'Maximum sequence length (default: {default_max_seq_len:,})')
    parser.add_argument('--nproc-per-node', type=int, default=default_nproc_per_node,
                       help=f'Number of GPUs per node (default: {default_nproc_per_node})')

    # Quantization options
    parser.add_argument('--qlora', action='store_true',
                       help='Enable QLoRA (4-bit quantization)')
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='Use 8-bit quantization instead of 4-bit')

    # Dataset options
    parser.add_argument('--dataset-type', default='chat_template', choices=['chat_template', 'alpaca', 'passthrough'],
                       help='Dataset format (default: chat_template)')
    parser.add_argument('--field-messages', default='messages',
                       help='Field name for messages (default: messages)')
    parser.add_argument('--field-instruction', default='instruction',
                       help='Field name for instruction (alpaca format, default: instruction)')
    parser.add_argument('--field-input', default='input',
                       help='Field name for input (alpaca format, default: input)')
    parser.add_argument('--field-output', default='output',
                       help='Field name for output (alpaca format, default: output)')


    # Logging options
    parser.add_argument('--wandb-project',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-entity',
                       help='Weights & Biases entity name')
    parser.add_argument('--wandb-run-name',
                        help='Weights & Biases run name')

    # Utility options
    parser.add_argument('--create-sample-data', action='store_true',
                       help='Create sample data file and exit')
    parser.add_argument('--sample-data-format', default='messages', choices=['messages', 'alpaca'],
                       help='Format for sample data (default: messages)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print configuration and exit without training')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()

    # Handle sample data creation
    if args.create_sample_data:
        sample_path = "./sample_data.jsonl"
        create_sample_data(sample_path, args.sample_data_format)
        print("\\nUse this sample data with:")
        if args.sample_data_format == 'alpaca':
            print(f"python {sys.argv[0]} --data-path {sample_path} --ckpt-output-dir ./outputs --dataset-type alpaca")
        else:
            print(f"python {sys.argv[0]} --data-path {sample_path} --ckpt-output-dir ./outputs")
        return

    # Validate required arguments
    if not args.data_path:
        print("❌ Error: --data-path is required (or use --create-sample-data to generate test data)")
        sys.exit(1)

    # Validate quantization options
    if args.qlora and args.load_in_8bit:
        print("❌ Error: --qlora (4-bit) and --load-in-8bit cannot be used together. Please choose one.")
        sys.exit(1)

    # Check for multi-GPU setup
    is_distributed = 'LOCAL_RANK' in os.environ or 'WORLD_SIZE' in os.environ

    # Warn if user wants multi-GPU but didn't launch with torchrun
    if args.nproc_per_node > 1 and not is_distributed:
        print("⚠️  WARNING: You specified --nproc-per-node > 1 but aren't using torchrun!")
        print("   For multi-GPU training, use:")
        script = Path(sys.argv[0]).name
        print(f"   torchrun --nproc-per-node={args.nproc_per_node} {script} --data-path {args.data_path} --ckpt-output-dir {args.ckpt_output_dir}")
        print()
        print("   Continuing with single-GPU training...")
        args.nproc_per_node = 1
        print()

    # Print configuration
    print("🚀 LoRA Training")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"Dataset type: {args.dataset_type}")
    if args.nproc_per_node > 1:
        print(f"GPUs: {args.nproc_per_node}")
    print()
    print("LoRA Configuration:")
    print(f"  LoRA rank (r): {args.lora_r}")
    print(f"  LoRA alpha: {args.lora_alpha}")
    print(f"  LoRA dropout: {args.lora_dropout}")
    print()
    print("Training Configuration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Micro batch size: {args.micro_batch_size}")
    if args.effective_batch_size:
        print(f"  Effective batch size: {args.effective_batch_size}")
    print(f"  Max sequence length: {args.max_seq_len:,}")
    print()
    print("Optimizations:")
    if args.qlora:
        print("  ✅ QLoRA (4-bit quantization)")
    elif args.load_in_8bit:
        print("  ✅ 8-bit quantization")
    else:
        print("  ✅ Standard LoRA (no quantization)")
    print("  ✅ Unsloth optimizations (automatic 2x speedup, 70% less VRAM)")
    if is_distributed:
        print("  ✅ Multi-GPU distributed training")
    if args.wandb_project:
        print(f"  ✅ W&B logging: {args.wandb_entity or 'default'}/{args.wandb_project}")
    print()

    # Dry run check
    if args.dry_run:
        print("🏃 Dry run mode - configuration validated, exiting without training")
        return

    # Validate data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found: {args.data_path}")
        print("💡 Use --create-sample-data to generate test data")
        sys.exit(1)

    # Prepare training arguments
    train_kwargs = {
        # Model and data
        'model_path': args.model_path,
        'data_path': args.data_path,
        'ckpt_output_dir': args.ckpt_output_dir,

        # LoRA configuration
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,

        # Training configuration
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'micro_batch_size': args.micro_batch_size,
        'max_seq_len': args.max_seq_len,

        # Dataset configuration
        'dataset_type': args.dataset_type,
        'field_messages': args.field_messages,
        'field_instruction': args.field_instruction,
        'field_input': args.field_input,
        'field_output': args.field_output,

        # Quantization
        'load_in_4bit': args.qlora,
        'load_in_8bit': args.load_in_8bit,

        # Optimization
        'bf16': True,
        'sample_packing': True,

        # Logging
        'logging_steps': 10,
        'save_steps': 200,
        'save_total_limit': 3,
    }

    # Add optional parameters
    if args.effective_batch_size:
        train_kwargs['effective_batch_size'] = args.effective_batch_size

    if args.wandb_project:
        train_kwargs['wandb_project'] = args.wandb_project
        if args.wandb_entity:
            train_kwargs['wandb_entity'] = args.wandb_entity
        if args.wandb_run_name:
            train_kwargs['wandb_run_name'] = args.wandb_run_name

    # Start training
    print(f"🚀 Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⏱️  Training in progress...")
    print()

    start_time = time.time()

    try:
        result = lora_sft(**train_kwargs)

        # Training completed successfully
        elapsed_time = time.time() - start_time
        print()
        print("=" * 60)
        print("✅ LoRA training completed successfully!")
        print(f"⏱️  Training time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"📁 Output saved to: {args.ckpt_output_dir}")
        print(f"🔧 Model type: {type(result['model'])}")
        print(f"🔤 Tokenizer type: {type(result['tokenizer'])}")
        print()
        print("🚀 Benefits achieved:")
        print("• 2x faster training with Unsloth optimizations")
        print("• 70% less VRAM usage")
        print("• Full compatibility with HuggingFace models")
        if args.qlora:
            print("• Additional memory savings from 4-bit quantization")

    except Exception as e:  # noqa: BLE001
        elapsed_time = time.time() - start_time
        print()
        print("=" * 60)
        print(f"❌ LoRA training failed after {elapsed_time:.1f} seconds")
        print(f"Error: {e}")
        print()
        print("💡 Troubleshooting tips:")
        print("• Enable quantization to reduce memory: --qlora")
        print("• Reduce batch size: --micro-batch-size 1")
        print("• Reduce sequence length: --max-seq-len 512")
        print("• Check your data format matches --dataset-type")
        sys.exit(1)


if __name__ == "__main__":
    main()