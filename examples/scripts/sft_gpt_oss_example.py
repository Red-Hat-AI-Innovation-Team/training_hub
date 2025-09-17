#!/usr/bin/env python3
"""
SFT Training Example: GPT-OSS 20B

This script demonstrates SFT training with GPT-OSS 20B model from OpenAI
using a single-node, multi-GPU setup with training_hub.

GPT-OSS 20B is a high-quality open source model that provides excellent
performance for supervised fine-tuning tasks.

Example usage:
    python sft_gpt_oss_example.py \
        --data-path /path/to/data.jsonl \
        --ckpt-output-dir /path/to/checkpoints
"""

import os
import sys
import time
from datetime import datetime
import argparse

from training_hub import sft


def main():
    parser = argparse.ArgumentParser(description='SFT Training Example: GPT-OSS 20B')

    # Required parameters
    parser.add_argument('--data-path', required=True,
                       help='Path to training data (JSONL format)')
    parser.add_argument('--ckpt-output-dir', required=True,
                       help='Directory to save checkpoints')

    # Optional overrides
    parser.add_argument('--model-path', default='openai/gpt-oss-20b',
                       help='Model path or HuggingFace name (default: openai/gpt-oss-20b)')
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--max-tokens-per-gpu', type=int, default=12000,
                       help='Max tokens per GPU (default: 12000 for 20B model)')
    parser.add_argument('--nproc-per-node', type=int, default=8,
                       help='Number of GPUs (default: 8)')

    args = parser.parse_args()

    # GPT-OSS 20B configuration
    print("🚀 SFT Training: GPT-OSS 20B")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print()

    # Training configuration optimized for GPT-OSS 20B
    start_time = time.time()

    try:
        result = sft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,

            # Training parameters optimized for GPT-OSS 20B
            num_epochs=args.num_epochs,
            effective_batch_size=32,            # Smaller batch size for 20B model
            learning_rate=2e-5,                # Conservative LR for larger model
            max_seq_len=8192,                  # Standard context length
            max_tokens_per_gpu=args.max_tokens_per_gpu,

            # Data processing
            data_output_dir="/dev/shm",        # Use RAM disk for speed
            warmup_steps=100,
            save_samples=0,                    # 0 disables sample-based checkpointing, use epoch-based only

            # Checkpointing
            checkpoint_at_epoch=True,
            accelerate_full_state_at_epoch=False, # Disable for smaller checkpoints (no auto-resumption)

            # Single-node multi-GPU setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=104,
            rdzv_endpoint="127.0.0.1:29500",
        )

        end_time = time.time()
        duration = end_time - start_time

        print("=" * 50)
        print("✅ Training completed successfully!")
        print(f"⏱️  Duration: {duration/3600:.2f} hours")
        print(f"📁 Checkpoints: {args.ckpt_output_dir}/hf_format/")
        print()
        print("🎯 Your GPT-OSS 20B model has been fine-tuned!")
        print("   The model is now specialized for your specific task")
        print("   while maintaining the high quality of the base model.")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print("=" * 50)
        print(f"❌ Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("💡 Troubleshooting tips:")
        print("   - Reduce --max-tokens-per-gpu if you see OOM errors")
        sys.exit(1)


if __name__ == "__main__":
    main()