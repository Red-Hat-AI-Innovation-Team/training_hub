#!/usr/bin/env python3
"""
SFT Training Example: Qwen3.5-4B on Medical Chatbot Dataset

This script demonstrates SFT (Supervised Fine-Tuning) of the Qwen3.5-4B model
on the ruslanmv/ai-medical-chatbot dataset using Training Hub.

Qwen3.5-4B uses a hybrid architecture combining Gated Delta Networks with
standard attention layers. Although the base model is multi-modal (VLM),
Training Hub loads it as a text-only CausalLM for efficient fine-tuning.

Dataset: https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot

Example usage:
    python sft_qwen35_medical_example.py \\
        --data-path /path/to/medical_chatbot_train.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

import argparse
import os
import sys
import tempfile
import time

from training_hub import sft


def main():
    parser = argparse.ArgumentParser(
        description="SFT Training Example: Qwen3.5-4B on Medical Chatbot Dataset"
    )

    # Required parameters
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to training data (JSONL messages format)",
    )
    parser.add_argument(
        "--ckpt-output-dir",
        required=True,
        help="Directory to save checkpoints",
    )

    # Optional overrides
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3.5-4B",
        help="Model path or HuggingFace name (default: Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of epochs (default: 3)"
    )
    parser.add_argument(
        "--max-tokens-per-gpu",
        type=int,
        default=8192,
        help="Max tokens per GPU (default: 8192)",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=8,
        help="Number of GPUs (default: 8)",
    )
    parser.add_argument(
        "--data-output-dir",
        default="/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir(),
        help="Directory for processed data output (default: /dev/shm or system temp)",
    )

    args = parser.parse_args()

    print("SFT Training: Qwen3.5-4B on Medical Chatbot Dataset")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print()

    start_time = time.time()

    try:
        sft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            # Training parameters
            num_epochs=args.num_epochs,
            effective_batch_size=32,
            learning_rate=1e-5,
            max_seq_len=4096,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            # Data processing
            data_output_dir=args.data_output_dir,
            warmup_steps=0,
            save_samples=0,
            # Checkpointing
            checkpoint_at_epoch=True,
            accelerate_full_state_at_epoch=False,
            # Multi-GPU setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=200,
            rdzv_endpoint="127.0.0.1:42067",
        )

        end_time = time.time()
        duration = end_time - start_time

        print("=" * 60)
        print("Training completed successfully!")
        print(f"Duration: {duration/3600:.2f} hours")
        print(f"Checkpoints: {args.ckpt_output_dir}/hf_format/")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print("=" * 60)
        print(f"Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("Try reducing --max-tokens-per-gpu if you see OOM errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
