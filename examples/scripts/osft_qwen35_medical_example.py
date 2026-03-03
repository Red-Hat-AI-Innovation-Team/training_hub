#!/usr/bin/env python3
"""
OSFT Training Example: Qwen3.5-4B on Medical Chatbot Dataset

This script demonstrates OSFT (Orthogonal Subspace Fine-Tuning) of the Qwen3.5-4B
model on the ruslanmv/ai-medical-chatbot dataset using Training Hub.

OSFT constrains weight updates to low-rank subspaces orthogonal to the model's
critical knowledge subspace, enabling domain adaptation without catastrophic
forgetting. This is useful for adding medical domain knowledge while
preserving the model's general reasoning and language abilities.

Qwen3.5-4B uses a hybrid architecture (Gated DeltaNet + standard attention).
OSFT targets both self_attn projections (8 attention layers) and
linear_attn.out_proj (24 DeltaNet layers), plus all MLP projections.

Dataset: https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot

Example usage:
    python osft_qwen35_medical_example.py \\
        --data-path /path/to/medical_chatbot_train.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

import argparse
import glob
import os
import sys
import time

from training_hub import osft


def find_most_recent_checkpoint(output_dir):
    """Find the most recent checkpoint in the training output directory."""
    checkpoint_pattern = os.path.join(output_dir, "hf_format", "samples_*.0")
    checkpoint_dirs = glob.glob(checkpoint_pattern)

    if not checkpoint_dirs:
        return None

    return max(checkpoint_dirs, key=os.path.getctime)


def main():
    parser = argparse.ArgumentParser(
        description="OSFT Training Example: Qwen3.5-4B on Medical Chatbot Dataset"
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
        "--num-epochs",
        type=int,
        default=3,
        help="Number of epochs (default: 3)",
    )
    parser.add_argument(
        "--unfreeze-rank-ratio",
        type=float,
        default=0.25,
        help="Unfreeze rank ratio for OSFT (0.0-1.0, default: 0.25)",
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

    args = parser.parse_args()

    print("OSFT Training: Qwen3.5-4B on Medical Chatbot Dataset")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Unfreeze Rank Ratio: {args.unfreeze_rank_ratio}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print()

    start_time = time.time()

    try:
        osft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            # OSFT-specific parameters
            unfreeze_rank_ratio=args.unfreeze_rank_ratio,
            # Training parameters
            num_epochs=args.num_epochs,
            effective_batch_size=32,
            learning_rate=5e-6,
            max_seq_len=4096,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            # Data processing
            data_output_dir="/dev/shm",
            warmup_steps=0,
            # Optimization
            use_liger=False,
            seed=42,
            lr_scheduler="cosine",
            # Checkpointing
            checkpoint_at_epoch=True,
            save_final_checkpoint=True,
            # Multi-GPU setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=201,
            rdzv_endpoint="127.0.0.1:29500",
        )

        end_time = time.time()
        duration = end_time - start_time

        most_recent = find_most_recent_checkpoint(args.ckpt_output_dir)

        print("=" * 60)
        print("OSFT Training completed successfully!")
        print(f"Duration: {duration/3600:.2f} hours")
        print(f"Checkpoints: {args.ckpt_output_dir}/hf_format")
        if most_recent:
            print(f"Most recent checkpoint: {most_recent}")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print("=" * 60)
        print(f"Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("Troubleshooting tips:")
        print("  - Reduce --max-tokens-per-gpu if you see OOM errors")
        print("  - Try --unfreeze-rank-ratio between 0.2-0.4 for domain adaptation")
        sys.exit(1)


if __name__ == "__main__":
    main()
