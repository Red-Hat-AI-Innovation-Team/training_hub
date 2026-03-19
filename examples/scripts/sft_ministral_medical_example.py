#!/usr/bin/env python3
"""
SFT Training Example: Ministral 3 3B on Medical Flashcards Dataset

This script demonstrates SFT (Supervised Fine-Tuning) of the Ministral 3 3B model
on a medical Q&A dataset using Training Hub.

Ministral 3 3B (Ministral3ForCausalLM) is a compact instruction-tuned model from
Mistral AI. Although it originates from a VLM wrapper (Mistral3ForConditionalGeneration),
Training Hub extracts and trains the CausalLM text backbone directly for efficient
text-only fine-tuning.

Recommended dataset: medalpaca/medical_meadow_medical_flashcards
    https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards

Data must be in JSONL messages format. Convert the flashcards dataset with:

    from datasets import load_dataset
    import json

    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    with open("medical_flashcards.jsonl", "w") as f:
        for row in ds:
            messages = [
                {"role": "user", "content": row["input"]},
                {"role": "assistant", "content": row["output"]},
            ]
            f.write(json.dumps({"messages": messages}) + "\\n")

Example usage:
    python sft_ministral_medical_example.py \\
        --data-path /path/to/medical_flashcards.jsonl \\
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
        description="SFT Training Example: Ministral 3 3B on Medical Flashcards Dataset"
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
        default="mistralai/Ministral-3-3B-Instruct-2512",
        help="Model path or HuggingFace name (default: mistralai/Ministral-3-3B-Instruct-2512)",
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

    print("SFT Training: Ministral 3 3B on Medical Flashcards Dataset")
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
            rdzv_id=300,
            rdzv_endpoint="127.0.0.1:42069",
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
