#!/usr/bin/env python3
"""
LoRA Training Example: Ministral 3 3B on Medical Flashcards Dataset

This script demonstrates LoRA (Low-Rank Adaptation) fine-tuning of the Ministral 3 3B
model on a medical Q&A dataset using Training Hub.

LoRA adds small trainable low-rank matrices to the model's attention and MLP layers,
enabling parameter-efficient fine-tuning with significantly reduced memory requirements.
This script also supports QLoRA (4-bit quantization) for even lower memory usage.

Ministral 3 3B (Ministral3ForCausalLM) is a compact instruction-tuned model from
Mistral AI. Its small size makes it well-suited for LoRA fine-tuning, fitting
comfortably on a single GPU.

Recommended dataset: medalpaca/medical_meadow_medical_flashcards
    https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards

See sft_ministral_medical_example.py for data conversion instructions.

Example usage:
    # Basic LoRA training (single GPU)
    python lora_ministral_medical_example.py \\
        --data-path /path/to/medical_flashcards.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    # QLoRA with 4-bit quantization
    python lora_ministral_medical_example.py \\
        --data-path /path/to/medical_flashcards.jsonl \\
        --ckpt-output-dir /path/to/checkpoints \\
        --qlora

    # Multi-GPU training
    python lora_ministral_medical_example.py \\
        --data-path /path/to/medical_flashcards.jsonl \\
        --ckpt-output-dir /path/to/checkpoints \\
        --nproc-per-node 8
"""

import argparse
import os
import sys
import time

from training_hub import lora_sft


def main():
    parser = argparse.ArgumentParser(
        description="LoRA Training Example: Ministral 3 3B on Medical Flashcards Dataset"
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
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--lora-r", type=int, default=32, help="LoRA rank (default: 32)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=64, help="LoRA alpha (default: 64)"
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Enable QLoRA (4-bit quantization) for reduced memory usage",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)",
    )

    args = parser.parse_args()

    quant_mode = "QLoRA (4-bit)" if args.qlora else "LoRA (full precision)"

    print(f"LoRA Training: Ministral 3 3B on Medical Flashcards Dataset")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"Mode: {quant_mode}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Max sequence length: {args.max_seq_len:,}")
    print()

    start_time = time.time()

    try:
        result = lora_sft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            # LoRA configuration
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            # Training parameters
            num_epochs=args.num_epochs,
            effective_batch_size=32,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            warmup_steps=10,
            # Quantization
            load_in_4bit=args.qlora,
            # Optimization
            bf16=True,
            sample_packing=True,
            # Logging and saving
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            # Multi-GPU setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=302,
            rdzv_endpoint="127.0.0.1:29502",
        )

        end_time = time.time()
        duration = end_time - start_time

        print("=" * 60)
        print("LoRA Training completed successfully!")
        print(f"Duration: {duration/3600:.2f} hours")
        print(f"Output: {args.ckpt_output_dir}")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print("=" * 60)
        print(f"Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("Troubleshooting tips:")
        print("  - Enable QLoRA to reduce memory: --qlora")
        print("  - Reduce sequence length: --max-seq-len 2048")
        print("  - Check your data format (expects JSONL with 'messages' field)")
        sys.exit(1)


if __name__ == "__main__":
    main()
