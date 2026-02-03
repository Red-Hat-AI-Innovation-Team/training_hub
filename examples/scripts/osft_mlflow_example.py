#!/usr/bin/env python3
"""
OSFT Training Example with MLflow Logging

This script demonstrates OSFT (Orthogonal Subspace Fine-Tuning) training
with MLflow integration using training_hub.

Example usage:
    python osft_mlflow_example.py --data-path /path/to/data.jsonl

    # With custom model:
    python osft_mlflow_example.py --data-path /path/to/data.jsonl --model-path meta-llama/Llama-3.1-8B-Instruct

    # With custom MLflow settings:
    python osft_mlflow_example.py --data-path /path/to/data.jsonl --mlflow-uri http://remote-server:5000
"""

import argparse
import os
import sys
from datetime import datetime

from training_hub import osft


def main():
    parser = argparse.ArgumentParser(description="OSFT Training with MLflow Logging")

    # Model and data paths
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model path or HuggingFace name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to training data (JSONL format)"
    )
    parser.add_argument(
        "--ckpt-output-dir",
        default="./osft-mlflow-checkpoints",
        help="Directory to save checkpoints",
    )

    # OSFT-specific parameters
    parser.add_argument(
        "--unfreeze-rank-ratio",
        type=float,
        default=0.1,
        help="Ratio of ranks to unfreeze (default: 0.1)",
    )

    # Training parameters
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of epochs (default: 1)"
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=32,
        help="Effective batch size (default: 32)",
    )
    parser.add_argument(
        "--max-tokens-per-gpu",
        type=int,
        default=4096,
        help="Max tokens per GPU (default: 4096)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--nproc-per-node", type=int, default=1, help="Number of GPUs (default: 1)"
    )

    # MLflow settings
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="osft-training",
        help="MLflow experiment name (default: osft-training)",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="MLflow run name (default: auto-generated with timestamp)",
    )

    args = parser.parse_args()

    # Generate run name if not provided
    run_name = (
        args.mlflow_run_name or f"osft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    # Create output directory
    os.makedirs(args.ckpt_output_dir, exist_ok=True)

    print("=" * 60)
    print("OSFT Training with MLflow Logging")
    print("=" * 60)
    print(f"Model:            {args.model_path}")
    print(f"Data:             {args.data_path}")
    print(f"Output:           {args.ckpt_output_dir}")
    print(f"GPUs:             {args.nproc_per_node}")
    print(f"Epochs:           {args.num_epochs}")
    print(f"Batch Size:       {args.effective_batch_size}")
    print(f"Max tokens/GPU:   {args.max_tokens_per_gpu}")
    print(f"Learning Rate:    {args.learning_rate}")
    print(f"Unfreeze Ratio:   {args.unfreeze_rank_ratio}")
    print("-" * 60)
    print(f"MLflow URI:       {args.mlflow_uri}")
    print(f"MLflow Experiment: {args.mlflow_experiment}")
    print(f"Run Name:         {run_name}")
    print("=" * 60)
    print()

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
            effective_batch_size=args.effective_batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=4096,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            warmup_steps=10,
            # Checkpointing
            checkpoint_at_epoch=True,
            # Logging - enable MLflow
            mlflow_tracking_uri=args.mlflow_uri,
            mlflow_experiment_name=args.mlflow_experiment,
            mlflow_run_name=run_name,
            # Distributed training
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=100,
            rdzv_endpoint="127.0.0.1:29500",
        )

        print()
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Checkpoints saved to: {args.ckpt_output_dir}")
        print(f"View results in MLflow: {args.mlflow_uri}")
        print("=" * 60)

    except Exception as e:  # noqa: BLE001
        print()
        print("=" * 60)
        print(f"Training failed: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
