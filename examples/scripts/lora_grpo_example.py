#!/usr/bin/env python3
"""
LoRA + GRPO Training Example

Demonstrates reinforcement learning from verifiable rewards (RLVR) using
adapter-based training with Group Relative Policy Optimization (GRPO).

Two modes are shown:
1. Built-in tool-call verification (single-turn, uses Toucan dataset)
2. Custom rollout function (multi-turn, user-defined environment)

Example usage:
    # Single-turn tool-call training with Toucan dataset
    python lora_grpo_example.py \\
        --ckpt-output-dir ./grpo_output

    # Custom model
    python lora_grpo_example.py \\
        --model-path ibm-granite/granite-4.0-h-tiny \\
        --ckpt-output-dir ./grpo_output \\
        --lora-r 128 --lora-alpha 256

    # Quick test run
    python lora_grpo_example.py \\
        --ckpt-output-dir ./grpo_output \\
        --num-iterations 2 --tasks-per-iteration 5
"""

import argparse
import os
import sys
import time
from datetime import datetime

from training_hub import lora_grpo


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

default_model_path = "Qwen/Qwen3-4B"
default_lora_r = 16
default_lora_alpha = 8
default_learning_rate = 1e-5
default_num_iterations = 15
default_group_size = 8
default_tasks_per_iteration = 100
default_data_path = "Agent-Ark/Toucan-1.5M"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LoRA + GRPO Training (Reinforcement Learning from Verifiable Rewards)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (2 iterations, 5 tasks each)
  python lora_grpo_example.py --ckpt-output-dir ./grpo_test --num-iterations 2 --tasks-per-iteration 5

  # Full training run
  python lora_grpo_example.py --ckpt-output-dir ./grpo_output --num-iterations 15 --tasks-per-iteration 100

  # Granite MoE model (needs higher LoRA rank)
  python lora_grpo_example.py --model-path ibm-granite/granite-4.0-h-tiny --ckpt-output-dir ./grpo_granite --lora-r 128 --lora-alpha 256
        """
    )

    # Required
    parser.add_argument('--ckpt-output-dir', required=True,
                        help='Directory to save checkpoints and results')

    # Model
    parser.add_argument('--model-path', default=default_model_path,
                        help=f'HuggingFace model ID or local path (default: {default_model_path})')

    # Dataset
    parser.add_argument('--data-path', default=default_data_path,
                        help=f'HuggingFace dataset ID or local JSON/JSONL (default: {default_data_path})')
    parser.add_argument('--data-config', default='Qwen3',
                        help='HuggingFace dataset config (default: Qwen3)')
    parser.add_argument('--n-train', type=int, default=5000,
                        help='Number of training samples (default: 5000)')

    # GRPO hyperparameters
    parser.add_argument('--num-iterations', type=int, default=default_num_iterations,
                        help=f'Number of GRPO iterations (default: {default_num_iterations})')
    parser.add_argument('--group-size', type=int, default=default_group_size,
                        help=f'Rollouts per task for advantage estimation (default: {default_group_size})')
    parser.add_argument('--tasks-per-iteration', type=int, default=default_tasks_per_iteration,
                        help=f'Tasks per iteration (default: {default_tasks_per_iteration})')
    parser.add_argument('--learning-rate', type=float, default=default_learning_rate,
                        help=f'Learning rate (default: {default_learning_rate})')
    parser.add_argument('--concurrency', type=int, default=32,
                        help='Max concurrent rollouts (default: 32)')

    # LoRA configuration
    parser.add_argument('--lora-r', type=int, default=default_lora_r,
                        help=f'LoRA rank (default: {default_lora_r})')
    parser.add_argument('--lora-alpha', type=int, default=default_lora_alpha,
                        help=f'LoRA alpha parameter (default: {default_lora_alpha})')

    # vLLM
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.45,
                        help='GPU memory fraction for vLLM (default: 0.45)')

    # Logging
    parser.add_argument('--wandb-project', help='Weights & Biases project name')
    parser.add_argument('--mlflow-tracking-uri', help='MLflow tracking URI')
    parser.add_argument('--mlflow-experiment-name', help='MLflow experiment name')

    # Utility
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configuration and exit without training')

    return parser.parse_args()


def main():
    args = parse_arguments()

    rollouts_per_iter = args.tasks_per_iteration * args.group_size
    total_rollouts = rollouts_per_iter * args.num_iterations

    # Print configuration
    print("LoRA + GRPO Training (RLVR)")
    print("=" * 60)
    print(f"Model:              {args.model_path}")
    print(f"Dataset:            {args.data_path} ({args.data_config})")
    print(f"Output:             {args.ckpt_output_dir}")
    print()
    print("GRPO Configuration:")
    print(f"  Iterations:       {args.num_iterations}")
    print(f"  Tasks/iteration:  {args.tasks_per_iteration}")
    print(f"  Group size:       {args.group_size}")
    print(f"  Rollouts/iter:    {rollouts_per_iter}")
    print(f"  Total rollouts:   {total_rollouts}")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  Concurrency:      {args.concurrency}")
    print()
    print("LoRA Configuration:")
    print(f"  Rank (r):         {args.lora_r}")
    print(f"  Alpha:            {args.lora_alpha}")
    print(f"  GPU mem util:     {args.gpu_memory_utilization}")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run mode - configuration validated, exiting without training")
        return

    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        result = lora_grpo(
            model_path=args.model_path,
            ckpt_output_dir=args.ckpt_output_dir,
            data_path=args.data_path,
            data_config=args.data_config,
            n_train=args.n_train,
            num_iterations=args.num_iterations,
            group_size=args.group_size,
            tasks_per_iteration=args.tasks_per_iteration,
            learning_rate=args.learning_rate,
            concurrency=args.concurrency,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            gpu_memory_utilization=args.gpu_memory_utilization,
            wandb_project=args.wandb_project,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
        )

        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print("GRPO training completed successfully!")
        print(f"Training time:     {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"Total rollouts:    {result['total_rollouts']}")
        print(f"Final reward:      {result['final_mean_reward']:.3f}")
        print(f"Reward history:    {[f'{r:.3f}' for r in result['reward_history']]}")
        print(f"Checkpoint saved:  {result['checkpoint_path']}")
        print("=" * 60)

    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print(f"GRPO training failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  - Ensure openpipe-art is installed: pip install openpipe-art")
        print("  - Reduce GPU memory usage: --gpu-memory-utilization 0.35")
        print("  - Reduce concurrency: --concurrency 8")
        print("  - Try fewer tasks: --tasks-per-iteration 10 --num-iterations 2")
        sys.exit(1)


if __name__ == "__main__":
    main()
