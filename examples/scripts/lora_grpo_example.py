#!/usr/bin/env python3
"""
LoRA + GRPO Training Example

Trains LoRA adapters using Group Relative Policy Optimization (GRPO) with
reinforcement learning from verifiable rewards (RLVR).

Defaults to the verl backend with Dr. GRPO (no reference model). Set
--n-gpus to scale across GPUs; use --backend art for single-GPU training.

Example usage:
    # Multi-GPU Dr. GRPO with verl (default)
    python lora_grpo_example.py \\
        --ckpt-output-dir ./grpo_output \\
        --n-gpus 4

    # Single-GPU with ART backend
    python lora_grpo_example.py \\
        --ckpt-output-dir ./grpo_output \\
        --backend art

    # Quick test run
    python lora_grpo_example.py \\
        --ckpt-output-dir ./grpo_output \\
        --num-iterations 2 --prompt-batch-size 10

    # Custom reward function
    python lora_grpo_example.py \\
        --ckpt-output-dir ./grpo_output \\
        --data-path ./my_data.jsonl
"""

import argparse
import os
import sys
import time
from datetime import datetime

from training_hub import lora_grpo


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LoRA + GRPO Training (RLVR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (2 iterations, 10 prompts each, verl backend)
  python lora_grpo_example.py --ckpt-output-dir ./test --num-iterations 2 --prompt-batch-size 10 --n-gpus 4

  # Full training run with ART (single GPU)
  python lora_grpo_example.py --ckpt-output-dir ./output --backend art

  # Standard GRPO with KL (instead of Dr. GRPO)
  python lora_grpo_example.py --ckpt-output-dir ./output --no-dr-grpo --n-gpus 4
        """
    )

    # Required
    parser.add_argument('--ckpt-output-dir', required=True,
                        help='Directory to save checkpoints and results')

    # Model
    parser.add_argument('--model-path', default='Qwen/Qwen3-4B',
                        help='HuggingFace model ID or local path (default: Qwen/Qwen3-4B)')

    # Dataset
    parser.add_argument('--data-path', default='Agent-Ark/Toucan-1.5M',
                        help='HuggingFace dataset ID or local JSON/JSONL (default: Agent-Ark/Toucan-1.5M)')
    parser.add_argument('--data-config', default='Qwen3',
                        help='HuggingFace dataset config (default: Qwen3)')
    parser.add_argument('--n-train', type=int, default=5000,
                        help='Number of training samples (default: 5000)')

    # Backend
    parser.add_argument('--backend', default='verl', choices=['verl', 'art'],
                        help='Training backend (default: verl)')
    parser.add_argument('--n-gpus', type=int, default=1,
                        help='Number of GPUs for verl backend (default: 1)')
    parser.add_argument('--no-dr-grpo', action='store_true',
                        help='Use standard GRPO+KL instead of Dr. GRPO (verl only)')

    # GRPO hyperparameters
    parser.add_argument('--num-iterations', type=int, default=15,
                        help='Number of GRPO iterations/epochs (default: 15)')
    parser.add_argument('--group-size', type=int, default=8,
                        help='Rollouts per prompt for advantage estimation (default: 8)')
    parser.add_argument('--prompt-batch-size', type=int, default=100,
                        help='Number of unique prompts per step (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--concurrency', type=int, default=32,
                        help='Max concurrent rollouts — ART only (default: 32)')

    # LoRA configuration
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank (default: 16)')
    parser.add_argument('--lora-alpha', type=int, default=8,
                        help='LoRA alpha parameter (default: 8)')

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

    rollouts_per_iter = args.prompt_batch_size * args.group_size
    total_rollouts = rollouts_per_iter * args.num_iterations
    use_dr_grpo = not args.no_dr_grpo

    print("LoRA + GRPO Training (RLVR)")
    print("=" * 60)
    print(f"Model:              {args.model_path}")
    print(f"Dataset:            {args.data_path} ({args.data_config})")
    print(f"Output:             {args.ckpt_output_dir}")
    print(f"Backend:            {args.backend}" +
          (f" ({args.n_gpus} GPUs, {'Dr. GRPO' if use_dr_grpo else 'GRPO+KL'})"
           if args.backend == 'verl' else " (single GPU)"))
    print()
    print("GRPO Configuration:")
    print(f"  Iterations:       {args.num_iterations}")
    print(f"  Prompts/batch:    {args.prompt_batch_size}")
    print(f"  Group size:       {args.group_size}")
    print(f"  Rollouts/iter:    {rollouts_per_iter}")
    print(f"  Total rollouts:   {total_rollouts}")
    print(f"  Learning rate:    {args.learning_rate}")
    print()
    print("LoRA Configuration:")
    print(f"  Rank (r):         {args.lora_r}")
    print(f"  Alpha:            {args.lora_alpha}")
    print(f"  GPU mem util:     {args.gpu_memory_utilization}")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run — configuration validated, exiting without training")
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
            backend=args.backend,
            n_gpus=args.n_gpus,
            use_dr_grpo=use_dr_grpo,
            num_iterations=args.num_iterations,
            group_size=args.group_size,
            prompt_batch_size=args.prompt_batch_size,
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
        print("Training completed successfully!")
        print(f"Training time:     {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"Total rollouts:    {result.get('total_rollouts', '?')}")
        print(f"Final reward:      {result.get('final_mean_reward', '?')}")
        if 'reward_history' in result:
            print(f"Reward history:    {[f'{r:.3f}' for r in result['reward_history']]}")
        if 'checkpoint_path' in result:
            print(f"Checkpoint saved:  {result['checkpoint_path']}")
        print("=" * 60)

    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print(f"Training failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  - For verl: ensure verl>=0.7 and vllm are installed")
        print("  - For ART: ensure openpipe-art is installed")
        print("  - Reduce GPU memory: --gpu-memory-utilization 0.3")
        print("  - Try fewer prompts: --prompt-batch-size 10 --num-iterations 2")
        print("  - For verl, ensure prompt_batch_size * group_size is divisible by n_gpus * 4")
        sys.exit(1)


if __name__ == "__main__":
    main()
