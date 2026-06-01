#!/usr/bin/env python3
"""
ITS Hub + GRPO Training Example

Uses ITS Hub inference-time scaling algorithms (SelfConsistency, BestOfN)
as the rollout mechanism for GRPO training. The ITS Hub algorithm generates
multiple candidate responses per prompt, selects the best one, and builds
a training trajectory from it.

Requires: its_hub[lm], openpipe-art

Example usage:
    # SelfConsistency rollout (default)
    python its_rollout_example.py \\
        --ckpt-output-dir ./its_grpo_output \\
        --data-path ./my_data.jsonl

    # BestOfN rollout (uses training model as judge)
    python its_rollout_example.py \\
        --ckpt-output-dir ./its_grpo_output \\
        --data-path ./my_data.jsonl \\
        --algorithm best_of_n

    # Quick test
    python its_rollout_example.py \\
        --ckpt-output-dir ./its_grpo_output \\
        --data-path ./my_data.jsonl \\
        --num-iterations 1 --prompt-batch-size 5 --budget 3
"""

import argparse
import sys
import time
from datetime import datetime

from training_hub import lora_grpo
from training_hub.algorithms.its_rollout import ITSRollout


# --- Algorithm factories (module-level for picklability) ---

def make_self_consistency(lm):
    from its_hub import SelfConsistency
    return SelfConsistency()


def make_best_of_n(lm):
    from its_hub import BestOfN, LLMJudge
    judge = LLMJudge(lm=lm, judge_prompt="Rate the accuracy of this response from 1 to 10. Reply with just the number.")
    return BestOfN(orm=judge)


ALGORITHM_FACTORIES = {
    "self_consistency": make_self_consistency,
    "best_of_n": make_best_of_n,
}


# --- Reward function (module-level for picklability) ---

def contains_answer_reward(response_text, task):
    """Simple reward: 1.0 if ground truth appears in response, else 0.0."""
    ground_truth = task.get("answer", task.get("ground_truth", ""))
    if not ground_truth:
        return 0.0
    return 1.0 if ground_truth.lower() in response_text.lower() else 0.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ITS Hub + GRPO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data format (JSONL, one object per line):
  {"messages": [{"role": "user", "content": "..."}], "answer": "expected answer"}

The 'messages' field is sent to the model. The 'answer' field is used by
the reward function to score responses.
        """
    )

    parser.add_argument('--ckpt-output-dir', required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--data-path', required=True,
                        help='JSONL file with messages + answer fields')

    # ITS Hub configuration
    parser.add_argument('--algorithm', default='self_consistency',
                        choices=list(ALGORITHM_FACTORIES.keys()),
                        help='ITS Hub algorithm (default: self_consistency)')
    parser.add_argument('--budget', type=int, default=8,
                        help='Candidates per rollout (default: 8)')
    parser.add_argument('--max-concurrency', type=int, default=32,
                        help='Max concurrent LM calls within each ITS rollout (default: 32)')

    # Model
    parser.add_argument('--model-path', default='Qwen/Qwen3-4B',
                        help='HuggingFace model ID (default: Qwen/Qwen3-4B)')

    # GRPO hyperparameters
    parser.add_argument('--num-iterations', type=int, default=10,
                        help='Training iterations (default: 10)')
    parser.add_argument('--group-size', type=int, default=8,
                        help='Rollouts per prompt (default: 8)')
    parser.add_argument('--prompt-batch-size', type=int, default=50,
                        help='Prompts per iteration (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--concurrency', type=int, default=16,
                        help='Max concurrent rollouts (default: 16)')

    # LoRA
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank (default: 16)')
    parser.add_argument('--lora-alpha', type=int, default=8,
                        help='LoRA alpha (default: 8)')

    # vLLM
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.3,
                        help='vLLM GPU memory fraction (default: 0.3)')

    parser.add_argument('--dry-run', action='store_true',
                        help='Print configuration and exit')

    return parser.parse_args()


def load_tasks(data_path):
    """Load tasks from JSONL file."""
    import json
    tasks = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def main():
    args = parse_arguments()

    tasks = load_tasks(args.data_path)
    factory = ALGORITHM_FACTORIES[args.algorithm]

    rollout = ITSRollout(
        algorithm_factory=factory,
        budget=args.budget,
        reward_fn=contains_answer_reward,
        max_concurrency=args.max_concurrency,
    )

    rollouts_per_iter = args.prompt_batch_size * args.group_size
    candidates_per_iter = rollouts_per_iter * args.budget

    print("ITS Hub + GRPO Training")
    print("=" * 60)
    print(f"Model:              {args.model_path}")
    print(f"Data:               {args.data_path} ({len(tasks)} tasks)")
    print(f"Algorithm:          {args.algorithm} (budget={args.budget})")
    print(f"Backend:            art (single GPU)")
    print()
    print("GRPO Configuration:")
    print(f"  Iterations:       {args.num_iterations}")
    print(f"  Prompts/batch:    {args.prompt_batch_size}")
    print(f"  Group size:       {args.group_size}")
    print(f"  Rollouts/iter:    {rollouts_per_iter}")
    print(f"  LM calls/iter:    {candidates_per_iter} ({rollouts_per_iter} x {args.budget})")
    print(f"  Outer concurrency:{args.concurrency}")
    print(f"  Inner concurrency:{args.max_concurrency}")
    print(f"  Learning rate:    {args.learning_rate}")
    print()
    print("LoRA Configuration:")
    print(f"  Rank (r):         {args.lora_r}")
    print(f"  Alpha:            {args.lora_alpha}")
    print(f"  GPU mem util:     {args.gpu_memory_utilization}")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run — exiting without training")
        return

    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    try:
        result = lora_grpo(
            model_path=args.model_path,
            ckpt_output_dir=args.ckpt_output_dir,
            rollout_fn=rollout,
            tasks=tasks,
            backend="art",
            num_iterations=args.num_iterations,
            group_size=args.group_size,
            prompt_batch_size=args.prompt_batch_size,
            learning_rate=args.learning_rate,
            concurrency=args.concurrency,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print("Training completed!")
        print(f"Time:              {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"Total rollouts:    {result.get('total_rollouts', '?')}")
        print(f"Final reward:      {result.get('final_mean_reward', '?')}")
        if 'reward_history' in result:
            print(f"Reward history:    {[f'{r:.3f}' for r in result['reward_history']]}")
        print("=" * 60)

    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print(f"Training failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  - Ensure its_hub is installed: pip install its_hub[lm]")
        print("  - Ensure openpipe-art is installed")
        print("  - Reduce budget: --budget 3")
        print("  - Reduce concurrency: --concurrency 8 --max-concurrency 8")
        print("  - Reduce GPU memory: --gpu-memory-utilization 0.2")
        sys.exit(1)


if __name__ == "__main__":
    main()
