# GRPO — Full Fine-Tuning with Reinforcement Learning from Verifiable Rewards

Full-parameter GRPO training via the verl backend. Trains all model weights using Group Relative Policy Optimization, without LoRA adapters.

> For LoRA-based GRPO (parameter-efficient), see [LoRA + GRPO](/algorithms/lora_grpo).

## When to Use

- You want maximum model capacity and are willing to train all parameters
- You have enough GPUs to fit the full model in FSDP (8B models need 8x H100 80GB)
- You want a standalone fine-tuned model (no base model + adapter at inference)

## Quick Start

```python
from training_hub import grpo

result = grpo(
    model_path="Qwen/Qwen3-8B",
    data_path="training_data.jsonl",
    ckpt_output_dir="./grpo_output",
    n_gpus=8,
    num_iterations=8,
    group_size=8,
    prompt_batch_size=48,
)
```

## Backend

Only the **verl** backend is supported. Full fine-tuning requires FSDP for multi-GPU parameter sharding, which the ART backend does not support.

## Comparison with LoRA + GRPO

| | LoRA + GRPO | GRPO (Full FT) |
|--|-------------|----------------|
| Trainable params | ~1-2% (adapter only) | 100% (all weights) |
| Checkpoint size | ~1-2 GB | ~16 GB (8B model) |
| Serving | Base model + adapter | Standalone model |
| Overfitting risk | Lower | Higher (may peak early) |
| Memory | Lower | Higher |

Despite a significant drop in trainable parameters, we find that in most cases the adapter-based approach sees similar or stronger quality, while providing more flexibility in deployment and agentic systems. Depending on the use-case, either method will carry different benefits (capacity vs flexibility).

## Data Format

Uses the same data formats as [LoRA + GRPO](/algorithms/lora_grpo) — tool-call traces or generic question/ground_truth JSONL.

## Checkpoints

Full fine-tuning produces FSDP sharded checkpoints. Use verl's model merger to consolidate into HuggingFace format:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir ./checkpoints/global_step_76/actor \
  --target_dir ./merged_checkpoint \
  --trust-remote-code
```

## API Reference

See [`grpo()`](/api/functions/grpo) for the full parameter reference.
