# ART Backend (GRPO)

> Single-GPU LoRA + GRPO training using OpenPipe ART with Unsloth GRPO.

## Overview

**Class:** `ARTLoRAGRPOBackend`

**Algorithm Support:** LoRA + GRPO

**Package:** `openpipe-art`

**Status:** Implemented

The ART backend runs LoRA + GRPO training on a single GPU using [OpenPipe ART](https://github.com/OpenPipe/ART) (Agent Reinforcement Trainer) with co-located vLLM inference and [Unsloth](https://github.com/unslothai/unsloth) GRPO training via time-sharing. During rollout generation, vLLM serves the model with the current LoRA adapter. During training, vLLM sleeps while Unsloth trains the LoRA adapter using its optimized GRPO implementation. This cycle repeats each iteration.

## Features

- Co-located vLLM + Unsloth GRPO on a single GPU
- Structured tool-call generation via OpenAI-compatible API
- Built-in tool-call reward verification (`tool_call_reward`)
- Automatic checkpoint saving and resume
- Support for custom rollout functions and reward functions
- Weights & Biases experiment tracking (auto-detects `WANDB_API_KEY`)

## Usage

### Via Convenience Function

```python
from training_hub import lora_grpo

result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
    backend="art",
    lora_r=32,
    lora_alpha=64,
    num_iterations=15,
)
```

### Via Class-Based API

```python
from training_hub import create_algorithm

algo = create_algorithm("lora_grpo", "art")
result = algo.train(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
)
```

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_memory_utilization` | `float` | `0.45` | GPU memory fraction for vLLM |
| `art_model_name` | `str` | auto | Model name for ART registration |
| `art_project` | `str` | `"training-hub-grpo"` | ART project name |
| `concurrency` | `int` | `32` | Max concurrent rollouts |

## Limitations

- Single GPU only (no multi-GPU support)
- Best suited for models up to ~8B parameters
- Requires `openpipe-art` package

## See Also

- [LoRA + GRPO Algorithm Guide](/algorithms/lora_grpo)
- [verl Backend](/api/backends/verl) — Multi-GPU alternative
- [`lora_grpo()` Function Reference](/api/functions/lora_grpo)
