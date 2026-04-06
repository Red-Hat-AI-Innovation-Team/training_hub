# verl Backend (GRPO)

> Multi-GPU distributed LoRA + GRPO training using [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLMs).

## Overview

**Class:** `VeRLLoRAGRPOBackend`

**Algorithm Support:** LoRA + GRPO

**Package:** `verl`

**Status:** Implemented

The verl backend uses FSDP for distributed LoRA training across multiple GPUs and vLLM for parallel rollout generation, orchestrated by Ray. It supports scaling to 70B+ models and provides epoch-based training over the full dataset.

## Features

- FSDP-sharded LoRA training across multiple GPUs
- vLLM rollout generation with co-located weight syncing
- Ray-based orchestration for distributed workers
- Epoch-based training for systematic data coverage
- Supports GRPO with group-based advantage estimation
- Automatic checkpoint saving and resume
- Experiment tracking via W&B, MLflow, TensorBoard, and more

## Usage

### Via Convenience Function

```python
from training_hub import lora_grpo

result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
    backend="verl",
    n_gpus=4,
    lora_r=32,
    lora_alpha=64,
    num_iterations=3,
)
```

### Via Class-Based API

```python
from training_hub import create_algorithm

algo = create_algorithm("lora_grpo", "verl")
result = algo.train(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
    n_gpus=4,
)
```

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_gpus` | `int` | `1` | Number of GPUs for distributed training |
| `tensor_parallel_size` | `int` | `1` | Tensor parallelism for vLLM inference |
| `gpu_memory_utilization` | `float` | `0.35` | GPU memory fraction for vLLM |

## Memory Considerations

verl co-locates FSDP training and vLLM inference on the same GPUs. Key memory factors:

- **Logits tensor**: `micro_batch * seq_len * vocab_size * 2 bytes` — can be large for models with large vocabularies (e.g., Qwen3's 152K vocab). Reduce micro batch size if OOM occurs.
- **FSDP sharding**: Model parameters are sharded across GPUs but gathered during forward/backward passes.
- **gpu_memory_utilization**: Set lower (0.3-0.4) to leave room for training. Default 0.35 works well for 4B models on H100.

## Checkpoints

verl saves checkpoints in:
```
ckpt_output_dir/checkpoints/global_step_N/actor/lora_adapter/
```

Each checkpoint contains `adapter_config.json` and `adapter_model.safetensors`, compatible with HuggingFace PEFT for loading.

## Limitations

- Requires Ray for distributed orchestration
- Tool calls generated as text (not structured API), parsed via regex
- `num_iterations` maps to epochs (full passes over the dataset), not random-sampling iterations
- Custom `rollout_fn` not supported (verl manages its own rollout pipeline)

## See Also

- [LoRA + GRPO Algorithm Guide](/algorithms/lora_grpo)
- [ART Backend](/api/backends/art-grpo) — Single-GPU alternative
- [`lora_grpo()` Function Reference](/api/functions/lora_grpo)
