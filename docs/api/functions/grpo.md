# `grpo()`

Full-parameter GRPO training via the verl backend. Equivalent to `lora_grpo(..., lora_r=0, backend="verl")`.

## Signature

```python
from training_hub import grpo

result = grpo(
    model_path: str,
    ckpt_output_dir: str,
    # Data
    data_path: str = None,
    data_config: str = "Qwen3",
    n_train: int = 5000,
    n_val: int = 500,
    reward_fn: Callable = None,
    # GRPO
    num_iterations: int = 15,
    group_size: int = 8,
    prompt_batch_size: int = 100,
    learning_rate: float = 1e-5,
    temperature: float = 0.7,
    max_tokens: int = 512,
    max_prompt_length: int = 16384,
    # GPU
    gpu_memory_utilization: float = 0.45,
    n_gpus: int = 1,
    nnodes: int = 1,
    tensor_parallel_size: int = 1,
    # Algorithm
    use_dr_grpo: bool = True,
    # Tracking
    wandb_project: str = None,
    mlflow_tracking_uri: str = None,
    **kwargs,
)
```

## Quick Example

```python
from training_hub import grpo

result = grpo(
    model_path="Qwen/Qwen3-8B",
    data_path="./training_data.jsonl",
    ckpt_output_dir="./grpo_output",
    n_gpus=8,
    num_iterations=8,
)
```

## Parameters

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | HuggingFace model ID or local path |
| `ckpt_output_dir` | `str` | Directory for checkpoints and results |

### Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | `None` | Dataset path (HuggingFace ID or local JSONL) |
| `data_config` | `"Qwen3"` | HuggingFace dataset config |
| `n_train` | `5000` | Training samples to load |
| `n_val` | `500` | Validation samples to load |
| `reward_fn` | `None` | Custom reward function for generic data |

### GRPO Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_iterations` | `15` | Training epochs |
| `group_size` | `8` | Rollouts per prompt for advantage estimation |
| `prompt_batch_size` | `100` | Unique prompts per step |
| `learning_rate` | `1e-5` | Learning rate |
| `temperature` | `0.7` | Sampling temperature |
| `max_tokens` | `512` | Max response tokens |
| `max_prompt_length` | `16384` | Max prompt tokens (filtered) |

### GPU Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_gpus` | `1` | Number of GPUs |
| `nnodes` | `1` | Number of nodes |
| `gpu_memory_utilization` | `0.45` | vLLM GPU memory fraction |
| `tensor_parallel_size` | `1` | vLLM tensor parallelism |

## Differences from `lora_grpo()`

- No `lora_r`, `lora_alpha`, `target_modules`, `max_lora_rank` parameters
- No `backend` parameter (always `"verl"`)
- No `rollout_fn`, `tasks`, `concurrency` (custom rollout is ART-only)
- Produces full model checkpoints (~16GB for 8B) instead of LoRA adapters (~1-2GB)
- Requires `python -m verl.model_merger merge` to consolidate FSDP checkpoints

## Related

- [`lora_grpo()`](/api/functions/lora_grpo) — LoRA-based GRPO (parameter-efficient, supports ART and verl)
- [GRPO Algorithm Guide](/algorithms/grpo)
- [verl Backend](/api/backends/verl)
