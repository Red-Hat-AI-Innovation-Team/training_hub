# `lora_grpo()` - LoRA + GRPO Training

> Convenience function for adapter-based RLVR — LoRA + GRPO training on tool-calling agents using reinforcement learning from verifiable rewards.

?> **New to LoRA + GRPO?** See the [Algorithm Guide](../../algorithms/lora_grpo.md) for overview and quick start.

## Signature

```python
def lora_grpo(
    model_path: str,
    ckpt_output_dir: str,
    # Data source
    data_path: Optional[str] = None,
    data_config: str = "Qwen3",
    n_train: int = 5000,
    n_val: int = 500,
    # Custom rollout
    rollout_fn: Optional[Callable] = None,
    tasks: Optional[List[Any]] = None,
    reward_fn: Optional[Callable] = None,
    # GRPO hyperparameters
    num_iterations: int = 15,
    group_size: int = 8,
    tasks_per_iteration: int = 100,
    learning_rate: float = 1e-5,
    temperature: float = 0.7,
    max_tokens: int = 512,
    concurrency: int = 32,
    # LoRA configuration
    lora_r: int = 16,
    lora_alpha: int = 8,
    target_modules: Optional[List[str]] = None,
    max_grad_norm: float = 0.1,
    # vLLM configuration
    gpu_memory_utilization: float = 0.45,
    # Multi-GPU
    n_gpus: int = 1,
    tensor_parallel_size: int = 1,
    # Backend
    backend: str = "art",
    # Experiment tracking
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
    **kwargs,
) -> dict
```

## Quick Example

```python
from training_hub import lora_grpo

result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./tool_call_traces.jsonl",
    ckpt_output_dir="./output",
    backend="art",
    lora_r=32,
    num_iterations=15,
)
```

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | HuggingFace model ID or local path to base model |
| `ckpt_output_dir` | `str` | Directory to save checkpoints and training results |

### Data Source

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | `str` | `None` | HuggingFace dataset ID (e.g., `Agent-Ark/Toucan-1.5M`) or local JSON/JSONL path |
| `data_config` | `str` | `"Qwen3"` | HuggingFace dataset config name |
| `n_train` | `int` | `5000` | Number of training samples to load |
| `n_val` | `int` | `500` | Number of validation samples to load |

### Custom Rollout

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rollout_fn` | `Callable` | `None` | Async function `(model, task) -> art.Trajectory`. Must be a top-level function (not lambda/closure). |
| `tasks` | `list` | `None` | List of task objects passed to `rollout_fn` |
| `reward_fn` | `Callable` | `None` | Custom reward function `(response, name, args) -> float` |

### GRPO Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | `int` | `15` | Number of GRPO training iterations (ART) or epochs (verl) |
| `group_size` | `int` | `8` | Rollouts per task for advantage estimation |
| `tasks_per_iteration` | `int` | `100` | Tasks sampled per iteration |
| `learning_rate` | `float` | `1e-5` | Learning rate |
| `temperature` | `float` | `0.7` | Sampling temperature for rollouts |
| `max_tokens` | `int` | `512` | Maximum response tokens per rollout |
| `concurrency` | `int` | `32` | Maximum concurrent rollouts |

### LoRA Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_r` | `int` | `16` | LoRA rank |
| `lora_alpha` | `int` | `8` | LoRA alpha scaling parameter |
| `target_modules` | `list[str]` | `None` | Modules to apply LoRA to (default: auto-detect) |
| `max_grad_norm` | `float` | `0.1` | Gradient clipping norm |

### Multi-GPU Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_gpus` | `int` | `1` | Number of GPUs (verl backend) |
| `tensor_parallel_size` | `int` | `1` | Tensor parallelism size for vLLM inference |
| `gpu_memory_utilization` | `float` | `0.45` | GPU memory fraction for vLLM |

### Backend Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"art"` | Backend to use: `"art"` (single-GPU) or `"verl"` (multi-GPU) |

### Experiment Tracking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wandb_project` | `str` | `None` | Weights & Biases project name. Enables W&B logging. |
| `wandb_entity` | `str` | `None` | Weights & Biases team/entity name. |
| `wandb_run_name` | `str` | `None` | Weights & Biases run name. |
| `mlflow_tracking_uri` | `str` | `None` | MLflow tracking server URI. Enables MLflow logging (verl backend only). |
| `mlflow_experiment_name` | `str` | `None` | MLflow experiment name (verl backend only). |
| `mlflow_run_name` | `str` | `None` | MLflow run name (verl backend only). |

> **Note:** The ART backend supports W&B natively (auto-detects `WANDB_API_KEY`). MLflow is only supported with the verl backend.

## Returns

`dict` with keys:
- `status` — `"success"` on completion
- `checkpoint_path` — Path to saved checkpoints
- `reward_history` — List of per-iteration mean rewards
- `full_match_history` — List of per-iteration full match rates
- `total_time_seconds` — Total training wall time
- `total_rollouts` — Total number of rollouts generated

## Examples

### Single-Turn Tool-Call Training

```python
from training_hub import lora_grpo

result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="Agent-Ark/Toucan-1.5M",
    ckpt_output_dir="./toucan_grpo",
    num_iterations=15,
    group_size=8,
    tasks_per_iteration=100,
    lora_r=32,
    lora_alpha=64,
)
```

### Multi-Turn Traces (e.g., tau-bench)

```python
result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./tau_retail_traces.jsonl",
    ckpt_output_dir="./tau_grpo",
    num_iterations=20,
    lora_r=32,
    lora_alpha=64,
    learning_rate=5e-6,
)
```

### Multi-GPU with verl

```python
result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./verl_output",
    backend="verl",
    n_gpus=4,
    num_iterations=3,
    group_size=4,
    lora_r=32,
    lora_alpha=64,
)
```

### With Experiment Tracking

```python
# W&B tracking (both backends)
result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
    wandb_project="my-grpo-project",
    wandb_entity="my-team",
    wandb_run_name="qwen3-grpo-run",
)

# MLflow tracking (verl backend only)
result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
    backend="verl",
    n_gpus=4,
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="grpo-experiments",
    mlflow_run_name="qwen3-grpo-run",
)
```

## See Also

- [LoRA + GRPO Algorithm Guide](/algorithms/lora_grpo) — Conceptual overview and detailed usage
- [ART Backend](/api/backends/art-grpo) — Single-GPU backend details
- [verl Backend](/api/backends/verl) — Multi-GPU backend details
- [`lora_sft()`](/api/functions/lora_sft) — LoRA + SFT (supervised, non-RL)
