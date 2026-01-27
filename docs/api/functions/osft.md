# `osft()` - Orthogonal Subspace Fine-Tuning

> Convenience function for continual learning with Orthogonal Subspace Fine-Tuning (OSFT), enabling model customization without catastrophic forgetting.

?> **New to OSFT?** See the [OSFT Algorithm Guide](../../algorithms/osft.md) for a conceptual overview and quick start guide.

## Signature

```python
def osft(
    model_path: str,
    data_path: str,
    unfreeze_rank_ratio: float,
    effective_batch_size: int,
    max_tokens_per_gpu: int,
    max_seq_len: int,
    learning_rate: float,
    ckpt_output_dir: str,
    data_output_dir: str | None = None,
    backend: str = "mini-trainer",
    target_patterns: list[str] | None = None,
    seed: int | None = None,
    use_liger: bool | None = None,
    use_processed_dataset: bool | None = None,
    unmask_messages: bool | None = None,
    is_pretraining: bool | None = None,
    block_size: int | None = None,
    document_column_name: str | None = None,
    lr_scheduler: str | None = None,
    warmup_steps: int | None = None,
    lr_scheduler_kwargs: dict[str, str] | None = None,
    checkpoint_at_epoch: bool | None = None,
    save_final_checkpoint: bool | None = None,
    num_epochs: int | None = None,
    nproc_per_node: int | None = None,
    nnodes: int | None = None,
    node_rank: int | None = None,
    rdzv_id: int | None = None,
    rdzv_endpoint: str | None = None,
    **kwargs
) -> Any
```

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Local path or HuggingFace model ID of the model to fine-tune (e.g., `"meta-llama/Llama-3.1-8B-Instruct"`). |
| `data_path` | `str` | Path to training data in JSONL format. See [Data Formats](../data-formats.md). |
| `unfreeze_rank_ratio` | `float` | **Key OSFT parameter.** Controls how much of each weight matrix is unfrozen for training. Range: 0.0-1.0. Lower values (0.1-0.3) preserve more of the original model behavior. Higher values (0.5-0.8) allow more adaptation. |
| `effective_batch_size` | `int` | Effective batch size for the entire minibatch, will be distributed across all GPUs and potentially use gradient accumulation. |
| `max_tokens_per_gpu` | `int` | Maximum number of tokens that can be on a single GPU at once. |
| `max_seq_len` | `int` | Maximum sequence length in tokens. |
| `learning_rate` | `float` | Learning rate for model updates. |
| `ckpt_output_dir` | `str` | Directory where checkpoints and final model will be saved. |

### Optional Parameters

#### Backend Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"mini-trainer"` | Backend implementation to use. Currently only `"mini-trainer"` is supported for OSFT. |

#### Data Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_output_dir` | `str` | `None` | Directory for data processing outputs. |
| `use_processed_dataset` | `bool` | Backend default | Whether the data at `data_path` is already preprocessed. Set to `True` to skip preprocessing. |
| `unmask_messages` | `bool` | Backend default | If `True`, unmasks all messages during data processing (excluding system messages). |

#### Pretraining

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_pretraining` | `bool` | `None` | When `True`, enables pretraining mode for training on raw documents. |
| `block_size` | `int` | `None` | Required when `is_pretraining=True`. Token length of each document block. |
| `document_column_name` | `str` | `"document"` | Column containing raw documents when `is_pretraining=True`. |

#### OSFT-Specific Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_patterns` | `list[str]` | Backend default | List of regex patterns for selecting which model modules to apply OSFT to. If not specified, applies to all compatible linear layers. Example: `["q_proj", "v_proj", "k_proj"]`. |

#### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_epochs` | `int` | Backend default | Number of training epochs. Typical values: 1-10. |
| `warmup_steps` | `int` | Backend default | Number of warmup steps for the learning rate scheduler. |
| `lr_scheduler` | `str` | Backend default | PyTorch learning rate scheduler name (e.g., `"cosine"`, `"linear"`, `"constant"`). |
| `lr_scheduler_kwargs` | `dict[str, str]` | `None` | Additional keyword arguments for the learning rate scheduler. |
| `use_liger` | `bool` | Backend default | Whether to use Liger kernels for improved performance. Requires `liger-kernel` package. |
| `seed` | `int` | Backend default | Random seed for reproducibility. |

#### Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_at_epoch` | `bool` | Backend default | If `True`, saves a checkpoint at the end of each epoch. |
| `save_final_checkpoint` | `bool` | Backend default | If `True`, saves the final model checkpoint after training completes. |

#### Distributed Training (Multi-GPU / Multi-Node)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nproc_per_node` | `int` | Auto-detected | Number of processes (GPUs) per node. Auto-detects if not specified. |
| `nnodes` | `int` | `1` | Total number of nodes for multi-node training. |
| `node_rank` | `int` | `0` | Rank of this node (0 to `nnodes-1`). Node 0 is the master. |
| `rdzv_id` | `int` | Random | Unique job ID for rendezvous. Must be the same across all nodes. |
| `rdzv_endpoint` | `str` | Required for multi-node | Master node endpoint in format `"hostname:port"`. |

#### Logging Configuration

Loggers are automatically enabled when their configuration parameters are set:

| Logger | Enabled By            | Env Variable Fallback |
| ------ | --------------------- | --------------------- |
| MLflow | `mlflow_tracking_uri` | `MLFLOW_TRACKING_URI` |
| W&B    | `wandb_project`       | `WANDB_PROJECT`       |

| Parameter                | Type  | Default                    | Description                                         |
| ------------------------ | ----- | -------------------------- | --------------------------------------------------- |
| `mlflow_tracking_uri`    | `str` | `MLFLOW_TRACKING_URI` env  | MLflow tracking server URI. Enables MLflow logging. |
| `mlflow_experiment_name` | `str` | `MLFLOW_EXPERIMENT_NAME` env | MLflow experiment name.                           |
| `mlflow_run_name`        | `str` | `None`                     | MLflow run name.                                    |
| `wandb_project`          | `str` | `WANDB_PROJECT` env        | W&B project name. Enables W&B logging.              |
| `wandb_entity`           | `str` | `WANDB_ENTITY` env         | W&B team/entity.                                    |
| `wandb_run_name`         | `str` | `None`                     | W&B run name.                                       |

> **Note:** OSFT does not support TensorBoard logging.

#### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `**kwargs` | `Any` | - | Additional backend-specific parameters passed to the backend implementation. |

## Returns

**Type:** `any`

Returns the training result from the backend. For the `mini-trainer` backend, this typically includes:
- Path to the final checkpoint
- Training metrics and statistics
- Model state information

## Raises

- **`ValueError`**: If required parameters are missing or invalid (e.g., `unfreeze_rank_ratio` out of range).
- **`RuntimeError`**: If training fails or the backend is not available.
- **`FileNotFoundError`**: If `model_path` or `data_path` do not exist.

## Examples

### Basic Continual Learning

Adapt a model to a new domain without forgetting its original capabilities:

```python
from training_hub import osft

result = osft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./medical_domain_data.jsonl",
    ckpt_output_dir="./checkpoints/medical_osft",
    unfreeze_rank_ratio=0.25,  # Preserve 75% of original behavior
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=5e-6,
    num_epochs=3
)
```

### Multi-GPU Training with Custom Patterns

Train only specific attention layers:

```python
from training_hub import osft

result = osft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./legal_domain_data.jsonl",
    ckpt_output_dir="./checkpoints/legal_osft",
    unfreeze_rank_ratio=0.3,
    effective_batch_size=32,
    max_tokens_per_gpu=4096,
    max_seq_len=2048,
    learning_rate=2e-5,
    nproc_per_node=4,  # 4 GPUs
    target_patterns=["q_proj", "v_proj", "k_proj", "o_proj"],  # Only attention layers
    use_liger=True,
    num_epochs=5,
    checkpoint_at_epoch=True
)
```

### High Adaptation (More Customization)

When you need more significant changes to the model:

```python
from training_hub import osft

result = osft(
    model_path="microsoft/Phi-4-mini-instruct",
    data_path="./specialized_task_data.jsonl",
    ckpt_output_dir="./checkpoints/phi_specialized",
    unfreeze_rank_ratio=0.7,  # More adaptation, less preservation
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=1e-5,
    num_epochs=10,
    warmup_steps=100,
    lr_scheduler="cosine"
)
```

### Multi-Node Distributed Training

**On master node (rank 0):**

Run this code on the master node. The `rdzv_endpoint` uses `0.0.0.0` to listen on all network interfaces for incoming connections from worker nodes.

```python
from training_hub import osft

result = osft(
    model_path="./models/gpt-oss-20b",
    data_path="./large_continual_dataset.jsonl",
    ckpt_output_dir="./checkpoints/multi_node_osft",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=64,
    max_tokens_per_gpu=2048,
    max_seq_len=2048,
    learning_rate=5e-6,
    nnodes=3,
    node_rank=0,
    nproc_per_node=8,
    rdzv_id=98765,
    rdzv_endpoint="0.0.0.0:29500",  # Listen on all interfaces
    num_epochs=5
)
```

**On worker nodes (ranks 1-2):**

Run this code on each worker node, replacing `192.168.1.100` with the actual IP address of your rank 0 node. Change `node_rank` to 1, 2, etc. for each worker.

```python
from training_hub import osft

result = osft(
    model_path="./models/gpt-oss-20b",
    data_path="./large_continual_dataset.jsonl",
    ckpt_output_dir="./checkpoints/multi_node_osft",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=64,
    max_tokens_per_gpu=2048,
    max_seq_len=2048,
    learning_rate=5e-6,
    nnodes=3,
    node_rank=1,  # Change to 2 for other workers
    nproc_per_node=8,
    rdzv_id=98765,
    rdzv_endpoint="192.168.1.100:29500",  # IP address of rank 0 node
    num_epochs=5
)
```

## Understanding `unfreeze_rank_ratio`

The `unfreeze_rank_ratio` parameter is the key to OSFT's continual learning capabilities:

- **0.1 - 0.2**: Minimal adaptation, maximum preservation. Use when you want to add a small amount of new knowledge while keeping the model's original behavior almost unchanged.
- **0.25 - 0.3**: Balanced approach (recommended for most cases). Good trade-off between adaptation and preservation.
- **0.4 - 0.6**: More significant adaptation. Use when the new task requires substantial changes but you still want some preservation.
- **0.7 - 0.9**: Heavy adaptation, minimal preservation. Similar to full fine-tuning but with some orthogonality constraints.

?> **Recommendation**: Start with `unfreeze_rank_ratio=0.25` and adjust based on your evaluation results.

## Implementation Notes

?> **Tip**: OSFT is particularly effective for continual learning scenarios where you want to add new capabilities without forgetting existing ones.

!> **Memory Warning**: If you encounter OOM errors, reduce `max_tokens_per_gpu` or `max_seq_len`.

?> **Performance Tip**: Enable `use_liger=True` for faster training when the `liger-kernel` package is installed.

> **Note**: OSFT is based on the paper "Orthogonal Subspace Fine-Tuning" by Nayak et al. (2025), [arXiv:2504.07097](https://arxiv.org/abs/2504.07097).

> **Note**: The `osft()` function is a convenience wrapper around [`OSFTAlgorithm`](/api/classes/OSFTAlgorithm). For advanced use cases, use the class-based API directly.

## See Also

- [**sft() Function**](/api/functions/sft) - Standard supervised fine-tuning
- [**OSFTAlgorithm Class**](/api/classes/OSFTAlgorithm) - Class-based API for more control
- [**Mini-Trainer Backend**](/api/backends/mini-trainer) - Backend implementation details
- [**Data Formats**](/api/data-formats) - Input data format specifications
- [**Distributed Training Guide**](/guides/distributed-training) - Multi-node setup guide
- [**OSFT Algorithm Overview**](/algorithms/osft) - Conceptual overview and theory
- [**OSFT Examples**](/examples/#orthogonal-subspace-fine-tuning-osft) - Notebooks and scripts

## Source

[View source on GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/osft.py)
