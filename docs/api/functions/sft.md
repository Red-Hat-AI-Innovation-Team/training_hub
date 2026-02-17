# `sft()` - Supervised Fine-Tuning

> Convenience function for running supervised fine-tuning on language models with support for single-node and multi-node distributed training.

?> **New to SFT?** See the [SFT Algorithm Guide](../../algorithms/sft.md) for a conceptual overview and quick start guide.

## Signature

```python
def sft(
    model_path: str,
    data_path: str,
    ckpt_output_dir: str,
    backend: str = "instructlab-training",
    num_epochs: Optional[int] = None,
    effective_batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    max_seq_len: Optional[int] = None,
    max_tokens_per_gpu: Optional[int] = None,
    data_output_dir: Optional[str] = None,
    save_samples: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    accelerate_full_state_at_epoch: Optional[bool] = None,
    checkpoint_at_epoch: Optional[bool] = None,
    is_pretraining: Optional[bool] = None,
    block_size: Optional[int] = None,
    document_column_name: Optional[str] = None,
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[int] = None,
    node_rank: Optional[int] = None,
    rdzv_id: Optional[int] = None,
    rdzv_endpoint: Optional[str] = None,
    **kwargs
) -> Any
```

## Quick Example

## Parameters

> **Note:** The parameter definitions below are inherited from the [`SFTAlgorithm.train()` method](../classes/SFTAlgorithm.md#train-kwargs---any), which is the authoritative source of truth for all SFT parameters.

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Path to the model to fine-tune. Can be a local directory path or a HuggingFace model ID (e.g., `"Qwen/Qwen2.5-7B-Instruct"`). |
| `data_path` | `str` | Path to training data. |
| `ckpt_output_dir` | `str` | Directory where model checkpoints will be saved during and after training. |
| `num_epochs` | `int` | Number of training epochs to run. |
| `effective_batch_size` | `int` | Effective batch size for the entire minibatch. This is the total number of samples processed before a gradient update. |
| `learning_rate` | `float` | Learning rate for the optimizer. |
| `max_seq_len` | `int` | Maximum sequence length in tokens. |
| `max_tokens_per_gpu` | `int` | Maximum number of tokens that can be on a single GPU at once. |

### Optional Parameters

#### Backend Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"instructlab-training"` | Backend implementation to use for training. Currently only `"instructlab-training"` is supported for SFT. |

#### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup_steps` | `int` | Backend default | Number of warmup steps for the learning rate scheduler. |

#### Data Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_output_dir` | `str` | `None` | Directory to save processed training data. If not specified, processed data may be stored in a temporary location. |

#### Pretraining

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_pretraining` | `bool` | `None` | When `True`, enables pretraining mode for training on raw documents. |
| `block_size` | `int` | `None` | Required when `is_pretraining=True`. Token length of each document block. |
| `document_column_name` | `str` | `"document"` | Column containing raw documents when `is_pretraining=True`. |

#### Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_at_epoch` | `bool` | Backend default | If `True`, saves a checkpoint at the end of each epoch. |
| `accelerate_full_state_at_epoch` | `bool` | Backend default | If `True`, saves the full training state (optimizer, scheduler, etc.) at each epoch for automatic resumption with Accelerate. |
| `save_samples` | `int` | Backend default | Enables frequency based checkpointing. The model will checkpoint after every `save_samples` number of samples seen. |

#### Distributed Training (Multi-GPU / Multi-Node)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nproc_per_node` | `int` | Auto-detected | Number of processes (GPUs) to use per node. If not specified, uses all available GPUs. |
| `nnodes` | `int` | `1` | Total number of nodes for multi-node training. |
| `node_rank` | `int` | `0` | Rank of this node (0 to `nnodes-1`). Node 0 is the master node. |
| `rdzv_id` | `int` | Random | Unique job ID for rendezvous. All nodes must use the same ID. |
| `rdzv_endpoint` | `str` | Required for multi-node | Endpoint of the master node (node 0) in the format `"hostname:port"` (e.g., `"192.168.1.10:29500"`). |

#### Logging Configuration

Loggers are automatically enabled when their configuration parameters are set:

| Logger      | Enabled By            | Env Variable Fallback |
| ----------- | --------------------- | --------------------- |
| MLflow      | `mlflow_tracking_uri` | `MLFLOW_TRACKING_URI` |
| W&B         | `wandb_project`       | `WANDB_PROJECT`       |
| TensorBoard | `tensorboard_log_dir` | -                     |

| Parameter                | Type  | Default                    | Description                                                                     |
| ------------------------ | ----- | -------------------------- | ------------------------------------------------------------------------------- |
| `mlflow_tracking_uri`    | `str` | `MLFLOW_TRACKING_URI` env  | MLflow tracking server URI. Enables MLflow logging.                             |
| `mlflow_experiment_name` | `str` | `MLFLOW_EXPERIMENT_NAME` env | MLflow experiment name.                                                       |
| `mlflow_run_name`        | `str` | `None`                     | MLflow run name (supports `{time}`, `{utc_time}`, `{rank}` placeholders).       |
| `wandb_project`          | `str` | `WANDB_PROJECT` env        | W&B project name. Enables W&B logging.                                          |
| `wandb_entity`           | `str` | `WANDB_ENTITY` env         | W&B team/entity.                                                                |
| `wandb_run_name`         | `str` | `None`                     | W&B run name.                                                                   |
| `tensorboard_log_dir`    | `str` | `None`                     | TensorBoard log directory. Enables TensorBoard.                                 |

#### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `**kwargs` | `Any` | - | Additional backend-specific parameters. These are passed directly to the backend implementation. |

## Returns

**Type:** `Any`

Returns the training result from the backend. The exact return type and structure depends on the backend implementation.

For the `instructlab-training` backend, this typically includes:
- Path to the final checkpoint
- Training metrics and logs
- Any generated samples (if `save_samples > 0`)

## Raises

- **`ValueError`**: If required parameters are missing or have invalid values.
- **`RuntimeError`**: If training fails or the specified backend is not available.
- **`FileNotFoundError`**: If `model_path` or `data_path` do not exist.

## Examples

### Basic Single-GPU Training

Here's a minimal example that you can copy and paste to get started:

```python
from training_hub import sft

# Basic single-GPU training
result = sft(
    model_path="Qwen/Qwen2.5-0.5B-Instruct",
    data_path="./training_data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=8,
    learning_rate=1e-5,
    max_seq_len=256,
    max_tokens_per_gpu=1024
)
```


### Single-Node Multi-GPU Training

Here's a more complex example showcasing a training job run across 4 GPUs.

```python
from training_hub import sft

result = sft(
    model_path="qwen/Qwen2.5-0.5B-Instruct",
    data_path="./data/training.jsonl",
    ckpt_output_dir="./checkpoints/qwen_sft",
    num_epochs=3,
    effective_batch_size=32,
    learning_rate=2e-5,
    max_seq_len=256,
    max_tokens_per_gpu=1024,
    nproc_per_node=4,  # Use 4 GPUs
    checkpoint_at_epoch=True
)
```

### Multi-Node Distributed Training

Finally, this example showcases a multi-node distributed training job run across 4 nodes, each with 8 GPUs.

**On the master node (rank 0):**

Run this code on the master node. The `rdzv_endpoint` uses `0.0.0.0` to listen on all network interfaces for incoming connections from worker nodes.

```python
from training_hub import sft

result = sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./data/large_dataset.jsonl",
    ckpt_output_dir="./checkpoints/llama_sft",
    num_epochs=5,
    effective_batch_size=128,
    learning_rate=1e-5,
    max_seq_len=4096,
    max_tokens_per_gpu=40_000,
    nnodes=4,
    node_rank=0,
    nproc_per_node=8,
    rdzv_id=12345,
    rdzv_endpoint="0.0.0.0:29500"  # Listen on all interfaces
)
```

**On worker nodes (ranks 1-3):**

Run this code on each worker node, replacing `192.168.1.10` with the actual IP address of your rank 0 node. Change `node_rank` to 1, 2, 3, etc. for each worker.

```python
from training_hub import sft

result = sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./data/large_dataset.jsonl",
    ckpt_output_dir="./checkpoints/llama_sft",
    num_epochs=5,
    effective_batch_size=128,
    learning_rate=1e-5,
    max_seq_len=4096,
    max_tokens_per_gpu=45000,
    nnodes=4,
    node_rank=1,  # Change to 2, 3 for other workers
    nproc_per_node=8,
    rdzv_id=12345,
    rdzv_endpoint="192.168.1.10:29500"  # IP address of rank 0 node
)
```

## Implementation Notes

?> **Tip**: Start with default values for most parameters. Only customize what you need based on your hardware and dataset.

!> **Memory Warning**: If you encounter CUDA out-of-memory (OOM) errors, reduce `max_tokens_per_gpu` or `max_seq_len`.

> **Note**: The `sft()` function is a convenience wrapper around [`SFTAlgorithm`](/api/classes/SFTAlgorithm). For more advanced use cases or custom backend configurations, use the class-based API directly.

## See Also

- [**osft() Function**](/api/functions/osft) - Alternative for continual learning without catastrophic forgetting
- [**SFTAlgorithm Class**](/api/classes/SFTAlgorithm) - Class-based API for more control
- [**InstructLab Training Backend**](/api/backends/instructlab-training) - Backend implementation details
- [**Data Formats**](/api/data-formats) - Required data format specifications
- [**Distributed Training Guide**](/guides/distributed-training) - Complete multi-node setup guide
- [**SFT Algorithm Overview**](/algorithms/sft) - Conceptual overview of supervised fine-tuning
- [**SFT Examples**](/examples/#supervised-fine-tuning-sft) - Working code examples and notebooks

## Source

[View source on GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/sft.py)
