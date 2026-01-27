# `SFTAlgorithm` - Supervised Fine-Tuning Algorithm Class

> Concrete implementation of the Algorithm interface for Supervised Fine-Tuning (SFT).

## Class Signature

```python
from training_hub import SFTAlgorithm, Backend

class SFTAlgorithm(Algorithm):
    """Supervised Fine-Tuning algorithm implementation."""

    def __init__(self, backend: Backend, **kwargs):
        """Initialize SFT algorithm with a backend."""

    def train(
        self,
        model_path: str,
        data_path: str,
        ckpt_output_dir: str,
        # ... all sft() parameters
        **kwargs
    ) -> Any:
        """Execute supervised fine-tuning."""

    def get_required_params(self) -> Dict[str, Type]:
        """Get required parameters."""

    def get_optional_params(self) -> Dict[str, Type]:
        """Get optional parameters."""
```

## Overview

`SFTAlgorithm` is the class-based implementation of supervised fine-tuning in Training Hub. It inherits from the [`Algorithm`](Algorithm.md) abstract base class and provides the concrete implementation for SFT.

This class is useful when you need:
- More control over the training process
- To reuse an algorithm instance across multiple training runs
- To inspect or modify algorithm behavior before training
- Direct access to the algorithm interface

For most use cases, the convenience function [`sft()`](../functions/sft.md) is simpler.

## Constructor

### `__init__(backend: Backend, **kwargs)`

Creates a new SFTAlgorithm instance.

**Parameters:**
- `backend` (`Backend`): The backend implementation to use for training (e.g., `InstructLabTrainingSFTBackend`)
- `**kwargs`: Additional configuration passed to the algorithm

**Example:**
```python
from training_hub import SFTAlgorithm, InstructLabTrainingSFTBackend

backend = InstructLabTrainingSFTBackend()
algorithm = SFTAlgorithm(backend=backend)
```

## Methods

### `train(**kwargs) -> Any`

Executes the supervised fine-tuning process.

#### Parameters

##### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Path to the model to fine-tune. Can be a local directory path or a HuggingFace model ID (e.g., `"Qwen/Qwen2.5-7B-Instruct"`). |
| `data_path` | `str` | Path to training data in JSONL format. Each line should contain a JSON object with a `messages` field. See [Data Formats](../data-formats.md). |
| `ckpt_output_dir` | `str` | Directory where model checkpoints will be saved during and after training. |
| `num_epochs` | `int` | Number of training epochs to run. Typical values: 1-10 depending on dataset size and task. |
| `effective_batch_size` | `int` | Effective batch size for the entire minibatch. This is the total number of samples processed before taking an optimizer step. |
| `learning_rate` | `float` | Learning rate for the optimizer. |
| `max_seq_len` | `int` | Maximum sequence length in tokens. |

##### Optional Parameters

###### Backend Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"instructlab-training"` | Backend implementation to use for training. Currently only `"instructlab-training"` is supported for SFT. |

###### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup_steps` | `int` | Backend default | Number of warmup steps for the learning rate scheduler. |
| `max_tokens_per_gpu` | `int` | Maximum number of tokens that can be on a single GPU at once. |

###### Data Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_output_dir` | `str` | `None` | Directory for data processing outputs. |

###### Pretraining Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_pretraining` | `bool` | `False` | When `True`, enables pretraining mode for training on raw documents. |
| `block_size` | `int` | `None` | Required when `is_pretraining=True`. Number of tokens per training block (recommend starting with 2048). |
| `document_column_name` | `str` | `"document"` | Column containing raw documents when `is_pretraining=True`. |

###### Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_at_epoch` | `bool` | Backend default | If `True`, saves a checkpoint at the end of each epoch. Useful for resuming training or selecting the best epoch. |
| `accelerate_full_state_at_epoch` | `bool` | Backend default | If `True`, saves the full training state (optimizer, scheduler, etc.) at each epoch for automatic resumption with Accelerate. |
| `save_samples` | `int` | Backend default | Enables frequency based checkpointing. The model will checkpoint after every `save_samples` number of samples seen. |

###### Distributed Training (Multi-GPU / Multi-Node)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nproc_per_node` | `int` | Auto-detected | Number of processes (GPUs) to use per node. If not specified, uses all available GPUs. |
| `nnodes` | `int` | `1` | Total number of nodes for multi-node training. |
| `node_rank` | `int` | `0` | Rank of this node (0 to `nnodes-1`). Node 0 is the master node. |
| `rdzv_id` | `int` | Random | Unique job ID for rendezvous. All nodes must use the same ID. |
| `rdzv_endpoint` | `str` | Required for multi-node | Endpoint of the master node in the format `"hostname:port"`. Master node uses `"0.0.0.0:port"`, workers use master's IP address. |

###### Logging Configuration

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

###### Additional Backend Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `**kwargs` | `Any` | - | Additional backend-specific parameters passed directly to `instructlab.training.TrainingArgs`. Examples include optimizer settings (e.g., `weight_decay`, `adam_beta1`), gradient clipping, and DeepSpeed configuration. |

#### Default Behavior

When optional parameters are set to `None` or omitted, the backend uses default values from `instructlab.training.TrainingArgs`. While the required parameters are typed as `Optional` in the implementation for flexibility, you should explicitly set `num_epochs`, `effective_batch_size`, `learning_rate`, `max_seq_len`, and `max_tokens_per_gpu` for production use to ensure predictable behavior.

#### Backend Support

The `SFTAlgorithm` currently supports the `instructlab-training` backend. See the [InstructLab Training Backend documentation](../backends/instructlab-training.md) for backend-specific details and additional parameters.

#### Returns

**Type:** `Any`

Returns the training result from the backend. For the `instructlab-training` backend, this typically includes:
- Path to the final checkpoint
- Training metrics and logs
- Generated samples (if `save_samples > 0`)

The exact structure depends on the backend implementation.

#### Raises

- **`ValueError`**:
  - When required parameters are missing or have invalid values
  - When parameter validation fails (e.g., negative values for `num_epochs`, invalid `model_path`)
  - When backend-specific validation fails

- **`RuntimeError`**:
  - When training execution fails
  - When the backend encounters errors during training
  - When out-of-memory errors occur during training execution
  - When distributed training coordination fails

### `get_required_params() -> Dict[str, Type]`

Returns the required parameters for SFT.

This method is useful for:
- Programmatic parameter validation in custom training frameworks
- Generating dynamic UI forms for training configuration
- Documentation generation tools
- Building parameter validators or type checkers

**Returns:**

```python
{
    "model_path": str,
    "data_path": str,
    "ckpt_output_dir": str,
    "num_epochs": int,
    "effective_batch_size": int,
    "learning_rate": float,
    "max_seq_len": int,
    "max_batch_len": int  # Note: User-facing API uses max_tokens_per_gpu
}
```

**Note on "Required" Parameters:**

While these parameters are identified as "required" for semantic clarity (they define the core aspects of SFT training), they have default values in the underlying `instructlab.training.TrainingArgs` implementation. The distinction helps identify which parameters are essential to the SFT algorithm versus convenience options. For production use, explicitly set these parameters rather than relying on defaults to ensure predictable, reproducible training behavior.

### `get_optional_params() -> Dict[str, Type]`

Returns the optional parameters for SFT.

This method is useful for discovering available configuration options programmatically, particularly when building dynamic training interfaces or validation systems.

**Returns:**

```python
{
    "backend": str,
    "data_output_dir": str,
    "save_samples": int,
    "warmup_steps": int,
    "is_pretraining": bool,
    "block_size": int,
    "document_column_name": str,
    "checkpoint_at_epoch": bool,
    "accelerate_full_state_at_epoch": bool,
    "nproc_per_node": int,
    "nnodes": int,
    "node_rank": int,
    "rdzv_id": int,
    "rdzv_endpoint": str,
    # Plus additional backend-specific parameters via **kwargs
}
```

**Example - Parameter Validation:**

```python
from training_hub import SFTAlgorithm, InstructLabTrainingSFTBackend

backend = InstructLabTrainingSFTBackend()
algorithm = SFTAlgorithm(backend=backend)

# Get parameter specifications
required = algorithm.get_required_params()
optional = algorithm.get_optional_params()

# Validate user input
user_params = {"model_path": "./model", "data_path": "./data.jsonl"}
missing = [p for p in required if p not in user_params]
if missing:
    print(f"Missing required parameters: {missing}")
```

## Examples

### Basic Usage

Demonstrates minimal SFTAlgorithm setup for single-GPU training with only the required core parameters.

```python
from training_hub import SFTAlgorithm, InstructLabTrainingSFTBackend

# Create backend
backend = InstructLabTrainingSFTBackend()

# Create algorithm
algorithm = SFTAlgorithm(backend=backend)

# Train
result = algorithm.train(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./training.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=8,
    learning_rate=1e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000
)
```

### Distributed Training

#### Multi-GPU (Single Node)

Demonstrates using SFTAlgorithm for multi-GPU training on a single machine, distributing the workload across all available GPUs.

```python
from training_hub import SFTAlgorithm, InstructLabTrainingSFTBackend

backend = InstructLabTrainingSFTBackend()
algorithm = SFTAlgorithm(backend=backend)

result = algorithm.train(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./large_dataset.jsonl",
    ckpt_output_dir="./checkpoints/multi_gpu",
    num_epochs=5,
    effective_batch_size=64,  # Distributed across 8 GPUs (8 per GPU)
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    nproc_per_node=8,  # Use 8 GPUs
    checkpoint_at_epoch=True
)
```

#### Multi-Node (Distributed)

Shows multi-node distributed training setup. Run the master node code on node 0 and worker node code on all other nodes, ensuring all nodes can communicate over the network.

**Master Node (rank 0):**

```python
from training_hub import SFTAlgorithm, InstructLabTrainingSFTBackend

backend = InstructLabTrainingSFTBackend()
algorithm = SFTAlgorithm(backend=backend)

result = algorithm.train(
    model_path="/shared/models/llama-70b",
    data_path="/shared/data/large_dataset.jsonl",
    ckpt_output_dir="/shared/checkpoints",
    num_epochs=10,
    effective_batch_size=256,  # Distributed across 32 GPUs (4 nodes × 8 GPUs)
    learning_rate=1e-5,
    max_seq_len=4096,
    max_tokens_per_gpu=45000,
    nnodes=4,
    node_rank=0,
    nproc_per_node=8,
    rdzv_id=12345,
    rdzv_endpoint="0.0.0.0:29500",  # Listen on all interfaces
    checkpoint_at_epoch=True
)
```

**Worker Nodes (ranks 1-3):**

```python
from training_hub import SFTAlgorithm, InstructLabTrainingSFTBackend

backend = InstructLabTrainingSFTBackend()
algorithm = SFTAlgorithm(backend=backend)

result = algorithm.train(
    model_path="/shared/models/llama-70b",
    data_path="/shared/data/large_dataset.jsonl",
    ckpt_output_dir="/shared/checkpoints",
    num_epochs=10,
    effective_batch_size=256,
    learning_rate=1e-5,
    max_seq_len=4096,
    max_tokens_per_gpu=45000,
    nnodes=4,
    node_rank=1,  # Change to 2, 3 for other workers
    nproc_per_node=8,
    rdzv_id=12345,
    rdzv_endpoint="192.168.1.10:29500",  # IP address of rank 0 node
    checkpoint_at_epoch=True
)
```

## Implementation Notes

**Relationship to sft() Function:**

The [`sft()`](../functions/sft.md) function is a convenience wrapper around `SFTAlgorithm`. Both use identical parameters, and **this page is the authoritative parameter reference** - the `sft()` function documentation inherits all parameter definitions from the `train()` method documented above.

?> **Tip**: For most use cases, prefer the [`sft()`](../functions/sft.md) function for its simplicity.

?> **Advanced Usage**: Use the class directly when building complex training pipelines or frameworks that require instance reuse.

> **Note**: The `SFTAlgorithm` class delegates all actual training execution to its backend. The algorithm is responsible for parameter validation and high-level orchestration.

## See Also

- [**sft() Function**](/api/functions/sft) - Convenience wrapper function
- [**Algorithm Class**](/api/classes/Algorithm) - Base class interface
- [**InstructLabTrainingSFTBackend**](/api/backends/instructlab-training) - Default backend
- [**OSFTAlgorithm Class**](/api/classes/OSFTAlgorithm) - OSFT alternative
- [**create_algorithm() Function**](/api/functions/create-algorithm) - Factory function
- [**SFT Algorithm Overview**](/algorithms/sft) - Conceptual overview

## Source

[View source on GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/sft.py)
