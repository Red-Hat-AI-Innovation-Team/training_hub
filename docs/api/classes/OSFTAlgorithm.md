# `OSFTAlgorithm` - Orthogonal Subspace Fine-Tuning Algorithm Class

> Concrete implementation of the Algorithm interface for Orthogonal Subspace Fine-Tuning (OSFT), enabling continual learning without catastrophic forgetting.

## Class Signature

```python
from training_hub import OSFTAlgorithm, Backend

class OSFTAlgorithm(Algorithm):
    """
    Implements the Orthogonal Subspace Fine-Tuning (OSFT) algorithm,
    based on Nayak et al. (2025), arXiv:2504.07097
    """

    def __init__(self, backend: Backend, **kwargs) -> None:
        """Initialize OSFT algorithm with a backend."""

    def train(
        self,
        model_path: str,
        data_path: str,
        unfreeze_rank_ratio: float,
        # ... all osft() parameters
        **kwargs
    ) -> any:
        """Execute OSFT training."""

    def get_required_params(self) -> dict[str, type]:
        """Get required parameters."""

    def get_optional_params(self) -> dict[str, type]:
        """Get optional parameters."""
```

## Overview

`OSFTAlgorithm` is the class-based implementation of Orthogonal Subspace Fine-Tuning in Training Hub. It inherits from the [`Algorithm`](Algorithm.md) abstract base class and provides concrete implementation for continual learning.

OSFT allows you to adapt models to new domains or tasks while controlling how much of the original model behavior is preserved, preventing catastrophic forgetting.

This class is useful when you need:
- More control over the OSFT training process
- To reuse an algorithm instance across multiple training runs
- Direct access to the algorithm interface
- Custom continual learning pipelines

For most use cases, the convenience function [`osft()`](../functions/osft.md) is simpler.

## Constructor

### `__init__(backend: Backend, **kwargs) -> None`

Creates a new OSFTAlgorithm instance.

**Parameters:**
- `backend` (`Backend`): The backend implementation to use for training (e.g., `MiniTrainerOSFTBackend`)
- `**kwargs`: Additional configuration passed to the algorithm

**Example:**
```python
from training_hub import OSFTAlgorithm, MiniTrainerOSFTBackend

backend = MiniTrainerOSFTBackend()
algorithm = OSFTAlgorithm(backend=backend)
```

## Methods

### `train(**kwargs) -> Any`

Executes the OSFT training process.

#### Parameters

##### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Local path or HuggingFace model ID of the model to fine-tune (e.g., `"meta-llama/Llama-3.1-8B-Instruct"`). |
| `data_path` | `str` | Path to training data in JSONL format. See [Data Formats](../data-formats.md). |
| `unfreeze_rank_ratio` | `float` | **Key OSFT parameter.** Controls how much of each weight matrix is unfrozen for training. Range: 0.0-1.0. Lower values (0.1-0.3) preserve more of the original model behavior. Higher values (0.5-0.8) allow more adaptation. |
| `effective_batch_size` | `int` | Effective batch size for the entire minibatch. This is the total number of samples processed before taking an optimizer step. |
| `max_tokens_per_gpu` | `int` | Maximum number of tokens that can be on a single GPU at once. |
| `max_seq_len` | `int` | Maximum sequence length in tokens. |
| `learning_rate` | `float` | Learning rate for model updates. |
| `ckpt_output_dir` | `str` | Directory where checkpoints and final model will be saved. |

##### Optional Parameters

###### Backend Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"mini-trainer"` | Backend implementation to use. Currently only `"mini-trainer"` is supported for OSFT. |

###### Data Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_output_dir` | `str` | `None` | Directory for data processing outputs. |
| `use_processed_dataset` | `bool` | Backend default | Whether the data at `data_path` is already preprocessed. Set to `True` to skip preprocessing. |
| `unmask_messages` | `bool` | Backend default | If `True`, unmasks all messages during data processing (excluding system messages). |

###### Pretraining Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_pretraining` | `bool` | `False` | When `True`, enables pretraining mode for training on raw documents. |
| `block_size` | `int` | `None` | Required when `is_pretraining=True`. Number of tokens per training block (recommend starting with 2048). |
| `document_column_name` | `str` | `"document"` | Column containing raw documents when `is_pretraining=True`. |

###### OSFT-Specific Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_patterns` | `list[str]` | Backend default | List of substring patterns for selecting which model modules to apply OSFT to. Each string is checked against the model's named parameters (e.g., if `"q_proj"` is in the list, any parameter name containing `"q_proj"` will be targeted for OSFT). Parameters not matching any pattern are trained using standard SFT (entire matrix). If not specified, applies OSFT to all compatible linear layers. Example: `["q_proj", "v_proj", "k_proj"]`. |

###### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_epochs` | `int` | Backend default | Number of training epochs. Typical values: 1-10. |
| `warmup_steps` | `int` | Backend default | Number of warmup steps for the learning rate scheduler. |
| `lr_scheduler` | `str` | Backend default | PyTorch learning rate scheduler name (e.g., `"cosine"`, `"linear"`, `"constant"`). |
| `lr_scheduler_kwargs` | `dict[str, str]` | `None` | Additional keyword arguments for the learning rate scheduler. |
| `use_liger` | `bool` | Backend default | Whether to use Liger kernels for improved performance. Requires `liger-kernel` package. |
| `seed` | `int` | Backend default | Random seed for reproducibility. |

###### Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_at_epoch` | `bool` | Backend default | If `True`, saves a checkpoint at the end of each epoch. |
| `save_final_checkpoint` | `bool` | Backend default | If `True`, saves the final model checkpoint after training completes. |

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

###### Additional Backend Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `**kwargs` | `Any` | - | Additional backend-specific parameters passed directly to the backend implementation. |

#### Default Behavior

When optional parameters are set to `None` or omitted, the backend uses default values from `rhai_innovation_mini_trainer.TrainingArgs`. While the required parameters are typed as `Optional` in the implementation for flexibility, you should explicitly set all required parameters for production use to ensure predictable, reproducible training behavior.

#### Backend Support

The `OSFTAlgorithm` currently supports the `mini-trainer` backend. See the [Mini-Trainer Backend documentation](../backends/mini-trainer.md) for backend-specific details and additional parameters.

#### Returns

**Type:** `Any`

Returns the training result from the backend. For the `mini-trainer` backend, this typically includes:
- Path to the final checkpoint
- Training metrics and statistics
- Model state information

The exact structure depends on the backend implementation.

#### Raises

- **`ValueError`**:
  - When required parameters are missing or have invalid values
  - When `unfreeze_rank_ratio` is out of range (0.0-1.0)
  - When parameter validation fails
  - When backend-specific validation fails

- **`RuntimeError`**:
  - When training execution fails
  - When the backend encounters errors during training
  - When out-of-memory errors occur during training execution
  - When distributed training coordination fails

### `get_required_params() -> Dict[str, Type]`

Returns the required parameters for OSFT.

**Returns:**

```python
{
    "model_path": str,
    "data_path": str,
    "unfreeze_rank_ratio": float,
    "effective_batch_size": int,
    "max_tokens_per_gpu": int,
    "max_seq_len": int,
    "learning_rate": float,
    "ckpt_output_dir": str
}
```

### `get_optional_params() -> Dict[str, Type]`

Returns the optional parameters for OSFT.

**Returns:**

```python
{
    "backend": str,
    "data_output_dir": str,
    "use_processed_dataset": bool,
    "unmask_messages": bool,
    "is_pretraining": bool,
    "block_size": int,
    "document_column_name": str,
    "target_patterns": list,
    "num_epochs": int,
    "warmup_steps": int,
    "lr_scheduler": str,
    "lr_scheduler_kwargs": dict,
    "use_liger": bool,
    "seed": int,
    "checkpoint_at_epoch": bool,
    "save_final_checkpoint": bool,
    "nproc_per_node": int,
    "nnodes": int,
    "node_rank": int,
    "rdzv_id": int,
    "rdzv_endpoint": str,
    # Plus additional backend-specific parameters via **kwargs
}
```

## Examples

### Basic Usage

```python
from training_hub import OSFTAlgorithm, MiniTrainerOSFTBackend

# Create backend
backend = MiniTrainerOSFTBackend()

# Create algorithm
algorithm = OSFTAlgorithm(backend=backend)

# Train
result = algorithm.train(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./domain_data.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=5e-6
)
```

### Continual Learning Pipeline

```python
from pathlib import Path
from training_hub import OSFTAlgorithm, MiniTrainerOSFTBackend

class ContinualLearningPipeline:
    """Pipeline for multi-stage continual learning."""

    def __init__(self, base_model: str):
        self.base_model = base_model
        backend = MiniTrainerOSFTBackend()
        self.algorithm = OSFTAlgorithm(backend=backend)

    def add_domain(self, domain_name: str, data_path: str, output_dir: str):
        """Add a new domain to the model."""
        print(f"Learning domain: {domain_name}")

        result = self.algorithm.train(
            model_path=self.base_model,
            data_path=data_path,
            ckpt_output_dir=output_dir,
            unfreeze_rank_ratio=0.25,  # Preserve 75% of behavior
            effective_batch_size=16,
            max_tokens_per_gpu=2048,
            max_seq_len=1024,
            learning_rate=5e-6,
            num_epochs=3
        )

        # Update base model to the most recent checkpoint
        # Mini-trainer backend stores checkpoints in hf_format directory
        hf_format_dir = Path(output_dir) / "hf_format"
        checkpoints = sorted(hf_format_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        self.base_model = str(checkpoints[-1])  # Most recent checkpoint

        return result

# Use the pipeline
pipeline = ContinualLearningPipeline(base_model="Qwen/Qwen2.5-7B-Instruct")

# Learn multiple domains sequentially without forgetting
pipeline.add_domain("medical", "./medical_data.jsonl", "./checkpoints/medical")
pipeline.add_domain("legal", "./legal_data.jsonl", "./checkpoints/legal")
pipeline.add_domain("finance", "./finance_data.jsonl", "./checkpoints/finance")
```

### Experimenting with unfreeze_rank_ratio

```python
from training_hub import OSFTAlgorithm, MiniTrainerOSFTBackend

backend = MiniTrainerOSFTBackend()
algorithm = OSFTAlgorithm(backend=backend)

# Test different preservation levels
ratios = [0.1, 0.25, 0.5, 0.75]

for ratio in ratios:
    print(f"\nTraining with unfreeze_rank_ratio={ratio}")

    result = algorithm.train(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        data_path="./specialized_data.jsonl",
        ckpt_output_dir=f"./checkpoints/ratio_{ratio}",
        unfreeze_rank_ratio=ratio,
        effective_batch_size=16,
        max_tokens_per_gpu=2048,
        max_seq_len=1024,
        learning_rate=5e-6,
        num_epochs=3
    )

    # Evaluate each checkpoint to find optimal ratio
    # ... your evaluation code ...
```

## Understanding OSFT Parameters

The key parameter that distinguishes OSFT from standard fine-tuning is `unfreeze_rank_ratio`:

| `unfreeze_rank_ratio` | Behavior | Use Case |
|-----------------------|----------|----------|
| 0.1 - 0.2 | Minimal adaptation, maximum preservation | Adding small amounts of knowledge |
| 0.25 - 0.3 | Balanced (recommended) | General continual learning |
| 0.4 - 0.6 | More adaptation | Significant domain shifts |
| 0.7 - 0.9 | Heavy adaptation | Near full fine-tuning with constraints |

## Relationship to osft() Function

The [`osft()`](../functions/osft.md) function is a convenience wrapper around `OSFTAlgorithm`. Both use identical parameters. For most use cases, prefer the `osft()` function for its simplicity. Use the class directly when building continual learning pipelines that require instance reuse across multiple training stages.

## Implementation Notes

?> **Tip**: For most use cases, prefer the [`osft()`](../functions/osft.md) function for its simplicity.

?> **Continual Learning**: OSFT is specifically designed for scenarios where you need to adapt a model sequentially to multiple domains without forgetting previous knowledge.

!> **Memory**: OSFT requires more memory than standard fine-tuning due to orthogonality constraints. Monitor GPU usage carefully.

> **Note**: Based on the paper "Orthogonal Subspace Fine-Tuning" by Nayak et al. (2025), [arXiv:2504.07097](https://arxiv.org/abs/2504.07097).

## See Also

- [**osft() Function**](/api/functions/osft) - Convenience wrapper function
- [**Algorithm Class**](/api/classes/Algorithm) - Base class interface
- [**MiniTrainerOSFTBackend**](/api/backends/mini-trainer) - Default backend
- [**SFTAlgorithm Class**](/api/classes/SFTAlgorithm) - Standard fine-tuning alternative
- [**create_algorithm() Function**](/api/functions/create-algorithm) - Factory function
- [**OSFT Algorithm Overview**](/algorithms/osft) - Conceptual overview and theory

## Source

[View source on GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/osft.py)
