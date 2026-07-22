# RHAI Innovation Mini-Trainer Backend

> Specialized backend for Orthogonal Subspace Fine-Tuning (OSFT) enabling continual learning without catastrophic forgetting.

## Overview

**Class:** `MiniTrainerOSFTBackend`

**Algorithm Support:** OSFT (Orthogonal Subspace Fine-Tuning)

**Package:** `rhai-innovation-mini-trainer`

**Status:** ✅ Fully implemented and tested

The RHAI Innovation Mini-Trainer backend provides production-ready OSFT capabilities optimized for continual learning scenarios. It implements the orthogonal subspace decomposition technique from Nayak et al. (2025).

## Features

- Specialized orthogonal subspace computations for OSFT
- Prevents catastrophic forgetting during continual learning
- Liger kernel support for improved performance
- Multi-GPU and multi-node distributed training
- Efficient parameter updates preserving orthogonality
- Flexible module targeting with regex patterns

## Usage

### Via Convenience Function

The simplest way to use this backend is through the [`osft()`](/api/functions/osft) function:

```python
from training_hub import osft

result = osft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./new_domain_data.jsonl",
    ckpt_output_dir="./checkpoints",
    backend="mini-trainer",  # Explicitly specify (or omit for default)
    unfreeze_rank_ratio=0.25,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=5e-6
)
```

### Via Class-Based API

For more control, use the backend directly with [`OSFTAlgorithm`](/api/classes/OSFTAlgorithm):

```python
from training_hub import OSFTAlgorithm, MiniTrainerOSFTBackend

# Create backend instance
backend = MiniTrainerOSFTBackend()

# Create algorithm with this backend
algorithm = OSFTAlgorithm(backend=backend)

# Train
result = algorithm.train(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./domain_data.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.3,
    effective_batch_size=32,
    max_tokens_per_gpu=4096,
    max_seq_len=2048,
    learning_rate=2e-5,
    num_epochs=3
)
```

## Supported Parameters

The Mini-Trainer backend supports all standard OSFT parameters. See the [`osft()` function reference](/api/functions/osft#parameters) for complete documentation.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `model_path` | Path to model or HuggingFace ID |
| `data_path` | JSONL training data file |
| `unfreeze_rank_ratio` | Control parameter (0.0-1.0) for adaptation vs preservation |
| `effective_batch_size` | Total batch size across all GPUs |
| `max_tokens_per_gpu` | GPU memory limit (tokens per GPU) |
| `max_seq_len` | Maximum sequence length |
| `learning_rate` | Learning rate for optimizer |
| `ckpt_output_dir` | Output directory for checkpoints |

### OSFT-Specific Parameters

| Parameter | Description |
|-----------|-------------|
| `target_patterns` | Regex patterns for module selection |
| `use_liger` | Enable Liger kernels for performance |
| `seed` | Random seed for reproducibility |

### Training Configuration

| Parameter | Description |
|-----------|-------------|
| `num_epochs` | Number of training epochs |
| `warmup_steps` | Learning rate warmup steps |
| `lr_scheduler` | PyTorch LR scheduler name |
| `lr_scheduler_kwargs` | Additional scheduler parameters |
| `checkpoint_at_epoch` | Save checkpoint after each epoch |
| `save_final_checkpoint` | Save final model |

### Optimizer (AdamW) Parameters

The Mini-Trainer backend uses the AdamW optimizer. Training Hub exposes these hyperparameters as first-class parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta1` | `float` | `0.9` | AdamW beta1 coefficient (first moment / momentum). |
| `beta2` | `float` | `0.95` | AdamW beta2 coefficient (second moment / RMSprop). Note: default is `0.95`, not PyTorch's `0.999`. |
| `eps` | `float` | `1e-8` | Epsilon for numerical stability. |
| `weight_decay` | `float` | `0.0` | Weight decay (L2 regularization). |

These parameter names match the mini-trainer `TrainingArgs` dataclass fields directly.

### Distributed Training Parameters

| Parameter | Description |
|-----------|-------------|
| `nproc_per_node` | Number of GPUs per node |
| `nnodes` | Total number of nodes |
| `node_rank` | This node's rank |
| `rdzv_id` | Rendezvous ID for multi-node |
| `rdzv_endpoint` | Master node endpoint |


## Validation

The Mini-Trainer backend supports validation loss monitoring with configurable event-based triggers and optional best-val-loss checkpointing. All validation parameters are first-class parameters of `osft()` and `OSFTAlgorithm.train()`.

### Validation Data Sources

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validation_split` | `float` | `0.0` | Fraction of training data to hold out for validation (`0.0` to `1.0`). `0.0` disables validation. Mutually exclusive with `validation_data_path`. |
| `validation_data_path` | `str` | `None` | Path to a separate validation dataset in JSONL format. The data is tokenized automatically using the same processing pipeline as the training data. Mutually exclusive with `validation_split`. |

### Validation Triggers

At least one trigger must be configured when validation data is present. Multiple triggers can be combined and are coalesced so validation runs at most once per step.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validation_frequency` | `int` | `None` | Run validation every N training steps. |
| `validate_at_epoch` | `bool` | `False` | Run validation at the end of each epoch. |
| `min_samples_per_validation` | `int` | `None` | Minimum accumulated samples between validation runs. Must be a positive integer. |
| `validate_at_final` | `bool` | `False` | Run validation at the end of training. |

### Best-Val-Loss Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_best_val_loss` | `bool` | `False` | Save a checkpoint whenever validation loss improves. |
| `val_loss_improvement_threshold` | `float` | `0.0` | Minimum improvement in validation loss required to trigger a best-val-loss checkpoint save. |

### Example: Step-Based Validation with Data Split

```python
osft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=5e-6,
    num_epochs=5,
    validation_split=0.1,
    validation_frequency=100,
    save_best_val_loss=True,
)
```

### Example: Separate Validation Dataset with Event-Based Triggers

```python
osft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./train_data.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=5e-6,
    num_epochs=5,
    validation_data_path="./val_data.jsonl",
    validate_at_epoch=True,
    validate_at_final=True,
)
```

Validation metrics (`val_loss`, `val_num_samples`, `val_num_loss_counted_tokens`, `val_num_batches`) are logged to JSONL, wandb, and MLflow alongside training metrics.

See the [Validation Loss Guide](/guides/validation-loss) for details.

## Additional Parameters

Beyond the parameters listed above, the Mini-Trainer backend supports many additional parameters beyond those documented above. For a complete list of all available parameters, refer to the [`TrainingArgs` class in the Mini-Trainer source code](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/main/src/mini_trainer/training_types.py).

The package provides many other useful parameters which we avoid documenting so our docs do not go out of sync with the package. 
For example, the package provides different modes of specifying training duration, as well as an "infinite" mode for training indefinitely.

The package also exposes wandb logging, however this also requires the user to have the `wandb` package installed.

**Algorithm Reference:**
Based on "Orthogonal Subspace Fine-Tuning" by Nayak et al. (2025), [arXiv:2504.07097](https://arxiv.org/abs/2504.07097)

## Additional Parameters

The backend passes additional parameters to the Mini-Trainer framework. These are framework-specific and may include:

- Data processing options
- Advanced optimization settings
- Debug and logging options

## See Also

- [**osft() Function**](/api/functions/osft) - Convenience function using this backend
- [**OSFTAlgorithm Class**](/api/classes/OSFTAlgorithm) - Algorithm that uses this backend
- [**Backend Base Class**](/api/classes/Backend) - Backend interface
- [**OSFT Algorithm Overview**](/algorithms/osft) - Conceptual overview and theory
- [**Distributed Training Guide**](/guides/distributed-training) - Multi-node setup
- [**Backends Overview**](/api/backends/README) - All backends

## External Resources

- [RHAI Innovation Mini-Trainer PyPI](https://pypi.org/project/rhai-innovation-mini-trainer/)
- [OSFT Paper (arXiv:2504.07097)](https://arxiv.org/abs/2504.07097)
