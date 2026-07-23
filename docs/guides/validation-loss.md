# Validation Loss

Training Hub supports validation loss monitoring through its underlying backends. When enabled, a fraction of the training data is held out and the model is periodically evaluated on it, giving you a signal for overfitting and helping you decide when to stop training.

## Support Matrix

| Algorithm | Backend | Validation Loss | Best-Val-Loss Checkpointing |
|-----------|---------|:---------------:|:---------------------------:|
| [SFT](/api/functions/sft) | [instructlab-training](/api/backends/instructlab-training) | Yes | No |
| [OSFT](/api/functions/osft) | [mini-trainer](/api/backends/mini-trainer) | Yes | Yes |
| [LoRA](/api/functions/lora_sft) | [Unsloth](/api/backends/unsloth) | No | No |
| [LoRA GRPO](/api/functions/lora_grpo) | [ART](/api/backends/art-grpo) / [verl](/api/backends/verl) | No | No |

?> **OSFT validation parameters are first-class.** As of mini-trainer v0.9.0, OSFT supports validation data, event-based triggers, and best-val-loss checkpointing as named parameters on `osft()` and `OSFTAlgorithm.train()`. SFT still supports validation through `**kwargs`.

## How It Works

Both supported backends follow the same approach:

1. **Data splitting**: The training dataset is automatically split into train and validation sets using the `validation_split` ratio. The split uses a fixed seed for reproducibility.
2. **Periodic evaluation**: Every `validation_frequency` training steps, the model is set to eval mode and the full validation set is processed under `torch.no_grad()`.
3. **Loss computation**: Per-token cross-entropy loss is computed, summed across all tokens and all distributed ranks, then divided by the total number of loss-counted tokens to produce the average validation loss.
4. **Metric logging**: Validation metrics are written to the same logging destinations as training metrics (JSONL file, wandb, MLflow, etc.).

## SFT (instructlab-training backend)

### Parameters

Pass these as `**kwargs` to `sft()` or `SFTAlgorithm.train()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validation_split` | `float` | `0.0` | Fraction of data to hold out for validation. Range: `[0.0, 1.0)`. Setting `0.0` (default) disables validation. |
| `validation_frequency` | `int` | `None` | How often to run validation, in training steps. **Required** when `validation_split > 0`. |

### Metrics

When validation runs, the following metrics are emitted alongside training metrics:

| Metric | Description |
|--------|-------------|
| `val_loss` | Average per-token validation loss |
| `val_num_tokens` | Total number of tokens evaluated |

These appear in:
- **JSONL file** (`training_params_and_metrics_global0.jsonl`)
- **TensorBoard** (rank 0 only)
- **wandb** (rank 0 only)
- **MLflow** (rank 0 only)

### Example

```python
from training_hub import sft

result = sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./training_data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=32,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    # Validation loss configuration
    validation_split=0.1,        # Hold out 10% of data
    validation_frequency=50,     # Evaluate every 50 steps
)
```

## OSFT (mini-trainer backend)

As of mini-trainer v0.9.0, OSFT validation parameters are first-class — pass them directly to `osft()` or `OSFTAlgorithm.train()`.

### Validation Data Sources

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validation_split` | `float` | `None` | Fraction of training data to hold out for validation. Range: `(0.0, 1.0)`. Mutually exclusive with `validation_data_path`. |
| `validation_data_path` | `str` | `None` | Path to a separate validation dataset in JSONL format. Tokenized automatically. Mutually exclusive with `validation_split`. |

### Validation Triggers

At least one trigger must be configured when validation data is present. Multiple triggers can be combined; they are coalesced so validation runs at most once per step.

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

### Metrics

When validation runs, the following metrics are emitted alongside training metrics:

| Metric | Description |
|--------|-------------|
| `val_loss` | Average per-token validation loss |
| `val_num_samples` | Total number of validation samples processed |
| `val_num_loss_counted_tokens` | Total tokens counted for loss computation |
| `val_num_batches` | Number of validation batches processed |

These appear in:
- **JSONL file** (`training_metrics.jsonl`)
- **wandb** (if configured)
- **MLflow** (if configured)

### Example: Step-Based Validation with Data Split

```python
from training_hub import osft

result = osft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./domain_data.jsonl",
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
    val_loss_improvement_threshold=0.01,
)
```

### Example: Separate Validation Dataset with Event-Based Triggers

```python
from training_hub import osft

result = osft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./train_data.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=32,
    max_tokens_per_gpu=4096,
    max_seq_len=2048,
    learning_rate=2e-5,
    num_epochs=3,
    nproc_per_node=4,
    validation_data_path="./val_data.jsonl",
    validate_at_epoch=True,
    validate_at_final=True,
)
```

### Example: Multiple Triggers Combined

```python
from training_hub import osft

result = osft(
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
    validation_frequency=200,
    validate_at_epoch=True,
    validate_at_final=True,
    save_best_val_loss=True,
)
```

## Interpreting Validation Loss

Validation loss is the primary signal for detecting overfitting:

- **Both training and validation loss decreasing**: Training is progressing well. Continue training.
- **Training loss decreasing but validation loss increasing**: The model is overfitting. Stop training or reduce the number of epochs.
- **Validation loss plateauing**: The model has learned what it can from the data. Further training provides diminishing returns.

?> **Tip**: Use `plot_loss()` to visualize training metrics. Validation loss metrics (`val_loss`) are included in the JSONL log files alongside training loss.

```python
from training_hub import plot_loss

plot_loss("./checkpoints")
```

## Choosing `validation_split` and `validation_frequency`

- **`validation_split`**: 5-15% is typical. Smaller datasets benefit from a smaller split (5%) to keep more data for training. Larger datasets can afford 10-15%.
- **`validation_frequency`**: Balance between monitoring granularity and training speed. Validation pauses training while it runs. Every 50-200 steps is a common range. For short runs, evaluate less frequently; for long runs, more frequently.

## See Also

- [**Experiment Tracking & Logging**](/guides/logging) - Configure where metrics are logged
- [**sft() Function**](/api/functions/sft) - SFT parameter reference
- [**osft() Function**](/api/functions/osft) - OSFT parameter reference
- [**InstructLab Training Backend**](/api/backends/instructlab-training) - Backend details
- [**Mini-Trainer Backend**](/api/backends/mini-trainer) - Backend details
- [**Data Preparation**](/guides/data-preparation) - Data splitting best practices
