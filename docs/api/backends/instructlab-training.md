# InstructLab Training Backend

> Production-grade backend for Supervised Fine-Tuning (SFT) using the InstructLab Training framework.

## Overview

**Class:** `InstructLabTrainingSFTBackend`

**Algorithm Support:** SFT (Supervised Fine-Tuning)

**Package:** `instructlab-training`

**Status:** ✅ Fully implemented and tested

The InstructLab Training backend provides robust, production-ready supervised fine-tuning capabilities with excellent support for distributed training across single-node and multi-node setups.

## Features

- Battle-tested distributed training with `torchrun`
- Optimized for instruction-tuned language models
- Comprehensive checkpointing and automatic resumption
- Full multi-GPU and multi-node support
- Memory-efficient training with dynamic batching
- Integrated with Accelerate for training management

## Usage

### Via Convenience Function

The simplest way to use this backend is through the [`sft()`](/api/functions/sft) function:

```python
from training_hub import sft

result = sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./training_data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=8,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    backend="instructlab-training"  # Explicitly specify (or omit for default)
)
```

### Via Class-Based API

For more control, use the backend directly with [`SFTAlgorithm`](/api/classes/SFTAlgorithm):

```python
from training_hub import SFTAlgorithm, InstructLabTrainingSFTBackend

# Create backend instance
backend = InstructLabTrainingSFTBackend()

# Create algorithm with this backend
algorithm = SFTAlgorithm(backend=backend)

# Train
result = algorithm.train(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=5,
    effective_batch_size=32,
    learning_rate=1e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000
)
```

## Supported Parameters

The InstructLab Training backend supports all standard SFT parameters. See the [`sft()` function reference](/api/functions/sft#parameters) for complete documentation.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `model_path` | Path to model or HuggingFace ID |
| `data_path` | JSONL training data file |
| `ckpt_output_dir` | Output directory for checkpoints |
| `num_epochs` | Number of training epochs |
| `effective_batch_size` | Total batch size across all GPUs |
| `learning_rate` | Learning rate for optimizer |
| `max_seq_len` | Maximum sequence length |
| `max_tokens_per_gpu` | GPU memory limit (tokens per GPU) |
| `warmup_steps` | Learning rate warmup steps |
| `checkpoint_at_epoch` | Save checkpoint after each epoch |
| `accelerate_full_state_at_epoch` | Save full state for resumption |

### Distributed Training Parameters

| Parameter | Description |
|-----------|-------------|
| `nproc_per_node` | Number of GPUs per node |
| `nnodes` | Total number of nodes |
| `node_rank` | This node's rank (0 to nnodes-1) |
| `rdzv_id` | Rendezvous ID for multi-node |
| `rdzv_endpoint` | Master node endpoint (host:port) |


## Validation Loss

The InstructLab Training backend supports validation loss monitoring. When enabled, a fraction of the training data is held out and the model is periodically evaluated on it.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validation_split` | `float` | `0.0` | Fraction of data to hold out for validation (`0.0` to `1.0`). `0.0` disables validation. |
| `validation_frequency` | `int` | `None` | Steps between validation evaluations. Required when `validation_split > 0`. |

Pass these as kwargs through `sft()`:

```python
sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=32,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    validation_split=0.1,
    validation_frequency=50,
)
```

Validation metrics (`val_loss`, `val_num_tokens`) are logged to JSONL, TensorBoard, wandb, and MLflow alongside training metrics.

See the [Validation Loss Guide](/guides/validation-loss) for details.

## Additional Parameters

The InstructLab Training backend supports many additional parameters beyond those documented above. For a complete list of all available parameters, refer to the [`TrainingArgs` class in the InstructLab Training source code](https://github.com/instructlab/training/blob/main/src/instructlab/training/config.py).

These parameters can be passed directly through the `sft()` function or `SFTAlgorithm.train()` method:

```python
# example of passing non-documented params
sft(
    # required params
    model_path="./model",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    # ...
    keep_last_checkpoint_only=True,  # non-surfaced, backend-specific kwarg
)
```



## Training Flow

The InstructLab Training backend follows this execution flow:

1. **Parameter validation** - Validates all required parameters
2. **Data processing** - Processes JSONL data into training format
3. **Distributed setup** - Configures `torchrun` for multi-GPU/multi-node
4. **Training execution** - Runs the training loop
5. **Checkpointing** - Saves model checkpoints
6. **Cleanup** - Handles resource cleanup

## Checkpointing and Resumption

### Epoch-Level Checkpoints

Enable checkpointing at each epoch:

```python
result = sft(
    model_path="./model",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=10,
    effective_batch_size=16,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    checkpoint_at_epoch=True  # Save after each epoch
)
```

Checkpoints will be saved as:
```
checkpoints/
├── samples_NNN/
├── samples_NNN/
├── ...
└── samples_NNN/
```

Where `NNN` is the number of samples seen at the time of checkpointing.

### Full State Checkpoints (Resumption)

For automatic resumption support with Accelerate:

```python
result = sft(
    model_path="./model",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=10,
    effective_batch_size=16,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    checkpoint_at_epoch=True,
    accelerate_full_state_at_epoch=True  # Save full training state
)
```

This saves optimizer state, scheduler state, and RNG state for exact resumption. To resume training, simply pass the initial `ckpt_output_dir` from the previous training run, and the backend will automatically resume training from the last checkpoint. You need to be careful to pass the same `num_epochs` as the previous training run, otherwise the learning rate scheduler will start from the beginning.

## Additional Parameters

The backend passes additional parameters to the InstructLab Training framework. Consult the [InstructLab Training documentation](https://github.com/instructlab/instructlab-training) for framework-specific parameters.

## Implementation Details

**Source File:** `src/training_hub/algorithms/sft.py:7-53`

**Key Methods:**
- `execute_training(algorithm_params)` - Main training execution

## See Also

- [**sft() Function**](/api/functions/sft) - Convenience function using this backend
- [**SFTAlgorithm Class**](/api/classes/SFTAlgorithm) - Algorithm that uses this backend
- [**Backend Base Class**](/api/classes/Backend) - Backend interface
- [**Distributed Training Guide**](/guides/distributed-training) - Multi-node setup guide
- [**Backends Overview**](/api/backends/README) - All backends

## External Resources

- [InstructLab Training GitHub](https://github.com/instructlab/training)
