# OSFT Algorithm Usage Examples

This document shows how to use the OSFT (Orthogonal Subspace Fine-Tuning) algorithm in training_hub.

## Overview

The OSFT algorithm implements Orthogonal Subspace Fine-Tuning based on Nayak et al. (2025), arXiv:2504.07097. This algorithm allows for continual training of pre-trained or instruction-tuned models without the need of a supplementary dataset to maintain the distribution of the original model/dataset that was trained.

**Key Benefits:**
- Enables continual learning without catastrophic forgetting
- No need for supplementary datasets to maintain original model distribution
- Significantly reduces data requirements for customizing instruction-tuned models
- Memory requirements similar to standard SFT

## Data Format Requirements

Training Hub's OSFT algorithm supports both **processed** and **unprocessed** data formats via the mini-trainer backend.

### Option 1: Standard Messages Format (Recommended)

Your training data should be a **JSON Lines (.jsonl)** file containing messages data:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there! How can I help you?"}]}
{"messages": [{"role": "user", "content": "What is OSFT?"}, {"role": "assistant", "content": "OSFT stands for Orthogonal Subspace Fine-Tuning..."}]}
```

### Message Structure
- **`role`**: One of `"system"`, `"user"`, `"assistant"`, or `"pretraining"`
- **`content`**: The text content of the message
- **`reasoning_content`** (optional): Additional reasoning traces

### Masking Control with `unmask_messages` Parameter

Control training behavior during data processing:

**Standard instruction tuning (default):**
```python
osft(..., unmask_messages=False)  # Only assistant responses used for loss
```

**Pretraining mode:**
```python
osft(..., unmask_messages=True)   # All content except system messages used for loss
```

### Option 2: Pre-processed Dataset

If you have pre-processed data with `input_ids` and `labels` fields:

```json
{"input_ids": [1, 2, 3, ...], "labels": [1, 2, 3, ...]}
{"input_ids": [4, 5, 6, ...], "labels": [4, 5, 6, ...]}
```

Use with:
```python
osft(..., use_processed_dataset=True)
```

## Simple Usage with Convenience Function

The easiest way to run OSFT training is using the convenience function:

```python
from training_hub import osft

# Basic OSFT training
result = osft(
    model_path="/path/to/your/model",
    data_path="/path/to/your/training/data.jsonl",
    ckpt_output_dir="/path/to/save/outputs",
    unfreeze_rank_ratio=0.3,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=2048,
    learning_rate=2e-5
)

# OSFT training with custom parameters
result = osft(
    model_path="/path/to/your/model",
    data_path="/path/to/your/training/data.jsonl",
    ckpt_output_dir="/path/to/save/outputs",
    unfreeze_rank_ratio=0.2,
    effective_batch_size=16,
    max_tokens_per_gpu=4096,
    max_seq_len=4096,
    learning_rate=1e-5,
    num_epochs=3,
    warmup_steps=100,
    use_liger=True,
    osft_memory_efficient_init=True,  # Recommended for OOMs at model load time
    seed=42
)
```

## Using the Factory Pattern

For more control over the algorithm instance:

```python
from training_hub import create_algorithm

# Create an OSFT algorithm instance
osft_algo = create_algorithm('osft', 'mini-trainer')

# Run training
result = osft_algo.train(
    model_path="/path/to/your/model",
    data_path="/path/to/your/training/data.jsonl",
    ckpt_output_dir="/path/to/save/outputs",
    unfreeze_rank_ratio=0.25,
    batch_size=6,
    max_tokens_per_gpu=3072,
    max_seq_len=2048,
    learning_rate=1.5e-5,
    num_epochs=2
)

# Check required parameters
required_params = osft_algo.get_required_params()
print("Required parameters:", list(required_params.keys()))
```

## Algorithm and Backend Discovery

Explore available algorithms and backends:

```python
from training_hub import AlgorithmRegistry

# List all available algorithms
algorithms = AlgorithmRegistry.list_algorithms()
print("Available algorithms:", algorithms)  # ['sft', 'osft']

# List backends for OSFT
osft_backends = AlgorithmRegistry.list_backends('osft')
print("OSFT backends:", osft_backends)  # ['mini-trainer']

# Get algorithm class directly
OSFTAlgorithm = AlgorithmRegistry.get_algorithm('osft')
```

## Parameter Reference

### Required Parameters

- `model_path` (str): Local path or HuggingFace model ID to be used for fine-tuning
- `data_path` (str): Path to the training data (processed or unprocessed)
- `ckpt_output_dir` (str): Directory where outputs from training will be saved
- `unfreeze_rank_ratio` (float): Controls the amount that each matrix is unfrozen during OSFT (0.0-1.0)
- `effective_batch_size` (int): Batch size for training
- `max_tokens_per_gpu` (int): Maximum number of tokens placed on a single GPU
- `max_seq_len` (int): Maximum sequence length (in tokens) for training samples
- `learning_rate` (float): Learning rate for model update size

### Optional Training Parameters

**OSFT-Specific Parameters:**
- `target_patterns` (list[str]): Patterns to match when selecting modules for OSFT
- `unfreeze_rank_ratio` (float): Valid values are between 0.0 and 1.0 (seldom need >0.5)

**Data Processing Parameters:**
- `use_processed_dataset` (bool): Whether to use pre-processed dataset format
- `unmask_messages` (bool): Whether to unmask messages during data processing

**Core Training Parameters:**
- `num_epochs` (int): Number of epochs to train for
- `seed` (int): Random seed for training
- `use_liger` (bool): Whether to use Liger kernels for training
- `osft_memory_efficient_init` (bool): Enable memory-efficient initialization to reduce memory usage during model loading (recommended for OOMs)

**Learning Rate Scheduler:**
- `lr_scheduler` (str): Name of the PyTorch learning rate scheduler to use
- `warmup_steps` (int): Number of warmup steps for the learning rate scheduler
- `lr_scheduler_kwargs` (dict[str, str]): Additional scheduler parameters

**Checkpointing:**
- `checkpoint_at_epoch` (bool): Whether to checkpoint at each epoch
- `save_final_checkpoint` (bool): Whether to save final checkpoint

**Multi-Node Parameters:**
- `nproc_per_node` (int): Number of processes (GPUs) per node
- `nnodes` (int): Total number of nodes in the cluster
- `node_rank` (int): Rank of this node (0 to nnodes-1)
- `rdzv_id` (int): Unique job ID for rendezvous
- `rdzv_endpoint` (str): Master node endpoint (format: "host:port")

### Backend Selection

- `backend` (str, default="mini-trainer"): Backend implementation to use

## Error Handling

```python
from training_hub import osft, AlgorithmRegistry

try:
    result = osft(
        model_path="/valid/model/path",
        data_path="/valid/data/path",
        ckpt_output_dir="/valid/output/path",
        unfreeze_rank_ratio=0.3,
        effective_batch_size=8,
        max_tokens_per_gpu=2048,
        max_seq_len=2048,
        learning_rate=2e-5
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Training error: {e}")

# Check if algorithm exists before using
if 'osft' in AlgorithmRegistry.list_algorithms():
    print("OSFT algorithm is available")

# Check if backend exists
if 'mini-trainer' in AlgorithmRegistry.list_backends('osft'):
    print("Mini-trainer backend is available")
```

## Multi-Node Training

The OSFT algorithm supports multi-node distributed training through torchrun parameters:

```python
from training_hub import osft

# Single-node, multi-GPU training (2 GPUs)
result = osft(
    model_path="/path/to/model",
    data_path="/path/to/data.jsonl",
    ckpt_output_dir="/path/to/outputs",
    unfreeze_rank_ratio=0.3,
    effective_batch_size=4,
    max_tokens_per_gpu=2048,
    max_seq_len=2048,
    learning_rate=2e-5,
    nproc_per_node=2,  # Number of GPUs per node
    nnodes=1,          # Single node
    node_rank=0,       # This node's rank
    rdzv_id=12345,     # Rendezvous ID
    rdzv_endpoint=""   # Empty for single node
)

# Multi-node training (2 nodes, 4 GPUs each)
# Run this on the first node (rank 0):
result = osft(
    model_path="/path/to/model",
    data_path="/path/to/data.jsonl",
    ckpt_output_dir="/path/to/outputs",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=2,
    max_tokens_per_gpu=1024,
    max_seq_len=2048,
    learning_rate=1e-5,
    nproc_per_node=4,           # 4 GPUs per node
    nnodes=2,                   # 2 total nodes
    node_rank=0,                # This is node 0
    rdzv_id=12345,              # Shared rendezvous ID
    rdzv_endpoint="node0:29500" # Master node endpoint
)
```

## Best Practices

1. **unfreeze_rank_ratio**: Start with values between 0.1-0.5. Values >0.5 are rarely needed for general continual-learning regimes.

2. **Memory Management**: OSFT doesn't reduce memory requirements compared to SFT, so adjust `max_tokens_per_gpu` accordingly. For memory-constrained environments or OOMs during model loading, set `osft_memory_efficient_init=True`.

3. **Data Processing**: The algorithm handles data processing automatically. Use `use_processed_dataset=True` only if you have pre-tokenized data.

4. **Continual Learning**: OSFT is particularly effective for adapting instruction-tuned models to new domains without catastrophic forgetting.
