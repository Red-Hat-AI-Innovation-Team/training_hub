# Distributed Training Guide

This guide covers multi-GPU and multi-node distributed training with Training Hub.

## Overview

Training Hub supports distributed training at three levels:

1. **Single-GPU Training** - Default behavior, simplest setup
2. **Multi-GPU Training (Single Node)** - Multiple GPUs on one machine
3. **Multi-Node Training** - Multiple machines with multiple GPUs each

[SFT](/api/functions/sft) and [OSFT](/api/functions/osft) support distributed training using the same parameters. [LoRA](/api/functions/lora_sft) has different multi-GPU options (see [LoRA Multi-GPU](#lora-multi-gpu) below).

## Single-GPU Training

No special configuration needed:

```python
from training_hub import sft

result = sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=8,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000
)
```

## Multi-GPU Training (Single Node)

Use the `nproc_per_node` parameter to specify the number of GPUs:

```python
from training_hub import sft

result = sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./large_dataset.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=5,
    effective_batch_size=64,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    nproc_per_node=8  # Use 8 GPUs
)
```

Training Hub uses `torchrun` under the hood. Data is automatically sharded across GPUs, and gradients are synchronized using Distributed Data Parallel (DDP).

### Example: 4-GPU OSFT

```python
from training_hub import osft

result = osft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./medical_data.jsonl",
    ckpt_output_dir="./checkpoints/medical",
    nproc_per_node=4,  # 4 GPUs
    unfreeze_rank_ratio=0.25,
    effective_batch_size=32,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=5e-6
)
```

## Multi-Node Training

Multi-node training distributes training across multiple machines, each with multiple GPUs.

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `nnodes` | Total number of nodes | `4` |
| `node_rank` | Rank of this node (0 to nnodes-1) | `0`, `1`, `2`, `3` |
| `nproc_per_node` | GPUs per node | `8` |
| `rdzv_endpoint` | Rendezvous address (master: `0.0.0.0:port`, workers: master's IP) | Master: `"0.0.0.0:29500"`<br/>Workers: `"192.168.1.10:29500"` |
| `rdzv_id` | Unique job ID (same for all nodes) | `12345` |

### Setup Example: 4 Nodes with 8 GPUs Each

#### Master Node (Node 0)

Run this code on the master node. Use `0.0.0.0` in `rdzv_endpoint` to listen on all network interfaces for incoming connections from worker nodes.

```python
from training_hub import sft

result = sft(
    model_path="/shared/models/llama-70b",
    data_path="/shared/data/large_dataset.jsonl",
    ckpt_output_dir="/shared/checkpoints",
    num_epochs=10,
    effective_batch_size=256,  # Distributed across 32 GPUs (4 nodes × 8 GPUs)
    learning_rate=1e-5,
    max_seq_len=4096,
    max_tokens_per_gpu=45000,
    nnodes=4,
    node_rank=0,  # Master node
    nproc_per_node=8,
    rdzv_id=12345,
    rdzv_endpoint="0.0.0.0:29500"  # Listen on all interfaces
)
```

#### Worker Nodes (Nodes 1-3)

Run this code on each worker node, replacing `node_rank` with the appropriate value (1, 2, or 3). The `rdzv_endpoint` should point to the master node's IP address.

**Worker Node 1:**

```python
from training_hub import sft

result = sft(
    model_path="/shared/models/llama-70b",
    data_path="/shared/data/large_dataset.jsonl",
    ckpt_output_dir="/shared/checkpoints",
    num_epochs=10,
    effective_batch_size=256,
    learning_rate=1e-5,
    max_seq_len=4096,
    max_tokens_per_gpu=45000,
    nnodes=4,
    node_rank=1,  # Worker node 1
    nproc_per_node=8,
    rdzv_id=12345,
    rdzv_endpoint="192.168.1.10:29500"  # Master node's IP address
)
```

**Worker Nodes 2 and 3:**

Use the same code as Worker Node 1, but change `node_rank` to `2` and `3` respectively.

### Complete Multi-Node Example (OSFT)

This example shows a single script that works for all nodes by detecting the node rank and configuring the rendezvous endpoint appropriately.

```python
from training_hub import osft
import os

# Get node rank from environment or command line
node_rank = int(os.environ.get("NODE_RANK", 0))

# Master node listens on all interfaces, workers connect to master's IP
master_ip = "192.168.1.10"
rdzv_endpoint = "0.0.0.0:29500" if node_rank == 0 else f"{master_ip}:29500"

result = osft(
    model_path="/shared/models/gpt-oss-20b",
    data_path="/shared/data/continual_dataset.jsonl",
    ckpt_output_dir="/shared/checkpoints/multi_node_osft",
    nnodes=4,
    node_rank=node_rank,
    nproc_per_node=8,
    rdzv_id=98765,
    rdzv_endpoint=rdzv_endpoint,
    unfreeze_rank_ratio=0.25,
    effective_batch_size=256,  # Across 32 GPUs
    max_tokens_per_gpu=2048,
    max_seq_len=2048,
    learning_rate=5e-6,
    num_epochs=10,
    checkpoint_at_epoch=True
)
```

## LoRA Multi-GPU

LoRA training has different multi-GPU options than SFT/OSFT:

| Approach | Use When | Key Parameter | Launch Command |
|----------|----------|---------------|----------------|
| **Data-Parallel** | Model fits on 1 GPU, want faster training | None (use torchrun) | `torchrun --nproc-per-node=N script.py` |
| **Model Splitting** | Model too large for 1 GPU | `enable_model_splitting=True` | `python script.py` |

**Note:** Use either data-parallel OR model splitting, not both. See the [LoRA documentation](/algorithms/lora) for complete examples.

## See Also

- [**sft() Function**](/api/functions/sft) - SFT distributed parameters
- [**osft() Function**](/api/functions/osft) - OSFT distributed parameters
- [**lora_sft() Function**](/api/functions/lora_sft) - LoRA distributed parameters
- [**InstructLab Training Backend**](/api/backends/instructlab-training) - SFT backend details
- [**Mini-Trainer Backend**](/api/backends/mini-trainer) - OSFT backend details
- [**Unsloth Backend**](/api/backends/unsloth) - LoRA backend details
