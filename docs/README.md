# Training Hub

**Training Hub** is an algorithm-focused interface for common LLM training, continual learning, and reinforcement learning techniques developed by the [Red Hat AI Innovation Team](https://ai-innovation.team).

<p align="center">
  <a href="https://pypi.org/project/training-hub/">
    <img src="https://img.shields.io/pypi/v/training-hub?style=for-the-badge" alt="PyPI version">
  </a>
  <a href="https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Red-Hat-AI-Innovation-Team/training_hub?style=for-the-badge" alt="License">
  </a>
  <a href="https://ai-innovation.team/training_hub">
    <img src="https://img.shields.io/badge/📚_Documentation_(WIP)-blue?style=for-the-badge" alt="Documentation (in progress)">
  </a>
</p>

**New to Training Hub?** Read our comprehensive introduction: [Get Started with Language Model Post-Training Using Training Hub](https://developers.redhat.com/articles/2025/11/19/get-started-language-model-post-training-using-training-hub)

## Support Matrix

| Algorithm | InstructLab-Training | RHAI Innovation Mini-Trainer | PEFT | Unsloth | verl | Status |
|-----------|----------------------|------------------------------|------|---------|------|--------|
| **Supervised Fine-tuning (SFT)** | ✅ | - | - | - | - | Implemented |
| Continual Learning (OSFT) | 🔄 | ✅ | 🔄 | - | - | Implemented |
| **Low-Rank Adaptation (LoRA) + SFT** | - | - | - | ✅ | - | Implemented |
| **LoRA + GRPO (Adapter-Based RLVR)** | - | - | - | ✅ | ✅ | Implemented |
| Direct Preference Optimization (DPO) | - | - | - | - | 🔄 | Planned |

**Legend:**
- ✅ Implemented and tested
- 🔄 Planned for future implementation
- \- Not applicable or not planned

## Implemented Algorithms

### [Supervised Fine-tuning (SFT)](./algorithms/sft)

Fine-tune language models on supervised datasets with support for:
- Single-node and multi-node distributed training
- Configurable training parameters (epochs, batch size, learning rate, etc.)
- InstructLab-Training backend integration

```python
from training_hub import sft

result = sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="/path/to/data",
    ckpt_output_dir="/path/to/checkpoints",
    num_epochs=3,
    effective_batch_size=8,
    learning_rate=1e-5,
    max_seq_len=256,
    max_tokens_per_gpu=1024,
)
```

### [Orthogonal Subspace Fine-Tuning (OSFT)](./algorithms/osft)

OSFT allows you to fine-tune models while controlling how much of its
existing behavior to preserve. Currently we have support for:

- Single-node and multi-node distributed training
- Configurable training parameters (epochs, batch size, learning rate, etc.)
- RHAI Innovation Mini-Trainer backend integration

Here's a quick and minimal way to get started with OSFT:

```python
from training_hub import osft

result = osft(
    model_path="/path/to/model",
    data_path="/path/to/data.jsonl", 
    ckpt_output_dir="/path/to/outputs",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=5e-6,
)
```

### [Low-Rank Adaptation (LoRA) + SFT](./algorithms/lora)


Parameter-efficient fine-tuning using LoRA with supervised fine-tuning. Features:
- Memory-efficient training with significantly reduced VRAM requirements
- Single-GPU and multi-GPU distributed training support
- Unsloth backend for 2x faster training and 70% less memory usage
- Support for QLoRA (4-bit quantization) for even lower memory usage
- Compatible with messages and Alpaca dataset formats

```python
from training_hub import lora_sft

result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="/path/to/data.jsonl",
    ckpt_output_dir="/path/to/outputs",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    learning_rate=2e-4
)
```


### [LoRA + GRPO (Adapter-Based RLVR)](./algorithms/lora_grpo)

Train LoRA adapters on tool-calling agents using Group Relative Policy Optimization with reinforcement learning from verifiable rewards. Features:
- Single-turn and multi-turn tool-call verification with automatic per-turn decomposition
- Two backends: OpenPipe ART + Unsloth GRPO (single-GPU, fast iteration) and verl (multi-GPU, scales to 70B+)
- Built-in reward functions for tool-call correctness, or bring your own
- Zero API cost training using ground-truth trace decomposition

```python
from training_hub import lora_grpo

# Single GPU (ART backend)
result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./tool_call_traces.jsonl",
    ckpt_output_dir="./grpo_output",
    backend="art",
    lora_r=32,
    lora_alpha=64,
    num_iterations=15,
)

# Multi GPU (verl backend)
result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./tool_call_traces.jsonl",
    ckpt_output_dir="./grpo_output",
    backend="verl",
    n_gpus=4,
)
```

## Installation

### Basic Installation

This installs the base package, but doesn't install the CUDA-related dependencies which are required for GPU training.

```bash
pip install training-hub
```

### Development Installation
```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/training_hub
cd training_hub
pip install -e .
```

**For developers:** See the [Development Guide](./DEVELOPING.md) for detailed instructions on setting up your development environment, running local documentation, and contributing to Training Hub.


### LoRA Support
For LoRA training with optimized dependencies:
```bash
pip install training-hub[lora]
# or for development
pip install -e .[lora]
```

**Note:** The LoRA extras include Unsloth optimizations and PyTorch-optimized xformers for better performance and compatibility.

### GRPO Support
For LoRA + GRPO training (both ART and verl backends):
```bash
pip install training-hub[grpo,lora]
```

### CUDA Support
For GPU training with CUDA support:
```bash
pip install training-hub[cuda] --no-build-isolation
# or for development
pip install -e .[cuda] --no-build-isolation
```

**Note:** If you encounter build issues with flash-attn, install the base package first:
```bash
# Install base package (provides torch, packaging, wheel, ninja)
pip install training-hub
# Then install with CUDA extras
pip install training-hub[cuda] --no-build-isolation

# For development installation:
pip install -e . && pip install -e .[cuda] --no-build-isolation
```

If you're using uv, you can use the following commands to install the package:

```bash
# Installs training-hub from PyPI
uv pip install training-hub && uv pip install training-hub[cuda] --no-build-isolation

# For development:
git clone https://github.com/Red-Hat-AI-Innovation-Team/training_hub
cd training_hub
uv pip install -e . && uv pip install -e .[cuda] --no-build-isolation
```

## Getting Started

For comprehensive tutorials, examples, and documentation, see the [examples directory](./examples/).
