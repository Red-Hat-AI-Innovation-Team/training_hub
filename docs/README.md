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

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/docs/assets/quickstart-dark.gif">
    <source media="(prefers-color-scheme: light)" srcset="/docs/assets/quickstart.gif">
    <img src="/docs/assets/quickstart.gif" alt="Training Hub quickstart examples" width="800" height="420">
  </picture>
</div>

**New to Training Hub?** Read our comprehensive introduction: [Get Started with Language Model Post-Training Using Training Hub](https://developers.redhat.com/articles/2025/11/19/get-started-language-model-post-training-using-training-hub)

## Support Matrix

| Algorithm | InstructLab-Training | RHAI Innovation Mini-Trainer | PEFT | Unsloth | verl | Status |
|-----------|----------------------|------------------------------|------|---------|------|--------|
| **Supervised Fine-tuning (SFT)** | ✅ | - | - | - | - | Implemented |
| Continual Learning (OSFT) | 🔄 | ✅ | 🔄 | - | - | Implemented |
| **Low-Rank Adaptation (LoRA) + SFT** | - | - | - | ✅ | - | Implemented |
| **LoRA + GRPO (Adapter-Based RLVR)** | - | - | - | ✅ | ✅ | Implemented |
| **GRPO (Full Fine-Tuning RLVR)** | - | - | - | - | ✅ | Implemented |
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

### [GRPO (Full Fine-Tuning RLVR)](/algorithms/grpo)

Full-parameter GRPO training via the verl backend. Trains all model weights instead of LoRA adapters. Same data formats and reward functions as LoRA + GRPO.

```python
from training_hub import grpo

result = grpo(
    model_path="Qwen/Qwen3-8B",
    data_path="./tool_call_traces.jsonl",
    ckpt_output_dir="./grpo_full_output",
    n_gpus=8,
    num_iterations=8,
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
pip install unsloth unsloth_zoo --no-deps
```

For development:
```bash
pip install -e .[lora]
pip install unsloth unsloth_zoo --no-deps
```

> **Why `--no-deps`?** Unsloth caps `transformers<=5.5.0`, which conflicts with
> `transformers>=5.13.0` required by `kernels>=0.15.1` (CUDA extras). The cap is
> overly conservative — unsloth works correctly with transformers 5.13.x.
> Installing with `--no-deps` bypasses this conflict.

### GRPO Support
For LoRA + GRPO training with the ART backend:
```bash
pip install training-hub[grpo,lora]
pip install unsloth unsloth_zoo --no-deps
```

For the verl backend (multi-GPU FSDP), add the `grpo-verl` extra:
```bash
pip install training-hub[grpo-verl,lora]
pip install unsloth unsloth_zoo --no-deps
```

> **Note:** Install `[cuda]` extras sequentially after `[grpo]`/`[grpo-verl]` to
> avoid dependency solver conflicts.

### CUDA Support
For GPU training with CUDA support (install after other extras):
```bash
pip install training-hub[cuda] --no-build-isolation
# or for development
pip install -e .[cuda] --no-build-isolation
```

### Full Install (recommended order)

```bash
# 1. Base + algorithm extras
pip install -e ".[grpo,lora,dev]"

# 2. CUDA extras (must come after step 1)
pip install -e ".[cuda]" --no-build-isolation

# 3. Unsloth (must use --no-deps to bypass transformers cap)
pip install unsloth unsloth_zoo --no-deps
```

With uv:
```bash
uv pip install -e ".[grpo,lora,dev]"
uv pip install -e ".[cuda]" --no-build-isolation
uv pip install unsloth unsloth_zoo --no-deps
```

## Coding Agent Plugin

Training Hub is available as a plugin for two coding agents, bringing LLM training capabilities directly into your coding workflow.

<details>
<summary><strong>Claude Code</strong></summary>

**Via org marketplace** (recommended — includes all Red Hat AI plugins):
```
/plugin marketplace add Red-Hat-AI-Innovation-Team/plugins
/plugin install training-hub@Red-Hat-AI-Innovation-Team/plugins
```

**Via this repo directly:**
```
/plugin marketplace add Red-Hat-AI-Innovation-Team/training_hub
/plugin install training-hub@Red-Hat-AI-Innovation-Team/training_hub
```

**From a local clone:**
```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git
/plugin marketplace add /path/to/training_hub
```
</details>

<details>
<summary><strong>Codex CLI</strong></summary>

```bash
codex plugin marketplace add Red-Hat-AI-Innovation-Team/plugins
```

Then install the plugin from the marketplace. See `.codex-plugin/INSTALL.md` for manual installation.
</details>

### After Installing

Invoke the `setup-guide` skill to configure your training algorithm, model, and data.

| Skill | Description |
|---|---|
| `setup-guide` | Guided first-time configuration |
| `training-guide` | Run LLM training or fine-tuning |
| `memory-estimation` | Estimate GPU memory requirements |

## Getting Started

For comprehensive tutorials, examples, and documentation, see the [examples directory](./examples/).
