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

## Algorithms

| Algorithm | Backends | GPU Support | Install Extra |
|-----------|----------|-------------|---------------|
| **SFT** | InstructLab-Training | Multi-GPU, multi-node | base |
| **OSFT** | Mini-Trainer | Multi-GPU, multi-node | base |
| **LoRA + SFT** | Unsloth | Single-GPU, multi-GPU, multi-node | `[lora]` |
| **LoRA + GRPO** | ART + Unsloth, verl | Single-GPU (ART), multi-GPU, multi-node (verl) | `[grpo,lora]` |
| **GRPO** | verl | Multi-GPU, multi-node | `[grpo]` |
| **GEPA** | GEPA, MLflow | CPU | `[gepa]` |
| **Speculative Decoding** | Speculators | Single-GPU, multi-GPU | `[speculative-decoding]` |

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
    learning_rate=2e-4,
)
```

### [LoRA + GRPO (Adapter-Based RLVR)](./algorithms/lora_grpo)

Train LoRA adapters on tool-calling agents using Group Relative Policy Optimization with reinforcement learning from verifiable rewards. Features:
- Single-turn and multi-turn tool-call verification with automatic per-turn decomposition
- Two backends: OpenPipe ART + Unsloth GRPO (single-GPU, fast iteration) and verl (multi-GPU/multi-node, scales to 70B+)
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

### [GEPA (Genetic-Pareto Prompt Optimization)](./algorithms/gepa)

Gradient-free prompt optimization using genetic algorithms with Pareto-optimal selection. Evolves system prompts to maximize task performance without modifying model weights. Features:
- Multi-objective optimization (accuracy, cost, latency)
- Works with any LLM via LiteLLM (OpenAI, Anthropic, local models)
- MLflow experiment tracking integration
- No GPU required — optimizes prompts, not weights

```python
from training_hub import gepa

result = gepa(
    model_path="gpt-4o-mini",
    data_path="./eval_data.jsonl",
    ckpt_output_dir="./gepa_output",
    population_size=10,
    generations=5,
)
```

### [Speculative Decoding (Draft Model Training)](./algorithms/speculative_decoding)

Train lightweight draft models (Eagle3, DFlash, MTP, PEagle) for speculative decoding inference acceleration using the [speculators](https://github.com/vllm-project/speculators) library. Features:
- Four pipeline modes: `offline` (bulk extract then train), `online` (extract on-demand), `train_only`, `data_only`
- Managed vLLM lifecycle or user-provided endpoints for hidden state extraction
- Tensor parallel or data parallel vLLM for multi-GPU extraction
- Single-GPU or multi-GPU training via torchrun
- GPU allocation controls (`vllm_gpu_ids`, `training_gpu_ids`)

```python
from training_hub import train_speculator

# Fully automated: data prep, hidden state extraction, training
result = train_speculator(
    verifier_name_or_path="Qwen/Qwen3-8B",
    ckpt_output_dir="./eagle3_output",
    data_path="sharegpt",
    speculator_type="eagle3",
    vllm_gpu_ids=[0, 1],       # vLLM for hidden state extraction
    training_gpu_ids=[2, 3],   # FSDP training
    epochs=3,
    draft_vocab_size=32000,
)
```

See [examples/speculative_decoding_examples.py](./examples/speculative_decoding_examples.py) for all configurations (offline/online, managed/endpoint, single/multi-GPU).

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
pip install 'training-hub[lora]'
```

### GRPO Support
For LoRA + GRPO training (both ART and verl backends):
```bash
pip install 'training-hub[grpo,lora]'
```

> **Note:** When combining `[grpo]` with `[cuda]` extras, install them sequentially
> to avoid dependency solver conflicts:
> ```bash
> pip install training-hub[grpo,lora]
> pip install training-hub[cuda]
> ```

### GEPA Support
For gradient-free prompt optimization:
```bash
pip install 'training-hub[gepa]'
```

### Speculative Decoding Support
For Eagle3 draft model training:
```bash
pip install 'training-hub[speculative-decoding]'
```

### CUDA Support
For GPU training with CUDA support:
```bash
pip install training-hub[cuda] --no-build-isolation
```

**Note:** If you encounter build issues with flash-attn, install the base package first:
```bash
pip install training-hub
pip install training-hub[cuda] --no-build-isolation
```

If you're using uv, you can use the following commands to install the package:

```bash
uv pip install training-hub && uv pip install training-hub[cuda] --no-build-isolation
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
