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

| Algorithm | Backends | GPU Support | Install Extra |
|-----------|----------|-------------|---------------|
| **Supervised Fine-tuning (SFT)** | InstructLab-Training | Multi-GPU, multi-node | base |
| **Continual Learning (OSFT)** | RHAI Innovation Mini-Trainer | Multi-GPU, multi-node | base |
| **Low-Rank Adaptation (LoRA) + SFT** | Unsloth | Single-GPU, multi-GPU, multi-node | `[lora]` |
| **LoRA + GRPO (Adapter-Based RLVR)** | ART + Unsloth, verl | Single-GPU (ART), multi-GPU, multi-node (verl) | `[grpo,lora]` |
| **GRPO (Full Fine-Tuning RLVR)** | verl | Multi-GPU, multi-node | `[grpo]` |
| **GEPA (Genetic-Pareto Prompt Optimization)** | GEPA, MLflow | CPU (API-based) | `[gepa]` |
| **Embedding Fine-Tuning** | SentenceTransformers | Single-GPU, multi-GPU, CPU | `[embedding]` |

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

### [GEPA (Genetic-Pareto Prompt Optimization)](/algorithms/gepa)

Gradient-free prompt optimization using evolutionary search with Pareto-based selection and LLM-driven reflection. GEPA evolves textual prompts to maximize task performance **without modifying model weights**, so it needs no local GPU — it optimizes prompts by calling an LLM endpoint (hosted API or local vLLM/OpenAI-compatible server via `api_base`). Features:
- Genetic-Pareto search with LLM reflection to propose improved prompts
- Works with any model reachable through LiteLLM (hosted APIs or local endpoints)
- Two backends: `gepa` (direct `gepa.optimize()`) and `mlflow` (MLflow prompt registry, scorers, and tracking)

```python
from training_hub import gepa

result = gepa(
    seed_candidate={"system_prompt": "You are a helpful assistant. Answer the question."},
    task_lm="openai/gpt-4o-mini",
    data_path="./eval_data.jsonl",
    output_dir="./gepa_output",
    reflection_lm="openai/gpt-4o",
    max_metric_calls=200,
)
```

### [Embedding Fine-Tuning](/algorithms/embedding_sft)

Contrastive fine-tuning of sentence embedding models (e.g. `all-MiniLM-L6-v2`) so that inputs with the same label cluster together in embedding space. Designed for **semantic routing / classification** — route a query to one of N specialist lanes by nearest-anchor cosine similarity — but applicable to any task that benefits from tighter embedding clusters (retrieval, deduplication, clustering). Features:
- Three contrastive losses: `batch_all_triplet`, `batch_hard_triplet`, and `mnrl` (Multiple Negatives Ranking Loss)
- `GROUP_BY_LABEL` batch sampling so every batch contains all classes (required for triplet mining)
- Auto-converts label datasets to (anchor, positive) pairs for MNRL
- Custom `loss_fn` support for extensibility
- Saves in standard sentence-transformers format

```python
from training_hub import embedding_sft

result = embedding_sft(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    data_path="routing_train.jsonl",    # {"text": "...", "label": 0}
    ckpt_output_dir="./routing_model",
    loss_type="batch_all_triplet",
    num_epochs=20,
    batch_size=32,
    learning_rate=2e-5,
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

> **Note:** When combining `[grpo]` with `[cuda]` extras, install them sequentially
> to avoid dependency solver conflicts:
> ```bash
> pip install training-hub[grpo,lora]
> pip install training-hub[cuda]
> ```
> The `[grpo]` extras constrain torch, vllm, and transformers versions for verl
> compatibility, which may conflict with versions pulled by `[cuda]`. Sequential
> installation lets the solver pick compatible versions.

### GEPA Support
For gradient-free prompt optimization (includes the MLflow backend):
```bash
pip install training-hub[gepa]
# or for development
pip install -e .[gepa]
```

**Note:** GEPA optimizes prompts via LLM API calls and does not require CUDA. To
optimize against a local model, run a vLLM (or other OpenAI-compatible) server and
pass its URL via the `api_base` parameter.

### Embedding Support
For contrastive embedding fine-tuning (sentence-transformers backend):
```bash
pip install training-hub[embedding]
# or for development
pip install -e .[embedding]
```

**Note:** Embedding fine-tuning uses `sentence-transformers>=5.0`. It runs on CPU
for small models (e.g. `all-MiniLM-L6-v2`, 23M params) and accelerates on a single
or multi-GPU when CUDA is available.

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
