# training_hub
An algorithm-focused interface for common llm training, continual learning, and reinforcement learning techniques.

## Support Matrix

| Algorithm | InstructLab-Training | RHAI Innovation Mini-Trainer | PEFT | VERL | Status |
|-----------|---------------------|---------------|------|------|--------|
| **Supervised Fine-tuning (SFT)** | ✅ | - | - | - | Implemented |
| Continual Learning (OSFT) | 🔄 | ✅ | 🔄 | - | Implemented |
| **Low-Rank Adaptation (LoRA) + SFT** | - | - | ✅ | - | Implemented |
| Direct Preference Optimization (DPO) | - | - | - | 🔄 | Planned |
| Group Relative Policy Optimization (GRPO) | - | - | - | 🔄 | Planned |

**Legend:**
- ✅ Implemented and tested
- 🔄 Planned for future implementation  
- \- Not applicable or not planned

## Implemented Algorithms

### [Supervised Fine-tuning (SFT)](examples/docs/sft_usage.md)

Fine-tune language models on supervised datasets with support for:
- Single-node and multi-node distributed training
- Configurable training parameters (epochs, batch size, learning rate, etc.)
- InstructLab-Training backend integration

```python
from training_hub import sft

result = sft(
    model_path="/path/to/model",
    data_path="/path/to/data",
    ckpt_output_dir="/path/to/checkpoints",
    num_epochs=3,
    learning_rate=1e-5
)
```

### [Orthogonal Subspace Fine-Tuning (OSFT)](examples/docs/osft_usage.md)

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

### [Low-Rank Adaptation (LoRA) + SFT](examples/docs/lora_usage.md)

Parameter-efficient fine-tuning using LoRA with supervised fine-tuning. Features:
- Memory-efficient training with significantly reduced VRAM requirements
- Single-GPU and multi-GPU distributed training support
- Unsloth backend for 2x faster training and 70% less memory usage
- Support for QLoRA (4-bit quantization) for even lower memory usage
- Compatible with messages and Alpaca dataset formats

```python
from training_hub import lora_sft

result = lora_sft(
    model_path="meta-llama/Llama-2-7b-hf",
    data_path="/path/to/data.jsonl",
    ckpt_output_dir="/path/to/outputs",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    learning_rate=2e-4
)
```

## Installation

### Basic Installation
```bash
pip install training-hub
```

### Development Installation
```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/training_hub
cd training_hub
pip install -e .
```

### CUDA Support
For GPU training with CUDA support:
```bash
pip install training-hub[cuda]
# or for development
pip install -e .[cuda]
```

### LoRA Support
For LoRA training with optimized dependencies:
```bash
pip install training-hub[lora]
# or for development
pip install -e .[lora]
```

**Note:** The LoRA extras include Unsloth optimizations and PyTorch-optimized xformers for better performance and compatibility.

**Note:** If you encounter build issues with flash-attn, install the base package first:
```bash
# Install base package (provides torch, packaging, wheel, ninja)
pip install training-hub
# Then install with CUDA extras
pip install training-hub[cuda] --no-build-isolation

# For development installation:
pip install -e .
pip install -e .[cuda] --no-build-isolation
```

**For uv users:** Use the same pattern with uv
```bash
uv pip install training-hub
uv pip install training-hub[cuda] --no-build-isolation

# For development:
uv pip install -e .
uv pip install -e .[cuda] --no-build-isolation
```

## Getting Started

For comprehensive tutorials, examples, and documentation, see the [examples directory](examples/).
