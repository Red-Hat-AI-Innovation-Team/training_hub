# training_hub
An algorithm-focused interface for common llm training, continual learning, and reinforcement learning techniques.

## Support Matrix

| Algorithm | InstructLab-Training | RHAI Innovation Mini-Trainer | PEFT | VERL | Status |
|-----------|---------------------|---------------|------|------|--------|
| **Supervised Fine-tuning (SFT)** | ✅ | - | - | - | Implemented |
| Continual Learning (OSFT) | 🔄 | ✅ | 🔄 | - | Implemented |
| Direct Preference Optimization (DPO) | - | - | - | 🔄 | Planned |
| Low-Rank Adaptation (LoRA) | 🔄 | - | 🔄 | - | Planned |
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

**Note:** If you encounter build issues with flash-attn, install the base package first:
```bash
# Install base package (provides torch, packaging, wheel, ninja)
pip install training-hub
# Then install with CUDA extras
pip install training-hub[cuda]

# For development installation:
pip install -e .
pip install -e .[cuda]
```

**For uv users:** You may need the `--no-build-isolation` flag:
```bash
uv pip install training-hub
uv pip install training-hub[cuda] --no-build-isolation

# For development:
uv pip install -e .
uv pip install -e .[cuda] --no-build-isolation
```

## Getting Started

For comprehensive tutorials, examples, and documentation, see the [examples directory](examples/).
