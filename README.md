# training_hub
An algorithm-focused interface for common llm training, continual learning, and reinforcement learning techniques.

## Support Matrix

| Algorithm | InstructLab-Training | PEFT | VERL | Status |
|-----------|---------------------|------|------|--------|
| **Supervised Fine-tuning (SFT)** | ✅ | - | - | Implemented |
| Continual Learning (OSFT) | 🔄 | 🔄 | - | Planned |
| Direct Preference Optimization (DPO) | - | - | 🔄 | Planned |
| Low-Rank Adaptation (LoRA) | 🔄 | 🔄 | - | Planned |
| Group Relative Policy Optimization (GRPO) | - | - | 🔄 | Planned |

**Legend:**
- ✅ Implemented and tested
- 🔄 Planned for future implementation  
- \- Not applicable or not planned

## Implemented Algorithms

### [Supervised Fine-tuning (SFT)](examples/sft_usage.md)
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

**Note:** If you encounter build issues with flash-attn, install torch first:
```bash
pip install torch
pip install training-hub[cuda]
```

## Getting Started
