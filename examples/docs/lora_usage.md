# LoRA + SFT Usage Guide

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that allows you to fine-tune large language models with significantly reduced memory requirements. Training hub implements LoRA combined with supervised fine-tuning (SFT) using the optimized Unsloth backend.

## Quick Start

### Basic LoRA Training

```python
from training_hub import lora_sft

result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./training_data.jsonl",
    ckpt_output_dir="./outputs",
    lora_r=16,              # LoRA rank
    lora_alpha=32,          # LoRA scaling parameter
    num_epochs=3,
    learning_rate=2e-4
)
```

### Single-GPU Launch

For standard single-GPU training, run your script directly with Python (same as other algorithms):

```bash
python my_training_script.py
```

### Multi-GPU Launch

**Important:** Unlike other algorithms in training-hub which handle distributed setup internally, LoRA training requires `torchrun` for multi-GPU setups due to Unsloth's distributed training requirements:

```bash
# For 4 GPUs
torchrun --nproc-per-node=4 my_training_script.py

# For 8 GPUs
torchrun --nproc-per-node=8 my_training_script.py
```

## Installation

```bash
pip install training-hub[lora]
```

This includes:
- Unsloth optimizations for 2x faster training and 70% less VRAM
- PyTorch-optimized xformers for better performance
- TRL for advanced training features

## LoRA Parameters

### Core LoRA Settings
- **`lora_r`**: LoRA rank (default: 16) - Higher values capture more information but use more memory
- **`lora_alpha`**: LoRA scaling parameter (default: 32) - Controls the magnitude of LoRA updates
- **`lora_dropout`**: Dropout rate for LoRA layers (default: 0.0) - Optimized for Unsloth
- **`target_modules`**: List of modules to apply LoRA to (default: auto-detect)

### Quantization (QLoRA)
For even lower memory usage, enable 4-bit quantization:

```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    lora_r=64,              # Higher rank for quantized model
    lora_alpha=128,
    load_in_4bit=True,      # Enable QLoRA
    learning_rate=1e-4      # Lower LR for quantized training
)
```

## Dataset Formats

LoRA training supports the same dataset formats as SFT:

### Messages Format (Recommended)
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

### Alpaca Format
```json
{
  "instruction": "Explain machine learning",
  "input": "",
  "output": "Machine learning is..."
}
```

## Memory Benefits

LoRA provides significant memory savings compared to full fine-tuning by only training low-rank adaptation matrices instead of the full model weights. The exact memory reduction depends on your specific model, LoRA configuration, and batch size settings.

## Multi-GPU Training

For distributed training across multiple GPUs:

```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./large_dataset.jsonl",
    ckpt_output_dir="./outputs",

    # LoRA settings
    lora_r=32,
    lora_alpha=64,

    # Distributed training
    effective_batch_size=128,  # Total across all GPUs
    micro_batch_size=2,        # Per GPU

    # Training settings
    num_epochs=3,
    learning_rate=2e-4
)
```

Launch with torchrun:
```bash
torchrun --nproc-per-node=4 my_script.py
```

## Performance Tips

1. **Use Unsloth optimizations** (included by default)
2. **Enable BF16** for better performance: `bf16=True`
3. **Use sample packing**: `sample_packing=True`
4. **Optimize batch sizes**: Start with `micro_batch_size=2` and adjust
5. **For large models**: Use `load_in_4bit=True` for QLoRA

## Advanced Configuration

### Custom Target Modules
```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention only
    lora_r=16,
    lora_alpha=32
)
```

### Weights & Biases Integration

**Note:** Weights & Biases is not included in the `[lora]` extras. Install it separately:
```bash
pip install wandb
```

```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    lora_r=16,
    lora_alpha=32,
    wandb_project="my-lora-project",
    wandb_entity="my-team"
)
```

## Examples

See [lora_example.py](../scripts/lora_example.py) for complete working examples including:
- Basic LoRA training
- QLoRA with 4-bit quantization
- Multi-GPU distributed training
- Different dataset format handling

## Troubleshooting

### Memory Issues
- Reduce `micro_batch_size`
- Enable `load_in_4bit=True` for QLoRA
- Lower the `lora_r` value

### Multi-GPU Issues
- Ensure you're using `torchrun` for multi-GPU (not direct Python execution)
- Check that `effective_batch_size` is divisible by `nproc_per_node * micro_batch_size`
- For very large models, try `enable_model_splitting=True`

### Installation Issues
- If xformers conflicts occur, the LoRA extras use PyTorch-optimized builds
- For CUDA version issues, try the appropriate extra: `[lora-cu129]` or `[lora-cu130]`