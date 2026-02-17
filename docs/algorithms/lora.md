# Low-Rank Adaptation (LoRA) + SFT

> **Conceptual Overview** - For complete API reference, see [lora_sft() Function Reference](/api/functions/lora_sft)

## What is LoRA?

Low-Rank Adaptation (LoRA) is a **parameter-efficient fine-tuning** technique that allows you to fine-tune large language models with significantly reduced memory requirements and faster training times. Instead of updating all model parameters, LoRA trains small low-rank matrices that are added to the existing weights, drastically reducing the number of trainable parameters.

In Training Hub, LoRA is combined with supervised fine-tuning (SFT) and powered by the optimized [Unsloth backend](/api/backends/unsloth), which provides up to 2x faster training and 70% less VRAM usage compared to standard implementations.

## When to Use LoRA

Use LoRA when you want to:

- **Fine-tune large models efficiently** with limited GPU memory (e.g., 7B+ models on consumer GPUs)
- **Reduce training costs** while maintaining competitive performance
- **Experiment quickly** with multiple fine-tuning configurations
- **Create model adapters** that can be easily shared and switched

LoRA works best when:
- You have limited GPU memory or computational resources
- You want faster iteration cycles during development
- You're fine-tuning for a specific task and don't need to modify all model parameters
- You want to create multiple task-specific adapters from a single base model

**Note:** For memory-rich environments or when you need maximum performance, standard [SFT (Supervised Fine-Tuning)](/algorithms/sft) may achieve slightly better results. For continual learning without forgetting, consider [OSFT](/algorithms/osft).

## Quick Start

Here's a minimal example to get started with LoRA:

```python
from training_hub import lora_sft

# Run LoRA fine-tuning
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",  # Model to fine-tune
    data_path="./training_data.jsonl",        # Your training data
    ckpt_output_dir="./outputs",              # Where to save results
    lora_r=16,                                # LoRA rank
    lora_alpha=32,                            # LoRA scaling parameter
    num_epochs=3,                             # Training epochs
    learning_rate=2e-4                        # Learning rate
)
```

Your training data should be in JSONL format with messages (same as SFT):

```json
{"messages": [{"role": "user", "content": "What is LoRA?"}, {"role": "assistant", "content": "LoRA is parameter-efficient fine-tuning..."}]}
{"messages": [{"role": "user", "content": "How does it work?"}, {"role": "assistant", "content": "It trains low-rank adaptation matrices..."}]}
```

### Launch Requirements

**Single-GPU:**
```bash
python my_training_script.py
```

**Multi-GPU - Two Options:**

LoRA supports two different multi-GPU strategies:

1. **Data-Parallel Training (DDP)** - Each GPU holds a full copy of the model and processes different data batches. Requires `torchrun`:

```bash
# For 4 GPUs with data parallelism
torchrun --nproc-per-node=4 my_training_script.py
```

2. **Model Splitting** - For very large models that don't fit on a single GPU (e.g., 70B models). The model is split across GPUs. No torchrun needed:

```python
result = lora_sft(
    model_path="meta-llama/Llama-3.1-70B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    enable_model_splitting=True,  # Splits model across available GPUs
    lora_r=16,
    lora_alpha=32
)
```

```bash
# Launch with standard Python (no torchrun)
python my_training_script.py
```

**When to use which:**
- **Data-Parallel (torchrun)**: Model fits on one GPU, you want faster training with multiple GPUs processing data in parallel
- **Model Splitting**: Model is too large for a single GPU, you need to distribute the model weights across GPUs

## Key Concepts

### Low-Rank Decomposition

LoRA works by decomposing weight updates into low-rank matrices. Instead of updating a large weight matrix W directly, LoRA learns two smaller matrices A and B such that:

```
W_updated = W_original + A × B
```

Where A and B have much smaller dimensions than W, dramatically reducing trainable parameters and memory usage.

### LoRA Parameters

**`lora_r` (LoRA Rank)**
- Controls the rank of the low-rank matrices
- Higher values capture more information but use more memory
- Typical values: 8, 16, 32, 64
- **Start with 16** and increase if needed

**`lora_alpha` (LoRA Alpha)**
- Scaling parameter that controls the magnitude of LoRA updates
- Often set to 2× the rank (e.g., alpha=32 for rank=16)
- Higher values = stronger adaptation

**`lora_dropout`**
- Dropout rate for LoRA layers (default: 0.0)
- Optimized for Unsloth backend

**`target_modules`**
- Which model modules to apply LoRA to
- Auto-detected if not specified
- Common choices: attention layers, all linear layers

### QLoRA (Quantized LoRA)

For even greater memory savings, use 4-bit quantization with QLoRA:

```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    lora_r=64,              # Higher rank for quantized model
    lora_alpha=128,
    load_in_4bit=True,      # Enable QLoRA
    learning_rate=1e-4      # Lower LR for quantized training
)
```

This enables fine-tuning of very large models (20B+) on consumer GPUs with 24GB VRAM.

### Dataset Formats

LoRA training supports multiple dataset formats:

**Messages Format (Recommended):**
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

**Alpaca Format:**
```json
{
  "instruction": "Explain machine learning",
  "input": "",
  "output": "Machine learning is..."
}
```

### Memory Benefits

LoRA dramatically reduces memory requirements:
- **Trainable Parameters**: Typically 0.1-1% of full model parameters
- **VRAM Usage**: 30-70% reduction with Unsloth optimizations
- **Training Speed**: Up to 2x faster with Unsloth

Exact savings depend on your model, LoRA configuration (rank, target modules), and batch size settings.

## Advanced Usage

### Multi-GPU Data-Parallel Training

For data-parallel training across multiple GPUs (each GPU processes different data batches):

```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
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

### Model Splitting for Large Models

For models too large to fit on a single GPU, use model splitting instead of data parallelism:

```python
result = lora_sft(
    model_path="meta-llama/Llama-3.1-70B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",

    # Enable model splitting across GPUs
    enable_model_splitting=True,

    # LoRA settings (lower rank may be needed for very large models)
    lora_r=16,
    lora_alpha=32,
    load_in_4bit=True,  # QLoRA recommended for large models

    # Training settings
    num_epochs=1,
    learning_rate=1e-4,
    micro_batch_size=1
)
```

Launch with standard Python (no torchrun needed):
```bash
python my_script.py
```

**Note:** Model splitting is slower than data parallelism but allows training models that exceed single-GPU memory capacity.

### Custom Target Modules

Specify which modules to apply LoRA to:

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

### Logging & Experiment Tracking

LoRA training supports multiple logging backends for experiment tracking. You can use any combination of these backends simultaneously:

- **MLflow** - Open-source platform for ML lifecycle management
- **Weights & Biases (wandb)** - Cloud-based experiment tracking
- **TensorBoard** - Visualization toolkit for training metrics

#### MLflow Integration

**Note:** MLflow is not included in the `[lora]` extras. Install separately:

```bash
pip install mlflow
```

Then configure MLflow logging:

```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    lora_r=16,
    lora_alpha=32,
    # MLflow configuration
    mlflow_tracking_uri="http://localhost:5000",  # Or your MLflow server
    mlflow_experiment_name="lora-experiments",
    mlflow_run_name="qwen-lora-run"
)
```

You can also configure MLflow via environment variables:
- `MLFLOW_TRACKING_URI` - Tracking server URI
- `MLFLOW_EXPERIMENT_NAME` - Experiment name

#### Weights & Biases Integration

**Note:** Weights & Biases is not included in the `[lora]` extras. Install separately:

```bash
pip install wandb
```

Then configure wandb logging:

```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    lora_r=16,
    lora_alpha=32,
    # Weights & Biases configuration
    wandb_project="my-lora-project",
    wandb_entity="my-team",
    wandb_run_name="qwen-lora-run"
)
```

#### TensorBoard Integration

**Note:** TensorBoard is not included in the `[lora]` extras. Install separately:

```bash
pip install tensorboard
```

Then configure TensorBoard logging:

```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    lora_r=16,
    lora_alpha=32,
    # TensorBoard configuration
    tensorboard_log_dir="./logs/tensorboard"
)
```

View logs with: `tensorboard --logdir=./logs/tensorboard`

#### Using Multiple Loggers

You can enable multiple logging backends simultaneously:

```python
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./outputs",
    lora_r=16,
    lora_alpha=32,
    # Enable all logging backends
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="lora-experiments",
    wandb_project="my-lora-project",
    tensorboard_log_dir="./logs/tensorboard"
)
```

### Performance Tips

1. **Use Unsloth optimizations** (included by default)
2. **Enable BF16** for better performance: `bf16=True`
3. **Use sample packing**: `sample_packing=True`
4. **Optimize batch sizes**: Start with `micro_batch_size=2` and adjust
5. **For large models**: Use `load_in_4bit=True` for QLoRA

## Troubleshooting

### Memory Issues
- Reduce `micro_batch_size`
- Enable `load_in_4bit=True` for QLoRA
- Lower the `lora_r` value
- Reduce `max_seq_len`

### Multi-GPU Issues
- **Data-Parallel (DDP)**: Ensure you're using `torchrun` (not direct Python execution)
- **Model Splitting**: Use `enable_model_splitting=True` with standard Python (no torchrun)
- Don't mix both approaches - use either DDP or model splitting, not both
- For DDP: Check that `effective_batch_size` is divisible by `nproc_per_node * micro_batch_size`
- For model splitting: Reduce `micro_batch_size` if you hit OOM errors

### Installation Issues
- If xformers conflicts occur, the LoRA extras use PyTorch-optimized builds
- For CUDA version issues, try the appropriate extra: `[lora-cu129]` or `[lora-cu130]`

## Installation

To use LoRA, install with the LoRA extras:

```bash
pip install training-hub[lora]
```

This includes:
- Unsloth optimizations for 2x faster training and 70% less VRAM
- PyTorch-optimized xformers for better performance
- TRL for advanced training features

## Next Steps

**Learn more about LoRA:**
- [lora_sft() Function Reference](/api/functions/lora_sft) - Complete parameter documentation and advanced examples
- [LoRASFTAlgorithm Class](/api/classes/LoRASFTAlgorithm) - Object-oriented API for advanced use cases
- [Unsloth Backend](/api/backends/unsloth) - Backend implementation details

**Related topics:**
- [SFT Algorithm](/algorithms/sft) - Standard full fine-tuning
- [OSFT Algorithm](/algorithms/osft) - Continual learning without forgetting
- [Data Formats](/api/data-formats) - Detailed data format specifications

**Working examples:**
- Check the [examples directory](/examples/) for Jupyter notebooks and scripts demonstrating LoRA in action
- See [lora_example.py](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/lora_example.py) for complete working examples
