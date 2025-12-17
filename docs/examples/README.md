# Training Hub Examples

This section provides documentation, tutorials, and examples for using training_hub algorithms.

## Directory Structure

The examples are located in the repository's `examples/` directory:

- **`notebooks/`** - Interactive Jupyter notebooks with step-by-step tutorials
- **`scripts/`** - Standalone Python scripts for automation and examples

## Supported Algorithms

### Supervised Fine-Tuning (SFT)

The SFT algorithm supports training language models on supervised datasets with both single-node and multi-node distributed training capabilities.

**Documentation:**
- [SFT Usage Guide](/algorithms/sft) - Comprehensive usage documentation with parameter reference and examples

**Tutorials:**
- [LAB Multi-Phase Training Tutorial](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/lab_multiphase_training_tutorial.ipynb) - Interactive notebook demonstrating LAB multi-phase training workflow
- [SFT Comprehensive Tutorial](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/sft_comprehensive_tutorial.ipynb) - Interactive notebook covering all SFT parameters with popular model examples
- [LAB Checkpoint Selection](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/lab_checkpoint_selection.ipynb) - Selecting the best checkpoints from LAB training

**Scripts:**
- [LAB Multi-Phase Training Script](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/lab_multiphase_training.py) - Example script for LAB multi-phase training with full command-line interface
- [SFT with Qwen 2.5 7B](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/sft_qwen_example.py) - Single-node multi-GPU training example
- [SFT with Llama 3.1 8B](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/sft_llama_example.py) - Single-node multi-GPU training example
- [SFT with Phi 4 Mini](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/sft_phi_example.py) - Single-node multi-GPU training example
- [SFT with GPT-OSS 20B](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/sft_gpt_oss_example.py) - Single-node multi-GPU training example
- [SFT with Granite 3.3 8B](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/sft_granite_example.py) - Single-node multi-GPU training example
- [SFT with Granite 4.0](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/sft_granite4_example.py) - Single-node multi-GPU training example

**Quick Example:**
```python
from training_hub import sft

result = sft(
    model_path="/path/to/model",
    data_path="/path/to/data",
    ckpt_output_dir="/path/to/checkpoints",
    num_epochs=3,
    effective_batch_size=8,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000
)
```

### Orthogonal Subspace Fine-Tuning (OSFT)

The OSFT algorithm supports continual training of pre-trained or instruction-tuned models without requiring supplementary datasets to maintain the original model distribution. Based on [Nayak et al. (2025)](https://arxiv.org/abs/2504.07097), it enables efficient customization while preventing catastrophic forgetting.

**Documentation:**
- [OSFT Usage Guide](/algorithms/osft) - Comprehensive usage documentation with parameter reference and examples

**Tutorials:**
- [OSFT Comprehensive Tutorial](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/osft_comprehensive_tutorial.ipynb) - Interactive notebook covering all OSFT parameters with popular model examples
- [OSFT Continual Learning](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/osft_continual_learning.ipynb) - Interactive notebook demonstrating continual learning capabilities
- [OSFT Multi-Phase Training Tutorial](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/osft_multiphase_training_tutorial.ipynb) - Interactive notebook demonstrating OSFT multi-phase training workflow
- [OSFT Dataset Scaling Guide](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/osft_dataset_scaling_guide.ipynb) - Understanding dataset size impact

**Scripts:**
- [OSFT Multi-Phase Training Script](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/osft_multiphase_training.py) - Example script for OSFT multi-phase training with full command-line interface
- [OSFT with Qwen 2.5 7B](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/osft_qwen_example.py) - Single-node multi-GPU training example
- [OSFT with Llama 3.1 8B](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/osft_llama_example.py) - Single-node multi-GPU training example
- [OSFT with Phi 4 Mini](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/osft_phi_example.py) - Single-node multi-GPU training example
- [OSFT with GPT-OSS 20B](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/osft_gpt_oss_example.py) - Single-node multi-GPU training example
- [OSFT with Granite 3.3 8B](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/osft_granite_example.py) - Single-node multi-GPU training example
- [OSFT Continual Learning Example](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/osft_continual_learning_example.py) - Example script demonstrating continual learning without catastrophic forgetting

**Quick Example:**
```python
from training_hub import osft

result = osft(
    model_path="/path/to/model",
    data_path="/path/to/data.jsonl",
    ckpt_output_dir="/path/to/outputs",
    unfreeze_rank_ratio=0.3,
    effective_batch_size=8,
    max_tokens_per_gpu=2048,
    max_seq_len=2048,
    learning_rate=2e-5
)
```

### Low-Rank Adaptation (LoRA) + SFT

LoRA provides parameter-efficient fine-tuning with significantly reduced memory requirements by training low-rank adaptation matrices instead of the full model weights. Training hub implements LoRA with supervised fine-tuning using the optimized Unsloth backend.

**Documentation:**
- [LoRA Usage Guide](/algorithms/lora) - Comprehensive usage documentation with parameter reference and examples

**Scripts:**
- [LoRA Example](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/lora_example.py) - Basic LoRA training examples with different configurations and dataset formats

**Launch Requirements:**
- **Single-GPU**: Standard Python launch: `python my_script.py`
- **Multi-GPU (Data-Parallel)**: For data-parallel training, use torchrun: `torchrun --nproc-per-node=4 my_script.py`
- **Multi-GPU (Model Splitting)**: For large models that don't fit on one GPU, use `enable_model_splitting=True` with standard Python launch

**Quick Example:**
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

### Memory Estimation (Experimental)

training_hub includes a library for estimating the expected amount of GPU memory that will be allocated during fine-tuning.

**Note:** This feature is still in-development. Estimates for OSFT may vary from actual results; the estimate mainly serves to give theoretical bounds. Estimates for SFT should be reasonably close to actual results when using training_hub.

**Tutorials:**
- [Memory Estimation Example](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/memory_estimator_example.ipynb) - Interactive notebook showcasing how to utilize the memory estimator methods

**Quick Example:**
```python
from training_hub import estimate

estimate(
    training_method='osft',
    num_gpus=2,
    model_path="/path/to/model",
    max_tokens_per_gpu=8192,
    use_liger=True,
    verbose=2,
    unfreeze_rank_ratio=0.25
)
```

### Model Interpolation (Experimental)

training_hub has a utility for merging two checkpoints of the same model into one with linear interpolation.

**Script:**
- [interpolator.py](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/interpolator.py) - Python script for model interpolation

**Command-Line Example:**
```bash
python interpolator.py --model-path /path/to/base/model --trained-model-path /path/to/trained/checkpoint
```

**Python Example:**
```python
from interpolator import interpolate_models

interpolate_models("/path/to/base/model", "/path/to/trained/checkpoint")
```

## Getting Started

1. **For detailed parameter documentation**: Check the relevant algorithm guide
2. **For hands-on learning**: Clone the repository and open the interactive notebooks in `examples/notebooks/`
3. **For automation scripts**: Refer to examples in `examples/scripts/`

## Repository

All examples are available in the [Training Hub GitHub repository](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/tree/main/examples).
