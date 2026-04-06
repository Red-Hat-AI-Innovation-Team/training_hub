# Training Hub Examples

This directory contains documentation, tutorials, and examples for using training_hub algorithms.

> **Looking for documentation?** Visit the comprehensive [Training Hub Documentation](https://ai-innovation.team/training_hub) for guides, API reference, and algorithm overviews.

## Directory Structure

- **`notebooks/`** - Interactive Jupyter notebooks with step-by-step tutorials
- **`scripts/`** - Standalone Python scripts for automation and examples

## Supported Algorithms

### Supervised Fine-Tuning (SFT)

The SFT algorithm supports training language models on supervised datasets with both single-node and multi-node distributed training capabilities.

**Documentation:**
- [SFT Usage Guide](https://ai-innovation.team/training_hub/#/algorithms/sft) - Comprehensive usage documentation with parameter reference and examples

**Tutorials:**
- [LAB Multi-Phase Training Tutorial](notebooks/lab_multiphase_training_tutorial.ipynb) - Interactive notebook demonstrating LAB multi-phase training workflow
- [SFT Comprehensive Tutorial](notebooks/sft_comprehensive_tutorial.ipynb) - Interactive notebook covering all SFT parameters with popular model examples
- [SFT with Ministral 3 3B (Medical)](notebooks/runnable_ministral_sft.ipynb) - End-to-end SFT notebook: medical flashcards dataset, training, and inference verification
- [SFT Continued Pretraining on Spreadsheets](notebooks/sft_cpt_spreadsheet.ipynb) - Interactive notebook demonstrating continued pretraining on Excel spreadsheet data

**Scripts:**
- [LAB Multi-Phase Training Script](scripts/lab_multiphase_training.py) - Example script for LAB multi-phase training with full command-line interface
- [SFT with Qwen 2.5 7B](scripts/sft_qwen_example.py) - Single-node multi-GPU training example with Qwen 2.5 7B Instruct
- [SFT with Llama 3.1 8B](scripts/sft_llama_example.py) - Single-node multi-GPU training example with Llama 3.1 8B Instruct
- [SFT with Phi 4 Mini](scripts/sft_phi_example.py) - Single-node multi-GPU training example with Phi 4 Mini Instruct
- [SFT with GPT-OSS 20B](scripts/sft_gpt_oss_example.py) - Single-node multi-GPU training example with GPT-OSS 20B
- [SFT with Granite 3.3 8B](scripts/sft_granite_example.py) - Single-node multi-GPU training example with Granite 3.3 8B Instruct
- [SFT with Granite 4.0](scripts/sft_granite4_example.py) - Single-node multi-GPU training example with Granite 4.0 models
- [SFT with Ministral 3 3B (Medical)](scripts/sft_ministral_medical_example.py) - Medical domain fine-tuning with Ministral 3 3B Instruct
- [SFT Continued Pretraining on Spreadsheets](scripts/sft_cpt_spreadsheet_example.py) - End-to-end continued pretraining example using SpreadsheetBench Excel data

**Quick Example:**
```python
from training_hub import sft

result = sft(
    model_path="/path/to/model",
    data_path="/path/to/data",
    ckpt_output_dir="/path/to/checkpoints",
    num_epochs=3,
    learning_rate=2e-5,
    max_tokens_per_gpu=45000
)
```

### Orthogonal Subspace Fine-Tuning (OSFT)

The OSFT algorithm supports continual training of pre-trained or instruction-tuned models without requiring supplementary datasets to maintain the original model distribution. Based on [Nayak et al. (2025)](https://arxiv.org/abs/2504.07097), it enables efficient customization while preventing catastrophic forgetting.

**Documentation:**
- [OSFT Usage Guide](https://ai-innovation.team/training_hub/#/algorithms/osft) - Comprehensive usage documentation with parameter reference and examples

**Tutorials:**
- [OSFT Comprehensive Tutorial](notebooks/osft_comprehensive_tutorial.ipynb) - Interactive notebook covering all OSFT parameters with popular model examples
- [OSFT Continual Learning](notebooks/osft_continual_learning.ipynb) - Interactive notebook demonstrating continual learning capabilities
- [OSFT Multi-Phase Training Tutorial](notebooks/osft_multiphase_training_tutorial.ipynb) - Interactive notebook demonstrating OSFT multi-phase training workflow
- [OSFT with Ministral 3 3B (Medical)](notebooks/runnable_ministral_osft.ipynb) - End-to-end OSFT notebook: medical flashcards, training, inference, and knowledge retention test
- [OSFT Continued Pretraining on Spreadsheets](notebooks/osft_cpt_spreadsheet.ipynb) - Interactive notebook demonstrating OSFT continued pretraining on Excel spreadsheet data

**Scripts:**
- [OSFT Multi-Phase Training Script](scripts/osft_multiphase_training.py) - Example script for OSFT multi-phase training with full command-line interface
- [OSFT with Qwen 2.5 7B](scripts/osft_qwen_example.py) - Single-node multi-GPU training example with Qwen 2.5 7B Instruct
- [OSFT with Llama 3.1 8B](scripts/osft_llama_example.py) - Single-node multi-GPU training example with Llama 3.1 8B Instruct
- [OSFT with Phi 4 Mini](scripts/osft_phi_example.py) - Single-node multi-GPU training example with Phi 4 Mini Instruct
- [OSFT with GPT-OSS 20B](scripts/osft_gpt_oss_example.py) - Single-node multi-GPU training example with GPT-OSS 20B
- [OSFT with Granite 3.3 8B](scripts/osft_granite_example.py) - Single-node multi-GPU training example with Granite 3.3 8B Instruct
- [OSFT Continual Learning Example](scripts/osft_continual_learning_example.py) - Example script demonstrating continual learning without catastrophic forgetting
- [OSFT with Ministral 3 3B (Medical)](scripts/osft_ministral_medical_example.py) - Medical domain OSFT with Ministral 3 3B Instruct
- [OSFT Continued Pretraining on Spreadsheets](scripts/osft_cpt_spreadsheet_example.py) - End-to-end OSFT continued pretraining example using SpreadsheetBench Excel data

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
- [LoRA Usage Guide](https://ai-innovation.team/training_hub/#/algorithms/lora) - Comprehensive usage documentation with parameter reference and examples

**Tutorials:**
- [LoRA with Ministral 3 3B (Medical)](notebooks/runnable_ministral_lora.ipynb) - End-to-end LoRA notebook: medical flashcards, training, and inference verification

**Scripts:**
- [LoRA Example](scripts/lora_example.py) - Basic LoRA training examples with different configurations and dataset formats
- [LoRA with Ministral 3 3B (Medical)](scripts/lora_ministral_medical_example.py) - Medical domain LoRA fine-tuning with Ministral 3 3B Instruct

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

### LoRA + GRPO (Adapter-Based RLVR)

LoRA + GRPO trains LoRA adapters on tool-calling agents using reinforcement learning from verifiable rewards. It supports both single-turn and multi-turn tool-call data, with automatic per-turn decomposition of multi-turn traces. Two backends are available: OpenPipe ART + Unsloth GRPO for single-GPU and verl for multi-GPU training.

**Documentation:**
- [LoRA + GRPO Usage Guide](https://ai-innovation.team/training_hub/#/algorithms/lora_grpo) - Comprehensive usage documentation with parameter reference and examples

**Scripts:**
- [LoRA GRPO Example](scripts/lora_grpo_example.py) - Tool-call GRPO training with ART backend, supports HuggingFace datasets and local JSONL

**Quick Example:**
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

### Memory Estimation (Experimental / In-Development)

training_hub includes a library for estimating the expected amount of GPU memory that will be allocated during the fine-tuning of a given model using SFT or OSFT. The calculations are built off of formulas presented in the blog post [How To Calculate GPU VRAM Requirements for an Large-Language Model](https://apxml.com/posts/how-to-calculate-vram-requirements-for-an-llm).
NOTE: This feature is still a work in-progress. In particular, the given estimates for OSFT may vary from your actual results; the estimate mainly serves to give theoretical bounds.  
The estimates for SFT should be reasonably close to actual results when using training_hub, but keep in mind that your actual results may still vary. 

**Tutorials:**
- [Memory Estimation Example](notebooks/memory_estimator_example.ipynb) - Interactive notebook showcasing how to utilize the memory estimator methods.

**Quick Example:**
```python
from training_hub import estimate

estimate(training_method='osft',
    num_gpus=2,
    model_path="/path/to/model",
    max_tokens_per_gpu=8192,
    use_liger=True,
    verbose=2,
    unfreeze_rank_ratio: float = 0.25
)
```

### Training Loss Visualization

training_hub includes a `plot_loss` utility for visualizing training loss curves after running SFT or OSFT training. This is useful for monitoring training progress, comparing different experiments, and identifying issues like overfitting.

**Tutorials:**
- [Plot Loss Example](notebooks/plot_loss_example.ipynb) - Interactive notebook demonstrating loss visualization features

**Quick Example:**
```python
from training_hub import sft, plot_loss

# After training
sft(model_path="...", ckpt_output_dir="./checkpoints", ...)

# Plot and save loss curve
plot_loss("./checkpoints")

# Compare multiple runs with EMA smoothing
plot_loss(
    ["./run1", "./run2", "./run3"],
    labels=["baseline", "lr=1e-5", "lr=5e-6"],
    ema=True
)
```

### Model Interpolation (Experimental / In-Development)

training_hub has a utility for merging two checkpoints of the same model into one with linear interpolation.

**Script:**
- [interpolator.py](scripts/interpolator.py) - Python script for model interpolation

**Command-Line Example:**
```bash
python interpolator.py --model-path /path/to/base/model --trained-model-path /path/to/trained/checkpoint
```

**Python Example:**
```python
from interpolator import interpolate_models

interpolate_models("/path/to/base/model", "/path/to/trained/checkpoint")
```

### Model Validation (Development / QA)

Validates that model architectures can train successfully with SFT and OSFT by overfitting on a single sample.

**Script:** [model_validation.py](../scripts/model_validation.py)

```bash
python scripts/model_validation.py --models llama --mode sft
python scripts/model_validation.py --run-all --mode both --liger-variants
python scripts/model_validation.py --list-models
```

## Getting Started

1. **For detailed parameter documentation**: Visit the [Training Hub Documentation](https://ai-innovation.team/training_hub)
2. **For hands-on learning**: Open the interactive notebooks in `notebooks/`
3. **For automation scripts**: Refer to examples in `scripts/`
