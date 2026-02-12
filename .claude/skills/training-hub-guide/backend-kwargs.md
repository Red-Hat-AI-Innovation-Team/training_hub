# Backend Kwargs and Hidden Features

Training Hub's algorithm functions (`sft()`, `osft()`, `lora_sft()`) expose a curated set of parameters. However, every backend supports additional options that are not directly surfaced in Training Hub's API. Any unrecognized keyword argument passed to an algorithm function is forwarded directly to the underlying backend.

## How it works

```python
from training_hub import osft

# 'osft' parameter is not in Training Hub's API but is forwarded to mini-trainer
result = osft(
    model_path="...",
    data_path="...",
    ckpt_output_dir="...",
    unfreeze_rank_ratio=0.5,
    effective_batch_size=128,
    max_tokens_per_gpu=2048,
    max_seq_len=2048,
    learning_rate=5e-6,
    osft=False,  # forwarded to mini-trainer: runs plain SFT through the OSFT backend
)
```

## Backend API references

To see all available options for each backend, consult their source definitions:

### OSFT (mini-trainer)

- **Training entry point**: https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/main/src/mini_trainer/api_train.py
- **Full parameter definitions**: https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/main/src/mini_trainer/training_types.py

Notable hidden options:
- `osft=False`: Run plain SFT through the OSFT backend without orthogonal subspace constraints
- Validation loss configuration (for convergence monitoring)
- Training duration modes including "infinite" mode for indefinite training

Note: W&B, MLflow, and TensorBoard logging are now first-class parameters on all algorithms (see SKILL.md "Experiment tracking" section) and no longer need to be passed as backend kwargs.

### SFT (instructlab-training)

- **Config definitions**: https://github.com/instructlab/training/blob/main/src/instructlab/training/config.py
- **Training entry point**: https://github.com/instructlab/training/blob/main/src/instructlab/training/main_ds.py

Note: The SFT backend does not currently support validation loss (planned for future release).

### LoRA (Unsloth)

The LoRA backend uses Unsloth's `FastLanguageModel` and HuggingFace's `SFTTrainer` from TRL. Additional parameters from `SFTTrainer` and `TrainingArguments` can be passed through as kwargs.

Notable options:
- `sample_packing`: Enable sample packing for efficiency
- `bf16`: Enable bfloat16 training

Note: W&B, MLflow, and TensorBoard logging are now first-class parameters on all algorithms and no longer need to be passed as backend kwargs.

## Validation loss

For determining optimal training duration, validation loss is more reliable than train loss.

| Backend | Validation loss support | How to enable |
|---------|------------------------|---------------|
| SFT (instructlab-training) | Not yet supported | Planned |
| OSFT (mini-trainer) | Supported | Pass appropriate kwargs from `TrainingArgs` (see mini-trainer source) |
| LoRA (Unsloth/TRL) | Supported | Configure via `SFTTrainer` evaluation kwargs |

When validation loss is available, look for the point where it stops decreasing or begins increasing. Training beyond this point leads to overfitting: the train loss continues to drop but the model's actual performance degrades.
