---
name: setup-guide
description: "Use when the user wants to set up LLM training for the first time, or when training_hub is not yet installed/configured in the current environment."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_estimate.sh:*)"]
---

# training_hub Setup Guide

You are helping the user set up LLM training. For algorithm selection guidance, hyperparameter tuning, and troubleshooting, consult the `training-hub-guide` skill.

## Step 1: Detect Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh"
```

## Step 2: Install if Needed

If `library=missing`:
- Ask permission: "training_hub isn't installed. I can install it for you — want me to proceed?"
- If yes and `installer=uv`: run `uv pip install training-hub`
- If yes and `installer=pip`: run `pip install training-hub`
- If `installer=none`: tell the user they need Python and pip/uv installed first
- Ask about extras:
  - CUDA: `training-hub[cuda]` — flash-attn, bitsandbytes for GPU acceleration
  - LoRA: `training-hub[lora]` — Unsloth, TRL for parameter-efficient fine-tuning
  - GRPO: `training-hub[grpo]` — ART, veRL for reinforcement learning

For installation issues, consult the `training-hub-guide` skill (installation-troubleshooting section).

## Step 3: Check GPU

If `gpu=unavailable`, warn: "No GPU detected. Training requires CUDA-capable GPUs. You can still configure, but training will fail without a GPU."

Report GPU count if available.

## Step 4: Quick Setup or Custom

If the user has a clear task ("fine-tune Llama on my data"), offer a fast path with sensible defaults:

> "I detected N GPU(s). I can set up with these defaults:
> - Algorithm: `lora_sft` (parameter-efficient, works on a single GPU)
> - Learning rate: `1e-5`
> - Epochs: `2`
> - Batch size: `64`
> - Max sequence length: `4096`
>
> You'll just need to provide your model path and data path. Accept these defaults, or customize?"

If the user accepts, ask only for model path and data path, then skip to Step 7.

**If the user wants to customize**, proceed with the full configuration.

## Full Configuration

Ask these questions **one at a time**:

1. **Algorithm**: "Which training algorithm do you want to use?" — consult the `training-hub-guide` skill for algorithm selection guidance if the user is unsure.
2. **Model path**: "What's the model identifier?" — e.g., `meta-llama/Llama-3.1-8B-Instruct`, or a local path.
3. **Data path**: "Where is your training data?" — Path to a JSONL file with `messages` field.
4. **Output directory**: "Where should checkpoints be saved?" — Default: `./output`
5. **GPU count**: "How many GPUs should be used?" — Default: detected count or 1.

## Step 5: Hyperparameters

Collect hyperparameters based on the chosen algorithm. Consult the `training-hub-guide` skill (hyperparameter-guide section) for recommended defaults by dataset size and algorithm.

## Step 6: Logging Config (Optional)

Ask: "Do you want to configure experiment tracking?" (W&B, MLflow, or TensorBoard). See the `training-hub-guide` skill for logger details.

## Step 7: Save Config

Write the config to `.training-hub/config.json`:

```json
{
  "algorithm": "<algorithm>",
  "model_path": "<model_path>",
  "data_path": "<data_path>",
  "ckpt_output_dir": "<output_dir>",
  "nproc_per_node": N,
  "hyperparams": { ... },
  "algorithm_config": { ... },
  "logging": { ... }
}
```

Add `.training-hub/` to `.gitignore` if not already present.

## Step 8: Offer Memory Estimation

Ask: "Want me to estimate GPU memory requirements before training?"

If yes, run:
```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_estimate.sh"
```

## Updating Config

If this skill is invoked again and a config already exists, ask: "You already have a configuration. Do you want to update it or start fresh?"
