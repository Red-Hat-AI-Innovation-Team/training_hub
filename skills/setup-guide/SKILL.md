---
name: setup-guide
description: "Use when the user wants to set up LLM training for the first time, or when training_hub is not yet installed/configured in the current environment."
---

# training_hub Setup Guide

You are helping the user set up LLM training for the first time.

## Detection

First, detect the environment by running:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh"
```

## If Nothing is Installed

1. Explain what training_hub does: "training_hub is a library for LLM post-training — it provides a unified interface for SFT, OSFT, LoRA fine-tuning, and GRPO reinforcement learning across multiple backends."
2. Ask permission: "I can install it for you. This will add the `training-hub` Python package to your environment. Want me to proceed?"
3. If yes: install using the detected installer (`uv pip install training-hub` or `pip install training-hub`)
4. Ask about optional extras:
   - CUDA: `training-hub[cuda]` — flash-attn, bitsandbytes for GPU acceleration
   - LoRA: `training-hub[lora]` — Unsloth, TRL for parameter-efficient fine-tuning
   - GRPO: `training-hub[grpo]` — ART, veRL for reinforcement learning
5. Proceed to configuration

## Configuration

Invoke the `/th-setup` command to walk through configuration:
- Algorithm selection (SFT, OSFT, LoRA, GRPO)
- Model, data, and output paths
- Hyperparameters
- Experiment tracking setup

## After Setup

Once configured, hand off to the `training-guide` skill if the user had an original training request. Otherwise, tell the user:
- "You're all set! You can now use `/th-train` to start training."
- Mention `/th-estimate` for GPU memory estimation before training.
