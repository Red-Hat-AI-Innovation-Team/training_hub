---
name: setup-guide
description: "Use when the user wants to set up LLM training for the first time, or when training_hub is not yet installed/configured in the current environment."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_estimate.sh:*)"]
---

# training_hub Setup Guide

You are helping the user set up LLM training.

## Step 1: Detect Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh"
```

## Step 2: Install if Needed

If `library=missing`:
- Explain: "training_hub is a library for LLM post-training — it provides a unified interface for SFT, OSFT, LoRA fine-tuning, and GRPO reinforcement learning across multiple backends."
- Ask permission: "I can install it for you. Want me to proceed?"
- If yes and `installer=uv`: run `uv pip install training-hub`
- If yes and `installer=pip`: run `pip install training-hub`
- If `installer=none`: tell the user they need Python and pip/uv installed first
- Ask about extras:
  - CUDA: `training-hub[cuda]` — flash-attn, bitsandbytes for GPU acceleration
  - LoRA: `training-hub[lora]` — Unsloth, TRL for parameter-efficient fine-tuning
  - GRPO: `training-hub[grpo]` — ART, veRL for reinforcement learning

## Step 3: Check GPU

If `gpu=unavailable`, warn: "No GPU detected. Training requires CUDA-capable GPUs. You can still configure, but training will fail without a GPU."

Report GPU count if available.

## Step 4: Collect Configuration

Ask these questions **one at a time**:

1. **Algorithm**: "Which training algorithm do you want to use?"
   - **SFT** — Supervised fine-tuning. Full-weight training with InstructLab backend. Best for: general instruction tuning with sufficient VRAM.
   - **OSFT** — Orthogonal supervised fine-tuning. Preserves pre-trained knowledge while learning new tasks. Best for: continual learning, avoiding catastrophic forgetting.
   - **LoRA-SFT** — Low-Rank Adaptation fine-tuning with Unsloth backend. Best for: parameter-efficient fine-tuning when VRAM is limited.
   - **LoRA-GRPO** — LoRA with Group Relative Policy Optimization. Best for: reinforcement learning with tool-calling or reward signals.
   - **GRPO** — Full-weight Group Relative Policy Optimization with veRL backend. Best for: large-scale RL training on multi-GPU clusters.

2. **Model path**: "What's the model identifier?" — e.g., `meta-llama/Llama-3.1-8B-Instruct`, or a local path.

3. **Data path**: "Where is your training data?" — Path to a JSONL file with `messages` field.

4. **Output directory**: "Where should checkpoints be saved?" — Default: `./output`

5. **GPU count**: "How many GPUs should be used?" — Default: detected count or 1.

## Step 5: Algorithm-Specific Config

Based on the algorithm choice:

**SFT:**
- Ask: "Learning rate?" (default: 5e-6)
- Ask: "Number of epochs?" (default: 2)
- Ask: "Effective batch size?" (default: 64)
- Ask: "Max sequence length?" (default: 4096)

**OSFT:**
- Same base hyperparams as SFT
- Ask: "Unfreeze rank ratio?" (default: 0.5) — higher preserves less pre-trained knowledge

**LoRA-SFT:**
- Same base hyperparams as SFT
- Ask: "LoRA rank (r)?" (default: 16)
- Ask: "LoRA alpha?" (default: 32)
- Ask: "LoRA target modules?" (default: all linear layers)

**LoRA-GRPO / GRPO:**
- Same base hyperparams
- Ask: "Reward function?" — `tool_call_reward` (for tool-calling tasks) or `binary_reward` (for pass/fail tasks)
- Ask: "Number of generations per prompt?" (default: 4)

## Step 6: Logging Config (Optional)

Ask: "Do you want to configure experiment tracking?"
- **W&B**: project name, entity
- **MLflow**: tracking URI
- **TensorBoard**: log directory
- Skip if not needed

## Step 7: Save Config

Write the config to `.training-hub/config.json`:

```json
{
  "algorithm": "<algorithm>",
  "model_path": "<model_path>",
  "data_path": "<data_path>",
  "ckpt_output_dir": "<output_dir>",
  "nproc_per_node": N,
  "hyperparams": {
    "learning_rate": 5e-6,
    "num_epochs": 2,
    "effective_batch_size": 64,
    "max_seq_len": 4096
  },
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

Report the estimates and suggest adjustments if memory is tight (reduce batch size, sequence length, or switch to LoRA).

## Updating Config

If this skill is invoked again and a config already exists, ask: "You already have a configuration. Do you want to update it or start fresh?"
