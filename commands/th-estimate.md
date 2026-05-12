---
description: "Estimate GPU memory requirements for a training configuration"
argument-hint: "[--method METHOD] [--model PATH] [--gpus N] [--seq-len N]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_estimate.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh:*)"]
---

# training-hub Memory Estimation

Estimate GPU VRAM requirements before committing to a training run.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh"
```

If `library=missing`, tell the user to install training_hub first.

## Step 2: Run Estimation

Execute the estimation script with user-provided parameters or config defaults:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_estimate.sh" $ARGUMENTS
```

## Step 3: Present Results

Parse the JSON output and present clearly:

1. **Memory estimates** — Show low/mid/high VRAM estimates in GB
2. **GPU fit** — Report whether the configuration fits on the available GPU(s)
3. **Recommendations** — If memory is tight, suggest:
   - Reduce `max_seq_len` (e.g., 4096 → 2048)
   - Reduce `effective_batch_size`
   - Switch to LoRA or QLoRA for lower memory
   - Add more GPUs for data parallelism

## Estimation Methods

| Method | For | Estimator |
|--------|-----|-----------|
| `basic` | SFT, GRPO | BasicEstimator |
| `osft` | OSFT | OSFTEstimator |
| `lora` | LoRA-SFT, LoRA-GRPO | LoRAEstimator |
| `qlora` | Quantized LoRA | QLoRAEstimator |

If no method is specified, the script infers it from the configured algorithm.
