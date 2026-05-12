---
description: "Run LLM training using saved configuration"
argument-hint: "[--algorithm ALG] [--data PATH] [--model PATH] [--gpus N]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_train.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh:*)"]
---

# training-hub Train

Run LLM training using the saved configuration.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh"
```

If `config=missing`, tell the user to run `/th-setup` first.

If `gpu=unavailable`, warn: "No GPU detected. Training requires CUDA-capable GPUs."

## Step 2: Execute Training

Run the training script with any user-provided overrides:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_train.sh" $ARGUMENTS
```

## Step 3: Present Results

Parse the output and present it clearly:

1. **Training status** — Whether training completed successfully
2. **Algorithm used** — Which algorithm and backend were used
3. **Checkpoint location** — Where the trained model was saved
4. **Loss summary** — If available, show final loss values

If training failed, show the error and suggest troubleshooting:
- OOM: Suggest reducing batch size, sequence length, or switching to LoRA
- Data format: Check JSONL structure
- Model not found: Verify the model path or HuggingFace ID

Remind the user they can visualize training loss with `training_hub.plot_loss()`.
