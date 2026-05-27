---
name: training-guide
description: "Use when the user wants to run a training job using a saved configuration. For algorithm selection, hyperparameter advice, or troubleshooting, use the training-hub-guide skill instead."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_train.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh:*)"]
---

# Run Training

Execute LLM training using a saved configuration.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh"
```

### If not ready

- `library=missing` or `config=missing`: invoke the `setup-guide` skill.
- `gpu=unavailable`: warn that training requires CUDA-capable GPUs.

### If ready (`library=installed`, `config=found`)

Proceed to Step 2.

## Step 2: Execute Training

Run the training script with any user-provided overrides:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_train.sh" $ARGUMENTS
```

## Step 3: Present Results

1. **Training status** — Whether training completed successfully
2. **Algorithm used** — Which algorithm and backend were used
3. **Checkpoint location** — Where the trained model was saved
4. **Loss summary** — If available, show final loss values

If training failed, consult the `training-hub-guide` skill for troubleshooting (OOM, loss interpretation, backend-specific issues).

Remind the user they can visualize training loss with `training_hub.plot_loss()`.
