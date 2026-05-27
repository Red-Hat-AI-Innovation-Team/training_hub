---
name: training-guide
description: "Use when the user wants to fine-tune or train a language model, run SFT/OSFT/LoRA training, or interpret training results like loss curves. Applies to tasks like: full fine-tuning, parameter-efficient fine-tuning, reinforcement learning, continual learning, or running a training job."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_train.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/th_estimate.sh:*)"]
---

# LLM Training

Help the user train or fine-tune a language model.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh"
```

### If not ready

- `library=missing` and `config=missing`: invoke the `setup-guide` skill.
- `library=installed` and `config=missing`: tell the user to run the `setup-guide` skill to configure.
- `gpu=unavailable`: warn that training requires CUDA-capable GPUs.

### If ready (`library=installed`, `config=found`)

Proceed to Step 2.

## Step 2: Algorithm Selection

If the user hasn't specified an algorithm, help them choose:

| User says | Algorithm | Why |
|---|---|---|
| "fine-tune", "SFT", "instruction tuning" | sft | Full-weight supervised fine-tuning |
| "continual learning", "preserve knowledge", "orthogonal" | osft | Preserves pre-trained knowledge |
| "LoRA", "efficient", "low rank", "limited VRAM" | lora_sft | Parameter-efficient, lower memory |
| "GRPO", "reinforcement learning", "reward", "tool calling" | lora_grpo | RL with LoRA for efficiency |
| "full GRPO", "large scale RL", "multi-GPU RL" | grpo | Full-weight RL on GPU clusters |

### Memory Questions

If the user asks about VRAM, memory, GPUs, or whether a model will fit, invoke the `memory-estimation` skill instead.

## Step 3: Execute Training

Run the training script with any user-provided overrides:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_train.sh" $ARGUMENTS
```

## Step 4: Present Results

1. **Training status** — Whether training completed successfully
2. **Algorithm used** — Which algorithm and backend were used
3. **Checkpoint location** — Where the trained model was saved
4. **Loss summary** — If available, show final loss values

If training failed, show the error and suggest troubleshooting:

| Symptom | Suggestion |
|---|---|
| OOM error | Reduce batch size, seq length, or switch to LoRA |
| Loss not decreasing | Check learning rate, data format, verify model path |
| Slow training | Check GPU utilization with `nvidia-smi`, consider flash-attn |

## Loss Interpretation

If the user asks about training progress, loss curves, or convergence:
- Suggest `training_hub.plot_loss("<ckpt_output_dir>")` to visualize
- Explain: loss should decrease over epochs; sudden spikes may indicate learning rate issues; plateaus suggest convergence
