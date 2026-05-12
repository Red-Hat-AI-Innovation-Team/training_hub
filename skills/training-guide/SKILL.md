---
name: training-guide
description: "Use when the user wants to fine-tune or train a language model, run SFT/OSFT/LoRA training, estimate VRAM requirements, or interpret training results like loss curves. Applies to tasks like: full fine-tuning, parameter-efficient fine-tuning, reinforcement learning, continual learning, or GPU memory planning."
---

# LLM Training

Help the user train or fine-tune a language model.

## Detection

First, check the environment:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/th_detect.sh"
```

## Routing

Based on detection results:

### Nothing available (`library=missing`, `config=missing`)
Invoke the `setup-guide` skill to walk through installation and configuration.

### Config missing but library installed (`library=installed`, `config=missing`)
Ask the user to run `/th-setup` to configure, or invoke the `setup-guide` skill.

### Ready (`library=installed`, `config=found`)
Proceed based on the user's intent.

## Algorithm Selection

If the user hasn't specified an algorithm, help them choose:

| User says | Algorithm | Why |
|---|---|---|
| "fine-tune", "SFT", "instruction tuning" | sft | Full-weight supervised fine-tuning |
| "continual learning", "preserve knowledge", "orthogonal" | osft | Preserves pre-trained knowledge |
| "LoRA", "efficient", "low rank", "limited VRAM" | lora_sft | Parameter-efficient, lower memory |
| "GRPO", "reinforcement learning", "reward", "tool calling" | lora_grpo | RL with LoRA for efficiency |
| "full GRPO", "large scale RL", "multi-GPU RL" | grpo | Full-weight RL on GPU clusters |

## Memory Estimation

If the user asks about VRAM, memory, GPUs, or whether a model will fit, route to `/th-estimate`.

## Training Execution

For training requests, route to `/th-train`.

## Loss Interpretation

If the user asks about training progress, loss curves, or convergence:
- Suggest `training_hub.plot_loss("<ckpt_output_dir>")` to visualize
- Explain: loss should decrease over epochs; sudden spikes may indicate learning rate issues; plateaus suggest convergence

## Common Issues

| Symptom | Suggestion |
|---|---|
| OOM error | Reduce batch size, seq length, or switch to LoRA |
| Loss not decreasing | Check learning rate, data format, verify model path |
| Slow training | Check GPU utilization with `nvidia-smi`, consider flash-attn |
