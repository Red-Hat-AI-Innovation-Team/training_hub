# LoRA + GRPO (Adapter-Based RLVR)

> **Conceptual Overview** - For complete API reference, see [`lora_grpo()` Function Reference](/api/functions/lora_grpo)

## What is LoRA + GRPO?

LoRA + GRPO trains LoRA adapters on tool-calling agents using Group Relative Policy Optimization (GRPO) with reinforcement learning from verifiable rewards (RLVR). Instead of supervised fine-tuning on demonstrations, the model learns by generating tool calls and receiving reward signals based on correctness.

Training Hub provides two backends for this algorithm:
- **ART** (`backend="art"`) — Single-GPU training using [OpenPipe ART](https://github.com/OpenPipe/ART) with co-located vLLM + [Unsloth](https://github.com/unslothai/unsloth) GRPO
- **verl** (`backend="verl"`) — Multi-GPU distributed training using [verl](https://github.com/volcengine/verl) with FSDP + vLLM via Ray

Both backends support single-turn and multi-turn tool-call data. Multi-turn traces are automatically decomposed into per-turn training samples, where each sample contains the ground-truth conversation prefix and the expected tool call at that turn.

## When to Use LoRA + GRPO

**Use when you want to:**
- Improve a model's tool-calling accuracy without expensive API-based training
- Train on ground-truth tool-call traces from stronger models (distillation)
- Scale RL training to larger models across multiple GPUs
- Fine-tune agents for specific tool-calling domains (retail, support, etc.)

**Works best when:**
- You have ground-truth tool-call traces (from GPT-4, Claude, etc.)
- The task involves structured tool/function calling with verifiable outputs
- You want to train LoRA adapters (not full fine-tuning) for efficiency
- You need fast iteration ($0 API cost, no user simulator required)

> **Note:** For supervised fine-tuning without RL, see [LoRA + SFT](/algorithms/lora).

## Quick Start

```python
from training_hub import lora_grpo

result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./tool_call_traces.jsonl",
    ckpt_output_dir="./grpo_output",
    num_iterations=15,
    group_size=8,
    lora_r=32,
    lora_alpha=64,
)
```

### Data Format

Two data formats are supported: **multi-turn traces** (recommended) and **pre-processed single-turn samples**.

#### Multi-Turn Traces (Recommended)

A JSONL file where each line is a full tool-calling conversation. Multi-turn traces are automatically decomposed into per-turn training samples (see [Per-Turn Decomposition](#per-turn-decomposition) below).

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Check weather alerts for California"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_abc123", "type": "function", "function": {"name": "get_alerts", "arguments": "{\"state\": \"CA\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_abc123", "name": "get_alerts", "content": "{\"alerts\": [...]}"},
    {"role": "assistant", "content": "Here are the current weather alerts..."}
  ],
  "question": "Check weather alerts for California"
}
```

Tool definitions can be provided either as a top-level `tools` field or embedded in the system message (e.g., using Qwen's `<|im_system|>tool_declare<|im_middle|>...<|im_end|>` format — both are auto-detected).

Messages must use the **modern `tool_calls`/`tool` format** (not the deprecated `function_call`/`function` format). The `messages` field can be either a JSON array or a JSON-encoded string.

#### Pre-Processed Single-Turn Samples

Alternatively, provide pre-processed samples where each line is one training example:

```json
{
  "question": "Check weather alerts for California",
  "tools": [{"type": "function", "function": {"name": "get_alerts", "parameters": {...}}}],
  "target_tool_name": "get_alerts",
  "target_arguments": {"state": "CA"},
  "system_prompt": "You are a helpful assistant."
}
```

#### HuggingFace Datasets

HuggingFace datasets like `Agent-Ark/Toucan-1.5M` are also supported. Specify the dataset ID as `data_path` and the config as `data_config` (default: `"Qwen3"`).

## Backends

### ART Backend (Single GPU)

Best for fast iteration with small models (4B-8B) on a single GPU.

```python
result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
    backend="art",
    num_iterations=15,
    tasks_per_iteration=100,
    group_size=8,
    lora_r=32,
    lora_alpha=64,
    learning_rate=1e-5,
)
```

ART uses co-located vLLM + Unsloth GRPO on the same GPU with time-sharing: vLLM generates rollouts, then sleeps while Unsloth trains the LoRA adapter using its optimized GRPO implementation, then wakes for the next rollout.

### verl Backend (Multi GPU)

Best for scaling to larger models and production training across multiple GPUs.

```python
result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
    backend="verl",
    n_gpus=4,
    num_iterations=3,
    group_size=4,
    lora_r=32,
    lora_alpha=64,
    learning_rate=1e-5,
)
```

verl uses FSDP for distributed LoRA training and vLLM for rollout generation, orchestrated by Ray. The model is sharded across GPUs during training and weights are synced to vLLM between iterations.

## Key Concepts

### GRPO (Group Relative Policy Optimization)

GRPO generates multiple rollouts (controlled by `group_size`) for each training sample and computes advantages relative to the group. This eliminates the need for a separate critic/value model (unlike PPO), reducing memory requirements.

### Per-Turn Decomposition

Multi-turn tool-call traces are decomposed into independent per-turn samples. A conversation with 5 tool calls becomes 5 training samples:

1. `[system, user] → tool_call_1` (no prior context)
2. `[system, user, gt_assistant_1, gt_tool_result_1] → tool_call_2`
3. `[system, user, gt_assistant_1, gt_tool_result_1, gt_assistant_2, gt_tool_result_2] → tool_call_3`
4. ...and so on

Each sample is scored independently using `tool_call_reward` (1.0 = correct name + args, 0.5 = correct name, 0.0 = wrong).

### Custom Reward Functions

You can provide your own reward function:

```python
def my_reward(response, expected_name, expected_args):
    # Your scoring logic
    return 1.0  # or 0.5, 0.0, etc.

result = lora_grpo(
    ...,
    reward_fn=my_reward,
)
```

For fully custom rollout logic (e.g., live environment interaction), provide a `rollout_fn`:

```python
async def my_rollout(model, task):
    client = model.openai_client()
    # ... your agentic loop ...
    trajectory = art.Trajectory(messages_and_choices=[...])
    trajectory.reward = compute_reward(...)
    return trajectory

result = lora_grpo(
    ...,
    rollout_fn=my_rollout,
    tasks=my_task_list,
)
```

> **Note:** Custom `rollout_fn` must be a top-level function (not a lambda or closure) due to subprocess pickling.

## Performance Tips

1. **Start with a small test** — Use `num_iterations=2, tasks_per_iteration=10` to verify the pipeline works before long runs
2. **Match LoRA rank to model size** — r=16-32 for 4B models, r=64-128 for larger or MoE models
3. **Lower learning rate for stability** — 5e-6 to 1e-5 works well; higher rates cause oscillation
4. **Strip thinking blocks** — For Qwen3 models, ensure `<think>` blocks are stripped from conversation history during evaluation
5. **Watch for overfitting** — With small datasets (<2000 samples), training reward can climb while eval performance degrades. Use early stopping or fewer epochs.

## Installation

```bash
pip install training-hub[grpo,lora]
```

This installs both ART and verl backends along with LoRA dependencies ([Unsloth](https://github.com/unslothai/unsloth), TRL, vLLM).

## Next Steps

**Learn more about LoRA + GRPO:**
- [`lora_grpo()` Function Reference](/api/functions/lora_grpo)
- [`LoRAGRPOAlgorithm` Class Reference](/api/classes/LoRAGRPOAlgorithm)
- [ART Backend](/api/backends/art-grpo)
- [verl Backend](/api/backends/verl)

**Related topics:**
- [Experiment Tracking & Logging](/guides/logging) — W&B and MLflow integration
- [LoRA + SFT](/algorithms/lora) — Supervised fine-tuning with LoRA (non-RL)
- [Data Preparation](/guides/data-preparation)
- [Extending the Framework](/guides/extending-framework)

**Working examples:**
- [Examples Directory](/examples/)
- [LoRA GRPO Example Script](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/lora_grpo_example.py)
