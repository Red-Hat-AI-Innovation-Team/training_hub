# `LoRAGRPOAlgorithm` - LoRA + GRPO Algorithm

> Adapter-based reinforcement learning from verifiable rewards for tool-calling agents using LoRA and Group Relative Policy Optimization.

## Class Signature

```python
class LoRAGRPOAlgorithm(Algorithm):
    def __init__(self, backend: Backend, **kwargs): ...
    def train(self, model_path, ckpt_output_dir, **kwargs) -> dict: ...
    def get_required_params(self) -> Dict[str, Type]: ...
    def get_optional_params(self) -> Dict[str, Type]: ...
```

## Overview

`LoRAGRPOAlgorithm` implements the GRPO training loop for tool-calling agents with LoRA parameter-efficient training. It supports two modes:

1. **Built-in tool-call verification** — Provide a dataset with tool-call traces, and the algorithm handles rollout generation, reward computation, and training automatically.
2. **Custom rollout** — Provide your own async rollout function for arbitrary environments.

## Backends

| Backend | Class | Use Case |
|---------|-------|----------|
| `art` | [`ARTLoRAGRPOBackend`](/api/backends/art-grpo) | Single-GPU with OpenPipe ART + Unsloth GRPO |
| `verl` | [`VeRLLoRAGRPOBackend`](/api/backends/verl) | Multi-GPU with FSDP + vLLM via Ray |

## Usage

```python
from training_hub import create_algorithm

# Using default backend (art)
algo = create_algorithm("lora_grpo")

# Using specific backend
algo = create_algorithm("lora_grpo", "verl")

# Train
result = algo.train(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
)
```

Most users should use the [`lora_grpo()`](/api/functions/lora_grpo) convenience function instead of the class-based API.

## See Also

- [`lora_grpo()` Function Reference](/api/functions/lora_grpo)
- [Algorithm Guide](/algorithms/lora_grpo)
- [`Algorithm` Base Class](/api/classes/Algorithm)
