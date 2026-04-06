# API Reference

Complete reference documentation for all Training Hub APIs.

## Quick Reference

### Core Functions

Training Hub provides convenient top-level functions for common training tasks:

| Function | Purpose | Learn More |
|----------|---------|------------|
| [`sft()`](/api/functions/sft) | Supervised fine-tuning of language models | [Details](/api/functions/sft) |
| [`osft()`](/api/functions/osft) | Orthogonal subspace fine-tuning for continual learning | [Details](/api/functions/osft) |
| [`lora_sft()`](/api/functions/lora_sft) | Parameter-efficient fine-tuning with LoRA | [Details](/api/functions/lora_sft) |
| [`lora_grpo()`](/api/functions/lora_grpo) | LoRA + GRPO for tool-calling agents (RLVR) | [Details](/api/functions/lora_grpo) |
| [`create_algorithm()`](/api/functions/create-algorithm) | Factory function to create algorithm instances | [Details](/api/functions/create-algorithm) |

### Classes

Training Hub uses an object-oriented architecture with algorithms and pluggable backends:

| Class | Purpose | Learn More |
|-------|---------|------------|
| [`Algorithm`](/api/classes/Algorithm) | Abstract base class for all training algorithms | [Details](/api/classes/Algorithm) |
| [`Backend`](/api/classes/Backend) | Abstract base class for backend implementations | [Details](/api/classes/Backend) |
| [`SFTAlgorithm`](/api/classes/SFTAlgorithm) | Supervised fine-tuning algorithm implementation | [Details](/api/classes/SFTAlgorithm) |
| [`OSFTAlgorithm`](/api/classes/OSFTAlgorithm) | OSFT algorithm implementation | [Details](/api/classes/OSFTAlgorithm) |
| [`LoRASFTAlgorithm`](/api/classes/LoRASFTAlgorithm) | LoRA fine-tuning algorithm implementation | [Details](/api/classes/LoRASFTAlgorithm) |
| [`LoRAGRPOAlgorithm`](/api/classes/LoRAGRPOAlgorithm) | LoRA + GRPO for tool-calling agents | [Details](/api/classes/LoRAGRPOAlgorithm) |
| [`PEFTExtender`](/api/classes/PEFTExtender) | Base class for parameter-efficient fine-tuning extensions | [Details](/api/classes/PEFTExtender) |
| [`LoRAPEFTExtender`](/api/classes/LoRAPEFTExtender) | LoRA-specific PEFT extension implementation | [Details](/api/classes/LoRAPEFTExtender) |
| [`AlgorithmRegistry`](/api/classes/AlgorithmRegistry) | Central registry for algorithms and backends | [Details](/api/classes/AlgorithmRegistry) |

### Backends

Backend implementations that power the algorithms:

| Backend | Algorithm Support | Learn More |
|---------|------------------|------------|
| [`InstructLabTrainingSFTBackend`](/api/backends/instructlab-training) | SFT | [Details](/api/backends/instructlab-training) |
| [`MiniTrainerOSFTBackend`](/api/backends/mini-trainer) | OSFT | [Details](/api/backends/mini-trainer) |
| [`UnslothLoRABackend`](/api/backends/unsloth) | LoRA | [Details](/api/backends/unsloth) |
| [`ARTLoRAGRPOBackend`](/api/backends/art-grpo) | LoRA + GRPO (single-GPU) | [Details](/api/backends/art-grpo) |
| [`VeRLLoRAGRPOBackend`](/api/backends/verl) | LoRA + GRPO (multi-GPU) | [Details](/api/backends/verl) |

For an overview of the backend system, see [Backends Overview](/api/backends/).

## Getting Started with the API

### For Most Users: Use the Convenience Functions

If you're fine-tuning models, use the simple function-based API:

```python
from training_hub import sft

# Supervised fine-tuning
result = sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./training_data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=8,
    learning_rate=2e-5,
    max_seq_len=256,
    max_tokens_per_gpu=1024,
)
```

```python
from training_hub import osft

# Continual learning with OSFT
result = osft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./training_data.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=5e-6
)
```

```python
from training_hub import lora_sft

# Parameter-efficient fine-tuning with LoRA
result = lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./training_data.jsonl",
    ckpt_output_dir="./checkpoints",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    learning_rate=2e-4
)
```

See:
- [sft() Function Reference](/api/functions/sft)
- [osft() Function Reference](/api/functions/osft)
- [lora_sft() Function Reference](/api/functions/lora_sft)

### For Advanced Users: Use the Class-Based API

For more control, use the algorithm classes directly:

```python
from training_hub import SFTAlgorithm, InstructLabTrainingSFTBackend

# Create backend
backend = InstructLabTrainingSFTBackend()

# Create algorithm instance
algorithm = SFTAlgorithm(backend=backend)

# Train with full control
result = algorithm.train(
    model_path="/path/to/model",
    data_path="/path/to/data",
    ckpt_output_dir="/path/to/output",
    num_epochs=5,
    effective_batch_size=16,
    learning_rate=2e-5,
    max_seq_len=256,
    max_tokens_per_gpu=1024,
)
```

```python
from training_hub import LoRASFTAlgorithm, UnslothLoRABackend

# Create backend
backend = UnslothLoRABackend()

# Create algorithm instance
algorithm = LoRASFTAlgorithm(backend=backend)

# Train with LoRA
result = algorithm.train(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="/path/to/data",
    ckpt_output_dir="/path/to/output",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    learning_rate=2e-4
)
```

See:
- [SFTAlgorithm Class Reference](/api/classes/SFTAlgorithm)
- [OSFTAlgorithm Class Reference](/api/classes/OSFTAlgorithm)
- [LoRASFTAlgorithm Class Reference](/api/classes/LoRASFTAlgorithm)

### For Framework Developers: Extend the Framework

Create custom algorithms and backends:

```python
from training_hub import Algorithm, Backend, AlgorithmRegistry

class MyCustomBackend(Backend):
    def execute_training(self, algorithm_params):
        # Your implementation
        pass

class MyCustomAlgorithm(Algorithm):
    def train(self, **kwargs):
        # Your implementation
        pass

    def get_required_params(self):
        return {"model_path": str, "data_path": str}

    def get_optional_params(self):
        return {"num_epochs": int}

# Register your algorithm
AlgorithmRegistry.register_algorithm("my_algo", MyCustomAlgorithm)
AlgorithmRegistry.register_backend("my_algo", "my_backend", MyCustomBackend)
```

See:
- [Extending the Framework Guide](/guides/extending-framework)
- [Algorithm Class Reference](/api/classes/Algorithm)
- [Backend Class Reference](/api/classes/Backend)
- [AlgorithmRegistry Reference](/api/classes/AlgorithmRegistry)

## Data Formats

All training functions expect data in specific formats. See [Data Formats](/api/data-formats) for detailed specifications.

## Additional Resources

- [Distributed Training Guide](/guides/distributed-training) - Multi-node training setup
- [Data Preparation Guide](/guides/data-preparation) - Best practices for preparing training data
- [Examples](/examples/) - Working code examples and tutorials
