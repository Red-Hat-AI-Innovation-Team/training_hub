# Supervised Fine-Tuning (SFT)

> **Conceptual Overview** - For complete API reference, see [sft() Function Reference](/api/functions/sft.md)

## What is SFT?

Supervised Fine-Tuning (SFT) is the standard approach for adapting pre-trained language models to new tasks or domains using labeled training data. The model learns to generate appropriate responses by training on input-output pairs.

In Training Hub, SFT is powered by the battle-tested [InstructLab Training backend](/api/backends/instructlab-training.md), which provides production-grade support for single-GPU, multi-GPU, and multi-node distributed training.

## When to Use SFT

Use SFT when you want to:

- **Adapt a pre-trained model** to a new domain or task (e.g., medical question-answering, coding assistance)
- **Create instruction-following models** from base language models
- **Improve model performance** on specific types of queries with labeled examples
- **Fine-tune openly available models** like Llama, Qwen, or Phi on custom data

SFT works best when:
- You have high-quality labeled training data (input-output pairs)
- You want straightforward, reliable fine-tuning without specialized techniques
- You're not concerned about catastrophic forgetting from previous training

**Note:** If you need to continually train a model without forgetting previous knowledge, consider [OSFT (Orthogonal Subspace Fine-Tuning)](osft.md) instead.

## Quick Start

Here's a minimal example to get started with SFT:

```python
from training_hub import sft

# Run supervised fine-tuning
result = sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",      # Model to fine-tune
    data_path="./training_data.jsonl",          # Your training data
    ckpt_output_dir="./checkpoints",            # Where to save results
    num_epochs=3,                               # Training epochs
    effective_batch_size=8,                     # Batch size across all GPUs
    learning_rate=2e-5,                         # Learning rate
    max_seq_len=2048,                           # Max sequence length
    max_tokens_per_gpu=45000                    # GPU memory limit
)
```

Your training data should be in JSONL format with messages:

```json
{"messages": [{"role": "user", "content": "What is SFT?"}, {"role": "assistant", "content": "SFT is supervised fine-tuning..."}]}
{"messages": [{"role": "user", "content": "How do I use it?"}, {"role": "assistant", "content": "You can use the sft() function..."}]}
```

That's it! The `sft()` function handles all the complexity of distributed training, data processing, and checkpointing automatically.

### Pretraining Mode (Optional)

To train on raw documents instead of chat-formatted data, enable pretraining mode by setting `is_pretraining=True` and specifying a `block_size`.

```python
result = sft(
    model_path="Qwen/Qwen2.5-7B",
    data_path="./raw_documents.jsonl",
    ckpt_output_dir="./checkpoints",
    is_pretraining=True,
    block_size=2048,
    document_column_name="text",  # optional, defaults to "document"
)
```

Your data should be a JSONL file where each line contains document text:

```json
{"text": "First document..."}
{"text": "Second document..."}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `is_pretraining` | Yes | Set to `True` to enable pretraining mode |
| `block_size` | Yes | Number of tokens per training block (recommend starting with 2048) |
| `document_column_name` | No | Column name in JSONL file (default: `"document"`) |

## Key Concepts

### Training Data

SFT requires training data in **messages format** - conversational exchanges between user and assistant. Your training data must be a **JSON Lines (.jsonl)** file where each line contains a conversation sample:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there! How can I help you?"}]}
{"messages": [{"role": "user", "content": "What is SFT?"}, {"role": "assistant", "content": "SFT stands for Supervised Fine-Tuning..."}]}
```

**Message Structure:**
- **`role`**: One of `"system"`, `"user"`, `"assistant"`, or `"pretraining"`
- **`content`**: The text content of the message
- **`reasoning_content`** (optional): Additional reasoning traces

**Masking Control with `unmask` Field:**

The backend trains only on assistant responses by default (instruction-tuning mode):

```json
{"messages": [...]}  // Only assistant responses used for loss
{"messages": [...], "unmask": false}  // Same as above
```

For pretraining-style training where all content (except system messages) is used for loss:

```json
{"messages": [...], "unmask": true}  // All content except system messages used for loss
```

See [Data Formats](/api/data-formats.md) for complete specifications.

### Memory Management

The `max_tokens_per_gpu` parameter is crucial for managing GPU memory. It sets a hard cap on the number of tokens processed per GPU in each mini-batch. The backend automatically calculates gradient accumulation steps to achieve your desired `effective_batch_size` while staying within memory limits.

**If you encounter out-of-memory errors**, reduce `max_tokens_per_gpu`, `effective_batch_size`, or `max_seq_len`.

### Distributed Training

Training Hub automatically handles distributed training across multiple GPUs and nodes. Simply specify:

- `nproc_per_node` - GPUs per machine (auto-detected if not specified)
- `nnodes` - Total number of machines
- `node_rank` - This machine's rank (0 for master)
- `rdzv_endpoint` - Master node address (for multi-node)

The backend uses PyTorch's `torchrun` under the hood for robust distributed execution.

See [Distributed Training Guide](/guides/distributed-training.md) for complete multi-node setup instructions.

## Advanced Usage

### Using the Factory Pattern

For more control over the algorithm instance, you can use the factory pattern:

```python
from training_hub import create_algorithm

# Create an SFT algorithm instance
sft_algo = create_algorithm('sft', 'instructlab-training')

# Run training
result = sft_algo.train(
    model_path="/path/to/your/model",
    data_path="/path/to/your/training/data",
    ckpt_output_dir="/path/to/save/checkpoints",
    num_epochs=2,
    learning_rate=2e-6
)

# Check required parameters
required_params = sft_algo.get_required_params()
print("Required parameters:", list(required_params.keys()))
```

### Algorithm Discovery

Explore available algorithms and backends programmatically:

```python
from training_hub import AlgorithmRegistry

# List all available algorithms
algorithms = AlgorithmRegistry.list_algorithms()
print("Available algorithms:", algorithms)

# List backends for SFT
sft_backends = AlgorithmRegistry.list_backends('sft')
print("SFT backends:", sft_backends)

# Get algorithm class directly
SFTAlgorithm = AlgorithmRegistry.get_algorithm('sft')
```

### Error Handling

```python
from training_hub import sft, AlgorithmRegistry

try:
    result = sft(
        model_path="/valid/model/path",
        data_path="/valid/data/path",
        ckpt_output_dir="/valid/output/path"
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Training error: {e}")

# Check if algorithm/backend exists before using
if 'sft' in AlgorithmRegistry.list_algorithms():
    print("SFT algorithm is available")

if 'instructlab-training' in AlgorithmRegistry.list_backends('sft'):
    print("InstructLab Training backend is available")
```

## Next Steps

**Learn more about SFT:**
- [sft() Function Reference](/api/functions/sft.md) - Complete parameter documentation and advanced examples
- [SFTAlgorithm Class](/api/classes/SFTAlgorithm.md) - Object-oriented API for advanced use cases
- [InstructLab Training Backend](/api/backends/instructlab-training.md) - Backend implementation details

**Related topics:**
- [OSFT Algorithm](/algorithms/osft.md) - Alternative for continual learning without catastrophic forgetting
- [Data Formats](/api/data-formats.md) - Detailed data format specifications
- [Distributed Training Guide](/guides/distributed-training.md) - Multi-node training setup
- [Data Preparation Guide](/guides/data-preparation.md) - Best practices for preparing training data

**Working examples:**
- Check the [examples directory](/examples/README.md) for Jupyter notebooks and scripts demonstrating SFT in action
