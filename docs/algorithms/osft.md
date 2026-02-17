# Orthogonal Subspace Fine-Tuning (OSFT)

> **Conceptual Overview** - For complete API reference, see [osft() Function Reference](/api/functions/osft)

## What is OSFT?

Orthogonal Subspace Fine-Tuning (OSFT) is a specialized training algorithm that enables **continual learning without catastrophic forgetting**. Based on research by Nayak et al. (2025) ([arXiv:2504.07097](https://arxiv.org/abs/2504.07097)), OSFT allows you to adapt pre-trained or instruction-tuned models to new tasks while preserving their original capabilities.

The key innovation: OSFT learns in a subspace **orthogonal** to the model's existing knowledge, preventing interference with previously learned information. This eliminates the need for supplementary datasets to maintain the original model's distribution.

In Training Hub, OSFT is powered by the [RHAI Innovation Mini-Trainer backend](/api/backends/mini-trainer), which provides efficient orthogonal subspace computation with support for distributed training.

## When to Use OSFT

Use OSFT when you want to:

- **Continually adapt models** to new domains without forgetting previous training
- **Customize instruction-tuned models** with domain-specific knowledge (e.g., adding medical expertise to a general assistant)
- **Train on small datasets** while preserving the model's general capabilities
- **Avoid catastrophic forgetting** that occurs with standard fine-tuning

OSFT works best when:
- You're adapting an already-trained model (pre-trained or instruction-tuned)
- You want to preserve the model's existing capabilities
- You don't have access to the original training data
- Your new training dataset is relatively small

**Note:** If you're doing initial training or have a large dataset and don't need to preserve previous knowledge, standard [SFT (Supervised Fine-Tuning)](/algorithms/sft) may be simpler and faster.

## Quick Start

Here's a minimal example to get started with OSFT:

```python
from training_hub import osft

# Run orthogonal subspace fine-tuning
result = osft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",  # Model to adapt
    data_path="./medical_qa.jsonl",                 # Your new training data
    ckpt_output_dir="./checkpoints",                # Where to save results
    unfreeze_rank_ratio=0.25,                       # How much to adapt (0.1-0.5)
    effective_batch_size=16,                        # Batch size
    max_tokens_per_gpu=2048,                        # GPU memory limit
    max_seq_len=2048,                               # Max sequence length
    learning_rate=2e-5                              # Learning rate
)
```

Your training data uses the same JSONL messages format as SFT:

```json
{"messages": [{"role": "user", "content": "What is diabetes?"}, {"role": "assistant", "content": "Diabetes is a condition..."}]}
{"messages": [{"role": "user", "content": "How is it treated?"}, {"role": "assistant", "content": "Treatment includes..."}]}
```

The model will learn the new medical domain while retaining its general conversational abilities.

## Data Format Requirements

OSFT supports both **processed** and **unprocessed** data formats via the mini-trainer backend.

### Standard Messages Format (Recommended)

Your training data should be a **JSON Lines (.jsonl)** file containing messages:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there! How can I help you?"}]}
{"messages": [{"role": "user", "content": "What is OSFT?"}, {"role": "assistant", "content": "OSFT stands for Orthogonal Subspace Fine-Tuning..."}]}
```

**Message Structure:**
- **`role`**: One of `"system"`, `"user"`, `"assistant"`, or `"pretraining"`
- **`content`**: The text content of the message
- **`reasoning_content`** (optional): Additional reasoning traces

**Masking Control with `unmask_messages` Parameter:**

Standard instruction tuning (default) - only assistant responses used for loss:
```python
osft(..., unmask_messages=False)  # Default
```

Instruction-style pretraining (unmask all conversational content except system messages):
```python
osft(..., unmask_messages=True)
```

### Pretraining Mode

To train on raw documents instead of chat-formatted data, enable pretraining mode by setting `is_pretraining=True` and specifying a `block_size`.

Your data should be a JSONL file where each line contains document text:

```json
{"document": "First document..."}
{"document": "Second document..."}
```

```python
result = osft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./domain_documents.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=2048,
    learning_rate=2e-5,

    # Enable pretraining mode
    is_pretraining=True,
    block_size=2048,
    document_column_name="text",  # optional; defaults to "document"
)
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `is_pretraining` | Yes | Set to `True` to enable pretraining mode |
| `block_size` | Yes | Number of tokens per training block (recommend starting with 2048) |
| `document_column_name` | No | Column name in JSONL file (default: `"document"`) |

### Pre-processed Dataset Format

If you have pre-processed data with `input_ids` and `labels` fields:

```json
{"input_ids": [1, 2, 3, ...], "labels": [1, 2, 3, ...]}
{"input_ids": [4, 5, 6, ...], "labels": [4, 5, 6, ...]}
```

Use with:
```python
osft(..., use_processed_dataset=True)
```

## Key Concepts

### Orthogonal Subspace Learning

OSFT works by identifying the subspace where the model's existing knowledge resides, then learning new information in a direction **orthogonal** (perpendicular) to that subspace. This mathematical property ensures new learning doesn't interfere with old learning.

Think of it like writing on a new sheet of paper instead of erasing and rewriting on the same sheet - both pieces of information coexist without conflict.

### Unfreeze Rank Ratio

The `unfreeze_rank_ratio` parameter (0.0-1.0) controls how much of each weight matrix is adapted during training:

- **0.1-0.3**: Conservative adaptation, minimal changes to the model (recommended for small datasets)
- **0.3-0.5**: Moderate adaptation, balanced preservation and learning
- **>0.5**: Aggressive adaptation (rarely needed, approaches standard fine-tuning)

**Start with 0.25** and adjust based on your needs. Higher values allow more adaptation but slightly increase the risk of forgetting.

### Use Cases

**Example 1: Domain Specialization**
- Start: General instruction-tuned model (e.g., Llama 3.1)
- New data: Medical question-answering pairs
- Result: Model with medical expertise + original general capabilities

**Example 2: Continual Learning**
- Start: Model trained on Task A
- New data: Task B examples
- Result: Model that can handle both Task A and Task B

**Example 3: Low-Resource Adaptation**
- Start: Pre-trained language model
- New data: 500 examples in a new language/domain
- Result: Model with new capabilities without corrupting base knowledge

### Memory Considerations

OSFT has similar memory requirements to standard SFT. If you encounter out-of-memory errors during model loading, use:

```python
result = osft(
    # ... other parameters ...
    osft_memory_efficient_init=True  # Reduces memory during initialization
)
```

For general memory management, adjust `max_tokens_per_gpu`, `effective_batch_size`, or `max_seq_len`.

## Advanced Usage

### Using the Factory Pattern

For more control over the algorithm instance, you can use the factory pattern:

```python
from training_hub import create_algorithm

# Create an OSFT algorithm instance
osft_algo = create_algorithm('osft', 'mini-trainer')

# Run training
result = osft_algo.train(
    model_path="/path/to/your/model",
    data_path="/path/to/your/training/data.jsonl",
    ckpt_output_dir="/path/to/save/outputs",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=6,
    max_tokens_per_gpu=3072,
    max_seq_len=2048,
    learning_rate=1.5e-5,
    num_epochs=2
)

# Check required parameters
required_params = osft_algo.get_required_params()
print("Required parameters:", list(required_params.keys()))
```

### Algorithm Discovery

Explore available algorithms and backends programmatically:

```python
from training_hub import AlgorithmRegistry

# List all available algorithms
algorithms = AlgorithmRegistry.list_algorithms()
print("Available algorithms:", algorithms)

# List backends for OSFT
osft_backends = AlgorithmRegistry.list_backends('osft')
print("OSFT backends:", osft_backends)

# Get algorithm class directly
OSFTAlgorithm = AlgorithmRegistry.get_algorithm('osft')
```

### Error Handling

```python
from training_hub import osft, AlgorithmRegistry

try:
    result = osft(
        model_path="/valid/model/path",
        data_path="/valid/data/path",
        ckpt_output_dir="/valid/output/path",
        unfreeze_rank_ratio=0.3,
        effective_batch_size=8,
        max_tokens_per_gpu=2048,
        max_seq_len=2048,
        learning_rate=2e-5
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Training error: {e}")

# Check if algorithm/backend exists before using
if 'osft' in AlgorithmRegistry.list_algorithms():
    print("OSFT algorithm is available")

if 'mini-trainer' in AlgorithmRegistry.list_backends('osft'):
    print("Mini-trainer backend is available")
```

### Best Practices

1. **unfreeze_rank_ratio**: Start with values between 0.1-0.5. Values >0.5 are rarely needed for general continual-learning regimes.

2. **Memory Management**: OSFT doesn't reduce memory requirements compared to SFT, so adjust `max_tokens_per_gpu` accordingly. For memory-constrained environments or OOMs during model loading, set `osft_memory_efficient_init=True`.

3. **Data Processing**: The algorithm handles data processing automatically. Use `use_processed_dataset=True` only if you have pre-tokenized data.

4. **Continual Learning**: OSFT is particularly effective for adapting instruction-tuned models to new domains without catastrophic forgetting.

## Next Steps

**Learn more about OSFT:**
- [osft() Function Reference](/api/functions/osft) - Complete parameter documentation and advanced examples
- [OSFTAlgorithm Class](/api/classes/OSFTAlgorithm) - Object-oriented API for advanced use cases
- [Mini-Trainer Backend](/api/backends/mini-trainer) - Backend implementation details

**Related topics:**
- [SFT Algorithm](/algorithms/sft) - Standard fine-tuning alternative
- [Data Formats](/api/data-formats) - Detailed data format specifications
- [Distributed Training Guide](/guides/distributed-training) - Multi-node training setup

**Research:**
- [Original OSFT Paper](https://arxiv.org/abs/2504.07097) - Nayak et al. (2025) - Mathematical foundations and empirical results

**Working examples:**
- Check the [examples directory](/examples/) for Jupyter notebooks and scripts demonstrating OSFT in action
