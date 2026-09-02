# Embedding SFT Backend

> Contrastive embedding fine-tuning. The [`SentenceTransformersBackend`](#) powers the [`embedding_sft()`](/api/functions/embedding_sft) algorithm.

## Overview

**Class:** `SentenceTransformersBackend`

**Algorithm Support:** Embedding SFT (contrastive embedding fine-tuning)

**Package:** `sentence-transformers>=5.0`

**Status:** Implemented

The backend wraps the `sentence-transformers` library's `SentenceTransformerTrainer`, exposing configurable contrastive losses and batch samplers. It fine-tunes a sentence-transformers model so that same-label inputs cluster together in embedding space, then saves the result in standard sentence-transformers format.

## Usage

```python
from training_hub import embedding_sft

result = embedding_sft(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    data_path="routing_train.jsonl",
    ckpt_output_dir="./routing_model",
    loss_type="batch_all_triplet",
    num_epochs=20,
    batch_size=32,
    learning_rate=2e-5,
)
```

### Via Class-Based API

```python
from training_hub import EmbeddingSFTAlgorithm, SentenceTransformersBackend

backend = SentenceTransformersBackend()
algorithm = EmbeddingSFTAlgorithm(backend=backend)
result = algorithm.train(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    data_path="routing_train.jsonl",
    ckpt_output_dir="./routing_model",
)
```

## Losses

The backend supports three contrastive losses (select via `loss_type`, or pass `loss_fn` for a custom one):

| `loss_type` | Loss class | Description |
|-------------|-----------|-------------|
| `batch_all_triplet` (default) | `BatchAllTripletLoss` | Mines all valid triplets per batch. Best for exhaustive boundary learning. |
| `batch_hard_triplet` | `BatchHardTripletLoss` | Mines the hardest triplet per anchor. Focused on worst-case boundaries. |
| `mnrl` | `MultipleNegativesRankingLoss` | In-batch negative ranking loss. General semantic similarity. |

For `mnrl`, the backend auto-converts `{"text", "label"}` datasets into `(anchor, positive)` pairs (all within-label combinations, capped at 10,000 pairs per label) and defaults the batch sampler to `no_duplicates`.

## Batch Samplers

| `batch_sampler` | When used | Description |
|-----------------|-----------|-------------|
| `group_by_label` | default for triplet losses | Every batch contains all classes — required for triplet mining. |
| `no_duplicates` | default for MNRL | No duplicate texts per batch. |
| `default` | — | Standard batching. |

A triplet loss with a non-`group_by_label` sampler logs a warning — batches may lack classes, degrading triplet mining.

## Data Format

JSONL (one object per line) with a text field and an integer label field (CSV and HuggingFace dataset IDs are also accepted):

```json
{"text": "What is the cabin pressure trend?", "label": 0}
{"text": "How much propellant remains?", "label": 1}
```

Column names default to `text` / `label`; override with `text_column` / `label_column`.

## Precision

The backend auto-enables `bf16` when a CUDA GPU with bf16 support is available, otherwise trains in fp32. Small models like `all-MiniLM-L6-v2` (23M params) also run on CPU.

## Output

- The fine-tuned model is saved to `ckpt_output_dir` in standard sentence-transformers format — reload with `SentenceTransformer("<ckpt_output_dir>")`.
- `training_metrics.jsonl` is appended with the final step / loss / epoch.

## Inference

The backend trains an **embedding model**, not a classifier. At inference, classification is a thin user-supplied layer: encode the query and a set of labeled anchor texts, then pick the class with the highest top-k mean cosine similarity. A confidence threshold (τ) drops low-confidence queries to a fallback. The [routing demo notebook](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/routing_demo.ipynb) ships a ready-to-use `Router` runtime.

## Installation

```bash
pip install training-hub[embedding]
```

The `[embedding]` extra includes `sentence-transformers>=5.0`.

## See Also

- [Embedding SFT Algorithm Overview](/algorithms/embedding_sft)
- [`embedding_sft()` Function Reference](/api/functions/embedding_sft)
- [`EmbeddingSFTAlgorithm` Class Reference](/api/classes/EmbeddingSFTAlgorithm)
- [Backends Overview](/api/backends/)
