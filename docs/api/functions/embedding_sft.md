# `embedding_sft()`

Contrastive fine-tuning of sentence embedding models so that inputs with the same label cluster together in embedding space. Designed for semantic routing / classification but applicable to any embedding task.

## Signature

```python
from training_hub import embedding_sft

result = embedding_sft(
    model_path: str,
    data_path: str,
    ckpt_output_dir: str,
    *,
    backend: str = "sentence-transformers",
    # Loss configuration
    loss_type: str = "batch_all_triplet",
    loss_fn: Optional[Callable] = None,
    # Training parameters
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    # Batch sampling
    batch_sampler: Optional[str] = None,
    # Evaluation
    eval_data_path: Optional[str] = None,
    # Data format
    text_column: str = "text",
    label_column: str = "label",
    # Standard
    seed: int = 42,
    **kwargs,
)
```

## Quick Example

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
print(result["status"], result["model_path"])
```

## Parameters

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | HuggingFace model ID or local path to a sentence-transformers model (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`). |
| `data_path` | `str` | Path to a JSONL/CSV file or a HuggingFace dataset ID. Each sample needs a text field and an integer label field. |
| `ckpt_output_dir` | `str` | Directory to save the fine-tuned model (standard sentence-transformers format). |

### Backend

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `"sentence-transformers"` | Training backend. Currently the only option is `"sentence-transformers"`. |

### Loss Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_type` | `"batch_all_triplet"` | One of `"batch_all_triplet"`, `"batch_hard_triplet"`, `"mnrl"`. Ignored if `loss_fn` is set. |
| `loss_fn` | `None` | Custom loss function. Overrides `loss_type` if provided. Pass any sentence-transformers-compatible loss. |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | `20` | Number of training epochs. |
| `batch_size` | `32` | Per-device train batch size. |
| `learning_rate` | `2e-5` | Learning rate. |
| `warmup_ratio` | `0.1` | Warmup fraction of total steps. |

### Batch Sampling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_sampler` | `None` | `None` auto-selects: `"group_by_label"` for triplet losses, `"no_duplicates"` for MNRL. Can be set explicitly to `"group_by_label"`, `"no_duplicates"`, or `"default"`. |

### Evaluation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eval_data_path` | `None` | Optional path to an evaluation JSONL/CSV (same format as `data_path`). Used by the trainer's built-in evaluation. |

### Data Format

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text_column` | `"text"` | Name of the text column in the dataset. |
| `label_column` | `"label"` | Name of the integer label column in the dataset. |

### Standard

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `42` | Random seed for reproducibility. |

## Returns

**Type:** `dict`

```python
{
    "status": "success",
    "model_path": "<ckpt_output_dir>",
    "num_samples": <int>,
    "num_epochs": <int>,
    "loss_type": "<loss_type>",
}
```

The fine-tuned model is saved to `ckpt_output_dir` in standard sentence-transformers format and can be reloaded with `SentenceTransformer("<ckpt_output_dir>")`. A `training_metrics.jsonl` file is also written there.

## Custom Loss Example

```python
from sentence_transformers import losses
from training_hub import embedding_sft
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

result = embedding_sft(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    data_path="routing_train.jsonl",
    ckpt_output_dir="./routing_model",
    loss_fn=losses.ContrastiveLoss(model),  # overrides loss_type
)
```

## Data Format

JSONL (one object per line), with a text field and an integer label field:

```json
{"text": "What is the cabin pressure trend?", "label": 0}
{"text": "How much propellant remains?", "label": 1}
```

CSV files and HuggingFace dataset IDs are also accepted. Override column names with `text_column` / `label_column`.

## Related

- [Embedding SFT Algorithm Guide](/algorithms/embedding_sft) — conceptual overview, losses, data format, and tips
- [`EmbeddingSFTAlgorithm`](/api/classes/EmbeddingSFTAlgorithm) — class-based API
- [Embedding SFT Backend](/api/backends/embedding_sft) — `SentenceTransformersBackend` details
