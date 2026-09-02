# `EmbeddingSFTAlgorithm` - Contrastive Embedding Fine-Tuning Algorithm Class

> Concrete implementation of the Algorithm interface for contrastive fine-tuning of sentence embedding models. Reshapes the embedding space so same-label inputs cluster together — the basis for semantic routing / classification.

## Class Signature

```python
from training_hub import EmbeddingSFTAlgorithm, Backend

class EmbeddingSFTAlgorithm(Algorithm):
    """
    Fine-tunes sentence embedding models using contrastive losses so that
    inputs with the same label cluster together in embedding space. Designed
    for semantic routing classifiers but applicable to any embedding
    classification task.
    """

    def __init__(self, backend: Backend, **kwargs) -> None:
        """Initialize the embedding SFT algorithm with a backend."""

    def train(
        self,
        model_path: str,
        data_path: str,
        ckpt_output_dir: str,
        **kwargs,
    ) -> Any:
        """Execute contrastive embedding fine-tuning."""

    def get_required_params(self) -> dict[str, type]:
        """Get required parameters."""

    def get_optional_params(self) -> dict[str, type]:
        """Get optional parameters."""
```

## Overview

`EmbeddingSFTAlgorithm` is the class-based implementation of contrastive embedding fine-tuning in Training Hub. It inherits from the [`Algorithm`](Algorithm.md) abstract base class.

Unlike weight-training algorithms (SFT, OSFT, LoRA, GRPO), it does not train a generative model — it reshapes the embedding space of a sentence-transformers model so that same-label inputs cluster together. The resulting model is used for **semantic routing / classification** via nearest-anchor cosine similarity (the runtime is supplied by the user; see the [routing demo notebook](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/routing_demo.ipynb)).

This class is useful when you need:
- To reuse an algorithm instance across multiple fine-tuning runs
- Direct access to the algorithm interface
- Custom embedding-training pipelines

For most use cases, the convenience function [`embedding_sft()`](../functions/embedding_sft.md) is simpler.

## Constructor

### `__init__(backend: Backend, **kwargs) -> None`

Creates a new `EmbeddingSFTAlgorithm` instance.

**Parameters:**
- `backend` (`Backend`): The backend implementation to use (`SentenceTransformersBackend`)
- `**kwargs`: Additional configuration passed to the algorithm

**Example:**
```python
from training_hub import EmbeddingSFTAlgorithm, SentenceTransformersBackend

backend = SentenceTransformersBackend()
algorithm = EmbeddingSFTAlgorithm(backend=backend)
```

## Methods

### `train(**kwargs) -> Any`

Executes contrastive embedding fine-tuning.

#### Parameters

##### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | HuggingFace model ID or local path to a sentence-transformers model. |
| `data_path` | `str` | Path to a JSONL/CSV file or HuggingFace dataset ID with text and integer label columns. |
| `ckpt_output_dir` | `str` | Directory to save the fine-tuned model. |

##### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"sentence-transformers"` | Training backend. |
| `loss_type` | `str` | `"batch_all_triplet"` | One of `"batch_all_triplet"`, `"batch_hard_triplet"`, `"mnrl"`. Ignored if `loss_fn` is set. |
| `loss_fn` | `Callable` | `None` | Custom loss function; overrides `loss_type`. |
| `num_epochs` | `int` | `20` | Number of training epochs. |
| `batch_size` | `int` | `32` | Per-device train batch size. |
| `learning_rate` | `float` | `2e-5` | Learning rate. |
| `warmup_ratio` | `float` | `0.1` | Warmup fraction of total steps. |
| `batch_sampler` | `str` | `None` | `"group_by_label"`, `"no_duplicates"`, `"default"`, or `None` (auto-select). |
| `eval_data_path` | `str` | `None` | Optional evaluation data path. |
| `text_column` | `str` | `"text"` | Name of the text column. |
| `label_column` | `str` | `"label"` | Name of the integer label column. |
| `seed` | `int` | `42` | Random seed. |

> Additional parameters are also accepted via `**kwargs`.

#### Returns

**Type:** `dict` — `{"status", "model_path", "num_samples", "num_epochs", "loss_type"}`. The model is saved to `ckpt_output_dir` in sentence-transformers format.

#### Raises

- **`ValueError`**: When `loss_type` is unknown and no `loss_fn` is provided, when `batch_sampler` is unknown, or when the dataset lacks the required text/label columns.
- **`ImportError`**: When `sentence-transformers` is not installed (install with `pip install training-hub[embedding]`).

### `get_required_params() -> Dict[str, Type]`

Returns the required parameters.

**Returns:**
```python
{
    "model_path": str,
    "data_path": str,
    "ckpt_output_dir": str,
}
```

### `get_optional_params() -> Dict[str, Type]`

Returns the optional parameters. See the [`embedding_sft()` function reference](../functions/embedding_sft.md) for the full list.

## Examples

### Basic Usage

```python
from training_hub import EmbeddingSFTAlgorithm, SentenceTransformersBackend

backend = SentenceTransformersBackend()
algorithm = EmbeddingSFTAlgorithm(backend=backend)

result = algorithm.train(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    data_path="./routing_train.jsonl",
    ckpt_output_dir="./routing_model",
    loss_type="batch_all_triplet",
    num_epochs=20,
)
```

## Relationship to embedding_sft() Function

The [`embedding_sft()`](../functions/embedding_sft.md) function is a convenience wrapper around `EmbeddingSFTAlgorithm`. Both use identical parameters. Prefer the function for most use cases; use the class directly when reusing an instance across multiple runs.

## See Also

- [**embedding_sft() Function**](/api/functions/embedding_sft) - Convenience wrapper function
- [**Algorithm Class**](/api/classes/Algorithm) - Base class interface
- [**Embedding SFT Backend**](/api/backends/embedding_sft) - `SentenceTransformersBackend`
- [**create_algorithm() Function**](/api/functions/create-algorithm) - Factory function
- [**Embedding SFT Algorithm Overview**](/algorithms/embedding_sft) - Conceptual overview and tips

## Source

[View source on GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/embedding_sft.py)
