# `ClassifierSFTAlgorithm` - Decision / Classification-Head Fine-Tuning Algorithm Class

> Concrete implementation of the Algorithm interface for fine-tuning a classification/decision head on any recognized backbone, using the HuggingFace `Trainer` with cross-entropy.

## Class Signature

```python
from training_hub import ClassifierSFTAlgorithm, Backend

class ClassifierSFTAlgorithm(Algorithm):
    """
    Fine-tunes a task head (sequence classification, multiple choice, or token
    classification) on any recognized HF backbone. Encoders are full fine-tuned;
    decoder backbones train via LoRA. Produces an inference-ready checkpoint with
    per-class logits and a saved label mapping.
    """

    def __init__(self, backend: Backend, **kwargs) -> None:
        """Initialize the classifier SFT algorithm with a backend."""

    def train(
        self,
        model_path: str,
        data_path: str,
        ckpt_output_dir: str,
        **kwargs,
    ) -> Any:
        """Execute classification-head fine-tuning."""

    def get_required_params(self) -> dict[str, type]:
        """Get required parameters."""

    def get_optional_params(self) -> dict[str, type]:
        """Get optional parameters."""
```

## Overview

`ClassifierSFTAlgorithm` is the class-based implementation of classification-head fine-tuning in Training Hub. It inherits from the [`Algorithm`](Algorithm.md) abstract base class.

Unlike [`EmbeddingSFTAlgorithm`](EmbeddingSFTAlgorithm.md) — which reshapes an embedding space and classifies by nearest-anchor similarity — it trains a dedicated **task head** and produces **per-class logits**, the best choice for a fixed, known label set. It is a convention-based harness: the `task` selector maps to a native HF Auto class, and the `trust_remote_code`/`revision`, `model_class`, and `loss_fn` hooks extend coverage with no vendored packages.

This class is useful when you need:
- To reuse an algorithm instance across multiple fine-tuning runs
- Direct access to the algorithm interface
- Custom classification-training pipelines

For most use cases, the convenience function [`classifier_sft()`](../functions/classifier_sft.md) is simpler.

## Constructor

### `__init__(backend: Backend, **kwargs) -> None`

Creates a new `ClassifierSFTAlgorithm` instance.

**Parameters:**
- `backend` (`Backend`): The backend implementation to use (`DecisionHeadBackend`)
- `**kwargs`: Accepted for API symmetry but not stored — all training configuration
  is passed to `train()`. (Use the [`classifier_sft()`](../functions/classifier_sft.md)
  convenience function to select the backend by name via the `backend` argument.)

**Example:**
```python
from training_hub import ClassifierSFTAlgorithm, DecisionHeadBackend

backend = DecisionHeadBackend()
algorithm = ClassifierSFTAlgorithm(backend=backend)
```

## Methods

### `train(**kwargs) -> Any`

Executes classification-head fine-tuning.

#### Parameters

##### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | HuggingFace model ID or local path (encoder, or decoder with `use_peft=True`). |
| `data_path` | `str` | Path to a `.jsonl/.json/.parquet/.csv` file or HuggingFace dataset ID. Columns depend on `task`. |
| `ckpt_output_dir` | `str` | Directory to save the model, tokenizer, and `label_mapping.json`. |

##### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `"sequence_classification"` | `"sequence_classification"`, `"multiple_choice"`, or `"token_classification"`. |
| `num_epochs` | `int` | `3` | Number of training epochs. |
| `batch_size` | `int` | `16` | Per-device train batch size. |
| `learning_rate` | `float` | `2e-5` | Learning rate (decoders need `~1e-4..1e-3`). |
| `max_seq_len` | `int` | `512` | Max token length. |
| `warmup_ratio` | `float` | `0.0` | Warmup fraction of total steps. |
| `weight_decay` | `float` | `0.01` | Weight decay. |
| `gradient_accumulation_steps` | `int` | `1` | Gradient accumulation steps. |
| `use_peft` | `bool` | `False` | LoRA fine-tuning (recommended for decoders). |
| `lora_r` / `lora_alpha` / `lora_dropout` | `int`/`int`/`float` | `16`/`32`/`0.05` | LoRA hyperparameters. |
| `lora_target_modules` | `list` | `None` | Explicit LoRA target modules (`None` = PEFT default). |
| `text_column` | `str` | `"text"` | Text/context column. |
| `label_column` | `str` | `None` | Label column (task default: `label` seq/MC, `ner_tags` token). |
| `options_column` | `str` | `"options"` | Options column (multiple choice). |
| `tokens_column` | `str` | `"tokens"` | Tokens column (token classification). |
| `trust_remote_code` | `bool` | `False` | Allow a custom `auto_map` architecture (pin `revision`). |
| `revision` | `str` | `None` | Model repo commit/branch/tag to pin. |
| `model_class` | `str` | `None` | Dotted path to a custom `*.from_pretrained` class. |
| `loss_fn` | `str` | `None` | Dotted path to a custom loss callable or `Trainer` subclass. |
| `seed` | `int` | `42` | Random seed. |

> The backend is selected in the constructor (`__init__`), not via `train()`.
> Additional parameters are also accepted via `train(**kwargs)` (unknown ones
> are logged as a warning to catch typos).

#### Returns

**Type:** `dict` — `{"ckpt_output_dir", "task", "id2label"}`. The inference-ready checkpoint (model + tokenizer + `label_mapping.json`) is written to `ckpt_output_dir`.

#### Raises

- **`ValueError`**: When `task` is unknown, the dataset lacks the required columns for the task, or labels are null/NaN.
- **`ImportError`**: When a dotted `model_class` / `loss_fn` path can't be imported (the user must install the package providing it).

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

Returns the optional parameters. See the [`classifier_sft()` function reference](../functions/classifier_sft.md) for the full list.

## Examples

### Basic Usage

```python
from training_hub import ClassifierSFTAlgorithm, DecisionHeadBackend

backend = DecisionHeadBackend()
algorithm = ClassifierSFTAlgorithm(backend=backend)

result = algorithm.train(
    model_path="answerdotai/ModernBERT-base",
    data_path="./intents.parquet",
    ckpt_output_dir="./intent_clf",
    label_column="category",
    num_epochs=10,
)
```

## Relationship to classifier_sft() Function

The [`classifier_sft()`](../functions/classifier_sft.md) function is a convenience wrapper around `ClassifierSFTAlgorithm`. Both use identical parameters. Prefer the function for most use cases; use the class directly when reusing an instance across multiple runs.

## See Also

- [**classifier_sft() Function**](/api/functions/classifier_sft) - Convenience wrapper function
- [**Algorithm Class**](/api/classes/Algorithm) - Base class interface
- [**Classifier SFT Backend**](/api/backends/classifier_sft) - `DecisionHeadBackend`
- [**create_algorithm() Function**](/api/functions/create-algorithm) - Factory function
- [**Classifier SFT Algorithm Overview**](/algorithms/classifier_sft) - Conceptual overview and tips

## Source

[View source on GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/classifier_sft.py)
