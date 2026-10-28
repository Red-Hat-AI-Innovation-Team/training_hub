# `classifier_sft()`

Fine-tunes a classification/decision **head** on any recognized backbone (HuggingFace `Trainer` + cross-entropy). A convention-based harness over the HF Auto-class family covering three decision primitives — sequence classification, multiple choice, and token classification — with no per-model code and no vendored packages.

## Signature

```python
from training_hub import classifier_sft

result = classifier_sft(
    model_path: str,
    data_path: str,
    ckpt_output_dir: str,
    *,
    backend: str = "decision-head",
    # Task / decision primitive
    task: str = "sequence_classification",
    # Training parameters
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_len: int = 512,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    # PEFT / LoRA (recommended for decoder backbones)
    use_peft: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
    # Data format (columns vary by task)
    text_column: str = "text",
    label_column: Optional[str] = None,   # task default: label (seq/MC), ner_tags (token)
    options_column: str = "options",
    tokens_column: str = "tokens",
    # Custom architectures / objectives
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    model_class: Optional[str] = None,
    loss_fn: Optional[str] = None,
    # Standard
    seed: int = 42,
    **kwargs,
)
```

## Quick Example

```python
from training_hub import classifier_sft

result = classifier_sft(
    model_path="answerdotai/ModernBERT-base",
    data_path="intents.parquet",
    ckpt_output_dir="./intent_clf",
    text_column="text",
    label_column="category",
    num_epochs=10,
)
print(result["ckpt_output_dir"], result["task"])
```

## Parameters

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | HuggingFace model ID or local path. An encoder (ModernBERT/DeBERTa/BERT, full fine-tune) or a decoder (Qwen/Llama, use `use_peft=True`). |
| `data_path` | `str` | Path to a `.jsonl/.json/.parquet/.csv` file or a HuggingFace dataset ID. Expected columns depend on `task`. |
| `ckpt_output_dir` | `str` | Directory to save the model, tokenizer, and `label_mapping.json`. |

### Backend / Task

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `"decision-head"` | Backend implementation. Currently the only option is `"decision-head"`. |
| `task` | `"sequence_classification"` | One of `"sequence_classification"`, `"multiple_choice"`, `"token_classification"`. |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | `3` | Number of training epochs. Decoder backbones usually need more. |
| `batch_size` | `16` | Per-device train batch size. |
| `learning_rate` | `2e-5` | Learning rate (tuned for encoders). Decoders with a fresh head typically need `~1e-4..1e-3`. |
| `max_seq_len` | `512` | Max token length. |
| `warmup_ratio` | `0.0` | Fraction of total steps used for LR warmup. |
| `weight_decay` | `0.01` | Weight decay. |
| `gradient_accumulation_steps` | `1` | Steps to accumulate before an optimizer update. |

### PEFT / LoRA

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_peft` | `False` | LoRA fine-tuning instead of a full fine-tune (recommended for decoder backbones). Adapters are merged before saving. |
| `lora_r` | `16` | LoRA rank. |
| `lora_alpha` | `32` | LoRA alpha. |
| `lora_dropout` | `0.05` | LoRA dropout. |
| `lora_target_modules` | `None` | Explicit list of module names to adapt; `None` lets PEFT choose sensible targets. |

### Data Format

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text_column` | `"text"` | Text/context column (`sequence_classification` & `multiple_choice`). |
| `label_column` | `None` | Label column. Defaults per task: `label` (seq/MC), `ner_tags` (token). String labels are encoded to ints. |
| `options_column` | `"options"` | Options-list column (`multiple_choice`). |
| `tokens_column` | `"tokens"` | Pre-split tokens column (`token_classification`). |

### Custom Architectures / Objectives

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trust_remote_code` | `False` | Allow a custom architecture that ships an HF `auto_map` (opt-in; the model author's code runs). Pin `revision` with it. |
| `revision` | `None` | Model repo commit/branch/tag to pin — strongly recommended with `trust_remote_code`. |
| `model_class` | `None` | Dotted path to an importable `*.from_pretrained` class, for a pip-packaged backbone. |
| `loss_fn` | `None` | Dotted path to a custom `loss_fn(logits, labels)` callable or a `Trainer` subclass. |

### Standard

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `42` | Random seed for reproducibility. |

## Returns

**Type:** `dict`

```python
{
    "ckpt_output_dir": "<ckpt_output_dir>",
    "task": "<task>",
    "id2label": {0: "...", 1: "...", ...},
}
```

The checkpoint is inference-ready: the model, tokenizer, and `label_mapping.json` (`task` + `id2label`/`label2id`) are written to `ckpt_output_dir`. Reload with the task's Auto class, e.g. `AutoModelForSequenceClassification.from_pretrained("<ckpt_output_dir>")`.

## Decoder Backbone Example (LoRA)

```python
from training_hub import classifier_sft

result = classifier_sft(
    model_path="Qwen/Qwen2.5-1.5B",
    data_path="intents.parquet",
    ckpt_output_dir="./intent_clf_qwen",
    use_peft=True,
    learning_rate=1e-3,     # decoders need more than the 2e-5 encoder default
    num_epochs=30,
    label_column="category",
)
```

## Data Format

Expected columns depend on `task`:

```json
// sequence_classification: text, label
{"text": "I was double charged this month", "label": "billing"}
// multiple_choice: text, options (list), label (index)
{"text": "The capital of France is", "options": ["Berlin", "Paris", "Rome"], "label": 1}
// token_classification: tokens (list), ner_tags (list)
{"tokens": ["Barack", "Obama", "visited", "Paris"], "ner_tags": ["B-PER", "I-PER", "O", "B-LOC"]}
```

String labels are encoded to contiguous ints (recorded in `label_mapping.json`). Null/NaN labels are rejected with a clear error.

## Related

- [Classifier SFT Algorithm Guide](/algorithms/classifier_sft) — conceptual overview, tasks, extension hooks, and tips
- [`ClassifierSFTAlgorithm`](/api/classes/ClassifierSFTAlgorithm) — class-based API
- [Classifier SFT Backend](/api/backends/classifier_sft) — `DecisionHeadBackend` details
