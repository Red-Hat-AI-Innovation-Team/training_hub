# Classifier SFT Backend

> Classification-head fine-tuning. The `DecisionHeadBackend` powers the [`classifier_sft()`](/api/functions/classifier_sft) algorithm.

## Overview

**Class:** `DecisionHeadBackend`

**Algorithm Support:** Classifier SFT (sequence classification, multiple choice, token classification)

**Package:** `transformers>=4.57.6`, `peft>=0.15` (both base dependencies)

**Status:** Implemented

The backend wraps the HuggingFace `Trainer`. It loads the task's `Auto*` class, prepares the dataset for the task, fine-tunes with cross-entropy (optionally LoRA), and saves an inference-ready checkpoint with a label mapping. It is convention-based: a `task` selector maps to a native HF Auto class, and extension hooks (`trust_remote_code`/`revision`, `model_class`, `loss_fn`) add custom architectures and objectives without vendoring any package.

## Usage

```python
from training_hub import classifier_sft

result = classifier_sft(
    model_path="answerdotai/ModernBERT-base",
    data_path="intents.parquet",
    ckpt_output_dir="./intent_clf",
    label_column="category",
    num_epochs=10,
)
```

### Via Class-Based API

```python
from training_hub import ClassifierSFTAlgorithm, DecisionHeadBackend

backend = DecisionHeadBackend()
algorithm = ClassifierSFTAlgorithm(backend=backend)
result = algorithm.train(
    model_path="answerdotai/ModernBERT-base",
    data_path="intents.parquet",
    ckpt_output_dir="./intent_clf",
    label_column="category",
)
```

## Tasks

Select via `task`; each maps to a native HF Auto class with its own expected columns:

| `task` | Auto class | Data columns |
|--------|-----------|--------------|
| `sequence_classification` (default) | `AutoModelForSequenceClassification` | `text`, `label` |
| `multiple_choice` | `AutoModelForMultipleChoice` | `text`, `options` (list), `label` (index) |
| `token_classification` | `AutoModelForTokenClassification` | `tokens` (list), `ner_tags` (list) |

For `multiple_choice`, the collator pads variable option counts to the batch max with empty choices and emits a `choice_mask`; the default trainer masks padded slots out of the softmax so mixed-K batches are unbiased.

## Backbones & LoRA

- **Encoders** (ModernBERT/DeBERTa/BERT) are full fine-tuned — fine at the `2e-5` default on CPU or a single GPU.
- **Decoders** (Qwen/Llama) use LoRA (`use_peft=True`); a fresh decoder head needs `~1e-4..1e-3` and more epochs to avoid single-class collapse. LoRA adapters are merged into the base weights before saving.

## Label Encoding

String/bool/float labels are mapped to contiguous integer codes; integer labels keep their value (the head spans `0..max`). The mapping is written to `label_mapping.json`. **Null/NaN labels are rejected** with a clear error — a classification head has no null consumer.

## Extension Hooks

| Hook | Purpose |
|------|---------|
| `trust_remote_code` + `revision` | Load any architecture that ships an HF `auto_map` (pin `revision` to a commit — remote code runs). |
| `model_class` (dotted path) | Load a backbone distributed as a pip package rather than via `auto_map` (the user installs it). |
| `loss_fn` (dotted path) | A custom `loss_fn(logits, labels)` callable or a `Trainer` subclass. |

## Precision

The backend auto-enables `bf16` when a CUDA GPU is available, otherwise trains in fp32. Encoder classifiers also run on CPU.

## Output

- An inference-ready checkpoint in `ckpt_output_dir`: model, tokenizer, and `label_mapping.json` (`{"task", "id2label", "label2id"}`).
- Reload with the task's Auto class, e.g. `AutoModelForSequenceClassification.from_pretrained("<ckpt_output_dir>")`.

> Calibration (temperature scaling / ECE) is an evaluation-time step — the head emits raw logits.

## Installation

```bash
pip install training-hub
```

No extra is required — `transformers`, `peft`, `datasets`, and `accelerate` are base dependencies.

## See Also

- [Classifier SFT Algorithm Overview](/algorithms/classifier_sft)
- [`classifier_sft()` Function Reference](/api/functions/classifier_sft)
- [`ClassifierSFTAlgorithm` Class Reference](/api/classes/ClassifierSFTAlgorithm)
- [Backends Overview](/api/backends/)
