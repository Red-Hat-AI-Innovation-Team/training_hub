# Classifier SFT — Decision / Classification-Head Fine-Tuning

Classifier SFT fine-tunes a **task head** on top of any recognized backbone using the HuggingFace `Trainer` with cross-entropy. Unlike [Embedding SFT](/algorithms/embedding_sft) — which reshapes an embedding space and classifies by nearest-anchor similarity — this trains a dedicated head and produces **per-class logits**, the best choice for a *fixed, known* label set (calibrated confidence, highest accuracy).

> It is a thin, **convention-based** harness over the HF Auto-class family: it covers the common "decision model" primitives through native HF classes with **no per-model code and no vendored packages**, and its escape hatches (`trust_remote_code` + `revision`, `model_class`, `loss_fn`) let coverage grow with the ecosystem at flat maintenance cost.

## When to Use

- You want to **classify text into a fixed set of N categories** (intent, triage, topic, sentiment, yes/no judgments) with the best accuracy and calibrated probabilities.
- You need **span / entity extraction** (token classification) or **pick-one-of-K** (multiple choice).
- You want to fine-tune a modern encoder (ModernBERT/DeBERTa/BERT) or a decoder backbone (via LoRA) as a classifier without writing model-specific code.

If your label set will **grow over time**, or you want reusable embeddings / nearest-match routing, use [Embedding SFT](/algorithms/embedding_sft) instead. For generative weight-training, see [SFT](/algorithms/sft), [OSFT](/algorithms/osft), [LoRA + SFT](/algorithms/lora), or [GRPO](/algorithms/grpo).

## Quick Start

```python
from training_hub import classifier_sft

result = classifier_sft(
    model_path="answerdotai/ModernBERT-base",
    data_path="intents.parquet",         # columns: text, label (strings are auto-encoded)
    ckpt_output_dir="./intent_clf",
    text_column="text",
    label_column="category",
    num_epochs=10,
    batch_size=32,
    learning_rate=2e-5,
)
# ./intent_clf is a standard HF sequence-classification checkpoint + label_mapping.json:
#   AutoModelForSequenceClassification.from_pretrained("./intent_clf")
```

## Tasks (the decision primitives)

The `task` parameter selects a native HF Auto class — each with its own expected columns:

| `task` | Auto class | Use | Data columns |
|--------|-----------|-----|--------------|
| `sequence_classification` (default) | `AutoModelForSequenceClassification` | fixed-label classification, yes/no | `text`, `label` |
| `multiple_choice` | `AutoModelForMultipleChoice` | pick one of K candidate options per example | `text`, `options` (list), `label` (index) |
| `token_classification` | `AutoModelForTokenClassification` | span / entity extraction (NER-style) | `tokens` (list), `ner_tags` (list) |

## How It Works

1. **Load** the tokenizer and the task's Auto class from `model_path` (a HF id or local path).
2. **Prepare** the dataset for the task (tokenize, encode labels, align token tags, or collate choices).
3. **Encode labels**: string/bool/float labels are mapped to contiguous integer codes; integer labels keep their value (the head spans `0..max`). Null/NaN labels are rejected with a clear error — a classification head has no null consumer.
4. **Train** with the HF `Trainer` and cross-entropy (optionally LoRA, a custom loss, or a custom model class — see below).
5. **Save** the model, tokenizer, and `label_mapping.json` (`task` + `id2label`/`label2id`) to `ckpt_output_dir` — an inference-ready checkpoint.

The call returns `{"ckpt_output_dir", "task", "id2label"}`.

## Backbones: encoders vs. decoders

- **Encoders** (ModernBERT, DeBERTa, BERT) are **full fine-tuned** and work well at the `2e-5` default on CPU or a single GPU.
- **Decoders** (Qwen, Llama, …) should use **LoRA** (`use_peft=True`). A freshly initialized head on a decoder trains far slower: the `2e-5` default typically **collapses to a single class** — use `~1e-4..1e-3` and more epochs. (Verified empirically: Qwen2.5-1.5B + LoRA collapsed at `2e-5`, fit its training set at `1e-3`.) LoRA adapters are merged into the base weights before saving, so the checkpoint loads like any standard model.

```python
result = classifier_sft(
    model_path="Qwen/Qwen2.5-1.5B",
    data_path="intents.parquet",
    ckpt_output_dir="./intent_clf_qwen",
    use_peft=True,
    learning_rate=1e-3,     # decoders need a higher LR than the 2e-5 encoder default
    num_epochs=30,
    label_column="category",
)
```

## Custom Architectures & Objectives (extension hooks)

Coverage of the OSS "decision model" space grows by **convention and configuration**, not by vendoring packages or adding per-model branches:

- **`trust_remote_code=True` + `revision`** — load any architecture that ships an HF `auto_map` (GLiClass, Jina, Nomic, GTE, …). The model author's code runs, so **pin `revision`** to a commit/branch/tag for safety.
- **`model_class`** (dotted path) — point at an importable `*.from_pretrained` class for a backbone distributed as a pip package rather than via `auto_map`. The user installs that package; this library takes on no dependency.
- **`loss_fn`** (dotted path) — a custom `loss_fn(logits, labels)` callable or a `Trainer` subclass, for a bespoke objective. (For `multiple_choice`, the default path masks padded choices out of the softmax; a custom 2-arg `loss_fn` owns any masking it needs.)

```python
result = classifier_sft(
    model_path="knowledgator/gliclass-base-v1.0",
    data_path="intents.parquet",
    ckpt_output_dir="./gliclass_clf",
    trust_remote_code=True,
    revision="<commit-sha>",            # pin custom code to a known commit
    label_column="category",
)
```

## Data Format

Accepts JSONL/JSON, Parquet, CSV, or a HuggingFace dataset id. Columns default per task (above) and are overridable with `text_column`, `label_column`, `options_column`, `tokens_column`. String labels are encoded to contiguous ints; the mapping is recorded in `label_mapping.json`.

```json
// sequence_classification
{"text": "I was double charged this month", "label": "billing"}
// multiple_choice
{"text": "The capital of France is", "options": ["Berlin", "Paris", "Rome"], "label": 1}
// token_classification
{"tokens": ["Barack", "Obama", "visited", "Paris"], "ner_tags": ["B-PER", "I-PER", "O", "B-LOC"]}
```

## Output

- An inference-ready checkpoint in `ckpt_output_dir`: the model, tokenizer, and `label_mapping.json` (`{"task", "id2label", "label2id"}`).
- Load with the task's Auto class, e.g. `AutoModelForSequenceClassification.from_pretrained(ckpt_output_dir)`, then map argmax ids back through `id2label`.

> **Calibration** (temperature scaling / ECE) is an **evaluation-time** step, not part of training — the head emits raw logits.

## Tips

- **Encoders are the default.** ModernBERT-base is fast, strong, and CPU-trainable; reach for a decoder only when you specifically need one (then use LoRA + a higher LR).
- **Fixed labels → this; growing labels → embeddings.** A head is tied to the labels it was trained on. If categories will be added later, use [Embedding SFT](/algorithms/embedding_sft).
- **Clean your labels.** Null/NaN labels are rejected up front — fill or drop them before training.
- **Pin `revision` whenever `trust_remote_code=True`.** Remote model code can change under you; a pinned commit keeps runs reproducible and safe.
- **Multiple choice with variable K is fine.** Padded choices are masked out of the loss, so mixed option counts in a batch don't bias training.

## API Reference

See [`classifier_sft()`](/api/functions/classifier_sft) for the full parameter reference and [`ClassifierSFTAlgorithm`](/api/classes/ClassifierSFTAlgorithm) for the class-based API.
