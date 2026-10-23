"""classifier_sft — supervised fine-tuning of a sequence-classification head.

Trains an N-way text classifier (``AutoModelForSequenceClassification`` + cross-entropy)
for classification / "decision" tasks — intent routing, triage, topic/sentiment, and
multi-choice or yes/no judgments over a *fixed* label set. This is distinct from
``embedding_sft``: that learns a contrastive *embedding space* classified by
nearest-centroid, whereas this learns a classification *head* directly and produces
per-class logits (temperature-scaling / ECE calibration is applied at evaluation time
on a held-out set, not here).

One algorithm, two backbone families:
  * Encoder models (ModernBERT, DeBERTa, BERT) — full fine-tune.
  * Decoder models (e.g. Qwen) — parameter-efficient via LoRA (``use_peft=True``),
    which puts a classification head on the last token.

The saved checkpoint carries its ``id2label`` / ``label2id`` mapping (on the model
config and in ``label_mapping.json``), so it is inference-ready.
"""

import json
import logging
import os
from typing import Any, Optional

from . import Algorithm, Backend, AlgorithmRegistry

logger = logging.getLogger(__name__)

_KNOWN_PARAMS: set[str] = {
    "model_path",
    "data_path",
    "ckpt_output_dir",
    "num_epochs",
    "batch_size",
    "learning_rate",
    "max_seq_len",
    "text_column",
    "label_column",
    "use_peft",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "trust_remote_code",
    "warmup_ratio",
    "weight_decay",
    "gradient_accumulation_steps",
    "seed",
}


def _load_classification_dataset(
    data_path: str, text_column: str, label_column: str
) -> tuple[Any, dict[int, str]]:
    """Load a (text, label) dataset and encode labels to contiguous ints.

    Reuses the shared ``load_training_dataset`` loader (so .jsonl/.json/.parquet/.csv
    and HuggingFace dataset IDs all work), renames the requested columns to the
    canonical ``text``/``label``, and maps string/categorical labels (e.g. "billing",
    "technical") to contiguous integer codes.

    Returns:
        (dataset, id2label) where ``dataset`` has int64 ``label``s and ``id2label``
        maps each code back to its original string (for an inference-ready config).
    """
    from datasets import Value

    from training_hub.utils import load_training_dataset

    dataset = load_training_dataset(data_path)

    if text_column != "text" and text_column in dataset.column_names:
        dataset = dataset.rename_column(text_column, "text")
    if label_column != "label" and label_column in dataset.column_names:
        dataset = dataset.rename_column(label_column, "label")
    for col in ("text", "label"):
        if col not in dataset.column_names:
            raise ValueError(
                f"Dataset must have a '{col}' column (set text_column/label_column). "
                f"Found: {dataset.column_names}"
            )

    labels = [v for v in dataset["label"] if v is not None]
    if not labels:
        raise ValueError("Dataset has no non-null labels to train on.")

    sample = labels[0]
    already_int = isinstance(sample, int) and not isinstance(sample, bool)
    if already_int:
        # Plain integer labels: id2label spans the observed range so the head width
        # matches the label space even if some classes are absent from this split.
        max_id = max(int(v) for v in labels)
        id2label = {i: str(i) for i in range(max_id + 1)}
    else:
        # Strings / bools / floats -> contiguous int codes, mapping kept for id2label.
        uniques = sorted({str(v) for v in labels})
        label2id = {name: idx for idx, name in enumerate(uniques)}
        id2label = {idx: name for name, idx in label2id.items()}
        dataset = dataset.map(
            lambda b: {"label": [label2id[str(v)] for v in b["label"]]}, batched=True
        )
        logger.info(
            "Encoded %d categorical label(s) to integers%s",
            len(uniques),
            f": {label2id}" if len(label2id) <= 20 else " (>20, omitted)",
        )

    dataset = dataset.cast_column("label", Value("int64"))
    return dataset, id2label


class SequenceClassificationBackend(Backend):
    """``AutoModelForSequenceClassification`` + HuggingFace ``Trainer`` (cross-entropy)."""

    def execute_training(self, algorithm_params: dict[str, Any]) -> Any:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        unknown = [k for k in algorithm_params if k not in _KNOWN_PARAMS]
        if unknown:
            logger.warning(
                "Ignoring unrecognized classifier_sft parameter(s): %s. "
                "Check for typos against the documented parameters.",
                ", ".join(sorted(unknown)),
            )

        model_path = algorithm_params["model_path"]
        data_path = algorithm_params["data_path"]
        ckpt_output_dir = algorithm_params["ckpt_output_dir"]
        num_epochs = algorithm_params.get("num_epochs", 3)
        batch_size = algorithm_params.get("batch_size", 16)
        learning_rate = algorithm_params.get("learning_rate", 2e-5)
        max_seq_len = algorithm_params.get("max_seq_len", 512)
        text_column = algorithm_params.get("text_column", "text")
        label_column = algorithm_params.get("label_column", "label")
        use_peft = algorithm_params.get("use_peft", False)
        trust_remote_code = algorithm_params.get("trust_remote_code", False)
        warmup_ratio = algorithm_params.get("warmup_ratio", 0.0)
        weight_decay = algorithm_params.get("weight_decay", 0.01)
        grad_accum = algorithm_params.get("gradient_accumulation_steps", 1)
        seed = algorithm_params.get("seed", 42)

        logger.info("Loading classification data: %s", data_path)
        dataset, id2label = _load_classification_dataset(data_path, text_column, label_column)
        label2id = {name: idx for idx, name in id2label.items()}
        num_labels = len(id2label)
        logger.info("Training a %d-class classifier on %d examples", num_labels, len(dataset))

        logger.info("Loading model: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=trust_remote_code,
        )

        # Decoder backbones typically have no pad token; sequence classification (which
        # pads to the batch max and reads the last non-pad token) needs one.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if use_peft:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=algorithm_params.get("lora_r", 16),
                lora_alpha=algorithm_params.get("lora_alpha", 32),
                lora_dropout=algorithm_params.get("lora_dropout", 0.05),
                target_modules=algorithm_params.get("lora_target_modules"),
            )
            model = get_peft_model(model, lora_config)
            logger.info("Applied LoRA (SEQ_CLS) — parameter-efficient fine-tuning")

        def _tokenize(batch: dict[str, Any]) -> Any:
            return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

        drop = [c for c in dataset.column_names if c != "label"]
        tokenized = dataset.map(_tokenize, batched=True, remove_columns=drop)

        args = TrainingArguments(
            output_dir=ckpt_output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            # transformers v5 removed `warmup_ratio`; `warmup_steps` as a float is the ratio.
            warmup_steps=warmup_ratio,
            weight_decay=weight_decay,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            seed=seed,
            bf16=torch.cuda.is_available(),
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        logger.info("Starting classifier training (%d epochs, lr=%s)", num_epochs, learning_rate)
        trainer.train()

        os.makedirs(ckpt_output_dir, exist_ok=True)
        # Merge LoRA adapters so the saved artifact is a standalone classifier.
        to_save = model.merge_and_unload() if use_peft else model
        to_save.save_pretrained(ckpt_output_dir)
        tokenizer.save_pretrained(ckpt_output_dir)
        with open(os.path.join(ckpt_output_dir, "label_mapping.json"), "w") as f:
            json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)
        logger.info("Saved %d-class classifier to %s", num_labels, ckpt_output_dir)

        return {
            "ckpt_output_dir": ckpt_output_dir,
            "num_labels": num_labels,
            "id2label": id2label,
        }


class ClassifierSFTAlgorithm(Algorithm):
    """Algorithm for supervised classification-head fine-tuning."""

    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend

    def train(self, **kwargs) -> Any:
        return self.backend.execute_training(kwargs)

    def get_required_params(self) -> dict[str, type]:
        return {
            "model_path": str,
            "data_path": str,
            "ckpt_output_dir": str,
        }

    def get_optional_params(self) -> dict[str, type]:
        return {
            "num_epochs": int,
            "batch_size": int,
            "learning_rate": float,
            "max_seq_len": int,
            "text_column": str,
            "label_column": str,
            "use_peft": bool,
            "lora_r": int,
            "lora_alpha": int,
            "lora_dropout": float,
            "trust_remote_code": bool,
            "warmup_ratio": float,
            "weight_decay": float,
            "gradient_accumulation_steps": int,
            "seed": int,
        }


def classifier_sft(
    model_path: str,
    data_path: str,
    ckpt_output_dir: str,
    *,
    backend: str = "sequence-classification",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_len: int = 512,
    text_column: str = "text",
    label_column: str = "label",
    use_peft: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
    trust_remote_code: bool = False,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """Fine-tune a sequence-classification head on (text, label) data.

    Args:
        model_path: Encoder (ModernBERT/DeBERTa/BERT) or decoder (Qwen, with
            ``use_peft=True``) backbone — a local path or HuggingFace ID.
        data_path: .jsonl/.json/.parquet/.csv or a HF dataset ID with text + label
            columns. String/categorical labels are encoded to ints automatically.
        ckpt_output_dir: Where to save the trained classifier (+ label mapping).
        use_peft: LoRA fine-tuning (recommended for decoder backbones).
        trust_remote_code: Allow custom architectures from the model repo (opt-in).

    Returns:
        dict with ``ckpt_output_dir``, ``num_labels``, and ``id2label``.

    Example:
        result = classifier_sft(
            model_path="answerdotai/ModernBERT-base",
            data_path="intents_train.parquet",
            ckpt_output_dir="./intent_classifier",
            num_epochs=3,
            text_column="text",
            label_column="category",
        )
    """
    from . import create_algorithm

    algorithm = create_algorithm("classifier_sft", backend)
    return algorithm.train(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        text_column=text_column,
        label_column=label_column,
        use_peft=use_peft,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        trust_remote_code=trust_remote_code,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        **kwargs,
    )


AlgorithmRegistry.register_algorithm("classifier_sft", ClassifierSFTAlgorithm)
AlgorithmRegistry.register_backend(
    "classifier_sft", "sequence-classification", SequenceClassificationBackend
)
