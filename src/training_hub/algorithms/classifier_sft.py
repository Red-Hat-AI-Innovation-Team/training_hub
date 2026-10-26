"""classifier_sft — supervised fine-tuning of a decision/classification head.

A thin, convention-based trainer over the HuggingFace Auto-class family: it fine-tunes
a task head on any recognized backbone, covering the "decision model" primitives via
native HF classes (no per-model code, no vendored packages):

    task="sequence_classification"  -> AutoModelForSequenceClassification
        fixed-label classification and yes/no judgments (data: text + label).
    task="multiple_choice"          -> AutoModelForMultipleChoice
        pick one of K candidate options per example (data: text + options[] + label-index).
    task="token_classification"     -> AutoModelForTokenClassification
        span / entity extraction (data: tokens[] + tags[]).

Backbones: encoder models (ModernBERT/DeBERTa/BERT) are full fine-tuned; decoder models
(Qwen, Llama, ...) via LoRA (``use_peft=True``). Custom architectures that follow HF's
``auto_map`` convention load with ``trust_remote_code=True`` (pin ``revision`` for safety)
— the model author's code, nothing vendored here. For a backbone distributed as a pip
package rather than ``auto_map``, point ``model_class`` at an importable
``*.from_pretrained`` class (the user installs that package; we take on no dependency).

Calibration (temperature scaling / ECE) is applied at evaluation time, not here.
"""

import importlib
import json
import logging
import os
from typing import Any, Callable, Optional

from . import Algorithm, Backend, AlgorithmRegistry

logger = logging.getLogger(__name__)

_TASKS = ("sequence_classification", "multiple_choice", "token_classification")

_KNOWN_PARAMS: set[str] = {
    "model_path",
    "data_path",
    "ckpt_output_dir",
    "task",
    "num_epochs",
    "batch_size",
    "learning_rate",
    "max_seq_len",
    "text_column",
    "label_column",
    "options_column",
    "tokens_column",
    "use_peft",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "trust_remote_code",
    "revision",
    "model_class",
    "loss_fn",
    "warmup_ratio",
    "weight_decay",
    "gradient_accumulation_steps",
    "seed",
}


def _resolve_dotted(path: str) -> Any:
    """Import a dotted path like ``pkg.mod.Class`` -> the attribute. Used for the
    ``model_class`` / ``loss_fn`` escape hatches so users bring their own code (that
    they have installed) without this package depending on it."""
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Expected a dotted import path like 'package.module.Name', got {path!r}")
    return getattr(importlib.import_module(module_path), attr)


def _encode_labels(values: list[Any]) -> tuple[dict[int, str], dict[Any, int], bool]:
    """Return (id2label, label2id, already_int) for a flat list of labels.

    Integer labels keep their value (head spans 0..max); strings/bools/floats map to
    contiguous sorted codes."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        raise ValueError("Dataset has no non-null labels to train on.")
    sample = non_null[0]
    if isinstance(sample, int) and not isinstance(sample, bool):
        max_id = max(int(v) for v in non_null)
        id2label = {i: str(i) for i in range(max_id + 1)}
        return id2label, {v: v for v in range(max_id + 1)}, True
    uniques = sorted({str(v) for v in non_null})
    label2id = {name: idx for idx, name in enumerate(uniques)}
    id2label = {idx: name for name, idx in label2id.items()}
    return id2label, label2id, False


class DecisionHeadBackend(Backend):
    """Fine-tunes an HF Auto-class task head with the ``Trainer`` + cross-entropy."""

    def execute_training(self, algorithm_params: dict[str, Any]) -> Any:
        import torch
        import transformers
        from transformers import AutoTokenizer, Trainer, TrainingArguments

        unknown = [k for k in algorithm_params if k not in _KNOWN_PARAMS]
        if unknown:
            logger.warning(
                "Ignoring unrecognized classifier_sft parameter(s): %s", ", ".join(sorted(unknown))
            )

        p = algorithm_params
        task = p.get("task", "sequence_classification")
        if task not in _TASKS:
            raise ValueError(f"Unknown task {task!r}. Choose from: {list(_TASKS)}")

        model_path = p["model_path"]
        data_path = p["data_path"]
        ckpt_output_dir = p["ckpt_output_dir"]
        max_seq_len = p.get("max_seq_len", 512)
        use_peft = p.get("use_peft", False)
        trust_remote_code = p.get("trust_remote_code", False)
        revision = p.get("revision")  # pin custom-code repos to a commit for safety
        seed = p.get("seed", 42)

        from_pretrained_kw: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision:
            from_pretrained_kw["revision"] = revision

        logger.info("Loading tokenizer: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **from_pretrained_kw)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Preparing data for task=%s", task)
        prepared = _PREP[task](self, data_path, tokenizer, max_seq_len, p)
        tokenized, collator, id2label, head_kwargs = prepared
        label2id = {name: idx for idx, name in id2label.items()}

        # Model: an explicit user-supplied class (escape hatch) or the task's Auto class.
        if p.get("model_class"):
            model_cls = _resolve_dotted(p["model_class"])
            logger.info("Loading model via user-supplied class: %s", p["model_class"])
        else:
            model_cls = getattr(transformers, _AUTO_CLASS[task])
        model = model_cls.from_pretrained(
            model_path, id2label=id2label, label2id=label2id, **head_kwargs, **from_pretrained_kw
        )
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if use_peft:
            from peft import LoraConfig, TaskType, get_peft_model

            peft_task = {
                "sequence_classification": TaskType.SEQ_CLS,
                "multiple_choice": TaskType.SEQ_CLS,
                "token_classification": TaskType.TOKEN_CLS,
            }[task]
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type=peft_task,
                    r=p.get("lora_r", 16),
                    lora_alpha=p.get("lora_alpha", 32),
                    lora_dropout=p.get("lora_dropout", 0.05),
                    target_modules=p.get("lora_target_modules"),
                ),
            )
            logger.info("Applied LoRA (%s)", peft_task)

        args = TrainingArguments(
            output_dir=ckpt_output_dir,
            num_train_epochs=p.get("num_epochs", 3),
            per_device_train_batch_size=p.get("batch_size", 16),
            gradient_accumulation_steps=p.get("gradient_accumulation_steps", 1),
            learning_rate=p.get("learning_rate", 2e-5),
            # transformers v5 removed `warmup_ratio`; `warmup_steps` as a float is the ratio.
            warmup_steps=p.get("warmup_ratio", 0.0),
            weight_decay=p.get("weight_decay", 0.01),
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            seed=seed,
            bf16=torch.cuda.is_available(),
        )

        trainer_kwargs: dict[str, Any] = dict(
            model=model, args=args, train_dataset=tokenized, data_collator=collator
        )
        # Optional custom objective (dotted path) — a callable loss or a Trainer subclass.
        if p.get("loss_fn"):
            loss_obj = _resolve_dotted(p["loss_fn"])
            if isinstance(loss_obj, type) and issubclass(loss_obj, Trainer):
                trainer = loss_obj(**trainer_kwargs)
            else:
                trainer = _trainer_with_loss(loss_obj, **trainer_kwargs)
            logger.info("Using custom loss/trainer: %s", p["loss_fn"])
        else:
            trainer = Trainer(**trainer_kwargs)

        logger.info("Training task=%s on %d examples", task, len(tokenized))
        trainer.train()

        os.makedirs(ckpt_output_dir, exist_ok=True)
        to_save = model.merge_and_unload() if use_peft else model
        to_save.save_pretrained(ckpt_output_dir)
        tokenizer.save_pretrained(ckpt_output_dir)
        with open(os.path.join(ckpt_output_dir, "label_mapping.json"), "w") as f:
            json.dump({"task": task, "id2label": id2label, "label2id": label2id}, f, indent=2)
        logger.info("Saved %s model to %s", task, ckpt_output_dir)
        return {"ckpt_output_dir": ckpt_output_dir, "task": task, "id2label": id2label}

    # ---- per-task data preparation -------------------------------------------------
    def _prep_sequence_classification(self, data_path, tokenizer, max_seq_len, p):
        from datasets import Value
        from transformers import DataCollatorWithPadding

        from training_hub.utils import load_training_dataset

        text_col = p.get("text_column") or "text"
        label_col = p.get("label_column") or "label"
        ds = load_training_dataset(data_path)
        if text_col != "text" and text_col in ds.column_names:
            ds = ds.rename_column(text_col, "text")
        if label_col != "label" and label_col in ds.column_names:
            ds = ds.rename_column(label_col, "label")
        for col in ("text", "label"):
            if col not in ds.column_names:
                raise ValueError(f"sequence_classification needs a '{col}' column. Found: {ds.column_names}")
        id2label, label2id, already_int = _encode_labels(ds["label"])
        if not already_int:
            ds = ds.map(lambda b: {"label": [label2id[str(v)] for v in b["label"]]}, batched=True)
        ds = ds.cast_column("label", Value("int64"))
        drop = [c for c in ds.column_names if c != "label"]
        tokenized = ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=max_seq_len),
            batched=True,
            remove_columns=drop,
        )
        return tokenized, DataCollatorWithPadding(tokenizer), id2label, {"num_labels": len(id2label)}

    def _prep_token_classification(self, data_path, tokenizer, max_seq_len, p):
        from transformers import DataCollatorForTokenClassification

        from training_hub.utils import load_training_dataset

        tokens_col = p.get("tokens_column") or "tokens"
        label_col = p.get("label_column") or "ner_tags"
        ds = load_training_dataset(data_path)
        for col in (tokens_col, label_col):
            if col not in ds.column_names:
                raise ValueError(f"token_classification needs a '{col}' column. Found: {ds.column_names}")
        flat = [t for row in ds[label_col] for t in (row or [])]
        id2label, label2id, already_int = _encode_labels(flat)

        def _align(example):
            enc = tokenizer(
                example[tokens_col], truncation=True, max_length=max_seq_len, is_split_into_words=True
            )
            word_ids = enc.word_ids()
            tags = example[label_col]
            labels, prev = [], None
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)  # special tokens
                elif wid != prev:
                    t = tags[wid]
                    labels.append(t if already_int else label2id[str(t)])
                else:
                    labels.append(-100)  # subword continuation
                prev = wid
            enc["labels"] = labels
            return enc

        tokenized = ds.map(_align, remove_columns=ds.column_names)
        collator = DataCollatorForTokenClassification(tokenizer, max_length=max_seq_len)
        return tokenized, collator, id2label, {"num_labels": len(id2label)}

    def _prep_multiple_choice(self, data_path, tokenizer, max_seq_len, p):
        from datasets import Value

        from training_hub.utils import load_training_dataset

        text_col = p.get("text_column") or "text"
        options_col = p.get("options_column") or "options"
        label_col = p.get("label_column") or "label"
        ds = load_training_dataset(data_path)
        for col in (text_col, options_col, label_col):
            if col not in ds.column_names:
                raise ValueError(f"multiple_choice needs a '{col}' column. Found: {ds.column_names}")

        def _tok(example):
            context = example[text_col]
            options = list(example[options_col])
            enc = tokenizer(
                [context] * len(options), options, truncation=True, max_length=max_seq_len
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": int(example[label_col]),
            }

        tokenized = ds.map(_tok, remove_columns=ds.column_names)
        tokenized = tokenized.cast_column("label", Value("int64"))
        # Multiple-choice has a dynamic per-question option set; id2label indexes positions.
        max_k = max(len(x) for x in tokenized["input_ids"])
        id2label = {i: f"option_{i}" for i in range(max_k)}
        return tokenized, _MultipleChoiceCollator(tokenizer), id2label, {}


class _MultipleChoiceCollator:
    """Pads a batch of per-example choice lists into [batch, num_choices, seq] tensors.

    Variable option counts are padded up to the batch max with empty (all-pad) choices;
    the correct-option index (``label``) is unaffected."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        import torch

        labels = [f["label"] for f in features]
        num_choices = max(len(f["input_ids"]) for f in features)
        flat: list[dict] = []
        for f in features:
            ids = list(f["input_ids"])
            masks = list(f["attention_mask"])
            while len(ids) < num_choices:  # pad missing choices with an empty sequence
                ids.append([self.tokenizer.pad_token_id])
                masks.append([0])
            for c in range(num_choices):
                flat.append({"input_ids": ids[c], "attention_mask": masks[c]})
        padded = self.tokenizer.pad(flat, return_tensors="pt")
        bsz, seq = len(features), padded["input_ids"].shape[-1]
        batch = {k: v.view(bsz, num_choices, seq) for k, v in padded.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def _trainer_with_loss(loss_fn: Callable, **trainer_kwargs):
    """Wrap a callable ``loss_fn(logits, labels) -> loss`` in a Trainer subclass."""
    from transformers import Trainer

    class _CustomLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    return _CustomLossTrainer(**trainer_kwargs)


# task -> (data-prep method name, Auto class name)
_PREP = {
    "sequence_classification": DecisionHeadBackend._prep_sequence_classification,
    "multiple_choice": DecisionHeadBackend._prep_multiple_choice,
    "token_classification": DecisionHeadBackend._prep_token_classification,
}
_AUTO_CLASS = {
    "sequence_classification": "AutoModelForSequenceClassification",
    "multiple_choice": "AutoModelForMultipleChoice",
    "token_classification": "AutoModelForTokenClassification",
}


class ClassifierSFTAlgorithm(Algorithm):
    """Algorithm for decision/classification-head fine-tuning."""

    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend

    def train(self, **kwargs) -> Any:
        return self.backend.execute_training(kwargs)

    def get_required_params(self) -> dict[str, type]:
        return {"model_path": str, "data_path": str, "ckpt_output_dir": str}

    def get_optional_params(self) -> dict[str, type]:
        return {
            "task": str,
            "num_epochs": int,
            "batch_size": int,
            "learning_rate": float,
            "max_seq_len": int,
            "text_column": str,
            "label_column": str,
            "options_column": str,
            "tokens_column": str,
            "use_peft": bool,
            "lora_r": int,
            "lora_alpha": int,
            "lora_dropout": float,
            "trust_remote_code": bool,
            "revision": str,
            "model_class": str,
            "loss_fn": str,
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
    backend: str = "decision-head",
    task: str = "sequence_classification",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_len: int = 512,
    text_column: str = "text",
    # None lets each task apply its own default (label for seq/MC, ner_tags for token-cls).
    label_column: Optional[str] = None,
    options_column: str = "options",
    tokens_column: str = "tokens",
    use_peft: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    model_class: Optional[str] = None,
    loss_fn: Optional[str] = None,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """Fine-tune a decision/classification head. See the module docstring for tasks.

    Args:
        task: ``sequence_classification`` (default), ``multiple_choice``, or
            ``token_classification``.
        trust_remote_code / revision: load custom architectures that ship an HF
            ``auto_map`` (pin ``revision`` to a commit for safety).
        model_class: dotted path to an importable ``*.from_pretrained`` class, for a
            backbone distributed as a pip package rather than ``auto_map``.
        loss_fn: dotted path to a custom ``loss_fn(logits, labels)`` callable or a
            ``Trainer`` subclass.

    Example:
        classifier_sft(
            model_path="answerdotai/ModernBERT-base",
            data_path="intents.parquet",
            ckpt_output_dir="./intent_clf",
            text_column="text", label_column="category",
        )
    """
    from . import create_algorithm

    algorithm = create_algorithm("classifier_sft", backend)
    return algorithm.train(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        task=task,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        text_column=text_column,
        label_column=label_column,
        options_column=options_column,
        tokens_column=tokens_column,
        use_peft=use_peft,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        trust_remote_code=trust_remote_code,
        revision=revision,
        model_class=model_class,
        loss_fn=loss_fn,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        **kwargs,
    )


# Back-compat alias: the sequence-classification backend name used by the first prototype.
SequenceClassificationBackend = DecisionHeadBackend

AlgorithmRegistry.register_algorithm("classifier_sft", ClassifierSFTAlgorithm)
AlgorithmRegistry.register_backend("classifier_sft", "decision-head", DecisionHeadBackend)
