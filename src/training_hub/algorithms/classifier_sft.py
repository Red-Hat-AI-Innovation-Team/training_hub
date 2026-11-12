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

Decoder note: a fresh classification head on a decoder backbone trains far slower than
an encoder's — the default ``learning_rate=2e-5`` suits encoders, but decoders with a
newly initialized head typically need ~1e-4..1e-3 and more epochs to move off a
single-class collapse (verified empirically: Qwen2.5-1.5B + LoRA collapsed at 2e-5,
fit its training set at 1e-3).

Calibration (temperature scaling / ECE) is applied at evaluation time, not here.
"""

import importlib
import json
import logging
import math
import os
from typing import Any, Callable, Optional

from . import Algorithm, Backend, AlgorithmRegistry

logger = logging.getLogger(__name__)

_TASKS = ("sequence_classification", "multiple_choice", "token_classification")
# Token-classification label that the cross-entropy loss ignores (HF convention).
_IGNORE_INDEX = -100
# Per-choice model inputs carried through multiple-choice prep + collation, with the value a
# padded (placeholder) choice uses for each: one attended pad token, segment 0.
_MC_INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids")
# Task-head module names PEFT keeps fully trainable (modules_to_save) for SEQ_CLS / TOKEN_CLS;
# never LoRA-wrap them.
_HEAD_MODULE_NAMES = ("classifier", "score")

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
    contiguous sorted codes. Float labels are treated as categorical (remapped to
    contiguous codes) — unlike embedding_sft, which preserves integer-valued floats;
    the mapping is recorded in label_mapping.json either way."""
    # Arrow nulls arrive as None, but a parquet/pandas float column carries missing
    # values as NaN (a float, so `is not None` lets it through) — a NaN would otherwise
    # become a literal "nan" class. Treat both as null.
    non_null = [v for v in values if not (v is None or (isinstance(v, float) and math.isnan(v)))]
    if not non_null:
        raise ValueError("Dataset has no non-null labels to train on.")
    if len(non_null) < len(values):
        # Unlike embedding_sft (whose losses tolerate None), a classification head
        # has no null consumer: the string path would KeyError on str(None) and the
        # int path would fail at tensorification. Fail loudly with the count instead.
        raise ValueError(
            f"Dataset has {len(values) - len(non_null)} null label(s) of {len(values)} rows. "
            "Remove or fill them before training — a classification head cannot train on "
            "unlabeled rows."
        )
    sample = non_null[0]
    if isinstance(sample, int) and not isinstance(sample, bool):
        negatives = sorted({int(v) for v in non_null if int(v) < 0})
        if negatives:
            # A negative id is outside the head's 0..N-1 class range; left alone it surfaces
            # only as an opaque CUDA device-side assert inside the loss.
            raise ValueError(
                f"Integer class labels must be non-negative class ids (0..N-1); found negative "
                f"label(s) {negatives[:5]}. Remap or drop them before training."
            )
        max_id = max(int(v) for v in non_null)
        id2label = {i: str(i) for i in range(max_id + 1)}
        return id2label, {v: v for v in range(max_id + 1)}, True
    uniques = sorted({str(v) for v in non_null})
    label2id = {name: idx for idx, name in enumerate(uniques)}
    id2label = {idx: name for name, idx in label2id.items()}
    return id2label, label2id, False


def _set_pad_token_id(config: Any, pad_token_id: Any) -> None:
    """Set ``pad_token_id`` where the classification head will read it. Decoder heads need it
    to find each row's last real token; models with a nested text config (e.g. Qwen3.5, whose
    head reads ``config.text_config.pad_token_id``) need it on the sub-config too. Never
    overrides a value the checkpoint already defines."""
    configs = [config]
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        text_config = get_text_config()
        if text_config is not None and text_config is not config:
            configs.append(text_config)
    for cfg in configs:
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = pad_token_id


def _require_columns(ds: Any, task: str, **columns: str) -> None:
    """Fail clearly when a configured column is missing, naming the parameter and the column
    the user passed (not a canonical default)."""
    missing = [f"{param}={col!r}" for param, col in columns.items() if col not in ds.column_names]
    if missing:
        raise ValueError(
            f"{task}: column(s) not found in the dataset: {', '.join(missing)}. "
            f"Available columns: {ds.column_names}"
        )


def _align_word_labels(
    word_ids: list[Optional[int]],
    tags: list[Any],
    label2id: dict[Any, int],
    already_int: bool,
) -> list[int]:
    """Map per-word token-classification tags onto tokenizer sub-tokens.

    Given the ``word_ids()`` of a tokenized (pre-split) sentence and that sentence's
    per-word ``tags``, return a per-token label list: ``-100`` for special tokens and
    for sub-word continuations (only the first sub-token of each word carries the label,
    the standard HF convention so the loss ignores the rest), and the encoded tag id for
    each word's first sub-token."""
    labels: list[int] = []
    prev = None
    for wid in word_ids:
        if wid is None:
            labels.append(-100)  # special tokens
        elif wid != prev:
            t = tags[wid]
            labels.append(t if already_int or t == _IGNORE_INDEX else label2id[str(t)])
        else:
            labels.append(-100)  # sub-word continuation
        prev = wid
    return labels


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

        # Seed BEFORE loading: from_pretrained randomly initializes the new task head (and
        # get_peft_model the LoRA weights), which happens before the Trainer seeds in its
        # __init__ — without this, two runs with the same seed get different heads.
        transformers.set_seed(seed)

        # Model: an explicit user-supplied class (escape hatch) or the task's Auto class.
        if p.get("model_class"):
            model_cls = _resolve_dotted(p["model_class"])
            logger.info("Loading model via user-supplied class: %s", p["model_class"])
        else:
            model_cls = getattr(transformers, _AUTO_CLASS[task])
        model = model_cls.from_pretrained(
            model_path, id2label=id2label, label2id=label2id, **head_kwargs, **from_pretrained_kw
        )
        _set_pad_token_id(model.config, tokenizer.pad_token_id)

        if use_peft:
            model = _apply_lora(model, task, p)

        # Warmup is version-dependent: transformers 4.x has a `warmup_ratio` field and
        # rejects a float passed to `warmup_steps`; transformers 5.x removed
        # `warmup_ratio` and accepts a float `warmup_steps` as the ratio. The dependency
        # floor (>=4.57.6) spans both, so route the ratio to whichever field exists.
        from dataclasses import fields as _dataclass_fields

        warmup = p.get("warmup_ratio", 0.0)
        if any(f.name == "warmup_ratio" for f in _dataclass_fields(TrainingArguments)):
            warmup_kwargs = {"warmup_ratio": warmup}
        else:
            warmup_kwargs = {"warmup_steps": warmup}

        args = TrainingArguments(
            output_dir=ckpt_output_dir,
            num_train_epochs=p.get("num_epochs", 3),
            per_device_train_batch_size=p.get("batch_size", 16),
            gradient_accumulation_steps=p.get("gradient_accumulation_steps", 1),
            learning_rate=p.get("learning_rate", 2e-5),
            weight_decay=p.get("weight_decay", 0.01),
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            seed=seed,
            # bf16 needs GPU support (Ampere+), not just CUDA: TrainingArguments raises on a
            # pre-Ampere GPU when bf16=True, so fall back to fp32 there.
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            **warmup_kwargs,
        )

        trainer_kwargs: dict[str, Any] = dict(
            model=model, args=args, train_dataset=tokenized, data_collator=collator
        )
        # Optional custom objective (dotted path) — a callable loss or a Trainer subclass.
        trainer_cls: Optional[type] = None
        if p.get("loss_fn"):
            loss_obj = _resolve_dotted(p["loss_fn"])
            if isinstance(loss_obj, type) and issubclass(loss_obj, Trainer):
                trainer_cls = loss_obj
                trainer = loss_obj(**trainer_kwargs)
            else:
                trainer = _trainer_with_loss(loss_obj, **trainer_kwargs)
            logger.info("Using custom loss/trainer: %s", p["loss_fn"])
        elif task == "multiple_choice":
            # Mixed-K batches are padded with empty choices by _MultipleChoiceCollator;
            # mask those slots out of the softmax so they never compete with real options
            # in the default cross-entropy.
            trainer = _multiple_choice_trainer(**trainer_kwargs)
        else:
            trainer = Trainer(**trainer_kwargs)
        _normalize_loss_over_accumulation(trainer, trainer_cls)

        logger.info("Training task=%s on %d examples", task, len(tokenized))
        trainer.train()

        os.makedirs(ckpt_output_dir, exist_ok=True)
        # Kubeflow derives progress from max_steps in training_metrics.jsonl (see AGENTS.md);
        # mirror embedding_sft's entry.
        with open(os.path.join(ckpt_output_dir, "training_metrics.jsonl"), "a") as f:
            f.write(json.dumps(_training_metrics_entry(trainer.state, task)) + "\n")

        to_save = model.merge_and_unload() if use_peft else model
        to_save.save_pretrained(ckpt_output_dir)
        tokenizer.save_pretrained(ckpt_output_dir)
        with open(os.path.join(ckpt_output_dir, "label_mapping.json"), "w") as f:
            json.dump({"task": task, "id2label": id2label, "label2id": label2id}, f, indent=2)
        logger.info("Saved %s model to %s", task, ckpt_output_dir)
        return {"ckpt_output_dir": ckpt_output_dir, "task": task, "id2label": id2label}

    # ---- per-task data preparation -------------------------------------------------
    def _prep_sequence_classification(
        self, data_path: str, tokenizer: Any, max_seq_len: int, p: dict[str, Any]
    ) -> tuple[Any, Any, dict[int, str], dict[str, Any]]:
        from datasets import Value
        from transformers import DataCollatorWithPadding

        from training_hub.utils import load_training_dataset

        text_col = p.get("text_column") or "text"
        label_col = p.get("label_column") or "label"
        if text_col == label_col:
            raise ValueError(f"text_column and label_column must differ (both are {text_col!r}).")
        ds = load_training_dataset(data_path)
        _require_columns(ds, "sequence_classification", text_column=text_col, label_column=label_col)
        # Keep only the two user columns and read them by name — no renaming, so an existing
        # `text`/`label` column can't collide with (or silently stand in for) the requested one.
        ds = ds.select_columns([text_col, label_col])
        id2label, label2id, already_int = _encode_labels(ds[label_col])

        def _tok(batch: dict[str, list[Any]]) -> dict[str, Any]:
            enc = tokenizer(batch[text_col], truncation=True, max_length=max_seq_len)
            enc["label"] = batch[label_col] if already_int else [label2id[str(v)] for v in batch[label_col]]
            return enc

        tokenized = ds.map(_tok, batched=True, remove_columns=ds.column_names)
        # `map` keeps the source column's Arrow type; force int64 so float-typed labels can't
        # round-trip back to floats (the embedding_sft #154 lesson).
        tokenized = tokenized.cast_column("label", Value("int64"))
        return tokenized, DataCollatorWithPadding(tokenizer), id2label, {"num_labels": len(id2label)}

    def _prep_token_classification(
        self, data_path: str, tokenizer: Any, max_seq_len: int, p: dict[str, Any]
    ) -> tuple[Any, Any, dict[int, str], dict[str, Any]]:
        from transformers import DataCollatorForTokenClassification

        from training_hub.utils import load_training_dataset

        tokens_col = p.get("tokens_column") or "tokens"
        label_col = p.get("label_column") or "ner_tags"
        ds = load_training_dataset(data_path)
        _require_columns(ds, "token_classification", tokens_column=tokens_col, label_column=label_col)
        # -100 is the standard "ignore this token" marker (the loss skips it); keep it in the
        # data but out of the label vocabulary.
        flat = [t for row in ds[label_col] for t in (row or []) if t != _IGNORE_INDEX]
        id2label, label2id, already_int = _encode_labels(flat)

        def _align(example):
            enc = tokenizer(
                example[tokens_col], truncation=True, max_length=max_seq_len, is_split_into_words=True
            )
            enc["labels"] = _align_word_labels(
                enc.word_ids(), example[label_col], label2id, already_int
            )
            return enc

        tokenized = ds.map(_align, remove_columns=ds.column_names)
        # Truncation to max_seq_len already happened in _align; the collator pads dynamically
        # (passing max_length here is ignored and only emits a warning).
        collator = DataCollatorForTokenClassification(tokenizer)
        return tokenized, collator, id2label, {"num_labels": len(id2label)}

    def _prep_multiple_choice(self, data_path, tokenizer, max_seq_len, p):
        from datasets import Value

        from training_hub.utils import load_training_dataset

        text_col = p.get("text_column") or "text"
        options_col = p.get("options_column") or "options"
        label_col = p.get("label_column") or "label"
        ds = load_training_dataset(data_path)
        _require_columns(
            ds, "multiple_choice", text_column=text_col, options_column=options_col, label_column=label_col
        )

        def _tok(example: dict[str, Any]) -> dict[str, Any]:
            context = example[text_col]
            options = list(example[options_col])
            label = int(example[label_col])
            if not 0 <= label < len(options):
                raise ValueError(
                    f"multiple_choice: {label_col}={label} is not a valid option index for a "
                    f"question with {len(options)} option(s) (expected 0..{len(options) - 1})."
                )
            enc = tokenizer(
                [context] * len(options), options, truncation=True, max_length=max_seq_len
            )
            # Keep every model input the tokenizer produced — including token_type_ids, which
            # BERT-family models use to tell the context segment from the option segment.
            out = {k: enc[k] for k in _MC_INPUT_KEYS if k in enc}
            out["label"] = label
            return out

        tokenized = ds.map(_tok, remove_columns=ds.column_names)
        tokenized = tokenized.cast_column("label", Value("int64"))
        # Multiple-choice has a dynamic per-question option set; id2label indexes positions.
        # Compute the max option count from the raw options column, not from the
        # tokenized input_ids — materializing every tokenized option list at once
        # is a memory blowup on large datasets; the raw strings are far lighter.
        max_k = max(len(x) for x in ds[options_col])
        id2label = {i: f"option_{i}" for i in range(max_k)}
        return tokenized, _MultipleChoiceCollator(tokenizer), id2label, {}


class _MultipleChoiceCollator:
    """Pads a batch of per-example choice lists into [batch, num_choices, seq] tensors.

    Variable option counts are padded up to the batch max with placeholder choices (a single
    attended pad token, so pooling stays finite); the correct-option index (``label``) is
    unaffected. A boolean ``choice_mask`` (True = real choice) marks the padded slots so the
    loss can exclude them."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        import torch

        labels = [f["label"] for f in features]
        real_counts = [len(f["input_ids"]) for f in features]
        num_choices = max(real_counts)
        keys = [k for k in _MC_INPUT_KEYS if k in features[0]]
        # Placeholder for a missing choice: ONE ATTENDED pad token (attention 1, not 0) in
        # segment 0. An all-zero mask makes mean-pooling heads (e.g. ModernBERT's default)
        # divide by zero -> NaN logits; masked_fill(-inf) hides that in the forward pass but
        # the backward pass still writes NaN into every weight. A real token keeps the forward
        # finite, and choice_mask below still drops the slot from the softmax (zero gradient).
        placeholder = {"input_ids": [self.tokenizer.pad_token_id], "attention_mask": [1], "token_type_ids": [0]}
        flat: list[dict] = []
        for f in features:
            per_key = {k: list(f[k]) for k in keys}
            for _ in range(num_choices - len(per_key["input_ids"])):
                for k in keys:
                    per_key[k].append(placeholder[k])
            for c in range(num_choices):
                flat.append({k: per_key[k][c] for k in keys})
        padded = self.tokenizer.pad(flat, return_tensors="pt")
        bsz, seq = len(features), padded["input_ids"].shape[-1]
        batch = {k: v.view(bsz, num_choices, seq) for k, v in padded.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        # True for real choices, False for padded slots; _multiple_choice_trainer masks the
        # padded slots to -inf before cross-entropy so variable-K batches stay unbiased.
        choice_mask = torch.zeros(bsz, num_choices, dtype=torch.bool)
        for i, n in enumerate(real_counts):
            choice_mask[i, :n] = True
        batch["choice_mask"] = choice_mask
        return batch


def _training_metrics_entry(state: Any, task: str) -> dict[str, Any]:
    """The ``training_metrics.jsonl`` record for a finished run (``state`` = ``trainer.state``).

    The last ``log_history`` record may lack ``train_loss`` (e.g. an eval/save entry), so walk
    back to the most recent one that has it. Non-finite measurements (a diverged run's NaN
    loss) become ``None``: ``json.dumps`` would otherwise write bare ``NaN``/``Infinity``,
    which strict JSON readers reject."""

    def _finite(v: Any) -> Any:
        return v if not isinstance(v, float) or math.isfinite(v) else None

    train_loss = next(
        (e["train_loss"] for e in reversed(state.log_history) if e.get("train_loss") is not None),
        None,
    )
    return {
        "max_steps": getattr(state, "max_steps", None),
        "global_step": state.global_step,
        "train_loss": _finite(train_loss),
        "epoch": _finite(state.epoch),
        "task": task,
    }


def _apply_lora(model: Any, task: str, p: dict[str, Any]) -> Any:
    """Wrap ``model`` with LoRA for the task.

    ``lora_target_modules=None`` uses PEFT's built-in targets for the architecture. PEFT has
    no defaults for some backbones (e.g. ModernBERT, Qwen3.5) and raises "Please specify
    `target_modules`"; for those, fall back to ``"all-linear"``, which works across
    architectures. The task head itself stays fully trainable either way (PEFT puts it in
    ``modules_to_save`` for SEQ_CLS / TOKEN_CLS)."""
    from peft import LoraConfig, TaskType, get_peft_model

    peft_task = {
        "sequence_classification": TaskType.SEQ_CLS,
        "multiple_choice": TaskType.SEQ_CLS,
        "token_classification": TaskType.TOKEN_CLS,
    }[task]

    def _config(target_modules: Any, exclude_modules: Optional[list[str]] = None) -> Any:
        kwargs: dict[str, Any] = {}
        if exclude_modules:
            kwargs["exclude_modules"] = exclude_modules
        return LoraConfig(
            task_type=peft_task,
            r=p.get("lora_r", 16),
            lora_alpha=p.get("lora_alpha", 32),
            lora_dropout=p.get("lora_dropout", 0.05),
            target_modules=target_modules,
            **kwargs,
        )

    targets = p.get("lora_target_modules")
    try:
        wrapped = get_peft_model(model, _config(targets))
    except ValueError as e:
        if targets is not None or "target_modules" not in str(e):
            raise
        logger.info(
            "PEFT has no default LoRA targets for %s; using target_modules='all-linear'",
            getattr(model.config, "model_type", "this architecture"),
        )
        targets = "all-linear"
        # "all-linear" skips the classification head for SEQ_CLS but NOT for TOKEN_CLS, so the
        # head would be LoRA-wrapped and then can't go into modules_to_save. Exclude the head
        # names PEFT keeps trainable (modules_to_save) for these task types.
        wrapped = get_peft_model(model, _config(targets, exclude_modules=list(_HEAD_MODULE_NAMES)))
    logger.info("Applied LoRA (%s, target_modules=%s)", peft_task, targets or "PEFT default")
    return wrapped


def _normalize_loss_over_accumulation(trainer: Any, trainer_cls: Optional[type] = None) -> None:
    """Make the Trainer divide the loss by the gradient-accumulation steps.

    Every objective here — the HF task heads' cross-entropy and this module's
    ``compute_loss`` overrides — is a per-batch mean that ignores ``num_items_in_batch``.
    When the model's ``forward`` accepts ``**kwargs`` (e.g. ModernBERT) the Trainer sets
    ``model_accepts_loss_kwargs=True``, assumes the loss is already normalized across
    accumulation steps, and skips its own division — so gradients (and the logged loss)
    come out N x too large with ``gradient_accumulation_steps=N``. A user ``Trainer``
    subclass whose ``compute_loss`` genuinely consumes ``num_items_in_batch`` can opt out
    by declaring the class attribute ``model_accepts_loss_kwargs = True``."""
    if trainer_cls is not None and getattr(trainer_cls, "model_accepts_loss_kwargs", None) is True:
        return
    trainer.model_accepts_loss_kwargs = False


def _accepts_choice_mask(loss_fn: Callable) -> bool:
    """Whether a custom ``loss_fn`` can take the multiple-choice ``choice_mask`` keyword."""
    import inspect

    try:
        params = inspect.signature(loss_fn).parameters.values()
    except (TypeError, ValueError):  # builtins / C callables without a signature
        return False
    return any(p.name == "choice_mask" or p.kind is p.VAR_KEYWORD for p in params)


def _call_loss_fn(loss_fn: Callable, logits: Any, labels: Any, choice_mask: Any = None, pass_mask: bool = False) -> Any:
    """Call a custom ``loss_fn(logits, labels)``. For multiple choice, padded (placeholder)
    choices get ``-inf`` logits first — they carry finite logits from their attended pad
    token, so a plain two-argument cross-entropy would otherwise score them as real options.
    A ``loss_fn`` that accepts a ``choice_mask`` keyword also receives the boolean mask."""
    extra: dict[str, Any] = {}
    if choice_mask is not None:
        logits = logits.masked_fill(~choice_mask, float("-inf"))
        if pass_mask:
            extra["choice_mask"] = choice_mask
    return loss_fn(logits, labels, **extra)


def _trainer_with_loss(loss_fn: Callable, **trainer_kwargs):
    """Wrap a callable ``loss_fn(logits, labels) -> loss`` in a Trainer subclass."""
    from transformers import Trainer

    pass_mask = _accepts_choice_mask(loss_fn)

    class _CustomLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            # The MC collator adds a choice_mask the backbone doesn't accept: pop it so
            # model(**inputs) is clean, and use it to keep padded choices out of the loss.
            choice_mask = inputs.pop("choice_mask", None)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = _call_loss_fn(loss_fn, outputs.logits, labels, choice_mask, pass_mask)
            return (loss, outputs) if return_outputs else loss

    return _CustomLossTrainer(**trainer_kwargs)


def _multiple_choice_trainer(**trainer_kwargs):
    """Trainer for the default multiple-choice path: masks padded choice slots out of the
    softmax so variable-K batches are unbiased (padded options contribute no gradient).

    The backbone (``AutoModelForMultipleChoice``) is called without ``labels`` so it
    returns raw per-choice logits; we mask the padded slots to ``-inf`` and compute
    cross-entropy ourselves. ``choice_mask`` is popped so it is never passed to the model.
    """
    import torch.nn.functional as F
    from transformers import Trainer

    class _MultipleChoiceTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            choice_mask = inputs.pop("choice_mask", None)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if choice_mask is not None:
                logits = logits.masked_fill(~choice_mask, float("-inf"))
            loss = F.cross_entropy(logits, labels)
            outputs.loss = loss
            return (loss, outputs) if return_outputs else loss

    return _MultipleChoiceTrainer(**trainer_kwargs)


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
            "lora_target_modules": list,
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
    """Fine-tune a decision/classification head. See the module docstring for the tasks.

    Args:
        model_path: Backbone — an encoder (ModernBERT/DeBERTa/BERT, full fine-tune) or a
            decoder (Qwen/Llama, use ``use_peft=True``). HF hub id or local path.
        data_path: Training data (.jsonl/.json/.parquet/.csv or an HF dataset id). Columns
            expected depend on ``task`` (see the column args below).
        ckpt_output_dir: Directory to write the fine-tuned model, tokenizer, and
            ``label_mapping.json`` (task + id2label/label2id).
        backend: Backend implementation name (default ``decision-head``).
        task: ``sequence_classification`` (default), ``multiple_choice``, or
            ``token_classification``.
        num_epochs: Training epochs (default 3). Decoder backbones usually need more.
        batch_size: Per-device batch size (default 16).
        learning_rate: Learning rate (default ``2e-5``, tuned for encoders). A freshly
            initialized head on a decoder backbone typically needs ``~1e-4..1e-3`` and
            more epochs to avoid single-class collapse (see the module docstring).
        max_seq_len: Max token length (default 512).
        text_column: Text/context column for ``sequence_classification`` and
            ``multiple_choice`` (default ``text``).
        label_column: Label column. Default ``label`` for ``sequence_classification`` /
            ``multiple_choice``, ``ner_tags`` for ``token_classification``. String labels
            are encoded to contiguous ints (recorded in ``label_mapping.json``); integer
            labels must be non-negative class ids (token classification may use ``-100`` to
            ignore a token), and a multiple-choice label must index one of its options.
        options_column: Options-list column for ``multiple_choice`` (default ``options``).
        tokens_column: Pre-split tokens column for ``token_classification``
            (default ``tokens``).
        use_peft: Fine-tune with LoRA instead of a full fine-tune (recommended for
            decoder backbones). Adapters are merged into the base weights before saving.
        lora_r: LoRA rank (default 16).
        lora_alpha: LoRA alpha (default 32).
        lora_dropout: LoRA dropout (default 0.05).
        lora_target_modules: Explicit list of module names to adapt. ``None`` (default)
            uses PEFT's built-in targets for the architecture, falling back to
            ``"all-linear"`` when PEFT has none (e.g. ModernBERT, Qwen3.5).
        trust_remote_code: Allow loading a custom architecture that ships an HF
            ``auto_map`` (opt-in; the model author's code runs). Pin ``revision`` with it.
        revision: Model repo commit/branch/tag to pin — strongly recommended with
            ``trust_remote_code`` so the executed code can't change under you.
        model_class: Dotted path to an importable ``*.from_pretrained`` class, for a
            backbone distributed as a pip package rather than via ``auto_map``.
        loss_fn: Dotted path to a custom ``loss_fn(logits, labels)`` callable or a
            ``Trainer`` subclass, for a bespoke objective. Losses are treated as per-batch
            means (normalized across gradient-accumulation steps by the Trainer); a
            ``Trainer`` subclass that consumes ``num_items_in_batch`` itself can opt out by
            setting the class attribute ``model_accepts_loss_kwargs = True``. For
            ``multiple_choice``, padded placeholder choices reach a callable loss with
            ``-inf`` logits (a loss accepting a ``choice_mask`` keyword also gets the boolean
            mask); a ``Trainer`` subclass finds ``choice_mask`` in its batch and must pop it
            before calling the model.
        warmup_ratio: Fraction of total steps used for LR warmup (default 0.0).
        weight_decay: Weight decay (default 0.01).
        gradient_accumulation_steps: Steps to accumulate before an optimizer update
            (default 1).
        seed: Random seed (default 42). Seeds the task-head / LoRA initialization too, so
            identical runs reproduce (bit-exact on CPU).

    Returns:
        dict: ``{"ckpt_output_dir": str, "task": str, "id2label": dict[int, str]}`` —
        the saved checkpoint directory, the trained task, and the label mapping.

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


AlgorithmRegistry.register_algorithm("classifier_sft", ClassifierSFTAlgorithm)
AlgorithmRegistry.register_backend("classifier_sft", "decision-head", DecisionHeadBackend)
