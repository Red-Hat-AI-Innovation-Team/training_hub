"""Embedding SFT — contrastive fine-tuning for sentence embedding models.

Fine-tunes embedding models (e.g. all-MiniLM-L6-v2) using contrastive losses
so that inputs with the same label cluster together in embedding space. Designed
for semantic routing classifiers but applicable to any embedding classification task.

Example (semantic routing with triplet loss):
    from training_hub import embedding_sft

    result = embedding_sft(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        data_path="training_data.jsonl",
        ckpt_output_dir="./embedding_output",
        loss_type="batch_all_triplet",
        num_epochs=20,
        batch_size=32,
        learning_rate=2e-5,
    )

Data format (JSONL):
    {"text": "What is the capital of France?", "label": 0}
    {"text": "Design a microservices architecture", "label": 2}
"""

import json
import logging
import os
import random
from typing import Any, Callable, Optional

from . import Algorithm, Backend, AlgorithmRegistry

logger = logging.getLogger(__name__)

LOSS_REGISTRY: dict[str, str] = {
    "batch_all_triplet": "BatchAllTripletLoss",
    "batch_hard_triplet": "BatchHardTripletLoss",
    "mnrl": "MultipleNegativesRankingLoss",
}

TRIPLET_LOSSES = {"batch_all_triplet", "batch_hard_triplet"}

# All recognized embedding_sft parameters (used to warn on unknown kwargs/typos).
# Kept in sync with EmbeddingSFTAlgorithm.get_required_params / get_optional_params.
_KNOWN_PARAMS: set[str] = {
    "model_path", "data_path", "ckpt_output_dir",
    "loss_type", "loss_fn", "num_epochs", "batch_size", "learning_rate",
    "warmup_ratio", "batch_sampler", "eval_data_path", "text_column",
    "label_column", "seed",
}


def _load_dataset(
    data_path: str,
    text_column: str = "text",
    label_column: str = "label",
    *,
    require_text: bool = True,
    require_label: bool = True,
) -> Any:
    """Load a text/label dataset from a JSONL/CSV file or HuggingFace dataset ID.

    Args:
        data_path: Path to a .jsonl/.json/.csv file, or a HuggingFace dataset ID.
        text_column: Name of the column to rename to ``text`` (if present).
        label_column: Name of the column to rename to ``label`` (if present).
        require_text: When True (default), require a ``text`` column after renaming.
            Set False for custom losses that use a different column schema
            (e.g. ``sentence1``/``sentence2``).
        require_label: When True (default), require a ``label`` column after
            renaming. Set False for pair-formatted or custom-loss datasets.

    Returns:
        A HuggingFace ``Dataset`` with the requested columns (``text`` and/or
        ``label`` after renaming, or whatever the custom loss expects when
        requirements are relaxed).

    Raises:
        ValueError: If the file extension is unsupported, or if a required
            column is missing after renaming.
    """
    from datasets import load_dataset

    if os.path.isfile(data_path):
        ext = os.path.splitext(data_path)[1].lower()
        if ext in (".jsonl", ".json"):
            dataset = load_dataset("json", data_files=data_path, split="train")
        elif ext == ".csv":
            dataset = load_dataset("csv", data_files=data_path, split="train")
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}' for data_path '{data_path}'. "
                f"Use .jsonl, .json, or .csv (or pass a HuggingFace dataset ID)."
            )
    else:
        dataset = load_dataset(data_path, split="train")

    if text_column != "text" and text_column in dataset.column_names:
        dataset = dataset.rename_column(text_column, "text")
    if label_column != "label" and label_column in dataset.column_names:
        dataset = dataset.rename_column(label_column, "label")

    if require_text and "text" not in dataset.column_names:
        raise ValueError(
            f"Dataset must have a 'text' column (or specify text_column). "
            f"Found: {dataset.column_names}"
        )
    if require_label and "label" not in dataset.column_names:
        raise ValueError(
            f"Dataset must have a 'label' column (or specify label_column). "
            f"Found: {dataset.column_names}"
        )

    return dataset


def _to_pair_dataset(dataset: Any, max_pairs_per_label: int = 10_000, seed: int = 42) -> Any:
    """Convert a label-based dataset to (anchor, positive) pairs for MNRL.

    To avoid the O(n^2) memory blow-up of materializing all within-label pairs,
    each label's texts are down-sampled to at most the number needed to produce
    ``max_pairs_per_label`` pairs *before* pair construction. For a label with n
    texts this caps memory at ~max_pairs_per_label pairs rather than n*(n-1)/2.

    Args:
        dataset: A HuggingFace ``Dataset`` with ``text`` and ``label`` columns.
        max_pairs_per_label: Upper bound on pairs generated per label. <= 0 means
            no cap (use with care on large labels).
        seed: Random seed for the source-text down-sampling and pair sub-sampling.

    Returns:
        A HuggingFace ``Dataset`` with ``anchor`` and ``positive`` columns.
    """
    from datasets import Dataset
    from itertools import combinations
    from math import ceil

    groups: dict[int, list[str]] = {}
    for row in dataset:
        groups.setdefault(row["label"], []).append(row["text"])

    rng = random.Random(seed)
    pairs: list[dict[str, str]] = []
    for _label, texts in groups.items():
        # Cap the number of source texts so the pair count stays bounded. With k
        # texts there are k*(k-1)/2 pairs; pick k so that's <= max_pairs_per_label.
        if max_pairs_per_label <= 0:
            cap = len(texts)
        else:
            cap = ceil((1 + (1 + 8 * max_pairs_per_label) ** 0.5) / 2)
        if len(texts) > cap:
            texts = rng.sample(texts, cap)
        label_pairs = [{"anchor": a, "positive": b} for a, b in combinations(texts, 2)]
        if len(label_pairs) > max_pairs_per_label:
            label_pairs = rng.sample(label_pairs, max_pairs_per_label)
        pairs.extend(label_pairs)

    return Dataset.from_list(pairs)


class SentenceTransformersBackend(Backend):
    """Backend using the sentence-transformers library for embedding fine-tuning."""

    def execute_training(self, algorithm_params: dict[str, Any]) -> Any:
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
        # Prefer the nested module paths (sentence-transformers >=5.4) which avoid
        # the deprecation warning emitted by the top-level aliases in 6.x. Fall back
        # to the top-level paths for 5.0-5.3.x where the nested modules do not exist.
        try:
            from sentence_transformers.sentence_transformer.training_args import (
                SentenceTransformerTrainingArguments, BatchSamplers,
            )
            from sentence_transformers.sentence_transformer.losses import (
                BatchAllTripletLoss, BatchHardTripletLoss, MultipleNegativesRankingLoss,
            )
        except ImportError:
            from sentence_transformers.training_args import (
                SentenceTransformerTrainingArguments, BatchSamplers,
            )
            from sentence_transformers.losses import (
                BatchAllTripletLoss, BatchHardTripletLoss, MultipleNegativesRankingLoss,
            )
        import torch

        model_path = algorithm_params["model_path"]
        data_path = algorithm_params["data_path"]
        ckpt_output_dir = algorithm_params["ckpt_output_dir"]

        loss_type = algorithm_params.get("loss_type", "batch_all_triplet")
        loss_fn = algorithm_params.get("loss_fn")
        num_epochs = algorithm_params.get("num_epochs", 20)
        batch_size = algorithm_params.get("batch_size", 32)
        learning_rate = algorithm_params.get("learning_rate", 2e-5)
        warmup_ratio = algorithm_params.get("warmup_ratio", 0.1)
        batch_sampler_name = algorithm_params.get("batch_sampler")
        eval_data_path = algorithm_params.get("eval_data_path")
        seed = algorithm_params.get("seed", 42)
        text_column = algorithm_params.get("text_column", "text")
        label_column = algorithm_params.get("label_column", "label")

        # Warn about unrecognized parameters so typos (e.g. ``learing_rate``) don't
        # silently fall back to defaults.
        unknown = [k for k in algorithm_params if k not in _KNOWN_PARAMS]
        if unknown:
            logger.warning(
                "Ignoring unrecognized embedding_sft parameter(s): %s. "
                "Check for typos against the documented parameters.",
                ", ".join(sorted(unknown)),
            )

        # Validate loss_fn *before* loading the model, so an invalid custom loss
        # fails fast rather than after a potentially large model download.
        using_custom_loss = loss_fn is not None
        if using_custom_loss and not callable(loss_fn):
            raise TypeError(
                f"loss_fn must be callable, got {type(loss_fn).__name__}. "
                f"Pass a sentence-transformers loss or set loss_type to one of {list(LOSS_REGISTRY.keys())}."
            )

        logger.info("Loading model: %s", model_path)
        model = SentenceTransformer(model_path)

        # Resolve the loss. A custom loss_fn takes precedence over loss_type.
        if using_custom_loss:
            loss = loss_fn
            loss_label = f"custom ({type(loss_fn).__name__})"
            logger.info("Using custom loss function: %s", type(loss_fn).__name__)
        elif loss_type in LOSS_REGISTRY:
            loss_classes = {
                "batch_all_triplet": BatchAllTripletLoss,
                "batch_hard_triplet": BatchHardTripletLoss,
                "mnrl": MultipleNegativesRankingLoss,
            }
            loss = loss_classes[loss_type](model)
            loss_label = loss_type
            logger.info("Using loss: %s", loss_type)
        else:
            raise ValueError(
                f"Unknown loss_type '{loss_type}'. Choose from: {list(LOSS_REGISTRY.keys())} "
                f"or pass a custom loss_fn."
            )

        logger.info("Loading training data: %s", data_path)
        # A custom loss_fn owns its data format (e.g. ContrastiveLoss wants
        # sentence1/sentence2 + label), so don't force the text/label schema on it.
        train_dataset = _load_dataset(
            data_path, text_column, label_column,
            require_text=not using_custom_loss, require_label=not using_custom_loss,
        )

        # MNRL expects (anchor, positive) pairs rather than text/label rows. We
        # only auto-convert when using the built-in MNRL loss — a custom loss_fn
        # is responsible for its own data format.
        use_mnrl_pairs = (loss_type == "mnrl" and not using_custom_loss)
        if use_mnrl_pairs:
            train_dataset = _to_pair_dataset(train_dataset, seed=seed)
            logger.info("Converted to %d (anchor, positive) pairs for MNRL", len(train_dataset))
        else:
            logger.info("Training samples: %d", len(train_dataset))

        # Auto-select the batch sampler based on the *effective* loss, so a custom
        # loss_fn doesn't inherit a triplet-oriented sampler from the loss_type
        # default. Custom losses default to the generic sampler unless overridden.
        if batch_sampler_name is None:
            if use_mnrl_pairs:
                batch_sampler_name = "no_duplicates"
            elif using_custom_loss:
                batch_sampler_name = "default"
            else:
                batch_sampler_name = "group_by_label"

        eval_dataset = None
        if eval_data_path:
            # Eval data must match the training format: pairs for MNRL, text/label
            # otherwise. A custom loss_fn owns its format, so we relax requirements.
            if use_mnrl_pairs:
                eval_dataset = _load_dataset(eval_data_path, text_column, label_column)
                eval_dataset = _to_pair_dataset(eval_dataset, seed=seed)
                logger.info("Eval samples: %d (anchor, positive) pairs", len(eval_dataset))
            else:
                eval_dataset = _load_dataset(
                    eval_data_path, text_column, label_column,
                    require_text=not using_custom_loss, require_label=not using_custom_loss,
                )
                logger.info("Eval samples: %d", len(eval_dataset))

        sampler_map = {
            "group_by_label": BatchSamplers.GROUP_BY_LABEL,
            "no_duplicates": BatchSamplers.NO_DUPLICATES,
            "default": BatchSamplers.BATCH_SAMPLER,
        }
        batch_sampler = sampler_map.get(batch_sampler_name)
        if batch_sampler is None:
            raise ValueError(
                f"Unknown batch_sampler '{batch_sampler_name}'. "
                f"Choose from: {list(sampler_map.keys())}"
            )

        if (not using_custom_loss) and loss_type in TRIPLET_LOSSES and batch_sampler_name != "group_by_label":
            logger.warning(
                "Triplet losses work best with batch_sampler='group_by_label'. "
                "Using '%s' may produce batches with missing classes.",
                batch_sampler_name,
            )

        os.makedirs(ckpt_output_dir, exist_ok=True)

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        training_args = SentenceTransformerTrainingArguments(
            output_dir=ckpt_output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            seed=seed,
            batch_sampler=batch_sampler,
            save_strategy="epoch",
            save_total_limit=1,
            # Only enable eval when an eval dataset was provided; otherwise the
            # default "no" skips evaluation (and avoids the trainer requiring one).
            eval_strategy="epoch" if eval_dataset is not None else "no",
            logging_steps=10,
            fp16=False,
            bf16=use_bf16,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )

        logger.info(
            "Starting training: %d epochs, batch_size=%d, lr=%s, loss=%s, sampler=%s",
            num_epochs, batch_size, learning_rate, loss_label, batch_sampler_name,
        )
        trainer.train()

        # The last log_history entry may be an eval/save record without train_loss;
        # walk backwards to find the most recent entry that recorded a training loss.
        train_loss = None
        for entry in reversed(trainer.state.log_history):
            if entry.get("train_loss") is not None:
                train_loss = entry["train_loss"]
                break

        metrics_path = os.path.join(ckpt_output_dir, "training_metrics.jsonl")
        metrics_entry = {
            "max_steps": getattr(trainer.state, "max_steps", None),
            "global_step": trainer.state.global_step,
            "train_loss": train_loss,
            "epoch": num_epochs,
            "loss": loss_label,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics_entry) + "\n")

        model.save_pretrained(ckpt_output_dir)
        logger.info("Model saved to %s", ckpt_output_dir)

        return {
            "status": "success",
            "model_path": ckpt_output_dir,
            "num_samples": len(train_dataset),
            "num_epochs": num_epochs,
            "loss": loss_label,
        }


class EmbeddingSFTAlgorithm(Algorithm):
    """Algorithm for contrastive fine-tuning of embedding models."""

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
            "loss_type": str,
            "loss_fn": Callable,
            "num_epochs": int,
            "batch_size": int,
            "learning_rate": float,
            "warmup_ratio": float,
            "batch_sampler": str,
            "eval_data_path": str,
            "text_column": str,
            "label_column": str,
            "seed": int,
        }


def embedding_sft(
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
) -> Any:
    """Fine-tune an embedding model using contrastive losses.

    Trains sentence embedding models so that inputs with the same label
    cluster together in embedding space. Supports triplet and ranking
    losses with configurable batch sampling strategies.

    Args:
        model_path: HuggingFace model ID or local path to a sentence-transformers model.
        data_path: Path to JSONL file or HF dataset ID. Each sample needs
            a text field and an integer label field.
        ckpt_output_dir: Directory to save the fine-tuned model.
        backend: Training backend. Default "sentence-transformers".
        loss_type: Loss function name. One of "batch_all_triplet" (default),
            "batch_hard_triplet", or "mnrl".
        loss_fn: Custom loss function. Overrides loss_type if provided. Must be
            callable (e.g. a sentence-transformers loss instance). When set, the
            batch sampler defaults to "default" unless overridden.
        num_epochs: Number of training epochs. Default 20.
        batch_size: Per-device batch size. Default 32.
        learning_rate: Learning rate. Default 2e-5.
        warmup_ratio: Warmup fraction of total steps. Default 0.1.
        batch_sampler: Batch sampling strategy. None (default) auto-selects:
            "group_by_label" for triplet losses, "no_duplicates" for MNRL, and
            "default" for a custom loss_fn. Can also be set explicitly to
            "group_by_label", "no_duplicates", or "default".
        eval_data_path: Optional path to evaluation data (same format as
            data_path; converted to pairs automatically for MNRL).
        text_column: Name of the text column in the dataset. Default "text".
        label_column: Name of the label column in the dataset. Default "label".
        seed: Random seed. Default 42.
        **kwargs: Forwarded to the backend. Unknown parameters are logged as a
            warning (to catch typos) and otherwise ignored.

    Returns:
        Dict with status, model_path, num_samples, num_epochs, and loss (the
        effective loss label — the loss_type name, or "custom (<ClassName>)"
        when loss_fn is provided).

    Example:
        result = embedding_sft(
            model_path="sentence-transformers/all-MiniLM-L6-v2",
            data_path="routing_train.jsonl",
            ckpt_output_dir="./routing_model",
            loss_type="batch_all_triplet",
            num_epochs=20,
            batch_size=32,
        )
    """
    from . import create_algorithm

    algorithm = create_algorithm("embedding_sft", backend)
    return algorithm.train(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        loss_type=loss_type,
        loss_fn=loss_fn,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        batch_sampler=batch_sampler,
        eval_data_path=eval_data_path,
        text_column=text_column,
        label_column=label_column,
        seed=seed,
        **kwargs,
    )


AlgorithmRegistry.register_algorithm("embedding_sft", EmbeddingSFTAlgorithm)
AlgorithmRegistry.register_backend("embedding_sft", "sentence-transformers", SentenceTransformersBackend)
