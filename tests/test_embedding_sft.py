"""Tests for the embedding_sft algorithm.

Covers the input-validation and data-helper logic that does not require a GPU or
a trained model: parameter schemas, loss_fn validation, dataset loading,
MNRL pair conversion, and the unknown-kwarg/unknown-extension guards.

The full training path is exercised by the routing_demo notebook; these tests
guard the lightweight logic that runs *before* the heavy sentence-transformers
imports kick in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from training_hub import EmbeddingSFTAlgorithm, SentenceTransformersBackend
from training_hub.algorithms.embedding_sft import (
    LOSS_REGISTRY,
    _KNOWN_PARAMS,
    _load_dataset,
    _to_pair_dataset,
    embedding_sft,
)


# ---------------------------------------------------------------------------
# Param schemas
# ---------------------------------------------------------------------------


class TestParamSchemas:
    def test_required_params(self):
        params = EmbeddingSFTAlgorithm(SentenceTransformersBackend()).get_required_params()
        assert set(params) == {"model_path", "data_path", "ckpt_output_dir"}
        for p in params.values():
            assert p is str

    def test_optional_params(self):
        import typing

        params = EmbeddingSFTAlgorithm(SentenceTransformersBackend()).get_optional_params()
        assert "loss_type" in params and params["loss_type"] is str
        # loss_fn must be typed as Callable, not the opaque `object` that lets
        # non-callables through silently.
        assert params["loss_fn"] is typing.Callable

    def test_known_params_in_sync_with_schemas(self):
        algo = EmbeddingSFTAlgorithm(SentenceTransformersBackend())
        schema = set(algo.get_required_params()) | set(algo.get_optional_params())
        assert schema == _KNOWN_PARAMS

    def test_loss_registry_covers_documented_losses(self):
        assert set(LOSS_REGISTRY) == {"batch_all_triplet", "batch_hard_triplet", "mnrl"}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


class TestLoadDataset:
    def test_loads_jsonl(self, tmp_path):
        path = _write_jsonl(tmp_path / "train.jsonl", [
            {"text": "a", "label": 0}, {"text": "b", "label": 1},
        ])
        ds = _load_dataset(path)
        assert "text" in ds.column_names and "label" in ds.column_names
        assert len(ds) == 2

    def test_loads_csv(self, tmp_path):
        path = tmp_path / "train.csv"
        path.write_text("text,label\na,0\nb,1\n")
        ds = _load_dataset(str(path))
        assert len(ds) == 2 and "label" in ds.column_names

    def test_column_rename(self, tmp_path):
        path = _write_jsonl(tmp_path / "train.jsonl", [
            {"sentence": "a", "category": 0}, {"sentence": "b", "category": 1},
        ])
        ds = _load_dataset(path, text_column="sentence", label_column="category")
        assert ds.column_names == ["text", "label"]

    def test_unknown_extension_raises(self, tmp_path):
        path = tmp_path / "data.parquet"
        path.write_text("not actually parquet")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            _load_dataset(str(path))

    def test_missing_text_column_raises(self, tmp_path):
        path = _write_jsonl(tmp_path / "train.jsonl", [{"foo": "a", "label": 0}])
        with pytest.raises(ValueError, match="text"):
            _load_dataset(path)

    def test_missing_label_column_raises(self, tmp_path):
        path = _write_jsonl(tmp_path / "train.jsonl", [{"text": "a", "foo": 0}])
        with pytest.raises(ValueError, match="label"):
            _load_dataset(path)

    def test_require_label_false_allows_text_only(self, tmp_path):
        path = _write_jsonl(tmp_path / "train.jsonl", [{"text": "a"}, {"text": "b"}])
        ds = _load_dataset(path, require_label=False)
        assert "text" in ds.column_names and "label" not in ds.column_names

    def test_custom_loss_schema_not_forced_into_text_label(self, tmp_path):
        """A custom loss (e.g. ContrastiveLoss) may use sentence1/sentence2 + label
        instead of text/label. With requirements relaxed, loading must not fail."""
        path = _write_jsonl(tmp_path / "train.jsonl", [
            {"sentence1": "a", "sentence2": "b", "label": 1},
            {"sentence1": "c", "sentence2": "d", "label": 0},
        ])
        ds = _load_dataset(path, require_text=False, require_label=False)
        assert "sentence1" in ds.column_names and "text" not in ds.column_names


# ---------------------------------------------------------------------------
# MNRL pair conversion
# ---------------------------------------------------------------------------


class TestToPairDataset:
    def test_pairs_cover_within_label_combos(self):
        from datasets import Dataset
        ds = Dataset.from_list([
            {"text": "a", "label": 0}, {"text": "b", "label": 0},
            {"text": "c", "label": 1}, {"text": "d", "label": 1},
        ])
        pairs = _to_pair_dataset(ds, max_pairs_per_label=1000)
        # 2 texts per label -> 1 pair each -> 2 total
        assert len(pairs) == 2
        assert set(pairs.column_names) == {"anchor", "positive"}

    def test_caps_pairs_without_materializing_all(self):
        """A large label must not blow up to O(n^2) pairs."""
        from datasets import Dataset
        big = Dataset.from_list([{"text": f"t{i}", "label": 0} for i in range(5000)])
        pairs = _to_pair_dataset(big, max_pairs_per_label=1000, seed=1)
        assert len(pairs) <= 1000
        assert len(pairs) > 0

    def test_respects_max_pairs_per_label(self):
        from datasets import Dataset
        # 20 texts -> 190 pairs; cap at 50
        ds = Dataset.from_list([{"text": f"t{i}", "label": 0} for i in range(20)])
        pairs = _to_pair_dataset(ds, max_pairs_per_label=50, seed=1)
        assert len(pairs) <= 50

    def test_is_deterministic_for_same_seed(self):
        from datasets import Dataset
        ds = Dataset.from_list([{"text": f"t{i}", "label": 0} for i in range(20)])
        a = _to_pair_dataset(ds, max_pairs_per_label=50, seed=7)
        b = _to_pair_dataset(ds, max_pairs_per_label=50, seed=7)
        assert [r["anchor"] for r in a] == [r["anchor"] for r in b]


# ---------------------------------------------------------------------------
# Loss-fn validation (runs without a GPU; fails before the model loads)
# ---------------------------------------------------------------------------


class TestLossFnValidation:
    def test_non_callable_loss_fn_raises_type_error(self, tmp_path):
        path = _write_jsonl(tmp_path / "train.jsonl", [{"text": "a", "label": 0}])
        with pytest.raises(TypeError, match="loss_fn must be callable"):
            embedding_sft(
                model_path="sentence-transformers/all-MiniLM-L6-v2",
                data_path=path,
                ckpt_output_dir=str(tmp_path / "out"),
                loss_fn=12345,
            )

    def test_non_callable_loss_fn_raises_before_model_load(self, tmp_path):
        """A non-existent model_path must NOT be the error surfaced when loss_fn
        is invalid — validation should happen first."""
        path = _write_jsonl(tmp_path / "train.jsonl", [{"text": "a", "label": 0}])
        with pytest.raises(TypeError, match="loss_fn"):
            embedding_sft(
                model_path="this/does/not/exist",
                data_path=path,
                ckpt_output_dir=str(tmp_path / "out"),
                loss_fn="not a loss",
            )
