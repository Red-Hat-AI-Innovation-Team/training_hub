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


# ---------------------------------------------------------------------------
# Column-rename collision guard
# ---------------------------------------------------------------------------


class TestColumnRenameCollision:
    def test_text_column_collision_raises(self, tmp_path):
        """A dataset with both 'text' and 'sentence' columns can't rename
        'sentence'->'text'; surface a clear error rather than an opaque one."""
        path = _write_jsonl(tmp_path / "train.jsonl", [
            {"text": "a", "sentence": "b", "label": 0},
        ])
        with pytest.raises(ValueError, match="Cannot rename column 'sentence' to 'text'"):
            _load_dataset(path, text_column="sentence")

    def test_label_column_collision_raises(self, tmp_path):
        path = _write_jsonl(tmp_path / "train.jsonl", [
            {"text": "a", "label": 0, "category": 1},
        ])
        with pytest.raises(ValueError, match="Cannot rename column 'category' to 'label'"):
            _load_dataset(path, label_column="category")


# ---------------------------------------------------------------------------
# Mocked integration: exercise execute_training validation/branching without a
# GPU or model download. A fake sentence_transformers package is installed into
# sys.modules so the lazy imports inside execute_training resolve to stubs.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_st(monkeypatch):
    """Install a stub sentence_transformers package in sys.modules.

    The nested module layout (sentence_transformers.sentence_transformer.*) is
    populated so the preferred import branch resolves; this mirrors a real
    5.4+/6.x install.
    """
    import sys
    import types

    created_trainers: list = []

    class FakeModel:
        def __init__(self, path):
            self.path = path

        def save_pretrained(self, d):
            os.makedirs(d, exist_ok=True)

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, eval_dataset, loss):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.loss = loss
            self.state = types.SimpleNamespace(
                log_history=[{"train_loss": 0.42, "global_step": 3}],
                max_steps=3,
                global_step=3,
            )

        def train(self):
            created_trainers.append(self)

    class BatchSamplers:
        GROUP_BY_LABEL = "group_by_label"
        NO_DUPLICATES = "no_duplicates"
        BATCH_SAMPLER = "default"

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeLoss:
        def __init__(self, model):
            self.model = model

    # Build the fake package + nested submodules.
    top = types.ModuleType("sentence_transformers")
    top.SentenceTransformer = FakeModel
    top.SentenceTransformerTrainer = FakeTrainer

    inner = types.ModuleType("sentence_transformers.sentence_transformer")
    training_args_mod = types.ModuleType(
        "sentence_transformers.sentence_transformer.training_args"
    )
    training_args_mod.SentenceTransformerTrainingArguments = FakeTrainingArguments
    training_args_mod.BatchSamplers = BatchSamplers
    losses_mod = types.ModuleType(
        "sentence_transformers.sentence_transformer.losses"
    )
    losses_mod.BatchAllTripletLoss = FakeLoss
    losses_mod.BatchHardTripletLoss = FakeLoss
    losses_mod.MultipleNegativesRankingLoss = FakeLoss
    inner.training_args = training_args_mod
    inner.losses = losses_mod
    top.sentence_transformer = inner

    # Note: we deliberately do NOT stub torch. The real torch is installed in
    # the test env and torch.cuda.is_available() returns False on CPU, which is
    # exactly the branch we want. Stubbing torch breaks datasets' dill pickling
    # (which inspects torch.Tensor).

    monkeypatch.setitem(sys.modules, "sentence_transformers", top)
    monkeypatch.setitem(sys.modules, "sentence_transformers.sentence_transformer", inner)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers.sentence_transformer.training_args",
        training_args_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers.sentence_transformer.losses",
        losses_mod,
    )

    return types.SimpleNamespace(trainers=created_trainers, BatchSamplers=BatchSamplers)


def _train_rows():
    return [
        {"text": "a", "label": 0}, {"text": "b", "label": 0},
        {"text": "c", "label": 1}, {"text": "d", "label": 1},
    ]


class TestExecuteTrainingValidation:
    def test_unknown_loss_type_raises(self, tmp_path, fake_st):
        path = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        with pytest.raises(ValueError, match="Unknown loss_type 'bogus'"):
            embedding_sft(
                model_path="fake/model",
                data_path=path,
                ckpt_output_dir=str(tmp_path / "out"),
                loss_type="bogus",
            )

    def test_unknown_batch_sampler_raises(self, tmp_path, fake_st):
        path = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        with pytest.raises(ValueError, match="Unknown batch_sampler 'bogus'"):
            embedding_sft(
                model_path="fake/model",
                data_path=path,
                ckpt_output_dir=str(tmp_path / "out"),
                batch_sampler="bogus",
            )

    def test_unknown_kwarg_warns(self, tmp_path, fake_st, caplog):
        path = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        with caplog.at_level("WARNING", logger="training_hub.algorithms.embedding_sft"):
            embedding_sft(
                model_path="fake/model",
                data_path=path,
                ckpt_output_dir=str(tmp_path / "out"),
                # typo: 'learing_rate' instead of 'learning_rate'
                learing_rate=1e-3,
            )
        joined = " ".join(r.message for r in caplog.records)
        assert "learing_rate" in joined and "unrecognized" in joined.lower()


class TestBatchSamplerAutoSelection:
    def test_triplet_defaults_to_group_by_label(self, tmp_path, fake_st):
        path = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        embedding_sft(
            model_path="fake/model",
            data_path=path,
            ckpt_output_dir=str(tmp_path / "out"),
            loss_type="batch_hard_triplet",
            num_epochs=1,
        )
        assert fake_st.trainers[-1].args.batch_sampler == fake_st.BatchSamplers.GROUP_BY_LABEL

    def test_mnrl_defaults_to_no_duplicates(self, tmp_path, fake_st):
        path = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        embedding_sft(
            model_path="fake/model",
            data_path=path,
            ckpt_output_dir=str(tmp_path / "out"),
            loss_type="mnrl",
            num_epochs=1,
        )
        assert fake_st.trainers[-1].args.batch_sampler == fake_st.BatchSamplers.NO_DUPLICATES

    def test_custom_loss_defaults_to_default_sampler(self, tmp_path, fake_st):
        path = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        embedding_sft(
            model_path="fake/model",
            data_path=path,
            ckpt_output_dir=str(tmp_path / "out"),
            loss_fn=lambda model: object(),  # callable custom loss
            num_epochs=1,
        )
        assert fake_st.trainers[-1].args.batch_sampler == fake_st.BatchSamplers.BATCH_SAMPLER

    def test_explicit_sampler_overrides_auto_selection(self, tmp_path, fake_st):
        path = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        embedding_sft(
            model_path="fake/model",
            data_path=path,
            ckpt_output_dir=str(tmp_path / "out"),
            loss_type="mnrl",
            batch_sampler="group_by_label",
            num_epochs=1,
        )
        assert fake_st.trainers[-1].args.batch_sampler == fake_st.BatchSamplers.GROUP_BY_LABEL


class TestEvalDatasetLoading:
    def test_eval_dataset_loaded_for_triplet(self, tmp_path, fake_st):
        train = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        eval_rows = [
            {"text": "e", "label": 0}, {"text": "f", "label": 1},
        ]
        eval_path = _write_jsonl(tmp_path / "eval.jsonl", eval_rows)
        embedding_sft(
            model_path="fake/model",
            data_path=train,
            ckpt_output_dir=str(tmp_path / "out"),
            eval_data_path=eval_path,
            num_epochs=1,
        )
        trainer = fake_st.trainers[-1]
        assert trainer.eval_dataset is not None
        assert len(trainer.eval_dataset) == 2
        assert trainer.args.eval_strategy == "epoch"

    def test_no_eval_dataset_defaults_to_no_eval(self, tmp_path, fake_st):
        path = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        embedding_sft(
            model_path="fake/model",
            data_path=path,
            ckpt_output_dir=str(tmp_path / "out"),
            num_epochs=1,
        )
        trainer = fake_st.trainers[-1]
        assert trainer.eval_dataset is None
        assert trainer.args.eval_strategy == "no"

    def test_eval_dataset_converted_to_pairs_for_mnrl(self, tmp_path, fake_st):
        train = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        eval_rows = [
            {"text": "e", "label": 0}, {"text": "f", "label": 0},
            {"text": "g", "label": 1}, {"text": "h", "label": 1},
        ]
        eval_path = _write_jsonl(tmp_path / "eval.jsonl", eval_rows)
        embedding_sft(
            model_path="fake/model",
            data_path=train,
            ckpt_output_dir=str(tmp_path / "out"),
            loss_type="mnrl",
            eval_data_path=eval_path,
            num_epochs=1,
        )
        trainer = fake_st.trainers[-1]
        # MNRL eval data is converted to (anchor, positive) pairs.
        assert trainer.eval_dataset is not None
        assert set(trainer.eval_dataset.column_names) == {"anchor", "positive"}
        assert trainer.train_dataset is not None
        assert set(trainer.train_dataset.column_names) == {"anchor", "positive"}


class TestMockedEndToEnd:
    def test_full_training_run_returns_success(self, tmp_path, fake_st):
        path = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        result = embedding_sft(
            model_path="fake/model",
            data_path=path,
            ckpt_output_dir=str(tmp_path / "out"),
            loss_type="batch_all_triplet",
            num_epochs=2,
            batch_size=4,
        )
        assert result["status"] == "success"
        assert result["model_path"] == str(tmp_path / "out")
        assert result["num_epochs"] == 2
        assert result["loss"] == "batch_all_triplet"
        assert result["num_samples"] == 4
        # The trainer was actually invoked (not just constructed).
        assert len(fake_st.trainers) == 1

    def test_metrics_file_written(self, tmp_path, fake_st):
        path = _write_jsonl(tmp_path / "train.jsonl", _train_rows())
        out_dir = tmp_path / "out"
        embedding_sft(
            model_path="fake/model",
            data_path=path,
            ckpt_output_dir=str(out_dir),
            num_epochs=1,
        )
        metrics = (out_dir / "training_metrics.jsonl").read_text()
        assert json.loads(metrics.splitlines()[0])["loss"] == "batch_all_triplet"
