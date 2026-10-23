"""Tests for the classifier_sft algorithm (sequence-classification head fine-tuning).

Covers the param spec and the (text, label) loader with label encoding — the training
loop itself is exercised end-to-end elsewhere (it needs a real model download)."""
from __future__ import annotations

import pytest

from training_hub import ClassifierSFTAlgorithm, SequenceClassificationBackend
from training_hub.algorithms.classifier_sft import _load_classification_dataset


class TestParams:
    def test_required_params(self):
        params = ClassifierSFTAlgorithm(SequenceClassificationBackend()).get_required_params()
        assert set(params) == {"model_path", "data_path", "ckpt_output_dir"}

    def test_optional_params_include_peft_and_trust_remote_code(self):
        params = ClassifierSFTAlgorithm(SequenceClassificationBackend()).get_optional_params()
        for key in ("use_peft", "lora_r", "trust_remote_code", "num_epochs", "label_column"):
            assert key in params


class TestLoadClassificationDataset:
    def _write(self, tmp_path, rows):
        import json

        p = tmp_path / "data.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return str(p)

    def test_string_categories_encoded_to_contiguous_ints(self, tmp_path):
        path = self._write(tmp_path, [
            {"text": "a", "label": "billing"},
            {"text": "b", "label": "technical"},
            {"text": "c", "label": "billing"},
        ])
        ds, id2label = _load_classification_dataset(path, "text", "label")
        assert sorted(set(ds["label"])) == [0, 1]
        # Sorted-unique mapping: billing -> 0, technical -> 1
        assert id2label == {0: "billing", 1: "technical"}
        assert all(isinstance(v, int) for v in ds["label"])

    def test_text_and_label_columns_are_renamed(self, tmp_path):
        path = self._write(tmp_path, [
            {"sentence": "hi", "category": "greeting"},
            {"sentence": "bug", "category": "issue"},
        ])
        ds, id2label = _load_classification_dataset(path, "sentence", "category")
        assert "text" in ds.column_names and "label" in ds.column_names
        assert set(id2label.values()) == {"greeting", "issue"}

    def test_integer_labels_preserved_with_full_range(self, tmp_path):
        # Already-int labels: head width spans the observed range even if a class is absent.
        path = self._write(tmp_path, [
            {"text": "a", "label": 0},
            {"text": "b", "label": 2},
        ])
        ds, id2label = _load_classification_dataset(path, "text", "label")
        assert sorted(set(ds["label"])) == [0, 2]
        assert id2label == {0: "0", 1: "1", 2: "2"}  # 3-wide head (0..2)

    def test_missing_label_column_raises(self, tmp_path):
        path = self._write(tmp_path, [{"text": "a"}])
        with pytest.raises(ValueError, match="label"):
            _load_classification_dataset(path, "text", "label")


class TestCliRegistration:
    def test_classifier_sft_subcommand_registered(self):
        from training_hub import cli

        assert cli._SUBCOMMAND_TO_FUNC["classifier-sft"] == "classifier_sft"
        assert cli._FUNC_IMPORTS["classifier_sft"][0] == "training_hub.algorithms.classifier_sft"
        # Parser builds without error and exposes the subcommand.
        parser = cli._build_parser()
        assert parser is not None
