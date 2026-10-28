"""Tests for the classifier_sft algorithm (decision/classification-head fine-tuning).

Covers the param spec, label encoding, the dotted-path resolver, and CLI wiring. The
training loop + per-task data prep are exercised end-to-end elsewhere (they need a real
model download and a tokenizer)."""
from __future__ import annotations

import pytest

from training_hub import ClassifierSFTAlgorithm
from training_hub.algorithms.classifier_sft import (
    DecisionHeadBackend,
    _encode_labels,
    _resolve_dotted,
)


class TestParams:
    def test_required_params(self):
        params = ClassifierSFTAlgorithm(DecisionHeadBackend()).get_required_params()
        assert set(params) == {"model_path", "data_path", "ckpt_output_dir"}

    def test_optional_params_cover_tasks_and_extension_hooks(self):
        params = ClassifierSFTAlgorithm(DecisionHeadBackend()).get_optional_params()
        for key in (
            "task",
            "options_column",
            "tokens_column",
            "use_peft",
            "trust_remote_code",
            "revision",
            "model_class",
            "loss_fn",
        ):
            assert key in params


class TestEncodeLabels:
    def test_string_labels_to_contiguous_sorted_ints(self):
        id2label, label2id, already_int = _encode_labels(["billing", "technical", "billing"])
        assert already_int is False
        assert id2label == {0: "billing", 1: "technical"}
        assert label2id == {"billing": 0, "technical": 1}

    def test_integer_labels_span_full_range(self):
        id2label, _, already_int = _encode_labels([0, 2])
        assert already_int is True
        assert id2label == {0: "0", 1: "1", 2: "2"}  # 3-wide head even with a gap

    def test_bool_labels_treated_as_categorical(self):
        id2label, _, already_int = _encode_labels([True, False, True])
        assert already_int is False
        assert set(id2label.values()) == {"True", "False"}

    def test_empty_labels_raise(self):
        with pytest.raises(ValueError, match="no non-null labels"):
            _encode_labels([None, None])

    def test_null_labels_raise_clear_error(self):
        # A null label in otherwise-labeled data previously crashed the string path
        # with an opaque KeyError('None'); the int path failed later at
        # tensorification. Both must fail loudly at label-encoding time instead.
        with pytest.raises(ValueError, match=r"2 null label\(s\) of 3"):
            _encode_labels(["billing", None, "support"])

        with pytest.raises(ValueError, match="null label"):
            _encode_labels([0, 1, None])


class TestResolveDotted:
    def test_resolves_attribute(self):
        assert _resolve_dotted("json.dumps").__name__ == "dumps"

    def test_bare_name_raises(self):
        with pytest.raises(ValueError, match="dotted import path"):
            _resolve_dotted("dumps")


class TestCliRegistration:
    def test_classifier_sft_subcommand_registered_and_builds(self):
        from training_hub import cli

        assert cli._SUBCOMMAND_TO_FUNC["classifier-sft"] == "classifier_sft"
        assert cli._FUNC_IMPORTS["classifier_sft"][0] == "training_hub.algorithms.classifier_sft"
        # _build_parser() triggers _define_params(), which populates _ALGO_PARAM_DEFS.
        assert cli._build_parser() is not None
        assert "--task" in cli._ALGO_PARAM_DEFS["classifier-sft"]
