"""Tests for the classifier_sft algorithm (decision/classification-head fine-tuning).

Covers the param spec, label encoding, the dotted-path resolver, and CLI wiring. The
training loop + per-task data prep are exercised end-to-end elsewhere (they need a real
model download and a tokenizer)."""
from __future__ import annotations

import pytest

from training_hub import ClassifierSFTAlgorithm
from training_hub.algorithms.classifier_sft import (
    DecisionHeadBackend,
    _MultipleChoiceCollator,
    _align_word_labels,
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
            "lora_target_modules",  # accepted by classifier_sft() + read by the backend
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
        with pytest.raises(ValueError, match=r"1 null label\(s\) of 3"):
            _encode_labels(["billing", None, "support"])

        with pytest.raises(ValueError, match="null label"):
            _encode_labels([0, 1, None])

    def test_nan_float_labels_rejected(self):
        # A parquet/pandas float column carries missing values as NaN — a float, so the
        # `is not None` filter lets it through — which would otherwise become a "nan" class.
        import math

        with pytest.raises(ValueError, match="null label"):
            _encode_labels([1.0, math.nan, 2.0])
        with pytest.raises(ValueError, match="no non-null"):
            _encode_labels([float("nan"), float("nan")])


class TestAlignWordLabels:
    """Token-classification sub-word alignment (only the first sub-token of each word
    carries the label; special tokens and continuations are -100)."""

    def test_int_tags_with_subword_split(self):
        # word_ids: [CLS]=None, word0, word0 (sub-word), word1, [SEP]=None
        out = _align_word_labels([None, 0, 0, 1, None], tags=[3, 5], label2id={}, already_int=True)
        assert out == [-100, 3, -100, 5, -100]

    def test_string_tags_mapped_through_label2id(self):
        out = _align_word_labels(
            [None, 0, 0, 1, None], tags=["B", "I"], label2id={"B": 0, "I": 1}, already_int=False
        )
        assert out == [-100, 0, -100, 1, -100]


class TestMultipleChoiceCollator:
    """Variable option counts pad to the batch max with empty choices; a boolean
    choice_mask marks the padded slots so the loss can exclude them."""

    class _StubTokenizer:
        pad_token_id = 0

        def pad(self, features, return_tensors=None):
            import torch

            maxlen = max(len(f["input_ids"]) for f in features)
            ids = [f["input_ids"] + [self.pad_token_id] * (maxlen - len(f["input_ids"])) for f in features]
            masks = [f["attention_mask"] + [0] * (maxlen - len(f["attention_mask"])) for f in features]
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    def test_variable_k_padding_shapes_and_mask(self):
        collator = _MultipleChoiceCollator(self._StubTokenizer())
        features = [
            {"input_ids": [[1, 2], [3, 4], [5, 6]], "attention_mask": [[1, 1], [1, 1], [1, 1]], "label": 2},
            {"input_ids": [[7, 8]], "attention_mask": [[1, 1]], "label": 0},
        ]
        batch = collator(features)
        assert batch["input_ids"].shape == (2, 3, 2)  # bsz, max K, seq
        assert batch["labels"].tolist() == [2, 0]
        # the second example had 1 real choice of 3 → only its first slot is real
        assert batch["choice_mask"].tolist() == [[True, True, True], [True, False, False]]


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
