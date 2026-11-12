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
    _accepts_choice_mask,
    _align_word_labels,
    _apply_lora,
    _call_loss_fn,
    _encode_labels,
    _model_load_dtype,
    _normalize_loss_over_accumulation,
    _resolve_dotted,
    _set_pad_token_id,
    _training_metrics_entry,
    _upcast_trainable_params,
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
            out = {}
            for key in features[0]:
                fill = self.pad_token_id if key == "input_ids" else 0
                out[key] = torch.tensor([f[key] + [fill] * (maxlen - len(f[key])) for f in features])
            return out

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

    def test_token_type_ids_are_carried_and_padded(self):
        collator = _MultipleChoiceCollator(self._StubTokenizer())
        features = [
            {"input_ids": [[1, 2], [3, 4]], "attention_mask": [[1, 1]] * 2, "token_type_ids": [[0, 1]] * 2, "label": 0},
            {"input_ids": [[5, 6]], "attention_mask": [[1, 1]], "token_type_ids": [[0, 1]], "label": 0},
        ]
        batch = collator(features)
        assert batch["token_type_ids"].shape == (2, 2, 2)
        assert batch["token_type_ids"][0].tolist() == [[0, 1], [0, 1]]
        assert batch["token_type_ids"][1, 1].tolist() == [0, 0]  # placeholder choice: segment 0

    def test_padded_choices_have_an_attended_token(self):
        # An all-zero attention mask on a padded choice makes mean-pooling heads (ModernBERT's
        # default) divide by zero -> NaN that backprop writes into every weight. Every choice,
        # padded or not, must attend to at least one token; choice_mask handles exclusion.
        collator = _MultipleChoiceCollator(self._StubTokenizer())
        features = [
            {"input_ids": [[1, 2], [3, 4], [5, 6], [7, 8]], "attention_mask": [[1, 1]] * 4, "label": 0},
            {"input_ids": [[9, 9]], "attention_mask": [[1, 1]], "label": 0},
        ]
        batch = collator(features)
        assert (batch["attention_mask"].sum(dim=-1) > 0).all()


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
        assert cli._ALGO_PARAM_DEFS["classifier-sft"]["--lora-target-modules"]["nargs"] == "+"


class _RecordingTokenizer:
    """Offline stand-in for a HF tokenizer: records which strings it was asked to encode."""

    pad_token_id = 0

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts, truncation=True, max_length=None):
        self.seen.extend(texts)
        return {"input_ids": [[1] * len(t) for t in texts], "attention_mask": [[1] * len(t) for t in texts]}


class TestSequenceClassificationColumns:
    def _prep(self, tmp_path, rows, **params):
        import json

        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        tok = _RecordingTokenizer()
        out = DecisionHeadBackend()._prep_sequence_classification(str(path), tok, 32, params)
        return out, tok

    def test_missing_label_column_names_it_instead_of_falling_back(self, tmp_path):
        rows = [{"text": "a", "label": "x"}, {"text": "b", "label": "y"}]
        with pytest.raises(ValueError, match=r"label_column='nope'"):
            self._prep(tmp_path, rows, label_column="nope")

    def test_custom_text_column_alongside_existing_text_column(self, tmp_path):
        rows = [{"text": "raw", "clean_text": "CLEAN-A", "category": "x"},
                {"text": "raw", "clean_text": "CLEAN-B", "category": "y"}]
        (tokenized, _c, id2label, _kw), tok = self._prep(
            tmp_path, rows, text_column="clean_text", label_column="category"
        )
        assert tok.seen == ["CLEAN-A", "CLEAN-B"]  # encoded the requested column, not `text`
        assert set(tokenized.column_names) == {"input_ids", "attention_mask", "label"}
        assert id2label == {0: "x", 1: "y"}

    def test_swapped_column_names(self, tmp_path):
        # text lives in a column called "label" and labels in one called "text" — no renames,
        # so no collision.
        rows = [{"label": "hello there", "text": "pos"}, {"label": "go away", "text": "neg"}]
        (tokenized, _c, id2label, _kw), tok = self._prep(
            tmp_path, rows, text_column="label", label_column="text"
        )
        assert tok.seen == ["hello there", "go away"]
        assert sorted(id2label.values()) == ["neg", "pos"]

    def test_same_text_and_label_column_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must differ"):
            self._prep(tmp_path, [{"text": "a"}], text_column="text", label_column="text")


class _FakeModel:
    class config:  # noqa: N801
        model_type = "modernbert"


class TestApplyLora:
    def test_falls_back_to_all_linear_when_peft_has_no_default_targets(self, monkeypatch):
        import peft

        calls = []

        def fake_get_peft_model(model, cfg):
            calls.append(cfg.target_modules)
            if cfg.target_modules is None:
                raise ValueError("Please specify `target_modules` or `target_parameters`in `peft_config`")
            return "wrapped"

        monkeypatch.setattr(peft, "get_peft_model", fake_get_peft_model)
        assert _apply_lora(_FakeModel(), "sequence_classification", {}) == "wrapped"
        assert calls[0] is None and calls[1] == "all-linear"

    def test_explicit_targets_are_not_overridden(self, monkeypatch):
        import peft

        def boom(model, cfg):
            raise ValueError("Target modules {'nope'} not found — target_modules mismatch")

        monkeypatch.setattr(peft, "get_peft_model", boom)
        with pytest.raises(ValueError, match="nope"):
            _apply_lora(_FakeModel(), "sequence_classification", {"lora_target_modules": ["nope"]})


class TestLossNormalization:
    def test_trainer_divides_by_accumulation_steps(self):
        class T:
            model_accepts_loss_kwargs = True

        t = T()
        _normalize_loss_over_accumulation(t)
        assert t.model_accepts_loss_kwargs is False

    def test_trainer_subclass_can_opt_out(self):
        class OptOut:
            model_accepts_loss_kwargs = True  # consumes num_items_in_batch itself

        t = OptOut()
        _normalize_loss_over_accumulation(t, OptOut)
        assert t.model_accepts_loss_kwargs is True


class TestTrainingMetricsEntry:
    class _State:
        def __init__(self, log_history, epoch=3.0):
            self.log_history = log_history
            self.max_steps = 30
            self.global_step = 30
            self.epoch = epoch

    def test_walks_back_to_latest_train_loss(self):
        state = self._State([{"loss": 0.9}, {"train_loss": 0.42}, {"eval_loss": 0.5}])
        entry = _training_metrics_entry(state, "sequence_classification")
        assert entry == {"max_steps": 30, "global_step": 30, "train_loss": 0.42, "epoch": 3.0,
                         "task": "sequence_classification"}

    def test_non_finite_values_serialize_as_valid_json(self):
        import json
        import math

        entry = _training_metrics_entry(self._State([{"train_loss": math.nan}], epoch=math.inf), "x")
        assert entry["train_loss"] is None and entry["epoch"] is None
        json.loads(json.dumps(entry, allow_nan=False))  # strict JSON round-trips



class TestNegativeAndOutOfRangeLabels:
    def test_negative_integer_labels_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            _encode_labels([0, 1, -1])

    def test_multiple_choice_label_must_index_an_option(self, tmp_path):
        import json

        class _PairTokenizer:
            pad_token_id = 0

            def __call__(self, contexts, options, truncation=True, max_length=None):
                n = len(options)
                return {"input_ids": [[1, 2]] * n, "attention_mask": [[1, 1]] * n, "token_type_ids": [[0, 1]] * n}

        path = tmp_path / "mc.jsonl"
        path.write_text(json.dumps({"text": "q", "options": ["a", "b"], "label": 2}))
        with pytest.raises(ValueError, match="not a valid option index"):
            DecisionHeadBackend()._prep_multiple_choice(str(path), _PairTokenizer(), 16, {})

    def test_multiple_choice_keeps_token_type_ids(self, tmp_path):
        import json

        class _PairTokenizer:
            pad_token_id = 0

            def __call__(self, contexts, options, truncation=True, max_length=None):
                n = len(options)
                return {"input_ids": [[1, 2]] * n, "attention_mask": [[1, 1]] * n, "token_type_ids": [[0, 1]] * n}

        path = tmp_path / "mc.jsonl"
        path.write_text(json.dumps({"text": "q", "options": ["a", "b"], "label": 1}))
        tokenized, *_ = DecisionHeadBackend()._prep_multiple_choice(str(path), _PairTokenizer(), 16, {})
        assert "token_type_ids" in tokenized.column_names


class TestCustomLossMasking:
    def test_padded_choices_masked_before_two_arg_loss(self):
        import torch

        seen = {}

        def loss(logits, labels):
            seen["logits"] = logits
            return logits.sum() * 0

        logits = torch.zeros(1, 3)
        mask = torch.tensor([[True, True, False]])
        _call_loss_fn(loss, logits, torch.tensor([0]), mask, pass_mask=_accepts_choice_mask(loss))
        assert seen["logits"][0, 2] == float("-inf") and torch.isfinite(seen["logits"][0, :2]).all()

    def test_choice_mask_passed_only_to_losses_that_accept_it(self):
        def two_arg(logits, labels):
            return 0

        def with_mask(logits, labels, choice_mask=None):
            return 0

        def with_kwargs(logits, labels, **kw):
            return 0

        assert not _accepts_choice_mask(two_arg)
        assert _accepts_choice_mask(with_mask) and _accepts_choice_mask(with_kwargs)


class TestSetPadTokenId:
    def test_sets_nested_text_config(self):
        # Qwen3.5-style: the head reads config.text_config.pad_token_id
        class Text:
            pad_token_id = None

        class Top:
            pad_token_id = None
            text_config = Text()

            def get_text_config(self):
                return self.text_config

        cfg = Top()
        _set_pad_token_id(cfg, 7)
        assert cfg.pad_token_id == 7 and cfg.text_config.pad_token_id == 7

    def test_does_not_override_existing(self):
        class Flat:
            pad_token_id = 3

            def get_text_config(self):
                return self

        cfg = Flat()
        _set_pad_token_id(cfg, 7)
        assert cfg.pad_token_id == 3


class TestApplyLoraRealPeft:
    """Real PEFT (no mock) on a tiny ModernBERT built from config — PEFT has no default LoRA
    targets for ModernBERT, so this exercises the all-linear fallback for each head type."""

    @staticmethod
    def _tiny(cls_name):
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "ModernBertConfig"):
            pytest.skip("this transformers has no ModernBERT")
        cfg = transformers.ModernBertConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, max_position_embeddings=64, num_labels=3,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
        )
        return getattr(transformers, cls_name)(cfg)

    @pytest.mark.parametrize(
        "cls_name,task",
        [
            ("ModernBertForSequenceClassification", "sequence_classification"),
            ("ModernBertForTokenClassification", "token_classification"),
        ],
    )
    def test_all_linear_fallback_keeps_head_trainable(self, cls_name, task):
        pytest.importorskip("peft")
        model = _apply_lora(self._tiny(cls_name), task, {})
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert any("lora_" in n for n in trainable)  # LoRA applied to the backbone
        assert any("classifier" in n and "lora_" not in n for n in trainable)  # head fully trainable



class TestTrainableDtype:
    """transformers 5 loads checkpoints in their stored dtype; AdamW on fp16 weights NaNs
    (DeBERTa-v3 ships fp16). Optimized weights must be fp32."""

    def test_load_dtype_policy(self):
        import torch

        assert _model_load_dtype(use_peft=False, use_bf16=True) is torch.float32  # full FT
        assert _model_load_dtype(use_peft=False, use_bf16=False) is torch.float32
        assert _model_load_dtype(use_peft=True, use_bf16=True) is torch.bfloat16  # frozen base
        assert _model_load_dtype(use_peft=True, use_bf16=False) is torch.float32

    def test_lora_trainables_upcast_while_frozen_base_stays_bf16(self):
        import torch

        pytest.importorskip("peft")
        model = TestApplyLoraRealPeft._tiny("ModernBertForSequenceClassification").to(torch.bfloat16)
        model = _apply_lora(model, "sequence_classification", {})
        _upcast_trainable_params(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        frozen = [p for p in model.parameters() if not p.requires_grad]
        assert trainable and all(p.dtype is torch.float32 for p in trainable)
        assert frozen and all(p.dtype is torch.bfloat16 for p in frozen)
