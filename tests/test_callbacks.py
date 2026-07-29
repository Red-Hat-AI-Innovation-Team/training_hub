"""Tests for the unified callback abstraction layer.

Covers:
- TrainingHubContext defaults and custom values
- TrainingHubCallback no-op defaults and selective override
- UnslothCallbackAdapter: context mapping, exception isolation, rank guard
- adapt_hub_callbacks utility
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from training_hub.callbacks import TrainingHubCallback, TrainingHubContext


# ---------------------------------------------------------------------------
# TrainingHubContext
# ---------------------------------------------------------------------------


class TestTrainingHubContext:
    """Tests for the TrainingHubContext dataclass."""

    def test_defaults(self):
        ctx = TrainingHubContext()
        assert ctx.step == 0
        assert ctx.epoch == 0
        assert ctx.loss is None
        assert ctx.learning_rate is None
        assert ctx.is_main_process is True
        assert ctx.output_dir == ""
        assert ctx.metrics == {}

    def test_custom_values(self):
        ctx = TrainingHubContext(
            step=42,
            epoch=3,
            loss=0.5,
            learning_rate=1e-4,
            is_main_process=False,
            output_dir="/tmp/ckpt",
            metrics={"grad_norm": 1.2},
        )
        assert ctx.step == 42
        assert ctx.epoch == 3
        assert ctx.loss == 0.5
        assert ctx.learning_rate == 1e-4
        assert ctx.is_main_process is False
        assert ctx.output_dir == "/tmp/ckpt"
        assert ctx.metrics == {"grad_norm": 1.2}

    def test_metrics_not_shared_across_instances(self):
        a = TrainingHubContext()
        b = TrainingHubContext()
        a.metrics["key"] = "val"
        assert "key" not in b.metrics


# ---------------------------------------------------------------------------
# TrainingHubCallback
# ---------------------------------------------------------------------------

ALL_HOOKS = [
    "on_train_begin",
    "on_epoch_begin",
    "on_step_begin",
    "on_log",
    "on_evaluate",
    "on_save",
    "on_step_end",
    "on_epoch_end",
    "on_train_end",
]


class TestTrainingHubCallback:
    """Tests for the TrainingHubCallback base class."""

    def test_all_hooks_are_noop(self):
        """Every hook should be callable and return None."""
        cb = TrainingHubCallback()
        ctx = TrainingHubContext()
        for hook_name in ALL_HOOKS:
            result = getattr(cb, hook_name)(ctx)
            assert result is None

    def test_selective_override(self):
        """Subclass can override a single hook; others stay no-op."""
        calls = []

        class StepCounter(TrainingHubCallback):
            def on_step_end(self, context):
                calls.append(context.step)

        cb = StepCounter()
        ctx = TrainingHubContext(step=10)

        cb.on_step_end(ctx)
        assert calls == [10]

        # Other hooks still no-op
        cb.on_train_begin(ctx)
        cb.on_epoch_end(ctx)
        assert calls == [10]

    def test_is_not_abstract(self):
        """TrainingHubCallback should be directly instantiable (not ABC)."""
        cb = TrainingHubCallback()
        assert cb is not None


# ---------------------------------------------------------------------------
# UnslothCallbackAdapter (requires transformers)
# ---------------------------------------------------------------------------

try:
    from training_hub.adapters.unsloth import (
        UnslothCallbackAdapter,
        adapt_hub_callbacks,
    )

    _HAS_TRANSFORMERS = True
except ImportError:
    UnslothCallbackAdapter = None  # type: ignore[misc, assignment]
    adapt_hub_callbacks = None  # type: ignore[misc, assignment]
    _HAS_TRANSFORMERS = False


class _RecordingCallback(TrainingHubCallback):
    """Records last context per hook for adapter tests."""

    def __init__(self) -> None:
        self.calls: dict[str, TrainingHubContext] = {}

    def on_train_begin(self, context):
        self.calls["on_train_begin"] = context

    def on_epoch_begin(self, context):
        self.calls["on_epoch_begin"] = context

    def on_step_begin(self, context):
        self.calls["on_step_begin"] = context

    def on_log(self, context):
        self.calls["on_log"] = context

    def on_evaluate(self, context):
        self.calls["on_evaluate"] = context

    def on_save(self, context):
        self.calls["on_save"] = context

    def on_step_end(self, context):
        self.calls["on_step_end"] = context

    def on_epoch_end(self, context):
        self.calls["on_epoch_end"] = context

    def on_train_end(self, context):
        self.calls["on_train_end"] = context


def _make_hf_state(
    global_step: int = 0,
    epoch: float = 0.0,
    is_world_process_zero: bool = True,
    log_history: list | None = None,
) -> SimpleNamespace:
    """Create a fake HuggingFace TrainerState."""
    return SimpleNamespace(
        global_step=global_step,
        epoch=epoch,
        is_world_process_zero=is_world_process_zero,
        log_history=log_history or [],
    )


def _make_hf_args(output_dir: str = "/tmp/output") -> SimpleNamespace:
    """Create a fake HuggingFace TrainingArguments."""
    return SimpleNamespace(output_dir=output_dir)


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
class TestUnslothCallbackAdapter:
    """Tests for the UnslothCallbackAdapter."""

    def test_rejects_non_callback(self):
        with pytest.raises(TypeError, match="TrainingHubCallback"):
            UnslothCallbackAdapter(object())  # type: ignore[arg-type]

    def test_context_mapping(self):
        """Adapter builds correct TrainingHubContext from HF state."""
        cb = _RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args = _make_hf_args("/checkpoints")
        state = _make_hf_state(global_step=5, epoch=1.5)
        logs = {"loss": 0.42, "learning_rate": 2e-5}

        adapter.on_log(args, state, None, logs=logs)

        ctx = cb.calls["on_log"]
        assert isinstance(ctx, TrainingHubContext)
        assert ctx.step == 5
        assert ctx.epoch == 1
        assert ctx.loss == 0.42
        assert ctx.learning_rate == 2e-5
        assert ctx.is_main_process is True
        assert ctx.output_dir == "/checkpoints"
        assert ctx.metrics == {"loss": 0.42, "learning_rate": 2e-5}

    def test_loss_from_log_history_fallback(self):
        """When logs don't have loss, fall back to log_history."""
        cb = _RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args = _make_hf_args()
        state = _make_hf_state(
            global_step=10,
            log_history=[{"loss": 0.99, "learning_rate": 1e-4}],
        )

        adapter.on_step_end(args, state, None)

        ctx = cb.calls["on_step_end"]
        assert ctx.loss == 0.99
        assert ctx.learning_rate == 1e-4

    def test_evaluate_forwards_metrics(self):
        """on_evaluate must populate context.metrics from HF metrics kwarg."""
        cb = _RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args = _make_hf_args("/checkpoints")
        state = _make_hf_state(global_step=2, epoch=0.5)
        metrics = {"eval_loss": 2.72, "epoch": 0.5}

        adapter.on_evaluate(args, state, None, metrics=metrics)

        ctx = cb.calls["on_evaluate"]
        assert isinstance(ctx, TrainingHubContext)
        assert ctx.step == 2
        assert ctx.metrics == metrics
        assert ctx.loss == 2.72
        assert ctx.output_dir == "/checkpoints"

    def test_empty_logs_dict_preserved(self):
        """Empty logs={} must not be treated as missing logs."""
        cb = _RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args = _make_hf_args()
        state = _make_hf_state(
            global_step=3,
            log_history=[{"loss": 0.5, "learning_rate": 1e-4}],
        )

        adapter.on_log(args, state, None, logs={})

        ctx = cb.calls["on_log"]
        assert ctx.metrics == {}
        # Empty logs still fall back to log_history for loss/lr
        assert ctx.loss == 0.5
        assert ctx.learning_rate == 1e-4

    def test_all_hooks_dispatch(self):
        """Every adapter hook dispatches to the corresponding user hook."""
        cb = _RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args = _make_hf_args()
        state = _make_hf_state()

        for hook_name in ALL_HOOKS:
            getattr(adapter, hook_name)(args, state, None)

        assert set(cb.calls) == set(ALL_HOOKS)

    def test_exception_isolation(self, caplog):
        """Callback exceptions are caught and logged, never propagated."""

        class Exploder(TrainingHubCallback):
            def on_step_end(self, context):
                raise ValueError("boom")

        adapter = UnslothCallbackAdapter(Exploder())
        args = _make_hf_args()
        state = _make_hf_state()

        with caplog.at_level(logging.ERROR):
            adapter.on_step_end(args, state, None)

        assert "boom" in caplog.text
        assert "Exploder.on_step_end" in caplog.text

    def test_rank_guard_skips_non_main(self):
        """Callbacks should not fire on non-main processes by default."""
        cb = _RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args = _make_hf_args()
        state = _make_hf_state(is_world_process_zero=False)

        adapter.on_step_end(args, state, None)

        assert "on_step_end" not in cb.calls

    def test_run_on_all_ranks_opt_in(self):
        """run_on_all_ranks=True allows worker-rank dispatch."""

        class AllRanks(TrainingHubCallback):
            run_on_all_ranks = True

            def __init__(self) -> None:
                self.called = False

            def on_step_end(self, context):
                self.called = True

        cb = AllRanks()
        adapter = UnslothCallbackAdapter(cb)
        args = _make_hf_args()
        state = _make_hf_state(is_world_process_zero=False)

        adapter.on_step_end(args, state, None)
        assert cb.called is True

    def test_inherits_trainer_callback(self):
        """Adapter must be a TrainerCallback subclass (required by Trainer)."""
        from transformers import TrainerCallback as HFTrainerCallback

        adapter = UnslothCallbackAdapter(TrainingHubCallback())
        assert isinstance(adapter, HFTrainerCallback)


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
class TestAdaptHubCallbacks:
    """Tests for the adapt_hub_callbacks utility."""

    def test_converts_list(self):
        cbs = [TrainingHubCallback(), TrainingHubCallback()]
        adapted = adapt_hub_callbacks(cbs)
        assert len(adapted) == 2
        assert all(isinstance(a, UnslothCallbackAdapter) for a in adapted)

    def test_empty_list(self):
        assert adapt_hub_callbacks([]) == []
