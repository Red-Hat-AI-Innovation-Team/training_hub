"""Tests for the unified callback abstraction layer.

Covers:
- TrainingHubContext defaults and custom values
- TrainingHubCallback no-op defaults and selective override
- UnslothCallbackAdapter: context mapping, exception isolation, rank guard
- adapt_hub_callbacks utility
"""

from __future__ import annotations

import logging
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Serialize helpers + InstructLab / Mini-Trainer bridges
# ---------------------------------------------------------------------------


class _SerializableLogger(TrainingHubCallback):
    """Module-level callback for serialization tests."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_log(self, context: TrainingHubContext) -> None:
        self.events.append(f"log:{context.step}:{context.loss}")

    def on_train_begin(self, context: TrainingHubContext) -> None:
        self.events.append("begin")


class _AllRanksLogger(TrainingHubCallback):
    run_on_all_ranks = True

    def __init__(self) -> None:
        self.calls = 0

    def on_step_end(self, context: TrainingHubContext) -> None:
        self.calls += 1


class _BoomCallback(TrainingHubCallback):
    def on_log(self, context: TrainingHubContext) -> None:
        raise RuntimeError("boom")


class TestSerializeHubCallbacks:
    """Tests for torchrun payload encode/decode."""

    def test_round_trip(self, tmp_path):
        from training_hub.adapters.serialize import (
            decode_hub_callbacks,
            encode_hub_callbacks,
            load_hub_callbacks_payload,
            write_hub_callbacks_payload,
        )

        original = _SerializableLogger()
        encoded = encode_hub_callbacks([original])
        restored = decode_hub_callbacks(encoded)
        assert len(restored) == 1
        assert type(restored[0]).__name__ == "_SerializableLogger"

        path = write_hub_callbacks_payload([original], payload_dir=str(tmp_path))
        loaded = load_hub_callbacks_payload(path)
        assert len(loaded) == 1
        assert oct(Path(path).stat().st_mode & 0o777) == oct(0o600)
        loaded[0].on_train_begin(TrainingHubContext(step=1))
        # Fresh instance — events list starts empty then gets begin
        assert loaded[0].events == ["begin"]

    def test_rejects_non_callback(self):
        from training_hub.adapters.serialize import normalize_hub_callbacks

        with pytest.raises(TypeError, match="TrainingHubCallback"):
            normalize_hub_callbacks([object()])  # type: ignore[list-item]

    def test_dynamic_class_fails_getsource(self):
        from training_hub.adapters.serialize import encode_hub_callback

        Dyn = type("DynCallback", (TrainingHubCallback,), {})
        with pytest.raises(ValueError, match="Cannot serialize"):
            encode_hub_callback(Dyn())


class TestDistributedContextAndDispatch:
    """Shared native-context mapping + dispatcher behavior."""

    def test_build_context_maps_fields(self):
        from training_hub.adapters.distributed import build_hub_context_from_native

        native = SimpleNamespace(
            step=7,
            epoch=2,
            loss=0.3,
            learning_rate=1e-5,
            is_world_process_zero=True,
            output_dir="/out",
            batch_metrics={"loss": 0.3, "tok": 10},
            val_metrics={"eval_loss": 0.4},
            checkpoint_path="/out/ckpt-7",
        )
        ctx = build_hub_context_from_native(native)
        assert ctx.step == 7
        assert ctx.epoch == 2
        assert ctx.loss == 0.3
        assert ctx.learning_rate == 1e-5
        assert ctx.is_main_process is True
        assert ctx.output_dir == "/out"
        assert ctx.metrics["tok"] == 10
        assert ctx.metrics["eval_loss"] == 0.4
        assert ctx.metrics["checkpoint_path"] == "/out/ckpt-7"

    def test_build_context_preserves_zero_eval_loss(self):
        from training_hub.adapters.distributed import build_hub_context_from_native

        native = SimpleNamespace(
            step=1,
            epoch=0,
            loss=None,
            learning_rate=None,
            is_world_process_zero=True,
            output_dir="/out",
            batch_metrics={},
            val_metrics={"eval_loss": 0.0},
            checkpoint_path=None,
        )
        ctx = build_hub_context_from_native(native)
        assert ctx.loss == 0.0

    def test_rank_guard_and_exception_isolation(self, tmp_path, caplog):
        from training_hub.adapters.distributed import HubCallbackDispatcher
        from training_hub.adapters.serialize import (
            set_callbacks_payload_env,
            write_hub_callbacks_payload,
        )

        path = write_hub_callbacks_payload(
            [_SerializableLogger(), _BoomCallback(), _AllRanksLogger()],
            payload_dir=str(tmp_path),
        )
        set_callbacks_payload_env(path)
        dispatcher = HubCallbackDispatcher()

        native_main = SimpleNamespace(
            step=1,
            epoch=0,
            loss=0.1,
            learning_rate=None,
            is_world_process_zero=True,
            output_dir="/out",
            batch_metrics={},
            val_metrics={},
            checkpoint_path=None,
        )
        with caplog.at_level(logging.ERROR):
            dispatcher.dispatch("on_log", native_main)
        assert "boom" in caplog.text or "BoomCallback" in caplog.text or "ignored" in caplog.text

        # Non-main: only run_on_all_ranks callbacks fire
        dispatcher2 = HubCallbackDispatcher()
        native_worker = SimpleNamespace(
            step=2,
            epoch=0,
            loss=0.2,
            learning_rate=None,
            is_world_process_zero=False,
            output_dir="/out",
            batch_metrics={},
            val_metrics={},
            checkpoint_path=None,
        )
        dispatcher2.dispatch("on_step_end", native_worker)
        # Fresh load — AllRanksLogger should have been called once
        ranks = [
            cb for cb in dispatcher2._callbacks() if type(cb).__name__ == "_AllRanksLogger"
        ]
        assert len(ranks) == 1
        assert ranks[0].calls == 1

    def test_load_failure_disables_callbacks(self, tmp_path, caplog):
        from training_hub.adapters.distributed import HubCallbackDispatcher
        from training_hub.adapters.serialize import set_callbacks_payload_env

        set_callbacks_payload_env(str(tmp_path / "missing_payload.json"))
        dispatcher = HubCallbackDispatcher()
        native = SimpleNamespace(
            step=1,
            epoch=0,
            loss=0.1,
            learning_rate=None,
            is_world_process_zero=True,
            output_dir="/out",
            batch_metrics={},
            val_metrics={},
            checkpoint_path=None,
        )
        with caplog.at_level(logging.ERROR):
            dispatcher.dispatch("on_log", native)
        assert dispatcher._callbacks() == []
        assert "Failed to load hub callback payload" in caplog.text


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("instructlab") is None,
    reason="instructlab-training not installed",
)
class TestInstructLabBridge:
    def test_adapt_writes_payload_and_returns_bridge(self, tmp_path):
        from training_hub.adapters.instructlab import (
            InstructLabCallbackBridge,
            adapt_hub_callbacks,
        )
        from training_hub.adapters.serialize import CALLBACKS_PATH_ENV
        import os

        adapted = adapt_hub_callbacks([_SerializableLogger()], payload_dir=str(tmp_path))
        assert len(adapted) == 1
        assert isinstance(adapted[0], InstructLabCallbackBridge)
        assert os.environ.get(CALLBACKS_PATH_ENV)
        assert (tmp_path / "training_hub_callbacks.json").exists()

        native = SimpleNamespace(
            step=3,
            epoch=1,
            loss=0.5,
            learning_rate=2e-4,
            is_world_process_zero=True,
            output_dir=str(tmp_path),
            batch_metrics={"loss": 0.5},
            val_metrics={},
            checkpoint_path=None,
        )
        adapted[0].on_log(native)

    def test_bridge_survives_upstream_serialize_roundtrip(self, tmp_path):
        from instructlab.training.callbacks import (
            deserialize_callback,
            serialize_callback,
        )
        from training_hub.adapters.instructlab import (
            InstructLabCallbackBridge,
            adapt_hub_callbacks,
        )

        adapt_hub_callbacks([_SerializableLogger()], payload_dir=str(tmp_path))
        bridge = InstructLabCallbackBridge()
        restored = deserialize_callback(serialize_callback(bridge))
        assert type(restored).__name__ == "InstructLabCallbackBridge"

        native = SimpleNamespace(
            step=9,
            epoch=0,
            loss=0.9,
            learning_rate=None,
            is_world_process_zero=True,
            output_dir=str(tmp_path),
            batch_metrics={},
            val_metrics={},
            checkpoint_path=None,
        )
        restored.on_train_begin(native)


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("mini_trainer") is None,
    reason="mini_trainer not installed",
)
class TestMiniTrainerBridge:
    def test_adapt_and_upstream_serialize(self, tmp_path):
        from mini_trainer.callbacks import deserialize_callback, serialize_callback
        from training_hub.adapters.mini_trainer import (
            MiniTrainerCallbackBridge,
            adapt_hub_callbacks,
        )

        adapted = adapt_hub_callbacks([_SerializableLogger()], payload_dir=str(tmp_path))
        assert isinstance(adapted[0], MiniTrainerCallbackBridge)
        restored = deserialize_callback(serialize_callback(adapted[0]))
        assert type(restored).__name__ == "MiniTrainerCallbackBridge"
        restored.on_train_end(
            SimpleNamespace(
                step=1,
                epoch=0,
                loss=None,
                learning_rate=None,
                is_world_process_zero=True,
                output_dir=str(tmp_path),
                batch_metrics={},
                val_metrics={},
                checkpoint_path=None,
            )
        )
