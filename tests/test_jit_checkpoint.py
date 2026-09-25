"""Tests for checkpoint_utils and JIT checkpoint integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from training_hub.callbacks import TrainingHubControl, merge_default_callbacks
from training_hub.checkpoint_utils import (
    INCOMPLETE_SENTINEL,
    find_latest_valid_checkpoint,
    incomplete_sidecar_path,
    is_valid_checkpoint_dir,
    jit_checkpoint_enabled,
    mark_checkpoint_complete,
    mark_checkpoint_incomplete,
)
from training_hub.jit_checkpoint import JITCheckpointCallback


@pytest.fixture(autouse=True)
def _reset_preemption_state():
    """Clear the module-level preemption flags between tests.

    on_train_begin resets them in a real run, but tests drive the hooks
    directly, so without this a test that trips preemption leaks into the next
    one and makes failures order-dependent.
    """
    from training_hub import jit_checkpoint as jc

    for name in ("_PREEMPT_REQUESTED", "_PREEMPT_LOGGED", "_PREEMPT_SAVE_REQUESTED"):
        setattr(jc, name, False)
    jc._PREEMPT_SIGNUM = None
    yield
    for name in ("_PREEMPT_REQUESTED", "_PREEMPT_LOGGED", "_PREEMPT_SAVE_REQUESTED"):
        setattr(jc, name, False)
    jc._PREEMPT_SIGNUM = None


class TestCheckpointUtils:
    def test_incomplete_sidecar_skipped_on_resume(self, tmp_path: Path):
        valid = tmp_path / "checkpoint-10"
        valid.mkdir()
        incomplete = tmp_path / "checkpoint-20"
        incomplete.mkdir()
        incomplete_sidecar_path(tmp_path, 20).touch()

        assert find_latest_valid_checkpoint(str(tmp_path)) == str(valid.resolve())

    def test_legacy_in_dir_sentinel_still_skipped(self, tmp_path: Path):
        valid = tmp_path / "checkpoint-10"
        valid.mkdir()
        incomplete = tmp_path / "checkpoint-20"
        incomplete.mkdir()
        (incomplete / INCOMPLETE_SENTINEL).touch()

        assert find_latest_valid_checkpoint(str(tmp_path)) == str(valid.resolve())

    def test_latest_hf_checkpoint_wins(self, tmp_path: Path):
        (tmp_path / "checkpoint-3").mkdir()
        (tmp_path / "checkpoint-12").mkdir()
        assert find_latest_valid_checkpoint(str(tmp_path)).endswith("checkpoint-12")

    def test_mini_trainer_layout(self, tmp_path: Path):
        step_dir = tmp_path / "full_state_checkpoints" / "step_7"
        step_dir.mkdir(parents=True)
        (step_dir / "training_state.pt").touch()
        assert find_latest_valid_checkpoint(str(tmp_path)) == str(step_dir.resolve())

    def test_mark_complete_removes_sidecar(self, tmp_path: Path):
        mark_checkpoint_incomplete(tmp_path, 1)
        sidecar = incomplete_sidecar_path(tmp_path, 1)
        assert sidecar.exists()
        assert not is_valid_checkpoint_dir(tmp_path / "checkpoint-1", tmp_path)

        (tmp_path / "checkpoint-1").mkdir()
        mark_checkpoint_complete(tmp_path, 1)
        assert not sidecar.exists()
        assert is_valid_checkpoint_dir(tmp_path / "checkpoint-1", tmp_path)

    def test_incomplete_does_not_precreate_checkpoint_dir(self, tmp_path: Path):
        mark_checkpoint_incomplete(tmp_path, 5)
        assert incomplete_sidecar_path(tmp_path, 5).exists()
        assert not (tmp_path / "checkpoint-5").exists()

    def test_layouts_are_not_ranked_against_each_other(self, tmp_path: Path):
        """HF step numbers and InstructLab epoch numbers are different counters.
        Ranking them together would let checkpoint-500 beat epoch_1 on an
        unrelated comparison, so selection stays within a layout."""
        from training_hub.checkpoint_utils import HF_LAYOUT, INSTRUCTLAB_LAYOUT

        hf = tmp_path / "checkpoint-500"
        hf.mkdir()
        ilab = tmp_path / "full_state" / "epoch_1"
        ilab.mkdir(parents=True)
        (ilab / "training_metadata.json").write_text("{}")

        assert find_latest_valid_checkpoint(
            str(tmp_path), layouts=(HF_LAYOUT,)
        ) == str(hf.resolve())
        assert find_latest_valid_checkpoint(
            str(tmp_path), layouts=(INSTRUCTLAB_LAYOUT,)
        ) == str(ilab.resolve())
        # asking for both prefers the order given, never a cross-layout compare
        assert find_latest_valid_checkpoint(
            str(tmp_path), layouts=(INSTRUCTLAB_LAYOUT, HF_LAYOUT)
        ) == str(ilab.resolve())

    def test_jit_checkpoint_enabled_requires_both(self):
        assert not jit_checkpoint_enabled(False, "/tmp")
        assert not jit_checkpoint_enabled(True, None)
        assert jit_checkpoint_enabled(True, "/tmp")


class TestMergeDefaultCallbacks:
    def test_prepends_jit_for_lora_backend(self):
        merged = merge_default_callbacks(
            [],
            enable_jit_checkpoint=True,
            ckpt_output_dir="/ckpt",
            backend="lora_sft",
        )
        assert len(merged) == 1
        assert isinstance(merged[0], JITCheckpointCallback)

    def test_skips_jit_for_native_backends(self):
        for backend in ("sft", "osft"):
            merged = merge_default_callbacks(
                [],
                enable_jit_checkpoint=True,
                ckpt_output_dir="/ckpt",
                backend=backend,
            )
            assert merged == []

    def test_user_callbacks_after_defaults(self):
        from training_hub.callbacks import TrainingHubCallback

        class UserCb(TrainingHubCallback):
            pass

        merged = merge_default_callbacks(
            [UserCb()],
            enable_jit_checkpoint=True,
            ckpt_output_dir="/ckpt",
            backend="lora_sft",
        )
        assert isinstance(merged[0], JITCheckpointCallback)
        assert type(merged[1]).__name__ == "UserCb"


class TestJITCheckpointCallback:
    def test_preemption_sets_control_flags(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(
            "training_hub.jit_checkpoint.preempt_requested",
            lambda: True,
        )
        cb = JITCheckpointCallback()
        control = TrainingHubControl()
        ctx = SimpleNamespace(
            output_dir=str(tmp_path),
            step=5,
            is_main_process=True,
            metrics={},
            control=control,
        )
        cb.on_step_end(ctx)
        assert control.should_save is True
        assert control.should_training_stop is True
        assert incomplete_sidecar_path(tmp_path, 5).exists()
        assert not (tmp_path / "checkpoint-5").exists()

    def test_sigterm_handler_does_no_logging(self):
        """logging takes non-reentrant locks, so a handler that logs can
        deadlock the main thread and lose the checkpoint entirely."""
        import signal as signal_mod

        from training_hub import jit_checkpoint as jc

        calls = []

        class Boom:
            def __getattr__(self, name):
                def record(*args, **kwargs):
                    calls.append(name)
                    raise AssertionError("signal handler must not log")

                return record

        original_logger = jc.logger
        jc.logger = Boom()
        try:
            jc._handle_sigterm(signal_mod.SIGTERM, None)
        finally:
            jc.logger = original_logger
        assert calls == []
        assert jc.preempt_requested() is True
        assert jc._PREEMPT_SIGNUM == int(signal_mod.SIGTERM)
        jc._PREEMPT_REQUESTED = False
        jc._PREEMPT_LOGGED = False

    def test_signal_notice_logged_at_step_boundary(self, monkeypatch, tmp_path, caplog):
        """The cluster harness greps for this line, so it must still be emitted
        — just from the hook rather than the handler."""
        import signal as signal_mod

        from training_hub import jit_checkpoint as jc

        monkeypatch.setattr(jc, "_PREEMPT_REQUESTED", False)
        monkeypatch.setattr(jc, "_PREEMPT_LOGGED", False)
        jc._handle_sigterm(signal_mod.SIGTERM, None)
        try:
            control = TrainingHubControl()
            ctx = SimpleNamespace(
                output_dir=str(tmp_path),
                step=5,
                is_main_process=True,
                metrics={},
                control=control,
            )
            with caplog.at_level("WARNING"):
                JITCheckpointCallback().on_step_end(ctx)
                JITCheckpointCallback().on_step_end(ctx)
            assert sum("Received signal" in r.message for r in caplog.records) == 1
        finally:
            jc._PREEMPT_REQUESTED = False
            jc._PREEMPT_LOGGED = False

    def test_preemption_requests_the_save_only_once(self, monkeypatch, tmp_path: Path):
        """_handle_preemption runs from on_step_end AND on_epoch_end while the
        flag stays set. Without a one-shot guard HF writes the same checkpoint
        twice and the mirror uploads it twice (seen on cluster: checkpoint-55
        mirrored 2x during one grace period)."""
        from training_hub import jit_checkpoint as jc

        monkeypatch.setattr(jc, "preempt_requested", lambda: True)
        monkeypatch.setattr(jc, "_PREEMPT_SAVE_REQUESTED", False)
        cb = JITCheckpointCallback()
        saves = 0
        for hook in (cb.on_step_end, cb.on_epoch_end, cb.on_step_end):
            control = TrainingHubControl()
            cb_ctx = SimpleNamespace(
                output_dir=str(tmp_path),
                step=55,
                is_main_process=True,
                metrics={},
                control=control,
            )
            hook(cb_ctx)
            saves += int(control.should_save)
            # stopping must stay sticky on every hook, only saving is one-shot
            assert control.should_training_stop is True
        assert saves == 1

    def test_no_preempt_is_noop(self):
        cb = JITCheckpointCallback()
        control = TrainingHubControl()
        ctx = SimpleNamespace(
            output_dir="/out",
            step=1,
            is_main_process=True,
            metrics={},
            control=control,
        )
        cb.on_step_end(ctx)
        assert control.should_save is False
        assert control.should_training_stop is False


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("transformers") is None,
    reason="transformers not installed",
)
class TestUnslothControlWiring:
    def test_step_end_returns_hf_control_flags(self):
        from training_hub.adapters.unsloth import adapt_hub_callbacks
        from training_hub.callbacks import TrainingHubCallback, TrainingHubContext

        class Preempt(TrainingHubCallback):
            run_on_all_ranks = True

            def on_step_end(self, context: TrainingHubContext) -> None:
                assert context.control is not None
                context.control.should_save = True
                context.control.should_training_stop = True

        adapter = adapt_hub_callbacks([Preempt()])[0]
        args = SimpleNamespace(output_dir="/out")
        state = SimpleNamespace(
            global_step=2,
            epoch=0.0,
            is_world_process_zero=True,
            log_history=[],
        )
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        result = adapter.on_step_end(args, state, control)
        assert result.should_save is True
        assert result.should_training_stop is True

    def test_stale_should_save_is_not_reapplied(self):
        """A preemption stops training, so on_step_begin never runs again. If
        the adapter does not consume should_save, on_epoch_end re-applies it
        and HF writes (and the mirror uploads) the same checkpoint twice —
        observed on cluster as 'Mirrored checkpoint checkpoint-18' x2."""
        from training_hub.adapters.unsloth import adapt_hub_callbacks
        from training_hub.callbacks import TrainingHubCallback, TrainingHubContext

        class Preempt(TrainingHubCallback):
            run_on_all_ranks = True
            fired = False

            def on_step_end(self, context: TrainingHubContext) -> None:
                if not Preempt.fired:
                    Preempt.fired = True
                    context.control.should_save = True
                context.control.should_training_stop = True

        adapter = adapt_hub_callbacks([Preempt()])[0]
        args = SimpleNamespace(output_dir="/out")
        state = SimpleNamespace(
            global_step=18, epoch=1.0, is_world_process_zero=True, log_history=[]
        )

        c1 = SimpleNamespace(should_save=False, should_training_stop=False)
        adapter.on_step_end(args, state, c1)
        assert c1.should_save is True

        # training is stopping, so no on_step_begin resets anything
        c2 = SimpleNamespace(should_save=False, should_training_stop=False)
        adapter.on_epoch_end(args, state, c2)
        assert c2.should_save is False, "stale should_save caused a duplicate save"
        assert c2.should_training_stop is True

    def test_on_save_uses_global_step_not_best_checkpoint(self):
        from training_hub.adapters.unsloth import adapt_hub_callbacks
        from training_hub.callbacks import TrainingHubCallback, TrainingHubContext

        captured: dict[str, str] = {}

        class CaptureSave(TrainingHubCallback):
            def on_save(self, context: TrainingHubContext) -> None:
                captured["path"] = context.metrics["checkpoint_path"]

        adapter = adapt_hub_callbacks([CaptureSave()])[0]
        args = SimpleNamespace(output_dir="/runs/out")
        state = SimpleNamespace(
            global_step=42,
            best_model_checkpoint="/runs/out/checkpoint-10",
            epoch=1.0,
            is_world_process_zero=True,
            log_history=[],
        )
        control = SimpleNamespace()
        adapter.on_save(args, state, control)
        assert captured["path"] == "/runs/out/checkpoint-42"
