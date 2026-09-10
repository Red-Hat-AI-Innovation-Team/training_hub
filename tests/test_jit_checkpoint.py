"""Tests for checkpoint_utils and JIT checkpoint integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from training_hub.callbacks import TrainingHubControl, merge_default_callbacks
from training_hub.checkpoint_utils import (
    INCOMPLETE_SENTINEL,
    INCOMPLETE_SIDECAR_PREFIX,
    find_latest_valid_checkpoint,
    incomplete_sidecar_path,
    is_valid_checkpoint_dir,
    jit_checkpoint_enabled,
    mark_checkpoint_complete,
    mark_checkpoint_incomplete,
)
from training_hub.jit_checkpoint import JITCheckpointCallback


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
