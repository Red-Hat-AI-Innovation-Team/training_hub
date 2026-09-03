"""Tests for checkpoint_storage selection (pvc/s3) and S3 sync plumbing."""

import os
from pathlib import Path

import pytest

from training_hub.callbacks import merge_default_callbacks
from training_hub.checkpoint_utils import (
    UPLOAD_URI_ENV,
    apply_checkpoint_storage_env,
    resolve_checkpoint_storage,
)
from training_hub.jit_checkpoint import JITCheckpointCallback, S3CheckpointSyncCallback


class TestResolveCheckpointStorage:
    def test_none_and_pvc_mean_filesystem_only(self):
        assert resolve_checkpoint_storage(None) is None
        assert resolve_checkpoint_storage("") is None
        assert resolve_checkpoint_storage("pvc") is None

    def test_s3_uri_passes_through(self):
        assert resolve_checkpoint_storage("s3://bucket/prefix") == "s3://bucket/prefix"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="checkpoint_storage"):
            resolve_checkpoint_storage("gs://bucket/x")

    def test_apply_env(self, monkeypatch):
        monkeypatch.delenv(UPLOAD_URI_ENV, raising=False)
        apply_checkpoint_storage_env("pvc")
        assert UPLOAD_URI_ENV not in os.environ
        apply_checkpoint_storage_env("s3://b/p")
        assert os.environ[UPLOAD_URI_ENV] == "s3://b/p"


class TestMergeDefaultsWithStorage:
    def test_s3_sync_prepended_for_hub_jit_backend(self):
        cbs = merge_default_callbacks(
            None,
            enable_jit_checkpoint=True,
            ckpt_output_dir="/tmp/x",
            backend="lora_sft",
            checkpoint_storage="s3://b/p",
        )
        assert isinstance(cbs[0], JITCheckpointCallback)
        assert isinstance(cbs[1], S3CheckpointSyncCallback)

    def test_s3_sync_prepended_even_for_native_backends(self):
        cbs = merge_default_callbacks(
            None,
            enable_jit_checkpoint=True,
            ckpt_output_dir="/tmp/x",
            backend="osft",
            checkpoint_storage="s3://b/p",
        )
        # native backend: no JIT callback, but S3 sync still present
        assert len(cbs) == 1
        assert isinstance(cbs[0], S3CheckpointSyncCallback)

    def test_pvc_adds_no_sync_callback(self):
        cbs = merge_default_callbacks(
            None,
            enable_jit_checkpoint=True,
            ckpt_output_dir="/tmp/x",
            backend="sft",
            checkpoint_storage="pvc",
        )
        assert cbs == []


class TestUploadLayout:
    def test_enqueue_preserves_relative_layout(self, monkeypatch, tmp_path: Path):
        """Nested checkpoint dirs keep their relative path in the queue item."""
        import training_hub.checkpoint_manager as cm

        monkeypatch.setenv(UPLOAD_URI_ENV, "s3://b/p")
        captured = []

        class FakeQueue:
            def put(self, item):
                captured.append(item)

        monkeypatch.setattr(cm, "_UPLOAD_QUEUE", FakeQueue())
        monkeypatch.setattr(cm, "ensure_upload_worker_started", lambda: None)

        ckpt = tmp_path / "full_state_checkpoints" / "step_9"
        ckpt.mkdir(parents=True)
        cm.enqueue_checkpoint_upload(ckpt, base_dir=tmp_path)

        local, base = captured[0]
        assert local == str(ckpt.resolve())
        assert base == str(tmp_path.resolve())


class TestEnvClearedForPvc:
    def test_s3_then_pvc_clears_stale_uri(self, monkeypatch):
        """Switching from S3 to PVC in one process must not leak the URI."""
        monkeypatch.delenv(UPLOAD_URI_ENV, raising=False)
        apply_checkpoint_storage_env("s3://b/p")
        assert os.environ[UPLOAD_URI_ENV] == "s3://b/p"
        apply_checkpoint_storage_env(None)
        assert UPLOAD_URI_ENV not in os.environ


class TestTrainParamPath:
    """Full train() param path with a mock backend: catches unsupported-kwarg
    regressions (e.g. apply_native_jit_params signature drift) that pure
    helper tests miss."""

    class _CaptureBackend:
        def __init__(self):
            self.params = None

        def execute_training(self, params):
            self.params = params
            return "ok"

    def test_sft_train_params_path(self, tmp_path):
        from training_hub.algorithms.sft import SFTAlgorithm

        backend = self._CaptureBackend()
        result = SFTAlgorithm(backend).train(
            model_path="m",
            data_path="d",
            ckpt_output_dir=str(tmp_path),
            enable_jit_checkpoint=True,
            checkpoint_storage="pvc",
        )
        assert result == "ok"
        assert backend.params["on_demand_checkpointing"] is True

    def test_osft_train_params_path(self, tmp_path):
        from training_hub.algorithms.osft import OSFTAlgorithm

        backend = self._CaptureBackend()
        result = OSFTAlgorithm(backend).train(
            model_path="m",
            data_path="d",
            unfreeze_rank_ratio=0.25,
            effective_batch_size=8,
            max_tokens_per_gpu=4096,
            max_seq_len=512,
            learning_rate=1e-5,
            ckpt_output_dir=str(tmp_path),
            enable_jit_checkpoint=True,
            checkpoint_storage="pvc",
        )
        assert result == "ok"
        assert backend.params["on_demand_checkpointing"] is True
