"""Tests for checkpoint_storage selection and remote mirroring/restore.

Remote storage is exercised end to end against fsspec's in-memory filesystem
(``memory://``), so staging, the completion marker, LIFO draining, error
surfacing and restore all run without S3.
"""

import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import fsspec
import pytest

import training_hub.checkpoint_manager as cm
from training_hub.callbacks import TrainingHubControl, merge_default_callbacks
from training_hub.checkpoint_utils import (
    UPLOAD_URI_ENV,
    apply_checkpoint_storage_env,
    find_latest_valid_checkpoint,
    resolve_checkpoint_storage,
)
from training_hub.jit_checkpoint import JITCheckpointCallback, RemoteCheckpointSyncCallback

REMOTE = "memory://bucket/run1"


def _clear_memory_fs():
    mem = fsspec.filesystem("memory")
    mem.store.clear()
    mem.pseudo_dirs[:] = [""]


@pytest.fixture
def remote(monkeypatch):
    """Point checkpoint storage at a clean in-memory bucket; returns its filesystem."""
    _clear_memory_fs()
    monkeypatch.setenv(UPLOAD_URI_ENV, REMOTE)
    yield cm.remote_filesystem(REMOTE)
    cm.shutdown_upload_worker(timeout=30)
    cm._uploader = None
    _clear_memory_fs()


def _raise_oserror(*args, **kwargs):
    raise OSError("bucket exploded")


def _call_capture(fn, *args):
    """Run *fn*, returning the exception it raised (or None)."""
    try:
        fn(*args)
        return None
    except Exception as e:  # noqa: BLE001 - the test asserts on the result
        return e


def _make_checkpoint(root: Path, name: str, files=("model.safetensors", "trainer_state.json")) -> Path:
    ckpt = root / name
    ckpt.mkdir(parents=True)
    for f in files:
        (ckpt / f).write_text(f"{name}:{f}")
    return ckpt


def _ctx(tmp_path: Path, step: int, checkpoint: Path, main: bool = True, control=None):
    return SimpleNamespace(
        is_main_process=main,
        output_dir=str(tmp_path),
        step=step,
        metrics={"checkpoint_path": str(checkpoint)},
        control=control,
    )


class TestResolveCheckpointStorage:
    def test_none_and_pvc_mean_filesystem_only(self):
        assert resolve_checkpoint_storage(None) is None
        assert resolve_checkpoint_storage("") is None
        assert resolve_checkpoint_storage("pvc") is None

    def test_any_fsspec_uri_passes_through(self):
        assert resolve_checkpoint_storage("s3://bucket/prefix") == "s3://bucket/prefix"
        assert resolve_checkpoint_storage("gs://bucket/prefix") == "gs://bucket/prefix"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="checkpoint_storage"):
            resolve_checkpoint_storage("bucket/prefix")

    def test_apply_env(self, monkeypatch):
        monkeypatch.delenv(UPLOAD_URI_ENV, raising=False)
        apply_checkpoint_storage_env("pvc")
        assert UPLOAD_URI_ENV not in os.environ
        apply_checkpoint_storage_env("s3://b/p")
        assert os.environ[UPLOAD_URI_ENV] == "s3://b/p"

    def test_s3_then_pvc_clears_stale_uri(self, monkeypatch):
        """Switching from remote to PVC in one process must not leak the URI."""
        monkeypatch.delenv(UPLOAD_URI_ENV, raising=False)
        apply_checkpoint_storage_env("s3://b/p")
        apply_checkpoint_storage_env(None)
        assert UPLOAD_URI_ENV not in os.environ


class TestVerifyStorageAccess:
    def test_missing_fsspec_backend_names_it(self):
        with pytest.raises(ImportError, match="'gs' backend"):
            cm.verify_storage_access("gs://bucket/x")

    def test_concurrent_ranks_do_not_clobber_each_others_probe(self, remote):
        """Every rank probes the same prefix from train(); a shared probe key
        made ranks delete each other's object and fail startup."""
        import concurrent.futures as cf

        with cf.ThreadPoolExecutor(8) as pool:
            errors = [
                e
                for e in pool.map(
                    lambda _: _call_capture(cm.verify_storage_access, REMOTE), range(8)
                )
                if e is not None
            ]
        assert errors == []
        assert remote.find("") == []  # no probe objects left behind

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
    def test_unwritable_uri_raises(self, tmp_path):
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o500)
        try:
            with pytest.raises(RuntimeError, match="not writable"):
                cm.verify_storage_access(f"file://{ro}")
        finally:
            ro.chmod(0o700)

    def test_reachable_uri_passes_and_leaves_no_probe(self, remote):
        cm.verify_storage_access(REMOTE)
        assert remote.find("") == []


class TestMergeDefaultsWithStorage:
    def test_sync_prepended_after_jit_for_lora(self):
        cbs = merge_default_callbacks(
            None,
            enable_jit_checkpoint=True,
            ckpt_output_dir="/tmp/x",
            backend="lora_sft",
            checkpoint_storage="s3://b/p",
        )
        assert isinstance(cbs[0], JITCheckpointCallback)
        assert isinstance(cbs[1], RemoteCheckpointSyncCallback)

    def test_native_backends_get_no_sync_callback(self):
        """sft/osft workers exit without firing on_save; the launcher mirrors instead."""
        for backend in ("sft", "osft"):
            cbs = merge_default_callbacks(
                None,
                enable_jit_checkpoint=True,
                ckpt_output_dir="/tmp/x",
                backend=backend,
                checkpoint_storage="s3://b/p",
            )
            assert cbs == []

    def test_pvc_adds_no_sync_callback(self):
        cbs = merge_default_callbacks(
            None,
            enable_jit_checkpoint=True,
            ckpt_output_dir="/tmp/x",
            backend="lora_sft",
            checkpoint_storage="pvc",
        )
        assert [type(c) for c in cbs] == [JITCheckpointCallback]


class TestBackgroundUpload:
    def test_upload_mirrors_layout_keeps_original_and_writes_marker(self, remote, tmp_path):
        ckpt = _make_checkpoint(tmp_path, "checkpoint-10")
        cm.enqueue_checkpoint_upload(ckpt, base_dir=tmp_path)
        cm.shutdown_upload_worker(timeout=30)

        assert remote.cat("checkpoint-10/model.safetensors") == b"checkpoint-10:model.safetensors"
        assert remote.exists("checkpoint-10/.upload_complete")
        assert ckpt.is_dir()  # original stays for local resume
        assert not (tmp_path / cm.STAGING_DIR / "checkpoint-10").exists()  # staging cleaned

    def test_nested_layout_preserved(self, remote, tmp_path):
        ckpt = _make_checkpoint(
            tmp_path / "full_state_checkpoints", "step_9", files=("training_state.pt",)
        )
        cm.enqueue_checkpoint_upload(ckpt, base_dir=tmp_path)
        cm.shutdown_upload_worker(timeout=30)
        assert remote.exists("full_state_checkpoints/step_9/.upload_complete")

    def test_staged_copy_survives_rotation_of_original(self, remote, tmp_path, monkeypatch):
        """save_total_limit may delete checkpoint-N mid-upload; the staged copy must not care."""
        gate = threading.Event()
        real = cm._upload_dir

        def slow(fs, local_dir, prefix):
            gate.wait(10)
            real(fs, local_dir, prefix)

        monkeypatch.setattr(cm, "_upload_dir", slow)
        ckpt = _make_checkpoint(tmp_path, "checkpoint-10")
        cm.enqueue_checkpoint_upload(ckpt, base_dir=tmp_path)
        shutil.rmtree(ckpt)  # rotation
        gate.set()
        cm.shutdown_upload_worker(timeout=30)
        assert remote.cat("checkpoint-10/model.safetensors") == b"checkpoint-10:model.safetensors"

    def test_shutdown_drains_backlog_newest_first(self, remote, tmp_path, monkeypatch):
        """An in-band stop sentinel on a LIFO queue would drop everything queued behind it."""
        order = []
        entered, gate = threading.Event(), threading.Event()
        real = cm._upload_dir

        def slow(fs, local_dir, prefix):
            entered.set()
            gate.wait(10)
            order.append(prefix)
            real(fs, local_dir, prefix)

        monkeypatch.setattr(cm, "_upload_dir", slow)
        cm.enqueue_checkpoint_upload(_make_checkpoint(tmp_path, "checkpoint-1"), base_dir=tmp_path)
        assert entered.wait(10)  # worker is busy; everything below queues up
        for step in (10, 20, 30):
            cm.enqueue_checkpoint_upload(
                _make_checkpoint(tmp_path, f"checkpoint-{step}"), base_dir=tmp_path
            )
        gate.set()
        cm.shutdown_upload_worker(timeout=30)

        assert order == ["checkpoint-1", "checkpoint-30", "checkpoint-20", "checkpoint-10"]
        assert all(remote.exists(f"{name}/.upload_complete") for name in order)

    def test_upload_failure_is_surfaced_not_swallowed(self, remote, tmp_path, monkeypatch):
        def boom(fs, local_dir, prefix):
            raise OSError("bucket exploded")

        monkeypatch.setattr(cm, "_upload_dir", boom)
        cm.enqueue_checkpoint_upload(_make_checkpoint(tmp_path, "checkpoint-10"), base_dir=tmp_path)
        cm.shutdown_upload_worker(timeout=30)

        assert cm.has_pending_upload_error()
        with pytest.raises(RuntimeError, match="checkpoint-10") as exc:
            cm.raise_pending_upload_error()
        assert isinstance(exc.value.__cause__, OSError)
        assert not cm.has_pending_upload_error()

    def test_no_marker_when_a_file_fails(self, remote, tmp_path, monkeypatch):
        monkeypatch.setattr(cm, "UPLOAD_ATTEMPTS", 1)

        class Flaky:
            def __getattr__(self, name):
                return getattr(remote, name)

            def put_file(self, lpath, rpath, **kw):
                if rpath.endswith("trainer_state.json"):
                    raise OSError("nope")
                return remote.put_file(lpath, rpath, **kw)

        with pytest.raises(OSError):
            cm._upload_dir(Flaky(), _make_checkpoint(tmp_path, "checkpoint-10"), "checkpoint-10")
        assert remote.exists("checkpoint-10/model.safetensors")
        assert not remote.exists("checkpoint-10/.upload_complete")


class TestRestore:
    def _upload(self, tmp_path: Path, name: str) -> Path:
        ckpt = _make_checkpoint(tmp_path / "src", name)
        cm.upload_checkpoint_now(ckpt, base_dir=tmp_path / "src")
        return ckpt

    def test_restores_latest_complete_and_skips_incomplete(self, remote, tmp_path):
        self._upload(tmp_path, "checkpoint-10")
        self._upload(tmp_path, "checkpoint-20")
        remote.pipe("checkpoint-30/model.safetensors", b"partial")  # no marker: interrupted upload

        dest = tmp_path / "dest"
        assert cm.restore_latest_checkpoint(REMOTE, dest) == str(dest / "checkpoint-20")
        assert (dest / "checkpoint-20" / "model.safetensors").read_text() == "checkpoint-20:model.safetensors"
        assert not (dest / "checkpoint-30").exists()
        assert not (dest / "checkpoint-20" / ".upload_complete").exists()
        assert not any(p.name.startswith(cm.RESTORE_TMP_DIR) for p in dest.iterdir())

    def test_ranking_uses_trailing_number(self, remote, tmp_path):
        self._upload(tmp_path, "checkpoint-9")
        self._upload(tmp_path, "checkpoint-10")
        assert cm.restore_latest_checkpoint(REMOTE, tmp_path / "dest").endswith("checkpoint-10")

    def test_nested_layout_restores_in_place(self, remote, tmp_path):
        src = tmp_path / "src"
        ckpt = _make_checkpoint(src / "full_state_checkpoints", "step_9", files=("training_state.pt",))
        cm.upload_checkpoint_now(ckpt, base_dir=src)

        dest = tmp_path / "dest"
        expected = dest / "full_state_checkpoints" / "step_9"
        assert cm.restore_latest_checkpoint(REMOTE, dest) == str(expected)
        assert (expected / "training_state.pt").exists()
        assert find_latest_valid_checkpoint(str(dest)) == str(expected)

    def test_interrupted_download_leaves_nothing_resumable(self, remote, tmp_path, monkeypatch):
        self._upload(tmp_path, "checkpoint-10")

        class Failing:
            def __getattr__(self, name):
                return getattr(remote, name)

            def get_file(self, rpath, lpath, **kw):
                if rpath.endswith("trainer_state.json"):
                    raise OSError("network")
                return remote.get_file(rpath, lpath, **kw)

        monkeypatch.setattr(cm, "remote_filesystem", lambda uri: Failing())
        dest = tmp_path / "dest"
        with pytest.raises(OSError):
            cm.restore_latest_checkpoint(REMOTE, dest)
        assert not (dest / "checkpoint-10").exists()
        assert find_latest_valid_checkpoint(str(dest)) is None

    def test_path_traversal_key_rejected(self, remote, tmp_path, monkeypatch):
        self._upload(tmp_path, "checkpoint-10")

        class Hostile:
            def __getattr__(self, name):
                return getattr(remote, name)

            def find(self, path):
                keys = remote.find(path)
                return keys + ["checkpoint-10/../../escape.txt"] if path else keys

        monkeypatch.setattr(cm, "remote_filesystem", lambda uri: Hostile())
        dest = tmp_path / "dest"
        with pytest.raises(ValueError, match="outside the checkpoint directory"):
            cm.restore_latest_checkpoint(REMOTE, dest)
        assert not list(tmp_path.rglob("escape.txt"))

    def test_maybe_restore_skips_when_local_checkpoint_exists(self, remote, tmp_path):
        self._upload(tmp_path, "checkpoint-20")
        local = tmp_path / "local"
        _make_checkpoint(local, "checkpoint-10")
        assert cm.maybe_restore_checkpoint(local) is None
        assert not (local / "checkpoint-20").exists()

    def test_invalid_local_copy_is_replaced_not_kept(self, remote, tmp_path):
        """A pod killed mid-save leaves a truncated checkpoint-20 plus its
        sidecar; the complete remote copy of the same step must still land,
        otherwise training silently restarts from step 0."""
        from training_hub.checkpoint_utils import mark_checkpoint_incomplete

        self._upload(tmp_path, "checkpoint-20")
        local = tmp_path / "local"
        truncated = local / "checkpoint-20"
        truncated.mkdir(parents=True)
        (truncated / "model.safetensors").write_text("TRUNCATED")
        mark_checkpoint_incomplete(local, 20)
        assert find_latest_valid_checkpoint(str(local)) is None

        assert cm.maybe_restore_checkpoint(local) == str(truncated)
        assert (truncated / "model.safetensors").read_text() == "checkpoint-20:model.safetensors"
        # the stale sidecar must go too, or resume still skips the checkpoint
        assert find_latest_valid_checkpoint(str(local)) == str(truncated.resolve())

    def test_valid_local_copy_is_not_redownloaded(self, remote, tmp_path):
        self._upload(tmp_path, "checkpoint-20")
        local = tmp_path / "local"
        kept = _make_checkpoint(local, "checkpoint-20", files=("model.safetensors",))
        (kept / "model.safetensors").write_text("LOCAL")
        assert cm.restore_latest_checkpoint(REMOTE, local) == str(kept)
        assert (kept / "model.safetensors").read_text() == "LOCAL"

    def test_maybe_restore_downloads_when_local_empty(self, remote, tmp_path):
        self._upload(tmp_path, "checkpoint-20")
        local = tmp_path / "local"
        local.mkdir()
        assert cm.maybe_restore_checkpoint(local) == str(local / "checkpoint-20")

    def test_maybe_restore_raises_instead_of_retraining(self, remote, tmp_path, monkeypatch):
        def bad_creds(uri):
            raise RuntimeError("bad creds")

        monkeypatch.setattr(cm, "remote_filesystem", bad_creds)
        with pytest.raises(RuntimeError, match="bad creds"):
            cm.maybe_restore_checkpoint(tmp_path)

    def test_noop_without_storage(self, monkeypatch, tmp_path):
        monkeypatch.delenv(UPLOAD_URI_ENV, raising=False)
        assert cm.maybe_restore_checkpoint(tmp_path) is None


class TestLauncherSync:
    def test_sync_latest_mirrors_native_layout_and_skips_unchanged(self, remote, tmp_path):
        ckpt = tmp_path / "full_state_checkpoints" / "step_9"
        ckpt.mkdir(parents=True)
        (ckpt / "training_state.pt").write_bytes(b"s")

        assert cm.sync_latest_checkpoint(tmp_path) == str(ckpt.resolve())
        assert remote.exists("full_state_checkpoints/step_9/.upload_complete")

        remote.rm_file("full_state_checkpoints/step_9/training_state.pt")
        cm.sync_latest_checkpoint(tmp_path)  # unchanged content: skipped
        assert not remote.exists("full_state_checkpoints/step_9/training_state.pt")

        (ckpt / "training_state.pt").write_bytes(b"ss")  # re-saved: mirrored again
        cm.sync_latest_checkpoint(tmp_path)
        assert remote.cat("full_state_checkpoints/step_9/training_state.pt") == b"ss"

    def test_instructlab_layout_recognized(self, tmp_path):
        done = tmp_path / "full_state" / "epoch_0"
        done.mkdir(parents=True)
        (done / "training_metadata.json").write_bytes(b"m")
        (tmp_path / "full_state" / "epoch_1").mkdir()  # metadata not written yet
        assert find_latest_valid_checkpoint(str(tmp_path)) == str(done.resolve())

    def test_noop_without_storage(self, monkeypatch, tmp_path):
        monkeypatch.delenv(UPLOAD_URI_ENV, raising=False)
        _make_checkpoint(tmp_path, "checkpoint-1")
        assert cm.sync_latest_checkpoint(tmp_path) is None

    @pytest.mark.parametrize(
        "node_rank,env,should_mirror",
        [
            (0, {}, True),
            (1, {}, False),                      # explicit torchrun arg
            (0, {"NODE_RANK": "1"}, False),      # get_torchrun_params ignores this
            (0, {"GROUP_RANK": "2"}, False),     # torchrun's own per-node var
            (0, {"NODE_RANK": "0"}, True),
        ],
    )
    def test_only_one_node_mirrors(
        self, remote, tmp_path, monkeypatch, node_rank, env, should_mirror
    ):
        for key in ("NODE_RANK", "GROUP_RANK"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        _make_checkpoint(tmp_path, "checkpoint-1")
        result = cm.sync_latest_checkpoint(tmp_path, node_rank=node_rank)
        assert (result is not None) is should_mirror
        assert remote.exists("checkpoint-1/.upload_complete") is should_mirror

    def test_best_effort_swallows_errors(self, remote, tmp_path, monkeypatch):
        monkeypatch.setattr(cm, "upload_checkpoint_now", _raise_oserror)
        _make_checkpoint(tmp_path, "checkpoint-1")
        cm.sync_latest_checkpoint_best_effort(tmp_path)  # must not raise
        with pytest.raises(OSError):
            cm.sync_latest_checkpoint(tmp_path)


class TestRemoteSyncCallback:
    def test_on_save_mirrors_from_main_process_only(self, remote, tmp_path):
        cb = RemoteCheckpointSyncCallback()
        ckpt = _make_checkpoint(tmp_path, "checkpoint-5")
        cb.on_save(_ctx(tmp_path, 5, ckpt, main=False))
        assert cm._uploader is None
        cb.on_save(_ctx(tmp_path, 5, ckpt, main=True))
        cb.on_train_end(_ctx(tmp_path, 5, ckpt))
        assert remote.exists("checkpoint-5/.upload_complete")

    def test_pending_upload_error_stops_training(self, remote, tmp_path, monkeypatch):
        def boom(fs, local_dir, prefix):
            raise OSError("bucket exploded")

        monkeypatch.setattr(cm, "_upload_dir", boom)
        cb = RemoteCheckpointSyncCallback()
        cb.on_save(_ctx(tmp_path, 5, _make_checkpoint(tmp_path, "checkpoint-5")))
        cm.shutdown_upload_worker(timeout=30)

        control = TrainingHubControl()
        cb.on_save(_ctx(tmp_path, 10, _make_checkpoint(tmp_path, "checkpoint-10"), control=control))
        assert control.should_training_stop is True
        with pytest.raises(RuntimeError, match="checkpoint-5"):
            cm.raise_pending_upload_error()


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

    def test_lora_train_verifies_remote_storage_at_entry(self, tmp_path, monkeypatch):
        from training_hub.algorithms.lora import LoRASFTAlgorithm

        monkeypatch.delenv(UPLOAD_URI_ENV, raising=False)
        _clear_memory_fs()
        backend = self._CaptureBackend()
        LoRASFTAlgorithm(backend).train(
            model_path="m",
            data_path="d",
            ckpt_output_dir=str(tmp_path),
            enable_jit_checkpoint=True,
            checkpoint_storage=REMOTE,
        )
        assert [type(c).__name__ for c in backend.params["callbacks"]] == [
            "JITCheckpointCallback",
            "RemoteCheckpointSyncCallback",
        ]
        with pytest.raises(ImportError, match="'gs' backend"):
            LoRASFTAlgorithm(backend).train(
                model_path="m",
                data_path="d",
                ckpt_output_dir=str(tmp_path),
                checkpoint_storage="gs://bucket/run1",
            )
