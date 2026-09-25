"""Remote checkpoint mirroring and restore for ``checkpoint_storage="<scheme>://..."``.

Any fsspec-supported URI works (``s3://``, ``gs://``, ``abfs://``, ``file://``);
S3 needs ``pip install 'training-hub[s3]'``. The URI is read from
``TRAINING_HUB_CHECKPOINT_UPLOAD_URI`` so torchrun workers inherit it.

The remote layout mirrors ``output_dir`` (``<uri>/checkpoint-10/...``,
``<uri>/full_state_checkpoints/step_9/...``). Every upload finishes by writing
an ``.upload_complete`` marker; restore only considers prefixes carrying it.

Uploads from inside the training loop (lora_sft) are staged with a hardlink
copy and pushed by a background worker; failures are surfaced to the caller
through ``raise_pending_upload_error``. sft/osft mirror their on-demand
checkpoint synchronously from the launcher process after the workers exit.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from training_hub.checkpoint_utils import (
    ALL_LAYOUTS,
    UPLOAD_URI_ENV,
    clear_stale_incomplete_marker,
    find_latest_valid_checkpoint,
    is_valid_checkpoint_dir,
)

logger = logging.getLogger(__name__)

COMPLETE_MARKER = ".upload_complete"
STAGING_DIR = ".upload_staging"
RESTORE_TMP_DIR = ".restore_tmp"
UPLOAD_WORKERS = 4
UPLOAD_ATTEMPTS = 3
SHUTDOWN_TIMEOUT = 3600.0

_TRAILING_INT_RE = re.compile(r"(\d+)$")


def storage_uri() -> str | None:
    return os.environ.get(UPLOAD_URI_ENV) or None


def remote_filesystem(uri: str):
    """fsspec filesystem rooted at *uri*, so every path is relative to it."""
    import fsspec

    protocol, sep, base_path = uri.partition("://")
    if not sep or not protocol or not base_path:
        raise ValueError(
            f"checkpoint_storage must look like '<scheme>://bucket/prefix'; got {uri!r}"
        )
    kwargs: dict = {}
    if protocol == "s3":
        endpoint = (
            os.environ.get("AWS_ENDPOINT_URL_S3")
            or os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("AWS_S3_ENDPOINT")
        )
        if endpoint:
            kwargs["client_kwargs"] = {"endpoint_url": endpoint}
    try:
        underlying = fsspec.filesystem(protocol, **kwargs)
    except ImportError as e:
        raise ImportError(
            f"checkpoint_storage={uri!r} needs the fsspec '{protocol}' backend "
            f"({e}). For S3: pip install 'training-hub[s3]'"
        ) from e
    return fsspec.filesystem("dir", path=base_path, fs=underlying)


def verify_storage_access(uri: str) -> None:
    """Fail at train() entry rather than hours later in a background thread."""
    fs = remote_filesystem(uri)
    # Unique per caller: every rank runs this against the same prefix, and a
    # shared key would let one rank delete another's probe mid-check.
    probe = f".training-hub-access-test.{uuid.uuid4().hex}"
    try:
        fs.pipe(probe, b"ok")
        fs.cat(probe)
    except Exception as e:
        raise RuntimeError(
            f"checkpoint_storage={uri!r} is not writable: {e}. Check credentials, "
            "the endpoint (AWS_ENDPOINT_URL_S3) and bucket permissions."
        ) from e
    finally:
        try:
            fs.rm_file(probe)
        except Exception:
            logger.debug("Could not remove access probe %s", probe)


def wait_for_all_ranks() -> None:
    """Barrier when torch.distributed is initialized; no-op otherwise."""
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def any_rank(flag: bool) -> bool:
    """Reduce a local boolean across ranks (MAX) so every rank agrees.

    Returns the local value when torch.distributed is not in play. Callers must
    invoke this on every rank — it is a collective.
    """
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            t = torch.tensor([1 if flag else 0], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return bool(t.item())
    except Exception:
        logger.exception("Rank sync failed; falling back to the local value")
    return flag


def _is_local_rank_zero() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _is_node_zero(node_rank: int = 0) -> bool:
    """Whether this launcher is the one that should mirror to remote storage.

    ``get_torchrun_params`` only reads ``PET_NODE_RANK``, so the passed value is
    0 on every node of a job that advertises ``NODE_RANK``/``GROUP_RANK``
    instead. Requiring both to say zero keeps exactly one node uploading.
    """
    if int(node_rank) != 0:
        return False
    env = os.environ.get("NODE_RANK") or os.environ.get("GROUP_RANK") or "0"
    return env.strip() in ("", "0")


def _relative_prefix(checkpoint_dir: Path, base_dir: Path) -> str:
    try:
        return checkpoint_dir.relative_to(base_dir).as_posix()
    except ValueError:
        return checkpoint_dir.name


def _fingerprint(local_dir: Path) -> bytes:
    size = newest = 0
    for path in local_dir.rglob("*"):
        if path.is_file():
            st = path.stat()
            size += st.st_size
            newest = max(newest, st.st_mtime_ns)
    return f"{size}:{newest}".encode()


def _upload_dir(fs, local_dir: Path, remote_prefix: str) -> None:
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    targets = {
        p: str(PurePosixPath(remote_prefix) / p.relative_to(local_dir).as_posix())
        for p in files
    }
    for parent in {str(PurePosixPath(r).parent) for r in targets.values()}:
        try:
            fs.makedirs(parent, exist_ok=True)
        except Exception:  # object stores have no directories
            pass

    def put(path: Path) -> None:
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            try:
                fs.put_file(str(path), targets[path])
                return
            except Exception:
                if attempt == UPLOAD_ATTEMPTS:
                    raise
                time.sleep(attempt)

    if files:
        with ThreadPoolExecutor(max_workers=min(UPLOAD_WORKERS, len(files))) as pool:
            list(pool.map(put, files))  # re-raises the first failure
    # written last: restore only considers checkpoints that carry it
    fs.pipe(f"{remote_prefix}/{COMPLETE_MARKER}", _fingerprint(local_dir))


class _Uploader:
    """Background LIFO uploader owning its queue and thread.

    Shutdown is an Event checked between polls, so the backlog drains newest
    first instead of being dropped by an in-band sentinel. The worker never
    reads module globals, so it cannot race a shutdown.
    """

    def __init__(self, fs) -> None:
        self.fs = fs
        self.queue: queue.LifoQueue = queue.LifoQueue()
        self.stop = threading.Event()
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run, name="training-hub-checkpoint-upload", daemon=False
        )
        self.thread.start()

    def _run(self) -> None:
        while True:
            try:
                local_dir, remote_prefix, cleanup = self.queue.get(timeout=1.0)
            except queue.Empty:
                if self.stop.is_set():
                    return
                continue
            try:
                _upload_dir(self.fs, Path(local_dir), remote_prefix)
                if cleanup:
                    shutil.rmtree(local_dir, ignore_errors=True)
                logger.warning(
                    "Mirrored checkpoint %s to %s", remote_prefix, storage_uri()
                )
            except Exception as e:
                logger.exception("Checkpoint upload failed: %s", remote_prefix)
                with self._lock:
                    self._error = RuntimeError(
                        f"Background checkpoint upload failed for {remote_prefix}: {e}"
                    )
                    self._error.__cause__ = e
            finally:
                self.queue.task_done()

    def peek_error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def take_error(self) -> BaseException | None:
        with self._lock:
            error, self._error = self._error, None
        return error


_uploader: _Uploader | None = None
_uploader_lock = threading.Lock()


def _get_uploader() -> _Uploader | None:
    global _uploader
    uri = storage_uri()
    if not uri:
        return None
    with _uploader_lock:
        if _uploader is None or not _uploader.thread.is_alive():
            _uploader = _Uploader(remote_filesystem(uri))
        return _uploader


def enqueue_checkpoint_upload(
    checkpoint_dir: str | Path, base_dir: str | Path | None = None
) -> None:
    """Stage *checkpoint_dir* and queue it for background upload (newest first).

    The staging copy is made with hardlinks (instant, no extra data), so
    ``save_total_limit`` rotation deleting the original cannot truncate the
    upload, and the original stays in place for local resume. Falls back to a
    real copy on filesystems without hardlinks. The remote key preserves the
    path relative to *base_dir* so nested layouts restore correctly.
    """
    uploader = _get_uploader()
    if uploader is None:
        return
    src = Path(checkpoint_dir).resolve()
    base = Path(base_dir).resolve() if base_dir else src.parent
    rel = _relative_prefix(src, base)
    staged = base / STAGING_DIR / rel
    shutil.rmtree(staged, ignore_errors=True)
    staged.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(src, staged, copy_function=os.link)
    except OSError:
        shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(src, staged)
    uploader.queue.put((str(staged), rel, True))


def has_pending_upload_error() -> bool:
    with _uploader_lock:
        uploader = _uploader
    return uploader is not None and uploader.peek_error() is not None


def raise_pending_upload_error() -> None:
    """Re-raise a background upload failure on the calling thread (consumes it)."""
    with _uploader_lock:
        uploader = _uploader
    if uploader is not None:
        error = uploader.take_error()
        if error is not None:
            raise error


def shutdown_upload_worker(timeout: float = SHUTDOWN_TIMEOUT) -> None:
    """Drain queued uploads (newest first) and stop the worker.

    A pending failure is kept for ``raise_pending_upload_error``.
    """
    with _uploader_lock:
        uploader = _uploader
    if uploader is None or not uploader.thread.is_alive():
        return
    uploader.stop.set()
    uploader.thread.join(timeout=timeout)
    if uploader.thread.is_alive():
        logger.warning(
            "Checkpoint upload worker still running after %.0fs; the remote copy "
            "stays unusable for resume until its %s marker is written.",
            timeout,
            COMPLETE_MARKER,
        )


def upload_checkpoint_now(
    checkpoint_dir: str | Path, base_dir: str | Path | None = None
) -> None:
    """Upload synchronously from the launcher process.

    No staging: the local copy stays for the backend's native auto-resume.
    Skips a checkpoint whose remote marker already matches the local content.
    """
    uri = storage_uri()
    if not uri:
        return
    src = Path(checkpoint_dir).resolve()
    base = Path(base_dir).resolve() if base_dir else src.parent
    rel = _relative_prefix(src, base)
    fs = remote_filesystem(uri)
    marker = f"{rel}/{COMPLETE_MARKER}"
    if fs.exists(marker) and fs.cat(marker) == _fingerprint(src):
        logger.info("Checkpoint %s already mirrored to %s", rel, uri)
        return
    _upload_dir(fs, src, rel)
    logger.warning("Mirrored checkpoint %s to %s", rel, uri)


def sync_latest_checkpoint(
    output_dir: str | Path,
    node_rank: int = 0,
    layouts: tuple[str, ...] = ALL_LAYOUTS,
) -> str | None:
    """Mirror the newest valid local checkpoint under *output_dir*.

    Used by the sft/osft launchers: their on-demand (SIGTERM) save happens in
    torchrun workers that exit without firing ``on_save``. Pass *layouts* naming
    the layout that backend writes. Returns the local path mirrored, or None.
    """
    if not storage_uri() or not _is_node_zero(node_rank):
        return None
    latest = find_latest_valid_checkpoint(str(output_dir), layouts=layouts)
    if latest is None:
        return None
    upload_checkpoint_now(latest, base_dir=output_dir)
    return latest


def sync_latest_checkpoint_best_effort(
    output_dir: str | Path,
    node_rank: int = 0,
    layouts: tuple[str, ...] = ALL_LAYOUTS,
) -> None:
    """Mirror on a failure path without masking the exception being propagated."""
    try:
        sync_latest_checkpoint(output_dir, node_rank=node_rank, layouts=layouts)
    except Exception:
        logger.exception("Could not mirror the checkpoint after a training failure")


def _checkpoint_order(prefix: str) -> int:
    match = _TRAILING_INT_RE.search(PurePosixPath(prefix).name)
    return int(match.group(1)) if match else -1


def _complete_checkpoints(fs) -> list[str]:
    try:
        keys = fs.find("")
    except FileNotFoundError:
        return []
    suffix = "/" + COMPLETE_MARKER
    names = [key[: -len(suffix)] for key in keys if key.endswith(suffix)]
    return sorted(names, key=_checkpoint_order, reverse=True)


def _safe_target(root: Path, rel: str) -> Path:
    parts = PurePosixPath(rel).parts
    if PurePosixPath(rel).is_absolute() or ".." in parts:
        raise ValueError(
            f"Refusing to restore an object outside the checkpoint directory: {rel!r}"
        )
    target = root.joinpath(*parts)
    if root.resolve() not in target.resolve().parents:
        raise ValueError(
            f"Refusing to restore an object outside the checkpoint directory: {rel!r}"
        )
    return target


def restore_latest_checkpoint(uri: str, local_dir: str | Path) -> str | None:
    """Download the newest complete checkpoint under *uri* into *local_dir*.

    Files land in a temporary directory and are moved into place only once the
    download finished, so an interrupted restore never looks resumable.
    Returns the local checkpoint path, or None when nothing complete exists.
    """
    fs = remote_filesystem(uri)
    local = Path(local_dir).resolve()
    for name in _complete_checkpoints(fs):
        dest = local / name
        # Only a *valid* local copy makes the download unnecessary. A partial
        # one (interrupted save, stale sentinel) must be replaced, otherwise we
        # would keep it and silently retrain from step 0.
        if is_valid_checkpoint_dir(dest, local):
            logger.warning(
                "Checkpoint %s already present locally; skipping download", dest
            )
            return str(dest)
        tmp_root = local / f"{RESTORE_TMP_DIR}-{os.getpid()}"
        tmp = tmp_root / name
        shutil.rmtree(tmp_root, ignore_errors=True)
        tmp.mkdir(parents=True)
        # Announced before the transfer: a multi-GB checkpoint takes minutes and
        # would otherwise look like a hang.
        keys = [
            k
            for k in fs.find(name)
            if k.startswith(name + "/") and k[len(name) + 1:] != COMPLETE_MARKER
        ]
        try:
            total_mb = sum(fs.size(k) or 0 for k in keys) / (1024 * 1024)
        except Exception:
            total_mb = 0
        logger.warning(
            "Restoring checkpoint %s from %s (%d files, %.0f MB) into %s",
            name, uri, len(keys), total_mb, dest,
        )
        try:
            for key in keys:
                rel = key[len(name) + 1:]
                if not rel:
                    continue
                target = _safe_target(tmp, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                fs.get_file(key, str(target))
            dest.parent.mkdir(parents=True, exist_ok=True)
            if is_valid_checkpoint_dir(dest, local):
                return str(dest)  # another launcher on a shared volume won
            # Swap in only now that a verified-complete download is in hand.
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(tmp), str(dest))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
        clear_stale_incomplete_marker(local, dest)
        # WARNING, not INFO: nothing configures logging in a training pod, so
        # INFO is invisible and the operator cannot tell a resume from a silent
        # restart at step 0 — the one thing this feature exists to prevent.
        logger.warning("Restored checkpoint %s from %s -> %s", name, uri, dest)
        return str(dest)
    return None


def maybe_restore_checkpoint(
    local_dir: str | Path, layouts: tuple[str, ...] = ALL_LAYOUTS
) -> str | None:
    """Restore from remote storage when configured and *local_dir* holds no
    valid checkpoint. Local rank 0 downloads; other ranks wait at the barrier.

    *layouts* must name the layout the caller consumes. Gating on every layout
    lets an unrelated leftover — say a native ``full_state/`` dir in an
    output_dir now used for LoRA — count as "already have one", skip the
    download, and leave the caller resuming from nothing.

    Raises on failure: silently retraining from step 0 is the one outcome this
    feature exists to prevent.
    """
    uri = storage_uri()
    if not uri:
        return None
    restored = None
    if _is_local_rank_zero():
        local_existing = find_latest_valid_checkpoint(str(local_dir), layouts=layouts)
        if local_existing is None:
            restored = restore_latest_checkpoint(uri, local_dir)
            if restored is None:
                logger.warning(
                    "No complete checkpoint under %s; training starts from step 0", uri
                )
        else:
            logger.warning(
                "Local checkpoint %s already present; not restoring from %s",
                local_existing,
                uri,
            )
    wait_for_all_ranks()
    return restored
