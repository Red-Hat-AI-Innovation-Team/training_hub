"""Background checkpoint upload queue (S3-compatible object storage).

Set ``TRAINING_HUB_CHECKPOINT_UPLOAD_URI`` to enqueue each completed checkpoint
for background upload after JIT save:

* ``s3://bucket/prefix`` — uses a lazy ``boto3`` import (optional dependency;
  install boto3 in the runtime image if S3 upload is required).
* Any other URI — treated as a local directory; checkpoints are copied there.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_UPLOAD_QUEUE: queue.LifoQueue[str | None] | None = None
_UPLOAD_THREAD: threading.Thread | None = None
_UPLOAD_LOCK = threading.Lock()


def _upload_uri() -> str | None:
    return os.environ.get("TRAINING_HUB_CHECKPOINT_UPLOAD_URI")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Unsupported upload URI (expected s3://): {uri}")
    without_scheme = uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, prefix.strip("/")


def _upload_directory_to_s3(local_dir: Path, uri: str) -> None:
    import boto3

    bucket, prefix = _parse_s3_uri(uri)
    client = boto3.client("s3")
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        client.upload_file(str(path), bucket, key)


def _upload_worker() -> None:
    assert _UPLOAD_QUEUE is not None
    uri = _upload_uri()
    if not uri:
        return

    while True:
        local_path = _UPLOAD_QUEUE.get()
        try:
            if local_path is None:
                return
            path = Path(local_path)
            if not path.exists():
                continue
            if uri.startswith("s3://"):
                _upload_directory_to_s3(path, uri)
            else:
                dest = Path(uri) / path.name
                if path.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(path, dest)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
            logger.info("Uploaded checkpoint %s to %s", path, uri)
        except Exception:
            logger.exception("Failed to upload checkpoint %s", local_path)
        finally:
            _UPLOAD_QUEUE.task_done()


def ensure_upload_worker_started() -> None:
    """Start the background uploader when an upload URI is configured."""
    global _UPLOAD_QUEUE, _UPLOAD_THREAD
    if not _upload_uri():
        return
    with _UPLOAD_LOCK:
        if _UPLOAD_THREAD is not None:
            return
        _UPLOAD_QUEUE = queue.LifoQueue()
        _UPLOAD_THREAD = threading.Thread(
            target=_upload_worker,
            name="training-hub-checkpoint-upload",
            daemon=True,
        )
        _UPLOAD_THREAD.start()


def enqueue_checkpoint_upload(checkpoint_dir: str | Path) -> None:
    """Queue a checkpoint directory for background upload (LIFO)."""
    if not _upload_uri():
        return
    ensure_upload_worker_started()
    assert _UPLOAD_QUEUE is not None
    _UPLOAD_QUEUE.put(str(Path(checkpoint_dir).resolve()))


def shutdown_upload_worker(timeout: float = 30.0) -> None:
    """Signal the upload worker to exit and wait for completion."""
    global _UPLOAD_QUEUE, _UPLOAD_THREAD
    with _UPLOAD_LOCK:
        if _UPLOAD_QUEUE is None or _UPLOAD_THREAD is None:
            return
        _UPLOAD_QUEUE.put(None)
        thread = _UPLOAD_THREAD
        _UPLOAD_THREAD = None
        _UPLOAD_QUEUE = None
    thread.join(timeout=timeout)
