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

_UPLOAD_QUEUE: queue.LifoQueue[tuple[str, str | None] | None] | None = None
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


def _s3_client():
    """boto3 S3 client honoring AWS_ENDPOINT_URL(_S3) for MinIO/on-prem."""
    import boto3

    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get(
        "AWS_ENDPOINT_URL"
    )
    return boto3.client("s3", endpoint_url=endpoint)


def _upload_directory_to_s3(local_dir: Path, uri: str) -> None:
    bucket, prefix = _parse_s3_uri(uri)
    client = _s3_client()
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        client.upload_file(str(path), bucket, key)
    # marker written last: restore only considers checkpoints that carry it
    marker = f"{prefix}/.upload_complete" if prefix else ".upload_complete"
    client.put_object(Bucket=bucket, Key=marker, Body=b"")


def _upload_worker() -> None:
    assert _UPLOAD_QUEUE is not None
    uri = _upload_uri()
    if not uri:
        return

    while True:
        item = _UPLOAD_QUEUE.get()
        try:
            if item is None:
                return
            local_path, base_dir = item
            path = Path(local_path)
            if not path.exists():
                continue
            # preserve layout relative to base_dir (e.g. full_state_checkpoints/step_9)
            rel_prefix = path.name
            if base_dir:
                try:
                    rel_prefix = path.resolve().relative_to(
                        Path(base_dir).resolve()
                    ).as_posix()
                except ValueError:
                    pass
            if uri.startswith("s3://"):
                _upload_directory_to_s3(path, f"{uri.rstrip('/')}/{rel_prefix}")
            else:
                dest = Path(uri) / rel_prefix
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


def enqueue_checkpoint_upload(
    checkpoint_dir: str | Path, base_dir: str | Path | None = None
) -> None:
    """Queue a checkpoint directory for background upload (LIFO).

    base_dir: when given, the S3 key prefix preserves the checkpoint's path
    relative to it (so nested layouts restore correctly).
    """
    if not _upload_uri():
        return
    ensure_upload_worker_started()
    assert _UPLOAD_QUEUE is not None
    _UPLOAD_QUEUE.put(
        (str(Path(checkpoint_dir).resolve()),
         str(Path(base_dir).resolve()) if base_dir else None)
    )


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


def restore_latest_from_s3(uri: str, local_dir: str | Path) -> str | None:
    """Download the latest complete checkpoint under *uri* into *local_dir*.

    Checkpoints are laid out as ``<uri>/<checkpoint_name>/...`` with an
    ``.upload_complete`` marker written last; prefixes without the marker
    are skipped. Returns the local checkpoint path, or None when the bucket
    holds no complete checkpoint.
    """
    bucket, prefix = _parse_s3_uri(uri)
    client = _s3_client()

    paginator = client.get_paginator("list_objects_v2")
    base = f"{prefix}/" if prefix else ""
    marker_suffix = "/.upload_complete"
    names: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=base):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(marker_suffix):
                names.append(key[len(base):-len(marker_suffix)])

    def _step(name: str) -> int:
        digits = "".join(ch for ch in name.rsplit("/", 1)[-1] if ch.isdigit())
        return int(digits) if digits else -1

    for name in sorted(names, key=_step, reverse=True):
        dest = Path(local_dir) / name
        dest.mkdir(parents=True, exist_ok=True)
        ckpt_prefix = f"{base}{name}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=ckpt_prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(ckpt_prefix):]
                if not rel or rel == ".upload_complete":
                    continue
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, obj["Key"], str(target))
        logger.info("Restored checkpoint from S3: %s -> %s", name, dest)
        return str(dest)

    return None


def maybe_restore_from_s3(local_dir: str | Path) -> str | None:
    """Restore latest S3 checkpoint when the upload URI is set and *local_dir*
    holds no checkpoints yet. No-op (returns None) otherwise."""
    uri = _upload_uri()
    if not uri or not uri.startswith("s3://"):
        return None
    local = Path(local_dir)
    if local.is_dir() and any(
        child.is_dir() and not child.name.startswith(".")
        and child.name not in ("data", "hf_cache", "_internal_data_processing")
        for child in local.iterdir()
    ):
        return None  # local checkpoints present; native/local resume handles it
    try:
        return restore_latest_from_s3(uri, local)
    except Exception:
        logger.exception("Failed to restore checkpoint from S3 (%s)", uri)
        return None
