"""Checkpoint discovery and incomplete-sentinel helpers for JIT resume."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Legacy in-dir sentinel (still honored on read for backward compatibility).
INCOMPLETE_SENTINEL = ".incomplete"
INCOMPLETE_SIDECAR_PREFIX = ".incomplete-checkpoint-"

_HF_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_MINI_TRAINER_STEP_RE = re.compile(r"^step_(\d+)$")


def incomplete_sidecar_path(output_dir: str | Path, step: int) -> Path:
    """Return the sidecar file marking an in-progress HF checkpoint at *step*."""
    return Path(output_dir) / f"{INCOMPLETE_SIDECAR_PREFIX}{step}"


def mark_checkpoint_incomplete(output_dir: str | Path, step: int) -> None:
    """Mark a checkpoint step as in-progress without pre-creating ``checkpoint-{step}``.

    Uses a sidecar file in *output_dir* so HuggingFace Trainer can create the
    checkpoint directory itself (some transformers versions rename into a new dir).
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    incomplete_sidecar_path(root, step).touch()


def mark_checkpoint_complete(output_dir: str | Path, step: int) -> None:
    """Clear incomplete markers for a finished checkpoint at *step*."""
    root = Path(output_dir)
    incomplete_sidecar_path(root, step).unlink(missing_ok=True)
    legacy = root / f"checkpoint-{step}" / INCOMPLETE_SENTINEL
    try:
        legacy.unlink(missing_ok=True)
    except OSError:
        pass


def is_valid_checkpoint_dir(
    checkpoint_dir: Path,
    output_dir: Path | None = None,
) -> bool:
    """Return True when the directory exists and has no incomplete sentinel."""
    if not checkpoint_dir.is_dir():
        return False
    if (checkpoint_dir / INCOMPLETE_SENTINEL).exists():
        return False
    if output_dir is not None:
        step = _checkpoint_step(checkpoint_dir, _HF_CHECKPOINT_RE)
        if step is not None and incomplete_sidecar_path(output_dir, step).exists():
            return False
    return True


def _checkpoint_step(path: Path, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def find_latest_valid_checkpoint(output_dir: str | None) -> str | None:
    """Return the newest valid checkpoint path under *output_dir*, if any.

    Supports HuggingFace ``checkpoint-{step}`` dirs and Mini-Trainer
    ``full_state_checkpoints/step_{n}`` dirs. Directories containing
    ``.incomplete`` or a matching ``.incomplete-checkpoint-{step}`` sidecar
    are skipped.
    """
    if not output_dir:
        return None

    root = Path(output_dir)
    if not root.is_dir():
        return None

    candidates: list[tuple[int, str]] = []

    for child in root.iterdir():
        if not child.is_dir() or not is_valid_checkpoint_dir(child, root):
            continue
        step = _checkpoint_step(child, _HF_CHECKPOINT_RE)
        if step is not None:
            candidates.append((step, str(child.resolve())))

    mini_root = root / "full_state_checkpoints"
    if mini_root.is_dir():
        for child in mini_root.iterdir():
            if not child.is_dir() or not is_valid_checkpoint_dir(child):
                continue
            if not (child / "training_state.pt").exists():
                continue
            step = _checkpoint_step(child, _MINI_TRAINER_STEP_RE)
            if step is not None:
                candidates.append((step, str(child.resolve())))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def jit_checkpoint_enabled(
    enable_jit_checkpoint: bool | None,
    ckpt_output_dir: str | None,
) -> bool:
    """Return True when explicit JIT opt-in and output dir are both set."""
    return bool(enable_jit_checkpoint and ckpt_output_dir)


def apply_native_jit_params(
    params: dict[str, object],
    *,
    enable_jit_checkpoint: bool | None,
    backend: str,
) -> None:
    """Enable backend-native on-demand checkpointing for SFT/OSFT.

    InstructLab SFT workers call ``load_latest_full_state()`` on startup and
    resume from ``{ckpt_output_dir}/full_state`` when on-demand checkpoints
    exist (no extra resume parameter from Training Hub). Mini-Trainer OSFT has
    the same auto-resume behavior for ``full_state_checkpoints/``.
    """
    if not (enable_jit_checkpoint and backend in {"sft", "osft"}):
        return

    if params.get("on_demand_checkpointing") is False:
        logger.warning(
            "enable_jit_checkpoint=True overrides on_demand_checkpointing=False "
            "for backend %s",
            backend,
        )
    params["on_demand_checkpointing"] = True


UPLOAD_URI_ENV = "TRAINING_HUB_CHECKPOINT_UPLOAD_URI"


def resolve_checkpoint_storage(checkpoint_storage: str | None) -> str | None:
    """Validate the checkpoint_storage selector and return the S3 URI, if any.

    Accepted values: None / "pvc" (filesystem only, the default) or an
    "s3://bucket/prefix" URI (mirror checkpoints to S3, restore on resume).
    """
    if checkpoint_storage in (None, "", "pvc"):
        return None
    if isinstance(checkpoint_storage, str) and checkpoint_storage.startswith("s3://"):
        return checkpoint_storage
    raise ValueError(
        "checkpoint_storage must be None, 'pvc', or an 's3://bucket/prefix' "
        f"URI; got {checkpoint_storage!r}"
    )


def apply_checkpoint_storage_env(checkpoint_storage: str | None) -> None:
    """Export the S3 upload URI so callbacks and torchrun workers inherit it."""
    import os

    uri = resolve_checkpoint_storage(checkpoint_storage)
    if uri:
        os.environ[UPLOAD_URI_ENV] = uri
