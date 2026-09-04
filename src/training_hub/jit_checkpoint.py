"""JIT (just-in-time) preemption checkpoint callback for HuggingFace-backed training."""

from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path

from training_hub.callbacks import TrainingHubCallback, TrainingHubContext
from training_hub.checkpoint_utils import mark_checkpoint_complete, mark_checkpoint_incomplete

logger = logging.getLogger(__name__)

_PREEMPT_REQUESTED = False
_ORIGINAL_SIGTERM_HANDLER: signal.Handlers | int | None = None


def preempt_requested() -> bool:
    """Return whether a preemption signal has been received."""
    return _PREEMPT_REQUESTED


def _handle_sigterm(signum: int, frame) -> None:  # noqa: ARG001
    global _PREEMPT_REQUESTED
    _PREEMPT_REQUESTED = True
    logger.warning(
        "Received signal %s; will checkpoint at the next training step boundary.",
        signum,
    )


def register_preemption_handler() -> None:
    """Register SIGTERM handler on the main thread."""
    global _ORIGINAL_SIGTERM_HANDLER
    if threading.current_thread() is not threading.main_thread():
        logger.warning(
            "JIT checkpoint: not on main thread; skipping signal registration."
        )
        return
    try:
        _ORIGINAL_SIGTERM_HANDLER = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except (OSError, ValueError):
        logger.exception("JIT checkpoint: failed to register SIGTERM handler")


def restore_preemption_handler() -> None:
    """Restore the original SIGTERM handler."""
    global _ORIGINAL_SIGTERM_HANDLER
    if _ORIGINAL_SIGTERM_HANDLER is None:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        signal.signal(signal.SIGTERM, _ORIGINAL_SIGTERM_HANDLER)
    except (OSError, ValueError):
        logger.exception("JIT checkpoint: failed to restore SIGTERM handler")
    finally:
        _ORIGINAL_SIGTERM_HANDLER = None


class JITCheckpointCallback(TrainingHubCallback):
    """Save a full checkpoint on SIGTERM at the next step/epoch boundary.

    Uses ``TrainingHubControl`` (via ``context.control``) to request a
    HuggingFace ``TrainerControl.should_save`` / ``should_training_stop``.
    No constructor arguments — reads ``context.output_dir`` at hook time.

    ponytail: ``run_on_all_ranks`` is set for future multi-GPU LoRA, but SIGTERM
    can land between ranks' ``on_step_end`` checks and cause mismatched save
    flags across processes. Unsloth LoRA is single-process today; a distributed
    fix would need a collective preemption barrier before ``should_save``.
    """

    run_on_all_ranks = True

    def on_train_begin(self, context: TrainingHubContext) -> None:
        global _PREEMPT_REQUESTED
        _PREEMPT_REQUESTED = False
        register_preemption_handler()

    def on_train_end(self, context: TrainingHubContext) -> None:
        restore_preemption_handler()

    def on_step_end(self, context: TrainingHubContext) -> None:
        self._handle_preemption(context)

    def on_epoch_end(self, context: TrainingHubContext) -> None:
        self._handle_preemption(context)

    def on_save(self, context: TrainingHubContext) -> None:
        # Clear the incomplete sidecar for the just-saved checkpoint.
        # S3 mirroring is owned by S3CheckpointSyncCallback, not this hook.
        if not context.is_main_process:
            return
        if context.output_dir and context.step > 0:
            mark_checkpoint_complete(context.output_dir, context.step)

    @staticmethod
    def _preempt_requested_any_rank() -> bool:
        """Aggregate the process-local SIGTERM flag across ranks (MAX), so a
        signal seen by one rank stops all ranks at the same step boundary."""
        flag = preempt_requested()
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
            logger.exception("JIT checkpoint: preemption rank-sync failed")
        return flag

    def _handle_preemption(self, context: TrainingHubContext) -> None:
        if not self._preempt_requested_any_rank():
            return
        control = context.control
        if control is None:
            logger.error(
                "JIT checkpoint: preemption requested but no TrainingHubControl "
                "is attached to the callback context."
            )
            return

        if context.is_main_process and context.output_dir and context.step > 0:
            mark_checkpoint_incomplete(context.output_dir, context.step)

        control.should_save = True
        control.should_training_stop = True


class S3CheckpointSyncCallback(TrainingHubCallback):
    """Mirror every saved checkpoint to S3 (checkpoint_storage="s3://...").

    Serialization-safe for torchrun backends: no constructor arguments —
    the S3 URI is read from TRAINING_HUB_CHECKPOINT_UPLOAD_URI inside hooks.
    Uploads happen on rank 0 only, via the background upload queue.
    """

    def on_save(self, context: TrainingHubContext) -> None:
        from training_hub.checkpoint_manager import enqueue_checkpoint_upload

        checkpoint_path = context.metrics.get("checkpoint_path")
        if not checkpoint_path and context.output_dir and context.step > 0:
            candidate = f"{context.output_dir}/checkpoint-{context.step}"
            import os

            if os.path.isdir(candidate):
                checkpoint_path = candidate
        if checkpoint_path:
            enqueue_checkpoint_upload(checkpoint_path, base_dir=context.output_dir or None)

    def on_train_end(self, context: TrainingHubContext) -> None:
        from training_hub.checkpoint_manager import shutdown_upload_worker

        shutdown_upload_worker()
