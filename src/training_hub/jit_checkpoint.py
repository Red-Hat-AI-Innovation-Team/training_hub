"""JIT (just-in-time) preemption checkpoint callback for HuggingFace-backed training."""

from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path

from training_hub.callbacks import TrainingHubCallback, TrainingHubContext
from training_hub.checkpoint_manager import enqueue_checkpoint_upload
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
        if not context.output_dir or context.step <= 0:
            return

        if context.is_main_process:
            mark_checkpoint_complete(context.output_dir, context.step)

        checkpoint_path = context.metrics.get("checkpoint_path")
        if checkpoint_path and context.is_main_process:
            enqueue_checkpoint_upload(checkpoint_path)
            return

        if context.is_main_process:
            ckpt_dir = Path(context.output_dir) / f"checkpoint-{context.step}"
            if ckpt_dir.is_dir():
                enqueue_checkpoint_upload(ckpt_dir)

    def _handle_preemption(self, context: TrainingHubContext) -> None:
        if not preempt_requested():
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
