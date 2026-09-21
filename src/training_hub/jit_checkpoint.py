"""JIT (just-in-time) preemption checkpoint callback for HuggingFace-backed training."""

from __future__ import annotations

import logging
import signal
import threading

from training_hub.callbacks import TrainingHubCallback, TrainingHubContext
from training_hub.checkpoint_utils import mark_checkpoint_complete, mark_checkpoint_incomplete

logger = logging.getLogger(__name__)

# getsignal() returns None for a handler installed from C, so None cannot mean
# "nothing saved"; a private sentinel does.
_UNSET = object()
_PREEMPT_REQUESTED = False
_PREEMPT_SIGNUM: int | None = None
_PREEMPT_LOGGED = False
_PREEMPT_SAVE_REQUESTED = False
_ORIGINAL_SIGTERM_HANDLER: object = _UNSET


def preempt_requested() -> bool:
    """Return whether a preemption signal has been received."""
    return _PREEMPT_REQUESTED


def _handle_sigterm(signum: int, frame) -> None:  # noqa: ARG001
    # Set flags only. `logging` takes non-reentrant locks, so logging here
    # deadlocks the main thread whenever the signal lands while another thread
    # holds a handler lock — and then no checkpoint is ever saved.
    global _PREEMPT_REQUESTED, _PREEMPT_SIGNUM
    _PREEMPT_SIGNUM = signum
    _PREEMPT_REQUESTED = True


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
        _ORIGINAL_SIGTERM_HANDLER = _UNSET
        logger.exception("JIT checkpoint: failed to register SIGTERM handler")


def restore_preemption_handler() -> None:
    """Restore the SIGTERM handler that was active before registration."""
    global _ORIGINAL_SIGTERM_HANDLER
    if _ORIGINAL_SIGTERM_HANDLER is _UNSET:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    original = _ORIGINAL_SIGTERM_HANDLER
    try:
        signal.signal(
            signal.SIGTERM, signal.SIG_DFL if original is None else original
        )
    except (OSError, ValueError):
        logger.exception("JIT checkpoint: failed to restore SIGTERM handler")
    finally:
        _ORIGINAL_SIGTERM_HANDLER = _UNSET


class JITCheckpointCallback(TrainingHubCallback):
    """Save a full checkpoint on SIGTERM at the next step/epoch boundary.

    Uses ``TrainingHubControl`` (via ``context.control``) to request a
    HuggingFace ``TrainerControl.should_save`` / ``should_training_stop``.
    No constructor arguments — reads ``context.output_dir`` at hook time.

    Runs on every rank: the process-local SIGTERM flag is reduced across ranks
    (MAX) before any rank sets the control flags, so all ranks save and stop at
    the same step boundary.
    """

    run_on_all_ranks = True

    def on_train_begin(self, context: TrainingHubContext) -> None:
        global _PREEMPT_REQUESTED, _PREEMPT_LOGGED, _PREEMPT_SAVE_REQUESTED
        _PREEMPT_REQUESTED = False
        _PREEMPT_LOGGED = False
        _PREEMPT_SAVE_REQUESTED = False
        register_preemption_handler()

    def on_train_end(self, context: TrainingHubContext) -> None:
        restore_preemption_handler()

    def on_step_end(self, context: TrainingHubContext) -> None:
        self._handle_preemption(context)

    def on_epoch_end(self, context: TrainingHubContext) -> None:
        self._handle_preemption(context)

    def on_save(self, context: TrainingHubContext) -> None:
        # Clear the incomplete sidecar for the just-saved checkpoint.
        # Remote mirroring is owned by RemoteCheckpointSyncCallback, not this hook.
        if not context.is_main_process:
            return
        if context.output_dir and context.step > 0:
            mark_checkpoint_complete(context.output_dir, context.step)

    @staticmethod
    def _preempt_requested_any_rank() -> bool:
        """Aggregate the process-local SIGTERM flag across ranks (MAX), so a
        signal seen by one rank stops all ranks at the same step boundary."""
        from training_hub.checkpoint_manager import any_rank

        return any_rank(preempt_requested())

    @staticmethod
    def _log_preemption_once() -> None:
        """Emit the signal notice here rather than in the handler itself."""
        global _PREEMPT_LOGGED
        if _PREEMPT_LOGGED:
            return
        _PREEMPT_LOGGED = True
        logger.warning(
            "Received signal %s; checkpointing at this training step boundary.",
            _PREEMPT_SIGNUM,
        )

    def _handle_preemption(self, context: TrainingHubContext) -> None:
        global _PREEMPT_SAVE_REQUESTED
        if not self._preempt_requested_any_rank():
            return
        self._log_preemption_once()
        control = context.control
        if control is None:
            logger.error(
                "JIT checkpoint: preemption requested but no TrainingHubControl "
                "is attached to the callback context."
            )
            return

        # This runs from both on_step_end and on_epoch_end and the flag stays
        # set, so without a one-shot guard HF writes the same checkpoint twice
        # and the mirror uploads it twice — wasted grace-period seconds on a
        # multi-GB save. Keep asking to stop, ask to save only once.
        control.should_training_stop = True
        if _PREEMPT_SAVE_REQUESTED:
            return
        _PREEMPT_SAVE_REQUESTED = True

        if context.is_main_process and context.output_dir and context.step > 0:
            mark_checkpoint_incomplete(context.output_dir, context.step)

        control.should_save = True


class RemoteCheckpointSyncCallback(TrainingHubCallback):
    """Mirror every saved checkpoint to ``checkpoint_storage`` (any fsspec URI).

    Runs on every rank: ``on_save`` first waits at a barrier so all ranks have
    finished writing their shards, then rank 0 stages the checkpoint and queues
    the upload. Reads the URI from TRAINING_HUB_CHECKPOINT_UPLOAD_URI, so it
    needs no constructor arguments.
    """

    run_on_all_ranks = True

    def on_save(self, context: TrainingHubContext) -> None:
        import os

        from training_hub.checkpoint_manager import (
            any_rank,
            enqueue_checkpoint_upload,
            has_pending_upload_error,
            wait_for_all_ranks,
        )

        wait_for_all_ranks()
        # Only rank 0 owns an uploader, so the failure has to be reduced across
        # ranks before anyone acts on it: stopping on rank 0 alone would leave
        # the others blocked on the next collective.
        if any_rank(has_pending_upload_error()):
            # Adapters isolate hook exceptions, so raising here would be
            # swallowed; stop training and let the backend re-raise afterwards.
            logger.error("Checkpoint upload failed; stopping training.")
            if context.control is not None:
                context.control.should_training_stop = True
            return
        if not context.is_main_process:
            return
        checkpoint_path = context.metrics.get("checkpoint_path")
        if not checkpoint_path and context.output_dir and context.step > 0:
            checkpoint_path = os.path.join(
                context.output_dir, f"checkpoint-{context.step}"
            )
        if checkpoint_path and os.path.isdir(checkpoint_path):
            enqueue_checkpoint_upload(
                checkpoint_path, base_dir=context.output_dir or None
            )

    def on_train_end(self, context: TrainingHubContext) -> None:
        from training_hub.checkpoint_manager import shutdown_upload_worker

        shutdown_upload_worker()
