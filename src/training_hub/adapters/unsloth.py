"""Unsloth/HuggingFace adapter for TrainingHubCallback.

Translates TrainingHubCallback instances into HuggingFace TrainerCallback
objects that can be passed to SFTTrainer.add_callback().

Note:
    Trainer.add_callback() requires actual TrainerCallback subclasses,
    not duck-typed objects. UnslothCallbackAdapter inherits from
    TrainerCallback to satisfy this requirement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from transformers import TrainerCallback

from training_hub.callbacks import TrainingHubCallback, TrainingHubContext

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class UnslothCallbackAdapter(TrainerCallback):
    """Adapts a TrainingHubCallback to HuggingFace's TrainerCallback interface.

    Wraps a single TrainingHubCallback, building a normalized
    TrainingHubContext from HuggingFace's TrainerState on each event
    and dispatching to the user's hook.

    Exception isolation: user callback exceptions are caught and logged,
    never propagated — callbacks cannot crash training.

    Rank-0 guard: callbacks only fire on the main process
    (state.is_world_process_zero).

    Args:
        hub_callback: A TrainingHubCallback instance to adapt.
    """

    def __init__(self, hub_callback: TrainingHubCallback) -> None:
        super().__init__()
        self._hub_callback = hub_callback

    @staticmethod
    def _build_context(
        args: TrainingArguments,
        state: TrainerState,
        logs: dict | None = None,
    ) -> TrainingHubContext:
        """Build normalized context from HuggingFace trainer state."""
        metrics = dict(logs) if logs else {}

        loss = metrics.get("loss")
        if loss is None:
            loss = metrics.get("eval_loss")
        if loss is None and state.log_history:
            # Prefer most recent training loss; otherwise last eval_loss
            eval_loss_fallback = None
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    loss = entry["loss"]
                    break
                if eval_loss_fallback is None and "eval_loss" in entry:
                    eval_loss_fallback = entry["eval_loss"]
            if loss is None:
                loss = eval_loss_fallback

        learning_rate = metrics.get("learning_rate")
        if learning_rate is None and state.log_history:
            for entry in reversed(state.log_history):
                if "learning_rate" in entry:
                    learning_rate = entry["learning_rate"]
                    break

        return TrainingHubContext(
            step=state.global_step,
            epoch=int(state.epoch) if state.epoch is not None else 0,
            loss=loss,
            learning_rate=learning_rate,
            is_main_process=state.is_world_process_zero,
            output_dir=args.output_dir,
            metrics=metrics,
        )

    def _safe_call(
        self,
        method_name: str,
        args: TrainingArguments,
        state: TrainerState,
        logs: dict | None = None,
    ) -> None:
        """Dispatch to user callback with exception isolation and rank guard."""
        if not state.is_world_process_zero:
            return
        try:
            ctx = self._build_context(args, state, logs)
            getattr(self._hub_callback, method_name)(ctx)
        except Exception:
            logger.exception(
                "TrainingHubCallback.%s raised an exception (ignored)",
                method_name,
            )

    # --- HuggingFace TrainerCallback hooks → TrainingHubCallback hooks ---

    def on_train_begin(self, args, state, control, **kwargs):
        self._safe_call("on_train_begin", args, state)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._safe_call("on_epoch_begin", args, state)

    def on_step_begin(self, args, state, control, **kwargs):
        self._safe_call("on_step_begin", args, state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._safe_call("on_log", args, state, logs=logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # HF Trainer passes evaluation metrics via the `metrics` kwarg
        self._safe_call("on_evaluate", args, state, logs=metrics)

    def on_save(self, args, state, control, **kwargs):
        self._safe_call("on_save", args, state)

    def on_step_end(self, args, state, control, **kwargs):
        self._safe_call("on_step_end", args, state)

    def on_epoch_end(self, args, state, control, **kwargs):
        self._safe_call("on_epoch_end", args, state)

    def on_train_end(self, args, state, control, **kwargs):
        self._safe_call("on_train_end", args, state)


def adapt_hub_callbacks(
    callbacks: list[TrainingHubCallback],
) -> list[TrainerCallback]:
    """Convert a list of TrainingHubCallbacks to HuggingFace TrainerCallbacks.

    Args:
        callbacks: List of TrainingHubCallback instances.

    Returns:
        List of UnslothCallbackAdapter instances ready for
        trainer.add_callback().
    """
    return [UnslothCallbackAdapter(cb) for cb in callbacks]
