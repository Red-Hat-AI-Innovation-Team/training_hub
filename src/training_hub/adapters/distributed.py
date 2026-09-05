"""Shared helpers for InstructLab / Mini-Trainer TrainingHub bridges.

These helpers are imported inside bridge method bodies so the module-level
native TrainerCallback classes remain ``inspect.getsource``-serializable.
"""

from __future__ import annotations

import logging
from typing import Any

from training_hub.adapters.serialize import load_hub_callbacks_payload
from training_hub.callbacks import TrainingHubCallback, TrainingHubContext

logger = logging.getLogger(__name__)

HUB_HOOKS = (
    "on_train_begin",
    "on_epoch_begin",
    "on_step_begin",
    "on_log",
    "on_evaluate",
    "on_save",
    "on_step_end",
    "on_epoch_end",
    "on_train_end",
)


def build_hub_context_from_native(context: Any) -> TrainingHubContext:
    """Map InstructLab/Mini-Trainer TrainingContext → TrainingHubContext."""
    metrics: dict[str, Any] = {}
    batch_metrics = getattr(context, "batch_metrics", None) or {}
    val_metrics = getattr(context, "val_metrics", None) or {}
    metrics.update(batch_metrics)
    metrics.update(val_metrics)
    checkpoint_path = getattr(context, "checkpoint_path", None)
    if checkpoint_path is not None:
        metrics["checkpoint_path"] = checkpoint_path

    loss = getattr(context, "loss", None)
    if loss is None:
        loss = metrics.get("loss")
    if loss is None:
        loss = metrics.get("eval_loss")
    if loss is None:
        loss = metrics.get("val_loss")

    return TrainingHubContext(
        step=int(getattr(context, "step", 0) or 0),
        epoch=int(getattr(context, "epoch", 0) or 0),
        loss=loss,
        learning_rate=getattr(context, "learning_rate", None),
        is_main_process=bool(getattr(context, "is_world_process_zero", True)),
        output_dir=str(getattr(context, "output_dir", "") or ""),
        metrics=metrics,
    )


class HubCallbackDispatcher:
    """Lazy-load hub callbacks from payload and dispatch with isolation/rank guard."""

    def __init__(self) -> None:
        self._hub_callbacks: list[TrainingHubCallback] | None = None

    def _callbacks(self) -> list[TrainingHubCallback]:
        if self._hub_callbacks is None:
            try:
                self._hub_callbacks = load_hub_callbacks_payload()
            except Exception:
                logger.exception(
                    "Failed to load hub callback payload; callbacks are disabled "
                    "for this worker"
                )
                self._hub_callbacks = []
        return self._hub_callbacks

    def dispatch(self, method_name: str, native_context: Any) -> None:
        hub_ctx = build_hub_context_from_native(native_context)
        for cb in self._callbacks():
            if not hub_ctx.is_main_process and not getattr(
                cb, "run_on_all_ranks", False
            ):
                continue
            try:
                getattr(cb, method_name)(hub_ctx)
            except Exception:
                logger.exception(
                    "%s.%s raised an exception (ignored)",
                    type(cb).__name__,
                    method_name,
                )
