"""Mini-Trainer adapter for TrainingHubCallback.

Produces a module-level ``mini_trainer.TrainerCallback`` bridge that survives
torchrun serialization. Hub callback class sources are written to a payload
file and reloaded in the worker via ``TRAINING_HUB_CALLBACKS_PATH``.
"""

from __future__ import annotations

from mini_trainer.callbacks import TrainerCallback

from training_hub.adapters.serialize import (
    normalize_hub_callbacks,
    set_callbacks_payload_env,
    write_hub_callbacks_payload,
)
from training_hub.callbacks import TrainingHubCallback


class MiniTrainerCallbackBridge(TrainerCallback):
    """Native Mini-Trainer callback that dispatches to TrainingHubCallbacks.

    Must remain module-level with a no-arg constructor so Mini-Trainer's
    ``inspect.getsource`` + base64 torchrun serialize/deserialize works.
    Imports that load hub callbacks happen inside methods.

    Class body must not reference free names from this module — upstream
    ``exec`` only injects ``TrainerCallback`` / ``TrainingContext``.
    """

    # Literal string so getsource/exec reconstruct works without module imports.
    _path_env = "TRAINING_HUB_CALLBACKS_PATH"

    def __init__(self) -> None:
        self._dispatcher = None

    def _get_dispatcher(self):
        if self._dispatcher is None:
            from training_hub.adapters.distributed import HubCallbackDispatcher

            self._dispatcher = HubCallbackDispatcher()
        return self._dispatcher

    def _dispatch(self, method_name, context) -> None:
        self._get_dispatcher().dispatch(method_name, context)

    def on_train_begin(self, context) -> None:
        self._dispatch("on_train_begin", context)

    def on_epoch_begin(self, context) -> None:
        self._dispatch("on_epoch_begin", context)

    def on_step_begin(self, context) -> None:
        self._dispatch("on_step_begin", context)

    def on_log(self, context) -> None:
        self._dispatch("on_log", context)

    def on_evaluate(self, context) -> None:
        self._dispatch("on_evaluate", context)

    def on_save(self, context) -> None:
        self._dispatch("on_save", context)

    def on_step_end(self, context) -> None:
        self._dispatch("on_step_end", context)

    def on_epoch_end(self, context) -> None:
        self._dispatch("on_epoch_end", context)

    def on_train_end(self, context) -> None:
        self._dispatch("on_train_end", context)


def adapt_hub_callbacks(
    callbacks: list[TrainingHubCallback] | TrainingHubCallback,
    payload_dir: str | None = None,
) -> list[TrainerCallback]:
    """Write hub callback payload and return Mini-Trainer bridge callbacks.

    Args:
        callbacks: TrainingHubCallback instance or list.
        payload_dir: Directory for the payload file (prefer output_dir).

    Returns:
        List containing a single ``MiniTrainerCallbackBridge`` for
        ``TrainingArgs.callbacks``.
    """
    normalized = normalize_hub_callbacks(callbacks)
    if not normalized:
        return []
    path = write_hub_callbacks_payload(normalized, payload_dir=payload_dir)
    set_callbacks_payload_env(path)
    return [MiniTrainerCallbackBridge()]
