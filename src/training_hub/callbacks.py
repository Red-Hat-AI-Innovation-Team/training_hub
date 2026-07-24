"""Unified callback abstraction for Training Hub.

Provides TrainingHubCallback (base class) and TrainingHubContext (normalized
training state) so users can write lifecycle hooks once and run them across
backends (Unsloth first; InstructLab / Mini-Trainer adapters follow).

Users subclass TrainingHubCallback and override only the hooks they need.
Backend adapters translate these to each trainer's native callback interface.

Callbacks are fire-and-forget: adapter layers catch exceptions so a failing
user hook cannot abort training.

Example:
    from training_hub import TrainingHubCallback, TrainingHubContext, lora_sft

    class MetricsLogger(TrainingHubCallback):
        def on_log(self, context: TrainingHubContext) -> None:
            print(f"step={context.step} loss={context.loss} lr={context.learning_rate}")

        def on_evaluate(self, context: TrainingHubContext) -> None:
            print(f"eval step={context.step} metrics={context.metrics}")

    lora_sft(
        model_path="...",
        data_path="train.jsonl",
        ckpt_output_dir="...",
        eval_data_path="eval.jsonl",
        eval_steps=100,
        callbacks=[MetricsLogger()],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingHubContext:
    """Normalized training state passed to all callback hooks.

    Hides backend-specific internals, providing a consistent view
    regardless of whether the underlying trainer is Unsloth/HuggingFace,
    InstructLab Training, or Mini-Trainer.

    Attributes:
        step: Current global training step.
        epoch: Current epoch number.
        loss: Current training loss, if available.
        learning_rate: Current learning rate from scheduler, if available.
        is_main_process: Whether this is rank 0 in distributed training.
        output_dir: Checkpoint output directory.
        metrics: Backend-specific metrics dict, flattened.
    """

    step: int = 0
    epoch: int = 0
    loss: float | None = None
    learning_rate: float | None = None
    is_main_process: bool = True
    output_dir: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class TrainingHubCallback:
    """Base class for unified training callbacks.

    All lifecycle hooks default to no-op. Subclass and override only the
    hooks you care about. Exceptions in callbacks never crash training —
    backend adapters catch and log them.

    Note:
        This is intentionally *not* an ABC. All hooks are optional.
    """

    def on_train_begin(self, context: TrainingHubContext) -> None:
        """Called after initialization, before the training loop."""

    def on_epoch_begin(self, context: TrainingHubContext) -> None:
        """Called at the start of each epoch."""

    def on_step_begin(self, context: TrainingHubContext) -> None:
        """Called at the start of each training step."""

    def on_log(self, context: TrainingHubContext) -> None:
        """Called when metrics are logged."""

    def on_evaluate(self, context: TrainingHubContext) -> None:
        """Called after validation/evaluation."""

    def on_save(self, context: TrainingHubContext) -> None:
        """Called after a checkpoint is saved."""

    def on_step_end(self, context: TrainingHubContext) -> None:
        """Called at the end of each training step."""

    def on_epoch_end(self, context: TrainingHubContext) -> None:
        """Called at the end of each epoch."""

    def on_train_end(self, context: TrainingHubContext) -> None:
        """Called after training completes."""
