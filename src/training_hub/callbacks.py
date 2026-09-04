"""Unified callback abstraction for Training Hub.

Provides TrainingHubCallback (base class) and TrainingHubContext (normalized
training state) so users can write lifecycle hooks once and run them across
backends (Unsloth, InstructLab Training, Mini-Trainer).

Users subclass TrainingHubCallback and override only the hooks they need.
Backend adapters translate these to each trainer's native callback interface.

Callbacks are fire-and-forget: adapter layers catch exceptions so a failing
user hook cannot abort training.

Torchrun backends (InstructLab / Mini-Trainer):
    Callbacks cross the subprocess boundary via class-source serialization.
    Requirements:
    - Define subclasses at **module level** (not nested / not ephemeral notebook cells)
    - Put imports **inside method bodies**
    - Use a no-arg (or all-default) constructor — **instance state is not preserved**
    - Keep helpers/constants **inside the callback class** — module-level names
      outside the class are not included in class-source serialization
    - Prefer class attributes or re-read config from files/env inside hooks

Example:
    from training_hub import TrainingHubCallback, TrainingHubContext, lora_sft, sft, osft

    class MetricsLogger(TrainingHubCallback):
        def on_log(self, context: TrainingHubContext) -> None:
            print(f"step={context.step} loss={context.loss} lr={context.learning_rate}")

        def on_evaluate(self, context: TrainingHubContext) -> None:
            print(f"eval step={context.step} metrics={context.metrics}")

    # Same callback class works across backends:
    lora_sft(..., callbacks=[MetricsLogger()])
    sft(..., callbacks=[MetricsLogger()])
    osft(..., callbacks=[MetricsLogger()])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Backends with native on_demand_checkpointing (no hub JIT callback injection).
_NATIVE_JIT_BACKENDS = frozenset({"sft", "osft"})


@dataclass
class TrainingHubControl:
    """Mutable control flags for hub default callbacks (HF TrainerControl-like)."""

    should_save: bool = False
    should_training_stop: bool = False


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
        control: Mutable control bag for default-flow callbacks (optional).
    """

    step: int = 0
    epoch: int = 0
    loss: float | None = None
    learning_rate: float | None = None
    is_main_process: bool = True
    output_dir: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    control: TrainingHubControl | None = None


class TrainingHubCallback:
    """Base class for unified training callbacks.

    All lifecycle hooks default to no-op. Subclass and override only the
    hooks you care about. Exceptions in callbacks never crash training —
    backend adapters catch and log them.

    Note:
        This is intentionally *not* an ABC. All hooks are optional.

    Attributes:
        run_on_all_ranks: When False (default), adapters only dispatch on
            rank 0. Set True on a subclass to opt into per-rank hooks
            (e.g. per-node GPU memory logging).
    """

    run_on_all_ranks: bool = False

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


def merge_default_callbacks(
    user_callbacks: list[TrainingHubCallback] | TrainingHubCallback | None,
    *,
    enable_jit_checkpoint: bool = False,
    ckpt_output_dir: str | None = None,
    backend: str | None = None,
    checkpoint_storage: str | None = None,
) -> list[TrainingHubCallback]:
    """Prepend platform default callbacks before user callbacks.

    Hub defaults run first on each lifecycle event. JIT checkpointing is
    injected only when ``enable_jit_checkpoint=True`` and ``ckpt_output_dir``
    is set; SFT/OSFT use native ``on_demand_checkpointing`` instead. An
    S3 sync callback is prepended for every backend when
    ``checkpoint_storage`` is an s3:// URI.
    """
    from training_hub.adapters.serialize import normalize_hub_callbacks
    from training_hub.checkpoint_utils import resolve_checkpoint_storage
    from training_hub.jit_checkpoint import (
        JITCheckpointCallback,
        S3CheckpointSyncCallback,
    )

    user_cbs = normalize_hub_callbacks(user_callbacks)
    defaults: list[TrainingHubCallback] = []
    if (
        enable_jit_checkpoint
        and ckpt_output_dir
        and backend not in _NATIVE_JIT_BACKENDS
    ):
        defaults.append(JITCheckpointCallback())
    if resolve_checkpoint_storage(checkpoint_storage):
        defaults.append(S3CheckpointSyncCallback())
    return [*defaults, *user_cbs]
