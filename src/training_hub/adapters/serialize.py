"""Serialize TrainingHubCallback classes across the torchrun boundary.

InstructLab and Mini-Trainer reconstruct native TrainerCallback subclasses via
``inspect.getsource`` + base64. Live hub callback instances cannot survive that
path, so we:

1. Extract each hub callback **class** source with ``inspect.getsource``
2. Write the payload to a file (shared filesystem / local tmp)
3. Point workers at the file via ``TRAINING_HUB_CALLBACKS_PATH``
4. Register a module-level native bridge that reloads hub callbacks in-worker

Instance state is not preserved (same constraint as upstream native callbacks).
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from training_hub.callbacks import TrainingHubCallback

CALLBACKS_PATH_ENV = "TRAINING_HUB_CALLBACKS_PATH"


def normalize_hub_callbacks(
    callbacks: list[TrainingHubCallback] | TrainingHubCallback | None,
) -> list[TrainingHubCallback]:
    """Normalize a single callback or list into a list of instances."""
    if callbacks is None:
        return []
    if isinstance(callbacks, TrainingHubCallback):
        callbacks = [callbacks]
    for cb in callbacks:
        if not isinstance(cb, TrainingHubCallback):
            raise TypeError(
                "Expected TrainingHubCallback instance, "
                f"got {type(cb).__name__}"
            )
    return list(callbacks)


def encode_hub_callback(callback: TrainingHubCallback) -> dict[str, str]:
    """Encode one hub callback class source for payload transport.

    Args:
        callback: TrainingHubCallback instance (class source is extracted).

    Returns:
        Dict with ``name`` and base64-encoded ``source``.

    Raises:
        ValueError: If class source cannot be extracted (nested/lambda/etc.).
    """
    cls = type(callback)
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError) as e:
        raise ValueError(
            f"Cannot serialize callback {cls.__name__}: {e}. "
            "Define TrainingHubCallback subclasses at module level "
            "(not nested, not in __main__/notebook cells without a file), "
            "with imports inside method bodies and a no-arg constructor."
        ) from e
    return {
        "name": cls.__name__,
        "source": base64.b64encode(source.encode("utf-8")).decode("ascii"),
    }


def encode_hub_callbacks(callbacks: list[TrainingHubCallback]) -> str:
    """Encode a list of hub callbacks to a base64 JSON payload string."""
    encoded = [encode_hub_callback(cb) for cb in callbacks]
    return base64.b64encode(json.dumps(encoded).encode("utf-8")).decode("ascii")


def decode_hub_callback(entry: dict[str, str]) -> TrainingHubCallback:
    """Reconstruct a TrainingHubCallback instance from an encoded entry."""
    source = base64.b64decode(entry["source"]).decode("utf-8")
    namespace: dict[str, Any] = {
        "TrainingHubCallback": TrainingHubCallback,
        "TrainingHubContext": __import__(
            "training_hub.callbacks", fromlist=["TrainingHubContext"]
        ).TrainingHubContext,
    }
    # Payload written by encode_hub_callback in this process / TrainJob — not untrusted input
    exec(source, namespace)  # noqa: S102
    classes = [
        v
        for v in namespace.values()
        if isinstance(v, type)
        and issubclass(v, TrainingHubCallback)
        and v is not TrainingHubCallback
    ]
    if len(classes) != 1:
        raise ValueError(
            f"Expected exactly one TrainingHubCallback subclass in payload "
            f"entry {entry.get('name')!r}, got {len(classes)}"
        )
    return classes[0]()


def decode_hub_callbacks(encoded: str) -> list[TrainingHubCallback]:
    """Reconstruct hub callbacks from a base64 JSON payload string."""
    entries = json.loads(base64.b64decode(encoded).decode("utf-8"))
    return [decode_hub_callback(entry) for entry in entries]


def write_hub_callbacks_payload(
    callbacks: list[TrainingHubCallback],
    payload_dir: str | None = None,
) -> str:
    """Write encoded hub callbacks to a JSON file and return its path.

    Args:
        callbacks: Hub callback instances to serialize.
        payload_dir: Directory for the payload file. Defaults to a temp dir.
            Prefer checkpoint output_dir so multi-node workers can share it.

    Returns:
        Absolute path to the written payload file.
    """
    if payload_dir is None:
        payload_dir = tempfile.mkdtemp(prefix="training_hub_callbacks_")
    else:
        os.makedirs(payload_dir, exist_ok=True)

    path = Path(payload_dir) / "training_hub_callbacks.json"
    payload = {
        "callbacks": [encode_hub_callback(cb) for cb in callbacks],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path.resolve())


def load_hub_callbacks_payload(path: str | None = None) -> list[TrainingHubCallback]:
    """Load hub callbacks from a payload file.

    Args:
        path: Payload file path. If None, reads ``TRAINING_HUB_CALLBACKS_PATH``.

    Returns:
        List of reconstructed TrainingHubCallback instances.
    """
    if path is None:
        path = os.environ.get(CALLBACKS_PATH_ENV)
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [decode_hub_callback(entry) for entry in data.get("callbacks", [])]


def set_callbacks_payload_env(path: str) -> None:
    """Set ``TRAINING_HUB_CALLBACKS_PATH`` for torchrun worker inheritance."""
    os.environ[CALLBACKS_PATH_ENV] = path
