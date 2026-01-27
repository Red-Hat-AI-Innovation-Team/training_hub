"""Visualization utilities for training metrics."""

import glob
import json
import os
from pathlib import Path

import pandas as pd


COLORS = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]

DEFAULT_METRIC_KEYS = ['avg_loss', 'loss', 'avg_loss_backwards', 'train_loss']


def _exponential_moving_average(data: list[float], span: int) -> pd.Series:
    """Calculate the Exponential Moving Average."""
    return pd.Series(data).ewm(span=span, adjust=False).mean()


def _find_metrics_file(ckpt_dir: str, metrics_file: str | None) -> str:
    """
    Find the metrics JSONL file in a checkpoint directory.

    Args:
        ckpt_dir: Path to the checkpoint directory
        metrics_file: Explicit filename to use, or None for auto-detection

    Returns:
        Path to the metrics file

    Raises:
        FileNotFoundError: If no metrics file is found
    """
    if metrics_file:
        path = os.path.join(ckpt_dir, metrics_file)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Metrics file not found: {path}")

    # Try common patterns
    patterns = ['training_log.jsonl', 'training_metrics.jsonl', 'metrics.jsonl']
    for pattern in patterns:
        path = os.path.join(ckpt_dir, pattern)
        if os.path.exists(path):
            return path

    # Fallback: find any .jsonl file (excluding data.jsonl which is training data)
    jsonl_files = glob.glob(os.path.join(ckpt_dir, '*.jsonl'))
    jsonl_files = [f for f in jsonl_files if not f.endswith('data.jsonl')]
    if jsonl_files:
        return jsonl_files[0]

    raise FileNotFoundError(
        f"No metrics file found in {ckpt_dir}. "
        f"Tried: {patterns} and any *.jsonl file. "
        f"Use the metrics_file parameter to specify the filename explicitly."
    )


def _read_losses_from_jsonl(
    filepath: str,
    metric_keys: list[str] | None = None
) -> tuple[list[float], str | None]:
    """
    Read loss values from a JSONL file.

    Tries each key in metric_keys in order, uses the first match found.

    Args:
        filepath: Path to the JSONL file
        metric_keys: List of keys to try in order. Defaults to DEFAULT_METRIC_KEYS.

    Returns:
        Tuple of (list of loss values, key that was used)
    """
    if metric_keys is None:
        metric_keys = DEFAULT_METRIC_KEYS

    losses = []
    key_used = None

    with open(filepath, 'r') as file:
        for line in file:
            try:
                info = json.loads(line)
                for key in metric_keys:
                    if key in info:
                        losses.append(float(info[key]))
                        if key_used is None:
                            key_used = key
                        break
            except json.JSONDecodeError:
                continue

    return losses, key_used


def plot_loss(
    ckpt_output_dirs: str | list[str],
    *,
    metrics_file: str | None = None,
    output_path: str | None = None,
    labels: list[str] | None = None,
    ema: bool = False,
    ema_span: int = 30,
    metric_keys: list[str] | None = None,
    show: bool = False,
) -> str:
    """
    Plot training loss curves from one or more training runs.

    This function reads metrics JSONL files from checkpoint directories
    and generates a loss curve visualization. Supports comparing multiple
    runs and optional EMA smoothing.

    Args:
        ckpt_output_dirs: Path to checkpoint directory, or list of paths
            for comparing multiple runs.
        metrics_file: Name of the metrics file within the checkpoint directory.
            If None, auto-detects common patterns (training_log.jsonl,
            training_metrics.jsonl, metrics.jsonl, or any .jsonl file).
        output_path: Path to save the plot. If None, saves to 'loss_plot.png'
            in the first checkpoint directory.
        labels: Labels for each run in the legend. If None, uses directory names.
        ema: If True, overlay an Exponential Moving Average smoothed line.
        ema_span: Span for EMA smoothing (default: 30).
        metric_keys: List of metric keys to try in order when reading the JSONL.
            Defaults to ['avg_loss', 'loss', 'avg_loss_backwards', 'train_loss'].
        show: If True, display the plot interactively in addition to saving.

    Returns:
        Path to the saved plot file.

    Raises:
        FileNotFoundError: If no metrics file is found in a checkpoint directory.
        ValueError: If no loss data is found in any of the metrics files.

    Example:
        >>> from training_hub import sft, plot_loss
        >>> sft(model_path="...", ckpt_output_dir="./checkpoints", ...)
        >>> plot_path = plot_loss("./checkpoints")
        >>> print(f"Plot saved to: {plot_path}")

        # Compare multiple runs
        >>> plot_loss(
        ...     ["./run1", "./run2"],
        ...     labels=["baseline", "improved"],
        ...     ema=True
        ... )
    """
    # Import matplotlib here to avoid import overhead when not using visualization
    import matplotlib.pyplot as plt

    # Normalize to list
    if isinstance(ckpt_output_dirs, str):
        ckpt_output_dirs = [ckpt_output_dirs]

    # Auto-generate labels from directory names if not provided
    if labels is None:
        labels = [Path(d).name for d in ckpt_output_dirs]

    # Determine output path
    if output_path is None:
        output_path = os.path.join(ckpt_output_dirs[0], 'loss_plot.png')

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')

    keys_found = set()
    has_data = False

    for i, ckpt_dir in enumerate(ckpt_output_dirs):
        color = COLORS[i % len(COLORS)]

        try:
            metrics_path = _find_metrics_file(ckpt_dir, metrics_file)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        losses, key_used = _read_losses_from_jsonl(metrics_path, metric_keys)

        if not losses:
            print(f"Warning: No matching metric found in {metrics_path}")
            continue

        has_data = True
        keys_found.add(key_used)
        label = labels[i] if i < len(labels) else Path(ckpt_dir).name

        ax.plot(losses, label=label, color=color)
        print(f"{ckpt_dir}: {len(losses)} data points (using '{key_used}')")

        if ema:
            ema_values = _exponential_moving_average(losses, span=ema_span)
            ax.plot(
                ema_values,
                label=f'{label} (EMA)',
                color=color,
                linestyle='--',
                alpha=0.7
            )

    if not has_data:
        plt.close(fig)
        raise ValueError(
            "No loss data found in any of the provided directories. "
            "Check that the metrics files exist and contain valid loss values."
        )

    ax.legend()
    metric_label = ', '.join(keys_found) if keys_found else 'loss'
    plt.title(f'{metric_label} over training steps')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
