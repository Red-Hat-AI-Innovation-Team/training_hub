#!/usr/bin/env python3
"""Minimal InstructLab SFT callback smoke for RHOAIENG-77627.

Example:
    python callback_smoke_instructlab.py --help
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from training_hub.callbacks import TrainingHubCallback, TrainingHubContext

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUT = "/tmp/callback_smoke_ilab_out"
DEFAULT_DATA = "/tmp/callback_smoke_ilab_data.jsonl"


def _clear_mlflow_env() -> None:
    # Workbench injects MLflow env; ilab torchrun worker crashes on missing experiment.
    for key in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_RUN_NAME",
        "MLFLOW_TRACKING_AUTH",
        "MLFLOW_K8S_INTEGRATION",
    ):
        os.environ.pop(key, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test TrainingHubCallback on InstructLab sft() (RHOAIENG-77627).",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--nproc-per-node", default="1")
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class InstructLabSmokeLogger(TrainingHubCallback):
    """Module-level callback — class source must survive torchrun serialize."""

    def on_train_begin(self, context: TrainingHubContext) -> None:
        print(f"[BEGIN] out={context.output_dir} main={context.is_main_process}", flush=True)

    def on_log(self, context: TrainingHubContext) -> None:
        print(
            f"[LOG] step={context.step} loss={context.loss} lr={context.learning_rate}",
            flush=True,
        )

    def on_step_end(self, context: TrainingHubContext) -> None:
        print(f"[STEP_END] step={context.step}", flush=True)

    def on_train_end(self, context: TrainingHubContext) -> None:
        print(f"[END] step={context.step}", flush=True)


def main() -> None:
    args = parse_args()
    _clear_mlflow_env()
    from training_hub import sft

    out = Path(args.out_dir)
    data = Path(args.data_path)
    out.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Capital of France?"},
                {"role": "assistant", "content": "Paris"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Say hello"},
                {"role": "assistant", "content": "Hello!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Color of the sky?"},
                {"role": "assistant", "content": "Blue"},
            ]
        },
    ]
    _write_jsonl(data, rows)

    print("Starting sft smoke with InstructLabSmokeLogger...", flush=True)
    sft(
        model_path=args.model_path,
        data_path=str(data),
        ckpt_output_dir=str(out),
        num_epochs=1,
        max_seq_len=128,
        effective_batch_size=4,
        learning_rate=2e-5,
        nproc_per_node=args.nproc_per_node,
        save_samples=0,
        checkpoint_at_epoch=True,
        callbacks=[InstructLabSmokeLogger()],
    )
    print("SMOKE OK — InstructLab sft() accepted callbacks and finished", flush=True)


if __name__ == "__main__":
    main()
