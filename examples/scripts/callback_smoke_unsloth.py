#!/usr/bin/env python3
"""Minimal Unsloth callback e2e smoke for RHOAIENG-77626.

Covers the unified TrainingHubCallback hooks that fire on a tiny LoRA run,
including on_save and on_evaluate.

Example:
    python callback_smoke_unsloth.py
    python callback_smoke_unsloth.py --help
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUT = "/opt/app-root/src/callback_smoke_out"
DEFAULT_DATA = "/opt/app-root/src/callback_smoke_data.jsonl"
DEFAULT_EVAL = "/opt/app-root/src/callback_smoke_eval.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test TrainingHubCallback hooks on a tiny Unsloth LoRA SFT run "
            "(RHOAIENG-77626)."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help=f"HF model id or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT,
        help=f"Checkpoint / output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA,
        help=f"Train JSONL path to write/use (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--eval-data-path",
        default=DEFAULT_EVAL,
        help=f"Eval JSONL path to write/use (default: {DEFAULT_EVAL})",
    )
    return parser.parse_args()


def _clear_mlflow_env() -> None:
    # Workbench injects MLflow env; HF Trainer would then crash on missing experiment.
    for key in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_RUN_NAME",
        "MLFLOW_TRACKING_AUTH",
    ):
        os.environ.pop(key, None)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    _clear_mlflow_env()

    # Import after argparse so `--help` works without loading training deps.
    from training_hub import TrainingHubCallback, TrainingHubContext, lora_sft

    out = Path(args.out_dir)
    data = Path(args.data_path)
    eval_data = Path(args.eval_data_path)
    out.mkdir(parents=True, exist_ok=True)

    class SmokeLogger(TrainingHubCallback):
        """Records hook calls so we can assert they fired."""

        def __init__(self) -> None:
            self.events: list[str] = []

        def _record(self, name: str, msg: str) -> None:
            self.events.append(name)
            print(msg, flush=True)

        def on_train_begin(self, context: TrainingHubContext) -> None:
            self._record(
                "on_train_begin",
                f"[BEGIN] output_dir={context.output_dir} main={context.is_main_process}",
            )

        def on_epoch_begin(self, context: TrainingHubContext) -> None:
            self._record("on_epoch_begin", f"[EPOCH_BEGIN] epoch={context.epoch}")

        def on_step_begin(self, context: TrainingHubContext) -> None:
            self._record("on_step_begin", f"[STEP_BEGIN] step={context.step}")

        def on_log(self, context: TrainingHubContext) -> None:
            self._record(
                "on_log",
                f"[LOG] step={context.step} epoch={context.epoch} "
                f"loss={context.loss} lr={context.learning_rate}",
            )

        def on_evaluate(self, context: TrainingHubContext) -> None:
            self._record(
                "on_evaluate",
                f"[EVAL] step={context.step} metrics={context.metrics}",
            )

        def on_save(self, context: TrainingHubContext) -> None:
            self._record(
                "on_save",
                f"[SAVE] step={context.step} output_dir={context.output_dir}",
            )

        def on_step_end(self, context: TrainingHubContext) -> None:
            self._record(
                "on_step_end",
                f"[STEP_END] step={context.step} loss={context.loss}",
            )

        def on_epoch_end(self, context: TrainingHubContext) -> None:
            self._record("on_epoch_end", f"[EPOCH_END] epoch={context.epoch}")

        def on_train_end(self, context: TrainingHubContext) -> None:
            self._record(
                "on_train_end",
                f"[END] step={context.step} output_dir={context.output_dir}",
            )

    train_rows = [
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
    eval_rows = train_rows[:2]
    _write_jsonl(data, train_rows)
    _write_jsonl(eval_data, eval_rows)

    callback = SmokeLogger()
    print("Starting lora_sft smoke (incl. save + eval)...", flush=True)
    lora_sft(
        model_path=args.model_path,
        data_path=str(data),
        ckpt_output_dir=str(out),
        num_epochs=1,
        max_seq_len=128,
        micro_batch_size=1,
        logging_steps=1,
        # Hit on_save / on_evaluate during the short run
        save_steps=2,
        eval_steps=2,
        eval_data_path=str(eval_data),
        save_total_limit=2,
        warmup_steps=0,
        learning_rate=2e-4,
        lora_r=8,
        lora_alpha=16,
        sample_packing=False,
        callbacks=[callback],
    )

    counts = Counter(callback.events)
    print("Hook counts:", dict(counts), flush=True)

    required = [
        "on_train_begin",
        "on_epoch_begin",
        "on_step_begin",
        "on_log",
        "on_evaluate",
        "on_save",
        "on_step_end",
        "on_epoch_end",
        "on_train_end",
    ]
    missing = [h for h in required if counts[h] < 1]
    if missing:
        raise SystemExit(f"SMOKE FAILED — missing hooks: {missing}")
    print("SMOKE OK — all 9 unified hooks fired on Unsloth", flush=True)


if __name__ == "__main__":
    main()
