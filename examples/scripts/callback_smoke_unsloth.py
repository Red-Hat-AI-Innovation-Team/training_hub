"""Minimal Unsloth callback e2e smoke for RHOAIENG-77626.

Covers the unified TrainingHubCallback hooks that fire on a tiny LoRA run,
including on_save and on_evaluate.

Run on a CUDA workbench:
  python callback_smoke_unsloth.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

# Workbench injects MLflow env; HF Trainer would then crash on missing experiment.
for _key in (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_RUN_NAME",
    "MLFLOW_TRACKING_AUTH",
):
    os.environ.pop(_key, None)

from training_hub import TrainingHubCallback, TrainingHubContext, lora_sft

OUT = Path("/opt/app-root/src/callback_smoke_out")
DATA = Path("/opt/app-root/src/callback_smoke_data.jsonl")
EVAL = Path("/opt/app-root/src/callback_smoke_eval.jsonl")
OUT.mkdir(parents=True, exist_ok=True)


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
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
    _write_jsonl(DATA, train_rows)
    _write_jsonl(EVAL, eval_rows)

    callback = SmokeLogger()
    print("Starting lora_sft smoke (incl. save + eval)...", flush=True)
    lora_sft(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        data_path=str(DATA),
        ckpt_output_dir=str(OUT),
        num_epochs=1,
        max_seq_len=128,
        micro_batch_size=1,
        logging_steps=1,
        # Hit on_save / on_evaluate during the short run
        save_steps=2,
        eval_steps=2,
        eval_data_path=str(EVAL),
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

    # Unified API hooks that this Unsloth smoke is expected to exercise
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
