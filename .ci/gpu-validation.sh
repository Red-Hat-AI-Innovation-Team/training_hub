#!/usr/bin/env bash
# Model validation, run by .github/workflows/gpu-validation.yml inside a
# MiniCloud job: a throwaway copy of the CI account's `ci-base` workspace on
# the CI node, with this checkout at ~/src and a venv at
# ~/venvs/training_hub that already holds torch and the [cuda] extras. The
# install below only adds what this branch changed.
#
# Knobs (environment, set by the workflow):
#   CI_MODELS  model keys, space separated, or "all" (default)
#   CI_MODES   modes, space separated: sft osft lora, or all (default
#              "sft osft": lora needs the [lora] extra, which does not
#              resolve next to mini_trainer's datasets pin today)
#   CI_SIMPLE  1 = --simple smoke on 2 GPUs (default), 0 = the real runs
#   CI_GPUS    --nproc-per-node for a non-simple run (default 8)
#
# scripts/model_validation.py takes one mode per call and exits 0
# whatever happened; this runs it once per mode and reads every results
# file: each run must be "success" or "skipped".
set -euo pipefail

echo "== $(date -u +%FT%TZ) $(git rev-parse --short HEAD) on $(hostname) =="
nvidia-smi -L

source ~/venvs/training_hub/bin/activate
uv pip install -e ".[cuda]" --no-build-isolation
python -c "import torch, training_hub; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

OUT="$HOME/validation"
rm -rf "$OUT"; mkdir -p "$OUT"
MODES="${CI_MODES:-sft osft}"
[ "$MODES" = all ] && MODES="sft osft lora"
for MODE in $MODES; do
  ARGS=(--output-dir "$OUT/$MODE" --dataset-dir "$OUT/data" --mode "$MODE")
  if [ -n "${CI_MODELS:-}" ] && [ "${CI_MODELS}" != all ]; then
    # shellcheck disable=SC2206  # the keys are space separated on purpose
    ARGS+=(--models ${CI_MODELS})
  else
    ARGS+=(--run-all)
  fi
  if [ "${CI_SIMPLE:-1}" = 1 ]; then
    ARGS+=(--simple)
  else
    ARGS+=(--nproc-per-node "${CI_GPUS:-8}")
  fi
  echo "== python scripts/model_validation.py ${ARGS[*]}"
  python scripts/model_validation.py "${ARGS[@]}"
done

python - "$OUT" <<'PY'
import glob, json, os, sys
files = sorted(glob.glob(os.path.join(sys.argv[1], "*", "validation_results_*.json")))
if not files:
    print("no validation_results_*.json written"); sys.exit(1)
results = [r for f in files for r in json.load(open(f))]
bad = [r for r in results if r.get("status") not in ("success", "skipped")]
n = lambda s: sum(1 for r in results if r.get("status") == s)
print(f"\n== verdict: {len(results)} runs, {n('success')} success, {n('skipped')} skipped, {len(bad)} failed")
for r in bad:
    print(f"   FAIL {r.get('model_key')} {r.get('mode')} liger={r.get('use_liger')} qlora={r.get('use_qlora')}: {(r.get('error') or r.get('status'))[:300]}")
sys.exit(1 if bad else 0)
PY
