#!/usr/bin/env bash
# Estimate GPU memory requirements for a training configuration
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG_PATH="${TRAINING_HUB_CONFIG:-.training-hub/config.json}"

usage() {
    echo "Usage: th_estimate.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --method METHOD    Estimation method (basic|osft|lora|qlora)"
    echo "  --model PATH       Model path or HuggingFace ID"
    echo "  --gpus N           Number of GPUs"
    echo "  --seq-len N        Max sequence length"
    exit 1
}

die() { echo "ERROR: $1" >&2; exit 1; }

# Parse arguments
METHOD=""
MODEL_PATH=""
GPUS=""
SEQ_LEN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --help) usage ;;
        *) die "Unknown option: $1" ;;
    esac
done

# Read defaults from config if available
if [ -f "$CONFIG_PATH" ]; then
    [ -z "$MODEL_PATH" ] && MODEL_PATH=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('model_path', ''))" 2>/dev/null || echo "")
    [ -z "$GPUS" ] && GPUS=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('nproc_per_node', 1))" 2>/dev/null || echo "1")
    [ -z "$METHOD" ] && METHOD=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "
import json, os
alg = json.load(open(os.environ['CONFIG_PATH'])).get('algorithm', 'sft')
method_map = {'sft': 'basic', 'osft': 'osft', 'lora_sft': 'lora', 'lora_grpo': 'lora', 'grpo': 'basic'}
print(method_map.get(alg, 'basic'))
" 2>/dev/null || echo "basic")
fi

[ -z "$MODEL_PATH" ] && die "No model path specified. Use --model or run /th-setup."
[ -z "$METHOD" ] && METHOD="basic"
[ -z "$GPUS" ] && GPUS="1"
[ -z "$SEQ_LEN" ] && SEQ_LEN="4096"

# Run estimation
TH_METHOD="$METHOD" TH_MODEL="$MODEL_PATH" TH_GPUS="$GPUS" TH_SEQ_LEN="$SEQ_LEN" \
$PYTHON -c "
import json, os, sys

method = os.environ['TH_METHOD']
model_path = os.environ['TH_MODEL']
gpus = int(os.environ['TH_GPUS'])
seq_len = int(os.environ['TH_SEQ_LEN'])

from training_hub import estimate

result = estimate(
    model_name_or_path=model_path,
    method=method,
    num_gpus=gpus,
    max_seq_len=seq_len,
)

print(json.dumps(result, default=str, indent=2))
"
