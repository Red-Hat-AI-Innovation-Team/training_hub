#!/usr/bin/env bash
# Execute LLM training using saved configuration
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG_PATH="${TRAINING_HUB_CONFIG:-.training-hub/config.json}"

usage() {
    echo "Usage: th_train.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --algorithm ALG    Override algorithm (sft|osft|lora_sft|lora_grpo|grpo)"
    echo "  --data PATH        Override training data path"
    echo "  --model PATH       Override model path or HuggingFace ID"
    echo "  --output DIR       Override checkpoint output directory"
    echo "  --gpus N           Override nproc_per_node"
    exit 1
}

die() { echo "ERROR: $1" >&2; exit 1; }

# Parse arguments
ALGORITHM=""
DATA_PATH=""
MODEL_PATH=""
OUTPUT_DIR=""
GPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm) ALGORITHM="$2"; shift 2 ;;
        --data) DATA_PATH="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --help) usage ;;
        *) die "Unknown option: $1" ;;
    esac
done

[ -f "$CONFIG_PATH" ] || die "Config not found at $CONFIG_PATH. Run setup-guide skill first."

# Read config values via env vars to avoid shell injection
[ -z "$ALGORITHM" ] && ALGORITHM=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('algorithm', ''))")
[ -z "$DATA_PATH" ] && DATA_PATH=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('data_path', ''))")
[ -z "$MODEL_PATH" ] && MODEL_PATH=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('model_path', ''))")
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('ckpt_output_dir', './output'))")
[ -z "$GPUS" ] && GPUS=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('nproc_per_node', 1))")

[ -z "$ALGORITHM" ] && die "No algorithm specified. Run setup-guide skill to configure."
[ -z "$MODEL_PATH" ] && die "No model path specified. Run setup-guide skill to configure."
[ -z "$DATA_PATH" ] && die "No data path specified. Run setup-guide skill to configure."

# Execute training
TH_CONFIG="$CONFIG_PATH" TH_ALGORITHM="$ALGORITHM" TH_DATA="$DATA_PATH" \
TH_MODEL="$MODEL_PATH" TH_OUTPUT="$OUTPUT_DIR" TH_GPUS="$GPUS" \
$PYTHON -c "
import json, os, sys

config = json.load(open(os.environ['TH_CONFIG']))
algorithm = os.environ['TH_ALGORITHM']
data_path = os.environ['TH_DATA']
model_path = os.environ['TH_MODEL']
output_dir = os.environ['TH_OUTPUT']
gpus = int(os.environ['TH_GPUS'])

hyperparams = config.get('hyperparams', {})
alg_config = config.get('algorithm_config', {})

import training_hub

func_map = {
    'sft': training_hub.sft,
    'osft': training_hub.osft,
    'lora_sft': training_hub.lora_sft,
    'lora_grpo': training_hub.lora_grpo,
    'grpo': training_hub.grpo,
}

func = func_map.get(algorithm)
if func is None:
    print(f'ERROR: Unknown algorithm: {algorithm}', file=sys.stderr)
    sys.exit(1)

kwargs = {
    'model_path': model_path,
    'data_path': data_path,
    'ckpt_output_dir': output_dir,
    'nproc_per_node': gpus,
}

kwargs.update(hyperparams)
kwargs.update(alg_config)

# Remove None values
kwargs = {k: v for k, v in kwargs.items() if v is not None}

print(f'Starting {algorithm} training...')
print(f'  Model: {model_path}')
print(f'  Data: {data_path}')
print(f'  Output: {output_dir}')
print(f'  GPUs: {gpus}')

func(**kwargs)

print(json.dumps({'status': 'complete', 'algorithm': algorithm, 'output_dir': output_dir}))
"
