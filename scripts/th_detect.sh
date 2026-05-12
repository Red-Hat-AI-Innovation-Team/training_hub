#!/usr/bin/env bash
# Detect training_hub environment: library, installer, config, GPU
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG_PATH="${TRAINING_HUB_CONFIG:-.training-hub/config.json}"

# Check library
if $PYTHON -c "import training_hub" 2>/dev/null; then
    echo "library=installed"
else
    echo "library=missing"
fi

# Check installer
if command -v uv > /dev/null 2>&1; then
    echo "installer=uv"
elif command -v pip > /dev/null 2>&1; then
    echo "installer=pip"
else
    echo "installer=none"
fi

# Check config
if [ -f "$CONFIG_PATH" ]; then
    echo "config=found"
else
    echo "config=missing"
fi

# Check GPU
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    echo "gpu=available gpus=$GPU_COUNT"
else
    echo "gpu=unavailable gpus=0"
fi
