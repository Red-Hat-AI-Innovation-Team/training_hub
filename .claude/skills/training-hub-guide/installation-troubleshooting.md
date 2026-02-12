# Installation Troubleshooting

## CUDA kernel import errors

Symptoms: `cannot import from flash_attn`, `unknown symbol`, or similar errors from flash attention, liger kernels, causal-conv1d, mamba-ssm, or other CUDA-compiled packages.

### Full cleanup procedure

```bash
# 1. Clean the uv/pip cache
uv cache clean

# 2. Remove GPU-related packages from ~/.cache
rm -rf ~/.cache/torch
rm -rf ~/.cache/triton
rm -rf ~/.cache/flash_attn
rm -rf ~/.cache/vllm
# Also remove any other GPU-related caches (bitsandbytes, liger, etc.)

# 3. Remove triton kernel cache
rm -rf ~/.triton

# 4. Delete and recreate the venv
rm -rf .venv
uv venv .venv
source .venv/bin/activate

# 5. Reinstall from scratch (two-step for [cuda])
uv pip install training_hub && uv pip install training_hub[cuda] --no-build-isolation
```

The key insight is that stale cached builds of CUDA extensions can persist across installs. Simply reinstalling is often insufficient; the caches and venv must be fully cleared first.

### Common causes

- Upgrading CUDA toolkit or driver without clearing caches
- Switching between torch versions
- Partial installs that leave incompatible compiled artifacts
- Cached triton kernels compiled against a different CUDA version

## flash-attn build failures during [cuda] install

If `flash-attn` fails to build, ensure the base package is installed first:

```bash
# This provides torch, packaging, wheel, ninja needed by flash-attn
uv pip install training_hub

# Then install CUDA extras
uv pip install training_hub[cuda] --no-build-isolation
```

The `--no-build-isolation` flag is critical because flash-attn needs access to the already-installed torch during its build process.

## CUDA version-specific package requirements

Depending on your CUDA version, some packages (e.g., PyTorch) may require downloading from a specific index URL or following additional installation steps. For example, PyTorch often needs a version-specific index like `--index-url https://download.pytorch.org/whl/cu124` for CUDA 12.4.

It is impossible to predict every failure mode with GPU-enabled systems. When encountering installation errors, standard debugging best practices apply: read the error message carefully, check CUDA/driver version compatibility, and consult the specific package's installation docs for your CUDA version.

## LoRA installation

LoRA does not require `[cuda]`. Install with:

```bash
uv pip install training_hub[lora]
```

If xformers conflicts occur, the `[lora]` extras include PyTorch-optimized builds. For specific CUDA versions, try `[lora-cu129]` or `[lora-cu130]` if available.
