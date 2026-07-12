# Installing for AIPCC 3.5 GA (Universal Workbench Image)

This guide covers installing training_hub with dependency versions that match
the AIPCC 3.5 GA package index. This is required for the downstream Red Hat AI
universal workbench image.

## Why a constraints file?

The AIPCC index ships specific package versions. A plain `pip install` lets the
resolver pick versions that may be mutually incompatible — for example, an old
`unsloth` that doesn't work with `transformers 5.13.x`. The constraints file
pins every resolved version to the validated set.

Additionally, `unsloth` and `unsloth_zoo` must be installed with `--no-deps`
because they cap `transformers<=5.5.0`, conflicting with the required 5.13.0.
The cap is overly conservative — unsloth 2026.7.2 works correctly with
transformers 5.13.0.

## Install recipe

```bash
# 1. Create and activate venv
uv venv --python=3.12 .venv
source .venv/bin/activate

# 2. Bootstrap build tools
uv pip install setuptools wheel packaging setuptools-scm

# 3. Install base + grpo + dev with AIPCC constraints
uv pip install -e ".[grpo,lora,dev]" -c constraints-aipcc.txt

# 4. Install CUDA extras (order matters — after step 3)
uv pip install -e ".[cuda]" -c constraints-aipcc.txt --no-build-isolation

# 5. Install vLLM (not declared as a dependency by openpipe-art)
uv pip install vllm -c constraints-aipcc.txt --no-build-isolation

# 6. Install unsloth with --no-deps (bypasses transformers<=5.5.0 cap)
uv pip install unsloth==2026.7.2 --no-deps
uv pip install unsloth_zoo==2026.7.2 --no-deps

# 7. Install megatron-core for ART 0.5.18 import stubs
uv pip install megatron-core --no-deps

# 8. Re-pin trl (may have been overwritten by unsloth's deps)
uv pip install trl -c constraints-aipcc.txt

# 9. Verify key versions
python -c "
import importlib.metadata as m
for p in ['torch','transformers','vllm','kernels','peft','trl','unsloth']:
    print(f'{p}=={m.version(p)}')
"
```

Expected output:
```
torch==2.11.0
transformers==5.13.0
vllm==0.21.0
kernels==0.15.2
peft==0.19.1
trl==1.6.0
unsloth==2026.7.2
```

## Environment variables

Set these before training:

```bash
export TMPDIR=/mnt/nvme0n1/tmp          # Large filesystem for CUDA compilation
export HF_HOME=/mnt/nvme0n1/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/nvme0n1/huggingface/hub
export HF_TOKEN=<your-token>
```

## Validating the install

```bash
# Smoke test (tiny random models, fast)
python scripts/model_validation.py --run-all --mode all --simple

# Full validation
python scripts/model_validation.py --run-all --mode all --liger both --qlora both

# GRPO smoke test (single GPU)
CUDA_VISIBLE_DEVICES=0 python test_grpo.py
```

## Troubleshooting

**`auto_docstring` NameError from unsloth:** You installed unsloth without
`--no-deps`, and the resolver picked an old version (e.g. 2026.3.11) that is
incompatible with transformers 5.13.x. Reinstall with:
```bash
uv pip install unsloth==2026.7.2 --no-deps
uv pip install unsloth_zoo==2026.7.2 --no-deps
```

**`LayerRepository` ValueError from kernels:** Your transformers version is too
old for the installed kernels version. Use the constraints file to pin
`transformers==5.13.0` and `kernels==0.15.2`.

**flash-attn build failure:** Set `TMPDIR` to a large filesystem and rebuild:
```bash
export TMPDIR=/mnt/nvme0n1/tmp
uv cache clean flash-attn
uv pip install flash-attn --no-build-isolation --force-reinstall
```
