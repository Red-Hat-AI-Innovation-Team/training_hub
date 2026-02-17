---
name: training-hub-guide
description: Guides users through LLM post-training with Training Hub, including installation, algorithm selection (SFT, OSFT, LoRA), hyperparameter tuning, troubleshooting OOM errors, interpreting loss curves, and leveraging backend-specific features. Use when the user is working with training_hub, fine-tuning language models, asking about SFT/OSFT/LoRA training, or debugging GPU/CUDA training issues.
---

# Training Hub Guide

Training Hub is an abstraction layer for LLM post-training algorithms. It packages SFT, OSFT, and LoRA behind a unified interface so users do not need to learn multiple backend APIs. Backends are wired together internally; users interact with a single API surface.

For API reference and conceptual overviews, consult the live documentation at https://ai-innovation.team/training_hub/#/ and the `docs/` directory in the repo root. This skill covers practical knowledge, decision frameworks, and troubleshooting that supplements the official docs.

## Installation

### Install targets

```bash
# Minimal (no backends, no GPU training)
uv pip install training_hub

# SFT + OSFT (high-scale distributed fine-tuning via CUDA backends)
# IMPORTANT: base install MUST come first, then [cuda] with --no-build-isolation
uv pip install training_hub && uv pip install training_hub[cuda] --no-build-isolation

# LoRA (budget-friendly, single/few-GPU via Unsloth — does NOT require [cuda])
uv pip install training_hub[lora]
```

The `[cuda]` extra is only needed for SFT and OSFT algorithms. LoRA uses the Unsloth backend which handles its own CUDA dependencies through `[lora]`.

The two-step install for `[cuda]` is required because `flash-attn` and other CUDA packages need torch and packaging to already be present at build time.

### Third-party loggers

Loggers are not bundled. Install separately as needed:

```bash
uv pip install wandb       # Weights & Biases
uv pip install mlflow      # MLflow
uv pip install tensorboard # TensorBoard
```

### Fixing CUDA/kernel import errors

When users encounter errors like `cannot import from flash_attn: unknown symbol` or similar issues with optimized kernels (flash attention, liger, causal-conv1d, mamba-ssm), the root cause is usually stale cached builds. Fix with:

1. `uv cache clean`
2. Remove GPU-related caches from `~/.cache/` (torch, triton, flash_attn, vllm, and similar)
3. Remove `~/.triton/` if it exists (triton kernel cache)
4. Delete the current venv and recreate it fresh
5. Reinstall with the two-step process above

See [installation-troubleshooting.md](installation-troubleshooting.md) for the full cleanup procedure.

## Algorithm selection

Read the algorithm guides at https://ai-innovation.team/training_hub/#/algorithms/sft, https://ai-innovation.team/training_hub/#/algorithms/osft, and https://ai-innovation.team/training_hub/#/algorithms/lora for conceptual overviews. The decision framework:

| Need | Algorithm | Why |
|------|-----------|-----|
| Compute-constrained or simple task fine-tuning | **LoRA** | Low VRAM, fast iteration, but lower capacity and higher forgetting |
| Maximum capacity, forgetting is acceptable | **SFT** | Full-parameter training, distributed multi-node support |
| New knowledge while preserving existing capabilities | **OSFT** | Orthogonal subspace prevents catastrophic forgetting |

Always try multiple algorithms and pick the one that performs best on your evaluation. See the example notebooks in `examples/notebooks/` that compare SFT vs OSFT for continual learning scenarios.

## Hyperparameter configuration

The three hyperparameters that matter most for any algorithm are **learning rate**, **effective batch size**, and **number of epochs**. Detailed guidance including dataset-size-dependent recommendations lives in [hyperparameter-guide.md](hyperparameter-guide.md).

### Quick reference

**SFT / OSFT:**
- Learning rate: start at 5e-6 for smaller datasets, 1e-6 for larger datasets
- Epochs: 2-3 is typical
- Batch size: 32-64 for <1k samples, 128 for 1k-10k, 256+ beyond 10k

**LoRA:**
- Learning rate: start at 1e-5, adjust up toward 1e-4 only if needed
- Rank (`lora_r`): start at 16, increase if underfitting
- Alpha (`lora_alpha`): typically 2x rank

**OSFT-specific:**
- `unfreeze_rank_ratio`: recommended default 0.5. Rarely need above 0.5 for models around 8B parameters. Larger models generally need less; smaller models may need more (see [hyperparameter-guide.md](hyperparameter-guide.md) for why).
- `target_patterns`: optionally restrict OSFT to specific modules (e.g., only MLPs or only attention)

### Key parameters

- **`effective_batch_size`**: Taken as the exact minibatch size on any backend. The algorithm translates this into whatever gradient accumulation is needed internally.
- **`max_seq_len`**: Samples exceeding this length are dropped. Important for long-context data and affects training speed and memory.
- **`unmask_messages`** (OSFT) / **`unmask` field** (SFT): Unmasks all messages except the system message for loss computation. When training on knowledge data where documents are embedded in user messages (e.g., from sdg_hub), this significantly boosts knowledge ingestion.
- **`is_pretraining`**: Enables pretraining mode for document-style data. Uses `block_size` to pack documents into fixed-length sample blocks. Start with block_size=2048, or 512 for short/numerous documents.
- **`accelerate_full_state_at_epoch`** (SFT only): Saves FP32 full-state checkpoints at every epoch. Very expensive (an 8B model checkpoint is ~108GB).

## Memory model and OOM troubleshooting

SFT and OSFT train in FP32 + mixed precision. Memory requirement to load a model: **16 bytes per parameter** (4 bytes x 4 copies: parameter, gradient, 2x AdamW optimizer states).

When you hit OOM, these are the available knobs:

- **`nproc_per_node`**: Use all available GPUs to distribute the workload
- **`max_seq_len`**: Reduce if sequences are longer than what fits in a single forward/backward pass
- **`max_tokens_per_gpu`**: Reduce to fit fewer tokens per GPU per step
- **Liger kernels**: Enable `use_liger=True` to reduce memory via fused kernels
- **Flash attention**: Ensure flash-attn is installed and importable for memory-efficient attention
- **LoRA-specific**: Decrease rank or target fewer modules
- **OSFT-specific**: Decrease `unfreeze_rank_ratio` to reduce SVD memory overhead

If all knobs are exhausted, choose a smaller model or a node with more GPU memory.

Use `from training_hub import estimate` for upfront memory estimation. See `examples/notebooks/memory_estimator_example.ipynb`.

## Experiment tracking

All three algorithms (`sft()`, `osft()`, `lora_sft()`) expose logging configuration as first-class parameters. Loggers are **auto-detected**: they are automatically enabled when their configuration parameters are set.

### Logging parameters

All algorithms accept the same logging parameters:

| Parameter | Logger | Description |
|-----------|--------|-------------|
| `wandb_project` | W&B | Project name (enables W&B logging) |
| `wandb_entity` | W&B | Team or user entity |
| `wandb_run_name` | W&B | Run display name |
| `mlflow_tracking_uri` | MLflow | Tracking server URI (enables MLflow logging) |
| `mlflow_experiment_name` | MLflow | Experiment name |
| `mlflow_run_name` | MLflow | Run name |
| `tensorboard_log_dir` | TensorBoard | Log directory (enables TensorBoard logging) |

### Example

```python
from training_hub import sft

sft(
    model_path="my-model",
    data_path="data.jsonl",
    ckpt_output_dir="./checkpoints",
    # W&B logging — enabled automatically because wandb_project is set
    wandb_project="my-finetune",
    wandb_entity="my-team",
    wandb_run_name="sft-run-1",
    # MLflow — also enabled, multiple loggers can run simultaneously
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="sft-experiment",
)
```

### Environment variable fallback

If logging parameters are not passed explicitly, backends will check these environment variables as fallback:

| Parameter | Environment variable |
|-----------|---------------------|
| `wandb_project` | `WANDB_PROJECT` |
| `wandb_entity` | `WANDB_ENTITY` |
| `wandb_run_name` | `WANDB_RUN_NAME` |
| `mlflow_tracking_uri` | `MLFLOW_TRACKING_URI` |
| `mlflow_experiment_name` | `MLFLOW_EXPERIMENT_NAME` |
| `mlflow_run_name` | `MLFLOW_RUN_NAME` |

Explicit kwargs always take precedence over environment variables.

### Logger support matrix

| Logger | SFT | OSFT | LoRA |
|--------|-----|------|------|
| W&B | Yes | Yes | Yes |
| MLflow | Yes | Yes | Yes |
| TensorBoard | Yes | Limited | Yes |

## Loss monitoring and convergence

Each backend emits loss in a different format. `plot_loss()` auto-detects all of them:

```python
from training_hub import plot_loss
plot_loss(["./run1", "./run2"], labels=["baseline", "tuned"], ema=True)
```

### Log formats

| Backend | Format | File | Loss key |
|---------|--------|------|----------|
| SFT (instructlab-training) | JSONL | `training_log.jsonl` | `avg_loss` |
| OSFT (mini-trainer) | JSONL | `training_log.jsonl` | `loss` |
| LoRA (Unsloth/TRL) | JSON | `checkpoint-*/trainer_state.json` | `loss` |

### Interpreting convergence

- **Train loss alone is insufficient.** A model can converge on train loss forever while overfitting badly.
- **Validation loss is the real signal.** Look for where validation loss stops decreasing or starts increasing (optimal training regime vs overfitting).
- SFT backend does not currently support validation loss (planned). OSFT and LoRA do support it but it must be explicitly enabled via backend kwargs. See [backend-kwargs.md](backend-kwargs.md).
- When validation loss is unavailable, rely on downstream evaluation to gauge actual model quality.

## Backend kwargs passthrough

Every algorithm exposes a curated parameter set, but backends support many more options. Any parameter not directly exposed can be passed as a kwarg to the algorithm function, and it will be forwarded to the backend.

This is covered in detail in [backend-kwargs.md](backend-kwargs.md), including links to each backend's full parameter definitions and practical examples like running plain SFT through the OSFT backend with `osft=False`.

## Evaluation

Validation loss is a useful proxy but not always sufficient. Ideally, evaluate the model on a downstream benchmark or task-specific eval harness both before and after training to measure the actual impact. Evaluation itself is outside Training Hub's scope.

## Additional resources

- **Live docs**: https://ai-innovation.team/training_hub/#/
- **Example notebooks**: `examples/notebooks/` has comprehensive tutorials for each algorithm
- **Example scripts**: `examples/scripts/` has ready-to-run training scripts for various models
- For installation issues: [installation-troubleshooting.md](installation-troubleshooting.md)
- For hyperparameter tuning: [hyperparameter-guide.md](hyperparameter-guide.md)
- For backend options and hidden features: [backend-kwargs.md](backend-kwargs.md)
