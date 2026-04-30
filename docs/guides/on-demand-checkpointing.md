# On-Demand Checkpointing Guide

This guide covers on-demand full-state checkpointing for preemptible training environments like Kubernetes/OpenShift AI, KubeFlow, and SLURM.

## Overview

On-demand checkpointing enables **graceful checkpoint-and-exit** when termination signals are received during training. Unlike epoch-boundary checkpoints which only save at fixed intervals, on-demand checkpointing can save at any point during training — preserving all progress since the last checkpoint.

This feature is available for both [SFT](/algorithms/sft) and [OSFT](/algorithms/osft) algorithms.

### When to Use

Use on-demand checkpointing when:
- Training in **Kubernetes/OpenShift AI** where pods can be preempted
- Using **SLURM** job schedulers with time limits
- Running long training jobs where losing mid-epoch progress is costly
- Training in any environment where **graceful preemption signals** are available

## How It Works

When `on_demand_checkpointing=True`:

1. **Signal handlers** are installed in the parent (launcher) process, catching SIGTERM, SIGINT, SIGUSR1, SIGUSR2, SIGXCPU, and SIGHUP — covering Kubernetes, SLURM, PBS, and LSF schedulers.
2. On signal receipt, a **trigger file** is atomically written to `/dev/shm` (tmpfs, node-local, zero disk I/O).
3. **Worker processes** check for the trigger file at synchronization points in the training loop and coordinate via `all_reduce(MAX)` for distributed consensus across all ranks on all nodes.
4. When any rank detects the trigger, **all ranks collectively save** a full-state distributed checkpoint, then exit gracefully.
5. The parent process waits up to 300 seconds for workers to complete the checkpoint before proceeding with shutdown.

## Quick Start

### SFT

```python
from training_hub import sft

# Enable on-demand checkpointing
result = sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./training_data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=10,
    effective_batch_size=32,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    on_demand_checkpointing=True,
)
```

**Resuming:** Simply re-run the same command. The backend automatically detects the checkpoint in `ckpt_output_dir` and resumes from where training was interrupted.

### OSFT

```python
from training_hub import osft

# Enable on-demand checkpointing
result = osft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./domain_data.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=2048,
    learning_rate=2e-5,
    on_demand_checkpointing=True,
)
```

**Resuming:** Specify the checkpoint path explicitly:

```python
result = osft(
    # ... same parameters as above ...
    on_demand_checkpointing=True,
    resume_from_full_state_checkpoint="./checkpoints/full_state_checkpoints",
)
```

## What Gets Saved

### SFT Checkpoints

The SFT backend saves a full distributed checkpoint including:
- Model state (all parameters)
- Optimizer state
- LR scheduler state
- Checkpoint metadata (epoch, samples seen, global step)

On resume, training continues from the **exact step** within the epoch where it was interrupted, skipping already-completed batches.

### OSFT Checkpoints

The OSFT backend uses DCP (Distributed Checkpoint) sharded saves, where each rank saves its own shard. The checkpoint includes:
- Model OSFT factors (decomposed format, not reconstructed dense weights)
- Optimizer state
- LR scheduler state
- Per-rank RNG states
- Training counters and checkpointer state

On resume, the model structure is initialized normally (with SVD computation for meta tensor materialization), then all parameters are overwritten with checkpoint values via DCP in-place load — enabling **bit-identical optimization trajectories**.

## Multi-Node Behavior

The trigger file mechanism works correctly across multiple nodes:

- The trigger file lives on `/dev/shm`, which is **node-local**. Each node's parent process writes its own trigger file when it receives a signal.
- Workers use `all_reduce(MAX)` to synchronize: if **any rank on any node** detects a trigger, all ranks on all nodes agree to save.
- The checkpoint itself is saved to the **shared filesystem** (the configured `ckpt_output_dir`), accessible by all nodes on resume.
- You only need to trigger on **one node** — the `all_reduce` ensures all nodes participate.

## Kubernetes / OpenShift Configuration

### terminationGracePeriodSeconds

The default Kubernetes grace period of 30 seconds may not be enough for large models. Increase it to give workers time to save:

```yaml
spec:
  terminationGracePeriodSeconds: 300  # 5 minutes
```

The required time depends on model size, number of GPUs, and filesystem speed.

### Stale Trigger Files

If a previous training run was killed before workers could clean up the trigger file, the new run's signal handler detects and removes it during initialization. This prevents a new job from immediately checkpointing and exiting due to a leftover trigger from a prior run.

## Manually Triggering a Checkpoint

You can trigger a checkpoint-and-exit without sending a signal by writing the trigger file directly. This is useful for debugging, testing, or integration with custom orchestration.

Both backends use the same default trigger filename — `checkpoint_requested` — located on the tmpfs mount at `/dev/shm`:

```bash
touch /dev/shm/checkpoint_requested
```

Or from Python (SFT backend):

```python
from instructlab.training.on_demand_checkpoint import write_trigger_file

write_trigger_file()
```

### Custom Trigger Filename

Both `instructlab-training` and `mini-trainer` support overriding the trigger filename via the `CHECKPOINT_TRIGGER_FILENAME` environment variable. Set it before launching training:

```bash
export CHECKPOINT_TRIGGER_FILENAME=my_custom_trigger
```

Then trigger with:

```bash
touch /dev/shm/my_custom_trigger
```

This is useful when running multiple independent training jobs on the same node and you need each job to have its own trigger file.

## Differences Between SFT and OSFT

| Feature | SFT | OSFT |
|---------|-----|------|
| **Parameter** | `on_demand_checkpointing` | `on_demand_checkpointing` + `resume_from_full_state_checkpoint` |
| **Resume** | Automatic (detects checkpoint in `ckpt_output_dir`) | Explicit path via `resume_from_full_state_checkpoint` |
| **Checkpoint format** | Distributed checkpoint | DCP sharded (each rank saves its own shard) |
| **What's saved** | Model, optimizer, scheduler, metadata | OSFT factors, optimizer, scheduler, RNG, counters |
| **Resume fidelity** | Resumes from exact step | Bit-identical optimization trajectories |
| **Default trigger file** | `/dev/shm/checkpoint_requested` | `/dev/shm/checkpoint_requested` |
| **Custom trigger env var** | `CHECKPOINT_TRIGGER_FILENAME` | `CHECKPOINT_TRIGGER_FILENAME` |

## Interaction with Other Checkpointing

On-demand checkpoints are **independent** of the existing checkpoint systems:

- `checkpoint_at_epoch` — epoch-boundary checkpoints continue to work as before
- `accelerate_full_state_at_epoch` (SFT) — Accelerate full-state saves are separate
- `save_final_checkpoint` (OSFT) — final checkpoint saving is separate
- `save_samples` (SFT) — sample-count-based checkpointing is separate

On-demand checkpoints are designed as **opaque resume tokens** for the preemption system. They coexist with, but do not replace, any other checkpoint mechanisms.

## See Also

- [SFT Algorithm](/algorithms/sft) — SFT overview with on-demand checkpointing examples
- [OSFT Algorithm](/algorithms/osft) — OSFT overview with on-demand checkpointing examples
- [sft() Function Reference](/api/functions/sft) — Complete SFT parameter documentation
- [osft() Function Reference](/api/functions/osft) — Complete OSFT parameter documentation
- [Distributed Training Guide](/guides/distributed-training) — Multi-node training setup
