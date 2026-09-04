# JIT Preemption Checkpointing

This guide covers just-in-time (JIT) checkpointing with Training Hub: saving a checkpoint automatically when a training job is interrupted (SIGTERM), and resuming from it automatically on restart.

## Overview

Training jobs on shared infrastructure can be interrupted at any time: Kueue preemption, spot-instance reclaim, node drain, or manual scale-down. Without protection, every interruption loses all training progress.

With JIT checkpointing enabled, Training Hub:

1. **Catches the termination signal** (SIGTERM) sent before the pod is killed
2. **Saves a full checkpoint** (model, optimizer, scheduler, RNG state) at the next safe point
3. **Stops training cleanly** once the checkpoint is written
4. **Resumes automatically** from the latest valid checkpoint when the job restarts. No manual intervention, no configuration changes

[SFT](/api/functions/sft), [OSFT](/api/functions/osft), and [LoRA](/api/functions/lora_sft) all support JIT checkpointing through the same two parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_jit_checkpoint` | `bool` | `False` | Save a checkpoint on SIGTERM and auto-resume on restart |
| `checkpoint_storage` | `str` | `None` | Where checkpoints live: `None`/`"pvc"` for the filesystem, or an `"s3://bucket/prefix"` URI to mirror checkpoints to S3 |

## Quick Start

Enable JIT checkpointing with a single parameter:

```python
from training_hub import sft

result = sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="/mnt/checkpoints/run1",
    num_epochs=3,
    effective_batch_size=8,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    enable_jit_checkpoint=True,
)
```

The same parameter works for OSFT and LoRA:

```python
from training_hub import osft

result = osft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="/mnt/checkpoints/osft-run",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=32,
    max_tokens_per_gpu=8192,
    max_seq_len=2048,
    learning_rate=5e-6,
    enable_jit_checkpoint=True,
)
```

```python
from training_hub import lora_sft

result = lora_sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="/mnt/checkpoints/lora-run",
    lora_r=16,
    num_epochs=3,
    max_seq_len=2048,
    enable_jit_checkpoint=True,
)
```

If the pod is killed mid-training, the logs show the save:

```
Received signal 15; will checkpoint at the next training step boundary.
```

When the job restarts with the same `ckpt_output_dir`, training continues from where it left off: the step counter, learning-rate schedule, and epoch all pick up from the checkpoint rather than restarting at zero.

## Checkpoint Storage: PVC or S3

The `checkpoint_storage` parameter selects where checkpoints are kept:

### PVC / filesystem (default)

```python
sft(..., enable_jit_checkpoint=True)                        # or checkpoint_storage="pvc"
```

Checkpoints stay on the filesystem at `ckpt_output_dir`. On Kubernetes, mount a persistent volume there so checkpoints survive pod restarts (the Kubeflow SDK handles the mount when you use a `pvc://` output location).

### S3-compatible object storage

```python
sft(
    ...,
    enable_jit_checkpoint=True,
    checkpoint_storage="s3://my-bucket/experiments/run1",
)
```

With an `s3://` URI, in addition to the local save:

- Every saved checkpoint is **mirrored to S3 in the background** (training is not blocked by uploads)
- On restart, if the local checkpoint directory is empty, the latest complete checkpoint is **downloaded from S3 before training starts**

Use S3 storage when local disk is ephemeral, for example spot nodes with `emptyDir`, where the replacement pod starts with a blank volume.

**Credentials and endpoint** use the standard AWS environment variables:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
# For MinIO or other S3-compatible stores:
export AWS_ENDPOINT_URL_S3=http://minio.my-namespace.svc:9000
```

S3 support requires `boto3`:

```bash
pip install training-hub[s3]
```

**Upload integrity:** each checkpoint upload finishes by writing an `.upload_complete` marker object. Restore only considers checkpoints that carry the marker, so a partially uploaded checkpoint is never resumed from.

## How It Works

JIT checkpointing is implemented with Training Hub's unified callback system (`TrainingHubCallback`): platform default callbacks are prepended before any user-provided `callbacks=[...]`, so they run first on every lifecycle event. Your own callbacks continue to work unchanged.

Each backend uses the mechanism best suited to it:

| Algorithm | Backend | Mechanism |
|-----------|---------|-----------|
| `lora_sft` | Unsloth / HuggingFace | `JITCheckpointCallback` registers the SIGTERM handler and triggers a full HF Trainer checkpoint via `TrainerControl`. Interrupted saves are marked with an `.incomplete` sentinel and skipped on resume. |
| `sft` | instructlab-training | Delegates to the backend's native `on_demand_checkpointing`: a parent-process signal handler saves full state to `full_state/` and resume is automatic. |
| `osft` | Mini-Trainer | Delegates to the backend's native `on_demand_checkpointing`: `GracefulShutdownHandler` performs a distributed (DCP) save to `full_state_checkpoints/step_N/` and resume is automatic. |

The saved checkpoint always contains everything needed for exact resumption: model weights, optimizer state, LR scheduler state, and RNG state.

## Requirements

Training Hub raises a clear error, rather than silently training without protection, when the installed backend cannot honor `enable_jit_checkpoint`:

| Algorithm | Requirement |
|-----------|-------------|
| `sft` | `instructlab-training >= 0.16.2` |
| `osft` | `rhai-innovation-mini-trainer >= 0.8.3` |
| `lora_sft` | no additional requirement |
| S3 storage | `boto3` (`pip install training-hub[s3]`) |

## Kubernetes Deployment Notes

- **Grace period:** set `terminationGracePeriodSeconds` long enough for the checkpoint to be written after SIGTERM. 120 seconds is a reasonable start for small models; large models need more.
- **torchrun 30-second limit:** PyTorch's elastic launcher force-kills workers ~30 seconds after SIGTERM regardless of the pod grace period ([pytorch/pytorch#119856](https://github.com/pytorch/pytorch/issues/119856)). This affects the torchrun-based backends (`sft`, `osft`) with large models.
- **RWO volumes:** don't let a replacement pod mount the checkpoint volume while the dying pod is still saving. The volume handover can revoke the old pod's write access mid-save and corrupt the checkpoint. On plain Kubernetes Jobs, `podReplacementPolicy: Failed` prevents the overlap.
- **Job restart semantics:** instructlab-training exits with code 0 after a successful preemption save. A plain Kubernetes Job counts that as success and will not restart the pod; restart orchestration is the platform's responsibility (e.g. Kubeflow TrainJob).

## Troubleshooting

**"instructlab-training does not support TrainingArgs.on_demand_checkpointing"**
The installed instructlab-training predates on-demand checkpointing. Upgrade: `pip install "instructlab-training>=0.16.2"`.

**Training restarts from step 0 instead of resuming**
Check that the restarted job uses the same `ckpt_output_dir` and that the directory survived the restart (persistent volume, or S3 storage configured). A checkpoint directory containing an `.incomplete` sentinel is intentionally skipped.

**Checkpoints not appearing in S3**
Verify `boto3` is installed in the training environment and the AWS credential variables are set. Upload failures are logged with full tracebacks in the training logs.

**Pod killed before the checkpoint finished**
Increase `terminationGracePeriodSeconds`. For `sft`/`osft` also note the 30-second torchrun limit above.

