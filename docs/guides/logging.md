# Experiment Tracking & Logging

Training Hub supports multiple experiment tracking backends so you can monitor training metrics, compare runs, and log hyperparameters. Loggers are **auto-detected** based on which configuration parameters you provide -- just set the relevant parameters and Training Hub handles the rest.

## Supported Loggers

| Logger | Description | SFT | OSFT | LoRA |
|--------|-------------|:---:|:----:|:----:|
| [MLflow](#mlflow) | Open-source platform for ML lifecycle management | Yes | Yes | Yes |
| [Weights & Biases](#weights--biases-wandb) | Cloud-based experiment tracking and visualization | Yes | Yes | Yes |
| [TensorBoard](#tensorboard) | TensorFlow's visualization toolkit for training metrics | Yes | No | Yes |
| [JSONL Metrics](#built-in-jsonl-metrics) | Built-in local file logging (always active) | Yes | Yes | Yes |

?> **Zero-config detection** -- Training Hub automatically enables each logger when its trigger parameter is set. No additional flags or boilerplate code is needed.

## Quick Start

Pick a logger and pass its configuration parameters to any training function:

```python
from training_hub import sft

# MLflow -- just set the tracking URI
sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="my-experiment",
)

# wandb -- just set the project name
sft(
    ...,
    wandb_project="my-project",
)

# TensorBoard -- just set the log directory
sft(
    ...,
    tensorboard_log_dir="./logs/tensorboard",
)
```

## Configuration Parameters

All logging parameters are optional and consistent across [sft()](/api/functions/sft), [osft()](/api/functions/osft), and [lora_sft()](/api/functions/lora_sft).

### MLflow

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mlflow_tracking_uri` | `str` | `MLFLOW_TRACKING_URI` env | MLflow tracking server URI. **Setting this enables MLflow logging.** |
| `mlflow_experiment_name` | `str` | `MLFLOW_EXPERIMENT_NAME` env | Experiment name for organizing runs. |
| `mlflow_run_name` | `str` | `None` | Display name for the run. SFT supports `{time}`, `{utc_time}`, `{rank}` placeholders. |

### Weights & Biases

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wandb_project` | `str` | `WANDB_PROJECT` env | W&B project name. **Setting this enables wandb logging.** |
| `wandb_entity` | `str` | `WANDB_ENTITY` env | W&B team or entity name. |
| `wandb_run_name` | `str` | `None` | Display name for the run. |

### TensorBoard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensorboard_log_dir` | `str` | `None` | Directory for TensorBoard logs. **Setting this enables TensorBoard logging.** |

!> **OSFT does not support TensorBoard.** The OSFT backend (mini-trainer) does not include TensorBoard integration. Use MLflow or wandb instead.

## Environment Variables

Each logger can also be configured through environment variables. When both a parameter and its corresponding environment variable are set, the **parameter takes precedence**.

| Environment Variable | Maps To | Logger |
|---------------------|---------|--------|
| `MLFLOW_TRACKING_URI` | `mlflow_tracking_uri` | MLflow |
| `MLFLOW_EXPERIMENT_NAME` | `mlflow_experiment_name` | MLflow |
| `MLFLOW_RUN_NAME` | `mlflow_run_name` | MLflow |
| `WANDB_PROJECT` | `wandb_project` | W&B |
| `WANDB_ENTITY` | `wandb_entity` | W&B |
| `WANDB_RUN_NAME` | `wandb_run_name` | W&B |

This is useful for configuring loggers without changing code, or for setting defaults across multiple training scripts:

```bash
# Configure MLflow for all training runs in this shell session
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="nightly-training"
```

```python
from training_hub import sft

# MLflow is auto-enabled from environment variables -- no logging params needed
sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=32,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
)
```

## Installation

The logging backends are **not included** in Training Hub's default dependencies. Install the one(s) you need:

```bash
# MLflow
pip install mlflow

# Weights & Biases
pip install wandb

# TensorBoard
pip install tensorboard
```

?> **LoRA extras** -- If you installed Training Hub with `pip install training-hub[lora]`, TensorBoard is already included. MLflow and wandb still need to be installed separately.

## MLflow

[MLflow](https://mlflow.org/) is an open-source platform for managing the full ML lifecycle, including experiment tracking, model versioning, and deployment.

### Setup

Start a local MLflow tracking server:

```bash
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000
```

### Usage with SFT

```python
from training_hub import sft

sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=32,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    # MLflow configuration
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="sft-training",
    mlflow_run_name="qwen-sft-run",
)
```

### Usage with OSFT

```python
from training_hub import osft

osft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.25,
    num_epochs=3,
    effective_batch_size=32,
    learning_rate=5e-6,
    max_seq_len=2048,
    max_tokens_per_gpu=2048,
    # MLflow configuration
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="osft-training",
    mlflow_run_name="qwen-osft-run",
)
```

### Usage with LoRA

```python
from training_hub import lora_sft

lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    # MLflow configuration
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="lora-training",
    mlflow_run_name="qwen-lora-run",
)
```

### Viewing Results

Open your MLflow UI in a browser at the tracking URI (e.g., `http://localhost:5000`) to view logged metrics, compare runs, and inspect hyperparameters.

## Weights & Biases (wandb)

[Weights & Biases](https://wandb.ai/) provides cloud-hosted experiment tracking with rich visualization, team collaboration, and model management features.

### Setup

```bash
pip install wandb
wandb login  # Authenticate with your API key
```

### Usage with SFT

```python
from training_hub import sft

sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=32,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    # W&B configuration
    wandb_project="sft-training",
    wandb_entity="my-team",
    wandb_run_name="qwen-sft-run",
)
```

### Usage with OSFT

```python
from training_hub import osft

osft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.25,
    num_epochs=3,
    effective_batch_size=32,
    learning_rate=5e-6,
    max_seq_len=2048,
    max_tokens_per_gpu=2048,
    # W&B configuration
    wandb_project="osft-training",
    wandb_entity="my-team",
    wandb_run_name="qwen-osft-run",
)
```

### Usage with LoRA

```python
from training_hub import lora_sft

lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    # W&B configuration
    wandb_project="lora-training",
    wandb_entity="my-team",
    wandb_run_name="qwen-lora-run",
)
```

### Viewing Results

After training starts, wandb prints a URL to the run dashboard. You can also view all runs at `https://wandb.ai/<entity>/<project>`.

## TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a visualization toolkit that provides dashboards for tracking metrics like loss and learning rate during training.

!> **SFT and LoRA only** -- TensorBoard is not supported with OSFT. Use MLflow or wandb for OSFT experiment tracking.

### Setup

```bash
pip install tensorboard
```

### Usage with SFT

```python
from training_hub import sft

sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=32,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    # TensorBoard configuration
    tensorboard_log_dir="./logs/tensorboard",
)
```

### Usage with LoRA

```python
from training_hub import lora_sft

lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    # TensorBoard configuration
    tensorboard_log_dir="./logs/tensorboard",
)
```

### Viewing Results

Launch the TensorBoard UI to visualize your training metrics:

```bash
tensorboard --logdir=./logs/tensorboard
```

Then open `http://localhost:6006` in your browser.

## Using Multiple Loggers

You can enable any combination of loggers simultaneously by providing their respective parameters. Training Hub will send metrics to all enabled backends:

```python
from training_hub import sft

sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=32,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    # Enable MLflow + wandb + TensorBoard simultaneously
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="sft-comparison",
    wandb_project="sft-comparison",
    wandb_entity="my-team",
    tensorboard_log_dir="./logs/tensorboard",
)
```

```python
from training_hub import lora_sft

lora_sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    # Enable all three loggers
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="lora-experiments",
    wandb_project="lora-experiments",
    tensorboard_log_dir="./logs/tensorboard",
)
```

## Built-in JSONL Metrics

In addition to the external loggers above, Training Hub automatically writes training metrics to a local JSONL file during every training run. This requires no configuration and serves as a baseline record of your training metrics.

- **SFT** and **OSFT** backends produce a `training_metrics.jsonl` (or similarly named) file in the output directory
- **LoRA** backend writes a `training_metrics.jsonl` file alongside checkpoints

Each line is a JSON object with fields like:

```json
{"step": 10, "epoch": 1, "loss": 2.345, "learning_rate": 1e-05}
```

You can visualize these local metrics using Training Hub's built-in `plot_loss()` function:

```python
from training_hub import plot_loss

# Plot loss for a single run
plot_loss("./checkpoints")

# Compare multiple runs
plot_loss(
    ["./checkpoints/run1", "./checkpoints/run2"],
    labels=["Run 1", "Run 2"],
    ema=True,  # Exponential moving average smoothing
)
```

## Distributed Training

In distributed (multi-GPU or multi-node) training, loggers are handled automatically:

- **MLflow** and **wandb** log from **rank 0 only**, avoiding duplicate metrics
- **TensorBoard** logs from **rank 0 only** with run names that include the rank
- **JSONL metrics** are written per-node, with filenames that include the node rank

No special logging configuration is needed for distributed runs -- just use the same logging parameters as single-GPU training:

```python
from training_hub import sft

sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    effective_batch_size=64,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000,
    # Distributed training
    nproc_per_node=8,
    # Logging works the same as single-GPU
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="distributed-sft",
    wandb_project="distributed-sft",
)
```

## See Also

- [**sft() Function**](/api/functions/sft) - Full SFT parameter reference including logging
- [**osft() Function**](/api/functions/osft) - Full OSFT parameter reference including logging
- [**lora_sft() Function**](/api/functions/lora_sft) - Full LoRA parameter reference including logging
- [**Distributed Training Guide**](/guides/distributed-training) - Multi-GPU and multi-node setup
- [**LoRA Logging Examples**](/algorithms/lora#logging--experiment-tracking) - LoRA-specific logging details
- [**Example Scripts**](/examples/) - Runnable scripts including MLflow and wandb examples
