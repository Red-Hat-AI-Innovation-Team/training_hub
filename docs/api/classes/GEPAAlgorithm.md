# `GEPAAlgorithm` - Genetic-Pareto Prompt Optimization Algorithm Class

> Concrete implementation of the Algorithm interface for GEPA, a gradient-free prompt optimization algorithm that evolves prompt text without modifying model weights.

## Class Signature

```python
from training_hub import GEPAAlgorithm, Backend

class GEPAAlgorithm(Algorithm):
    """
    Optimizes textual prompts using evolutionary search with Pareto-based
    selection and LLM-driven reflection. Does not modify model weights.
    """

    def __init__(self, backend: Backend, **kwargs) -> None:
        """Initialize GEPA algorithm with a backend."""

    def train(
        self,
        seed_candidate: dict[str, str],
        task_lm: str,
        # ... all gepa() parameters
        **kwargs
    ) -> any:
        """Execute GEPA prompt optimization."""

    def get_required_params(self) -> dict[str, type]:
        """Get required parameters."""

    def get_optional_params(self) -> dict[str, type]:
        """Get optional parameters."""
```

## Overview

`GEPAAlgorithm` is the class-based implementation of GEPA (Genetic-Pareto) prompt optimization in Training Hub. It inherits from the [`Algorithm`](Algorithm.md) abstract base class.

Unlike weight-training algorithms (SFT, OSFT, LoRA, GRPO), GEPA optimizes the *prompt itself* — useful for improving system prompts, few-shot templates, and agent instructions. It requires no local GPU; the model is reached through an LLM endpoint (hosted API or local vLLM/OpenAI-compatible server).

This class is useful when you need:
- To reuse an algorithm instance across multiple optimization runs
- Direct access to the algorithm interface
- Custom prompt-optimization pipelines

For most use cases, the convenience function [`gepa()`](../functions/gepa.md) is simpler.

## Constructor

### `__init__(backend: Backend, **kwargs) -> None`

Creates a new GEPAAlgorithm instance.

**Parameters:**
- `backend` (`Backend`): The backend implementation to use (`GEPABackend` or `MLflowGEPABackend`)
- `**kwargs`: Additional configuration passed to the algorithm

**Example:**
```python
from training_hub import GEPAAlgorithm, GEPABackend

backend = GEPABackend()
algorithm = GEPAAlgorithm(backend=backend)
```

## Methods

### `train(**kwargs) -> Any`

Executes GEPA prompt optimization.

#### Parameters

##### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed_candidate` | `dict[str, str]` | Initial prompt to optimize, as a dict of field name → text (e.g. `{"system_prompt": "..."}`). |
| `task_lm` | `str` | Model to optimize for, as a litellm model string (e.g. `"openai/gpt-4o-mini"`). |

##### Optional Parameters

###### Data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | `str` | `None` | Path to a JSONL file with `input`/`answer` (and optional `additional_context`) per line. |
| `trainset` | `list` | `None` | Training examples as a list of dicts. Alternative to `data_path`. |
| `valset` | `list` | `None` | Optional validation set (same format as `trainset`). |
| `output_dir` | `str` | `None` | Directory to save `best_candidate.json` and `result.json`. |

###### Backend Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"gepa"` | `"gepa"` (direct) or `"mlflow"` (MLflow prompt registry + scorers). |

###### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluator` | `Callable` | `None` | Custom scoring function `(data, response) -> (score, feedback, objective_scores)`. Defaults to gepa's `ContainsAnswerEvaluator`. |
| `reflection_lm` | `str` | `None` | Model for reflection/mutation. Defaults to `task_lm` if omitted. |
| `api_base` | `str` | `None` | Base URL for a local vLLM/OpenAI-compatible endpoint. |

###### Optimization Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_metric_calls` | `int` | `None` | Evaluation budget (typically 100–500). |
| `candidate_selection_strategy` | `str` | `None` | `"pareto"`, `"current_best"`, `"epsilon_greedy"`, `"top_k_pareto"`. |
| `frontier_type` | `str` | `None` | `"instance"`, `"objective"`, `"hybrid"`, `"cartesian"`. |
| `skip_perfect_score` | `bool` | `None` | Whether to skip perfect-scoring candidates. |
| `perfect_score` | `float` | `None` | Score considered perfect (default `1.0`). |
| `reflection_minibatch_size` | `int` | `None` | Examples examined per reflection step. |
| `seed` | `int` | `None` | Random seed for reproducibility. |

###### Tracking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_wandb` | `bool` | `None` | Enable Weights & Biases logging. |
| `use_mlflow` | `bool` | `None` | Enable MLflow logging. |
| `mlflow_tracking_uri` | `str` | `None` | MLflow tracking server URI. |
| `mlflow_experiment_name` | `str` | `None` | MLflow experiment name. |

###### MLflow Backend Only (`backend="mlflow"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predict_fn` | `Callable` | `None` | Required. Callable using MLflow registered prompts to generate output. |
| `prompt_uris` | `list` | `None` | Required. MLflow prompt URIs to optimize. |
| `scorers` | `list` | `None` | MLflow `Scorer` instances. |
| `aggregation` | `Callable` | `None` | Combines individual scorer outputs into an overall score. |
| `enable_tracking` | `bool` | `None` | Log optimization progress to MLflow (default `True`). |
| `gepa_kwargs` | `dict` | `None` | Extra kwargs forwarded to `gepa.optimize()`. |

> Additional advanced parameters (`adapter`, `run_dir`, `batch_sampler`, `reflection_prompt_template`, `custom_candidate_proposer`, `module_selector`, `use_merge`, `stop_callbacks`, `callbacks`, `display_progress_bar`, `cache_evaluation`, `raise_on_exception`, W&B keys) are also accepted via `**kwargs`.

#### Returns

**Type:** `Any`

- `GEPAResult` for the `gepa` backend (`result.best_candidate` holds the optimized prompt).
- `PromptOptimizationResult` for the `mlflow` backend.

When `output_dir` is set, `best_candidate.json` and `result.json` are also written to disk.

#### Raises

- **`ValueError`**: When neither `trainset` nor `data_path` is provided, or when the MLflow backend is missing `predict_fn`/`prompt_uris`/reflection model.
- **`ImportError`**: When the `gepa` package (or `mlflow>=3.5.0` for the MLflow backend) is not installed.

### `get_required_params() -> Dict[str, Type]`

Returns the required parameters for GEPA.

**Returns:**

```python
{
    "seed_candidate": dict,
    "task_lm": str,
}
```

### `get_optional_params() -> Dict[str, Type]`

Returns the optional parameters for GEPA (data, model configuration, optimization hyperparameters, tracking, and MLflow-backend parameters). See the [`gepa()` function reference](../functions/gepa.md) for the full list.

## Examples

### Basic Usage

```python
from training_hub import GEPAAlgorithm, GEPABackend

backend = GEPABackend()
algorithm = GEPAAlgorithm(backend=backend)

result = algorithm.train(
    seed_candidate={"system_prompt": "You are a helpful assistant. Answer the question."},
    task_lm="openai/gpt-4o-mini",
    data_path="./eval_data.jsonl",
    output_dir="./gepa_output",
    max_metric_calls=200,
)

print(result.best_candidate)
```

### MLflow Backend

```python
from training_hub import GEPAAlgorithm, MLflowGEPABackend

backend = MLflowGEPABackend()
algorithm = GEPAAlgorithm(backend=backend)

result = algorithm.train(
    seed_candidate={"qa": "Answer: {{question}}"},
    task_lm="openai/gpt-4o-mini",
    predict_fn=my_predict_fn,
    prompt_uris=["prompts:/qa/1"],
    data_path="./qa_data.jsonl",
)
```

## Relationship to gepa() Function

The [`gepa()`](../functions/gepa.md) function is a convenience wrapper around `GEPAAlgorithm`. Both use identical parameters. Prefer the function for most use cases; use the class directly when reusing an instance across multiple optimization runs.

## See Also

- [**gepa() Function**](/api/functions/gepa) - Convenience wrapper function
- [**Algorithm Class**](/api/classes/Algorithm) - Base class interface
- [**GEPA Backends**](/api/backends/gepa) - `gepa` and `mlflow` backends
- [**create_algorithm() Function**](/api/functions/create-algorithm) - Factory function
- [**GEPA Algorithm Overview**](/algorithms/gepa) - Conceptual overview and tips

## Source

[View source on GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/gepa.py)
