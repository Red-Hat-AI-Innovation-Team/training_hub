# `gepa()`

Gradient-free prompt optimization using GEPA (Genetic-Pareto) evolutionary search. Optimizes the *text* of prompts without modifying model weights.

## Signature

```python
from training_hub import gepa

result = gepa(
    seed_candidate: dict[str, str],
    task_lm: str,
    # Data
    data_path: str = None,
    trainset: list[dict] = None,
    valset: list[dict] = None,
    output_dir: str = None,
    # Backend
    backend: str = "gepa",
    # Model configuration
    evaluator: Callable = None,
    reflection_lm: str = None,
    api_base: str = None,
    # Optimization
    max_metric_calls: int = None,
    candidate_selection_strategy: str = None,
    frontier_type: str = None,
    skip_perfect_score: bool = None,
    perfect_score: float = None,
    reflection_minibatch_size: int = None,
    seed: int = None,
    # Tracking
    use_wandb: bool = None,
    use_mlflow: bool = None,
    mlflow_tracking_uri: str = None,
    mlflow_experiment_name: str = None,
    # MLflow backend (backend="mlflow")
    predict_fn: Callable = None,
    prompt_uris: list[str] = None,
    scorers: list = None,
    aggregation: Callable = None,
    enable_tracking: bool = None,
    gepa_kwargs: dict = None,
    **kwargs,
)
```

## Quick Example

```python
from training_hub import gepa

result = gepa(
    seed_candidate={"system_prompt": "You are a helpful assistant. Answer the question."},
    task_lm="openai/gpt-4o-mini",
    data_path="./eval_data.jsonl",
    output_dir="./gepa_output",
    max_metric_calls=200,
)
print(result.best_candidate)
```

## Parameters

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed_candidate` | `dict[str, str]` | Initial prompt to optimize, as a dict of field name → text (e.g. `{"system_prompt": "..."}`). |
| `task_lm` | `str` | Model to optimize for, as a litellm model string (e.g. `"openai/gpt-4o-mini"`). |

### Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | `None` | Path to a JSONL file with `input`/`answer` (and optional `additional_context`) per line. |
| `trainset` | `None` | Training examples as a list of dicts. Alternative to `data_path`. |
| `valset` | `None` | Optional validation set (same format as `trainset`). |
| `output_dir` | `None` | Directory to save `best_candidate.json` and `result.json`. |

### Backend

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `"gepa"` | `"gepa"` calls `gepa.optimize()` directly; `"mlflow"` uses `mlflow.genai.optimize_prompts()`. |

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evaluator` | `None` | Custom scoring function `(data, response) -> (score, feedback, objective_scores)`. Defaults to gepa's `ContainsAnswerEvaluator`. |
| `reflection_lm` | `None` | Model for reflection/mutation (e.g. `"openai/gpt-4o"`). Defaults to `task_lm` if omitted. |
| `api_base` | `None` | Base URL for a local vLLM/OpenAI-compatible endpoint (e.g. `"http://localhost:8000/v1"`). |

### Optimization Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_metric_calls` | `None` | Evaluation budget. GEPA typically needs 100–500 calls. |
| `candidate_selection_strategy` | `None` | One of `"pareto"`, `"current_best"`, `"epsilon_greedy"`, `"top_k_pareto"`. |
| `frontier_type` | `None` | Pareto frontier type: `"instance"`, `"objective"`, `"hybrid"`, `"cartesian"`. |
| `skip_perfect_score` | `None` | Whether to skip perfect-scoring candidates. |
| `perfect_score` | `None` | Score considered perfect (default `1.0`). |
| `reflection_minibatch_size` | `None` | Number of examples examined per reflection step. |
| `seed` | `None` | Random seed for reproducibility. |

### Tracking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_wandb` | `None` | Enable Weights & Biases logging. |
| `use_mlflow` | `None` | Enable MLflow logging. |
| `mlflow_tracking_uri` | `None` | MLflow tracking server URI. |
| `mlflow_experiment_name` | `None` | MLflow experiment name. |

### MLflow Backend Only (`backend="mlflow"`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `predict_fn` | `None` | **Required.** Callable that uses MLflow registered prompts to generate output. |
| `prompt_uris` | `None` | **Required.** List of MLflow prompt URIs to optimize (e.g. `["prompts:/my_prompt/1"]`). |
| `scorers` | `None` | List of MLflow `Scorer` instances. |
| `aggregation` | `None` | Callable computing an overall score from individual scorer outputs. |
| `enable_tracking` | `None` | Whether to log optimization progress to MLflow (default `True`). |
| `gepa_kwargs` | `None` | Additional kwargs forwarded to `gepa.optimize()` via `GepaPromptOptimizer`. |

> Additional advanced parameters (`adapter`, `run_dir`, `batch_sampler`, `reflection_prompt_template`, `custom_candidate_proposer`, `module_selector`, `use_merge`, `stop_callbacks`, `callbacks`, `display_progress_bar`, `cache_evaluation`, `raise_on_exception`, and W&B keys) are also accepted — see the [`gepa()` docstring](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/gepa.py) for the complete list.

## Returns

**Type:** `Any`

- `GEPAResult` for the `gepa` backend (`result.best_candidate` holds the optimized prompt).
- `PromptOptimizationResult` for the `mlflow` backend.

When `output_dir` is set, `best_candidate.json` and `result.json` are also written to disk.

## MLflow Backend Example

```python
import mlflow
from mlflow.genai.scorers import Correctness
from training_hub import gepa

prompt = mlflow.genai.register_prompt(name="qa", template="Answer: {{question}}")

result = gepa(
    seed_candidate={"qa": prompt.template},
    task_lm="openai/gpt-4o-mini",
    backend="mlflow",
    predict_fn=my_predict_fn,
    prompt_uris=[prompt.uri],
    scorers=[Correctness(model="openai:/gpt-4o")],
    data_path="./qa_data.jsonl",
)
```

## Related

- [GEPA Algorithm Guide](/algorithms/gepa) — conceptual overview and tips
- [`GEPAAlgorithm`](/api/classes/GEPAAlgorithm) — class-based API
- [GEPA Backends](/api/backends/gepa) — `gepa` and `mlflow` backend details
