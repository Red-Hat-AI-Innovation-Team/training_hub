# GEPA Backends

> Gradient-free prompt optimization. Two backends power the [`gepa()`](/api/functions/gepa) algorithm: the default `gepa` backend and an `mlflow` backend.

## Overview

**Classes:** `GEPABackend`, `MLflowGEPABackend`

**Algorithm Support:** GEPA (prompt optimization)

**Package:** `gepa` (and `mlflow>=3.5.0` for the MLflow backend)

**Status:** Implemented

GEPA optimizes prompt *text* rather than model weights, so neither backend requires a local GPU — both reach the model through an LLM endpoint (hosted API or local vLLM/OpenAI-compatible server via `api_base`).

## `GEPABackend` (default, `backend="gepa"`)

Calls `gepa.optimize()` directly. Best for standalone prompt optimization.

### Usage

```python
from training_hub import gepa

result = gepa(
    seed_candidate={"system_prompt": "You are a helpful assistant. Answer the question."},
    task_lm="openai/gpt-4o-mini",
    data_path="./eval_data.jsonl",
    output_dir="./gepa_output",
    reflection_lm="openai/gpt-4o",
    max_metric_calls=200,
)
print(result.best_candidate)
```

### Via Class-Based API

```python
from training_hub import GEPAAlgorithm, GEPABackend

algo = GEPAAlgorithm(backend=GEPABackend())
result = algo.train(
    seed_candidate={"system_prompt": "Answer the question."},
    task_lm="openai/gpt-4o-mini",
    data_path="./eval_data.jsonl",
)
```

### Notes

- **Scoring:** Defaults to gepa's `ContainsAnswerEvaluator` (checks whether the expected answer appears in the response). Pass a custom `evaluator` for anything more nuanced.
- **Reflection model:** `reflection_lm` defaults to `task_lm` if omitted.
- **Returns:** a `GEPAResult`. With `output_dir` set, writes `best_candidate.json` and `result.json`.

## `MLflowGEPABackend` (`backend="mlflow"`)

Wraps `mlflow.genai.optimize_prompts()`, integrating with MLflow's prompt registry, scorer framework, and experiment tracking. Requires `mlflow>=3.5.0`.

### Usage

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

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `predict_fn` | `Callable` | Uses MLflow registered prompts to generate output. |
| `prompt_uris` | `list[str]` | MLflow prompt URIs to optimize (e.g. `["prompts:/qa/1"]`). |
| `reflection_lm` or `task_lm` | `str` | Reflection model for the `GepaPromptOptimizer`. |

### Notes

- **Model URIs:** litellm model strings (`openai/model`) are automatically converted to MLflow URI format (`openai:/model`).
- **Data conversion:** GEPA-format data (`input`/`answer`) is automatically converted to MLflow's expected format.
- **Local endpoints:** Prefer custom `@scorer` functions over built-in scorers like `Correctness(model="openai:/...")` — the built-ins hardcode the OpenAI endpoint and do not route through `api_base`.
- **Ignored params:** Parameters not applicable to MLflow are logged as a warning rather than silently dropped.
- **Returns:** a `PromptOptimizationResult`. With `output_dir` set, writes `result.json`.

## Installation

```bash
pip install training-hub[gepa]
```

The `[gepa]` extra includes both `gepa` and `mlflow`, so both backends are available.

## Environment Handling

Both backends manage `OPENAI_API_BASE` and `OPENAI_API_KEY` for the duration of a run when `api_base` is set: a dummy API key is supplied if none is configured (litellm requires one even for local endpoints), and the previous environment is restored afterward.

## See Also

- [GEPA Algorithm Overview](/algorithms/gepa)
- [`gepa()` Function Reference](/api/functions/gepa)
- [`GEPAAlgorithm` Class Reference](/api/classes/GEPAAlgorithm)
- [Backends Overview](/api/backends/)
