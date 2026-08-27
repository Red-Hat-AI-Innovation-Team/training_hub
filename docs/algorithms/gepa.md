# GEPA — Genetic-Pareto Prompt Optimization

GEPA is a **gradient-free** prompt optimization algorithm. Instead of updating model weights, it evolves the *text* of your prompts using evolutionary search with Pareto-based selection and LLM-driven reflection, keeping the candidates that perform best on your task.

> Because GEPA optimizes prompts rather than weights, it needs **no local GPU**. It reaches the model through an LLM endpoint — a hosted API (OpenAI, Anthropic, …) or a local vLLM / OpenAI-compatible server via `api_base`.

## When to Use

- You want to improve a system prompt, few-shot template, or agent instructions without training the model
- You cannot (or do not want to) fine-tune weights — e.g. you are calling a hosted API model
- You have a small labeled evaluation set and a way to score responses
- You want a cheap, fast iteration loop before committing to weight training

For weight-training algorithms, see [SFT](/algorithms/sft), [OSFT](/algorithms/osft), [LoRA + SFT](/algorithms/lora), or [GRPO](/algorithms/grpo).

## Quick Start

```python
from training_hub import gepa

result = gepa(
    seed_candidate={"system_prompt": "You are a helpful assistant. Answer the question."},
    task_lm="openai/gpt-4o-mini",
    data_path="./eval_data.jsonl",
    output_dir="./gepa_output",
    reflection_lm="openai/gpt-4o",   # defaults to task_lm if omitted
    max_metric_calls=200,
)

print(result.best_candidate)  # the optimized prompt(s)
```

## How It Works

1. **Seed** — you provide a starting prompt as `seed_candidate`, a dict mapping prompt field names to their text (e.g. `{"system_prompt": "..."}`).
2. **Evaluate** — each candidate is scored against your training examples using an evaluator (the default checks whether the expected answer appears in the response).
3. **Reflect & mutate** — a reflection LLM inspects failures and proposes improved prompt candidates.
4. **Select** — GEPA keeps a Pareto frontier of candidates rather than a single best, preserving diversity across objectives.
5. **Repeat** — until the evaluation budget (`max_metric_calls`) is exhausted.

## Backends

GEPA ships with two backends, selected via the `backend` parameter:

| Backend | `backend=` | Description |
|---------|-----------|-------------|
| GEPA (default) | `"gepa"` | Calls `gepa.optimize()` directly. Returns a `GEPAResult`. Best for standalone prompt optimization. |
| MLflow | `"mlflow"` | Wraps `mlflow.genai.optimize_prompts()`. Integrates with the MLflow prompt registry, scorer framework, and experiment tracking. Requires `mlflow>=3.5.0`. |

See [GEPA Backends](/api/backends/gepa) for details.

## Data Format

Training data is a JSONL file where each line has an `input` and an `answer`, plus optional `additional_context`:

```json
{"input": "What is the capital of France?", "answer": "Paris"}
{"input": "2 + 2 * 3 = ?", "answer": "8", "additional_context": {"topic": "arithmetic"}}
```

You can also pass data directly as a list of dicts via `trainset` instead of `data_path`.

## Custom Scoring

The default evaluator checks whether the expected `answer` appears in the model's response. For anything more nuanced, pass a custom `evaluator` (gepa backend) matching the gepa Evaluator protocol:

```python
def evaluator(data, response):
    # returns (score, feedback, objective_scores)
    score = 1.0 if data["answer"].lower() in response.lower() else 0.0
    return score, "", {}
```

For the MLflow backend, supply MLflow `scorers` instead. When using a **local endpoint** with the MLflow backend, prefer custom `@scorer` functions — the built-in scorers (e.g. `Correctness`) hardcode the OpenAI endpoint and do not route through `api_base`.

## Using Local Models

Point GEPA at any OpenAI-compatible server (e.g. vLLM) with `api_base`. Model names use litellm format (`openai/<model-name>`):

```python
result = gepa(
    seed_candidate={"system_prompt": "Answer the question."},
    task_lm="openai/my-local-model",
    api_base="http://localhost:8000/v1",
    data_path="./eval_data.jsonl",
    max_metric_calls=200,
)
```

`api_base` is applied via `OPENAI_API_BASE` for the duration of the run and restored afterward. A dummy `OPENAI_API_KEY` is set automatically when none is configured, since litellm requires a key even for local endpoints.

## Output

When `output_dir` is set, results are written to disk:

- `best_candidate.json` — the optimized prompt(s)
- `result.json` — full optimization metadata (scores, candidates, etc.)

The call also returns the result object: a `GEPAResult` (gepa backend) or `PromptOptimizationResult` (mlflow backend).

## Tips

- **Model size matters for reflection.** Very small models (< 3B) tend to produce poor reflections. Use a capable model for `reflection_lm` even if `task_lm` is small.
- **Budget rule of thumb.** GEPA typically needs 100–500 evaluations (`max_metric_calls`). Roughly `num_examples * 20` is a reasonable starting budget.
- **Design data where the prompt matters.** If a trivial prompt already scores ~1.0, there is nothing to optimize. Choose examples where a vague prompt fails but a precise one succeeds.
- **Set `reflection_minibatch_size`** so reflection examines enough examples per step to catch failure patterns.

## API Reference

See [`gepa()`](/api/functions/gepa) for the full parameter reference and [`GEPAAlgorithm`](/api/classes/GEPAAlgorithm) for the class-based API.

## Example Notebook

- [GEPA Prompt Optimization](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/gepa_prompt_optimization.ipynb) — runnable notebook demonstrating both backends against a local vLLM server.
