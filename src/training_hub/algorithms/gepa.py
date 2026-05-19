"""GEPA (Genetic-Pareto) prompt optimization algorithm.

GEPA is a gradient-free prompt optimization algorithm that optimizes textual
prompts without modifying model weights. It uses evolutionary search with
Pareto-based selection and LLM-driven reflection to evolve prompts that
maximize task performance.

Unlike weight-training algorithms (SFT, OSFT, LoRA, GRPO), GEPA optimizes
the prompt itself, making it useful for improving system prompts, few-shot
templates, and agent instructions.

Requires the ``gepa`` package: ``pip install training-hub[gepa]``
"""

import json
import logging
import os
from typing import Any, Callable, Dict, Optional, Type

from . import Algorithm, AlgorithmRegistry, Backend

logger = logging.getLogger(__name__)


class GEPABackend(Backend):
    """Default backend for GEPA prompt optimization using the gepa library."""

    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute GEPA prompt optimization.

        Args:
            algorithm_params: Parameters for the optimization run.

        Returns:
            GEPAResult from the gepa library.
        """
        try:
            from gepa import optimize
        except ImportError as err:
            raise ImportError(
                "GEPA requires the 'gepa' package. "
                "Install it with: pip install training-hub[gepa]"
            ) from err

        # Load trainset from data_path if it's a file path
        trainset = algorithm_params.pop("trainset", None)
        data_path = algorithm_params.pop("data_path", None)

        if trainset is None and data_path is not None:
            trainset = self._load_data(data_path)
        elif trainset is None:
            raise ValueError("Either 'trainset' or 'data_path' must be provided")

        # Extract output_dir (not a gepa.optimize param)
        output_dir = algorithm_params.pop("output_dir", None)

        # Build the optimize() kwargs from algorithm_params
        optimize_kwargs = {k: v for k, v in algorithm_params.items() if v is not None}

        result = optimize(
            trainset=trainset,
            **optimize_kwargs,
        )

        # Save results if output_dir specified
        if output_dir is not None:
            self._save_result(result, output_dir)

        return result

    @staticmethod
    def _load_data(data_path: str) -> list[dict]:
        """Load training data from a JSONL file.

        Expected format per line::

            {"input": "...", "answer": "...", "additional_context": {...}}

        The ``additional_context`` field is optional and defaults to ``{}``.
        """
        data = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                entry = {
                    "input": record["input"],
                    "answer": record["answer"],
                    "additional_context": record.get("additional_context", {}),
                }
                data.append(entry)
        return data

    @staticmethod
    def _save_result(result: Any, output_dir: str) -> None:
        """Save the optimized prompt and metadata to output_dir."""
        os.makedirs(output_dir, exist_ok=True)

        # Save best candidate prompt
        best = result.best_candidate
        with open(os.path.join(output_dir, "best_candidate.json"), "w") as f:
            json.dump(best, f, indent=2)

        # Save full result dict
        try:
            result_dict = result.to_dict()
            with open(os.path.join(output_dir, "result.json"), "w") as f:
                json.dump(result_dict, f, indent=2, default=str)
        except (AttributeError, TypeError, OSError) as e:
            logger.warning("Failed to save result metadata to %s: %s",
                           os.path.join(output_dir, "result.json"), e)


class GEPAAlgorithm(Algorithm):
    """GEPA (Genetic-Pareto) prompt optimization algorithm.

    Optimizes textual prompts using evolutionary search with Pareto-based
    selection and LLM-driven reflection. Does not modify model weights.
    """

    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend
        self.config = kwargs

    def train(
        self,
        seed_candidate: dict[str, str],
        task_lm: str,
        data_path: Optional[str] = None,
        trainset: Optional[list[dict]] = None,
        valset: Optional[list[dict]] = None,
        output_dir: Optional[str] = None,
        # Model configuration
        evaluator: Optional[Callable] = None,
        reflection_lm: Optional[str] = None,
        # Optimization parameters
        max_metric_calls: Optional[int] = None,
        candidate_selection_strategy: Optional[str] = None,
        frontier_type: Optional[str] = None,
        skip_perfect_score: Optional[bool] = None,
        perfect_score: Optional[float] = None,
        reflection_minibatch_size: Optional[int] = None,
        seed: Optional[int] = None,
        # Adapter
        adapter: Optional[Any] = None,
        # Logging parameters
        run_dir: Optional[str] = None,
        use_wandb: Optional[bool] = None,
        wandb_api_key: Optional[str] = None,
        wandb_init_kwargs: Optional[dict] = None,
        use_mlflow: Optional[bool] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        # Advanced
        batch_sampler: Optional[Any] = None,
        reflection_prompt_template: Optional[str | dict[str, str]] = None,
        custom_candidate_proposer: Optional[Any] = None,
        module_selector: Optional[str] = None,
        use_merge: Optional[bool] = None,
        stop_callbacks: Optional[Any] = None,
        callbacks: Optional[list] = None,
        display_progress_bar: Optional[bool] = None,
        cache_evaluation: Optional[bool] = None,
        raise_on_exception: Optional[bool] = None,
        **kwargs,
    ) -> Any:
        """Execute GEPA prompt optimization.

        Args:
            seed_candidate: Initial prompt to optimize. Dict mapping prompt
                field names to their text content, e.g.
                ``{"system_prompt": "You are a helpful assistant..."}``.
            task_lm: The model to optimize for, as a litellm model string
                (e.g. ``"openai/gpt-4o-mini"``).
            data_path: Path to a JSONL file with training examples. Each line
                must have ``"input"`` and ``"answer"`` fields, and optionally
                ``"additional_context"``.
            trainset: Training examples as a list of dicts. Alternative to
                ``data_path`` for passing data directly.
            valset: Optional validation set (same format as trainset).
            output_dir: Directory to save the optimized prompt and results.
            evaluator: Custom scoring function matching the gepa Evaluator
                protocol: ``(data, response) -> (score, feedback, objective_scores)``.
                Defaults to gepa's ContainsAnswerEvaluator.
            reflection_lm: Model for reflection/mutation (e.g.
                ``"openai/gpt-4o"``). Defaults to task_lm if not specified.
            max_metric_calls: Maximum number of evaluation calls (budget).
                GEPA typically needs 100-500 evaluations.
            candidate_selection_strategy: Selection strategy for the
                evolutionary search. One of ``"pareto"``, ``"current_best"``,
                ``"epsilon_greedy"``, ``"top_k_pareto"``.
            frontier_type: Pareto frontier type. One of ``"instance"``,
                ``"objective"``, ``"hybrid"``, ``"cartesian"``.
            skip_perfect_score: Whether to skip candidates that achieve
                perfect score on all training examples.
            perfect_score: Score value considered perfect (default 1.0).
            reflection_minibatch_size: Number of examples per reflection batch.
            seed: Random seed for reproducibility.
            adapter: Custom GEPAAdapter instance for non-default data formats
                or evaluation pipelines.
            run_dir: Directory for GEPA's internal run logs.
            use_wandb: Enable Weights & Biases logging.
            wandb_api_key: W&B API key.
            wandb_init_kwargs: Additional W&B init kwargs.
            use_mlflow: Enable MLflow logging.
            mlflow_tracking_uri: MLflow tracking server URI.
            mlflow_experiment_name: MLflow experiment name.
            batch_sampler: Batch sampling strategy.
            reflection_prompt_template: Custom reflection prompt template.
            custom_candidate_proposer: Custom proposal function.
            module_selector: Module selection strategy for reflection.
            use_merge: Enable candidate merging.
            stop_callbacks: Custom stop conditions.
            callbacks: GEPA callbacks.
            display_progress_bar: Show progress bar during optimization.
            cache_evaluation: Cache evaluation results.
            raise_on_exception: Raise exceptions instead of logging them.
            **kwargs: Additional parameters passed to the backend.

        Returns:
            GEPAResult with best_candidate, metrics, and optimization history.
        """
        params = {
            "seed_candidate": seed_candidate,
            "task_lm": task_lm,
            "data_path": data_path,
            "trainset": trainset,
            "output_dir": output_dir,
        }

        optional_params = {
            "valset": valset,
            "evaluator": evaluator,
            "reflection_lm": reflection_lm,
            "max_metric_calls": max_metric_calls,
            "candidate_selection_strategy": candidate_selection_strategy,
            "frontier_type": frontier_type,
            "skip_perfect_score": skip_perfect_score,
            "perfect_score": perfect_score,
            "reflection_minibatch_size": reflection_minibatch_size,
            "seed": seed,
            "adapter": adapter,
            "run_dir": run_dir,
            "use_wandb": use_wandb,
            "wandb_api_key": wandb_api_key,
            "wandb_init_kwargs": wandb_init_kwargs,
            "use_mlflow": use_mlflow,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
            "batch_sampler": batch_sampler,
            "reflection_prompt_template": reflection_prompt_template,
            "custom_candidate_proposer": custom_candidate_proposer,
            "module_selector": module_selector,
            "use_merge": use_merge,
            "stop_callbacks": stop_callbacks,
            "callbacks": callbacks,
            "display_progress_bar": display_progress_bar,
            "cache_evaluation": cache_evaluation,
            "raise_on_exception": raise_on_exception,
        }

        for key, value in optional_params.items():
            if value is not None:
                params[key] = value

        params.update(kwargs)

        return self.backend.execute_training(params)

    def get_required_params(self) -> Dict[str, Type]:
        """Return required parameters for GEPA."""
        return {
            "seed_candidate": dict,
            "task_lm": str,
        }

    def get_optional_params(self) -> Dict[str, Type]:
        """Return optional parameters for GEPA."""
        return {
            "data_path": str,
            "trainset": list,
            "valset": list,
            "output_dir": str,
            "evaluator": Callable,
            "reflection_lm": str,
            "max_metric_calls": int,
            "candidate_selection_strategy": str,
            "frontier_type": str,
            "skip_perfect_score": bool,
            "perfect_score": float,
            "reflection_minibatch_size": int,
            "seed": int,
            "adapter": object,
            "run_dir": str,
            "use_wandb": bool,
            "wandb_api_key": str,
            "wandb_init_kwargs": dict,
            "use_mlflow": bool,
            "mlflow_tracking_uri": str,
            "mlflow_experiment_name": str,
            "batch_sampler": object,
            "reflection_prompt_template": object,  # accepts str | dict[str, str]
            "custom_candidate_proposer": object,
            "module_selector": str,
            "use_merge": bool,
            "stop_callbacks": object,
            "callbacks": list,
            "display_progress_bar": bool,
            "cache_evaluation": bool,
            "raise_on_exception": bool,
        }


# Register algorithm and backend
AlgorithmRegistry.register_algorithm("gepa", GEPAAlgorithm)
AlgorithmRegistry.register_backend("gepa", "gepa", GEPABackend)


def gepa(
    seed_candidate: dict[str, str],
    task_lm: str,
    data_path: Optional[str] = None,
    trainset: Optional[list[dict]] = None,
    valset: Optional[list[dict]] = None,
    output_dir: Optional[str] = None,
    backend: str = "gepa",
    # Model configuration
    evaluator: Optional[Callable] = None,
    reflection_lm: Optional[str] = None,
    # Optimization parameters
    max_metric_calls: Optional[int] = None,
    candidate_selection_strategy: Optional[str] = None,
    frontier_type: Optional[str] = None,
    skip_perfect_score: Optional[bool] = None,
    perfect_score: Optional[float] = None,
    reflection_minibatch_size: Optional[int] = None,
    seed: Optional[int] = None,
    # Adapter
    adapter: Optional[Any] = None,
    # Logging parameters
    run_dir: Optional[str] = None,
    use_wandb: Optional[bool] = None,
    wandb_api_key: Optional[str] = None,
    wandb_init_kwargs: Optional[dict] = None,
    use_mlflow: Optional[bool] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: Optional[str] = None,
    # Advanced
    batch_sampler: Optional[Any] = None,
    reflection_prompt_template: Optional[str | dict[str, str]] = None,
    custom_candidate_proposer: Optional[Any] = None,
    module_selector: Optional[str] = None,
    use_merge: Optional[bool] = None,
    stop_callbacks: Optional[Any] = None,
    callbacks: Optional[list] = None,
    display_progress_bar: Optional[bool] = None,
    cache_evaluation: Optional[bool] = None,
    raise_on_exception: Optional[bool] = None,
    **kwargs,
) -> Any:
    """Optimize a prompt using GEPA (Genetic-Pareto) evolutionary search.

    GEPA optimizes textual prompts without modifying model weights. It uses
    evolutionary search with Pareto-based selection and LLM-driven reflection
    to find prompts that maximize task performance.

    Args:
        seed_candidate: Initial prompt to optimize. Dict mapping prompt field
            names to their text content, e.g.
            ``{"system_prompt": "You are a helpful assistant..."}``.
        task_lm: The model to optimize for, as a litellm model string
            (e.g. ``"openai/gpt-4o-mini"``).
        data_path: Path to a JSONL file with training examples. Each line must
            have ``"input"`` and ``"answer"`` fields, and optionally
            ``"additional_context"``.
        trainset: Training examples as a list of dicts. Alternative to
            ``data_path`` for passing data directly.
        valset: Optional validation set (same format as trainset).
        output_dir: Directory to save the optimized prompt and results.
        backend: Backend to use (default: ``"gepa"``).
        evaluator: Custom scoring function matching the gepa Evaluator
            protocol: ``(data, response) -> (score, feedback, objective_scores)``.
            Defaults to gepa's ContainsAnswerEvaluator.
        reflection_lm: Model for reflection/mutation (e.g.
            ``"openai/gpt-4o"``). Defaults to task_lm if not specified.
        max_metric_calls: Maximum number of evaluation calls (budget).
            GEPA typically needs 100-500 evaluations.
        candidate_selection_strategy: Selection strategy. One of ``"pareto"``,
            ``"current_best"``, ``"epsilon_greedy"``, ``"top_k_pareto"``.
        frontier_type: Pareto frontier type. One of ``"instance"``,
            ``"objective"``, ``"hybrid"``, ``"cartesian"``.
        skip_perfect_score: Whether to skip perfect-scoring candidates.
        perfect_score: Score value considered perfect (default 1.0).
        reflection_minibatch_size: Number of examples per reflection batch.
        seed: Random seed for reproducibility.
        adapter: Custom GEPAAdapter for non-default data/eval pipelines.
        run_dir: Directory for GEPA's internal run logs.
        use_wandb: Enable Weights & Biases logging.
        wandb_api_key: W&B API key.
        wandb_init_kwargs: Additional W&B init kwargs.
        use_mlflow: Enable MLflow logging.
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_experiment_name: MLflow experiment name.
        batch_sampler: Batch sampling strategy.
        reflection_prompt_template: Custom reflection prompt template.
        custom_candidate_proposer: Custom proposal function.
        module_selector: Module selection strategy for reflection.
        use_merge: Enable candidate merging.
        stop_callbacks: Custom stop conditions.
        callbacks: GEPA callbacks.
        display_progress_bar: Show progress bar during optimization.
        cache_evaluation: Cache evaluation results.
        raise_on_exception: Raise exceptions instead of logging them.
        **kwargs: Additional parameters passed to the backend.

    Returns:
        GEPAResult with ``best_candidate`` (the optimized prompt),
        ``total_metric_calls``, and optimization history.

    Example::

        from training_hub import gepa

        result = gepa(
            seed_candidate={"system_prompt": "Answer the question."},
            task_lm="openai/gpt-4o-mini",
            data_path="qa_data.jsonl",
            max_metric_calls=200,
            output_dir="./optimized_prompt",
        )
        print(result.best_candidate)
    """
    from . import create_algorithm

    algorithm = create_algorithm("gepa", backend)
    return algorithm.train(
        seed_candidate=seed_candidate,
        task_lm=task_lm,
        data_path=data_path,
        trainset=trainset,
        valset=valset,
        output_dir=output_dir,
        evaluator=evaluator,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        candidate_selection_strategy=candidate_selection_strategy,
        frontier_type=frontier_type,
        skip_perfect_score=skip_perfect_score,
        perfect_score=perfect_score,
        reflection_minibatch_size=reflection_minibatch_size,
        seed=seed,
        adapter=adapter,
        run_dir=run_dir,
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        batch_sampler=batch_sampler,
        reflection_prompt_template=reflection_prompt_template,
        custom_candidate_proposer=custom_candidate_proposer,
        module_selector=module_selector,
        use_merge=use_merge,
        stop_callbacks=stop_callbacks,
        callbacks=callbacks,
        display_progress_bar=display_progress_bar,
        cache_evaluation=cache_evaluation,
        raise_on_exception=raise_on_exception,
        **kwargs,
    )
