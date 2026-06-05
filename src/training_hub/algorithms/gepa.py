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

        # Remove MLflow-backend-only params that don't apply here
        for mlflow_param in ("predict_fn", "prompt_uris", "scorers",
                             "aggregation", "enable_tracking", "gepa_kwargs"):
            algorithm_params.pop(mlflow_param, None)

        # Handle api_base: set via litellm env var, not passed to optimize()
        api_base = algorithm_params.pop("api_base", None)
        prev_api_base = os.environ.get("OPENAI_API_BASE")
        prev_api_key = os.environ.get("OPENAI_API_KEY")
        if api_base is not None:
            os.environ["OPENAI_API_BASE"] = api_base
            # litellm requires an API key even for local endpoints;
            # set a dummy value if none is configured
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = "dummy"

        # Default reflection_lm to task_lm if not provided
        if algorithm_params.get("reflection_lm") is None:
            task_lm = algorithm_params.get("task_lm")
            if task_lm is not None:
                algorithm_params["reflection_lm"] = task_lm

        # Build the optimize() kwargs from algorithm_params
        optimize_kwargs = {k: v for k, v in algorithm_params.items() if v is not None}

        try:
            result = optimize(
                trainset=trainset,
                **optimize_kwargs,
            )
        finally:
            # Restore previous environment state
            if api_base is not None:
                if prev_api_base is None:
                    os.environ.pop("OPENAI_API_BASE", None)
                else:
                    os.environ["OPENAI_API_BASE"] = prev_api_base
                if prev_api_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = prev_api_key

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


class MLflowGEPABackend(Backend):
    """MLflow backend for GEPA prompt optimization.

    Uses ``mlflow.genai.optimize_prompts()`` which wraps GEPA with MLflow's
    prompt registry, scorer framework, and experiment tracking. This backend
    is for users who want to optimize prompts registered in MLflow's prompt
    registry and leverage MLflow's evaluation/tracking infrastructure.

    Requires ``mlflow>=3.5.0`` and the ``gepa`` package.
    """

    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute GEPA prompt optimization via MLflow.

        Args:
            algorithm_params: Parameters including MLflow-specific ones
                (predict_fn, prompt_uris, scorers, aggregation, enable_tracking)
                and GEPA optimizer config (reflection_lm, max_metric_calls, etc.).

        Returns:
            PromptOptimizationResult from mlflow.genai.optimize_prompts().
        """
        try:
            from mlflow.genai import optimize_prompts
            from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
        except ImportError as err:
            raise ImportError(
                "MLflow GEPA backend requires 'mlflow>=3.5.0' and 'gepa'. "
                "Install with: pip install training-hub[gepa]"
            ) from err

        # Handle api_base: set via litellm env var, not passed to MLflow
        api_base = algorithm_params.pop("api_base", None)
        prev_api_base = os.environ.get("OPENAI_API_BASE")
        prev_api_key = os.environ.get("OPENAI_API_KEY")
        if api_base is not None:
            os.environ["OPENAI_API_BASE"] = api_base
            # litellm requires an API key even for local endpoints;
            # set a dummy value if none is configured
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = "dummy"

        # Extract MLflow-specific required params
        predict_fn = algorithm_params.pop("predict_fn", None)
        prompt_uris = algorithm_params.pop("prompt_uris", None)

        if predict_fn is None:
            raise ValueError(
                "MLflow backend requires 'predict_fn': a callable that uses "
                "MLflow registered prompts to generate output."
            )
        if prompt_uris is None:
            raise ValueError(
                "MLflow backend requires 'prompt_uris': list of MLflow prompt "
                "URIs to optimize (e.g. ['prompts:/my_prompt/1'])."
            )

        # Extract MLflow-specific optional params
        scorers = algorithm_params.pop("scorers", None)
        aggregation = algorithm_params.pop("aggregation", None)
        enable_tracking = algorithm_params.pop("enable_tracking", True)

        # Build train_data from trainset or data_path
        trainset = algorithm_params.pop("trainset", None)
        data_path = algorithm_params.pop("data_path", None)

        if trainset is None and data_path is not None:
            trainset = GEPABackend._load_data(data_path)

        # Convert GEPA data format to MLflow format if needed
        if trainset is not None:
            train_data = self._convert_to_mlflow_format(trainset)
        else:
            train_data = None

        # Build GepaPromptOptimizer from our params
        reflection_lm = algorithm_params.pop("reflection_lm", None)
        task_lm = algorithm_params.pop("task_lm", None)
        reflection_model = reflection_lm or task_lm
        if reflection_model is None:
            raise ValueError(
                "MLflow backend requires 'reflection_lm' or 'task_lm' to "
                "configure the GepaPromptOptimizer reflection model."
            )

        max_metric_calls = algorithm_params.pop("max_metric_calls", 100)
        display_progress_bar = algorithm_params.pop("display_progress_bar", False)
        gepa_kwargs = algorithm_params.pop("gepa_kwargs", None) or {}

        # Forward GEPA-specific optimization params through gepa_kwargs
        # These are accepted by gepa.optimize() but not by GepaPromptOptimizer directly
        gepa_forward_params = [
            "reflection_minibatch_size", "candidate_selection_strategy",
            "frontier_type", "skip_perfect_score", "perfect_score", "seed",
            "batch_sampler", "reflection_prompt_template",
            "custom_candidate_proposer", "module_selector", "use_merge",
            "stop_callbacks", "callbacks", "cache_evaluation",
            "raise_on_exception",
        ]
        for param in gepa_forward_params:
            value = algorithm_params.pop(param, None)
            if value is not None:
                gepa_kwargs.setdefault(param, value)

        # Warn about params that are silently ignored by the MLflow backend
        ignored_params = {
            k: v for k, v in algorithm_params.items()
            if k not in ("output_dir",) and v is not None
        }
        if ignored_params:
            logger.warning(
                "MLflow backend does not use these parameters (they will be "
                "ignored): %s",
                list(ignored_params.keys()),
            )

        # Convert litellm format (openai/model) to MLflow URI format (openai:/model)
        reflection_model = self._to_mlflow_uri(reflection_model)

        optimizer = GepaPromptOptimizer(
            reflection_model=reflection_model,
            max_metric_calls=max_metric_calls,
            display_progress_bar=display_progress_bar,
            gepa_kwargs=gepa_kwargs or None,
        )

        try:
            result = optimize_prompts(
                predict_fn=predict_fn,
                train_data=train_data,
                prompt_uris=prompt_uris,
                optimizer=optimizer,
                scorers=scorers,
                aggregation=aggregation,
                enable_tracking=enable_tracking,
            )
        finally:
            # Restore previous environment state
            if api_base is not None:
                if prev_api_base is None:
                    os.environ.pop("OPENAI_API_BASE", None)
                else:
                    os.environ["OPENAI_API_BASE"] = prev_api_base
                if prev_api_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = prev_api_key

        # Save results if output_dir specified
        output_dir = algorithm_params.pop("output_dir", None)
        if output_dir is not None:
            self._save_mlflow_result(result, output_dir)

        return result

    @staticmethod
    def _to_mlflow_uri(model_string: str) -> str:
        """Convert litellm model string to MLflow URI format.

        litellm uses ``provider/model`` (e.g. ``openai/gpt-4o-mini``),
        while MLflow expects ``provider:/model`` (e.g. ``openai:/gpt-4o-mini``).
        If already in MLflow format (contains ``:/``), returns as-is.
        """
        if ":/" in model_string:
            return model_string
        if "/" in model_string:
            provider, model = model_string.split("/", 1)
            return f"{provider}:/{model}"
        return model_string

    @staticmethod
    def _convert_to_mlflow_format(trainset: list[dict]) -> list[dict]:
        """Convert GEPA data format to MLflow format if needed.

        GEPA format: {"input": ..., "answer": ..., "additional_context": ...}
        MLflow format: {"inputs": {...}, "expectations": {"expected_response": ...}}

        If data is already in MLflow format (has "inputs" key), pass through.
        """
        if not trainset:
            return trainset

        sample = trainset[0]
        if "inputs" in sample:
            return trainset

        converted = []
        for record in trainset:
            entry = {
                "inputs": {"input": record["input"]},
                "expectations": {
                    "expected_response": record.get("answer", ""),
                },
            }
            if record.get("additional_context"):
                entry["inputs"]["additional_context"] = record["additional_context"]
            converted.append(entry)
        return converted

    @staticmethod
    def _save_mlflow_result(result: Any, output_dir: str) -> None:
        """Save MLflow optimization result metadata to output_dir."""
        os.makedirs(output_dir, exist_ok=True)

        result_dict = {
            "optimizer_name": result.optimizer_name,
            "initial_eval_score": result.initial_eval_score,
            "final_eval_score": result.final_eval_score,
            "initial_eval_score_per_scorer": result.initial_eval_score_per_scorer,
            "final_eval_score_per_scorer": result.final_eval_score_per_scorer,
            "optimized_prompt_uris": [p.uri for p in result.optimized_prompts],
        }

        try:
            with open(os.path.join(output_dir, "result.json"), "w") as f:
                json.dump(result_dict, f, indent=2, default=str)
        except (TypeError, OSError) as e:
            logger.warning("Failed to save MLflow result to %s: %s",
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
        api_base: Optional[str] = None,
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
        # MLflow backend parameters
        predict_fn: Optional[Callable] = None,
        prompt_uris: Optional[list[str]] = None,
        scorers: Optional[list] = None,
        aggregation: Optional[Callable] = None,
        enable_tracking: Optional[bool] = None,
        gepa_kwargs: Optional[dict] = None,
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
            api_base: Base URL for the LLM API endpoint. Use this to point
                at a local vLLM or compatible server (e.g.
                ``"http://localhost:8000/v1"``) instead of relying on
                environment variables.
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
            predict_fn: (MLflow backend only) Callable that uses MLflow
                registered prompts to generate output. Required for the
                ``"mlflow"`` backend.
            prompt_uris: (MLflow backend only) List of MLflow prompt URIs to
                optimize (e.g. ``["prompts:/my_prompt/1"]``). Required for the
                ``"mlflow"`` backend.
            scorers: (MLflow backend only) List of MLflow Scorer instances for
                evaluation (e.g. ``[Correctness(model="openai:/gpt-4o")]``).
            aggregation: (MLflow backend only) Callable that computes an
                overall score from individual scorer outputs.
            enable_tracking: (MLflow backend only) Whether to log optimization
                progress to MLflow (default True).
            gepa_kwargs: (MLflow backend only) Additional kwargs passed through
                to ``gepa.optimize()`` via the ``GepaPromptOptimizer``.
            **kwargs: Additional parameters passed to the backend.

        Returns:
            GEPAResult (gepa backend) or PromptOptimizationResult (mlflow
            backend) with optimization results.
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
            "api_base": api_base,
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
            "predict_fn": predict_fn,
            "prompt_uris": prompt_uris,
            "scorers": scorers,
            "aggregation": aggregation,
            "enable_tracking": enable_tracking,
            "gepa_kwargs": gepa_kwargs,
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
            "api_base": str,
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
            "predict_fn": Callable,
            "prompt_uris": list,
            "scorers": list,
            "aggregation": Callable,
            "enable_tracking": bool,
            "gepa_kwargs": dict,
        }


# Register algorithm and backend
AlgorithmRegistry.register_algorithm("gepa", GEPAAlgorithm)
AlgorithmRegistry.register_backend("gepa", "gepa", GEPABackend)
AlgorithmRegistry.register_backend("gepa", "mlflow", MLflowGEPABackend)


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
    api_base: Optional[str] = None,
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
    # MLflow backend parameters
    predict_fn: Optional[Callable] = None,
    prompt_uris: Optional[list[str]] = None,
    scorers: Optional[list] = None,
    aggregation: Optional[Callable] = None,
    enable_tracking: Optional[bool] = None,
    gepa_kwargs: Optional[dict] = None,
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
        backend: Backend to use. ``"gepa"`` (default) calls ``gepa.optimize()``
            directly; ``"mlflow"`` uses ``mlflow.genai.optimize_prompts()`` for
            integration with MLflow's prompt registry and scorer framework.
        evaluator: Custom scoring function matching the gepa Evaluator
            protocol: ``(data, response) -> (score, feedback, objective_scores)``.
            Defaults to gepa's ContainsAnswerEvaluator.
        reflection_lm: Model for reflection/mutation (e.g.
            ``"openai/gpt-4o"``). Defaults to task_lm if not specified.
        api_base: Base URL for the LLM API endpoint. Use this to point
            at a local vLLM or compatible server (e.g.
            ``"http://localhost:8000/v1"``) instead of relying on
            environment variables.
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
        predict_fn: (MLflow backend only) Callable that uses MLflow
            registered prompts to generate output.
        prompt_uris: (MLflow backend only) List of MLflow prompt URIs to
            optimize (e.g. ``["prompts:/my_prompt/1"]``).
        scorers: (MLflow backend only) List of MLflow Scorer instances.
        aggregation: (MLflow backend only) Callable that computes an overall
            score from individual scorer outputs.
        enable_tracking: (MLflow backend only) Whether to log optimization
            progress to MLflow (default True).
        gepa_kwargs: (MLflow backend only) Additional kwargs passed through
            to ``gepa.optimize()`` via the ``GepaPromptOptimizer``.
        **kwargs: Additional parameters passed to the backend.

    Returns:
        GEPAResult (gepa backend) or PromptOptimizationResult (mlflow backend).

    Example::

        from training_hub import gepa

        # Direct GEPA backend (default)
        result = gepa(
            seed_candidate={"system_prompt": "Answer the question."},
            task_lm="openai/gpt-4o-mini",
            data_path="qa_data.jsonl",
            max_metric_calls=200,
            output_dir="./optimized_prompt",
        )
        print(result.best_candidate)

        # With a local vLLM or OpenAI-compatible endpoint
        result = gepa(
            seed_candidate={"system_prompt": "Answer the question."},
            task_lm="openai/my-model",
            api_base="http://localhost:8000/v1",
            data_path="qa_data.jsonl",
        )

        # MLflow backend (requires mlflow>=3.5.0)
        import mlflow
        from mlflow.genai.scorers import Correctness

        prompt = mlflow.genai.register_prompt(
            name="qa", template="Answer: {{question}}"
        )
        result = gepa(
            seed_candidate={"qa": prompt.template},
            task_lm="openai/gpt-4o-mini",
            backend="mlflow",
            predict_fn=my_predict_fn,
            prompt_uris=[prompt.uri],
            scorers=[Correctness(model="openai:/gpt-4o")],
            data_path="qa_data.jsonl",
        )

    .. note::
        **Using local vLLM/OpenAI-compatible endpoints:**

        Both backends support local models via ``api_base``. Model names use
        litellm format (``openai/model-name``); the MLflow backend automatically
        converts to MLflow URI format (``openai:/model-name``) internally.

        For the MLflow backend with local endpoints, use custom ``@scorer``
        functions rather than built-in scorers like
        ``Correctness(model="openai:/...")``. The built-in MLflow scorers
        hardcode the OpenAI API endpoint and do not route through
        ``OPENAI_API_BASE``.
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
        api_base=api_base,
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
        predict_fn=predict_fn,
        prompt_uris=prompt_uris,
        scorers=scorers,
        aggregation=aggregation,
        enable_tracking=enable_tracking,
        gepa_kwargs=gepa_kwargs,
        **kwargs,
    )
