"""ITS Hub integration for GRPO rollouts.

Provides ITSRollout, a picklable adapter that uses ITS Hub generation
algorithms (BestOfN, SelfConsistency, etc.) as rollout functions for
lora_grpo() with the ART backend.

Requires: its_hub (optional dependency, imported lazily)
"""

import asyncio
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _dict_to_choice(response: dict):
    """Convert an ITS Hub response dict (with _raw_choice) to an OpenAI SDK Choice."""
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    raw = response.get("_raw_choice", {})
    msg_kwargs: dict[str, Any] = {
        "role": response.get("role", "assistant"),
        "content": response.get("content"),
    }
    if "tool_calls" in response:
        msg_kwargs["tool_calls"] = response["tool_calls"]
    return Choice(
        index=raw.get("index", 0),
        finish_reason=raw.get("finish_reason", "stop"),
        message=ChatCompletionMessage(**msg_kwargs),
    )


class ITSRollout:
    """Picklable rollout adapter that uses ITS Hub algorithms for GRPO training.

    Uses ITS Hub's generation strategies (BestOfN, SelfConsistency, BeamSearch,
    ParticleFiltering) as the rollout mechanism for ``lora_grpo()`` with the
    ART backend.

    The ``algorithm_factory`` and ``reward_fn`` must be defined at module level
    (not lambdas or closures) because ART runs in a spawned subprocess.

    Concurrency note: each rollout generates ``budget`` LM calls internally
    via ITS Hub's orchestrator. With ``group_size=8``, ``prompt_batch_size=50``,
    and ``budget=8``, that's 50*8*8=3200 concurrent LM calls. Use
    ``max_concurrency`` to limit the inner ITS Hub calls, and
    ``lora_grpo(concurrency=...)`` to limit the outer rollout calls.

    Args:
        algorithm_factory: Module-level function ``(lm) -> AbstractScalingAlgorithm``.
            Receives an ITS Hub LM pointed at the training model's vLLM server
            and returns a configured algorithm instance. If the algorithm needs
            a reward model (e.g., BestOfN with LLMJudge), construct it here
            using the provided ``lm`` — note that judge calls add to the total
            LM call count beyond ``budget``.
        budget: Number of candidates to generate per rollout.
        reward_fn: Module-level function ``(response_text, task) -> float``.
            Scores the selected response. This is separate from
            ``lora_grpo(reward_fn=...)`` — do not pass both.
        message_key: Key in the task dict containing the prompt messages.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum response tokens.
        max_concurrency: Max concurrent LM calls within each ITS Hub rollout.
            Controls the ITS Hub orchestrator's concurrency limit. Default 32.

    Example::

        from its_hub import SelfConsistency
        from training_hub.algorithms.its_rollout import ITSRollout

        def make_sc(lm):
            return SelfConsistency()

        def my_reward(response_text, task):
            return 1.0 if task["answer"] in response_text else 0.0

        rollout = ITSRollout(
            algorithm_factory=make_sc,
            budget=8,
            reward_fn=my_reward,
        )

        lora_grpo(
            model_path="Qwen/Qwen3-4B",
            rollout_fn=rollout,
            tasks=my_tasks,
            backend="art",
        )
    """

    def __init__(
        self,
        algorithm_factory: Callable,
        budget: int,
        reward_fn: Callable[[str, dict], float],
        message_key: str = "messages",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_concurrency: int = 32,
    ):
        self.algorithm_factory = algorithm_factory
        self.budget = budget
        self.reward_fn = reward_fn
        self.message_key = message_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency

        # Lazy-initialized after unpickling in ART's subprocess
        self._lm = None
        self._algorithm = None
        self._init_lock = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lm"] = None
        state["_algorithm"] = None
        state["_init_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    async def __call__(self, model, task) -> Any:
        """Run an ITS Hub algorithm and return an ART Trajectory.

        Args:
            model: ART TrainableModel with ``.openai_client()`` and
                ``.get_inference_name()`` methods.
            task: Task dict; must contain a key matching ``self.message_key``.

        Returns:
            ``art.Trajectory`` with ``.reward`` set.
        """
        import art
        from openai.types.chat.chat_completion import Choice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage

        if self._lm is None:
            if self._init_lock is None:
                self._init_lock = asyncio.Lock()
            async with self._init_lock:
                if self._lm is None:
                    self._init_its_hub(model)

        if self.message_key not in task:
            raise KeyError(
                f"Task is missing key '{self.message_key}'. "
                f"Available keys: {list(task.keys())}"
            )
        messages = task[self.message_key]

        try:
            result = await self._algorithm.ainfer(
                self._lm,
                messages,
                budget=self.budget,
                return_response_only=False,
            )

            selected = result.the_one
            if selected is None:
                raise ValueError(
                    "ITS Hub algorithm returned no selected candidate "
                    "(result.the_one is None)"
                )
            choice = _dict_to_choice(selected)
            response_text = selected.get("content") or ""

            trajectory = art.Trajectory(
                messages_and_choices=list(messages) + [choice],
            )
            trajectory.reward = float(self.reward_fn(response_text, task))
            return trajectory

        except (ConnectionError, TimeoutError) as e:
            logger.warning("ITS Hub rollout failed (retriable): %s", e)
            fallback_choice = Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant", content="[rollout failed]"
                ),
            )
            trajectory = art.Trajectory(
                messages_and_choices=list(messages) + [fallback_choice],
            )
            trajectory.reward = 0.0
            return trajectory

    def _init_its_hub(self, model) -> None:
        """Create ITS Hub LM and algorithm on first call."""
        from its_hub import OpenAICompatibleLanguageModel

        client = model.openai_client()

        lm_kwargs: dict[str, Any] = {
            "endpoint": str(client.base_url),
            "api_key": client.api_key or "EMPTY",
            "model_name": model.get_inference_name(),
            "include_raw_choices": True,
            "max_concurrency": self.max_concurrency,
        }
        if self.temperature is not None:
            lm_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            lm_kwargs["max_tokens"] = self.max_tokens

        self._lm = OpenAICompatibleLanguageModel(**lm_kwargs)
        self._algorithm = self.algorithm_factory(self._lm)
