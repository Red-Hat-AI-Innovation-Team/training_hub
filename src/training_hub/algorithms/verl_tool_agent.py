"""Single-turn tool-calling agent loop for verl.

A minimal agent loop that passes tools to the chat template so the model
generates structured tool calls. Used by the verl backend for tool-call
GRPO training.

This is a drop-in replacement for verl's SingleTurnAgentLoop that adds
tool support via tools_kwargs from the dataset.
"""
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_tool_agent")
class SingleTurnToolAgentLoop(AgentLoopBase):
    """Single-turn agent loop that passes tools to the chat template.

    Unlike the default SingleTurnAgentLoop, this passes tool definitions
    from the dataset's tools_kwargs to apply_chat_template, enabling
    models like Qwen3 to generate structured tool calls.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # Extract tools from dataset's tools_kwargs
        tools_kwargs = kwargs.get("tools_kwargs", {})
        tools = tools_kwargs.get("tools", None)

        # Extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # Apply chat template WITH tools
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=tools,
            images=images,
            videos=videos,
        )

        # Generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output
