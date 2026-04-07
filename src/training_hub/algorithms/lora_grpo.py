"""LoRA + GRPO (Group Relative Policy Optimization) algorithm.

Adapter-based reinforcement learning from verifiable rewards (RLVR) for tool-calling
agents. Uses the ART framework (OpenPipe) for co-located vLLM inference + Unsloth
LoRA training on a single GPU with time-sharing.

Supports:
- Single-turn tool-call verification with built-in reward (default)
- Multi-turn agentic rollouts with custom rollout functions
- Custom reward functions
- Co-located vLLM (default) with planned support for standalone vLLM

Example (single-turn tool-call verification):
    from training_hub import lora_grpo

    result = lora_grpo(
        model_path="Qwen/Qwen3-4B",
        data_path="Agent-Ark/Toucan-1.5M",
        ckpt_output_dir="./grpo_output",
        num_iterations=15,
        group_size=8,
        tasks_per_iteration=100,
    )

Example (custom multi-turn rollout):
    import art

    async def my_rollout(model, task):
        client = model.openai_client()
        model_name = model.get_inference_name()
        # ... run your agentic loop ...
        trajectory = art.Trajectory(messages_and_choices=[...])
        trajectory.reward = compute_reward(...)
        return trajectory

    result = lora_grpo(
        model_path="Qwen/Qwen3-4B",
        ckpt_output_dir="./grpo_output",
        rollout_fn=my_rollout,
        tasks=my_task_list,
        num_iterations=10,
    )
"""

import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from . import Algorithm, Backend, AlgorithmRegistry
from .rewards import tool_call_reward

logger = logging.getLogger(__name__)


async def _shutdown_art_backend(backend) -> None:
    """Shut down the ART LocalBackend and its vLLM engine cleanly.

    ART's backend.close() is a no-op in shared/in-process mode because it only
    handles the dedicated vLLM subprocess. In shared mode, vLLM runs as an
    AsyncLLM engine with a spawned EngineCore process. We need to explicitly
    call engine.shutdown() to terminate it.
    """
    for _, service in backend._services.items():
        # Shut down the vLLM engine if it exists (shared/in-process mode)
        llm_task = getattr(service, "__dict__", {}).get("llm")
        if llm_task is not None and hasattr(llm_task, "result"):
            try:
                engine = llm_task.result() if llm_task.done() else None
                if engine is not None and hasattr(engine, "shutdown"):
                    engine.shutdown()
            except Exception as e:
                logger.warning("Error shutting down vLLM engine: %s", e)

        # Also call the service's own close (handles dedicated mode subprocess)
        close_fn = getattr(service, "close", None)
        if close_fn is not None:
            try:
                close_fn()
            except Exception as e:
                logger.warning("Error closing service: %s", e)

    # Release GPU memory and clean up remaining threads.
    # asyncio.run() can hang on non-daemon threads left by torch/vLLM.
    # Since results are already saved to disk by the training loop,
    # force exit to avoid blocking indefinitely.
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    import threading
    non_daemon = [t for t in threading.enumerate()
                  if t.is_alive() and not t.daemon and t != threading.main_thread()]
    if non_daemon:
        logger.info(
            "Force-exiting to clean up %d non-daemon threads (torch/vLLM)",
            len(non_daemon),
        )
        os._exit(0)


# ---------------------------------------------------------------------------
# Built-in dataset loading (Toucan-style tool-call data)
# ---------------------------------------------------------------------------

def _load_tool_call_dataset(
    data_path: str,
    n_train: int = 5000,
    n_val: int = 500,
    data_config: str = "Qwen3",
    cache_dir: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    """Load a tool-call dataset for single-turn GRPO training.

    Supports HuggingFace dataset IDs (e.g. 'Agent-Ark/Toucan-1.5M') or
    local JSON/JSONL files with the required fields.

    For HuggingFace datasets, each row should contain:
    - messages: list of [system, user, assistant-with-tool-call] messages
    - available_tools: list of tool definitions (OpenAI format)
    - question: the user query

    For local files, each sample should be a dict with:
    - question: user query text
    - tools: list of tool definitions (OpenAI format)
    - target_tool_name: expected tool name
    - target_arguments: expected arguments dict
    - system_prompt: system prompt text

    Returns:
        (train_samples, val_samples) tuple
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "training_hub", "grpo_data")

    cache_path = Path(cache_dir)
    train_cache = cache_path / f"train_{n_train}.json"
    val_cache = cache_path / f"val_{n_val}.json"

    # Return cached if available
    if train_cache.exists() and val_cache.exists():
        logger.info("Loading cached data from %s", cache_dir)
        with open(train_cache) as f:
            train = json.load(f)
        with open(val_cache) as f:
            val = json.load(f)
        return train, val

    # Local file
    if data_path.endswith((".json", ".jsonl")):
        return _load_local_dataset(data_path, n_train, n_val)

    # HuggingFace dataset
    logger.info("Downloading dataset '%s' (config=%s) from HuggingFace...", data_path, data_config)
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace dataset loading. "
            "Install with: pip install datasets"
        )

    total_needed = n_train + n_val
    buffer = int(total_needed * 1.5)

    ds = load_dataset(data_path, data_config, split="train", streaming=True)

    processed = []
    skipped = 0
    for row in ds:
        if len(processed) >= buffer:
            break

        sample = _process_hf_row(row)
        if sample is None:
            skipped += 1
            continue
        processed.append(sample)

    logger.info("Processed %d samples (%d skipped)", len(processed), skipped)

    train = processed[:n_train]
    val = processed[n_train:n_train + n_val]

    # Cache
    cache_path.mkdir(parents=True, exist_ok=True)
    with open(train_cache, "w") as f:
        json.dump(train, f)
    with open(val_cache, "w") as f:
        json.dump(val, f)

    logger.info("Cached %d train, %d val samples to %s", len(train), len(val), cache_dir)
    return train, val


def _load_local_dataset(data_path: str, n_train: int, n_val: int) -> tuple[list[dict], list[dict]]:
    """Load dataset from local JSON/JSONL file.

    Auto-detects format:
    - Single-turn: each sample has {question, tools, target_tool_name, target_arguments}
    - Multi-turn: each sample has {messages} with multiple assistant/function turn pairs.
      Multi-turn traces are decomposed into per-turn samples, each with GT context prefix.
    """
    logger.info("Loading local dataset from %s", data_path)

    if data_path.endswith(".jsonl"):
        raw_samples = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    raw_samples.append(json.loads(line))
    else:
        with open(data_path) as f:
            raw_samples = json.load(f)

    # Auto-detect format from first sample
    if not raw_samples:
        return [], []

    first = raw_samples[0]
    single_turn_fields = {"question", "tools", "target_tool_name", "target_arguments"}

    if single_turn_fields.issubset(first.keys()):
        # Single-turn format
        valid = [s for s in raw_samples if single_turn_fields.issubset(s.keys())]
        if len(valid) < len(raw_samples):
            logger.warning(
                "Filtered %d samples missing required fields",
                len(raw_samples) - len(valid),
            )
        train = valid[:n_train]
        val = valid[n_train:n_train + n_val]
        return train, val

    if "messages" in first:
        # Multi-turn format — decompose into per-turn samples
        all_samples = _decompose_multiturn_traces(raw_samples)
        logger.info(
            "Decomposed %d multi-turn traces into %d per-turn samples",
            len(raw_samples), len(all_samples),
        )
        train = all_samples[:n_train]
        val = all_samples[n_train:n_train + n_val]
        return train, val

    logger.warning("Unrecognized dataset format — no valid samples loaded")
    return [], []


def _extract_tools_from_system_prompt(system_content: str) -> list[dict]:
    """Extract tool definitions from a system prompt that may embed them.

    Handles formats like Qwen chat template markers or raw JSON tool arrays
    embedded in the system prompt text.
    """
    if not system_content:
        return []

    # Look for a JSON array of tool definitions in the content
    start = system_content.find('[{"type"')
    if start < 0:
        start = system_content.find('[{\"type\"')
    if start < 0:
        return []

    # Find matching closing bracket
    depth = 0
    for i in range(start, len(system_content)):
        if system_content[i] == '[':
            depth += 1
        elif system_content[i] == ']':
            depth -= 1
            if depth == 0:
                try:
                    tools = json.loads(system_content[start:i + 1])
                    if isinstance(tools, list) and tools:
                        return tools
                except json.JSONDecodeError:
                    pass
                break

    return []


def _clean_system_prompt(system_content: str) -> str:
    """Remove chat template markers and embedded tool JSON from system prompt.

    Returns a clean system prompt suitable for passing as message content
    (tools will be passed separately via the tools= API parameter).
    """
    import re

    # Remove Qwen-style tool_declare blocks: <|im_system|>tool_declare<|im_middle|>[...]<|im_end|>
    cleaned = re.sub(
        r'<\|im_system\|>tool_declare<\|im_middle\|>.*?<\|im_end\|>',
        '',
        system_content,
        flags=re.DOTALL,
    )

    # If the entire content was just the tool declaration, provide a minimal system prompt
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = "You are a helpful assistant."

    return cleaned


def _normalize_message(msg: dict) -> dict:
    """Normalize a message from deprecated function_call/function format to tool_calls/tool.

    Converts:
    - assistant messages with function_call → assistant with tool_calls
    - function role messages → tool role messages
    This ensures compatibility with modern OpenAI API and vLLM servers.
    """
    role = msg.get("role")

    if role == "assistant" and "function_call" in msg:
        fc = msg["function_call"]
        import uuid
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
        normalized = {
            "role": "assistant",
            "content": msg.get("content") or None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": fc["name"],
                    "arguments": fc.get("arguments", "{}"),
                },
            }],
        }
        # Store tool_call_id so the paired function message can reference it
        normalized["_tool_call_id"] = tool_call_id
        return normalized

    if role == "function":
        return {
            "role": "tool",
            "content": msg.get("content", ""),
            "tool_call_id": msg.get("_tool_call_id", "call_unknown"),
            "name": msg.get("name", ""),
        }

    return msg


def _decompose_multiturn_traces(traces: list[dict]) -> list[dict]:
    """Decompose multi-turn traces into per-turn single-call samples.

    Each multi-turn trace with N tool calls becomes N samples. Sample k
    contains the full GT context up to turn k as the prompt prefix, and
    the expected tool call at turn k as the target.

    Input format (per trace):
        messages: [system, user, assistant+function_call, function_result,
                   assistant+function_call, function_result, ..., assistant_final]
        (messages can be a JSON string or list)

    Output format (per sample):
        system_prompt: system message content
        question: user message content
        context_messages: list of GT messages between user and this turn
        tools: extracted from system prompt or empty (tools passed via system prompt)
        target_tool_name: expected tool name at this turn
        target_arguments: expected tool arguments at this turn
    """
    samples = []

    for trace_idx, trace in enumerate(traces):
        messages = trace.get("messages", [])
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                continue

        if len(messages) < 3:
            continue

        # Extract system and user messages
        system_msg = messages[0] if messages[0].get("role") == "system" else {"role": "system", "content": ""}
        user_msg = None
        user_idx = None
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                user_msg = m
                user_idx = i
                break

        if user_msg is None:
            continue

        # Get tools: prefer top-level field, fall back to extracting from system prompt
        tools = trace.get("tools", [])
        system_content = system_msg.get("content", "")
        if not tools:
            tools = _extract_tools_from_system_prompt(system_content)
            if tools:
                system_content = _clean_system_prompt(system_content)

        # Walk through the conversation and find each assistant tool-call turn
        # Everything between user and current turn is GT context
        context_prefix = []  # GT messages between user and current turn
        turn_messages = messages[user_idx + 1:]  # everything after user

        i = 0
        while i < len(turn_messages):
            msg = turn_messages[i]

            if msg.get("role") != "assistant":
                # Non-assistant message (shouldn't happen at turn boundary, skip)
                context_prefix.append(msg)
                i += 1
                continue

            # Check if this assistant message has a tool call
            fc = msg.get("function_call")
            tcs = msg.get("tool_calls")

            if fc:
                target_name = fc.get("name", "")
                target_args_str = fc.get("arguments", "{}")
                if isinstance(target_args_str, str):
                    try:
                        target_args = json.loads(target_args_str)
                    except json.JSONDecodeError:
                        target_args = {}
                else:
                    target_args = target_args_str
            elif tcs and len(tcs) > 0:
                tc = tcs[0]
                target_name = tc.get("function", {}).get("name", "")
                target_args_str = tc.get("function", {}).get("arguments", "{}")
                if isinstance(target_args_str, str):
                    try:
                        target_args = json.loads(target_args_str)
                    except json.JSONDecodeError:
                        target_args = {}
                else:
                    target_args = target_args_str
            else:
                # Assistant message without tool call (text response, e.g., greeting
                # or clarification). Add to context and continue looking for tool calls.
                context_prefix.append(msg)
                i += 1
                continue

            # Create a sample for this turn
            # Strip internal _tool_call_id fields from context messages
            clean_context = [
                {k: v for k, v in m.items() if k != "_tool_call_id"}
                for m in context_prefix
            ]
            sample = {
                "id": f"{trace.get('id', trace_idx)}_turn{len(samples)}",
                "question": user_msg.get("content", ""),
                "system_prompt": system_content,
                "tools": tools,
                "context_messages": clean_context,
                "target_tool_name": target_name,
                "target_arguments": target_args,
            }
            samples.append(sample)

            # Add this assistant message + its tool result to context for next turn
            # Normalize deprecated function_call format to tool_calls format
            normalized_assistant = _normalize_message(msg)
            context_prefix.append(normalized_assistant)
            i += 1

            # Consume the following function/tool result message
            if i < len(turn_messages) and turn_messages[i].get("role") in ("function", "tool"):
                result_msg = turn_messages[i]
                # Link tool_call_id from the assistant message
                if result_msg.get("role") == "function":
                    result_msg = dict(result_msg)
                    result_msg["_tool_call_id"] = normalized_assistant.get("_tool_call_id", "call_unknown")
                normalized_result = _normalize_message(result_msg)
                context_prefix.append(normalized_result)
                i += 1

    return samples


def _process_hf_row(row: dict) -> Optional[dict]:
    """Process a HuggingFace dataset row into tool-call training format."""
    messages = row.get("messages", [])
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return None

    if len(messages) < 3:
        return None

    # Extract gold tool call from assistant message
    assistant_msg = messages[2]
    if assistant_msg.get("role") != "assistant":
        return None

    function_call = assistant_msg.get("function_call")
    tool_calls = assistant_msg.get("tool_calls")

    if function_call:
        target_name = function_call["name"]
        target_args_str = function_call.get("arguments", "{}")
        if isinstance(target_args_str, str):
            try:
                target_args = json.loads(target_args_str)
            except json.JSONDecodeError:
                target_args = {}
        else:
            target_args = target_args_str
    elif tool_calls and len(tool_calls) > 0:
        tc = tool_calls[0]
        target_name = tc["function"]["name"]
        target_args_str = tc["function"].get("arguments", "{}")
        if isinstance(target_args_str, str):
            try:
                target_args = json.loads(target_args_str)
            except json.JSONDecodeError:
                target_args = {}
        else:
            target_args = target_args_str
    else:
        return None

    tools = row.get("available_tools", [])
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None

    if not tools:
        return None

    system_prompt = messages[0].get("content", "") if messages[0].get("role") == "system" else ""

    return {
        "id": row.get("uuid", ""),
        "question": row.get("question", ""),
        "tools": tools,
        "target_tool_name": target_name,
        "target_arguments": target_args,
        "system_prompt": system_prompt,
    }


# ---------------------------------------------------------------------------
# Backend: ART-based LoRA GRPO
# ---------------------------------------------------------------------------

class ARTLoRAGRPOBackend(Backend):
    """ART backend for LoRA + GRPO training.

    Uses the ART framework (OpenPipe) which manages:
    - vLLM inference server (co-located, time-shared with training)
    - Unsloth LoRA adapter training
    - GRPO optimization with trajectory groups

    The backend supports two modes:
    1. Built-in tool-call verification (data_path provided, no rollout_fn)
    2. Custom rollout function (rollout_fn provided by user)
    """

    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute LoRA GRPO training using ART.

        Runs in a subprocess to isolate the caller from ART/vLLM's non-daemon
        threads that prevent clean Python exit. The subprocess performs proper
        engine shutdown (freeing GPU), saves results to disk, then force-exits.
        The parent reads results from disk.
        """
        import multiprocessing as mp

        ckpt_output_dir = algorithm_params["ckpt_output_dir"]
        os.makedirs(ckpt_output_dir, exist_ok=True)
        results_path = os.path.join(ckpt_output_dir, "training_results.json")
        error_path = os.path.join(ckpt_output_dir, ".training_error")

        if os.path.exists(error_path):
            os.remove(error_path)

        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=self._subprocess_entry,
            args=(algorithm_params, results_path, error_path),
        )
        proc.start()
        proc.join()

        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
            os.remove(error_path)
            raise RuntimeError(f"GRPO training failed: {error_msg}")

        if not os.path.exists(results_path):
            raise RuntimeError(f"GRPO training subprocess exited with code {proc.exitcode}")

        with open(results_path) as f:
            return json.load(f)

    @staticmethod
    def _subprocess_entry(algorithm_params, results_path, error_path):
        """Subprocess entry point for training."""
        os.environ["VLLM_USE_V1"] = "0"
        try:
            import art
            from art.local.backend import LocalBackend
            result = asyncio.run(
                ARTLoRAGRPOBackend()._run_training(algorithm_params, art, LocalBackend)
            )
            with open(results_path, "w") as f:
                json.dump(result, f, indent=2)
        except SystemExit:
            pass  # os._exit from shutdown — results already on disk
        except Exception as e:
            with open(error_path, "w") as f:
                f.write(str(e))

    async def _run_training(self, params: Dict[str, Any], art, LocalBackend) -> Dict[str, Any]:
        """Async training loop."""
        # Extract parameters
        model_path = params["model_path"]
        ckpt_output_dir = params["ckpt_output_dir"]

        # GRPO hyperparameters
        num_iterations = params.get("num_iterations", 15)
        group_size = params.get("group_size", 8)
        tasks_per_iteration = params.get("tasks_per_iteration", 100)
        learning_rate = params.get("learning_rate", 1e-5)
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 512)
        concurrency = params.get("concurrency", 32)

        # LoRA configuration
        lora_r = params.get("lora_r", 16)
        lora_alpha = params.get("lora_alpha", 8)
        target_modules = params.get("target_modules")
        max_grad_norm = params.get("max_grad_norm", 0.1)

        # vLLM configuration
        gpu_memory_utilization = params.get("gpu_memory_utilization", 0.45)
        max_lora_rank = params.get("max_lora_rank", None)
        # Custom rollout/reward
        rollout_fn = params.get("rollout_fn")
        tasks = params.get("tasks")
        reward_fn = params.get("reward_fn")

        # Dataset (for built-in tool-call mode)
        data_path = params.get("data_path")
        data_config = params.get("data_config", "Qwen3")
        n_train = params.get("n_train", 5000)
        n_val = params.get("n_val", 500)

        # Callbacks
        iteration_callback = params.get("iteration_callback")

        # Experiment tracking (with env var fallbacks, matching other algorithms)
        wandb_project = params.get("wandb_project") or os.environ.get("WANDB_PROJECT")
        wandb_entity = params.get("wandb_entity") or os.environ.get("WANDB_ENTITY")
        wandb_run_name = params.get("wandb_run_name") or os.environ.get("WANDB_RUN_NAME")
        mlflow_tracking_uri = params.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")

        if mlflow_tracking_uri:
            logger.warning(
                "MLflow is not supported by the ART backend. "
                "Use the verl backend for MLflow experiment tracking."
            )

        # Set W&B env vars so ART picks them up automatically
        if wandb_entity:
            os.environ["WANDB_ENTITY"] = wandb_entity

        # Model name for ART registration (also used as W&B run name)
        model_name_slug = model_path.split("/")[-1].lower().replace(".", "-")
        art_model_name = wandb_run_name or params.get("art_model_name", f"{model_name_slug}-grpo")
        art_project = wandb_project or params.get("art_project", "training-hub-grpo")
        art_path = params.get("art_path", os.path.join(ckpt_output_dir, ".art"))

        # Resolve mode: built-in tool-call vs custom rollout
        if rollout_fn is not None:
            if tasks is None:
                raise ValueError(
                    "When using a custom rollout_fn, you must also provide 'tasks' "
                    "(a list of task objects to pass to your rollout function)."
                )
            train_data = tasks
            val_data = []
            mode = "custom"
        elif data_path is not None:
            train_data, val_data = _load_tool_call_dataset(
                data_path, n_train=n_train, n_val=n_val, data_config=data_config,
            )
            mode = "tool_call"
        else:
            raise ValueError(
                "Either 'data_path' (for built-in tool-call verification) or "
                "'rollout_fn' + 'tasks' (for custom rollouts) must be provided."
            )

        # Build ART model config
        peft_kwargs = {"lora_alpha": lora_alpha}
        if lora_r != 16:
            peft_kwargs["r"] = lora_r
        if target_modules is not None:
            peft_kwargs["target_modules"] = target_modules

        init_kwargs = {"gpu_memory_utilization": gpu_memory_utilization}

        engine_kwargs = {}
        effective_max_lora_rank = max_lora_rank if max_lora_rank else lora_r
        if effective_max_lora_rank > 16:
            engine_kwargs["max_lora_rank"] = effective_max_lora_rank

        internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(**init_kwargs),
            peft_args=art.dev.PeftArgs(**peft_kwargs),
            trainer_args=art.dev.TrainerArgs(max_grad_norm=max_grad_norm),
            **({"engine_args": art.dev.EngineArgs(**engine_kwargs)} if engine_kwargs else {}),
        )

        model = art.TrainableModel(
            name=art_model_name,
            project=art_project,
            base_model=model_path,
            _internal_config=internal_config,
        )

        # Register with backend
        backend = LocalBackend(in_process=True, path=art_path)
        await model.register(backend)
        logger.info("Model registered with ART backend at %s", art_path)

        try:
            return await self._run_training_loop(
                model, backend, art, train_data,
                mode=mode,
                rollout_fn=rollout_fn,
                reward_fn=reward_fn,
                num_iterations=num_iterations,
                group_size=group_size,
                tasks_per_iteration=tasks_per_iteration,
                learning_rate=learning_rate,
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency=concurrency,
                ckpt_output_dir=ckpt_output_dir,
                art_path=art_path,
                model_path=model_path,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                iteration_callback=iteration_callback,
            )
        finally:
            logger.info("Shutting down ART backend...")
            await _shutdown_art_backend(backend)
            logger.info("ART backend shut down")

    async def _run_training_loop(
        self, model, backend, art, train_data, *,
        mode, rollout_fn, reward_fn,
        num_iterations, group_size, tasks_per_iteration,
        learning_rate, temperature, max_tokens, concurrency,
        ckpt_output_dir, art_path, model_path, lora_r, lora_alpha,
        iteration_callback,
    ) -> Dict[str, Any]:
        """Inner training loop."""
        # Build the rollout function for built-in tool-call mode
        sem = asyncio.Semaphore(concurrency)
        actual_reward_fn = reward_fn or tool_call_reward

        if mode == "tool_call":
            async def _builtin_rollout(mdl, sample):
                return await self._tool_call_rollout(
                    mdl, sample, sem, art,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reward_fn=actual_reward_fn,
                )
            effective_rollout = _builtin_rollout
        else:
            # Wrap user's rollout_fn with semaphore for concurrency control
            async def _wrapped_rollout(mdl, task):
                async with sem:
                    return await rollout_fn(mdl, task)
            effective_rollout = _wrapped_rollout

        # Check for resume
        current_step = await model.get_step()
        start_iteration = current_step if current_step > 0 else 0
        if start_iteration > 0:
            logger.info("Resuming from step %d", start_iteration)

        os.makedirs(ckpt_output_dir, exist_ok=True)
        rollouts_per_iter = tasks_per_iteration * group_size

        # Training loop
        reward_history = []
        timing_history = []
        full_match_history = []
        start_time = time.time()

        logger.info(
            "Starting GRPO training: %d iterations, %d tasks/iter, group_size=%d, "
            "lr=%s, mode=%s",
            num_iterations, tasks_per_iteration, group_size, learning_rate, mode,
        )

        for iteration in range(start_iteration, num_iterations):
            iter_start = time.time()

            # Sample tasks for this iteration
            random.seed(42 + iteration)
            n_tasks = min(tasks_per_iteration, len(train_data))
            iter_samples = random.sample(train_data, n_tasks)

            # Gather trajectory groups
            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        effective_rollout(model, sample) for _ in range(group_size)
                    )
                    for sample in iter_samples
                ),
                pbar_desc=f"Iter {iteration + 1}/{num_iterations}",
            )

            # Compute metrics
            rewards = [t.reward for g in train_groups for t in g.trajectories]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            full_match = sum(1 for r in rewards if r >= 1.0) / len(rewards) if rewards else 0.0
            name_match = sum(1 for r in rewards if r >= 0.5) / len(rewards) if rewards else 0.0

            reward_history.append(mean_reward)
            full_match_history.append(full_match)

            # Write rollout metrics to training_metrics.jsonl immediately
            # (before model.train which may os._exit on final iteration)
            metrics_path = os.path.join(ckpt_output_dir, "training_metrics.jsonl")
            rollout_entry = {
                "step": iteration + 1,
                "epoch": (iteration + 1) / num_iterations,
                "phase": "rollout",
                "mean_reward": mean_reward,
                "full_match_rate": full_match,
                "name_match_rate": name_match,
                "total_rollouts": len(rewards),
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rollout_entry) + "\n")
                f.flush()

            # Save results before training (survives os._exit)
            elapsed = time.time() - start_time
            results = {
                "framework": "art",
                "algorithm": "lora_grpo",
                "base_model": model_path,
                "mode": mode,
                "iteration": iteration + 1,
                "num_iterations": num_iterations,
                "group_size": group_size,
                "tasks_per_iteration": tasks_per_iteration,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "concurrency": concurrency,
                "reward_history": reward_history,
                "full_match_history": full_match_history,
                "timing_history": timing_history,
                "total_time_seconds": elapsed,
                "total_rollouts": (iteration + 1) * rollouts_per_iter,
                "final_mean_reward": mean_reward,
                "final_full_match": full_match,
            }
            results_path = os.path.join(ckpt_output_dir, "training_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            # Train GRPO step — ART writes loss metrics to its own
            # history.jsonl live during training
            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=learning_rate),
            )

            iter_time = time.time() - iter_start
            timing_history.append(iter_time)

            # Write training metrics (loss/grad/entropy) from ART's history.
            # ART writes history.jsonl live during model.train(), so this
            # data is available immediately after training completes.
            history_path = os.path.join(
                art_path, art_project, "models", art_model_name, "history.jsonl"
            )
            train_entry = {
                "step": iteration + 1,
                "epoch": (iteration + 1) / num_iterations,
                "phase": "train",
                "wall_time_s": iter_time,
            }
            if os.path.exists(history_path):
                try:
                    with open(history_path) as hf:
                        last_line = None
                        for last_line in hf:
                            pass
                        if last_line:
                            h = json.loads(last_line)
                            train_entry["loss"] = h.get("loss/train")
                            train_entry["grad_norm"] = h.get("loss/grad_norm")
                            train_entry["learning_rate"] = h.get("loss/learning_rate")
                            train_entry["entropy"] = h.get("loss/entropy")
                except Exception:
                    pass
            with open(metrics_path, "a") as f:
                f.write(json.dumps(train_entry) + "\n")
                f.flush()

            logger.info(
                "Iter %d/%d: mean_reward=%.3f full_match=%.1f%% name_match=%.1f%% "
                "rollouts=%d time=%.0fs elapsed=%.0fs",
                iteration + 1, num_iterations,
                mean_reward, full_match * 100, name_match * 100,
                len(rewards), iter_time, elapsed,
            )

            # User callback
            if iteration_callback is not None:
                iteration_callback(iteration + 1, results)

        total_time = time.time() - start_time

        logger.info(
            "GRPO training complete: %d iterations, %d total rollouts, "
            "final_reward=%.3f, total_time=%.0fs (%.1fh)",
            num_iterations, num_iterations * rollouts_per_iter,
            reward_history[-1] if reward_history else 0.0,
            total_time, total_time / 3600,
        )

        return {
            "status": "success",
            "checkpoint_path": art_path,
            "reward_history": reward_history,
            "full_match_history": full_match_history,
            "timing_history": timing_history,
            "total_time_seconds": total_time,
            "total_rollouts": num_iterations * rollouts_per_iter,
            "final_mean_reward": reward_history[-1] if reward_history else 0.0,
        }

    async def _tool_call_rollout(
        self,
        model,
        sample: dict,
        sem: asyncio.Semaphore,
        art,
        temperature: float = 0.7,
        max_tokens: int = 512,
        reward_fn: Callable = tool_call_reward,
    ):
        """Built-in rollout for tool-call verification (single-turn and multi-turn).

        For single-turn samples: messages = [system, user] → model generates tool call.
        For multi-turn samples: messages = [system, user, ...gt_context] → model generates
        the next tool call given ground-truth prefix context.
        """
        async with sem:
            client = model.openai_client()
            model_name = model.get_inference_name()

            # Build message list: system + user + optional GT context prefix
            messages = [
                {"role": "system", "content": sample.get("system_prompt", "")},
                {"role": "user", "content": sample["question"]},
            ]
            context_messages = sample.get("context_messages", [])
            messages.extend(context_messages)

            # Tools may be a list (single-turn) or empty (multi-turn, embedded in system prompt)
            tools = sample.get("tools") or None
            create_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                create_kwargs["tools"] = tools

            try:
                response = await client.chat.completions.create(**create_kwargs)

                choice = response.choices[0]

                reward = reward_fn(
                    choice,
                    expected_name=sample["target_tool_name"],
                    expected_args=sample["target_arguments"],
                )

                # Trajectory: all context messages as dicts (not trainable),
                # then the model's Choice (trainable, carries logprobs)
                messages_and_choices = []
                messages_and_choices.append(messages[0])  # system
                messages_and_choices.append(messages[1])  # user
                messages_and_choices.extend(context_messages)  # GT context (dicts, not trainable)
                messages_and_choices.append(choice)  # model output (trainable)

                trajectory = art.Trajectory(messages_and_choices=messages_and_choices)
                trajectory.reward = reward
                return trajectory

            except Exception as e:
                logger.warning("Rollout failed for sample %s: %s", sample.get("id", "?"), e)
                trajectory = art.Trajectory(
                    messages_and_choices=[
                        {"role": "system", "content": "failed"},
                        {"role": "user", "content": "failed"},
                    ]
                )
                trajectory.reward = 0.0
                return trajectory


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class LoRAGRPOAlgorithm(Algorithm):
    """LoRA + GRPO algorithm for reinforcement learning from verifiable rewards.

    Combines LoRA parameter-efficient training with GRPO optimization.
    Supports single-turn (tool-call verification) and multi-turn (custom rollout)
    training modes.
    """

    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend
        self.config = kwargs

    def train(
        self,
        model_path: str,
        ckpt_output_dir: str,
        # Data source (for built-in tool-call mode)
        data_path: Optional[str] = None,
        data_config: Optional[str] = None,
        n_train: Optional[int] = None,
        n_val: Optional[int] = None,
        # Custom rollout mode
        rollout_fn: Optional[Callable] = None,
        tasks: Optional[List[Any]] = None,
        reward_fn: Optional[Callable] = None,
        # GRPO hyperparameters
        num_iterations: Optional[int] = None,
        group_size: Optional[int] = None,
        tasks_per_iteration: Optional[int] = None,
        learning_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        concurrency: Optional[int] = None,
        # LoRA configuration
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        target_modules: Optional[List[str]] = None,
        max_grad_norm: Optional[float] = None,
        # vLLM configuration
        gpu_memory_utilization: Optional[float] = None,
        max_lora_rank: Optional[int] = None,
        # Multi-GPU configuration (verl backend)
        n_gpus: Optional[int] = None,
        tensor_parallel_size: Optional[int] = None,
        # ART configuration
        art_model_name: Optional[str] = None,
        art_project: Optional[str] = None,
        art_path: Optional[str] = None,
        # Callbacks
        iteration_callback: Optional[Callable] = None,
        # Logging / experiment tracking
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Execute LoRA + GRPO training.

        Two modes of operation:

        1. Built-in tool-call verification (provide data_path):
           Uses a tool-calling dataset (e.g., Toucan-1.5M) with built-in
           single-turn rollout and tool_call_reward verification.

        2. Custom rollout (provide rollout_fn + tasks):
           User supplies an async rollout function and task list. The rollout
           function receives (model, task) and returns an art.Trajectory with
           reward set. Supports both single-turn and multi-turn traces.

        Args:
            model_path: HuggingFace model ID or local path to base model.
            ckpt_output_dir: Directory to save checkpoints and results.

            Built-in Tool-Call Mode:
                data_path: HuggingFace dataset ID or local JSON/JSONL path.
                data_config: HuggingFace dataset config (default: 'Qwen3').
                n_train: Number of training samples (default: 5000).
                n_val: Number of validation samples (default: 500).

            Custom Rollout Mode:
                rollout_fn: Async function (model, task) -> art.Trajectory.
                    The returned trajectory must have .reward set.
                    Must be a top-level function (not a lambda or closure),
                    as it is serialized to a subprocess via pickle.
                tasks: List of task objects passed to rollout_fn.
                reward_fn: Optional reward function to override the default
                    tool_call_reward. Signature: (response, expected_name, expected_args) -> float.

            GRPO Hyperparameters:
                num_iterations: Number of GRPO training iterations (default: 15).
                group_size: Rollouts per task for advantage estimation (default: 8).
                tasks_per_iteration: Tasks sampled per iteration (default: 100).
                learning_rate: Learning rate (default: 1e-5).
                temperature: Sampling temperature for rollouts (default: 0.7).
                max_tokens: Max tokens per rollout response (default: 512).
                max_prompt_length: Max prompt length in tokens; prompts exceeding
                    this are filtered (verl backend only, default: 16384).
                concurrency: Max concurrent rollouts (default: 32).

            LoRA Configuration:
                lora_r: LoRA rank (default: 16).
                lora_alpha: LoRA alpha (default: 8).
                target_modules: Modules to apply LoRA to (default: auto).
                max_grad_norm: Gradient clipping norm (default: 0.1).

            vLLM Configuration:
                gpu_memory_utilization: GPU memory fraction for vLLM (default: 0.45).
                max_lora_rank: Max LoRA rank for vLLM engine (default: matches lora_r).
            ART Configuration:
                art_model_name: Model name for ART registration.
                art_project: ART project name.
                art_path: Path for ART local backend storage.

            Callbacks:
                iteration_callback: Called after each iteration with (iteration, results_dict).

            Experiment Tracking:
                wandb_project: Weights & Biases project name.
                wandb_entity: Weights & Biases team/entity name.
                wandb_run_name: Weights & Biases run name.
                mlflow_tracking_uri: MLflow tracking server URI.
                mlflow_experiment_name: MLflow experiment name.
                mlflow_run_name: MLflow run name.

        Returns:
            Dict with training results including reward_history, timing, and checkpoint path.
        """
        params = {
            "model_path": model_path,
            "ckpt_output_dir": ckpt_output_dir,
        }

        optional_params = {
            "data_path": data_path,
            "data_config": data_config,
            "n_train": n_train,
            "n_val": n_val,
            "rollout_fn": rollout_fn,
            "tasks": tasks,
            "reward_fn": reward_fn,
            "num_iterations": num_iterations,
            "group_size": group_size,
            "tasks_per_iteration": tasks_per_iteration,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_prompt_length": max_prompt_length,
            "concurrency": concurrency,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
            "max_grad_norm": max_grad_norm,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_lora_rank": max_lora_rank,
            "n_gpus": n_gpus,
            "tensor_parallel_size": tensor_parallel_size,
            "art_model_name": art_model_name,
            "art_project": art_project,
            "art_path": art_path,
            "iteration_callback": iteration_callback,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "wandb_run_name": wandb_run_name,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
            "mlflow_run_name": mlflow_run_name,
        }

        for key, value in optional_params.items():
            if value is not None:
                params[key] = value

        params.update(kwargs)

        return self.backend.execute_training(params)

    def get_required_params(self) -> Dict[str, Type]:
        return {
            "model_path": str,
            "ckpt_output_dir": str,
        }

    def get_optional_params(self) -> Dict[str, Type]:
        return {
            # Data source
            "data_path": str,
            "data_config": str,
            "n_train": int,
            "n_val": int,
            # Custom rollout
            "rollout_fn": Callable,
            "tasks": list,
            "reward_fn": Callable,
            # GRPO hyperparameters
            "num_iterations": int,
            "group_size": int,
            "tasks_per_iteration": int,
            "learning_rate": float,
            "temperature": float,
            "max_tokens": int,
            "max_prompt_length": int,
            "concurrency": int,
            # LoRA
            "lora_r": int,
            "lora_alpha": int,
            "target_modules": list,
            "max_grad_norm": float,
            # vLLM
            "gpu_memory_utilization": float,
            "max_lora_rank": int,
            "n_gpus": int,
            "tensor_parallel_size": int,
            # ART
            "art_model_name": str,
            "art_project": str,
            "art_path": str,
            # Callbacks
            "iteration_callback": Callable,
            # Logging / experiment tracking
            "wandb_project": str,
            "wandb_entity": str,
            "wandb_run_name": str,
            "mlflow_tracking_uri": str,
            "mlflow_experiment_name": str,
            "mlflow_run_name": str,
        }


# Register algorithm and backend
AlgorithmRegistry.register_algorithm("lora_grpo", LoRAGRPOAlgorithm)
AlgorithmRegistry.register_backend("lora_grpo", "art", ARTLoRAGRPOBackend)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def lora_grpo(
    model_path: str,
    ckpt_output_dir: str,
    # Data source (built-in tool-call mode)
    data_path: Optional[str] = None,
    data_config: str = "Qwen3",
    n_train: int = 5000,
    n_val: int = 500,
    # Custom rollout mode
    rollout_fn: Optional[Callable] = None,
    tasks: Optional[List[Any]] = None,
    reward_fn: Optional[Callable] = None,
    # GRPO hyperparameters
    num_iterations: int = 15,
    group_size: int = 8,
    tasks_per_iteration: int = 100,
    learning_rate: float = 1e-5,
    temperature: float = 0.7,
    max_tokens: int = 512,
    max_prompt_length: int = 16384,
    concurrency: int = 32,
    # LoRA configuration
    lora_r: int = 16,
    lora_alpha: int = 8,
    target_modules: Optional[List[str]] = None,
    max_grad_norm: float = 0.1,
    # vLLM configuration
    gpu_memory_utilization: float = 0.45,
    max_lora_rank: Optional[int] = None,
    # Multi-GPU configuration (verl backend)
    n_gpus: int = 1,
    tensor_parallel_size: int = 1,
    # ART configuration
    art_model_name: Optional[str] = None,
    art_project: Optional[str] = None,
    art_path: Optional[str] = None,
    # Backend selection
    backend: str = "art",
    # Callbacks
    iteration_callback: Optional[Callable] = None,
    # Logging / experiment tracking
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
    **kwargs,
) -> Any:
    """Run LoRA + GRPO training for reinforcement learning from verifiable rewards.

    Two modes of operation:

    1. Built-in tool-call verification (provide data_path):
       Uses a tool-calling dataset with single-turn rollout and automatic
       reward computation based on tool call correctness.

    2. Custom rollout (provide rollout_fn + tasks):
       User supplies an async rollout function for arbitrary environments.
       Supports both single-turn and multi-turn agentic traces.

    Args:
        model_path: HuggingFace model ID or local path (e.g., 'Qwen/Qwen3-4B').
        ckpt_output_dir: Directory to save checkpoints and training results.

        data_path: Dataset for built-in tool-call mode. HuggingFace ID
            (e.g., 'Agent-Ark/Toucan-1.5M') or local JSON/JSONL path.
        data_config: HuggingFace dataset config name (default: 'Qwen3').
        n_train: Number of training samples to load (default: 5000).
        n_val: Number of validation samples to load (default: 500).

        rollout_fn: Async function with signature (model, task) -> art.Trajectory.
            The returned Trajectory must have .reward set.
        tasks: List of task objects passed to rollout_fn each iteration.
        reward_fn: Custom reward function for built-in tool-call mode.
            Signature: (response, expected_name, expected_args) -> float.

        num_iterations: GRPO training iterations (default: 15).
        group_size: Rollouts per task for advantage estimation (default: 8).
        tasks_per_iteration: Tasks sampled per iteration (default: 100).
        learning_rate: Learning rate (default: 1e-5).
        temperature: Sampling temperature (default: 0.7).
        max_tokens: Max response tokens per rollout (default: 512).
        max_prompt_length: Max prompt length in tokens (verl backend, default: 16384).
        concurrency: Max concurrent rollouts (default: 32).

        lora_r: LoRA rank (default: 16).
        lora_alpha: LoRA alpha scaling parameter (default: 8).
        target_modules: Modules to apply LoRA to (default: auto-detect).
        max_grad_norm: Gradient clipping norm (default: 0.1).

        gpu_memory_utilization: vLLM GPU memory fraction (default: 0.45).
        max_lora_rank: Max LoRA rank for vLLM engine (default: matches lora_r).
        art_model_name: Custom name for ART model registration.
        art_project: ART project name (default: 'training-hub-grpo').
        art_path: Path for ART backend storage.

        backend: Backend to use (default: 'art').
        iteration_callback: Callback after each iteration: (iteration_num, results_dict).

        wandb_project: Weights & Biases project name.
        wandb_entity: Weights & Biases team/entity name.
        wandb_run_name: Weights & Biases run name.
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_experiment_name: MLflow experiment name.
        mlflow_run_name: MLflow run name.

    Returns:
        Dict with keys: status, checkpoint_path, reward_history,
        full_match_history, timing_history, total_time_seconds,
        total_rollouts, final_mean_reward.

    Examples:
        # Single-turn tool-call verification with Toucan dataset
        result = lora_grpo(
            model_path="Qwen/Qwen3-4B",
            data_path="Agent-Ark/Toucan-1.5M",
            ckpt_output_dir="./grpo_output",
            num_iterations=15,
            group_size=8,
            tasks_per_iteration=100,
            lora_r=16,
            lora_alpha=8,
        )

        # Multi-turn custom rollout
        import art

        async def agent_rollout(model, task):
            client = model.openai_client()
            model_name = model.get_inference_name()
            messages = [{"role": "system", "content": task["system"]}]
            choices = []

            for turn in range(task["max_turns"]):
                response = await client.chat.completions.create(
                    model=model_name, messages=messages, tools=task["tools"],
                    temperature=0.7, max_tokens=1024,
                )
                choice = response.choices[0]
                choices.append(choice)
                # ... execute tool calls, append results ...

            # Build trajectory with interleaved messages and choices
            messages_and_choices = []
            for msg in messages:
                if msg["role"] == "assistant":
                    messages_and_choices.append(choices.pop(0))
                else:
                    messages_and_choices.append(msg)

            trajectory = art.Trajectory(messages_and_choices=messages_and_choices)
            trajectory.reward = evaluate_success(messages)
            return trajectory

        result = lora_grpo(
            model_path="Qwen/Qwen3-4B",
            ckpt_output_dir="./grpo_output",
            rollout_fn=agent_rollout,
            tasks=my_task_list,
            num_iterations=10,
            concurrency=16,
        )
    """
    from . import create_algorithm

    algorithm = create_algorithm("lora_grpo", backend)
    return algorithm.train(
        model_path=model_path,
        ckpt_output_dir=ckpt_output_dir,
        data_path=data_path,
        data_config=data_config,
        n_train=n_train,
        n_val=n_val,
        rollout_fn=rollout_fn,
        tasks=tasks,
        reward_fn=reward_fn,
        num_iterations=num_iterations,
        group_size=group_size,
        tasks_per_iteration=tasks_per_iteration,
        learning_rate=learning_rate,
        temperature=temperature,
        max_tokens=max_tokens,
        max_prompt_length=max_prompt_length,
        concurrency=concurrency,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        max_grad_norm=max_grad_norm,
        gpu_memory_utilization=gpu_memory_utilization,
        max_lora_rank=max_lora_rank,
        n_gpus=n_gpus,
        tensor_parallel_size=tensor_parallel_size,
        art_model_name=art_model_name,
        art_project=art_project,
        art_path=art_path,
        iteration_callback=iteration_callback,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
        **kwargs,
    )
