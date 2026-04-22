"""verl backend for LoRA + GRPO training.

Provides multi-GPU distributed GRPO training using the verl framework
(Volcano Engine Reinforcement Learning for LLMs). Uses FSDP for distributed
LoRA training and vLLM for parallel rollout generation.

This backend supports scaling to large models (70B+) across multiple GPUs,
unlike the ART backend which is limited to single-GPU time-sharing.

Usage:
    from training_hub import lora_grpo

    # Single-turn tool-call training on 8 GPUs
    result = lora_grpo(
        model_path="Qwen/Qwen3-4B",
        data_path="Agent-Ark/Toucan-1.5M",
        ckpt_output_dir="./grpo_output",
        backend="verl",
        num_iterations=15,
        group_size=8,
        n_gpus=8,
    )
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import Backend, AlgorithmRegistry

logger = logging.getLogger(__name__)


def _prepare_verl_data(
    data_path: str,
    output_dir: str,
    n_train: int = 5000,
    n_val: int = 500,
    data_config: str = "Qwen3",
    reward_type: str = "tool_call",
    system_prompt: str = None,
    data_source: str = None,
) -> tuple[str, str]:
    """Convert training data to verl's expected Parquet format.

    verl expects a Parquet file with at minimum a 'prompt' column containing
    chat messages (list of dicts with role/content). Additional columns like
    'ground_truth' are passed through to the reward function.

    Supports two reward types:
    - "tool_call": Tool-call verification (decomposed multi-turn traces)
    - "math": Math answer verification via verl's built-in verifiers
      (data should have 'problem' and 'answer' fields)

    Returns:
        (train_parquet_path, val_parquet_path)
    """
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")

    if reward_type == "math":
        return _prepare_math_data(
            data_path, train_path, val_path,
            n_train=n_train, n_val=n_val,
            system_prompt=system_prompt, data_source=data_source,
        )

    if reward_type == "custom":
        return _prepare_generic_data(
            data_path, train_path, val_path,
            n_train=n_train, n_val=n_val,
            system_prompt=system_prompt, data_source=data_source or "custom",
        )

    # Tool-call data loading + decomposition
    from .lora_grpo import _load_local_dataset, _load_tool_call_dataset

    if data_path.endswith((".json", ".jsonl")):
        train_data, val_data = _load_local_dataset(data_path, n_train, n_val)
    else:
        train_data, val_data = _load_tool_call_dataset(
            data_path, n_train=n_train, n_val=n_val, data_config=data_config,
        )

    def samples_to_parquet(samples, path):
        rows = []
        for s in samples:
            # Build the prompt messages (everything the model sees before generating)
            prompt = [{"role": "system", "content": s.get("system_prompt", "")}]
            prompt.append({"role": "user", "content": s["question"]})
            # Add GT context for multi-turn samples
            for ctx_msg in s.get("context_messages", []):
                prompt.append(ctx_msg)

            tools = s.get("tools", [])

            # Embed tools in the system prompt so the model sees tool definitions
            # during generation. This is simpler than configuring verl's tool agent
            # loop and works with the default SingleTurnAgentLoop.
            if tools:
                tools_text = json.dumps(tools, indent=2)
                tools_instruction = (
                    "\n\nYou have access to the following tools. To call a tool, "
                    "respond with a JSON object in this format: "
                    '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n\n'
                    f"Tools:\n{tools_text}"
                )
                if prompt and prompt[0]["role"] == "system":
                    prompt[0] = {
                        "role": "system",
                        "content": prompt[0]["content"] + tools_instruction,
                    }
                else:
                    prompt.insert(0, {"role": "system", "content": tools_instruction})

            row = {
                "prompt": prompt,
                "reward_model": {
                    "ground_truth": json.dumps({
                        "tool_name": s["target_tool_name"],
                        "tool_args": s["target_arguments"],
                    }),
                },
                "extra_info": {
                    "index": len(rows),
                },
                "data_source": "tool_call",
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        logger.info("Saved %d samples to %s", len(rows), path)
        return path

    samples_to_parquet(train_data, train_path)
    if val_data:
        samples_to_parquet(val_data, val_path)
    else:
        # verl requires a val file, create a small one from train
        samples_to_parquet(train_data[:min(50, len(train_data))], val_path)

    return train_path, val_path


def _prepare_generic_data(
    data_path: str,
    train_path: str,
    val_path: str,
    n_train: int = 5000,
    n_val: int = 500,
    system_prompt: str = None,
    data_source: str = "custom",
) -> tuple[str, str]:
    """Prepare generic data for verl GRPO with custom reward functions.

    Expects JSONL with 'question' and 'ground_truth' fields.
    Optionally, samples can include 'messages' (list of chat messages)
    instead of 'question'.

    The ground_truth is passed to the reward function via
    reward_model.ground_truth in the parquet schema.
    """
    import pandas as pd

    with open(data_path) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    samples = raw[:n_train]
    val_samples = raw[n_train:n_train + n_val] if n_val > 0 else []

    default_sys = system_prompt or ""

    def build_rows(data):
        rows = []
        for i, s in enumerate(data):
            # Build prompt from 'messages' or 'question'
            if "messages" in s:
                prompt = s["messages"]
            elif "question" in s:
                prompt = []
                if default_sys:
                    prompt.append({"role": "system", "content": default_sys})
                prompt.append({"role": "user", "content": s["question"]})
            else:
                logger.warning("Sample %d has no 'question' or 'messages', skipping", i)
                continue

            gt = s.get("ground_truth", s.get("answer", ""))

            rows.append({
                "data_source": data_source,
                "prompt": prompt,
                "reward_model": {"style": "rule", "ground_truth": str(gt)},
                "extra_info": {"index": i},
            })
        return rows

    train_rows = build_rows(samples)
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)

    if val_samples:
        val_rows = build_rows(val_samples)
        pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    else:
        pd.DataFrame(train_rows[:min(5, len(train_rows))]).to_parquet(val_path, index=False)

    logger.info("Prepared generic data: %d train, %d val", len(train_rows),
                len(val_samples) if val_samples else min(5, len(train_rows)))
    return train_path, val_path


def _prepare_math_data(
    data_path: str,
    train_path: str,
    val_path: str,
    n_train: int = 5000,
    n_val: int = 500,
    system_prompt: str = None,
    data_source: str = None,
) -> tuple[str, str]:
    """Prepare math problem data for verl GRPO training.

    Expects JSONL with 'problem' and 'answer' fields. Uses verl's built-in
    math reward verifiers (routes via data_source field).

    Returns:
        (train_parquet_path, val_parquet_path)
    """
    import pandas as pd

    if system_prompt is None:
        system_prompt = (
            "Please reason step by step, and put your final answer "
            "within \\boxed{}."
        )

    with open(data_path) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    # Auto-detect data_source from filename if not specified
    if data_source is None:
        basename = os.path.basename(data_path).lower()
        if "aime" in basename:
            # Extract year if present
            import re
            year_match = re.search(r'(\d{4})', basename)
            data_source = f"aime{year_match.group(1)}" if year_match else "aime"
        elif "gsm8k" in basename:
            data_source = "openai/gsm8k"
        elif "math" in basename:
            data_source = "lighteval/MATH"
        else:
            data_source = "math"

    rows = []
    for i, p in enumerate(raw):
        # Support both 'problem' and 'question' field names
        problem_text = p.get("problem") or p.get("question", "")
        answer = str(p.get("answer", ""))

        row = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem_text},
            ],
            "reward_model": {
                "ground_truth": answer,
            },
            "extra_info": {
                "index": i,
            },
            "data_source": data_source,
        }
        rows.append(row)

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val] if n_val > 0 else rows[:min(50, len(rows))]

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    logger.info("Saved %d math problems to %s (data_source=%s)", len(train_rows), train_path, data_source)

    return train_path, val_path


def _write_math_reward_function(output_dir: str) -> str:
    """Write a math answer verification reward function for verl.

    Uses math_verify for robust semantic comparison (handles LaTeX
    equivalence like \\frac{1}{2} == 0.5). Falls back to string
    matching if math_verify is not installed.

    Returns:
        Path to the reward function file.
    """
    reward_code = '''
"""Math answer verification reward function for verl GRPO training.

Extracts answers from \\boxed{} or <answer> tags and compares to ground truth
using math_verify for semantic equivalence. Returns 1.0 for correct, 0.0 for wrong.

Requires: pip install math-verify
"""
import re
import logging

logging.getLogger("math_verify").setLevel(logging.ERROR)


def math_compute_score(data_source, solution_str, ground_truth, **kwargs):
    """Compute reward for math answer verification.

    Extracts the last \\boxed{} from the response, then uses math_verify
    for semantic comparison with ground truth. Falls back to normalized
    string matching if math_verify fails.

    Args:
        data_source: Dataset identifier (e.g., "aime2024", "math")
        solution_str: The model's generated solution text
        ground_truth: The correct answer as a string

    Returns:
        float: 1.0 if correct, 0.0 if wrong
    """
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("ground_truth", ground_truth)
    ground_truth = str(ground_truth).strip()

    try:
        predicted = _extract_answer(solution_str)
        if predicted is None:
            return 0.0
        return 1.0 if _verify(predicted, ground_truth) else 0.0
    except Exception:
        return 0.0


def _extract_answer(text):
    """Extract answer from model output.

    Priority:
    1. Last \\boxed{...}
    2. <answer>...</answer> or <answer>... (stop consumed closing tag)
    3. Full text (let math_verify handle extraction)
    """
    if not text:
        return None

    # Try last \\boxed{...}
    boxed = _last_boxed(text)
    if boxed is not None:
        return _remove_boxed(boxed)

    # Try <answer>...</answer>
    m = re.search(r"<answer>\\s*(.+?)\\s*</answer>", text, re.DOTALL)
    if m:
        ans = m.group(1).strip()
        # Remove boxed wrapper if present inside
        inner_boxed = _last_boxed(ans)
        if inner_boxed:
            return _remove_boxed(inner_boxed)
        return ans

    # Try <answer>... at end (stop sequence consumed </answer>)
    m = re.search(r"<answer>\\s*(.+?)\\s*$", text, re.DOTALL)
    if m:
        ans = m.group(1).strip()
        inner_boxed = _last_boxed(ans)
        if inner_boxed:
            return _remove_boxed(inner_boxed)
        return ans

    # Fall back to full text (math_verify can often extract from it)
    return text.strip()


def _last_boxed(string):
    """Find the last \\boxed{...} expression using brace counting."""
    idx = string.rfind("\\\\boxed{")
    if idx < 0:
        idx = string.rfind("\\\\fbox{")
        if idx < 0:
            return None

    i = idx
    depth = 0
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                return string[idx:i + 1]
        i += 1
    return None


def _remove_boxed(s):
    """Remove \\boxed{} or \\fbox{} wrapper."""
    for prefix in ["\\\\boxed{", "\\\\fbox{"]:
        if s.startswith(prefix) and s.endswith("}"):
            return s[len(prefix):-1]
    if "\\\\boxed " in s:
        return s.split("\\\\boxed ")[-1].split("$")[0]
    return s


def _normalize_latex(text):
    """Normalize LaTeX variants for comparison."""
    text = re.sub(r"\\\\[dt]frac", r"\\\\frac", text)
    text = re.sub(r"\\\\frac\\s*([^{\\s])\\s*([^{\\s])", r"\\\\frac{\\1}{\\2}", text)
    return text


def _verify(predicted, ground_truth):
    """Verify answer using math_verify, with string fallback."""
    predicted = predicted.strip()
    ground_truth = ground_truth.strip()

    # Quick exact match
    if predicted == ground_truth:
        return True

    # Normalize and try again
    pred_norm = _normalize_latex(predicted)
    truth_norm = _normalize_latex(ground_truth)
    if pred_norm == truth_norm:
        return True

    # Use math_verify for semantic comparison
    try:
        import math_verify
        pred_parsed = math_verify.parse(pred_norm, parsing_timeout=5)
        truth_parsed = math_verify.parse(truth_norm, parsing_timeout=5)
        if not pred_parsed or not truth_parsed:
            return False
        return math_verify.verify(truth_parsed, pred_parsed)
    except ImportError:
        # Fallback: simple numeric comparison
        try:
            return abs(float(predicted) - float(ground_truth)) < 1e-6
        except (ValueError, TypeError):
            return False
    except Exception:
        return False
'''
    reward_path = os.path.join(output_dir, "verl_math_reward.py")
    with open(reward_path, "w") as f:
        f.write(reward_code)
    return reward_path


def _write_custom_reward_function(reward_fn, output_dir: str) -> str:
    """Write a user-provided reward function to a file for verl to load.

    The function is serialized via inspect.getsource. It must be a standalone
    function (no closures over local variables) with the verl reward signature:
        compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float

    If the function has a different name, a wrapper is generated that calls it
    as `compute_score`.

    Returns:
        Path to the reward function file.
    """
    import inspect
    import textwrap

    # If reward_fn is a string, treat it as a path to a reward file
    if isinstance(reward_fn, str):
        if os.path.isfile(reward_fn):
            logger.info("Using reward function file directly: %s", reward_fn)
            return reward_fn
        raise FileNotFoundError(f"Reward function file not found: {reward_fn}")

    source = inspect.getsource(reward_fn)
    source = textwrap.dedent(source)
    fn_name = reward_fn.__name__

    # If the function isn't named compute_score, add an alias
    alias = ""
    if fn_name != "compute_score":
        alias = f"\ncompute_score = {fn_name}\n"

    reward_path = os.path.join(output_dir, "verl_custom_reward.py")
    with open(reward_path, "w") as f:
        f.write(source + alias)

    logger.info("Wrote custom reward function to %s (fn=%s)", reward_path, fn_name)
    return reward_path


def _write_reward_function(output_dir: str) -> str:
    """Write the tool-call reward function for verl to load.

    verl loads custom reward functions from a file path + function name.
    This writes a standalone reward module that verl can import.

    Returns:
        Path to the reward function file.
    """
    reward_code = '''
"""Tool-call reward function for verl GRPO training."""
import json


def tool_call_compute_score(data_source, solution_str, ground_truth, **kwargs):
    """Compute reward for tool-call verification.

    Called by verl's reward manager for each generated response.

    Args:
        data_source: Dataset identifier (unused, always "tool_call")
        solution_str: The model's generated response text
        ground_truth: JSON string with {"tool_name": ..., "tool_args": ...}

    Returns:
        float: 1.0 (correct name + args), 0.5 (correct name), 0.0 (wrong/missing)
    """
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    expected_name = gt["tool_name"]
    expected_args = gt.get("tool_args", {})

    # Extract tool call from the model's response
    # verl passes the raw generated text — we need to parse tool calls from it
    predicted_name, predicted_args = _extract_tool_call_from_text(solution_str)

    if predicted_name is None or not isinstance(predicted_name, str):
        return 0.0

    if not _names_match(predicted_name, expected_name):
        return 0.0

    if _args_match(predicted_args, expected_args):
        return 1.0

    return 0.5


def _extract_tool_call_from_text(text):
    """Extract tool call from model-generated text.

    Handles Qwen3 format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    Also handles thinking blocks and various JSON formats.
    """
    if not text:
        return None, {}

    import re

    # Strip thinking blocks first
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Try <tool_call> tags (Qwen3 format) - use greedy match for nested braces
    tc_match = re.search(r'<tool_call>\\s*(.+?)\\s*</tool_call>', text, re.DOTALL)
    if tc_match:
        try:
            obj = json.loads(tc_match.group(1))
            if isinstance(obj, dict):
                return obj.get("name"), obj.get("arguments", {})
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object with name+arguments
    # Find all { } balanced blocks
    for match in re.finditer(r'\\{', text):
        start = match.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i+1])
                        if isinstance(obj, dict) and "name" in obj:
                            return obj["name"], obj.get("arguments", {})
                    except json.JSONDecodeError:
                        pass
                    break

    return None, {}


def _names_match(predicted, expected):
    if predicted == expected:
        return True
    if predicted.endswith(expected) or expected.endswith(predicted):
        return True
    p = predicted.lower().replace("-", "_").replace(".", "_")
    e = expected.lower().replace("-", "_").replace(".", "_")
    return p == e or p.endswith(e) or e.endswith(p)


def _args_match(predicted, expected):
    if isinstance(predicted, str):
        try:
            predicted = json.loads(predicted)
        except json.JSONDecodeError:
            return False
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError:
            return False
    if not predicted and not expected:
        return True
    if not predicted or not expected:
        return False
    try:
        return json.dumps(_normalize(predicted), sort_keys=True) == json.dumps(_normalize(expected), sort_keys=True)
    except (TypeError, ValueError):
        return False


def _normalize(args):
    normalized = {}
    for k, v in args.items():
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v = v.strip()
        normalized[k] = v
    return normalized
'''
    reward_path = os.path.join(output_dir, "verl_reward.py")
    with open(reward_path, "w") as f:
        f.write(reward_code)
    return reward_path


def _write_agent_loop(output_dir: str) -> str:
    """Write a single-turn tool-calling agent loop for verl.

    verl's default SingleTurnAgentLoop doesn't pass tools to the chat template.
    This custom agent loop passes tools from the dataset's tools_kwargs field,
    enabling models like Qwen3 to generate structured tool calls.

    Returns:
        Path to the agent loop module in output_dir.
    """
    import shutil

    # Copy the agent loop module to the output dir
    src = os.path.join(os.path.dirname(__file__), "verl_tool_agent.py")
    dst = os.path.join(output_dir, "verl_tool_agent.py")
    shutil.copy2(src, dst)

    # Try to install it into verl's agent_loop directory so Ray workers can
    # import it. This may fail on read-only container filesystems (e.g., KubeRay
    # custom images), which is fine — the module is still available via PYTHONPATH.
    try:
        import verl.experimental.agent_loop as agent_loop_pkg
        agent_loop_dir = os.path.dirname(agent_loop_pkg.__file__)
        installed_path = os.path.join(agent_loop_dir, "verl_tool_agent.py")
        shutil.copy2(src, installed_path)

        # Patch the __init__.py to import it
        init_path = os.path.join(agent_loop_dir, "__init__.py")
        with open(init_path, "r") as f:
            init_content = f.read()
        if "verl_tool_agent" not in init_content:
            with open(init_path, "a") as f:
                f.write("\nfrom .verl_tool_agent import SingleTurnToolAgentLoop\n")
                f.write("_ = [*_, SingleTurnToolAgentLoop]\n")
    except (PermissionError, OSError) as e:
        logger.warning("Could not install agent loop into verl package dir: %s", e)

    return dst


def _normalize_verl_checkpoints(checkpoint_dir: str) -> list[str]:
    """Merge tokenizer files into verl's lora_adapter dirs for direct vLLM use.

    verl saves adapter weights and tokenizer files in separate subdirs:
        global_step_N/actor/lora_adapter/{adapter_config.json, adapter_model.safetensors}
        global_step_N/actor/huggingface/{tokenizer files, config.json}

    This copies the huggingface files into lora_adapter/ so that directory
    is self-contained and directly loadable by vLLM as a LoRA adapter path.
    The full checkpoint (FSDP state, optimizer, etc.) is preserved for resume.

    Returns:
        List of paths to the usable lora_adapter checkpoint directories.
    """
    import shutil

    adapter_paths = []
    if not os.path.exists(checkpoint_dir):
        return adapter_paths

    for entry in sorted(os.listdir(checkpoint_dir)):
        if not entry.startswith("global_step_"):
            continue

        raw_dir = os.path.join(checkpoint_dir, entry)
        lora_dir = os.path.join(raw_dir, "actor", "lora_adapter")
        hf_dir = os.path.join(raw_dir, "actor", "huggingface")

        if not os.path.isdir(lora_dir):
            continue

        # Copy tokenizer/config files into lora_adapter dir
        if os.path.isdir(hf_dir):
            for f in os.listdir(hf_dir):
                dst = os.path.join(lora_dir, f)
                if not os.path.exists(dst):  # don't overwrite adapter files
                    shutil.copy2(os.path.join(hf_dir, f), dst)

        adapter_paths.append(lora_dir)
        logger.info("Normalized checkpoint: %s/actor/lora_adapter/", entry)

    return adapter_paths


class VeRLLoRAGRPOBackend(Backend):
    """verl backend for distributed multi-GPU LoRA + GRPO training.

    Uses verl's FSDP for distributed LoRA training and vLLM for parallel
    rollout generation. Supports scaling to 70B+ models across multiple GPUs.
    """

    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute LoRA GRPO training using verl."""
        ckpt_output_dir = algorithm_params["ckpt_output_dir"]
        os.makedirs(ckpt_output_dir, exist_ok=True)

        # Prepare data in verl's Parquet format
        data_path = algorithm_params.get("data_path")
        if data_path is None:
            raise ValueError("data_path is required for verl backend")

        reward_fn = algorithm_params.get("reward_fn")
        reward_type = algorithm_params.get("reward_type", "tool_call")
        if reward_fn is not None and reward_type == "tool_call":
            reward_type = "custom"

        data_dir = os.path.join(ckpt_output_dir, "verl_data")
        train_path, val_path = _prepare_verl_data(
            data_path=data_path,
            output_dir=data_dir,
            n_train=algorithm_params.get("n_train", 5000),
            n_val=algorithm_params.get("n_val", 500),
            data_config=algorithm_params.get("data_config", "Qwen3"),
            reward_type=reward_type,
            system_prompt=algorithm_params.get("system_prompt"),
            data_source=algorithm_params.get("data_source"),
        )

        # Write reward function (and agent loop for tool_call mode)
        reward_fn = algorithm_params.get("reward_fn")
        if reward_fn is not None:
            reward_path = _write_custom_reward_function(reward_fn, ckpt_output_dir)
        elif reward_type == "tool_call":
            reward_path = _write_reward_function(ckpt_output_dir)
            _write_agent_loop(ckpt_output_dir)
        elif reward_type == "math":
            reward_path = _write_math_reward_function(ckpt_output_dir)
        else:
            reward_path = None

        # Extract parameters
        model_path = algorithm_params["model_path"]
        num_iterations = algorithm_params.get("num_iterations", 15)
        group_size = algorithm_params.get("group_size", 8)
        tasks_per_iteration = algorithm_params.get("tasks_per_iteration", 100)
        learning_rate = algorithm_params.get("learning_rate", 1e-5)
        lora_r = algorithm_params.get("lora_r", 16)
        lora_alpha = algorithm_params.get("lora_alpha", 8)
        max_tokens = algorithm_params.get("max_tokens", 512)
        max_prompt_length = algorithm_params.get("max_prompt_length", 16384)
        temperature = algorithm_params.get("temperature", 0.7)
        gpu_memory_utilization = algorithm_params.get("gpu_memory_utilization", 0.35)
        n_gpus = algorithm_params.get("n_gpus", 1)
        nnodes = algorithm_params.get("nnodes", 1)
        tp_size = algorithm_params.get("tensor_parallel_size", 1)
        use_dr_grpo = algorithm_params.get("use_dr_grpo", True)

        # train_batch_size = number of prompts per batch. verl's rollout.n
        # (set from group_size) controls how many rollouts per prompt, so
        # batch size should NOT be multiplied by group_size.
        train_batch_size = tasks_per_iteration

        # Calculate steps per epoch for checkpoint frequency.
        # Default: save once per epoch. Can be overridden via saves_per_epoch.
        import pandas as pd
        train_dataset_size = len(pd.read_parquet(train_path))
        steps_per_epoch = max(1, (train_dataset_size + train_batch_size - 1) // train_batch_size)
        saves_per_epoch = algorithm_params.get("saves_per_epoch", 1)
        save_freq = max(1, steps_per_epoch // saves_per_epoch)

        # Build verl command
        cmd = [
            sys.executable, "-m", "verl.trainer.main_ppo",
            # Algorithm
            "algorithm.adv_estimator=grpo",
            "algorithm.use_kl_in_reward=False",
        ]

        # Dr. GRPO: no reference model, token-level normalization
        if use_dr_grpo:
            cmd.append("algorithm.norm_adv_by_std_in_grpo=False")
            logger.info("Using Dr. GRPO: no ref model, token-level loss normalization")

        cmd.extend([
            # Data
            f"data.train_files={train_path}",
            f"data.val_files={val_path}",
            f"data.train_batch_size={train_batch_size}",
            f"data.max_prompt_length={max_prompt_length}",
            f"data.max_response_length={max_tokens}",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            # Model + LoRA
            f"actor_rollout_ref.model.path={model_path}",
            f"actor_rollout_ref.model.lora_rank={lora_r}",
            f"actor_rollout_ref.model.lora_alpha={lora_alpha}",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            "actor_rollout_ref.model.use_remove_padding=True",
            # Actor (training)
            f"actor_rollout_ref.actor.optim.lr={learning_rate}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={min(train_batch_size, 256)}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.actor.entropy_coeff=0",
            "actor_rollout_ref.actor.fsdp_config.param_offload=False",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        ])

        # Actor KL and loss settings depend on Dr. GRPO vs standard
        if use_dr_grpo:
            cmd.extend([
                "actor_rollout_ref.actor.use_kl_loss=False",
                "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
            ])
        else:
            cmd.extend([
                "actor_rollout_ref.actor.use_kl_loss=True",
                "actor_rollout_ref.actor.kl_loss_coef=0.01",
                "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            ])

        cmd.extend([
            # Rollout (vLLM)
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={tp_size}",
            f"actor_rollout_ref.rollout.n={group_size}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}",
            f"actor_rollout_ref.rollout.max_model_len={max_prompt_length + max_tokens}",
            "actor_rollout_ref.rollout.load_format=safetensors",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={min(train_batch_size // max(n_gpus, 1), 2)}",
            # Reference model (only used when not Dr. GRPO)
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={min(train_batch_size // max(n_gpus, 1), 2)}",
            # Trainer
            "trainer.critic_warmup=0",
            f"trainer.n_gpus_per_node={n_gpus}",
            f"trainer.nnodes={nnodes}",
            f"trainer.total_epochs={num_iterations}",
            f"trainer.save_freq={save_freq}",
            "trainer.test_freq=-1",
            "trainer.val_before_train=False",
            "trainer.project_name=training-hub-grpo",
            "trainer.experiment_name=lora_grpo",
            f"trainer.default_local_dir={ckpt_output_dir}/checkpoints",
            "trainer.resume_mode=auto",
        ])

        # Reward configuration
        if reward_fn is not None:
            cmd.extend([
                f"reward.custom_reward_function.path={reward_path}",
                "reward.custom_reward_function.name=compute_score",
            ])
        elif reward_type == "tool_call":
            cmd.extend([
                f"reward.custom_reward_function.path={reward_path}",
                "reward.custom_reward_function.name=tool_call_compute_score",
            ])
        elif reward_type == "math":
            cmd.extend([
                f"reward.custom_reward_function.path={reward_path}",
                "reward.custom_reward_function.name=math_compute_score",
            ])

        # Experiment tracking (with env var fallbacks, matching other algorithms)
        loggers = ["console"]
        wandb_project = algorithm_params.get("wandb_project") or os.environ.get("WANDB_PROJECT")
        mlflow_tracking_uri = algorithm_params.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")

        if wandb_project:
            loggers.append("wandb")
            cmd.append(f"trainer.project_name={wandb_project}")
        if mlflow_tracking_uri:
            loggers.append("mlflow")
            mlflow_experiment_name = (
                algorithm_params.get("mlflow_experiment_name")
                or os.environ.get("MLFLOW_EXPERIMENT_NAME")
            )
            if mlflow_experiment_name and not wandb_project:
                # verl maps project_name → MLflow experiment
                cmd.append(f"trainer.project_name={mlflow_experiment_name}")

        wandb_run_name = algorithm_params.get("wandb_run_name") or os.environ.get("WANDB_RUN_NAME")
        mlflow_run_name = algorithm_params.get("mlflow_run_name") or os.environ.get("MLFLOW_RUN_NAME")
        if wandb_run_name or mlflow_run_name:
            experiment_name = wandb_run_name or mlflow_run_name
            cmd.append(f"trainer.experiment_name={experiment_name}")

        cmd.append(f'trainer.logger={json.dumps(loggers)}')

        logger.info(
            "Starting verl GRPO training: model=%s, n_gpus=%d, "
            "lora_r=%d, lr=%s, epochs=%d, batch=%d, group=%d",
            model_path, n_gpus, lora_r, learning_rate,
            num_iterations, train_batch_size, group_size,
        )

        # Run verl as a subprocess
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "true"
        env["NCCL_DEBUG"] = "WARN"
        env["VLLM_LOGGING_LEVEL"] = "WARN"
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"

        # Pass tracking env vars to subprocess
        wandb_entity = algorithm_params.get("wandb_entity") or os.environ.get("WANDB_ENTITY")
        if wandb_entity:
            env["WANDB_ENTITY"] = wandb_entity
        if mlflow_tracking_uri:
            env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        # Add output dir to PYTHONPATH so verl can import the custom agent loop
        env["PYTHONPATH"] = ckpt_output_dir + ":" + env.get("PYTHONPATH", "")

        # Stream output to stdout, log file, and parse metrics live
        import re
        log_path = os.path.join(ckpt_output_dir, "verl_training.log")
        metrics_path = os.path.join(ckpt_output_dir, "training_metrics.jsonl")
        reward_history = []
        step_pattern = re.compile(r'step:(\d+)\s*-\s*(.*)')

        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                # Stream to stdout live
                sys.stdout.write(line)
                sys.stdout.flush()
                # Save to log file
                log_file.write(line)
                log_file.flush()

                # Parse step metrics and write training_metrics.jsonl live
                match = step_pattern.search(line)
                if match:
                    step = int(match.group(1))
                    metrics_str = match.group(2)
                    raw = {}
                    for kv in metrics_str.split(" - "):
                        kv = kv.strip()
                        if ":" not in kv:
                            continue
                        key, val = kv.split(":", 1)
                        try:
                            raw[key.strip()] = float(val.strip())
                        except ValueError:
                            pass

                    entry = {
                        "step": step,
                        "epoch": raw.get("training/epoch", 0),
                        "mean_reward": raw.get("critic/score/mean", 0),
                        "loss": raw.get("actor/pg_loss"),
                        "grad_norm": raw.get("actor/grad_norm"),
                        "learning_rate": raw.get("actor/lr"),
                        "entropy": raw.get("actor/entropy"),
                        "response_length_mean": raw.get("response_length/mean"),
                        "prompt_length_mean": raw.get("prompt_length/mean"),
                        "wall_time_s": raw.get("timing_s/step"),
                    }
                    entry = {k: v for k, v in entry.items() if v is not None}
                    reward_history.append(entry.get("mean_reward", 0))

                    with open(metrics_path, "a") as mf:
                        mf.write(json.dumps(entry) + "\n")
                        mf.flush()

            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"verl training failed with exit code {proc.returncode}"
            )

        # Re-parse log file to fill in any gaps in training_metrics.jsonl
        # (covers steps that were missed due to stdout buffering)
        logged_steps = set()
        if os.path.exists(metrics_path):
            with open(metrics_path) as mf:
                for line in mf:
                    if line.strip():
                        logged_steps.add(json.loads(line).get("step"))

        with open(log_path) as lf:
            for line in lf:
                match = step_pattern.search(line)
                if match and int(match.group(1)) not in logged_steps:
                    step = int(match.group(1))
                    raw = {}
                    for kv in match.group(2).split(" - "):
                        kv = kv.strip()
                        if ":" not in kv:
                            continue
                        key, val = kv.split(":", 1)
                        try:
                            raw[key.strip()] = float(val.strip())
                        except ValueError:
                            pass
                    entry = {
                        "step": step,
                        "epoch": raw.get("training/epoch", 0),
                        "mean_reward": raw.get("critic/score/mean", 0),
                        "loss": raw.get("actor/pg_loss"),
                        "grad_norm": raw.get("actor/grad_norm"),
                        "learning_rate": raw.get("actor/lr"),
                        "entropy": raw.get("actor/entropy"),
                        "response_length_mean": raw.get("response_length/mean"),
                        "prompt_length_mean": raw.get("prompt_length/mean"),
                        "wall_time_s": raw.get("timing_s/step"),
                    }
                    entry = {k: v for k, v in entry.items() if v is not None}
                    reward_history.append(entry.get("mean_reward", 0))
                    with open(metrics_path, "a") as mf:
                        mf.write(json.dumps(entry) + "\n")

        # Post-training: normalize checkpoints into flat adapter dirs
        # (matching ART's format for uniform vLLM loading)
        checkpoint_dir = os.path.join(ckpt_output_dir, "checkpoints")
        adapter_checkpoints = _normalize_verl_checkpoints(checkpoint_dir)

        return {
            "status": "success",
            "checkpoint_path": checkpoint_dir,
            "adapter_checkpoints": adapter_checkpoints,
            "reward_history": reward_history,
            "backend": "verl",
            "model_path": model_path,
            "n_gpus": n_gpus,
            "lora_r": lora_r,
            "num_iterations": num_iterations,
        }


# Register the verl backend for the lora_grpo algorithm
# (algorithm is already registered by lora_grpo.py, we just add a new backend)
try:
    AlgorithmRegistry.register_backend("lora_grpo", "verl", VeRLLoRAGRPOBackend)
except ValueError:
    pass  # Algorithm not registered yet (import order), will be registered on first use
