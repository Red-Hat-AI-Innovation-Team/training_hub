"""Pre-flight validation and configuration optimization for training runs.

Validates training configurations before launch to catch common issues that
would otherwise only surface at runtime (often after minutes of setup):
- Batch divisibility constraints for distributed training
- GPU memory availability and stale process detection
- Dependency version compatibility
- Parameter conflicts and silent misconfigurations

Also provides auto-optimization of training parameters based on hardware
and model characteristics.

Usage:
    from training_hub.profiling.preflight import preflight_check, optimize_config

    # Validate a config will work
    issues = preflight_check(
        algorithm="lora_grpo",
        backend="verl",
        model_path="Qwen/Qwen3-4B",
        n_gpus=4,
        nnodes=1,
        prompt_batch_size=50,
        group_size=8,
        lora_r=128,
    )
    for issue in issues:
        print(f"[{issue.severity}] {issue.message}")

    # Auto-optimize parameters
    config = optimize_config(
        algorithm="lora_grpo",
        backend="verl",
        model_path="Qwen/Qwen3-4B",
        n_gpus=4,
    )
    print(config.params)  # Suggested parameters
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Issue:
    """A single validation issue found during pre-flight check."""
    severity: Severity
    category: str
    message: str
    fix_hint: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] [{self.category}] {self.message}"]
        if self.fix_hint:
            parts.append(f"  Fix: {self.fix_hint}")
        return "\n".join(parts)


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    processes: list[dict] = field(default_factory=list)

    @property
    def utilization_pct(self) -> float:
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100


@dataclass
class OptimizedConfig:
    """Result of configuration optimization."""
    params: dict[str, Any]
    reasoning: list[str]
    estimated_memory_per_gpu_gb: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


def get_gpu_info() -> list[GPUInfo]:
    """Query nvidia-smi for GPU status.

    Returns:
        List of GPUInfo objects, one per GPU. Empty list if nvidia-smi
        is not available or no GPUs are detected.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            gpu = GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                total_memory_mb=int(parts[2]),
                used_memory_mb=int(parts[3]),
                free_memory_mb=int(parts[4]),
            )
            gpus.append(gpu)

        # Get process info
        proc_result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if proc_result.returncode == 0 and proc_result.stdout.strip():
            # Map processes to GPU indices via UUID
            uuid_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            uuid_to_idx = {}
            if uuid_result.returncode == 0:
                for line in uuid_result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        uuid_to_idx[parts[1]] = int(parts[0])

            for line in proc_result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpu_uuid = parts[0]
                    gpu_idx = uuid_to_idx.get(gpu_uuid)
                    if gpu_idx is not None and gpu_idx < len(gpus):
                        gpus[gpu_idx].processes.append({
                            "pid": int(parts[1]),
                            "memory_mb": int(parts[2]) if parts[2].isdigit() else 0,
                            "name": parts[3],
                        })

        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


# ---------------------------------------------------------------------------
# Batch divisibility validation
# ---------------------------------------------------------------------------

def _check_batch_divisibility(
    algorithm: str,
    backend: str,
    prompt_batch_size: int,
    group_size: int,
    n_gpus: int,
    nnodes: int,
    ppo_mini_batch_size: Optional[int] = None,
) -> list[Issue]:
    """Validate batch divisibility constraints for verl backend."""
    issues = []

    if backend != "verl":
        return issues

    world_size = n_gpus * nnodes

    # Constraint 1: total_rollouts must be divisible by n_workers
    # verl uses n_gpus * nnodes * 4 agent loop workers
    total_rollouts = prompt_batch_size * group_size
    n_workers = world_size * 4

    if total_rollouts % n_workers != 0:
        suggested = prompt_batch_size
        while (suggested * group_size) % n_workers != 0:
            suggested += 1
        issues.append(Issue(
            severity=Severity.ERROR,
            category="batch_divisibility",
            message=(
                f"prompt_batch_size ({prompt_batch_size}) * group_size ({group_size}) "
                f"= {total_rollouts} is not divisible by n_workers "
                f"({n_workers} = n_gpus * nnodes * 4 = {n_gpus} * {nnodes} * 4)."
            ),
            fix_hint=f"Use prompt_batch_size={suggested} (next valid value).",
        ))

    # Constraint 2: ppo_mini_batch_size must be divisible by world_size
    if ppo_mini_batch_size is None:
        ppo_mini_batch_size = min(prompt_batch_size, 256)
    if ppo_mini_batch_size % world_size != 0:
        suggested_mbs = max(world_size, (ppo_mini_batch_size // world_size) * world_size)
        issues.append(Issue(
            severity=Severity.ERROR,
            category="batch_divisibility",
            message=(
                f"ppo_mini_batch_size ({ppo_mini_batch_size}) is not divisible by "
                f"world_size ({world_size} = n_gpus * nnodes = {n_gpus} * {nnodes})."
            ),
            fix_hint=(
                f"Use ppo_mini_batch_size={suggested_mbs}. "
                f"This is auto-calculated from prompt_batch_size, but must divide evenly across all GPUs."
            ),
        ))

    return issues


# ---------------------------------------------------------------------------
# GPU environment checks
# ---------------------------------------------------------------------------

def _check_gpu_environment(
    n_gpus: int,
    gpu_memory_utilization: float = 0.45,
    min_free_memory_pct: float = 80.0,
) -> list[Issue]:
    """Check GPU availability and for stale processes."""
    issues = []
    gpus = get_gpu_info()

    if not gpus:
        issues.append(Issue(
            severity=Severity.WARNING,
            category="gpu",
            message="Could not query GPU information (nvidia-smi not available or no GPUs detected).",
        ))
        return issues

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        visible_indices = [int(x.strip()) for x in visible_devices.split(",") if x.strip().isdigit()]
        available_gpus = [g for g in gpus if g.index in visible_indices]
    else:
        available_gpus = gpus

    if len(available_gpus) < n_gpus:
        issues.append(Issue(
            severity=Severity.ERROR,
            category="gpu",
            message=(
                f"Requested {n_gpus} GPUs but only {len(available_gpus)} are "
                f"available (CUDA_VISIBLE_DEVICES={visible_devices or 'not set'})."
            ),
            fix_hint=f"Set CUDA_VISIBLE_DEVICES to include {n_gpus} GPU indices.",
        ))

    # Check for stale processes / insufficient free memory
    for gpu in available_gpus[:n_gpus]:
        free_pct = (gpu.free_memory_mb / gpu.total_memory_mb * 100) if gpu.total_memory_mb > 0 else 0
        if free_pct < min_free_memory_pct:
            proc_desc = ""
            if gpu.processes:
                proc_desc = " Processes: " + ", ".join(
                    f"PID {p['pid']} ({p['memory_mb']}MB)" for p in gpu.processes
                )
            issues.append(Issue(
                severity=Severity.WARNING,
                category="gpu",
                message=(
                    f"GPU {gpu.index} ({gpu.name}) has only {free_pct:.0f}% free memory "
                    f"({gpu.free_memory_mb}MB / {gpu.total_memory_mb}MB).{proc_desc}"
                ),
                fix_hint=(
                    "Kill stale processes with: "
                    "nvidia-smi --query-compute-apps=pid --format=csv,noheader | "
                    "sort -u | xargs -r kill -9"
                ),
            ))

    # Estimate if memory is sufficient for the requested gpu_memory_utilization
    if available_gpus and n_gpus <= len(available_gpus):
        target_gpus = available_gpus[:n_gpus]
        min_free = min(g.free_memory_mb for g in target_gpus)
        min_total = min(g.total_memory_mb for g in target_gpus)
        required_mb = int(min_total * gpu_memory_utilization)
        if min_free < required_mb:
            issues.append(Issue(
                severity=Severity.WARNING,
                category="gpu",
                message=(
                    f"gpu_memory_utilization={gpu_memory_utilization} requires "
                    f"~{required_mb}MB free, but the least-free GPU only has "
                    f"{min_free}MB available."
                ),
                fix_hint="Free GPU memory or reduce gpu_memory_utilization.",
            ))

    return issues


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _check_dependencies(backend: str, algorithm: str) -> list[Issue]:
    """Check that required dependencies are importable and compatible."""
    issues = []

    if backend == "verl" or algorithm in ("lora_grpo", "grpo"):
        # Check verl
        try:
            import verl  # noqa: F401
        except ImportError:
            issues.append(Issue(
                severity=Severity.ERROR,
                category="dependency",
                message="verl is not installed.",
                fix_hint="pip install -e .[grpo]",
            ))

        # Check vllm
        try:
            import vllm  # noqa: F401
        except ImportError:
            issues.append(Issue(
                severity=Severity.ERROR,
                category="dependency",
                message="vllm is not installed.",
                fix_hint="pip install vllm",
            ))

        # Check flashinfer version consistency
        try:
            import importlib.metadata as metadata
            fi_python = metadata.version("flashinfer-python")
            try:
                fi_cubin = metadata.version("flashinfer-cubin")
                # Compare major.minor.patch (ignore post releases for comparison)
                fi_py_base = fi_python.split(".post")[0]
                fi_cu_base = fi_cubin.split(".post")[0]
                if fi_py_base != fi_cu_base:
                    issues.append(Issue(
                        severity=Severity.ERROR,
                        category="dependency",
                        message=(
                            f"flashinfer version mismatch: flashinfer-python={fi_python}, "
                            f"flashinfer-cubin={fi_cubin}. This will crash vLLM's engine core."
                        ),
                        fix_hint=f"pip install flashinfer-cubin=={fi_py_base}",
                    ))
            except metadata.PackageNotFoundError:
                pass  # cubin not installed separately, OK
        except Exception:
            pass

    if backend == "art" or (algorithm == "lora_grpo" and backend == "art"):
        try:
            import art  # noqa: F401
        except ImportError:
            issues.append(Issue(
                severity=Severity.ERROR,
                category="dependency",
                message="openpipe-art is not installed.",
                fix_hint="pip install -e .[grpo]",
            ))

    if algorithm in ("lora_sft",):
        try:
            import unsloth  # noqa: F401
        except ImportError:
            issues.append(Issue(
                severity=Severity.ERROR,
                category="dependency",
                message="unsloth is not installed.",
                fix_hint="pip install -e .[lora]",
            ))

    if algorithm == "osft":
        try:
            import mini_trainer  # noqa: F401
        except ImportError:
            issues.append(Issue(
                severity=Severity.ERROR,
                category="dependency",
                message="mini-trainer is not installed.",
                fix_hint="pip install -e .",
            ))

    return issues


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def _check_parameter_conflicts(
    algorithm: str,
    backend: str,
    params: dict[str, Any],
) -> list[Issue]:
    """Check for parameter conflicts and silent misconfigurations."""
    issues = []

    if algorithm in ("lora_grpo", "grpo") and backend == "verl":
        lora_r = params.get("lora_r", 16)
        load_format = params.get("load_format", "dummy")

        # Large models need safetensors for LoRA weight sync
        if lora_r > 0 and load_format == "dummy":
            model_path = params.get("model_path") or ""
            if any(s in model_path.lower() for s in ["9b", "70b", "72b"]):
                issues.append(Issue(
                    severity=Severity.WARNING,
                    category="config",
                    message=(
                        f"Large model detected ({model_path}) with load_format='dummy'. "
                        f"For 9B+ models with LoRA, the first weight sync may OOM because "
                        f"FSDP.summon_full_params reconstructs the full unsharded model."
                    ),
                    fix_hint="Use load_format='safetensors' with layered_summon=True for 9B+ LoRA models.",
                ))

        # gpu_memory_utilization sanity check
        gpu_mem_util = params.get("gpu_memory_utilization", 0.45)
        if gpu_mem_util > 0.7:
            issues.append(Issue(
                severity=Severity.WARNING,
                category="config",
                message=(
                    f"gpu_memory_utilization={gpu_mem_util} is very high. "
                    f"FSDP weight sync requires headroom on top of vLLM's allocation."
                ),
                fix_hint="Use gpu_memory_utilization <= 0.5 for co-located FSDP+vLLM.",
            ))

        # Check for missing max_prompt_length with large data
        max_prompt_length = params.get("max_prompt_length", 16384)
        if max_prompt_length > 32768:
            issues.append(Issue(
                severity=Severity.INFO,
                category="config",
                message=f"max_prompt_length={max_prompt_length} is very large, may slow down training.",
            ))

    if algorithm == "lora_grpo" and backend == "art":
        n_gpus = params.get("n_gpus", 1)
        if n_gpus > 1:
            issues.append(Issue(
                severity=Severity.ERROR,
                category="config",
                message="ART backend only supports single-GPU training (n_gpus=1).",
                fix_hint="Use backend='verl' for multi-GPU GRPO training.",
            ))

        # Reward function picklability check
        reward_fn = params.get("reward_fn")
        if reward_fn is not None:
            if hasattr(reward_fn, "__qualname__") and "<locals>" in reward_fn.__qualname__:
                issues.append(Issue(
                    severity=Severity.ERROR,
                    category="config",
                    message=(
                        f"reward_fn '{reward_fn.__name__}' appears to be a closure or local function. "
                        f"ART uses mp.spawn which requires picklable functions."
                    ),
                    fix_hint="Define reward_fn at module level, not inside another function.",
                ))

    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preflight_check(
    algorithm: str,
    backend: Optional[str] = None,
    *,
    model_path: Optional[str] = None,
    n_gpus: int = 1,
    nnodes: int = 1,
    prompt_batch_size: int = 100,
    group_size: int = 8,
    gpu_memory_utilization: float = 0.45,
    check_gpu: bool = True,
    check_deps: bool = True,
    **params,
) -> list[Issue]:
    """Run pre-flight validation checks on a training configuration.

    Catches common issues before training starts:
    - Batch divisibility errors that would crash verl at runtime
    - Insufficient GPU memory or stale processes blocking GPUs
    - Missing or incompatible dependencies
    - Parameter conflicts (e.g., ART with n_gpus > 1)

    Args:
        algorithm: Algorithm name ("sft", "osft", "lora_sft", "lora_grpo", "grpo").
        backend: Backend name. If None, uses the algorithm's default.
        model_path: Model path for model-specific checks.
        n_gpus: Number of GPUs per node.
        nnodes: Number of nodes.
        prompt_batch_size: Prompts per training step (GRPO).
        group_size: Rollouts per prompt (GRPO).
        gpu_memory_utilization: vLLM GPU memory fraction.
        check_gpu: Whether to run GPU environment checks.
        check_deps: Whether to run dependency checks.
        **params: Additional algorithm parameters for validation.

    Returns:
        List of Issue objects. Empty list means all checks passed.

    Example:
        issues = preflight_check(
            algorithm="lora_grpo",
            backend="verl",
            n_gpus=8,
            nnodes=2,
            prompt_batch_size=48,
            group_size=8,
            lora_r=128,
            model_path="Qwen/Qwen3-8B",
        )
        errors = [i for i in issues if i.severity == Severity.ERROR]
        if errors:
            print("Cannot proceed:")
            for e in errors:
                print(f"  {e}")
    """
    if backend is None:
        defaults = {
            "sft": "instructlab-training",
            "osft": "mini-trainer",
            "lora_sft": "unsloth",
            "lora_grpo": "verl",
            "grpo": "verl",
            "gepa": "gepa",
        }
        backend = defaults.get(algorithm, "unknown")

    all_params = {
        "model_path": model_path,
        "n_gpus": n_gpus,
        "nnodes": nnodes,
        "prompt_batch_size": prompt_batch_size,
        "group_size": group_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        **params,
    }

    issues: list[Issue] = []

    # Batch divisibility (GRPO/verl)
    if algorithm in ("lora_grpo", "grpo"):
        issues.extend(_check_batch_divisibility(
            algorithm=algorithm,
            backend=backend,
            prompt_batch_size=prompt_batch_size,
            group_size=group_size,
            n_gpus=n_gpus,
            nnodes=nnodes,
            ppo_mini_batch_size=params.get("ppo_mini_batch_size"),
        ))

    # GPU environment
    if check_gpu:
        issues.extend(_check_gpu_environment(
            n_gpus=n_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
        ))

    # Dependencies
    if check_deps:
        issues.extend(_check_dependencies(backend=backend, algorithm=algorithm))

    # Parameter conflicts
    issues.extend(_check_parameter_conflicts(
        algorithm=algorithm,
        backend=backend,
        params=all_params,
    ))

    return issues


def optimize_config(
    algorithm: str,
    backend: Optional[str] = None,
    *,
    model_path: Optional[str] = None,
    n_gpus: int = 1,
    nnodes: int = 1,
    target_batch_size: Optional[int] = None,
    group_size: int = 8,
    lora_r: int = 16,
) -> OptimizedConfig:
    """Suggest optimal training parameters based on hardware and model.

    Auto-calculates batch sizes, memory settings, and other parameters
    that satisfy all divisibility constraints and fit within GPU memory.

    Args:
        algorithm: Algorithm name.
        backend: Backend name. If None, uses default.
        model_path: Model identifier for model-specific tuning.
        n_gpus: Number of GPUs per node.
        nnodes: Number of nodes.
        target_batch_size: Desired prompt_batch_size (will be adjusted for constraints).
        group_size: Rollouts per prompt for GRPO.
        lora_r: LoRA rank.

    Returns:
        OptimizedConfig with suggested parameters and reasoning.

    Example:
        config = optimize_config(
            algorithm="lora_grpo",
            backend="verl",
            model_path="Qwen/Qwen3-8B",
            n_gpus=8,
            nnodes=1,
        )
        print(config.params)
        # Use directly: lora_grpo(**config.params)
    """
    if backend is None:
        defaults = {
            "sft": "instructlab-training",
            "osft": "mini-trainer",
            "lora_sft": "unsloth",
            "lora_grpo": "verl",
            "grpo": "verl",
        }
        backend = defaults.get(algorithm, "verl")

    world_size = n_gpus * nnodes
    reasoning = []
    warnings = []
    params: dict[str, Any] = {}

    if algorithm in ("lora_grpo", "grpo") and backend == "verl":
        # Calculate optimal prompt_batch_size
        n_workers = world_size * 4
        if target_batch_size is None:
            target_batch_size = max(world_size * 6, 48)

        # Find valid prompt_batch_size >= target that satisfies divisibility
        pbs = target_batch_size
        while (pbs * group_size) % n_workers != 0:
            pbs += 1
        params["prompt_batch_size"] = pbs
        reasoning.append(
            f"prompt_batch_size={pbs}: satisfies (pbs * group_size) % "
            f"(n_gpus * nnodes * 4) == 0 constraint"
        )

        # Calculate ppo_mini_batch_size divisible by world_size
        ppo_mbs = min(pbs, 256)
        if ppo_mbs % world_size != 0:
            ppo_mbs = max(world_size, (ppo_mbs // world_size) * world_size)
        params["ppo_mini_batch_size"] = ppo_mbs
        reasoning.append(
            f"ppo_mini_batch_size={ppo_mbs}: divisible by world_size={world_size}"
        )

        params["group_size"] = group_size
        params["n_gpus"] = n_gpus
        params["nnodes"] = nnodes

        # Model-specific tuning
        is_large_model = False
        is_large_vocab = False
        if model_path:
            params["model_path"] = model_path
            model_lower = model_path.lower()
            is_large_model = any(s in model_lower for s in ["8b", "9b", "14b", "32b", "70b", "72b"])
            is_large_vocab = any(s in model_lower for s in ["qwen3", "qwen3.5"])

        # LoRA config
        if algorithm == "grpo":
            params["lora_r"] = 0
            reasoning.append("lora_r=0: full fine-tuning (grpo algorithm)")
        else:
            params["lora_r"] = lora_r
            params["lora_alpha"] = lora_r * 2
            reasoning.append(f"lora_alpha={lora_r * 2}: standard 2x lora_r ratio")

        # GPU memory utilization
        if is_large_model:
            params["gpu_memory_utilization"] = 0.5
            params["load_format"] = "dummy"
            params["update_weights_bucket_megabytes"] = 3072
            reasoning.append(
                "gpu_memory_utilization=0.5, load_format=dummy: "
                "large model needs headroom for FSDP weight sync"
            )
        else:
            params["gpu_memory_utilization"] = 0.3
            reasoning.append("gpu_memory_utilization=0.3: conservative for smaller models")

        # Large vocab models need micro_batch_size=2
        if is_large_vocab:
            params["micro_batch_size"] = 2
            reasoning.append(
                "micro_batch_size=2: large vocabulary (152K) causes OOM on logits "
                "with default micro_batch_size"
            )

        # Very large models with LoRA need safetensors
        if is_large_model and lora_r > 0:
            model_lower = model_path.lower() if model_path else ""
            if any(s in model_lower for s in ["9b", "70b", "72b"]):
                params["load_format"] = "safetensors"
                params["layered_summon"] = True
                reasoning.append(
                    "load_format=safetensors + layered_summon=True: "
                    "9B+ LoRA models need preloaded base weights to avoid OOM during weight sync"
                )

        # Qwen3.5 specific
        if model_path and "qwen3.5" in model_path.lower():
            params["language_model_only"] = True
            params["use_fused_kernels"] = True
            reasoning.append(
                "language_model_only=True, use_fused_kernels=True: "
                "Qwen3.5 GatedDeltaNet requires these for correct text-only inference"
            )
            warnings.append(
                "Qwen3.5 requires: vllm>=0.19.0, flash-linear-attention, "
                "tilelang, CUDA toolkit >= 12.5, VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200"
            )

        # Default learning rate
        if "learning_rate" not in params:
            params["learning_rate"] = 5e-6 if is_large_model else 1e-5
            reasoning.append(
                f"learning_rate={params['learning_rate']}: "
                f"{'conservative for large model' if is_large_model else 'standard for smaller model'}"
            )

        # Max prompt length
        params["max_prompt_length"] = 16384
        if is_large_vocab:
            params["max_prompt_length"] = 32768
            reasoning.append("max_prompt_length=32768: large vocab models benefit from longer context")

    elif algorithm == "lora_grpo" and backend == "art":
        params["n_gpus"] = 1
        params["gpu_memory_utilization"] = 0.3
        params["group_size"] = group_size
        params["prompt_batch_size"] = target_batch_size or 5
        params["lora_r"] = lora_r
        params["lora_alpha"] = lora_r // 2
        reasoning.append("ART backend: single-GPU, conservative memory settings")
        if model_path:
            params["model_path"] = model_path

    elif algorithm == "osft":
        params["n_gpus"] = n_gpus
        if model_path:
            params["model_path"] = model_path
        reasoning.append("OSFT: uses same memory as SFT, configure via memory estimator")

    elif algorithm == "sft":
        params["n_gpus"] = n_gpus
        if model_path:
            params["model_path"] = model_path
        reasoning.append("SFT: standard training, configure via memory estimator")

    return OptimizedConfig(
        params=params,
        reasoning=reasoning,
        warnings=warnings,
    )


def validate_and_fix(
    algorithm: str,
    backend: Optional[str] = None,
    **params,
) -> tuple[dict[str, Any], list[Issue]]:
    """Validate parameters and auto-fix what can be fixed.

    Runs preflight_check, then attempts to fix ERROR-level issues
    (like batch divisibility) by adjusting parameters to valid values.

    Args:
        algorithm: Algorithm name.
        backend: Backend name.
        **params: All training parameters.

    Returns:
        Tuple of (fixed_params, remaining_issues). Fixed params will have
        adjustments applied. Remaining issues are those that couldn't be
        auto-fixed (e.g., missing GPUs, missing dependencies).
    """
    if backend is None:
        defaults = {
            "sft": "instructlab-training",
            "osft": "mini-trainer",
            "lora_sft": "unsloth",
            "lora_grpo": "verl",
            "grpo": "verl",
        }
        backend = defaults.get(algorithm, "unknown")

    fixed_params = dict(params)
    remaining_issues = []

    n_gpus = fixed_params.get("n_gpus", 1)
    nnodes = fixed_params.get("nnodes", 1)
    world_size = n_gpus * nnodes

    if algorithm in ("lora_grpo", "grpo") and backend == "verl":
        prompt_batch_size = fixed_params.get("prompt_batch_size", 100)
        group_size = fixed_params.get("group_size", 8)
        n_workers = world_size * 4

        # Fix prompt_batch_size
        if (prompt_batch_size * group_size) % n_workers != 0:
            original = prompt_batch_size
            while (prompt_batch_size * group_size) % n_workers != 0:
                prompt_batch_size += 1
            fixed_params["prompt_batch_size"] = prompt_batch_size
            logger.info(
                "Auto-fixed prompt_batch_size: %d -> %d (divisibility constraint)",
                original, prompt_batch_size,
            )

        # Fix ppo_mini_batch_size
        ppo_mbs = fixed_params.get("ppo_mini_batch_size", min(prompt_batch_size, 256))
        if ppo_mbs % world_size != 0:
            original = ppo_mbs
            ppo_mbs = max(world_size, (ppo_mbs // world_size) * world_size)
            fixed_params["ppo_mini_batch_size"] = ppo_mbs
            logger.info(
                "Auto-fixed ppo_mini_batch_size: %d -> %d (world_size divisibility)",
                original, ppo_mbs,
            )

    # Run checks on the fixed config (skip fixable issues)
    check_gpu = fixed_params.pop("check_gpu", True)
    check_deps = fixed_params.pop("check_deps", True)
    issues = preflight_check(
        algorithm=algorithm,
        backend=backend,
        check_gpu=check_gpu,
        check_deps=check_deps,
        **fixed_params,
    )

    # Filter out batch_divisibility issues (we already fixed them)
    for issue in issues:
        if issue.category != "batch_divisibility":
            remaining_issues.append(issue)

    return fixed_params, remaining_issues
