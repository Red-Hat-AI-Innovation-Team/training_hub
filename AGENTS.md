# AGENTS.md - Training Hub

Guidelines for AI agents working in this codebase.

## Project Overview

**Training Hub** is an algorithm-focused interface for common LLM training, continual learning, and reinforcement learning techniques. The goal is to expose common training algorithms in an intuitive and easy-to-use way, abstracting away backend complexity. Training Hub is designed to support as many backends as necessary—the current implementations are just the starting point.

- **Language**: Python 3.11+
- **License**: Apache-2.0
- **Primary Author**: Red Hat AI Innovation Team

For the current list of supported algorithms, backends, and dependencies, see:
- `pyproject.toml` - Dependencies and optional extras
- `src/training_hub/__init__.py` - Public API exports
- `README.md` - User-facing documentation and support matrix

## Quick Commands

```bash
# Install in editable mode (development)
pip install -e .

# Install with LoRA support
pip install -e .[lora]

# Install with GRPO support (includes ART + verl backends)
pip install -e .[grpo,lora]

# Install with CUDA support (install sequentially after other extras)
pip install -e .[cuda] --no-build-isolation

# Install with development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Serve documentation locally (requires docsify-cli)
cd docs && docsify serve
```

See `pyproject.toml` for the full list of optional dependency groups.

## Code Organization

```text
src/training_hub/
├── __init__.py              # Public API exports
├── hub_core.py              # Core utilities
├── utils.py                 # Shared utilities (torchrun params, type formatting)
├── visualization.py         # plot_loss() for training curves
├── algorithms/
│   ├── __init__.py          # Base classes: Algorithm, Backend, AlgorithmRegistry
│   ├── sft.py               # Supervised Fine-Tuning
│   ├── osft.py              # Orthogonal Subspace Fine-Tuning
│   ├── lora.py              # LoRA + SFT
│   ├── lora_grpo.py         # LoRA + GRPO and GRPO (ART backend, algorithm, convenience fns)
│   ├── lora_grpo_verl.py    # verl backend for LoRA + GRPO and GRPO
│   ├── rewards.py           # Reward functions (tool_call_reward, binary_reward)
│   ├── verl_tool_agent.py   # Custom verl agent loop for tool-call training
│   └── peft_extender.py     # PEFT parameter handling for LoRA
└── profiling/
    └── memory_estimator.py  # GPU memory estimation for training
```

To see current algorithms and backends, check `AlgorithmRegistry` usage in `src/training_hub/algorithms/*.py`.

## Architecture Pattern

The codebase follows a **Strategy + Registry** pattern:

1. **Algorithm** (abstract base class): Defines `train()`, `get_required_params()`, `get_optional_params()`
2. **Backend** (abstract base class): Defines `execute_training(params)` - actual training implementation
3. **AlgorithmRegistry**: Maps algorithm names to classes, and backends to algorithms
4. **Convenience functions**: Top-level functions like `sft()`, `osft()` wrap the registry

See `src/training_hub/algorithms/__init__.py` for the base class definitions.

### Adding a New Algorithm

1. Create algorithm class inheriting from `Algorithm` in `src/training_hub/algorithms/`
2. Create backend class inheriting from `Backend`
3. Register both with `AlgorithmRegistry`:
   ```python
   AlgorithmRegistry.register_algorithm('my_algo', MyAlgorithm)
   AlgorithmRegistry.register_backend('my_algo', 'my-backend', MyBackend)
   ```
4. Add convenience function wrapper
5. Export in `__init__.py`
6. Add documentation in `docs/algorithms/` and `docs/api/`

Follow existing implementations in `sft.py`, `osft.py`, or `lora.py` as templates.

### Adding a New Backend

Training Hub is designed to support multiple backends per algorithm:

1. Create backend class inheriting from `Backend`
2. Implement `execute_training(algorithm_params: dict) -> Any`
3. Register with existing algorithm: `AlgorithmRegistry.register_backend('sft', 'new-backend', NewBackend)`
4. Users can then select your backend via the `backend` parameter: `sft(..., backend='new-backend')`

## Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase` (e.g., `SFTAlgorithm`, `MiniTrainerOSFTBackend`)
- **Functions**: `snake_case` (e.g., `sft()`, `osft()`, `lora_sft()`, `lora_grpo()`, `grpo()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `FLOAT32_BYTES_N`)
- **Backend names**: kebab-case strings (e.g., `"instructlab-training"`, `"mini-trainer"`, `"verl"`, `"art"`)
- **Algorithm names**: lowercase (e.g., `"sft"`, `"osft"`, `"lora_sft"`, `"lora_grpo"`, `"grpo"`)

## Code Style

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Type hints throughout (Python 3.11+ style: `list[str]` not `List[str]`)
- Use `Optional[T]` for optional parameters with None default
- Docstrings use Google format with Args/Returns sections
- Training parameters use keyword-only syntax with defaults

## Data Formats

Training data is expected in JSONL format. See `docs/api/data-formats.md` for full documentation.

Common formats:
- **Messages format**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- **Pretraining**: `{"document": "Raw text content..."}`
- **Alpaca** (LoRA): `{"instruction": "...", "input": "...", "output": "..."}`

## Parameter Translation

Different backends use different parameter names. Translation happens in each backend's `execute_training()` method.

To understand parameter mappings, read the relevant backend class:
- SFT: `InstructLabTrainingSFTBackend` in `src/training_hub/algorithms/sft.py`
- OSFT: `MiniTrainerOSFTBackend` in `src/training_hub/algorithms/osft.py`
- LoRA: `UnslothLoRABackend` in `src/training_hub/algorithms/lora.py`

Each backend class has a `renames` dict or inline translation showing the mapping.

## Torchrun Integration

Multi-GPU/multi-node training uses torchrun. See `utils.get_torchrun_params()` in `src/training_hub/utils.py` for:
- Precedence handling: args dict > environment variables > defaults
- Mutual exclusivity between `master_addr` and `rdzv_endpoint`
- Validation of `nproc_per_node` values

## Memory Estimation

The `profiling/memory_estimator.py` module provides VRAM estimation. See the module docstrings for usage:

```python
from training_hub import estimate

low, mid, high = estimate(
    training_method="osft",  # Check module for supported methods
    model_path="...",
    num_gpus=8,
    ...
)
```

## Visualization

See `src/training_hub/visualization.py` for the `plot_loss()` function:

```python
from training_hub import plot_loss

plot_loss("./checkpoints")  # Single run
plot_loss(["./run1", "./run2"], labels=["A", "B"], ema=True)  # Compare runs
```

## Important Gotchas

### Algorithm-specific constraints

Each algorithm has validation logic in its `train()` method. Read the method docstrings and validation code for current constraints:
- OSFT: See `OSFTAlgorithm.train()` in `src/training_hub/algorithms/osft.py`
- LoRA: See `LoRASFTAlgorithm.train()` in `src/training_hub/algorithms/lora.py`
- SFT: See `SFTAlgorithm.train()` in `src/training_hub/algorithms/sft.py`

### Installation

Install extras sequentially — `[grpo]` and `[cuda]` have conflicting transitive
dependencies that the solver can resolve when installed in order:
```bash
pip install -e .[grpo,lora]           # GRPO backends (ART + verl)
pip install -e .[cuda] --no-build-isolation  # CUDA kernels (after grpo)
```

### Testing

- Manual testing with example scripts in `examples/scripts/`
- Jupyter notebooks in `examples/notebooks/` for interactive testing

## Documentation

```text
docs/
├── README.md            # Home page
├── _sidebar.md          # Navigation sidebar
├── algorithms/          # Algorithm overviews
├── api/                 # API reference (functions, classes, backends)
├── guides/              # How-to guides
└── examples/            # Examples overview
```

- Uses [Docsify](https://docsify.js.org/)
- Use absolute paths for internal links: `/api/functions/sft` not `../functions/sft.md`
- See `docs/DEVELOPING.md` for documentation contribution guidelines

## CI/CD

See `.github/workflows/pypi.yaml` for the current GitHub CI pipeline:
- Build and package validation
- Test PyPI publishing on main branch pushes
- Production PyPI publishing on GitHub releases
- Sigstore package signing

## Version Management

Uses `setuptools_scm` for automatic versioning from git tags. See `pyproject.toml` `[tool.setuptools_scm]` section for configuration.

## Examples

Example scripts are in `examples/scripts/`. All examples accept `--help` for argument documentation.

See `examples/README.md` for an overview of available examples.

## Operational Practices for Development and Testing

These are hard-won lessons from development on this codebase. Follow them carefully.

### Environment Management

- **Use `uv` for venv creation and package management** when available. Prefer `uv venv`, `uv pip install`, and `uv pip compile` over `python -m venv` and `pip install`. `uv` is significantly faster and has better dependency resolution. Only fall back to basic `python -m venv` / `pip` if `uv` is not installed.
- **Track which venv you are using.** This project may have multiple venvs (e.g., `.venv`, `.venv-test`) for different purposes. Before running anything, confirm you are in the correct one. If you create a new venv for a workstream, note it explicitly so future sessions know it exists and what it's for.
- **The main branch has correctly resolving dependency versions** unless something upstream has changed. Do not preemptively downgrade packages to "fix" version conflicts. If something breaks after a new install, the problem is more likely a build artifact issue (see flash-attn below) than a fundamental version incompatibility.
- **Test install order in a fresh venv** when adding or changing dependencies. Development iteration can mask dependency resolution issues — packages may already be installed from prior work, hiding the fact that a clean `pip install` would fail. Always verify the install path documented in README/CLAUDE.md works from scratch before merging dependency changes.

### Debugging Build and Runtime Failures

- **Read the actual error before acting.** When a build or import fails, read the full traceback and identify the root cause. Do not guess based on the package name alone.
- **flash-attn must be rebuilt when torch changes.** If you see flash-attn errors after a torch upgrade, the fix is `uv pip install flash-attn --no-build-isolation --force-reinstall` (rebuilding against the new torch), NOT downgrading torch. The versions are not incompatible — the compiled extension just needs to match the torch it was built with. Run `uv cache clean flash-attn` first if needed.
- **Do not downgrade packages as a first resort.** If something worked before and breaks now, investigate what changed rather than assuming version incompatibility. Common actual causes: stale build cache, wrong venv, missing rebuild of compiled extensions, or something else already running on the GPU.

### GPU and Process Management

- **Check what is already running on GPUs before launching work AND after any failure.** Run `nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader` first. If GPUs are occupied, identify whether the processes are active work or leftover dead processes before deciding what to do. This applies to *every* GPU error — not just explicit OOM. NCCL errors, vLLM engine core crashes, `pure virtual method called` aborts, and FSDP init failures can all be caused by insufficient memory from stale processes, even when the error message doesn't mention OOM.
- **`ray stop` is unreliable for cleanup.** It often fails to kill worker subprocesses (vLLM workers, FSDP workers). After calling `ray stop`, always verify with `nvidia-smi` that GPU memory is actually freed. If processes remain, kill them directly: `nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9`.
- **Never run broad kill commands** like `pkill -f vllm` or `kill -9` on process patterns without first checking what matches. Other users or workstreams may have active processes. Use targeted kills on specific PIDs after confirming they are yours and no longer needed.
- **When you see any GPU error, check GPU state first.** An OOM, NCCL error, or engine crash on a training run may not be the run's fault — another process (or a zombie from a prior run) may already be consuming GPU memory. Check `nvidia-smi` before changing configs, downgrading packages, or reducing batch sizes.

### Running and Monitoring Training

- **Run long-running training in the background** so the user can provide input and you can do other work. Use `run_in_background=true` on Bash commands for training launches.
- **Always do a short initial check (30-60 seconds) to verify a run has started successfully** before switching to a longer monitoring cadence. Check for: process is alive, GPU utilization is nonzero, no immediate crash in logs. Do not sleep for 5-10 minutes only to discover the run failed on startup.
- **After confirming startup, check periodically but not excessively.** Read the last few lines of the log file rather than sleeping blindly. If the run produces metrics to a file (e.g., `training_metrics.jsonl`), read that for progress.

### Dependency Security

- **Check for known supply chain compromises when relaxing dependency version ranges.** litellm 1.82.7-1.82.8 were compromised by the TeamPCP attack (March 24, 2026) — they harvest credentials at import time. The upper cap `<1.82.7` exists specifically for this. When widening version ranges, search for CVEs and supply chain advisories in the expanded range before merging.
- **When a dependency is transitive (not imported directly), match the floor to the library that imports it.** Training Hub doesn't import litellm — `openpipe-art` does. The floor should match ART's declared minimum (`>=1.71.1`), not an arbitrary recent version.
- **Test dependency floors against RHOAI AIPCC indexes.** RHOAI ships pinned package versions (e.g., litellm 1.74.15/1.75.8, numba 0.61.2). If training_hub's floor is higher than what RHOAI ships, pip resolution fails at install time. When setting minimum versions, check what the target deployment platform actually provides.

### Kubeflow Integration

- **New algorithms must emit `max_steps` in `training_metrics.jsonl`.** Without it, the Kubeflow SDK reports progress as "unknown" (None) because it calculates `progressPercentage` from `max_steps`. For GRPO, set `max_steps` to `num_iterations` in both rollout and train phase entries.

### Avoiding Regressions

- **Track speed and quality baselines.** Before making changes, note the current performance (e.g., reward values, training time per step, memory usage). After changes, verify these haven't degraded. If a "fix" makes things slower or reduces reward, the fix introduced a regression.
- **Build quick test harnesses for bug fixes and new features.** Write a small script that exercises the specific broken or new path, run it, and confirm the fix. Also run (or at minimum read) any existing tests to make sure prior functionality isn't broken.
- **Save test results and note what was tested.** When a test passes, record: what was run, what config, what the output was, what commit it was on. This creates a reference point for future work. Use memory files or commit messages for this, not just conversation context.

### Validating Before Shipping

These are process lessons — not what broke, but what could have been done better to catch problems earlier.

- **Reproduce the bug before writing the fix.** When a bug is reported, the first step is to build a minimal reproduction on the current code — not to start coding the fix. If you can't reproduce the bug, you can't verify the fix. If the "fix" passes tests that also pass without the fix, the tests prove nothing. Use fresh output directories (not leftover state from prior runs), match the reporter's environment as closely as possible (same model, same ART/vLLM versions), and confirm you see the same error. Only then write the fix, and verify the reproduction now passes.
- **Do not create PRs before functionally testing the change.** Code review and syntax checks are not sufficient. If a fix touches a runtime code path (especially one involving subprocesses, GPU operations, or third-party libraries), run it end-to-end before opening a PR. A fix that passes `python -c "import ast; ast.parse(...)"` but crashes at runtime wastes reviewer time and erodes trust. Set up a test environment first, then write the fix, then test, then PR.
- **Read how the downstream library actually consumes your inputs.** When passing config to a third-party library (ART, vLLM, PEFT), read the library's source to understand which config dict it reads from and how. Do not assume two config sections (`init_args` vs `engine_args`) are merged. Do not assume a parameter set in one place is forwarded to another. `grep` for the parameter name in the library's installed source (`site-packages/art/`, etc.) to trace how it flows.
- **When a subprocess fails opaquely, find the real error before trying fixes.** `RuntimeError: Engine core initialization failed. Failed core proc(s): {}` tells you nothing. Do not start upgrading/downgrading packages or changing configs until you've found the actual exception. For vLLM, grep full output for `EngineCore_DP0.*ERROR`. For ART subprocess errors, read the `.training_error` file. Time spent on wrong fixes compounds — each attempt changes the environment and may introduce new problems.
- **Test with the actual return format, not assumed keys.** Before writing test assertions about a function's return value, call the function once and inspect what it actually returns. Writing assertions like `result["algorithm"]` based on reading an intermediate internal dict (not the return value) wastes a test cycle.
- **Test serialization of third-party objects before saving them.** When saving configs or state dicts from libraries like PEFT, verify the object is JSON/safetensors-serializable before building a pipeline around it. `peft_config.to_dict()` returning a `set` for `target_modules` is not obvious from the API — only a quick `json.dumps(config.to_dict())` test reveals it.

### Working With This Codebase Specifically

- **lora_grpo has two backends (verl and art) and three data modes (tool_call, generic, custom).** Changes to shared code paths (the `LoRAGRPOAlgorithm.train()` method, the convenience function, parameter forwarding) must be verified against both backends. A parameter added to the API must be forwarded in `optional_params` AND `get_optional_params()`.
- **verl backend launches training via torchrun subprocess.** Parameters are passed as Hydra-style CLI overrides. If you add a new parameter, it must be appended to the `cmd` list in `VeRLLoRAGRPOBackend.execute_training()`.
- **Batch divisibility matters for verl.** `prompt_batch_size * group_size` must be divisible by `n_gpus * nnodes * micro_batch_size`. `ppo_mini_batch_size` must be divisible by `n_gpus * nnodes`. Validation exists but verify it covers new configurations.
- **ART backend runs in-process.** It uses `asyncio.run()` and `LocalBackend(in_process=True)`. Errors propagate directly. The `os._exit()` issue in shutdown was fixed — do not re-introduce it, as it kills multi-iteration training silently.
- **ART post-training cleanup errors can mask successful runs.** ART's background health-check task throws `ConnectionRefusedError` after vLLM shuts down at end of training. If not handled, this writes to `error_path` and causes Kubeflow to restart the pod even though training succeeded. Check if `results_path` exists before treating a post-training exception as a failure — if training completed, log the error as a warning instead.
- **ART backend runs in a spawned subprocess.** `execute_training()` uses `mp.get_context("spawn")`. Functions passed as parameters (e.g., `reward_fn`, `rollout_fn`, `iteration_callback`) must be picklable — they must be defined at module level, not as closures or lambdas. Local functions defined inside test methods or `if __name__ == "__main__"` blocks will fail with `AttributeError: Can't get attribute ...`. The same applies to `python -c` scripts: define functions in importable files instead.
- **ART has two sets of config: `init_args` (Unsloth) and `engine_args` (vLLM).** Parameters that affect vLLM (like `gpu_memory_utilization`) must be set in `engine_args`, not just `init_args`. ART reads `engine_args` directly when constructing `AsyncEngineArgs` for vLLM. If a parameter is missing from `engine_args`, vLLM uses its default (e.g., `gpu_memory_utilization=0.9`), which will OOM when Unsloth already holds model weights on the same GPU.
- **vLLM V1 is the only engine in vLLM 0.15+.** The `VLLM_USE_V1=0` env var was removed in vLLM ~0.12. Do not attempt to force the V0 engine — it does not exist. vLLM V1 eagerly validates LoRA adapter paths on `add_lora`; if a checkpoint doesn't exist yet, create a seed adapter (see `_create_seed_lora_checkpoint`).
- **vLLM engine core errors are hidden in subprocess logs.** When `AsyncLLM.from_engine_args` fails with `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`, the actual error is in the EngineCore subprocess stderr, NOT the parent process. To find it, grep for `EngineCore_DP0.*ERROR` in the full output. Common causes: flashinfer version mismatch, GPU memory exhaustion, incompatible torch version for compiled extensions.

### Past Incidents (reference for future debugging)

These are real issues encountered during development. They illustrate *why* the rules above exist.

- **flash-attn "wrong torch" failure:** After torch was upgraded, flash-attn import failed. The attempted fix was downgrading torch, which would have broken other things. Actual fix: `uv cache clean flash-attn && uv pip install flash-attn --no-build-isolation --force-reinstall`. The extension just needed rebuilding.
- **ART multi-iteration silent failure:** `os._exit(0)` in `_shutdown_art_backend` was killing the process after iteration 1. A `NameError` on `art_project` (removed from method signature thinking closure would capture it — closures don't work across method boundaries) was the root cause, but `os._exit` masked the error. Multi-iteration training had worked before; the question was "what changed?" not "is this possible?"
- **Eval regression from history truncation:** A change to truncate observation history (`h_obs[:80]`) during an eval refactor caused scores to drop from ~43% to ~16%. The truncation was introduced incidentally during a different change and wasn't caught until explicit comparison with prior results.
- **Broad `ray stop` killed other users:** Running `ray stop` to clean up after training killed another user's active Ray cluster on the shared machine. Should have checked `ps aux` first and killed specific PIDs.
- **OOM blamed on new workload:** Training OOM'd and the response was to reduce batch size. Actual cause: stale GPU processes from a prior run were still consuming memory. `nvidia-smi` would have shown this immediately.
- **NCCL error misdiagnosed as driver incompatibility:** FSDP init failed with `ncclUnhandledCudaError: Cuda failure 'out of memory'`. This was misread as a CUDA driver/toolkit version mismatch, leading to attempted torch downgrade from cu128 to cu126. Actual cause: zombie ray/vLLM workers from prior failed runs were holding 79GB across all GPUs. `ray stop` had been called between runs but didn't kill the worker subprocesses. Fix: `nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9`. Lesson: any GPU error — not just OOM — should start with checking what's on the GPUs. NCCL errors are often memory errors in disguise.
- **`nnodes` silently dropped:** When hiding ART-internal params from the public API, `nnodes` was accidentally removed from the `optional_params` dict. Multi-node training silently fell back to 1 node. Caught by CodeRabbit review. Lesson: when adding parameters to the convenience function, they must also appear in `optional_params` in `train()` AND in `get_optional_params()`.
- **Install order dependency:** `[grpo]` and `[cuda]` extras must be installed sequentially. In development this worked because packages accumulated over time; in a fresh venv the solver couldn't resolve both simultaneously. Only caught when testing in a clean environment.
- **`VLLM_USE_V1=0` became a no-op:** The env var was set in `_subprocess_entry` to force vLLM's legacy V0 engine. vLLM ~0.12 removed the env var, and ART 0.5.17 pins vLLM 0.17.0 which only has V1. vLLM V1 eagerly validates LoRA adapter paths on `add_lora`, but ART calls `add_lora` before training creates the first checkpoint — causing `FileNotFoundError` on `adapter_config.json`. Lesson: env var workarounds for third-party behavior silently break when the dependency upgrades. Prefer structural fixes (creating the expected files) over environment hacks.
- **Seed checkpoint placed before `model.register()` was wiped by ART:** The initial fix created a seed LoRA checkpoint at the expected path before calling `model.register(backend)`. This worked locally because ART's `save_model` happened to produce valid adapter files for Qwen2.5-7B — so the seed was irrelevant and the test passed for the wrong reason. On RHOAI with Qwen3-4B, ART's registration recreated the project directory structure (deleting the seed), and then `save_model` did not produce a valid `adapter_config.json`. The fix had to be moved to a monkey-patch on `convert_checkpoint_if_needed`, which runs AFTER ART's directory setup + `save_model` but BEFORE vLLM's `add_lora`. Lesson: when a test passes, verify it's passing *because of your fix*, not because the bug doesn't reproduce in your environment. Reproduce the bug first, then fix it.
- **`gpu_memory_utilization` not forwarded to vLLM:** The parameter was passed in `init_args` (Unsloth initialization) but not in `engine_args` (vLLM engine args). ART reads `engine_args` when constructing `AsyncEngineArgs`, so vLLM defaulted to 0.9 and OOM'd because Unsloth already held ~14GB of model weights on the same GPU. Worse, `engine_args` was conditionally omitted entirely (`**({"engine_args": ...} if engine_kwargs else {})`) — when `max_lora_rank` wasn't set, no engine args were forwarded at all. Fix: always pass `engine_args` with at least `gpu_memory_utilization`. Lesson: read how the downstream library (ART) actually consumes config — don't assume `init_args` and `engine_args` are merged.
- **PEFT `target_modules` returned as a set:** `peft_config.to_dict()` returns `target_modules` as a Python `set`, which `json.dump` cannot serialize. This crashed the seed checkpoint creation with `TypeError: Object of type set is not JSON serializable`. Fix: convert to sorted list before serializing. Lesson: always test serialization of configs from third-party libraries — they may use non-JSON-safe types internally.
- **flashinfer-cubin version mismatch:** vLLM 0.17.0 installed `flashinfer-python==0.6.4` but `flashinfer-cubin==0.6.8.post1` was already present from a prior install. The version mismatch caused the vLLM engine core subprocess to crash silently with `RuntimeError: flashinfer-cubin version (0.6.8.post1) does not match flashinfer version (0.6.4)`. The parent process only showed `Engine core initialization failed. Failed core proc(s): {}` with no root cause. Fix: `uv pip install flashinfer-cubin==0.6.4`. Lesson: when vLLM engine core fails silently, grep for `EngineCore_DP0.*ERROR` to find the real error in subprocess logs.
- **vLLM engine core error invisible in spawned subprocess:** When ART runs inside Training Hub's `mp.spawn` subprocess, vLLM launches its engine core as a sub-subprocess. Errors in that sub-subprocess are printed to stderr but not captured by the parent. The parent only sees `RuntimeError: Engine core initialization failed` with an empty proc set. Spent significant time trying to install compatible versions and upgrade torch before discovering the actual error (flashinfer mismatch, then GPU memory). Lesson: always capture and grep the full stderr, including lines prefixed with `(EngineCore_DP0 pid=...)`, before trying to fix the problem.
- **ART cleanup `ConnectionRefusedError` restarted Kubeflow pods after successful training:** ART's background health-check task tried to reach vLLM after it had already shut down, throwing `ConnectionRefusedError`. The exception handler wrote to `error_path` unconditionally, which Kubeflow interpreted as a training failure and restarted the pod. Fix: check if `results_path` exists (indicating training completed) before writing to `error_path`; if training succeeded, log the cleanup error as a warning. Lesson: post-training cleanup errors should not overwrite successful results — always check completion state before marking a run as failed.
- **litellm supply chain attack (TeamPCP, March 2026):** Attackers compromised litellm's CI/CD pipeline and published malicious versions 1.82.7 and 1.82.8 that harvested credentials at import time. Upper cap `<1.82.7` in `pyproject.toml` prevents accidental upgrade. Lesson: when relaxing dependency version ranges, always check for known supply chain compromises and CVEs in the expanded range.

### Testing the ART Backend

There are no automated tests checked into the repo yet. When testing ART GRPO locally:

- **Use generic data mode** (not tool-call) for quick validation. Create a JSONL file with `{"question": "...", "ground_truth": "..."}` entries and pass a `reward_fn` that checks if `ground_truth` appears in the model's response. This avoids downloading the Toucan dataset.
- **Make questions hard enough to produce reward variance.** If the model gets 100% reward on trivial questions (e.g., "What is 2+2?"), ART skips the GRPO training step ("Skipping tuning as there is no suitable data") because all trajectories have the same reward and there's no advantage signal. Use questions the model sometimes gets wrong to exercise the actual training path.
- **Use minimal configs for speed:** `num_iterations=1`, `group_size=2`, `prompt_batch_size=5`, `lora_r=8`, `max_tokens=64`, `gpu_memory_utilization=0.3`. A single iteration takes ~90s on H100.
- **`reward_fn` must be picklable.** Define it at module level in an importable file. Do not define it as a closure, lambda, or inside `python -c`. The ART subprocess uses `mp.get_context("spawn")` which pickles all parameters.
- **Set `CUDA_VISIBLE_DEVICES=0`** to constrain to a single GPU (ART is single-GPU only). Without this, vLLM may try to use all visible GPUs.
- **Install vllm separately after `[grpo]`.** `openpipe-art` declares vllm as a dependency but the prebuilt wheel may need manual installation to match CUDA/torch versions. After install, verify: `python -c "from vllm import _C; print('OK')"`. If `_C` import fails, the wheel was built for a different torch version.
- **Verify flashinfer versions match:** `uv pip list | grep flashinfer` — `flashinfer-python` and `flashinfer-cubin` must be the same version. Mismatches crash the vLLM engine core subprocess silently.

### Disk and Cache Management

- **Set `TMPDIR` to a large filesystem** (e.g., NVMe) before launching training. Flash-attn and flashinfer JIT compilation write hundreds of MB of temp files to `/tmp`. If `/tmp` is on a small root partition, the build fills the disk silently and the compilation fails with no useful error. Also set `RAY_TMPDIR` for Ray socket paths (AF_UNIX paths have a 107-byte limit — deep NVMe paths can exceed this, use a short path like `/mnt/nvme0n1/tmp`).
- **Clear flashinfer JIT cache when changing CUDA toolkit versions.** The cache at `~/.cache/flashinfer/` contains compiled kernels for a specific CUDA version. After upgrading nvcc, delete the cache: `rm -rf ~/.cache/flashinfer/`.
- **Clear uv and pip caches periodically.** They can grow to 60-100GB on root. `uv cache clean` and `rm -rf ~/.cache/pip/`.
- **vLLM compile cache** at `~/.cache/vllm/torch_compile_cache/` can also grow large. Safe to delete between runs.

### Training Larger Models with verl (8B+)

These lessons apply to models larger than Qwen3-4B on 80GB H100s with co-located FSDP+vLLM.

- **Use `load_format=dummy` instead of `safetensors`** for the verl rollout config. With `safetensors`, vLLM loads the full model from disk during init on top of FSDP's copy, doubling GPU memory. With `dummy`, vLLM allocates empty tensors for profiling and gets real weights via the checkpoint engine's weight sync. This is verl's own default — we were overriding it unnecessarily.
- **Increase `update_weights_bucket_megabytes` to 3072.** The default 2048MB bucket can't fit the embedding weight for models with large vocabularies (e.g., Qwen3's 152K vocab × 4096 hidden × 4 bytes = 2.36GB in float32). Without this, the FSDP→vLLM weight sync fails.
- **`load_format=safetensors` is needed for LoRA on very large models (9B+).** With `dummy` format, the first weight sync calls `FSDP.summon_full_params` which reconstructs the full unsharded model on each GPU — OOM for 9B+. With `safetensors`, vLLM preloads base weights from disk, so subsequent syncs only transfer LoRA params (`base_sync_done=True`). Use with `layered_summon=True` for efficient LoRA-only syncs.
- **Full fine-tuning works by setting `lora_r=0`.** verl treats `lora_rank=0` as "no LoRA, train all parameters." No other changes needed — the same FSDP+vLLM pipeline handles full fine-tuning. Full FT is faster per step (no LoRA adapter management overhead) but uses more memory and may overfit earlier.
- **verl's FSDP→vLLM weight sync is the memory bottleneck.** During `update_weights`, FSDP gathers the full model in float32 onto each GPU to sync to vLLM. For 9B models in float32 this is ~36GB per GPU on top of FSDP's shards. This is why Qwen3-8B (8.2B params, 152K vocab) barely fits but Qwen3.5-9B (9.5B params, 248K vocab) doesn't without `safetensors` + `layered_summon`.

### OSFT (mini_trainer backend) Specific

- **AdamW leaks into the frozen subspace without post-step re-projection.** AdamW's element-wise moment rescaling (`m̂_t / (√v̂_t + ε)`) destroys the orthogonal structure of projected gradients, causing parameter updates to drift into the frozen subspace over time. The fix (mini_trainer PR #91) adds `project_parameters()` after each optimizer step. If you see gradual quality degradation during OSFT training with AdamW, check whether parameter re-projection is active.
- **OSFT V-projection uses factored form (not Gram matrix).** The original `dV -= dV @ (V_high^T @ V_high)` was replaced with `dV -= (dV @ V_high^T) @ V_high` (mini_trainer PR #74). Under FSDP2, this replaces an (M, M) all-reduce with a (k_high, M) all-gather — up to 25% speedup on 8B models. Optional caching of the all-gathered V_high is available via `OSFT_CACHE_V=1` (off by default).
- **OSFT supports on-demand checkpointing and auto-resume.** With `on_demand_checkpointing=True`, the training loop responds to termination signals (SIGTERM, SIGUSR1/2, etc.) by saving full state via `torch.distributed.checkpoint`. On restart, if no explicit `resume_from_full_state_checkpoint` is provided, it auto-detects the latest valid checkpoint from `{output_dir}/full_state_checkpoints/step_*/training_state.pt`.
- **The checkpoint trigger file is now product-agnostic.** Changed from `mini_trainer_checkpoint_trigger` to `checkpoint_requested`. Customizable via `CHECKPOINT_TRIGGER_FILENAME` env var.
- **`osft_output_dtype` was removed — use `train_dtype` instead.** The separate output dtype parameter was always falling back to `train_dtype` when unset (which was the common case). It was removed to simplify the API. Users control precision via `osft_upcast_dtype` (computation), `train_dtype` (training/output), and `save_dtype` (checkpoint).

### Notebook Development

These lessons apply to the Jupyter notebooks in `examples/notebooks/`.

- **Default configurations must work on available hardware.** The OSFT templated notebook shipped with `Llama-3.1-8B-Instruct` and `num_gpus=1`, which requires >80GB VRAM and fails on all single-GPU setups including 80GB A100/H100. Either use a smaller default model (e.g., `Llama-3.2-1B-Instruct`) or set `num_gpus=2` so the defaults work out of the box.
- **Free GPU memory before launching evaluation subprocesses.** Keeping the trained model resident while spawning `lm_eval` as a subprocess causes OOM because the subprocess loads a second copy of the model. Add `del trained_model; del trained_tokenizer; gc.collect(); torch.cuda.empty_cache()` before evaluation cells.
- **Use `sys.executable` instead of `"python"` in subprocess calls.** `subprocess.run(["python", "-m", "lm_eval", ...])` resolves against `PATH`, not the Jupyter kernel's interpreter. In a venv, this can invoke a different Python that lacks `lm_eval` or the trained checkpoint's dependencies. Use `sys.executable` to match the kernel's environment.
- **Handle `subprocess.TimeoutExpired` in evaluation cells.** `subprocess.run(..., timeout=3600)` raises `TimeoutExpired` rather than returning a non-zero code. Without a try/except, the cell aborts with no diagnostic, masking partial progress from the evaluation run.

### API Design Conventions

- **Always restore environment variables after mutation.** When a backend sets `os.environ` (e.g., `OPENAI_API_BASE` for GEPA), save the previous value and restore it in a `finally` block. Without cleanup, one training run can silently change the endpoint used by subsequent runs in the same process.
- **Enforce keyword-only parameters on training entrypoints.** Use `*` after `self` in `.train()` methods and as the first parameter in convenience functions (e.g., `def gepa(*, seed_candidate=..., task_lm=...)`). This prevents positional argument misuse and matches the convention established by `sft()`, `osft()`, and `lora_sft()`. This was flagged repeatedly in GEPA review (training_hub PR #77).

### Qwen3.5 (GatedDeltaNet) Specific

Qwen3.5 uses a hybrid architecture: 75% GatedDeltaNet linear attention layers + 25% full attention. This requires specific setup:

- **CUDA toolkit >= 12.5 required.** flashinfer's GDN kernels use PTX intrinsics (`tensormap_replace_global_dim`) added in CUDA 12.5. CUDA 12.4 fails at JIT compile time. Install the toolkit only — no driver upgrade needed: `sudo dnf install -y cuda-toolkit-12-8`.
- **Install `flash-linear-attention` and `tilelang`.** Without FLA's fused Triton kernels, GDN layers decompose into 15-20 separate GPU ops per step — 10-50x slower. `tilelang` is required on Hopper GPUs with Triton >= 3.4.0 to avoid incorrect gradients. Set `use_fused_kernels=True` in the verl model config.
- **vLLM >= 0.19.0 required.** Older versions don't support `Qwen3_5ForConditionalGeneration`.
- **Pass `language_model_only=True`.** Qwen3.5-9B is text-only but registers as a multimodal architecture. Without this flag, vLLM loads the full VL model with vision encoder. Install `qwen-vl-utils` as well (imported during model init even in text-only mode).
- **Do NOT use `tensor_parallel_size > 1`.** Qwen3.5 has known TP issues with vLLM 0.19 (NCCL hangs during inference).
- **Set `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200`.** Default is 300s. GDN inference on long prompts (20K+ tokens) with TP=1 can exceed 5 minutes, causing `TimeoutError: RPC call to sample_tokens timed out`.
- **Qwen3.5 chat template requires dict arguments in tool_calls.** Unlike Qwen3, the Qwen3.5 Jinja template calls `.items()` on tool call arguments. If `arguments` is a JSON string instead of a dict, tokenization fails with `"Can only get item pairs from a mapping"`. The data prep code must `json.loads()` string arguments before passing them to the tokenizer.
- **transformers >= 4.58 required** for the `qwen3_5` model type. Also upgrade `huggingface_hub >= 1.0`.
- **Force-reinstall `nvidia-nccl-cu12` after installing vLLM 0.19.** vLLM may leave a stale cu13 NCCL `.so` file even though pip shows cu12. Verify with `python -c "import ctypes; lib=ctypes.CDLL('path/to/libnccl.so.2'); v=ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v)); print(v.value)"`.

### Past Incidents (continued)

- **Root filesystem filled by flash-attn JIT compilation.** CUDA compilation of flash-attn wrote hundreds of 125-140MB temp files to `/tmp` on a 100GB root partition. The build failed silently with "Error compiling objects for extension" and no disk space error. Fix: set `TMPDIR=/mnt/nvme0n1/tmp` before building. Lesson: always set TMPDIR to a large filesystem for CUDA compilation.
- **Stale `libnccl.so` version after package reinstall.** `pip list` showed `nvidia-nccl-cu12==2.27.5` but the actual `.so` file was version 2.28.9 (from a prior cu13 install). Caused NCCL errors at runtime. Fix: `pip install nvidia-nccl-cu12==2.27.5 --force-reinstall`. Lesson: pip metadata and actual library files can diverge — verify with ctypes or ldd when debugging NCCL issues.
- **vLLM `sample_tokens` timeout on GDN models.** Qwen3.5-9B with 29K+ token prompts exceeded the 300s default `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`. The vLLM worker appeared to hang and the engine reported `EngineDeadError`. Fix: increase timeout to 1200s. Lesson: GDN linear attention is slower per token than standard attention — timeout values calibrated for transformer models may be too low.
- **FLA fused kernels crash without `tilelang` on Hopper.** `RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results for gated chunk_bwd_dqkwg (see #640). Please install tilelang`. The error message is clear and actionable — just `pip install tilelang`.
- **21 min/step without FLA vs 3.3 min/step with FLA.** The `[transformers] The fast path is not available because one of the required library is not installed` warning is easy to miss. Without FLA, GDN layers use a decomposed fallback that is 10-50x slower. Always install `flash-linear-attention` when training GDN/Qwen3.5 models.

### verl Backend Reference Configuration

These are known-good configuration values from validated training runs. Use as a starting point when debugging or configuring new runs.

```
# Validated on 1x H100, Qwen2.5-7B-Instruct, generic Q&A data (ART backend)
lora_r=8, lora_alpha=4
gpu_memory_utilization=0.3
num_iterations=1, group_size=2, prompt_batch_size=5
max_tokens=64, concurrency=4
Result: successful completion, 10 rollouts, reward=1.0 (trivial data)
Training time: ~90s total (including model load + vLLM warmup)
Environment: openpipe-art==0.5.17, vllm==0.17.0, torch==2.10.0

# Validated on 4x H100, Qwen3-4B, OpenShift tool-call data (1886 traces)
lora_r=128, lora_alpha=256
gpu_memory_utilization=0.3
micro_batch_size=2  (needed for large vocab models — seq_len * 152K vocab causes OOM on logits)
max_prompt_length=32768
prompt_batch_size=50, group_size=8
learning_rate=5e-6
Step 1 reward: 0.154 (baseline reference for OpenShift v5 data)
Training time: ~7 min/step

# Validated on 4x H100, Qwen3-4B, AIME math data (custom reward)
lora_r=16, lora_alpha=8
gpu_memory_utilization=0.3
num_iterations=2, group_size=8, prompt_batch_size=50
learning_rate=1e-5
Result: successful completion, both iterations

# Validated on 8x H100, Qwen3-8B, OpenShift v4 data (LoRA)
lora_r=128, lora_alpha=256
gpu_memory_utilization=0.5, load_format=dummy
update_weights_bucket_megabytes=3072
prompt_batch_size=48, group_size=8
learning_rate=5e-6, max_prompt_length=32768
Final reward: 0.80 avg, 0.89 peak (8 epochs, 41 hours)

# Validated on 8x H100, Qwen3-8B, OpenShift v4 data (full fine-tuning)
lora_r=0 (full fine-tuning)
gpu_memory_utilization=0.5, load_format=dummy
Peak reward: 0.804 at epoch 5, then overfitting (8 epochs, 26.5 hours)

# Validated on 8x H100, Qwen3.5-9B, OpenShift v4 data (LoRA)
lora_r=128, lora_alpha=256
gpu_memory_utilization=0.3, load_format=safetensors, layered_summon=True
language_model_only=True, use_fused_kernels=True
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
Requires: vllm==0.19.0, flash-linear-attention, tilelang, CUDA toolkit 12.5+
Step time: ~3.3 min/step, reward 0.85+ by epoch 2
```
