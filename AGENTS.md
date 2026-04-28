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
- **Functions**: `snake_case` (e.g., `sft()`, `osft()`, `lora_sft()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `FLOAT32_BYTES_N`)
- **Backend names**: kebab-case strings (e.g., `"instructlab-training"`, `"mini-trainer"`)
- **Algorithm names**: lowercase (e.g., `"sft"`, `"osft"`, `"lora_sft"`)

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

- **Check what is already running on GPUs before launching work.** Run `nvidia-smi` first. If GPUs are occupied, identify whether the processes are active work or leftover dead processes before deciding what to do.
- **Never run broad kill commands** like `ray stop`, `pkill -f vllm`, or `kill -9` on process patterns without first checking what matches. Other users or workstreams may have active processes. Use targeted kills on specific PIDs after confirming they are yours and no longer needed.
- **When you see OOM, check GPU state first.** An OOM error on a training run may not be the run's fault — another process may already be consuming GPU memory. Check `nvidia-smi` before reducing batch sizes or model configs.

### Running and Monitoring Training

- **Run long-running training in the background** so the user can provide input and you can do other work. Use `run_in_background=true` on Bash commands for training launches.
- **Always do a short initial check (30-60 seconds) to verify a run has started successfully** before switching to a longer monitoring cadence. Check for: process is alive, GPU utilization is nonzero, no immediate crash in logs. Do not sleep for 5-10 minutes only to discover the run failed on startup.
- **After confirming startup, check periodically but not excessively.** Read the last few lines of the log file rather than sleeping blindly. If the run produces metrics to a file (e.g., `training_metrics.jsonl`), read that for progress.

### Avoiding Regressions

- **Track speed and quality baselines.** Before making changes, note the current performance (e.g., reward values, training time per step, memory usage). After changes, verify these haven't degraded. If a "fix" makes things slower or reduces reward, the fix introduced a regression.
- **Build quick test harnesses for bug fixes and new features.** Write a small script that exercises the specific broken or new path, run it, and confirm the fix. Also run (or at minimum read) any existing tests to make sure prior functionality isn't broken.
- **Save test results and note what was tested.** When a test passes, record: what was run, what config, what the output was, what commit it was on. This creates a reference point for future work. Use memory files or commit messages for this, not just conversation context.

### Working With This Codebase Specifically

- **lora_grpo has two backends (verl and art) and three data modes (tool_call, generic, custom).** Changes to shared code paths (the `LoRAGRPOAlgorithm.train()` method, the convenience function, parameter forwarding) must be verified against both backends. A parameter added to the API must be forwarded in `optional_params` AND `get_optional_params()`.
- **verl backend launches training via torchrun subprocess.** Parameters are passed as Hydra-style CLI overrides. If you add a new parameter, it must be appended to the `cmd` list in `VeRLLoRAGRPOBackend.execute_training()`.
- **Batch divisibility matters for verl.** `prompt_batch_size * group_size` must be divisible by `n_gpus * nnodes * micro_batch_size`. `ppo_mini_batch_size` must be divisible by `n_gpus * nnodes`. Validation exists but verify it covers new configurations.
- **ART backend runs in-process.** It uses `asyncio.run()` and `LocalBackend(in_process=True)`. Errors propagate directly. The `os._exit()` issue in shutdown was fixed — do not re-introduce it, as it kills multi-iteration training silently.

### Past Incidents (reference for future debugging)

These are real issues encountered during development. They illustrate *why* the rules above exist.

- **flash-attn "wrong torch" failure:** After torch was upgraded, flash-attn import failed. The attempted fix was downgrading torch, which would have broken other things. Actual fix: `uv cache clean flash-attn && uv pip install flash-attn --no-build-isolation --force-reinstall`. The extension just needed rebuilding.
- **ART multi-iteration silent failure:** `os._exit(0)` in `_shutdown_art_backend` was killing the process after iteration 1. A `NameError` on `art_project` (removed from method signature thinking closure would capture it — closures don't work across method boundaries) was the root cause, but `os._exit` masked the error. Multi-iteration training had worked before; the question was "what changed?" not "is this possible?"
- **Eval regression from history truncation:** A change to truncate observation history (`h_obs[:80]`) during an eval refactor caused scores to drop from ~43% to ~16%. The truncation was introduced incidentally during a different change and wasn't caught until explicit comparison with prior results.
- **Broad `ray stop` killed other users:** Running `ray stop` to clean up after training killed another user's active Ray cluster on the shared machine. Should have checked `ps aux` first and killed specific PIDs.
- **OOM blamed on new workload:** Training OOM'd and the response was to reduce batch size. Actual cause: stale GPU processes from a prior run were still consuming memory. `nvidia-smi` would have shown this immediately.
- **`nnodes` silently dropped:** When hiding ART-internal params from the public API, `nnodes` was accidentally removed from the `optional_params` dict. Multi-node training silently fell back to 1 node. Caught by CodeRabbit review. Lesson: when adding parameters to the convenience function, they must also appear in `optional_params` in `train()` AND in `get_optional_params()`.
- **Install order dependency:** `[grpo]` and `[cuda]` extras must be installed sequentially. In development this worked because packages accumulated over time; in a fresh venv the solver couldn't resolve both simultaneously. Only caught when testing in a clean environment.

### verl Backend Reference Configuration

These are known-good configuration values from validated training runs. Use as a starting point when debugging or configuring new runs.

```
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
```
