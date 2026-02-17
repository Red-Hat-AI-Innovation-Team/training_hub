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

# Install with CUDA support (requires two-step for flash-attn)
pip install -e . && pip install -e .[cuda] --no-build-isolation

# Install with LoRA support
pip install -e .[lora]

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

CUDA extras require two-step install due to flash-attn build requirements:
```bash
pip install -e . && pip install -e .[cuda] --no-build-isolation
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
