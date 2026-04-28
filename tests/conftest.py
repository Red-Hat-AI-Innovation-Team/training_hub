"""Shared fixtures and markers for Training Hub tests."""

import os
import shutil
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: long-running test (>60s)")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for test outputs, cleaned up automatically."""
    return str(tmp_path / "output")


@pytest.fixture
def model_path():
    """Model path for tests that need a real model.

    Uses Qwen3.5-4B if cached locally, otherwise falls back to Qwen3-4B
    from HuggingFace (requires network).
    """
    # Prefer a locally-cached model to avoid downloads during tests.
    # Order matters: smaller/simpler models first for faster tests.
    candidates = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3.5-4B",
    ]
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    for name in candidates:
        model_dir = os.path.join(cache_dir, f"models--{name.replace('/', '--')}")
        if os.path.isdir(model_dir):
            return name
    pytest.skip("No locally cached model found; skipping to avoid download")
