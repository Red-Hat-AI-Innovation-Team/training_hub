"""Tests for the wandb stub used to keep ART 0.5.18 working without wandb."""

import importlib.util
import sys
import types

import pytest

from training_hub.algorithms.lora_grpo import _install_wandb_stub


@pytest.fixture(autouse=True)
def _clean_wandb_state(monkeypatch):
    """Snapshot and restore sys.modules / wandb availability around each test."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    saved = sys.modules.pop("wandb", None)
    yield
    if saved is not None:
        sys.modules["wandb"] = saved
    else:
        sys.modules.pop("wandb", None)


def _force_wandb_absent(monkeypatch):
    """Make importlib believe wandb is not installed."""
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name):
        return None if name == "wandb" else real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)


def test_stub_installed_when_wandb_absent(monkeypatch):
    _force_wandb_absent(monkeypatch)
    _install_wandb_stub()
    assert "wandb" in sys.modules
    # ART only needs the import to succeed; _get_wandb_run() then returns
    # None because WANDB_API_KEY is unset, so the module is never used.
    assert isinstance(sys.modules["wandb"], types.ModuleType)


def test_no_stub_when_wandb_installed(monkeypatch):
    real = types.ModuleType("wandb")
    sys.modules["wandb"] = real
    _install_wandb_stub()
    assert sys.modules["wandb"] is real


def test_no_stub_when_api_key_set(monkeypatch):
    _force_wandb_absent(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    _install_wandb_stub()
    assert "wandb" not in sys.modules


def test_stub_is_idempotent(monkeypatch):
    _force_wandb_absent(monkeypatch)
    _install_wandb_stub()
    first = sys.modules["wandb"]
    _install_wandb_stub()
    assert sys.modules["wandb"] is first
