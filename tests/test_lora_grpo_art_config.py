"""Tests for building art.TrainableModel kwargs across openpipe-art API versions."""

import types

import pytest

from training_hub.algorithms.lora_grpo import _build_art_model_kwargs


def _raise_peft_args(*_args, **_kwargs):
    raise ValueError("`peft_args` has been replaced by top-level `TrainableModel(lora_config=...)`.")


def _fake_art(new_api: bool):
    """Mimic the openpipe-art surface _build_art_model_kwargs touches (<0.5.20 vs >=0.5.20)."""
    if new_api:
        class TrainableModel:
            def __init__(self, *, name, run_name, project, base_model, lora_config=None,
                         _internal_config=None):
                pass

        dev = types.SimpleNamespace(LoRAConfig=dict, PeftArgs=_raise_peft_args,
                                    InternalModelConfig=dict)
    else:
        class TrainableModel:
            def __init__(self, *, name, project, base_model, _internal_config=None):
                pass

        dev = types.SimpleNamespace(PeftArgs=dict, InternalModelConfig=dict)
    return types.SimpleNamespace(TrainableModel=TrainableModel, dev=dev)


def _build(art, **overrides):
    kwargs = dict(
        name="qwen-grpo",
        project="training-hub-grpo",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        lora_r=16,
        lora_alpha=8,
        target_modules=None,
        internal_config_kwargs={"init_args": {"gpu_memory_utilization": 0.3}},
    )
    kwargs.update(overrides)
    return _build_art_model_kwargs(art, **kwargs)


def test_new_api_uses_top_level_lora_config_and_run_name():
    out = _build(_fake_art(new_api=True), target_modules=["q_proj", "v_proj"])
    assert out["lora_config"] == {"rank": 16, "alpha": 8, "target_modules": ["q_proj", "v_proj"]}
    assert out["run_name"] == "qwen-grpo"
    assert "peft_args" not in out["_internal_config"]
    assert out["_internal_config"]["init_args"] == {"gpu_memory_utilization": 0.3}


def test_old_api_keeps_peft_args_and_omits_new_kwargs():
    out = _build(_fake_art(new_api=False), target_modules=["q_proj"])
    assert out["_internal_config"]["peft_args"] == {"r": 16, "lora_alpha": 8,
                                                   "target_modules": ["q_proj"]}
    assert "lora_config" not in out
    assert "run_name" not in out


@pytest.mark.parametrize("new_api", [True, False])
def test_rank_is_always_explicit(new_api):
    """ART defaults to rank 8, so training_hub's default lora_r=16 must not be dropped."""
    out = _build(_fake_art(new_api=new_api), lora_r=16)
    rank = out["lora_config"]["rank"] if new_api else out["_internal_config"]["peft_args"]["r"]
    assert rank == 16


@pytest.mark.parametrize("new_api", [True, False])
def test_kwargs_are_accepted_by_the_constructor(new_api):
    art = _fake_art(new_api=new_api)
    art.TrainableModel(**_build(art))


def test_target_modules_omitted_when_none():
    out = _build(_fake_art(new_api=True), target_modules=None)
    assert "target_modules" not in out["lora_config"]


def test_internal_config_kwargs_are_not_mutated():
    internal = {"init_args": {"gpu_memory_utilization": 0.3}}
    _build(_fake_art(new_api=False), internal_config_kwargs=internal)
    assert internal == {"init_args": {"gpu_memory_utilization": 0.3}}


def test_installed_openpipe_art_accepts_the_kwargs():
    """Against whichever openpipe-art is installed (skipped when it is not)."""
    art = pytest.importorskip("art")
    if not hasattr(art, "TrainableModel"):
        pytest.skip("installed 'art' module is not openpipe-art")
    out = _build(art)
    params = __import__("inspect").signature(art.TrainableModel.__init__).parameters
    assert ("lora_config" in out) == ("lora_config" in params)
    assert ("run_name" in out) == ("run_name" in params)
