"""Tests for exporting PEFT adapters from verl FSDP checkpoints (_normalize_verl_checkpoints)."""

import json
import shutil
from pathlib import Path

import pytest
import torch

pytest.importorskip("verl.model_merger.fsdp_model_merger")
peft = pytest.importorskip("peft")
from transformers import Qwen3Config, Qwen3ForCausalLM

from training_hub.algorithms.lora_grpo_verl import _normalize_verl_checkpoints

TARGET_MODULES = ["q_proj", "v_proj"]
SHARD = "model_world_size_1_rank_0.pt"
TOKENIZER_CONFIG = {"tokenizer_class": "Qwen2Tokenizer"}
CHAT_TEMPLATE = "{{ messages }}"


# --- helpers ------------------------------------------------------------------------

def _tiny_base():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    )
    return Qwen3ForCausalLM(config)


def _trained_peft_model():
    model = peft.get_peft_model(
        _tiny_base(),
        peft.LoraConfig(r=4, lora_alpha=8, target_modules=TARGET_MODULES, task_type="CAUSAL_LM"),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn_like(param))
    return model


def _lora_weights(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def _write_verl_checkpoint(actor_dir: Path, model, lora_only: bool = True, lora: bool = True) -> Path:
    """Mimic verl 0.9.1 FSDPCheckpointManager.save_checkpoint output for world_size=1."""
    hf_dir = actor_dir / "huggingface"
    hf_dir.mkdir(parents=True)
    torch.save(_lora_weights(model) if lora_only else model.state_dict(), actor_dir / SHARD)
    (actor_dir / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 1, "world_size": 1}))
    if lora:
        (actor_dir / "lora_train_meta.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, "task_type": "CAUSAL_LM"}))
    (model.base_model.model.config if lora else model.config).save_pretrained(hf_dir)
    (hf_dir / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG))
    return actor_dir


def _checkpoint(tmp_path: Path, step: int = 3, model=None, **kwargs) -> Path:
    """Write global_step_<step>/actor/ under tmp_path/checkpoints and return the actor dir."""
    actor_dir = tmp_path / "checkpoints" / f"global_step_{step}" / "actor"
    return _write_verl_checkpoint(actor_dir, model if model is not None else _trained_peft_model(), **kwargs)


def _normalize(tmp_path: Path) -> list[Path]:
    return [Path(p) for p in _normalize_verl_checkpoints(str(tmp_path / "checkpoints"))]


def _add_chat_template(actor_dir: Path) -> None:
    """Tokenizers with several chat templates save them under additional_chat_templates/."""
    templates = actor_dir / "huggingface" / "additional_chat_templates"
    templates.mkdir()
    (templates / "tool_use.jinja").write_text(CHAT_TEMPLATE)


def _assert_loads_with_weights(lora_dir: Path, trained) -> None:
    """The adapter loads in PEFT and holds the trained LoRA weights (the verl merger exports bf16)."""
    loaded = _lora_weights(peft.PeftModel.from_pretrained(_tiny_base(), str(lora_dir)))
    expected = _lora_weights(trained)
    assert loaded.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(loaded[key], value.to(torch.bfloat16).to(value.dtype))


@pytest.fixture
def gloo_group(tmp_path):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


# --- export -------------------------------------------------------------------------

@pytest.mark.parametrize("lora_only", [True, False], ids=["save_lora_only", "full_shards"])
def test_exports_loadable_peft_adapter(tmp_path, lora_only):
    trained = _trained_peft_model()
    actor_dir = _checkpoint(tmp_path, model=trained, lora_only=lora_only)

    lora_dir = actor_dir / "lora_adapter"
    assert _normalize(tmp_path) == [lora_dir]
    assert {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"} <= {
        p.name for p in lora_dir.iterdir()}
    adapter_config = json.loads((lora_dir / "adapter_config.json").read_text())
    assert (adapter_config["r"], adapter_config["lora_alpha"]) == (4, 8)
    assert sorted(adapter_config["target_modules"]) == TARGET_MODULES
    _assert_loads_with_weights(lora_dir, trained)


def test_exports_dtensor_shards(tmp_path, gloo_group):
    """FSDP saves sharded params as DTensors; check the export unwraps them (world_size=1)."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    trained = _trained_peft_model()
    shard = _checkpoint(tmp_path, model=trained) / SHARD
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
    state_dict = torch.load(shard, weights_only=False)
    torch.save({k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in state_dict.items()}, shard)

    (lora_dir,) = _normalize(tmp_path)
    _assert_loads_with_weights(lora_dir, trained)


def test_skips_full_finetune_checkpoints(tmp_path):
    actor_dir = _checkpoint(tmp_path, model=_tiny_base(), lora_only=False, lora=False)

    assert _normalize(tmp_path) == []
    assert not (actor_dir / "lora_adapter").exists()


def test_redoes_a_partial_adapter_export(tmp_path):
    """A lora_adapter/ with adapter_config.json but no weights (interrupted export) is redone."""
    trained = _trained_peft_model()
    actor_dir = _checkpoint(tmp_path, model=trained)
    (actor_dir / "lora_adapter").mkdir()
    (actor_dir / "lora_adapter" / "adapter_config.json").write_text("{}")

    (lora_dir,) = _normalize(tmp_path)

    _assert_loads_with_weights(lora_dir, trained)
    assert not list(actor_dir.glob(".lora_export_*"))  # the temporary export dir is cleaned up


# --- checkpoint selection -----------------------------------------------------------

def test_checkpoints_are_in_step_order(tmp_path):
    for step in (9, 10, 2):
        _checkpoint(tmp_path, step)

    assert [p.parent.parent.name for p in _normalize(tmp_path)] == [
        "global_step_2", "global_step_9", "global_step_10"]


def test_skips_a_broken_earlier_checkpoint(tmp_path, caplog):
    (_checkpoint(tmp_path, 5) / SHARD).unlink()
    latest = _checkpoint(tmp_path, 10)

    assert _normalize(tmp_path) == [latest / "lora_adapter"]
    assert "Skipping checkpoint global_step_5" in caplog.text


def test_raises_if_the_latest_checkpoint_is_broken(tmp_path):
    _checkpoint(tmp_path, 5)
    (_checkpoint(tmp_path, 10) / SHARD).unlink()

    with pytest.raises(RuntimeError, match="latest checkpoint .*global_step_10"):
        _normalize(tmp_path)


def test_raises_if_the_latest_checkpoint_has_no_lora_meta(tmp_path):
    _checkpoint(tmp_path, 5)
    (_checkpoint(tmp_path, 10) / "lora_train_meta.json").unlink()

    with pytest.raises(RuntimeError, match="lora_train_meta.json"):
        _normalize(tmp_path)


# --- tokenizer/config files ---------------------------------------------------------

def test_copies_nested_tokenizer_folders(tmp_path):
    _add_chat_template(_checkpoint(tmp_path))

    (lora_dir,) = _normalize(tmp_path)

    assert (lora_dir / "additional_chat_templates" / "tool_use.jinja").read_text() == CHAT_TEMPLATE


def test_rerun_repairs_an_interrupted_tokenizer_copy(tmp_path):
    _add_chat_template(_checkpoint(tmp_path))
    (lora_dir,) = _normalize(tmp_path)
    # Simulate a copy interrupted on an earlier run: an empty folder and a truncated file
    shutil.rmtree(lora_dir / "additional_chat_templates")
    (lora_dir / "additional_chat_templates").mkdir()
    (lora_dir / "tokenizer_config.json").write_text("")

    _normalize(tmp_path)

    assert (lora_dir / "additional_chat_templates" / "tool_use.jinja").read_text() == CHAT_TEMPLATE
    assert json.loads((lora_dir / "tokenizer_config.json").read_text()) == TOKENIZER_CONFIG


def test_tokenizer_copy_never_overwrites_adapter_files(tmp_path):
    actor_dir = _checkpoint(tmp_path)
    (actor_dir / "huggingface" / "adapter_config.json").write_text('{"r": 999}')

    (lora_dir,) = _normalize(tmp_path)

    assert json.loads((lora_dir / "adapter_config.json").read_text())["r"] == 4
