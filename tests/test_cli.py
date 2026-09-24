"""Tests for the ``thub`` CLI (``training_hub.cli``).

Covers the CLI's pure logic without launching real training:
- flag/name conversion and boolean parsing
- dotted-path resolution, including that real errors are surfaced (not masked)
- value coercion from YAML config (bool/int/json/list/callable) and its errors
- YAML config loading (missing file, bad YAML, non-mapping)
- argument parsing (``--version``, help, missing-required validation)
- config + CLI-override merging and dispatch to the algorithm function
- unknown-config-key warnings and the ``--traceback`` flag

The algorithm backend is monkeypatched so these tests need no GPU/torch.
"""

from __future__ import annotations

import argparse
import importlib
from types import SimpleNamespace

import pytest

from training_hub import cli


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "flag,expected",
    [
        ("--learning-rate", "learning_rate"),
        ("--model-path", "model_path"),
        ("--bf16", "bf16"),
    ],
)
def test_flag_to_python_name(flag, expected):
    assert cli._flag_to_python_name(flag) == expected


@pytest.mark.parametrize("value", ["true", "True", "1", "yes", "on"])
def test_parse_bool_true(value):
    assert cli._parse_bool(value) is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off"])
def test_parse_bool_false(value):
    assert cli._parse_bool(value) is False


def test_parse_bool_invalid_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_bool("maybe")


# ---------------------------------------------------------------------------
# _resolve_dotted_path
# ---------------------------------------------------------------------------

def test_resolve_dotted_path_simple():
    assert cli._resolve_dotted_path("json.dumps") is __import__("json").dumps


def test_resolve_dotted_path_nested_attribute():
    # os.path.join — requires walking past the importable prefix into attributes
    assert cli._resolve_dotted_path("os.path.join") is __import__("os").path.join


def test_resolve_dotted_path_requires_dot():
    with pytest.raises(ValueError):
        cli._resolve_dotted_path("nodothere")


def test_resolve_dotted_path_missing_module():
    with pytest.raises(ImportError):
        cli._resolve_dotted_path("definitely_not_a_module_xyz.func")


def test_resolve_dotted_path_missing_attribute_is_surfaced():
    # Module imports fine but the attribute doesn't exist — the error must name
    # the attribute failure rather than the generic "no importable prefix".
    with pytest.raises(ImportError, match="attribute lookup failed"):
        cli._resolve_dotted_path("json.this_attr_does_not_exist")


def test_resolve_dotted_path_missing_dependency_is_surfaced(monkeypatch):
    # A module that exists but fails to import due to a *missing dependency*
    # must propagate the real error, not be masked as "prefix not found".
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "pkg.mod":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError, match="torch"):
        cli._resolve_dotted_path("pkg.mod.func")


# ---------------------------------------------------------------------------
# _coerce_value
# ---------------------------------------------------------------------------

def test_coerce_value_none_passthrough():
    assert cli._coerce_value(None, {"type": str}) is None


def test_coerce_value_bool_from_string():
    assert cli._coerce_value("true", {"type": bool}) is True
    assert cli._coerce_value("no", {"type": bool}) is False


def test_coerce_value_bool_passthrough():
    assert cli._coerce_value(True, {"type": bool}) is True


def test_coerce_value_int_cast():
    assert cli._coerce_value("5", {"type": int}) == 5


def test_coerce_value_json():
    assert cli._coerce_value('{"a": 1}', {"type": str, "json": True}) == {"a": 1}


def test_coerce_value_nargs_list():
    assert cli._coerce_value(["1", "2"], {"type": int, "nargs": "+"}) == [1, 2]


def test_coerce_value_callable():
    fn = cli._coerce_value("json.dumps", {"type": str, "callable": True})
    assert fn is __import__("json").dumps


def test_coerce_value_invalid_bool_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._coerce_value("nope", {"type": bool})


def test_coerce_value_invalid_json_raises():
    with pytest.raises(ValueError):
        cli._coerce_value("{bad}", {"type": str, "json": True})


# ---------------------------------------------------------------------------
# _load_yaml_config
# ---------------------------------------------------------------------------

def test_load_yaml_config_valid(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model_path: ./m\nlearning_rate: 0.001\n", encoding="utf-8")
    data = cli._load_yaml_config(str(cfg))
    assert data == {"model_path": "./m", "learning_rate": 0.001}


def test_load_yaml_config_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli._load_yaml_config(str(tmp_path / "nope.yaml"))
    assert exc.value.code == 1


def test_load_yaml_config_non_mapping_exits(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli._load_yaml_config(str(cfg))


def test_load_yaml_config_bad_yaml_exits(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("key: : : bad\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli._load_yaml_config(str(cfg))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "training-hub" in capsys.readouterr().out


def test_no_args_prints_help_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_missing_required_args_exits(capsys):
    # osft with none of its required flags
    with pytest.raises(SystemExit) as exc:
        cli.main(["osft"])
    assert exc.value.code == 1
    assert "required" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Dispatch: config + CLI merge, coercion, None-stripping, result printing
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sft(monkeypatch):
    """Replace the sft backend import with a recorder that captures kwargs."""
    calls = {}

    def recorder(**kwargs):
        calls["kwargs"] = kwargs
        return {"status": "ok"}

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "training_hub.algorithms.sft":
            return SimpleNamespace(sft=recorder)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    return calls


def test_dispatch_cli_only(fake_sft, capsys):
    cli.main([
        "sft",
        "--model-path", "./m",
        "--data-path", "./d.jsonl",
        "--ckpt-output-dir", "./out",
        "--num-epochs", "3",
        "--checkpoint-at-epoch", "true",
    ])
    kw = fake_sft["kwargs"]
    assert kw["model_path"] == "./m"
    assert kw["num_epochs"] == 3            # int coercion via argparse
    assert kw["checkpoint_at_epoch"] is True  # bool coercion
    # internal keys must never leak to the backend
    for internal in ("config", "algorithm", "version", "traceback"):
        assert internal not in kw
    # dict result is printed as JSON
    assert '"status": "ok"' in capsys.readouterr().out


def test_dispatch_config_with_cli_override(fake_sft, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "model_path: ./from_config\n"
        "data_path: ./d.jsonl\n"
        "ckpt_output_dir: ./out\n"
        "learning_rate: 0.001\n",
        encoding="utf-8",
    )
    cli.main([
        "sft", "--config", str(cfg),
        "--learning-rate", "0.5",  # overrides the config value
    ])
    kw = fake_sft["kwargs"]
    assert kw["model_path"] == "./from_config"  # from config
    assert kw["learning_rate"] == 0.5           # CLI wins over config


def test_unknown_config_key_warns(fake_sft, tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "model_path: ./m\n"
        "data_path: ./d.jsonl\n"
        "ckpt_output_dir: ./out\n"
        "lerning_rate: 0.001\n",  # typo — not a real flag
        encoding="utf-8",
    )
    cli.main(["sft", "--config", str(cfg)])
    err = capsys.readouterr().err.lower()
    assert "warning" in err and "lerning_rate" in err
    # still passed through so nothing is silently dropped
    assert fake_sft["kwargs"]["lerning_rate"] == 0.001


def test_backend_failure_without_traceback(monkeypatch, capsys):
    def boom(name, *a, **k):
        if name == "training_hub.algorithms.sft":
            return SimpleNamespace(sft=lambda **kw: (_ for _ in ()).throw(RuntimeError("kaboom")))
        return importlib.import_module(name, *a, **k)

    monkeypatch.setattr(cli.importlib, "import_module", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main(["sft", "--model-path", "m", "--data-path", "d", "--ckpt-output-dir", "o"])
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "error: kaboom" in out.err
    assert "Traceback" not in out.err  # no traceback unless requested


def test_backend_failure_with_traceback(monkeypatch, capsys):
    def boom(name, *a, **k):
        if name == "training_hub.algorithms.sft":
            return SimpleNamespace(sft=lambda **kw: (_ for _ in ()).throw(RuntimeError("kaboom")))
        return importlib.import_module(name, *a, **k)

    monkeypatch.setattr(cli.importlib, "import_module", boom)
    with pytest.raises(SystemExit):
        cli.main([
            "sft", "--traceback",
            "--model-path", "m", "--data-path", "d", "--ckpt-output-dir", "o",
        ])
    err = capsys.readouterr().err
    assert "Traceback" in err  # full traceback when --traceback is set


# ---------------------------------------------------------------------------
# _coerce_value type-safety hardening
# ---------------------------------------------------------------------------

def test_coerce_callable_non_string_rejected():
    with pytest.raises(ValueError):
        cli._coerce_value(["os", "system"], {"type": str, "callable": True})


def test_coerce_callable_resolving_to_non_callable_rejected():
    with pytest.raises(ValueError, match="non-callable"):
        cli._coerce_value("os.sep", {"type": str, "callable": True})  # os.sep is a str


def test_coerce_json_native_dict_passthrough():
    assert cli._coerce_value({"a": 1}, {"type": str, "json": True}) == {"a": 1}


def test_coerce_bool_from_int_0_1():
    assert cli._coerce_value(1, {"type": bool}) is True
    assert cli._coerce_value(0, {"type": bool}) is False


def test_coerce_bool_rejects_other_int():
    with pytest.raises(ValueError):
        cli._coerce_value(2, {"type": bool})


def test_coerce_int_rejects_bool():
    # `num_epochs: true` in YAML must not silently become 1
    with pytest.raises(ValueError, match="boolean"):
        cli._coerce_value(True, {"type": int})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "nan", "-inf"])
def test_coerce_float_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        cli._coerce_value(bad, {"type": float})


def test_coerce_str_from_yaml_scalar():
    # YAML may parse an intended string as a number (e.g. `nproc_per_node: 8`)
    assert cli._coerce_value(8, {"type": str}) == "8"


def test_coerce_nargs_scalar_is_wrapped():
    assert cli._coerce_value("q_proj", {"type": str, "nargs": "+"}) == ["q_proj"]


def test_coerce_nargs_list():
    assert cli._coerce_value(["a", "b"], {"type": str, "nargs": "+"}) == ["a", "b"]


# ---------------------------------------------------------------------------
# Spec defaults + callable/JSON through full dispatch
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_grpo(monkeypatch):
    calls = {}

    def recorder(**kwargs):
        calls["kwargs"] = kwargs
        return None

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "training_hub.algorithms.lora_grpo":
            return SimpleNamespace(grpo=recorder, lora_grpo=recorder)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    return calls


def test_dispatch_applies_spec_defaults_and_resolves_callable(fake_grpo):
    cli.main(["grpo", "--model-path", "m", "--ckpt-output-dir", "o",
              "--reward-fn", "json.dumps"])
    kw = fake_grpo["kwargs"]
    assert kw["reward_fn"] is __import__("json").dumps  # callable resolved through dispatch
    assert kw["data_config"] == "Qwen3"                 # spec default applied
    assert kw["n_train"] == 5000                        # spec default applied


def test_config_value_beats_spec_default(fake_grpo, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model_path: m\nckpt_output_dir: o\ndata_config: Custom\n", encoding="utf-8")
    cli.main(["grpo", "--config", str(cfg)])
    assert fake_grpo["kwargs"]["data_config"] == "Custom"  # config wins over spec default


def test_dispatch_json_param_via_cli(monkeypatch):
    calls = {}
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "training_hub.algorithms.gepa":
            return SimpleNamespace(gepa=lambda **kw: calls.update(kwargs=kw))
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    cli.main(["gepa", "--seed-candidate", '{"system_prompt": "x"}',
              "--task-lm", "openai/gpt-4o-mini"])
    assert calls["kwargs"]["seed_candidate"] == {"system_prompt": "x"}  # JSON parsed via CLI
