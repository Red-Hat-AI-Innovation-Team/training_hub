"""Tests for training_hub.utils dataset loading."""

import json

import pytest

from training_hub.utils import _FORMAT_MAP, load_training_dataset


def test_format_map_covers_supported_extensions():
    assert _FORMAT_MAP == {
        ".jsonl": "json",
        ".json": "json",
        ".parquet": "parquet",
        ".csv": "csv",
    }


@pytest.mark.parametrize(
    "path,expected_builder",
    [
        ("/data/train.jsonl", "json"),
        ("/data/train.json", "json"),
        ("/data/train.parquet", "parquet"),
        ("/data/train.csv", "csv"),
        ("/data/TRAIN.Parquet", "parquet"),  # extension match is case-insensitive
    ],
)
def test_selects_builder_by_extension(monkeypatch, path, expected_builder):
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "DATASET"

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    assert load_training_dataset(path) == "DATASET"
    assert calls["args"] == (expected_builder,)
    assert calls["kwargs"] == {"data_files": path, "split": "train"}


def test_unknown_extension_falls_back_to_hf_name(monkeypatch):
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "HF"

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    assert load_training_dataset("org/some-dataset") == "HF"
    assert calls["args"] == ("org/some-dataset",)
    assert calls["kwargs"] == {"split": "train"}


def test_custom_split_is_forwarded(monkeypatch):
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["kwargs"] = kwargs
        return "DATASET"

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    load_training_dataset("/data/train.parquet", split="validation")
    assert calls["kwargs"]["split"] == "validation"


def test_reads_real_jsonl(tmp_path):
    p = tmp_path / "train.jsonl"
    rows = [{"text": "a"}, {"text": "b"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    ds = load_training_dataset(str(p))

    assert len(ds) == 2
    assert [r["text"] for r in ds] == ["a", "b"]
