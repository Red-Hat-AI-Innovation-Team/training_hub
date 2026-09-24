"""Tests for training_hub.utils dataset loading."""

import json

import pytest

from training_hub.utils import (
    _FORMAT_MAP,
    load_training_dataset,
    normalize_messages_column,
)


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


def test_reads_real_csv(tmp_path):
    from datasets import Dataset

    p = tmp_path / "train.csv"
    Dataset.from_list([{"text": "a"}, {"text": "b"}]).to_csv(str(p))

    ds = load_training_dataset(str(p))

    assert len(ds) == 2
    assert [r["text"] for r in ds] == ["a", "b"]


def test_reads_real_parquet(tmp_path):
    from datasets import Dataset

    p = tmp_path / "train.parquet"
    Dataset.from_list([{"text": "a"}, {"text": "b"}]).to_parquet(str(p))

    ds = load_training_dataset(str(p))

    assert len(ds) == 2
    assert [r["text"] for r in ds] == ["a", "b"]


# ---------------------------------------------------------------------------
# normalize_messages_column
# ---------------------------------------------------------------------------

_CONVO = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]


def test_normalize_messages_decodes_json_strings():
    # CSV/parquet may store `messages` as a serialized JSON string.
    from datasets import Dataset

    ds = Dataset.from_list([{"messages": json.dumps(_CONVO)}])
    out = normalize_messages_column(ds)

    assert out[0]["messages"] == _CONVO


def test_normalize_messages_passes_through_lists():
    from datasets import Dataset

    ds = Dataset.from_list([{"messages": _CONVO}])
    out = normalize_messages_column(ds)

    assert out[0]["messages"] == _CONVO


def test_normalize_messages_missing_column_is_noop():
    from datasets import Dataset

    ds = Dataset.from_list([{"text": "a"}])
    assert normalize_messages_column(ds) is ds


def test_normalize_messages_rejects_non_json_string():
    from datasets import Dataset

    ds = Dataset.from_list([{"messages": "not json at all"}])
    with pytest.raises(ValueError, match="not.*valid JSON|list of message"):
        normalize_messages_column(ds)


def test_normalize_messages_rejects_wrong_shape():
    from datasets import Dataset

    # decodes to a JSON scalar, not a list of message dicts
    ds = Dataset.from_list([{"messages": json.dumps("just a string")}])
    with pytest.raises(ValueError, match="list of message"):
        normalize_messages_column(ds)


def test_normalize_messages_passes_through_none():
    # A null cell (common in CSV / nullable parquet) must not crash the map.
    from datasets import Dataset

    ds = Dataset.from_list([{"messages": None}, {"messages": json.dumps(_CONVO)}])
    out = normalize_messages_column(ds)
    assert out[0]["messages"] is None
    assert out[1]["messages"] == _CONVO


def test_normalize_messages_empty_list_ok():
    from datasets import Dataset

    ds = Dataset.from_list([{"messages": []}])
    assert normalize_messages_column(ds)[0]["messages"] == []


def test_normalize_messages_error_names_row_index():
    # A serialized-string column where the second row is invalid JSON.
    # (An Arrow column can't mix list and string values, so both rows are strings.)
    from datasets import Dataset

    ds = Dataset.from_list([{"messages": json.dumps(_CONVO)}, {"messages": "not json"}])
    with pytest.raises(ValueError, match="row 1"):
        normalize_messages_column(ds)
