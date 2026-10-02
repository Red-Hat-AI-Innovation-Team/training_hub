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
    "filename,expected_builder",
    [
        ("train.jsonl", "json"),
        ("train.json", "json"),
        ("train.parquet", "parquet"),
        ("train.csv", "csv"),
        ("TRAIN.Parquet", "parquet"),  # extension match is case-insensitive
    ],
)
def test_selects_builder_by_extension(monkeypatch, tmp_path, filename, expected_builder):
    # Must be a real local file — a recognized extension alone is not enough
    # (a HF dataset ID can also end in .csv); see the dataset-ID test below.
    path = tmp_path / filename
    path.write_text("")
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "DATASET"

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    assert load_training_dataset(str(path)) == "DATASET"
    assert calls["args"] == (expected_builder,)
    assert calls["kwargs"] == {"data_files": str(path), "split": "train"}


def test_dataset_id_with_extension_is_not_read_as_a_file(monkeypatch):
    # A HuggingFace dataset ID can end in a recognized extension; it must load as
    # a repo (load_dataset(id)), not via a file builder.
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "HF"

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    assert load_training_dataset("org/labels.csv") == "HF"
    assert calls["args"] == ("org/labels.csv",)  # HF branch, not ("csv",)
    assert "data_files" not in calls["kwargs"]


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
