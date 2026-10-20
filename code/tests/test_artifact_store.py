"""LocalArtifactStore: key layout, bytes and JSON round trips, keys that escape the root."""

import pytest

from pipeline.exec.local_backend import LocalArtifactStore


@pytest.fixture
def store(tmp_path):
    return LocalArtifactStore(tmp_path, run_id="t")


def test_key_layout(store):
    assert store.key("train", "checkpoint.pt") == "runs/t/train/checkpoint.pt"


def test_put_get_roundtrip(store):
    store.put("runs/t/x/a.bin", b"\x00\x01")
    assert store.get("runs/t/x/a.bin") == b"\x00\x01"


def test_get_missing_raises_keyerror(store):
    with pytest.raises(KeyError):
        store.get("runs/t/x/missing")
    assert not store.exists("runs/t/x/missing")


def test_list_prefix(store):
    store.put("runs/t/a/1", b"1")
    store.put("runs/t/a/2", b"2")
    store.put("runs/t/b/3", b"3")
    assert store.list("runs/t/a/") == ["runs/t/a/1", "runs/t/a/2"]
    assert store.list() == ["runs/t/a/1", "runs/t/a/2", "runs/t/b/3"]


def test_key_escape_rejected(store):
    with pytest.raises(ValueError):
        store.put("../outside", b"x")


def test_json_and_jsonl_helpers(store):
    store.put_json("runs/t/s/m.json", {"b": 2, "a": 1})
    assert store.get_json("runs/t/s/m.json") == {"a": 1, "b": 2}
    store.append_jsonl("runs/t/s/log.jsonl", {"i": 0})
    store.append_jsonl("runs/t/s/log.jsonl", {"i": 1})
    assert store.read_jsonl("runs/t/s/log.jsonl") == [{"i": 0}, {"i": 1}]
