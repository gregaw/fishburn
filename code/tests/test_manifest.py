"""Stage manifests: params hashes, completion, corrupt files, the git sha."""

from pipeline import manifest
from pipeline.exec.local_backend import LocalArtifactStore


def test_params_hash_stable_and_order_insensitive():
    a = manifest.params_hash({"x": 1, "y": "z"})
    b = manifest.params_hash({"y": "z", "x": 1})
    assert a == b and len(a) == 16
    assert manifest.params_hash({"x": 2, "y": "z"}) != a


def test_complete_lifecycle(tmp_path):
    store = LocalArtifactStore(tmp_path, run_id="t")
    phash = manifest.params_hash({"p": 1})
    assert not manifest.is_complete(store, "s", phash)
    assert manifest.load(store, "s") is None

    manifest.write_manifest(store, "s", phash, ["out.bin"], extra={"seed": 7})
    assert manifest.is_complete(store, "s", phash)
    m = manifest.load(store, "s")
    assert m["outputs"] == ["out.bin"] and m["seed"] == 7 and m["git_sha"]

    # changed params -> not complete -> stage re-runs
    assert not manifest.is_complete(store, "s", manifest.params_hash({"p": 2}))


def test_corrupt_manifest_is_incomplete(tmp_path):
    store = LocalArtifactStore(tmp_path, run_id="t")
    store.put(store.key("s", manifest.MANIFEST), b"{not json")
    assert not manifest.is_complete(store, "s", "whatever")
    assert manifest.load(store, "s") is None


def test_git_sha_env_override(monkeypatch):
    monkeypatch.setenv("NCM_GIT_SHA", "cafe123")
    assert manifest.git_sha() == "cafe123"
