"""Resume points for training stages (experiment 0114): a stage killed
mid-run continues from its last checkpoint when dispatched again."""

import os

import pytest
import torch

from pipeline import manifest
from pipeline.config import load_profile
from pipeline.exec.local_backend import LocalArtifactStore
from pipeline.legs import LEGS
from pipeline.resume import ResumePoint
from pipeline.stages import data_prep, relabel, train

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _checkpoint_dir(root, step, payload):
    d = root / f"checkpoint-{step}"
    (d / "sub").mkdir(parents=True)
    (d / "optimizer.pt").write_bytes(payload)
    (d / "sub" / "trainer_state.json").write_text("{}")
    return d


def test_save_load_alternates_slots_and_restores_files(tmp_path):
    store = LocalArtifactStore(tmp_path / "store", run_id="t")
    point = ResumePoint(store, "train_x", "h1")
    assert point.load(str(tmp_path / "w0")) == (None, {})

    point.save(str(_checkpoint_dir(tmp_path / "a", 2, b"two")), 2, {"n": 2})
    point.save(str(_checkpoint_dir(tmp_path / "b", 4, b"four")), 4, {"n": 4})
    assert store.get_json(store.key("train_x", "resume", "resume.json"))["slot"] == "slot1"

    path, state = point.load(str(tmp_path / "w1"))
    assert path.endswith("checkpoint-4") and state == {"n": 4}
    assert open(os.path.join(path, "optimizer.pt"), "rb").read() == b"four"
    assert os.path.isfile(os.path.join(path, "sub", "trainer_state.json"))

    # the third save reuses slot0 and leaves no stale file from step 2 behind
    point.save(str(_checkpoint_dir(tmp_path / "c", 6, b"six")), 6, {})
    slot0 = store.list(store.key("train_x", "resume", "slot0"))
    assert len(slot0) == 2

    point.clear()
    assert store.list(store.key("train_x", "resume")) == []


def test_a_resume_point_from_other_params_is_ignored(tmp_path):
    store = LocalArtifactStore(tmp_path / "store", run_id="t")
    ResumePoint(store, "train_x", "h1").save(
        str(_checkpoint_dir(tmp_path / "a", 2, b"x")), 2, {"n": 2}
    )
    assert ResumePoint(store, "train_x", "h2").load(str(tmp_path / "w")) == (None, {})


def test_resume_every_is_not_part_of_the_params_hash():
    p = load_profile("smoke")
    q = load_profile("smoke")
    q["train"]["resume_every"] = 2
    shared = train.load_shared(p)
    assert train._params(p, "dpo", shared, "u") == train._params(q, "dpo", shared, "u")


class Killed(RuntimeError):
    pass


def _prepared_store(tmp_path, name):
    profile = load_profile("smoke")
    profile["train"]["resume_every"] = 2
    store = LocalArtifactStore(tmp_path / name, run_id="smoke")
    data_prep.run(store, profile)
    relabel.run(store, profile)
    return profile, store


@pytest.mark.parametrize("leg", ["dpo", "online_dpo"])
def test_a_killed_stage_resumes_where_it_stopped(tmp_path, monkeypatch, leg):
    """Kill training after step 3 of 4 (resume point at step 2), dispatch the
    stage again, and get the same result as a run that was never killed."""
    from transformers import TrainerCallback

    run = train.make(leg)
    profile, clean = _prepared_store(tmp_path, "clean")
    run(clean, profile)

    profile, store = _prepared_store(tmp_path, "killed")
    real_train = LEGS[leg].train

    class KillAfterStep3(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step == 3:
                raise Killed

    def killed_train(*a, **kw):
        a = list(a)
        a[7] = a[7] + [KillAfterStep3()]
        return real_train(*a, **kw)

    monkeypatch.setattr(LEGS[leg], "train", killed_train)
    with pytest.raises(Killed):
        run(store, profile)
    assert manifest.load(store, f"train_{leg}") is None
    assert store.get_json(store.key(f"train_{leg}", "resume", "resume.json"))["step"] == 2

    monkeypatch.setattr(LEGS[leg], "train", real_train)
    result = run(store, profile)
    assert result["skipped"] is False
    m = manifest.load(store, f"train_{leg}")
    assert m["resumed_from_step"] == 2
    assert store.list(store.key(f"train_{leg}", "resume")) == []  # cleaned up

    stage = f"train_{leg}"
    rows = store.read_jsonl(store.key(stage, "metrics.jsonl"))
    clean_rows = clean.read_jsonl(clean.key(stage, "metrics.jsonl"))
    steps = [r["step"] for r in rows if r["kind"] == "trl" and "loss" in r]
    assert steps == [1, 2, 3, 4]  # no step lost, none repeated
    kl_steps = [r["step"] for r in rows if r["kind"] == "kl"]
    assert kl_steps == [r["step"] for r in clean_rows if r["kind"] == "kl"]

    a = torch.load(store._path(store.key(stage, "checkpoint.pt")), weights_only=True)
    b = torch.load(clean._path(clean.key(stage, "checkpoint.pt")), weights_only=True)
    for k in b:
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-5), k
