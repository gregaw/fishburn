"""Stage units: synthetic data prep, skip and force, eval scoring, DPO without a judge."""

import pytest

from pipeline.config import load_profile
from pipeline.exec.local_backend import LocalArtifactStore
from pipeline.stages import data_prep
from pipeline.stages.eval_cot import score_text


@pytest.fixture
def store(tmp_path):
    return LocalArtifactStore(tmp_path, run_id="t")


def test_sample_pairs_deterministic():
    a = data_prep.sample_pairs(8, seed=7)
    assert a == data_prep.sample_pairs(8, seed=7)
    assert a != data_prep.sample_pairs(8, seed=8)
    for row in a:
        assert set(row) == {"prompt", "chosen", "rejected"}
        assert "think" in row["chosen"] and "think" not in row["rejected"]


def test_data_prep_skip_and_force(store):
    profile = load_profile("smoke")
    first = data_prep.run(store, profile)
    assert first == {
        "skipped": False,
        "stage": "data_prep",
        "n_pairs": profile["data_prep"]["n_pairs"],
    }
    assert data_prep.run(store, profile)["skipped"] is True
    assert data_prep.run(store, profile, force=True)["skipped"] is False
    # force re-run overwrites rather than appends
    pairs = store.read_jsonl(store.key("data_prep", "prefs.jsonl"))
    assert len(pairs) == profile["data_prep"]["n_pairs"]


def test_data_prep_param_change_invalidates(store):
    data_prep.run(store, load_profile("smoke"))
    assert data_prep.run(store, load_profile("smoke", seed=99))["skipped"] is False


def test_data_prep_text_stays_in_tiny_vocab(store):
    from pipeline.models import TINY_VOCAB_WORDS

    data_prep.run(store, load_profile("smoke"))
    vocab = set(TINY_VOCAB_WORDS)
    for row in store.read_jsonl(store.key("data_prep", "prefs.jsonl")):
        for field in ("prompt", "chosen", "rejected"):
            assert set(row[field].split()) <= vocab, row[field]


def test_score_text():
    assert score_text("think step . the answer is yes .") == {"parsed": True, "cot": True}
    assert score_text("the answer is no .") == {"parsed": True, "cot": False}
    assert score_text(". . .") == {"parsed": False, "cot": False}


def test_dpo_trains_without_loading_the_judge(tmp_path, monkeypatch):
    """Experiment 0122: an 8B judge loaded beside DPO ran an L4 out of
    memory, and DPO never asks it anything."""
    from pipeline.config import load_profile
    from pipeline.exec.local_backend import LocalArtifactStore
    from pipeline.stages import data_prep, relabel, train

    def no_judge(*a, **k):
        raise AssertionError("DPO must not build the judge")

    store = LocalArtifactStore(tmp_path, run_id="t")
    profile = load_profile("smoke")
    profile["train"]["steps"] = 1
    data_prep.run(store, profile)
    relabel.run(store, profile)
    monkeypatch.setattr(train, "build_trl_judge", no_judge)
    assert train.make("dpo")(store, profile)["skipped"] is False
