"""Relabel stage (experiment 0112): swap where the judge
disagrees, never drop, keep the dataset label, cache every verdict."""

import pytest

from pipeline import manifest
from pipeline.config import load_profile
from pipeline.exec.local_backend import LocalArtifactStore
from pipeline.judge import CachedJudge, StubJudge
from pipeline.stages import data_prep, relabel


class CountingJudge(StubJudge):
    name = "counting"

    def __init__(self, fn):
        super().__init__(fn=fn)
        self.calls = 0

    def prob_prefer_batch(self, items):
        self.calls += len(items)
        return super().prob_prefer_batch(items)


@pytest.fixture
def store(tmp_path):
    return LocalArtifactStore(tmp_path, run_id="t")


def test_relabel_rows_swaps_disagreements_and_keeps_ties():
    rows = [{"prompt": "p", "chosen": "a", "rejected": "b"}] * 3
    out = relabel.relabel_rows(rows, [0.9, 0.1, 0.5])
    assert [(r["chosen"], r["rejected"], r["swapped"]) for r in out] == [
        ("a", "b", False),
        ("b", "a", True),
        ("a", "b", False),
    ]
    assert all(r["dataset_chosen"] == "a" for r in out)
    stats = relabel.split_stats(out)
    assert stats == {
        "n_pairs": 3,
        "n_swapped": 1,
        "swapped_fraction": 0.3333,
        "judge_disagrees_fraction": 0.3333,
        "n_exact_ties": 1,
    }


def test_relabel_rows_without_swap_keeps_every_dataset_label():
    """experiment 0123: train on the dataset's own preferences, verdicts kept"""
    rows = [{"prompt": "p", "chosen": "a", "rejected": "b"}] * 2
    out = relabel.relabel_rows(rows, [0.9, 0.1], swap=False)
    assert [(r["chosen"], r["rejected"], r["swapped"]) for r in out] == [
        ("a", "b", False),
        ("a", "b", False),
    ]
    assert [r["p_dataset_chosen"] for r in out] == [0.9, 0.1]
    stats = relabel.split_stats(out)
    assert stats["n_swapped"] == 0 and stats["judge_disagrees_fraction"] == 0.5


def test_no_swap_changes_the_hash_and_the_training_pairs(store):
    profile = load_profile("smoke")
    data_prep.run(store, profile)
    # disagrees everywhere (uncached: a constant would flip)
    judge = StubJudge(fn=lambda p, a, b: 0.0)
    relabel.run(store, profile, judge=judge)
    swapped = store.read_jsonl(store.key("relabel", "prefs.jsonl"))
    h1 = manifest.load(store, "relabel")["params_hash"]
    profile["relabel"] = {"swap": False}
    relabel.run(store, profile, judge=judge)
    kept = store.read_jsonl(store.key("relabel", "prefs.jsonl"))
    assert manifest.load(store, "relabel")["params_hash"] != h1
    assert all(r["swapped"] for r in swapped)
    assert not any(r["swapped"] for r in kept)
    assert [r["chosen"] for r in kept] == [r["dataset_chosen"] for r in kept]


def test_stage_never_drops_pairs_and_reports_seen_and_unseen(store):
    profile = load_profile("smoke")
    data_prep.run(store, profile)
    # disagree with every pair whose chosen response mentions "yes"
    backend = CountingJudge(lambda p, y1, y2: 0.2 if "yes" in y1 else 0.8)
    result = relabel.run(store, profile, judge=CachedJudge(backend, store))
    train_in = store.read_jsonl(store.key("data_prep", "prefs.jsonl"))
    train_out = store.read_jsonl(store.key("relabel", "prefs.jsonl"))
    assert len(train_out) == len(train_in)
    expected = sum("yes" in r["chosen"] for r in train_in)
    assert result["splits"]["train"]["n_swapped"] == expected
    assert set(result["splits"]) == {"train", "test"}
    assert result["verdicts_from_judge"] == backend.calls > 0
    m = manifest.load(store, "relabel")
    assert m["splits"] == result["splits"] and "est_usd" in m


def test_rerun_is_free_from_the_cache(store):
    profile = load_profile("smoke")
    data_prep.run(store, profile)
    first = CountingJudge(lambda p, y1, y2: 0.7)
    relabel.run(store, profile, judge=CachedJudge(first, store))
    second = CountingJudge(lambda p, y1, y2: 0.7)
    result = relabel.run(store, profile, force=True, judge=CachedJudge(second, store))
    assert second.calls == 0 and result["verdicts_from_judge"] == 0
    assert result["est_usd"] == 0.0


def test_cost_estimate_uses_the_profile_rate(store):
    profile = load_profile("smoke")
    profile["judge"]["usd_per_1k_pairs"] = 0.13
    data_prep.run(store, profile)
    result = relabel.run(store, profile, judge=CachedJudge(CountingJudge(lambda *a: 0.6), store))
    # priced on verdicts that reached the judge (duplicates hit the cache)
    assert result["verdicts_from_judge"] > 0
    assert result["est_usd"] == round(result["verdicts_from_judge"] / 1000 * 0.13, 4)


def test_relabel_needs_data_prep(store):
    with pytest.raises(RuntimeError, match="data_prep"):
        relabel.run(store, load_profile("smoke"))
