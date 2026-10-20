"""Judge interface, stub determinism, verdict cache, agreement metric.
Offline: the PairRM backend itself is cloud-only (lazy import) and is
exercised by the Modal agreement run, not here."""

import pytest

from pipeline.exec.local_backend import LocalArtifactStore
from pipeline.judge import CachedJudge, StubJudge, agreement_report


@pytest.fixture
def store(tmp_path):
    return LocalArtifactStore(tmp_path, run_id="t")


def test_stub_judge_deterministic_and_bounded():
    o = StubJudge()
    items = [
        ("p", "a longer answer indeed", "short"),
        ("p", "short", "a longer answer indeed"),
        ("p", "same-len", "len-same"),
    ]
    first = o.prob_prefer_batch(items)
    assert first == o.prob_prefer_batch(items)
    assert all(0.0 <= p <= 1.0 for p in first)
    assert first[0] > 0.5 > first[1]  # longer response preferred


def test_stub_judge_injected_fn():
    o = StubJudge(fn=lambda p, y1, y2: 0.9)
    assert o.prob_prefer("x", "a", "b") == 0.9


class CountingJudge(StubJudge):
    name = "counting"

    def __init__(self):
        super().__init__()
        self.calls = 0

    def prob_prefer_batch(self, items):
        self.calls += len(items)
        return super().prob_prefer_batch(items)


def test_cached_judge_hits_cache(store):
    backend = CountingJudge()
    cached = CachedJudge(backend, store)
    p1 = cached.prob_prefer("prompt", "yes yes yes", "no")
    assert backend.calls == 1
    assert cached.prob_prefer("prompt", "yes yes yes", "no") == p1
    assert backend.calls == 1  # second query served from the store
    # cache survives a fresh wrapper over the same store
    cached2 = CachedJudge(CountingJudge(), store)
    assert cached2.prob_prefer("prompt", "yes yes yes", "no") == p1
    assert cached2.backend.calls == 0


def test_cached_judge_flips_order(store):
    backend = CountingJudge()
    cached = CachedJudge(backend, store)
    p = cached.prob_prefer("prompt", "bbb", "aa")
    assert cached.prob_prefer("prompt", "aa", "bbb") == pytest.approx(1 - p)
    assert backend.calls == 1  # the flipped query is the same cached verdict


def test_cached_judge_batch_mixes_hits_and_misses(store):
    backend = CountingJudge()
    cached = CachedJudge(backend, store)
    cached.prob_prefer("p", "aaaa", "b")
    probs = cached.prob_prefer_batch([("p", "aaaa", "b"), ("p", "cccc", "d"), ("p", "b", "aaaa")])
    assert backend.calls == 2  # only ("p","cccc","d") was a miss
    assert probs[0] == pytest.approx(1 - probs[2])


def test_cached_judge_isolated_by_backend_name(store):
    a = CachedJudge(StubJudge(fn=lambda *_: 0.9), store)
    a.backend.name = "backend-a"
    b = CachedJudge(StubJudge(fn=lambda *_: 0.1), store)
    b.backend.name = "backend-b"
    assert a.prob_prefer("p", "x", "y") == 0.9
    assert b.prob_prefer("p", "x", "y") == 0.1  # not backend-a's verdict


def test_agreement_report():
    records = [
        {"prompt": "p", "chosen": "long correct answer", "rejected": "meh"},
        {"prompt": "p", "chosen": "x", "rejected": "a very long wrong one"},
        {"prompt": "p", "chosen": "ignored", "rejected": "row", "tie": True},
    ]
    report = agreement_report(records, StubJudge(), chunk_size=2)
    assert report["n_records"] == 3
    assert report["n_ties_excluded"] == 1
    assert report["n_scored"] == 2
    assert report["n_agree"] == 1  # stub prefers length: right on 0, wrong on 1
    assert report["agreement"] == 0.5
    assert 0.0 <= report["mean_p_chosen"] <= 1.0


def test_agreement_report_empty():
    report = agreement_report([], StubJudge())
    assert report["agreement"] is None and report["n_scored"] == 0


def test_by_score_margin_buckets_and_skips_unscored(store):
    from pipeline.judge.agreement import by_score_margin

    recs = [
        {
            "prompt": "p",
            "chosen": "long answer here",
            "rejected": "no",
            "score_chosen": 9.0,
            "score_rejected": 9.0 - m,
        }
        for m in (1, 2, 5)
    ]
    out = by_score_margin(recs, CachedJudge(StubJudge(), store))
    assert {k: v["n"] for k, v in out.items()} == {">=1": 3, ">=2": 2, ">=4": 1}
    assert by_score_margin([{"prompt": "p", "chosen": "a", "rejected": "b"}], StubJudge()) is None


def test_seen_unseen_summary_gap():
    from pipeline.judge.agreement import seen_unseen_summary

    reports = {
        ("uf", "test"): {"agreement": 0.8},
        ("uf", "train"): {"agreement": 0.9},
        ("hh", "test"): {"agreement": 0.6},
    }
    out = seen_unseen_summary(reports)
    assert out["uf"]["seen_minus_unseen"] == 0.1
    assert out["hh"] == {
        "unseen_test_agreement": 0.6,
        "seen_train_agreement": None,
        "seen_minus_unseen": None,
    }
