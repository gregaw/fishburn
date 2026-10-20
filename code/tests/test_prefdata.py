"""Loaders: pure normalization on fixture rows, with no network and no `datasets`."""

import pytest

from pipeline.config import load_profile
from pipeline.exec.local_backend import LocalArtifactStore
from pipeline.prefdata import loaders
from pipeline.prefdata.loaders import (
    normalize_hh,
    normalize_ultrafeedback,
    subsample,
)
from pipeline.stages import data_prep

UF_ROW = {
    "prompt": "What is the capital of France?",
    "chosen": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital is Paris."},
    ],
    "rejected": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "London."},
    ],
    "score_chosen": 9.0,
    "score_rejected": 2.0,
}


def test_normalize_ultrafeedback():
    rec = normalize_ultrafeedback(UF_ROW)
    assert rec == {
        "prompt": "What is the capital of France?",
        "chosen": "The capital is Paris.",
        "rejected": "London.",
        "source": "ultrafeedback",
        "score_chosen": 9.0,
        "score_rejected": 2.0,
        "tie": False,
    }


def test_normalize_ultrafeedback_tie_flag():
    row = dict(UF_ROW, score_chosen=5.0, score_rejected=5.0)
    assert normalize_ultrafeedback(row)["tie"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        {"prompt": ""},
        {"chosen": []},
        {"chosen": [{"role": "user", "content": "no assistant turn"}]},
        {"rejected": [{"role": "assistant", "content": None}]},
        # identical responses carry no preference
        {"rejected": [{"role": "assistant", "content": "The capital is Paris."}]},
    ],
)
def test_normalize_ultrafeedback_rejects_unusable(mutation):
    assert normalize_ultrafeedback(dict(UF_ROW, **mutation)) is None


HH_PROMPT = "\n\nHuman: Is water wet?\n\nAssistant: Yes.\n\nHuman: Why?\n\nAssistant:"


def test_normalize_hh_multi_turn():
    rec = normalize_hh(
        {"chosen": HH_PROMPT + " Because it is a liquid.", "rejected": HH_PROMPT + " No idea."}
    )
    assert rec == {
        "prompt": HH_PROMPT.strip(),
        "chosen": "Because it is a liquid.",
        "rejected": "No idea.",
        "source": "hh_rlhf",
    }


def test_normalize_hh_rejects_unusable():
    # no Assistant marker at all
    assert normalize_hh({"chosen": "\n\nHuman: hi", "rejected": "x"}) is None
    # prompt prefixes differ -> the pair is meaningless
    assert (
        normalize_hh(
            {
                "chosen": "\n\nHuman: A\n\nAssistant: yes",
                "rejected": "\n\nHuman: B\n\nAssistant: no",
            }
        )
        is None
    )
    # identical responses
    assert normalize_hh({"chosen": HH_PROMPT + " same", "rejected": HH_PROMPT + " same"}) is None


def test_subsample_deterministic():
    rows = [{"i": i} for i in range(100)]
    assert subsample(rows, 10, seed=7) == subsample(rows, 10, seed=7)
    assert subsample(rows, 10, seed=7) != subsample(rows, 10, seed=8)
    assert subsample(rows, 200, seed=7) == rows  # n >= len: all, stable order


def test_load_records_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        loaders.load_records("nope", 4, seed=1)


def test_data_prep_real_source_via_registry(tmp_path, monkeypatch):
    """The stage's real-source path, offline: monkeypatch the loader registry
    and check artifacts, manifest and skip logic."""
    fake = [
        {"prompt": f"p{i}", "chosen": f"c{i}", "rejected": f"r{i}", "source": "ultrafeedback"}
        for i in range(10)
    ]
    monkeypatch.setitem(loaders.LOADERS, "ultrafeedback", lambda n, seed, **kw: fake[:n])
    store = LocalArtifactStore(tmp_path, run_id="t")
    profile = load_profile("smoke")
    profile["data_prep"].update(source="ultrafeedback", n_pairs=6, n_eval_prompts=2)

    result = data_prep.run(store, profile)
    assert result == {"skipped": False, "stage": "data_prep", "n_pairs": 6}
    pairs = store.read_jsonl(store.key("data_prep", "prefs.jsonl"))
    prompts = store.read_jsonl(store.key("data_prep", "eval_prompts.jsonl"))
    assert len(pairs) == 6 and len(prompts) == 2
    # held-out eval prompts don't overlap the training pairs
    assert {p["prompt"] for p in prompts}.isdisjoint({r["prompt"] for r in pairs})
    assert data_prep.run(store, profile)["skipped"] is True
    # source is part of the params hash: switching invalidates
    profile["data_prep"]["source"] = "synthetic"
    assert data_prep.run(store, profile)["skipped"] is False


def test_within_chars_counts_prompt_plus_longer_response():
    rec = {"prompt": "abcd", "chosen": "x" * 6, "rejected": "y" * 3}
    assert loaders.within_chars(rec, None)
    assert loaders.within_chars(rec, 10)
    assert not loaders.within_chars(rec, 9)


def test_data_prep_test_split_and_length_filter_reach_the_loader(tmp_path, monkeypatch):
    """n_test_pairs loads the dataset's held-out split (experiment 0112);
    max_chars is passed through to both loads and enters the params hash."""
    calls = []

    def fake_loader(n, seed, **kw):
        calls.append(kw)
        tag = kw.get("split", "train")
        return [
            {"prompt": f"{tag}-p{i}", "chosen": "c", "rejected": "r", "source": "ultrafeedback"}
            for i in range(n)
        ]

    monkeypatch.setitem(loaders.LOADERS, "ultrafeedback", fake_loader)
    store = LocalArtifactStore(tmp_path, run_id="t")
    profile = load_profile("smoke")
    profile["data_prep"].update(
        source="ultrafeedback", n_pairs=4, n_eval_prompts=2, n_test_pairs=3, max_chars=500
    )
    data_prep.run(store, profile)
    assert calls == [
        {"max_chars": 500},
        {"max_chars": 500, "split": loaders.TEST_SPLITS["ultrafeedback"]},
    ]
    test = store.read_jsonl(store.key("data_prep", "test_prefs.jsonl"))
    assert [r["prompt"] for r in test] == ["test_prefs-p0", "test_prefs-p1", "test_prefs-p2"]
    profile["data_prep"]["max_chars"] = 400
    assert data_prep.run(store, profile)["skipped"] is False
