"""Dataset check (experiment 0123): did training raise log p(chosen)
relative to log p(rejected) on held-out labelled pairs?"""

import math

import pytest

from pipeline import prefcheck
from pipeline.models import build_causal_lm, build_tokenizer


def tiny():
    tok = build_tokenizer("tiny-random")
    return tok, build_causal_lm("tiny-random", "float32", tok)


def test_dataset_label_rows_undoes_the_swap_and_keeps_the_verdict():
    rel = [
        {"prompt": "p", "chosen": "a", "rejected": "b", "swapped": False, "p_dataset_chosen": 0.8},
        {"prompt": "q", "chosen": "d", "rejected": "c", "swapped": True, "p_dataset_chosen": 0.3},
    ]
    out = prefcheck.dataset_label_rows(rel)
    assert [(r["chosen"], r["rejected"], r["p_judge"]) for r in out] == [
        ("a", "b", 0.8),
        ("c", "d", 0.3),
    ]


def test_summarize_counts_the_gap_not_the_chosen_probability():
    rows = [{"p_judge": 0.9}, {"p_judge": 0.2}, {"p_judge": 0.5}, {"p_judge": None}]
    base = ([-10.0] * 4, [-10.0] * 4)
    # both answers fall on pair 1, the rejected one faster: still a hit
    trained = ([-12.0, -9.0, -10.0, -11.0], [-15.0, -8.0, -10.0, -10.0])
    out = prefcheck.summarize(rows, base, trained, n_boot=200, seed=1)
    # deltas: +3, -1, 0, -1 -> hits 1, 0, 0.5 (a tie), 0
    assert out["n_pairs"] == 4
    assert out["mean_change_logp_chosen"] == pytest.approx(-0.5)
    assert out["mean_delta"] == pytest.approx(0.25)
    assert out["accuracy_vs_dataset"] == pytest.approx(1.5 / 4)
    # judged pairs exclude ties and missing verdicts: pair 1 (judge agrees with
    # the label, delta > 0) and pair 2 (judge prefers the rejected answer and
    # training moved towards it): both follow the judge
    assert out["n_judged"] == 2
    assert out["judge_agrees_with_dataset"] == 0.5
    assert out["accuracy_vs_judge"] == 1.0


def test_summarize_without_verdicts_reports_dataset_accuracy_only():
    out = prefcheck.summarize(
        [{"p_judge": None}] * 2,
        ([0.0, 0.0], [0.0, 0.0]),
        ([1.0, 1.0], [0.0, 0.0]),
        n_boot=100,
        seed=0,
    )
    assert out["accuracy_vs_dataset"] == 1.0 and out["above_half_beyond_noise"]
    assert "accuracy_vs_judge" not in out


def test_response_logprob_is_a_negative_sum_over_the_response():
    tok, model = tiny()
    model.eval()
    short = prefcheck.response_logprob(model, tok, "is the answer yes or no .", "yes .", "cpu", 64)
    longer = prefcheck.response_logprob(
        model, tok, "is the answer yes or no .", "yes . yes . yes .", "cpu", 64
    )
    assert short < 0 and math.isfinite(short)
    assert longer < short  # more tokens, more negative log-probability


def test_response_logprob_respects_max_length():
    tok, model = tiny()
    model.eval()
    lp = prefcheck.response_logprob(model, tok, "the answer " * 20, "yes " * 40, "cpu", 16)
    assert math.isfinite(lp)


def test_onpolicy_summary_follows_the_judges_pick_and_skips_ties():
    rows = [{"p_judge": 0.8}, {"p_judge": 0.3}, {"p_judge": 0.5}]
    base = ([0.0] * 3, [0.0] * 3)
    trained = ([1.0, 1.0, 5.0], [0.0, 0.0, 0.0])  # moved towards the first answer each time
    out = prefcheck.summarize_onpolicy(rows, base, trained, n_boot=100, seed=0)
    assert out["n_judged"] == 2
    assert out["accuracy_vs_judge"] == 0.5  # right on pair 1, wrong on pair 2


def test_onpolicy_summary_with_only_ties():
    out = prefcheck.summarize_onpolicy(
        [{"p_judge": 0.5}], ([0.0], [0.0]), ([0.0], [0.0]), n_boot=10, seed=0
    )
    assert out["n_judged"] == 0 and out["above_half_beyond_noise"] is None
