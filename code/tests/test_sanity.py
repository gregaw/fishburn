"""Sanity-check helpers (experiments 0113, 0114)."""

import pytest

from pipeline.stages import sanity


def test_preference_averages_both_orderings():
    # a judge that always prefers whatever is shown first is neutral
    assert sanity.preference([0.8, 0.8], [0.8, 0.8]) == [0.5, 0.5]
    assert sanity.preference([0.9], [0.3]) == [0.8]


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    vals = [0.4, 0.6, 0.7, 0.55, 0.65, 0.5, 0.6, 0.7]
    lo, hi = sanity.bootstrap_ci(vals, 500, seed=1)
    assert lo <= sum(vals) / len(vals) <= hi
    assert (lo, hi) == sanity.bootstrap_ci(vals, 500, seed=1)
    assert sanity.bootstrap_ci([0.7] * 10, 100, seed=0) == pytest.approx((0.7, 0.7))


def test_quarter_means_and_distinct_ratio():
    assert sanity.quarter_means([4, 3, 2, 1]) == (4, 1)
    assert sanity.quarter_means([1]) is None
    assert sanity.distinct_ratio("a a a a") == 0.25
    assert sanity.distinct_ratio("") == 0.0


def test_training_checks_read_trl_and_kl_rows():
    metrics = (
        [
            {"kind": "trl", "loss": loss, "rewards/margins": m}
            for loss, m in [(0.7, 0.0), (0.6, 0.1), (0.5, 0.2), (0.4, 0.3)]
        ]
        + [{"kind": "trl", "train_loss": 0.5}]  # TRL's summary row
        + [{"kind": "kl", "kl": k} for k in (0.0, 0.5, 1.5)]
    )
    checks = sanity.training_checks(metrics)
    assert checks["loss"]["falls"] is True
    assert checks["kl"] == {"first": 0.0, "last": 1.5, "grows": True}
    assert checks["reward_margins"] == {"first_quarter": 0.0, "last_quarter": 0.3}


def test_training_checks_flag_flat_runs():
    metrics = [{"kind": "trl", "loss": 0.5}] * 4 + [
        {"kind": "kl", "kl": 0.0},
        {"kind": "kl", "kl": 0.0},
    ]
    checks = sanity.training_checks(metrics)
    assert checks["loss"]["falls"] is False and checks["kl"]["grows"] is False
