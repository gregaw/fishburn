"""Judge-vs-labels agreement: how often the judge prefers the labelled
`chosen` response, e.g. against UltraFeedback binarized labels.

Works on normalized records ({"prompt", "chosen", "rejected", ...}): the
judge agrees on a record iff P(chosen > rejected) > 0.5. Records flagged
`tie` carry no label signal and are excluded (and counted, so the report
shows how many)."""

from pipeline.judge.base import PreferenceJudge


def agreement_report(records: list[dict], judge: PreferenceJudge, chunk_size: int = 64) -> dict:
    scored = [r for r in records if not r.get("tie")]
    probs: list[float] = []
    for i in range(0, len(scored), chunk_size):
        chunk = scored[i : i + chunk_size]
        probs += judge.prob_prefer_batch([(r["prompt"], r["chosen"], r["rejected"]) for r in chunk])
    agree = sum(p > 0.5 for p in probs)
    return {
        "judge": judge.name,
        "n_records": len(records),
        "n_ties_excluded": len(records) - len(scored),
        "n_scored": len(scored),
        "n_agree": agree,
        "agreement": round(agree / len(scored), 4) if scored else None,
        "mean_p_chosen": round(sum(probs) / len(probs), 4) if probs else None,
    }


def by_score_margin(records: list[dict], judge: PreferenceJudge, margins=(1, 2, 4)) -> dict | None:
    """Agreement by label confidence, for datasets with scores (UltraFeedback's
    GPT-4 ratings). Free after a main pass through a CachedJudge."""
    if not records or "score_chosen" not in records[0]:
        return None
    out = {}
    for m in margins:
        subset = [r for r in records if r["score_chosen"] - r["score_rejected"] >= m]
        sub = agreement_report(subset, judge)
        out[f">={m}"] = {"n": sub["n_scored"], "agreement": sub["agreement"]}
    return out


def seen_unseen_summary(reports: dict) -> dict:
    """reports[(source, split_kind)] -> agreement report, split_kind in
    {"train", "test"}. Per source: test (unseen) agreement, train (seen)
    agreement, and the gap seen - unseen (experiment 0111)."""
    out = {}
    for source in sorted({s for s, _ in reports}):
        test = reports.get((source, "test"), {}).get("agreement")
        train = reports.get((source, "train"), {}).get("agreement")
        out[source] = {
            "unseen_test_agreement": test,
            "seen_train_agreement": train,
            "seen_minus_unseen": (
                round(train - test, 4) if test is not None and train is not None else None
            ),
        }
    return out
