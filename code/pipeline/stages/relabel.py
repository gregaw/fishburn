"""Stage: relabel the preference pairs with the shared judge (experiment
0112).

For each pair the judge gives P(chosen > rejected). Where it disagrees with
the dataset label (p < 0.5) the pair is swapped, never dropped: contested
pairs are where Borda and Maximal Lotteries part ways, so dropping them would
remove the signal the comparison exists to measure. A verdict of exactly 0.5 keeps
the dataset label (the tie rule in pipeline/shared.py).

The training pairs (`data_prep/prefs.jsonl`, from each dataset's training
split, which PairRM has probably seen) are what the legs train on. The
held-out pairs (`data_prep/test_prefs.jsonl`, the test split, unseen) are
relabelled only to report the swapped fraction on pairs the judge cannot
remember, beside the training figure.

Every verdict goes through the run's verdict cache, so a re-run, or a later
stage judging the same pair, costs nothing. The manifest records the
subsample sizes, the swapped fraction per split, how many verdicts actually
reached the judge, and what those cost at the profile's measured rate.
"""

import json

from pipeline import manifest
from pipeline.costs import stage_timer
from pipeline.exec.base import ArtifactStore
from pipeline.shared import TIE_RULE, build_judge, load_shared

STAGE = "relabel"
SPLITS = (("train", "prefs.jsonl"), ("test", "test_prefs.jsonl"))


def relabel_rows(rows: list[dict], probs: list[float], swap: bool = True) -> list[dict]:
    """Swap chosen/rejected where the judge prefers the rejected response.
    Each output row keeps the dataset's own label and the verdict. With
    `swap=False` every row keeps the dataset label and only the verdict is
    recorded (experiment 0123: train on the dataset's own preferences)."""
    out = []
    for r, p in zip(rows, probs, strict=True):
        swap_this = swap and p < 0.5
        out.append(
            {
                **r,
                "chosen": r["rejected"] if swap_this else r["chosen"],
                "rejected": r["chosen"] if swap_this else r["rejected"],
                "swapped": swap_this,
                "p_dataset_chosen": p,
                "dataset_chosen": r["chosen"],
            }
        )
    return out


def split_stats(rows: list[dict]) -> dict:
    n = len(rows)
    swapped = sum(r["swapped"] for r in rows)
    return {
        "n_pairs": n,
        "n_swapped": swapped,
        "swapped_fraction": round(swapped / n, 4) if n else None,
        "judge_disagrees_fraction": (
            round(sum(r["p_dataset_chosen"] < 0.5 for r in rows) / n, 4) if n else None
        ),
        "n_exact_ties": sum(r["p_dataset_chosen"] == 0.5 for r in rows),
    }


def run(store: ArtifactStore, profile: dict, force: bool = False, judge=None) -> dict:
    """`judge` overrides the shared judge (tests inject a counting stub)."""
    shared = load_shared(profile)
    up = manifest.load(store, "data_prep")
    if up is None:
        raise RuntimeError("relabel needs a completed data_prep stage")
    params = {
        "stage": STAGE,
        "data_prep_params_hash": up["params_hash"],
        "judge": shared.to_dict()["judge"],
        "tie_rule": TIE_RULE,
    }
    swap = (profile.get("relabel") or {}).get("swap", True)
    if not swap:  # in the hash only when set, so earlier runs keep their hashes
        params["swap"] = False
    phash = manifest.params_hash(params)
    if not force and manifest.is_complete(store, STAGE, phash):
        return {"skipped": True, "stage": STAGE}

    rate = float(profile["judge"].get("usd_per_1k_pairs", 0.0))
    chunk = 64
    with stage_timer(store, STAGE, gpu_type=profile.get("gpu", "none")):
        if judge is None:
            judge = build_judge(shared, store, device=profile["model"]["device"])
        calls_before = getattr(judge, "backend_calls", 0)
        stats, outputs = {}, []
        for split, name in SPLITS:
            key = store.key("data_prep", name)
            if not store.exists(key):
                continue
            rows = store.read_jsonl(key)
            probs: list[float] = []
            for i in range(0, len(rows), chunk):
                probs += judge.prob_prefer_batch(
                    [(r["prompt"], r["chosen"], r["rejected"]) for r in rows[i : i + chunk]]
                )
            relabelled = relabel_rows(rows, probs, swap)
            store.put(
                store.key(STAGE, name),
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in relabelled).encode(),
            )
            stats[split] = split_stats(relabelled)
            outputs.append(name)
        judged = getattr(judge, "backend_calls", 0) - calls_before
        summary = {
            "source": profile["data_prep"].get("source", "synthetic"),
            "splits": stats,
            "swap": swap,
            "judge": judge.name,
            "verdicts_from_judge": judged,
            "est_usd": round(judged / 1000 * rate, 4),
            "usd_per_1k_pairs": rate,
            "full_corpus_note": (
                "relabels a fixed seeded subsample (data_prep's n_pairs and "
                "n_test_pairs); the full corpora (~161k HH-RLHF, ~61k "
                "UltraFeedback training pairs) would cost ~$29 at this rate"
            ),
        }
        store.put_json(store.key(STAGE, "summary.json"), summary)
    manifest.write_manifest(store, STAGE, phash, outputs + ["summary.json"], extra=summary)
    return {"skipped": False, "stage": STAGE, **summary}
