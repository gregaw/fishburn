"""Stage registry. Stage functions are plain `run(store, profile, force)`,
with no Modal imports anywhere under stages/ (tests/test_no_modal_imports.py
checks it), so the local e2e test exercises the real code path.

Phase 1 (experiments 0108-0115) makes the training, sanity and eval stages
per leg: `train_<leg>`, `sanity_<leg>`, `eval_cot_<leg>`,
`eval_steering_<leg>` for each leg in pipeline/legs/. `stage_order(profile)`
gives the run order for the legs a profile selects.
"""

from pipeline.legs import LEGS
from pipeline.stages import data_prep, eval_cot, eval_steering, relabel, report, sanity, train

PER_LEG = ("train", "sanity", "eval_cot", "eval_steering")

STAGES = {
    "data_prep": data_prep.run,
    "relabel": relabel.run,
    **{f"train_{leg}": train.make(leg) for leg in LEGS},
    **{f"sanity_{leg}": sanity.make(leg) for leg in LEGS},
    **{f"eval_cot_{leg}": eval_cot.make(leg) for leg in LEGS},
    **{f"eval_steering_{leg}": eval_steering.make(leg) for leg in LEGS},
    "report": report.run,
}


def legs_of(profile: dict) -> tuple[str, ...]:
    legs = tuple(profile.get("legs") or LEGS)
    unknown = set(legs) - set(LEGS)
    if unknown:
        raise ValueError(f"unknown leg(s) {sorted(unknown)}, expected from {tuple(LEGS)}")
    return legs


def stage_order(profile: dict) -> tuple[str, ...]:
    """data_prep, relabel, then each leg's train -> sanity -> evals (one leg
    at a time, so a canary leg finishes before the next starts), then the
    report."""
    per_leg = tuple(f"{kind}_{leg}" for leg in legs_of(profile) for kind in PER_LEG)
    return ("data_prep", "relabel", *per_leg, "report")


# All legs, for callers that have no profile in hand.
ORDER = stage_order({})
