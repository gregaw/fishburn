"""Stage: aggregate every leg's results into report.json + report.md.

The markdown is for the person reading the run afterwards: one column per
leg, all from the same shared config (experiment 0114's side-by-side table).
report.json holds the same numbers for later analysis. The stage depends on
every upstream stage's params, so any hyperparameter change re-renders it.
"""

from pipeline import manifest
from pipeline.costs import stage_timer
from pipeline.exec.base import ArtifactStore
from pipeline.shared import load_shared

STAGE = "report"


def _fmt(v, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def leg_summary(store: ArtifactStore, leg: str) -> dict:
    train = store.read_jsonl(store.key(f"train_{leg}", "metrics.jsonl"))
    losses = [r["loss"] for r in train if r["kind"] == "trl" and "loss" in r]
    kls = [r["kl"] for r in train if r["kind"] == "kl"]
    sanity = store.get_json(store.key(f"sanity_{leg}", "sanity.json"))
    checks = sanity["checks"]
    return {
        "steps": len(losses),
        "final_loss": losses[-1] if losses else None,
        "final_kl": kls[-1] if kls else None,
        "milestones_reached": [
            r["milestone"] for r in store.get_json(store.key(f"train_{leg}", "checkpoints.json"))
        ],
        "loss_falls": checks["loss"]["falls"],
        "kl_grows": checks["kl"]["grows"],
        "generations_sane": checks["generations"]["sane"],
        "judge_pref": checks["judge_preference"]["mean_p_trained_over_base"],
        "judge_pref_ci95": checks["judge_preference"]["ci95"],
        "hard_win_rate": checks["judge_preference"]["hard_win_rate"],
        "sanity_all_passed": sanity["all_passed"],
        "eval_cot": store.get_json(store.key(f"eval_cot_{leg}", "metrics.json")),
        "eval_steering": store.get_json(store.key(f"eval_steering_{leg}", "metrics.json")),
    }


def render_md(run_id: str, profile: dict, shared_hash: str, legs: dict, relabel: dict) -> str:
    names = list(legs)
    rows = [
        ("train steps", lambda s: str(s["steps"])),
        ("final loss", lambda s: _fmt(s["final_loss"], ".4f")),
        ("final KL from base (nats/seq)", lambda s: _fmt(s["final_kl"], ".2f")),
        (
            "KL milestones reached",
            lambda s: ", ".join(f"{m:g}" for m in s["milestones_reached"]) or "none",
        ),
        ("loss falls", lambda s: str(s["loss_falls"])),
        ("KL grows", lambda s: str(s["kl_grows"])),
        ("generations sane", lambda s: str(s["generations_sane"])),
        (
            "judge P(trained > base)",
            lambda s: (
                f"{s['judge_pref']:.3f} "
                f"[{s['judge_pref_ci95'][0]:.3f}, {s['judge_pref_ci95'][1]:.3f}]"
            ),
        ),
        ("hard win rate vs base", lambda s: f"{s['hard_win_rate']:.3f}"),
        ("sanity checks all passed", lambda s: str(s["sanity_all_passed"])),
        ("eval_cot cot rate (unjudged)", lambda s: f"{s['eval_cot']['cot_rate']:.2f}"),
        (
            "eval_steering effect (unjudged)",
            lambda s: f"{s['eval_steering']['steering_effect']:+.2f}",
        ),
    ]
    lines = [
        f"# nlhf-cot-monitor run report — `{run_id}`",
        "",
        f"Profile `{profile['profile']}`, base model `{profile['model']['name']}`, "
        f"shared config `{shared_hash}` (identical for every leg).",
        "",
        "| metric | " + " | ".join(f"`{n}`" for n in names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    lines += [f"| {label} | " + " | ".join(f(legs[n]) for n in names) + " |" for label, f in rows]
    lines += [
        "",
        "Relabelling (swapped fraction by split): "
        + ", ".join(f"{split} {st['swapped_fraction']}" for split, st in relabel["splits"].items())
        + ".",
        "",
        "Provenance: `upstream_params_hashes` in report.json; per-stage "
        "manifest.json carries `git_sha`, seeds and `shared_config_hash`.",
        "",
    ]
    return "\n".join(lines)


def run(store: ArtifactStore, profile: dict, force: bool = False) -> dict:
    from pipeline.stages import legs_of, stage_order

    upstream = [s for s in stage_order(profile) if s != STAGE]
    upstream_hashes = {}
    for s in upstream:
        m = manifest.load(store, s)
        if m is None:
            raise RuntimeError(f"report needs completed upstream stage {s!r}")
        upstream_hashes[s] = m["params_hash"]
    params = {"stage": STAGE, "upstream": upstream_hashes}
    phash = manifest.params_hash(params)
    if not force and manifest.is_complete(store, STAGE, phash):
        return {"skipped": True, "stage": STAGE}

    shared_hash = load_shared(profile).hash
    with stage_timer(store, STAGE, gpu_type="none"):
        legs = {leg: leg_summary(store, leg) for leg in legs_of(profile)}
        relabel = store.get_json(store.key("relabel", "summary.json"))
        report = {
            "run_id": store.run_id,
            "profile": profile["profile"],
            "model": profile["model"]["name"],
            "shared_config_hash": shared_hash,
            "legs": legs,
            "relabel": relabel,
            "upstream_params_hashes": upstream_hashes,
        }
        store.put_json(store.key(STAGE, "report.json"), report)
        store.put(
            store.key(STAGE, "report.md"),
            render_md(store.run_id, profile, shared_hash, legs, relabel).encode(),
        )
    manifest.write_manifest(
        store, STAGE, phash, ["report.json", "report.md"], extra={"shared_config_hash": shared_hash}
    )
    return {
        "skipped": False,
        "stage": STAGE,
        "sanity_all_passed": {leg: s["sanity_all_passed"] for leg, s in legs.items()},
    }
