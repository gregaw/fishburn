"""Stage: sanity checks for one leg (`sanity_<leg>`), experiments 0113 and
0114.

These say whether a leg learned something, not whether the model got better:

- loss falls: mean logged loss, last quarter of steps vs first quarter.
- KL grows: last KL-from-base estimate above the first, and positive.
- generations sane: non-empty, not degenerate repetition (distinct-token ratio).
- judge check: mean P(trained > base) over both orderings on held-out
  prompts, with a bootstrap 95 % interval above 0.5.
- dataset check (with `sanity.pref_pairs`): on held-out labelled pairs,
  training raised log p(chosen) relative to log p(rejected) on more than half,
  beyond noise (pipeline/prefcheck.py).
- own-answers check (with `sanity.onpolicy_check`, reported only): the same gap
  on pairs of two base answers per prompt, with the judge's pick as "chosen".

At smoke scale the judge check is the mean probability, not a hard win
rate: a few hundred prompts make a hard win rate too noisy. The
hard win rate is reported beside it for the real run. TRL's logged reward
margins are reported too. Raw generations are kept so a person can read them.
"""

import json
import random

from pipeline import manifest, prefcheck
from pipeline.costs import stage_timer
from pipeline.exec.base import ArtifactStore
from pipeline.generate import sample_models
from pipeline.models import build_tokenizer, load_state
from pipeline.shared import build_judge, load_shared, render_prompt, with_judge
from pipeline.stages.train import CHECKPOINT, build_base, leg_train_config, stage_name

MIN_NONEMPTY = 0.9
MIN_DISTINCT = 0.3


def stage(leg: str) -> str:
    return f"sanity_{leg}"


def quarter_means(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    q = max(1, len(values) // 4)
    return sum(values[:q]) / q, sum(values[-q:]) / q


def distinct_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


def bootstrap_ci(
    values: list[float], n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def preference(p_tb: list[float], p_bt: list[float]) -> list[float]:
    """Per-prompt P(trained > base), averaged over both orderings so a judge
    with a position bias cannot pass the check by itself."""
    return [(a + (1.0 - b)) / 2 for a, b in zip(p_tb, p_bt, strict=True)]


def training_checks(metrics: list[dict]) -> dict:
    trl = [m for m in metrics if m["kind"] == "trl" and "loss" in m]
    kls = [m for m in metrics if m["kind"] == "kl"]
    out: dict = {}
    qm = quarter_means([m["loss"] for m in trl])
    out["loss"] = (
        {"first_quarter": qm[0], "last_quarter": qm[1], "falls": qm[1] < qm[0]}
        if qm
        else {"falls": None}
    )
    if len(kls) >= 2:
        out["kl"] = {
            "first": kls[0]["kl"],
            "last": kls[-1]["kl"],
            "grows": kls[-1]["kl"] > kls[0]["kl"] and kls[-1]["kl"] > 0,
        }
    else:
        out["kl"] = {"grows": None}
    margins = [m["rewards/margins"] for m in trl if "rewards/margins" in m]
    qm = quarter_means(margins)
    out["reward_margins"] = {"first_quarter": qm[0], "last_quarter": qm[1]} if qm else None
    return out


def make(leg: str):
    name = stage(leg)
    upstream = stage_name(leg)

    def run(store: ArtifactStore, profile: dict, force: bool = False) -> dict:
        shared = load_shared(profile)
        up = manifest.load(store, upstream)
        if up is None:
            raise RuntimeError(f"{name} needs a completed {upstream} stage")
        s = profile["sanity"]
        params = {"stage": name, "upstream": up["params_hash"], "sanity": s}
        phash = manifest.params_hash(params)
        if not force and manifest.is_complete(store, name, phash):
            return {"skipped": True, "stage": name}

        device = profile["model"]["device"]
        tcfg = leg_train_config(profile, leg)
        with stage_timer(store, name, gpu_type=profile.get("gpu", "none")):
            tokenizer = build_tokenizer(shared.base_model)
            tokenizer.padding_side = "left"
            base = build_base(shared, tokenizer, tcfg["seed"]).to(device).eval()
            trained = build_base(shared, tokenizer, tcfg["seed"])
            load_state(trained, store.get(store.key(upstream, CHECKPOINT)))
            trained.to(device).eval()

            # experiment 0123: the dataset check, while both models are
            # loaded (the vLLM path below hands them over and drops them)
            pref_check = None
            n_pairs = s.get("pref_pairs", 0)
            if n_pairs:
                rows = prefcheck.dataset_label_rows(
                    store.read_jsonl(store.key("relabel", "test_prefs.jsonl"))
                )[:n_pairs]
                q = [render_prompt(shared, tokenizer, r["prompt"]) for r in rows]
                max_len = tcfg["max_length"]
                pref_check = prefcheck.summarize(
                    rows,
                    prefcheck.pair_logprobs(base, tokenizer, q, rows, device, max_len),
                    prefcheck.pair_logprobs(trained, tokenizer, q, rows, device, max_len),
                    s["n_bootstrap"],
                    s["seed"],
                )

            raw = [
                r["prompt"] for r in store.read_jsonl(store.key("data_prep", "eval_prompts.jsonl"))
            ][: s["n_prompts"]]
            rendered = [render_prompt(shared, tokenizer, p) for p in raw]

            # backend "vllm" samples both models with vLLM (experiment 0122)
            models = [base, trained]
            del base, trained
            base_txt, trained_txt = sample_models(
                models,
                tokenizer,
                rendered,
                shared.sampling,
                s["seed"],
                device,
                backend=s.get("backend", "hf"),
                gpu_memory_utilization=s.get("gpu_memory_utilization", 0.4),
            )

            judge = build_judge(shared, store, device=device)
            p_tb = judge.prob_prefer_batch(list(zip(raw, trained_txt, base_txt)))
            p_bt = judge.prob_prefer_batch(list(zip(raw, base_txt, trained_txt)))
            pref = preference(p_tb, p_bt)
            mean = sum(pref) / len(pref)
            lo, hi = bootstrap_ci(pref, s["n_bootstrap"], s["seed"])
            wins = sum(1.0 if p > 0.5 else 0.5 if p == 0.5 else 0.0 for p in pref)

            # experiment 0123: the own-answers check. A second base
            # answer per prompt (another seed), the judge picks between the two
            # base answers, and the check asks whether training moved towards
            # the judge's pick. Unlike dataset pairs these are answers the
            # policy could have written, which is all an online leg shapes.
            onpolicy = None
            if s.get("onpolicy_check"):
                alt_base = build_base(shared, tokenizer, tcfg["seed"]).to(device).eval()
                [alt_txt] = sample_models(
                    [alt_base],
                    tokenizer,
                    rendered,
                    shared.sampling,
                    s["seed"] + 1,
                    device,
                    backend=s.get("backend", "hf"),
                    gpu_memory_utilization=s.get("gpu_memory_utilization", 0.4),
                )
                del alt_base
                p_ab = preference(
                    judge.prob_prefer_batch(list(zip(raw, base_txt, alt_txt))),
                    judge.prob_prefer_batch(list(zip(raw, alt_txt, base_txt))),
                )
                rows = [
                    {"prompt": q, "chosen": a, "rejected": b, "p_judge": p}
                    for q, a, b, p in zip(raw, base_txt, alt_txt, p_ab)
                ]
                base = build_base(shared, tokenizer, tcfg["seed"]).to(device).eval()
                lp_base = prefcheck.pair_logprobs(
                    base, tokenizer, rendered, rows, device, tcfg["max_length"]
                )
                del base
                trained = build_base(shared, tokenizer, tcfg["seed"])
                load_state(trained, store.get(store.key(upstream, CHECKPOINT)))
                trained.to(device).eval()
                lp_trained = prefcheck.pair_logprobs(
                    trained, tokenizer, rendered, rows, device, tcfg["max_length"]
                )
                del trained
                onpolicy = prefcheck.summarize_onpolicy(
                    rows, lp_base, lp_trained, s["n_bootstrap"], s["seed"]
                )
            del judge
            # second opinions, reported beside the shared judge's verdict but
            # never deciding the check (experiment 0121)
            others = {}
            for spec in s.get("extra_judges", []):
                o = build_judge(with_judge(shared, spec), store, device=device)
                q = preference(
                    o.prob_prefer_batch(list(zip(raw, trained_txt, base_txt))),
                    o.prob_prefer_batch(list(zip(raw, base_txt, trained_txt))),
                )
                qlo, qhi = bootstrap_ci(q, s["n_bootstrap"], s["seed"])
                others[o.backend.name] = {
                    "mean_p_trained_over_base": sum(q) / len(q),
                    "ci95": [qlo, qhi],
                    "per_prompt": q,
                }
                del o

            metrics = store.read_jsonl(store.key(upstream, "metrics.jsonl"))
            checks = training_checks(metrics)
            nonempty = sum(bool(t) for t in trained_txt) / len(trained_txt)
            distinct = sum(distinct_ratio(t) for t in trained_txt) / len(trained_txt)
            checks["generations"] = {
                "nonempty_fraction": nonempty,
                "mean_distinct_ratio": distinct,
                "mean_words": sum(len(t.split()) for t in trained_txt) / len(trained_txt),
                "base_mean_words": sum(len(t.split()) for t in base_txt) / len(base_txt),
                "sane": nonempty >= MIN_NONEMPTY and distinct >= MIN_DISTINCT,
            }
            checks["judge_preference"] = {
                "mean_p_trained_over_base": mean,
                "ci95": [lo, hi],
                "hard_win_rate": wins / len(pref),
                "n_prompts": len(pref),
                "above_half_beyond_noise": lo > 0.5,
            }
            if others:
                checks["other_judges"] = {
                    k: {kk: vv for kk, vv in v.items() if kk != "per_prompt"}
                    for k, v in others.items()
                }
            flags = [
                checks["loss"]["falls"],
                checks["kl"]["grows"],
                checks["generations"]["sane"],
                checks["judge_preference"]["above_half_beyond_noise"],
            ]
            if pref_check is not None:
                checks["preference_check"] = pref_check
                flags.append(pref_check["above_half_beyond_noise"])
            if onpolicy is not None:  # reported, not a pass flag: it reads the judge, not the data
                checks["onpolicy_check"] = onpolicy
            report = {
                "leg": leg,
                "checks": checks,
                "all_passed": all(f is True for f in flags),
                "shared_config_hash": shared.hash,
            }
            store.put_json(store.key(name, "sanity.json"), report)
            store.put(
                store.key(name, "samples.jsonl"),
                "".join(
                    json.dumps(
                        {
                            "prompt": p,
                            "base": b,
                            "trained": t,
                            "p_trained_over_base": q,
                            **{f"p_{k}": v["per_prompt"][i] for k, v in others.items()},
                        },
                        sort_keys=True,
                    )
                    + "\n"
                    for i, (p, b, t, q) in enumerate(zip(raw, base_txt, trained_txt, pref))
                ).encode(),
            )
        manifest.write_manifest(
            store,
            name,
            phash,
            ["sanity.json", "samples.jsonl"],
            extra={
                "leg": leg,
                "all_passed": report["all_passed"],
                "shared_config_hash": shared.hash,
            },
        )
        return {
            "skipped": False,
            "stage": name,
            "leg": leg,
            "all_passed": report["all_passed"],
            "mean_p_trained_over_base": mean,
            "ci95": [lo, hi],
        }

    run.__name__ = f"run_{name}"
    return run
