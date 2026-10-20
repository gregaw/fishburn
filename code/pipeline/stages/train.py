"""Stage: train one leg (`train_<leg>`), experiments 0108-0114.

Everything a leg must not vary lives here and is identical for all legs: the
base checkpoint and its frozen reference copy, the rendered prompts, the
judge (one cached judge, wrapped for TRL), the KL-milestone ladder and its
estimator, the artifact layout and the manifest. The leg module only says
which trainer turns those into updates (pipeline/legs/).

Artifacts under `train_<leg>/`:

| key | contents |
| --- | --- |
| `checkpoint.pt` | final policy state_dict (what eval stages load) |
| `checkpoints/kl_<m>.pt` | policy the first time KL from base reached m nats |
| `checkpoints.json` | the ladder index: milestone, step, KL, key |
| `metrics.jsonl` | TRL's log rows (`kind: trl`) and KL estimates (`kind: kl`) |
| `resume/` | the last resume point, while the stage is unfinished (pipeline/resume.py) |

With `train.resume_every: N` a stage that dies mid-run continues from its
last resume point when the same run id is dispatched again.
"""

import copy
import json
import tempfile

from pipeline import manifest
from pipeline.costs import stage_timer
from pipeline.exec.base import ArtifactStore
from pipeline.models import build_tokenizer, save_state
from pipeline.shared import build_judge, load_shared, raw_prompt_lookup, render_prompt

CHECKPOINT = "checkpoint.pt"


def stage_name(leg: str) -> str:
    return f"train_{leg}"


def leg_train_config(profile: dict, leg: str) -> dict:
    """profile["train"] with the leg's overrides applied (recorded in the
    params hash, so an override is never silent)."""
    t = dict(profile["train"])
    overrides = t.pop("overrides", {}) or {}
    return {**t, **overrides.get(leg, {})}


def build_base(shared, tokenizer, seed: int):
    """The base checkpoint. Deterministic in `seed` for the tiny random model,
    so later stages can rebuild the same base without storing it."""
    import torch

    from pipeline.models import build_causal_lm

    torch.manual_seed(seed)
    return build_causal_lm(shared.base_model, shared.dtype, tokenizer)


def kl_prompts(store: ArtifactStore, shared, tokenizer) -> list[str]:
    """Held-out prompts for the KL estimator: the first n eval prompts, so
    every leg measures KL on exactly the same strings."""
    rows = store.read_jsonl(store.key("data_prep", "eval_prompts.jsonl"))
    prompts = [r["prompt"] for r in rows][: shared.kl.n_prompts]
    return [render_prompt(shared, tokenizer, p) for p in prompts]


def training_rows(store: ArtifactStore, shared, tokenizer) -> tuple[list[dict], list[str]]:
    """Relabelled pairs with rendered prompts, plus the raw prompts."""
    rows = store.read_jsonl(store.key("relabel", "prefs.jsonl"))
    out = [
        {
            "prompt": render_prompt(shared, tokenizer, r["prompt"]),
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        for r in rows
    ]
    return out, [r["prompt"] for r in rows]


def build_trl_judge(store: ArtifactStore, shared, tokenizer, raw_prompts, device):
    """The one judge, as TRL sees it."""
    from pipeline.judge.trl_judge import make_trl_judge

    return make_trl_judge(
        build_judge(shared, store, device=device), raw_prompt_lookup(shared, tokenizer, raw_prompts)
    )


# Settings that change how a run is carried out but not what it computes stay
# out of the params hash, so adding a resume point never invalidates a result.
NOT_HASHED = ("resume_every",)


def _params(profile: dict, leg: str, shared, upstream: str) -> dict:
    tcfg = leg_train_config(profile, leg)
    return {
        "stage": stage_name(leg),
        "leg": leg,
        "shared_config_hash": shared.hash,
        "train": {k: v for k, v in tcfg.items() if k not in NOT_HASHED},
        "relabel_params_hash": upstream,
    }


def make(leg: str):
    from pipeline.legs import LEGS

    if leg not in LEGS:
        raise ValueError(f"unknown leg {leg!r}, expected one of {tuple(LEGS)}")
    stage = stage_name(leg)

    def run(store: ArtifactStore, profile: dict, force: bool = False) -> dict:
        import torch

        shared = load_shared(profile)
        up = manifest.load(store, "relabel")
        if up is None:
            raise RuntimeError(f"{stage} needs a completed relabel stage")
        params = _params(profile, leg, shared, up["params_hash"])
        phash = manifest.params_hash(params)
        if not force and manifest.is_complete(store, stage, phash):
            return {"skipped": True, "stage": stage}

        tcfg = leg_train_config(profile, leg)
        device = profile["model"]["device"]
        with stage_timer(store, stage, gpu_type=profile.get("gpu", "none")):
            tokenizer = build_tokenizer(shared.base_model)
            policy = build_base(shared, tokenizer, tcfg["seed"]).to(device)
            ref = copy.deepcopy(policy).eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            rows, raw = training_rows(store, shared, tokenizer)
            judge = (
                build_trl_judge(store, shared, tokenizer, raw, device)
                if getattr(LEGS[leg], "USES_JUDGE", True)
                else None
            )

            from pipeline.resume import ResumePoint

            workdir = tempfile.TemporaryDirectory(prefix="ncm-trl-")
            resume = ResumePoint(store, stage, phash)
            resume_from, saved = resume.load(workdir.name)
            metrics: list[dict] = saved.get("metrics", [])
            ladder: list[dict] = saved.get("ladder", [])

            def on_eval(step, est):
                metrics.append({"kind": "kl", "step": step, **est})

            def on_milestone(m, model, step, est):
                key = store.key(stage, "checkpoints", f"kl_{m:g}.pt")
                store.put(key, save_state(model, tcfg.get("checkpoint_dtype")))
                ladder.append({"milestone": m, "step": step, "kl": est["kl"], "key": key})

            from pipeline.kl import make_kl_callback

            callback = make_kl_callback(
                ref=ref,
                tokenizer=tokenizer,
                prompts=kl_prompts(store, shared, tokenizer),
                ladder=shared.kl,
                max_new_tokens=shared.sampling.max_new_tokens,
                device=device,
                seed=tcfg["seed"],
                on_eval=on_eval,
                on_milestone=on_milestone,
                reached=[r["milestone"] for r in ladder],
                measure_at_begin=resume_from is None,
            )
            callbacks = [callback]
            if tcfg.get("resume_every"):
                callbacks.append(resume.callback(lambda: {"metrics": metrics, "ladder": ladder}))
            torch.manual_seed(tcfg["seed"])
            with workdir:
                history = LEGS[leg].train(
                    policy,
                    ref,
                    tokenizer,
                    rows,
                    judge,
                    shared,
                    tcfg,
                    callbacks,
                    workdir.name,
                    resume_from=resume_from,
                )
            metrics = [{"kind": "trl", **h} for h in history] + metrics

            store.put(
                store.key(stage, "metrics.jsonl"),
                "".join(
                    json.dumps(m, sort_keys=True, default=float) + "\n" for m in metrics
                ).encode(),
            )
            store.put_json(store.key(stage, "checkpoints.json"), ladder)
            store.put(
                store.key(stage, CHECKPOINT), save_state(policy, tcfg.get("checkpoint_dtype"))
            )
            resume.clear()
        kls = [m["kl"] for m in metrics if m["kind"] == "kl"]
        manifest.write_manifest(
            store,
            stage,
            phash,
            [CHECKPOINT, "checkpoints.json", "metrics.jsonl"]
            + [r["key"].split(f"{stage}/", 1)[1] for r in ladder],
            extra={
                "leg": leg,
                "trainer": LEGS[leg].TRAINER,
                "shared_config_hash": shared.hash,
                "shared_config": shared.to_dict(),
                "seed": tcfg["seed"],
                "resumed_from_step": int(resume_from.rsplit("-", 1)[1]) if resume_from else None,
                "final_kl": kls[-1] if kls else None,
                "milestones_reached": [r["milestone"] for r in ladder],
            },
        )
        return {
            "skipped": False,
            "stage": stage,
            "leg": leg,
            "final_kl": kls[-1] if kls else None,
            "milestones_reached": [r["milestone"] for r in ladder],
        }

    run.__name__ = f"run_{stage}"
    return run
