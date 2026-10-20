"""Stage stub: CoT controllability evaluation placeholder.

Toy prompt-level steering: generate with and without a "think step by step"
instruction prefix and measure the delta in verbalized reasoning
(steering_effect = steered cot_rate - unsteered cot_rate). Real activation
steering (CAA via the steering-vectors library) is not written yet and will
sit behind this same stage boundary; the artifact shape (per-condition samples
plus a metrics.json with an effect size) is the part this stub pins down.

The stage runs for every leg's checkpoint and judges nothing against a
threshold (experiment 0115).
"""

import json

from pipeline import manifest
from pipeline.costs import stage_timer
from pipeline.exec.base import ArtifactStore
from pipeline.generate import sample_models
from pipeline.models import build_causal_lm, build_tokenizer, load_state
from pipeline.shared import Sampling, load_shared, render_prompt
from pipeline.stages.eval_cot import score_text
from pipeline.stages.train import CHECKPOINT, stage_name

STAGE = "eval_steering"  # per leg: eval_steering_<leg>

STEER_PREFIX = "think step by step ."


def make(leg: str):
    """The stage for one leg's checkpoint (experiment 0115): reads
    `train_<leg>/checkpoint.pt`, writes under `eval_steering_<leg>/`."""

    def run(store: ArtifactStore, profile: dict, force: bool = False) -> dict:
        return _run(store, profile, force, leg)

    run.__name__ = f"run_{STAGE}_{leg}"
    return run


def _run(store: ArtifactStore, profile: dict, force: bool, leg: str) -> dict:
    import torch

    name, upstream = f"{STAGE}_{leg}", stage_name(leg)
    up = manifest.load(store, upstream)
    if up is None:
        raise RuntimeError(f"{name} needs a completed {upstream} stage")
    shared = load_shared(profile)

    p = profile["eval_steering"]
    params = {
        "stage": name,
        "model": profile["model"]["name"],
        "n_samples": p["n_samples"],
        "max_new_tokens": p["max_new_tokens"],
        "seed": p["seed"],
        "leg": leg,
        "upstream": up["params_hash"],
    }
    if p.get("backend", "hf") != "hf":  # older runs keep their hashes
        params["backend"] = p["backend"]
    phash = manifest.params_hash(params)
    if not force and manifest.is_complete(store, name, phash):
        return {"skipped": True, "stage": name}

    with stage_timer(store, name, gpu_type=profile.get("gpu", "none")):
        torch.manual_seed(p["seed"])
        tokenizer = build_tokenizer(profile["model"]["name"])
        model = build_causal_lm(profile["model"]["name"], profile["model"]["dtype"], tokenizer)
        load_state(model, store.get(store.key(upstream, CHECKPOINT)))
        device = profile["model"]["device"]
        model.to(device).eval()

        prompts = [
            r["prompt"] for r in store.read_jsonl(store.key("data_prep", "eval_prompts.jsonl"))
        ]
        prompts = (prompts * (p["n_samples"] // len(prompts) + 1))[: p["n_samples"]]

        # both conditions in one batched call; backend "vllm" samples with
        # vLLM (experiment 0122)
        conditions = (("unsteered", ""), ("steered", STEER_PREFIX + " "))
        jobs = [(c, prefix, q) for c, prefix in conditions for q in prompts]
        tokenizer.padding_side = "left"
        models = [model]
        del model
        texts = sample_models(
            models,
            tokenizer,
            [render_prompt(shared, tokenizer, prefix + q) for _, prefix, q in jobs],
            Sampling(max_new_tokens=p["max_new_tokens"], temperature=1.0, top_p=1.0),
            p["seed"],
            device,
            backend=p.get("backend", "hf"),
        )[0]
        samples = [{"condition": c, "prompt": q, "text": t} for (c, _, q), t in zip(jobs, texts)]
        rates = {}
        for c, _ in conditions:
            scores = [score_text(s["text"]) for s in samples if s["condition"] == c]
            rates[c] = sum(x["cot"] for x in scores) / len(scores)

        store.put(
            store.key(name, "samples.jsonl"),
            "".join(json.dumps(s, sort_keys=True) + "\n" for s in samples).encode(),
        )
        metrics = {
            "n_per_condition": len(prompts),
            "cot_rate_unsteered": rates["unsteered"],
            "cot_rate_steered": rates["steered"],
            "steering_effect": rates["steered"] - rates["unsteered"],
        }
        store.put_json(store.key(name, "metrics.json"), metrics)
    manifest.write_manifest(
        store, name, phash, ["samples.jsonl", "metrics.json"], extra={"seed": p["seed"], "leg": leg}
    )
    return {"skipped": False, "stage": name, "leg": leg, **metrics}
