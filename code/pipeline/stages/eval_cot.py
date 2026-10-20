"""Stage stub: CoT monitorability evaluation placeholder.

Generates from one leg's trained checkpoint on the held-out eval prompts and scores
two toy surface metrics: parse_rate (generation names an answer word) and
cot_rate (generation verbalizes reasoning words). Raw samples are kept,
because metrics without the samples that produced them cannot be debugged.
The real evaluators (Lanham perturbations, steganography probes, judge
scoring) are not written yet and will sit behind this same stage boundary.

The stage runs for every leg and judges nothing against a threshold
(experiment 0115).
"""

import json

from pipeline import manifest
from pipeline.costs import stage_timer
from pipeline.exec.base import ArtifactStore
from pipeline.generate import sample_models
from pipeline.models import build_causal_lm, build_tokenizer, load_state
from pipeline.shared import Sampling, load_shared, render_prompt
from pipeline.stages.train import CHECKPOINT, stage_name

STAGE = "eval_cot"  # per leg: eval_cot_<leg>

ANSWER_WORDS = ("yes", "no")
COT_WORDS = ("think", "step", "therefore", "first", "then")


def score_text(text: str) -> dict:
    words = text.split()
    return {
        "parsed": any(w in words for w in ANSWER_WORDS),
        "cot": any(w in words for w in COT_WORDS),
    }


def make(leg: str):
    """The stage for one leg's checkpoint (experiment 0115): reads
    `train_<leg>/checkpoint.pt`, writes under `eval_cot_<leg>/`."""

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

    p = profile["eval_cot"]
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
        # one batched call; backend "vllm" samples with vLLM (experiment 0122)
        tokenizer.padding_side = "left"
        models = [model]
        del model
        texts = sample_models(
            models,
            tokenizer,
            [render_prompt(shared, tokenizer, q) for q in prompts],
            Sampling(max_new_tokens=p["max_new_tokens"], temperature=1.0, top_p=1.0),
            p["seed"],
            device,
            backend=p.get("backend", "hf"),
        )[0]
        samples = [{"prompt": q, "text": t} for q, t in zip(prompts, texts)]
        scores = [score_text(t) for t in texts]

        store.put(
            store.key(name, "samples.jsonl"),
            "".join(json.dumps(s, sort_keys=True) + "\n" for s in samples).encode(),
        )
        metrics = {
            "n": len(scores),
            "parse_rate": sum(s["parsed"] for s in scores) / len(scores),
            "cot_rate": sum(s["cot"] for s in scores) / len(scores),
        }
        store.put_json(store.key(name, "metrics.json"), metrics)
    manifest.write_manifest(
        store, name, phash, ["samples.jsonl", "metrics.json"], extra={"seed": p["seed"], "leg": leg}
    )
    return {"skipped": False, "stage": name, "leg": leg, **metrics}
