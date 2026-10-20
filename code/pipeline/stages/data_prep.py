"""Stage: pairwise preference data + eval prompts.

`profile["data_prep"]["source"]` selects the dataset:

- `synthetic` (smoke default): deterministic (prompt, chosen, rejected)
  records where the chosen response verbalizes its reasoning and the rejected
  one answers bare: a toy CoT-vs-no-CoT preference axis, offline, vocabulary
  inside models.TINY_VOCAB_WORDS so the smoke tokenizer sees no <unk>.
- `ultrafeedback` / `hh_rlhf`: real preference corpora via
  pipeline.prefdata.loaders (network, so canary and real profiles only).
"""

import json
import random

from pipeline import manifest
from pipeline.costs import stage_timer
from pipeline.exec.base import ArtifactStore

STAGE = "data_prep"

_TEMPLATES = [
    (
        "is the answer yes or no .",
        "think step by step . first the question then the answer . the answer is {a} .",
        "the answer is {b} .",
    ),
]
_ANSWERS = ("yes", "no")


def sample_pairs(n: int, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}-{STAGE}")
    rows = []
    for _ in range(n):
        prompt, chosen_t, rejected_t = rng.choice(_TEMPLATES)
        a = rng.choice(_ANSWERS)
        b = a if rng.random() < 0.5 else next(x for x in _ANSWERS if x != a)
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen_t.format(a=a),
                "rejected": rejected_t.format(b=b),
            }
        )
    return rows


def _put_jsonl(store: ArtifactStore, key: str, rows: list[dict]) -> None:
    # put(), not append_jsonl(): a --force re-run must overwrite, not
    # double the dataset.
    store.put(key, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())


def _run_synthetic(store: ArtifactStore, p: dict) -> tuple[list[str], int]:
    pairs = sample_pairs(p["n_pairs"], p["seed"])
    _put_jsonl(store, store.key(STAGE, "prefs.jsonl"), pairs)
    outputs = ["prefs.jsonl", "eval_prompts.jsonl"]
    if p.get("n_test_pairs"):
        # stands in for the held-out split: same generator, another seed
        _put_jsonl(
            store,
            store.key(STAGE, "test_prefs.jsonl"),
            sample_pairs(p["n_test_pairs"], p["seed"] + 1000),
        )
        outputs.append("test_prefs.jsonl")
    rng = random.Random(f"{p['seed']}-eval")
    prompts = [{"prompt": rng.choice(_TEMPLATES)[0]} for _ in range(p["n_eval_prompts"])]
    _put_jsonl(store, store.key(STAGE, "eval_prompts.jsonl"), prompts)
    return outputs, len(pairs)


def _run_loader(store: ArtifactStore, p: dict) -> tuple[list[str], int]:
    """Real corpora (network): n_pairs training records + n_eval_prompts
    held-out prompts, both deterministic in (source, sizes, seed), plus
    n_test_pairs from the dataset's test split for the seen-versus-unseen
    judge comparison (never trained on)."""
    from pipeline.prefdata import TEST_SPLITS, load_records

    kw = {"max_chars": p["max_chars"]} if p.get("max_chars") else {}
    records = load_records(p["source"], p["n_pairs"] + p["n_eval_prompts"], p["seed"], **kw)
    pairs, held_out = records[p["n_eval_prompts"] :], records[: p["n_eval_prompts"]]
    _put_jsonl(store, store.key(STAGE, "prefs.jsonl"), pairs)
    _put_jsonl(
        store, store.key(STAGE, "eval_prompts.jsonl"), [{"prompt": r["prompt"]} for r in held_out]
    )
    outputs = ["prefs.jsonl", "eval_prompts.jsonl"]
    if p.get("n_test_pairs"):
        test = load_records(
            p["source"], p["n_test_pairs"], p["seed"], split=TEST_SPLITS[p["source"]], **kw
        )
        _put_jsonl(store, store.key(STAGE, "test_prefs.jsonl"), test)
        outputs.append("test_prefs.jsonl")
    return outputs, len(pairs)


def run(store: ArtifactStore, profile: dict, force: bool = False) -> dict:
    p = profile["data_prep"]
    source = p.get("source", "synthetic")
    params = {
        "stage": STAGE,
        "source": source,
        "n_pairs": p["n_pairs"],
        "n_eval_prompts": p["n_eval_prompts"],
        "seed": p["seed"],
    }
    for opt in ("n_test_pairs", "max_chars"):  # in the hash only when set
        if p.get(opt):
            params[opt] = p[opt]
    phash = manifest.params_hash(params)
    if not force and manifest.is_complete(store, STAGE, phash):
        return {"skipped": True, "stage": STAGE}

    with stage_timer(store, STAGE, gpu_type=profile.get("gpu", "none")):
        if source == "synthetic":
            outputs, n = _run_synthetic(store, p)
        else:
            outputs, n = _run_loader(store, p)
    manifest.write_manifest(
        store, STAGE, phash, outputs, extra={"seed": p["seed"], "source": source}
    )
    return {"skipped": False, "stage": STAGE, "n_pairs": n}
