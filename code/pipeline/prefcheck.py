"""Preference check: did training move the model towards the labelled
preferences of held-out pairs? (experiment 0123).

For each held-out pair (prompt, chosen, rejected), with the dataset's own
label, the check compares how training changed the log-probability of each
answer:

    delta = [log p_trained(chosen) - log p_base(chosen)]
          - [log p_trained(rejected) - log p_base(rejected)]

delta > 0 means training moved the model towards the chosen answer relative
to the rejected one. The base model has delta = 0 on every pair, so the
chance level is exactly 0.5 and the check passes when the fraction of pairs
with delta > 0 has a bootstrap 95 % interval above 0.5. This is DPO's
implicit-reward accuracy, measured on pairs no leg trained on, and it needs no
generation and no judge.

The gap is used, not log p(chosen) alone: DPO is known to lower both answers'
probabilities, the rejected one faster, so the chosen answer's probability can
fall while training works. The mean change of each is reported beside it.

The same deltas are also scored against the judge's verdict on each pair
(relabel's `p_dataset_chosen`), which is what an online leg learns from, and
the judge's own agreement with the dataset label is the reference line: a
leg that learns only from the judge cannot be expected to beat it.
"""


def response_logprob(
    model, tokenizer, prompt: str, response: str, device: str, max_length: int
) -> float:
    """Summed log-probability of `response` (plus EOS) after the rendered
    `prompt`. Over max_length the response is cut at its end, after the prompt
    is cut to at most half the budget from its start. Only the response
    positions' logits are kept (Qwen's 152k vocabulary, see pipeline/kl.py)."""
    import torch

    p = tokenizer(prompt, add_special_tokens=False).input_ids
    r = tokenizer(response, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        r = r + [tokenizer.eos_token_id]
    if len(p) + len(r) > max_length:
        p = p[-(max_length // 2) :] if len(p) > max_length // 2 else p
        r = r[: max_length - len(p)]
    if not r:
        return 0.0
    ids = torch.tensor([p + r], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, logits_to_keep=len(r) + 1).logits[0, :-1]
    lp = torch.log_softmax(logits.float(), dim=-1)
    target = torch.tensor(r, device=device).unsqueeze(-1)
    return float(lp.gather(-1, target).sum())


def pair_logprobs(
    model, tokenizer, rendered: list[str], rows: list[dict], device: str, max_length: int
) -> tuple[list[float], list[float]]:
    """(log p(chosen), log p(rejected)) per row, under the dataset label."""
    chosen = [
        response_logprob(model, tokenizer, q, r["chosen"], device, max_length)
        for q, r in zip(rendered, rows, strict=True)
    ]
    rejected = [
        response_logprob(model, tokenizer, q, r["rejected"], device, max_length)
        for q, r in zip(rendered, rows, strict=True)
    ]
    return chosen, rejected


def dataset_label_rows(relabelled: list[dict]) -> list[dict]:
    """Relabel's rows back under the dataset's own label, with the judge's
    P(dataset chosen > dataset rejected)."""
    out = []
    for r in relabelled:
        swapped = r.get("swapped", False)
        out.append(
            {
                "prompt": r["prompt"],
                "chosen": r["rejected"] if swapped else r["chosen"],
                "rejected": r["chosen"] if swapped else r["rejected"],
                "p_judge": r.get("p_dataset_chosen"),
            }
        )
    return out


def _score(x: float) -> float:
    return 1.0 if x > 0 else 0.5 if x == 0 else 0.0


def summarize(
    rows: list[dict],
    base: tuple[list[float], list[float]],
    trained: tuple[list[float], list[float]],
    n_boot: int,
    seed: int,
) -> dict:
    """The check's report. `rows` carry `p_judge` (None when unjudged)."""
    from pipeline.stages.sanity import bootstrap_ci

    (bc, br), (tc, tr) = base, trained
    d_chosen = [t - b for t, b in zip(tc, bc, strict=True)]
    d_rejected = [t - b for t, b in zip(tr, br, strict=True)]
    delta = [c - r for c, r in zip(d_chosen, d_rejected)]
    hits = [_score(d) for d in delta]
    lo, hi = bootstrap_ci(hits, n_boot, seed)
    n = len(delta)
    out = {
        "n_pairs": n,
        "accuracy_vs_dataset": sum(hits) / n,
        "ci95": [lo, hi],
        "above_half_beyond_noise": lo > 0.5,
        "mean_delta": sum(delta) / n,
        "mean_change_logp_chosen": sum(d_chosen) / n,
        "mean_change_logp_rejected": sum(d_rejected) / n,
    }
    judged = [
        (d, r["p_judge"])
        for d, r in zip(delta, rows)
        if r.get("p_judge") is not None and r["p_judge"] != 0.5
    ]
    if judged:
        agree = [_score(d if p > 0.5 else -d) for d, p in judged]
        jlo, jhi = bootstrap_ci(agree, n_boot, seed)
        out["accuracy_vs_judge"] = sum(agree) / len(agree)
        out["accuracy_vs_judge_ci95"] = [jlo, jhi]
        out["judge_agrees_with_dataset"] = sum(p > 0.5 for _, p in judged) / len(judged)
        out["n_judged"] = len(judged)
    return out


def summarize_onpolicy(
    rows: list[dict],
    base: tuple[list[float], list[float]],
    trained: tuple[list[float], list[float]],
    n_boot: int,
    seed: int,
) -> dict:
    """The on-policy check: `rows` are pairs of two base answers with the
    judge's P(first > second) in `p_judge`. Reports how often training moved
    towards the judge's pick, over the pairs the judge does not tie."""
    from pipeline.stages.sanity import bootstrap_ci

    (bc, br), (tc, tr) = base, trained
    hits = []
    for r, b1, b2, t1, t2 in zip(rows, bc, br, tc, tr, strict=True):
        p = r["p_judge"]
        if p == 0.5:
            continue
        delta = (t1 - b1) - (t2 - b2)
        hits.append(_score(delta if p > 0.5 else -delta))
    n = len(hits)
    if not n:
        return {"n_pairs": len(rows), "n_judged": 0, "above_half_beyond_noise": None}
    lo, hi = bootstrap_ci(hits, n_boot, seed)
    return {
        "n_pairs": len(rows),
        "n_judged": n,
        "accuracy_vs_judge": sum(hits) / n,
        "ci95": [lo, hi],
        "above_half_beyond_noise": lo > 0.5,
    }
