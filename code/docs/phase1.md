# Phase 1: the pipeline

Phase 1 built a pipeline that compares RLHF-style and NLHF training on real text
end to end, and checked that it trains. It does not answer the scientific
question yet. This note describes what the code in this folder does and the one
run of it as published.

## What the code does

- One config-driven pipeline: prepare a preference dataset, optionally relabel
  it with a judge, train several legs from the same base model, run the same
  checks on each, and report them side by side.
- Three legs: **NashMP** (NLHF, our implementation of Nash Mirror Prox),
  **Online DPO** (the legacy method it is compared against) and plain **DPO**
  (a cheap check that a setup trains).
- One pairwise judge, **PairRM**, used for relabelling, as the referee during
  online training, and in the checks.
- vLLM sampling in both online legs and in the checks (see [`vllm.md`](vllm.md)).
- Two checks with bootstrap intervals: the judged check (P(trained ≻ base) on
  held-out prompts) and the judge-free preference check on held-out pairs.
- Plumbing: content-hashed stage caching, a cost record per stage, resume points
  for long runs, a fresh container per GPU stage, and an offline CPU test that
  runs every stage and leg in seconds before any money is spent.

## Why the code looks like this

- **NashMP is the NLHF leg.** Its opponents are ordinary policies, so it can
  sample with vLLM.
- **Online DPO is the comparison leg.** It samples its own answers and uses the
  same judge as NashMP, so the training rule is the only difference. Comparing
  offline DPO with NashMP would mix that with offline versus online.
- **The legacy side is DPO, with no reward model and PPO.** DPO's Bradley-Terry
  loss aims at the same Borda-like target as classic RLHF, without training a
  separate reward model or running PPO. Plain offline DPO stays as the quick
  check that a setup trains.
- **PairRM is the one judge.** A scalar judge is Borda-like by construction:
  against it NLHF and RLHF aim at the same target, and the comparison
  disappears.

## The pipeline as published, run once

The `real` profile (Online DPO and NashMP on vLLM, DPO as the check, PairRM)
ran end to end on Qwen3-0.6B and UltraFeedback, one seed. Every stage of every
leg completed. Total cost was $1.51 on one L4, after a $0.59 canary.

| leg | preference check (dataset pairs) | judged check P(trained ≻ base) | KL from base | train time |
| --- | --- | --- | --- | --- |
| DPO, 256 steps | **0.659 [0.628, 0.690]** | 0.520 [0.471, 0.569] | 0.91 | 27 min |
| Online DPO, 96 steps | 0.509 [0.477, 0.542] | 0.548 [0.499, 0.596] | 0.89 | 22 min |
| NashMP, 96 steps | 0.499 [0.466, 0.531] | 0.536 [0.491, 0.580] | 0.95 | 27 min |

DPO learns the dataset clearly. The two online legs reach the same KL and lean
the same way on the judged check, and neither clears 0.5 by its interval, so at
this size the comparison shows no difference yet. Both online legs train at
about 14 to 17 s per step with vLLM, including the judge.

## Known limits

- **The chain-of-thought evaluators are placeholders.** `eval_cot` and
  `eval_steering` save samples and toy metrics only.
- **PairRM is a weak judge for long answers** (about 400 tokens of context) and
  has seen UltraFeedback's and HH-RLHF's training splits.
- **KL milestones need care at real scale.** The first KL estimate (step 32 in
  the `real` profile) can already be past several milestones, which makes those
  milestone checkpoints the same model. Estimate KL more often early in a run if
  you compare legs at equal drift.
- **NashMP has no resume points.** Keep its runs under about an hour until it does.
- **Single seeds.** The result above is one seed.
