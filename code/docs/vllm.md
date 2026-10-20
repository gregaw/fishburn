# vLLM sampling, and how it changes the choice of algorithm

Both online legs and the sanity checks sample with vLLM (`sampler: vllm`); the
`hf` sampler uses transformers `generate` and is what the offline CPU test runs.
This note gives the benchmark behind that choice and why it decided which NLHF
method the pipeline uses. The numbers come from one GPU type (an L4) and two
small models.

## In short

- Sampling 256 answers through vLLM was **7.7× faster end to end** than the
  transformers path for Qwen3-0.6B, and **7.0×** for Qwen3-1.7B. Counting only
  the generation call on a running engine, it was **42 to 58×** faster.
- Most of vLLM's end-to-end time is **starting the engine**. The more you sample
  from one engine, the closer you get to the generation-only figure.
- Some preference-learning algorithms **cannot use an inference server at all**.
  Nash-MD samples from a token-by-token mixture of two models, which vLLM cannot
  serve. NashMP samples only from ordinary policies, so it can.

## The benchmark

The job: one answer per prompt for 256 held-out UltraFeedback prompts, at
temperature 0.9 and at most 192 new tokens, on one L4, with the same prompts and
seed throughout. This is what the sanity stage does after every training run.

| path | Qwen3-0.6B: seconds | speed-up | Qwen3-1.7B: seconds | speed-up |
| --- | --- | --- | --- | --- |
| transformers, fp32, batch 8 | 352.6 | 1.0× | 474.7 | 1.0× |
| transformers, bf16, batch 8 | 310.3 | 1.1× | 395.5 | 1.2× |
| transformers, bf16, batch 32 | 175.0 | 2.0× | 189.5 | 2.5× |
| **vLLM, end to end** (write model, start engine, generate) | **46.0** | **7.7×** | **67.7** | **7.0×** |
| vLLM, generation call only | 6.1 | 58× | 11.2 | 42× |

All runs produced about the same number of tokens (24k and 33k), so the
comparison is fair. vLLM runs in bf16, which is why the bf16 rows are the honest
transformers baseline; even against the best of them, vLLM is about 3× faster
end to end.

## Two settings in the code that come from this

- **`max_model_len` is capped at 4,096** (`MAX_MODEL_LEN` in
  `pipeline/generate.py`). Qwen3-1.7B advertises a 40,960-token context; at the
  memory share vLLM gets next to training, its KV cache did not fit and the
  engine refused to start. 4,096 is far above anything the pipeline generates.
- **One image, pinned for vLLM.** TRL 0.29.1 accepts vLLM up to 0.12.0, which
  needs torch 2.9.0 and numpy 2.2.6, so `configs/requirements.txt` pins those
  rather than keeping a separate vLLM image that would have to stay in step.

## Why this decides the NLHF method

Nash learning from human feedback (NLHF) trains a policy against copies of
itself, scored by a pairwise judge. The solvers differ in **who the opponent
is**, and that decides where the samples can come from.

- **Nash-MD** plays against a *geometric mixture* of the current policy and the
  reference model, π^(1−ηβ)·π_ref^(ηβ). Sampling that mixture needs both
  models' logits at every token, so it must run token by token in plain
  PyTorch, and no inference server can speed it up.
- **NashMP** (Nash Mirror Prox), the leg in this pipeline, plays against
  *earlier versions of the policy itself*. Every opponent is an ordinary
  policy, so one colocated vLLM engine serves all of its sampling.

NashMP also has the better guarantee: its last iterate converges, with no
averaging of checkpoints.

In the `real` profile on Qwen3-0.6B and one L4, with PairRM judging, both online
legs sample through vLLM: 96 steps took 22 minutes for Online DPO (about 14 s per
step, engine start included) and 27 minutes for NashMP. Both pay roughly the same
for sampling, so their comparison is not skewed by speed.

## What to take away

1. **Benchmark end to end, not just the generation call.** The 58× figure
   becomes 7.7× once you pay for starting the engine. Which one matters depends
   on how many times you sample per engine start.
2. **Online algorithms are where vLLM pays off most.** A training loop that
   samples every step from one long-lived engine gets close to the
   generation-only figure.
3. **Check whether an algorithm's sampler is "server-shaped".** If it needs
   logits from two models per token, a mixture, or a custom sampling rule, it
   will not run on vLLM, and that can make it an order of magnitude more
   expensive in practice.
