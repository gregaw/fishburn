# vLLM sampling, and how it changes the choice of algorithm

This repo trains a language model on preferences with several methods, each run
as a **leg** of the pipeline. Two legs are **online**: they sample fresh answers
from the model at every training step. One is Online DPO. The other is NashMP
(Nash Mirror Prox), a method for Nash learning from human feedback (NLHF). A
third leg, plain DPO, trains on fixed data and does not sample. After training,
the `sanity` stage checks each leg: it samples answers from both the trained and
the untrained model to see what the leg learned.

All of that sampling uses vLLM, a fast inference engine (the setting
`sampler: vllm` in a run's config). The alternative, `sampler: hf`, uses the
transformers library's `generate`; the CPU tests, which run without a GPU or
network, use that one. This note
gives the benchmark behind the choice, and explains why vLLM decided which NLHF
method the pipeline uses. The numbers come from one GPU type (an L4), two small
models and one timing per setting.

## In short

- Sampling 256 answers through vLLM was **3.8× faster end to end** than the best
  transformers setting for Qwen3-0.6B, and **2.8×** for Qwen3-1.7B. Counting
  only the generation call on a running engine, it was **29× and 17×** faster
  (the table below).
- Most of vLLM's end-to-end time is **saving the model to disk and starting the
  engine**. During training that cost is paid once; after that, each step only
  copies the new weights into the engine in GPU memory.
- Some preference-learning methods **cannot use an inference engine at all**.
  Nash-MD (Nash Mirror Descent), another NLHF method, samples from a
  token-by-token mixture of two models, which vLLM cannot serve. NashMP samples
  only from ordinary models, so it can.

## The benchmark

The job: one answer per prompt for 256 UltraFeedback prompts that no leg trains
on, at temperature 0.9 and at most 192 new tokens, on one L4, with the same
prompts and seed throughout. This is what the `sanity` stage does after every
training run, and it pays the whole end-to-end cost each time: it saves the
trained model, starts an engine and generates. The benchmark's record does not
say whether the transformers times include loading the model. If they do,
transformers generation alone is faster than shown, and the gap is smaller.

The speed-up column is relative to transformers in bf16 at batch 32, the fastest
transformers setting and the fair baseline, since vLLM also runs in bf16. Higher
is faster; below 1.0× is slower than that baseline.

| path | Qwen3-0.6B: seconds | speed-up | Qwen3-1.7B: seconds | speed-up |
| --- | --- | --- | --- | --- |
| transformers, fp32, batch 8 (the `sanity` stage before vLLM) | 352.6 | 0.5× | 474.7 | 0.4× |
| transformers, bf16, batch 8 | 310.3 | 0.6× | 395.5 | 0.5× |
| transformers, bf16, batch 32 | 175.0 | 1.0× | 189.5 | 1.0× |
| **vLLM, end to end** (save model, start engine, generate) | **46.0** | **3.8×** | **67.7** | **2.8×** |
| vLLM, generation call only (all 256 prompts in one call) | 6.1 | 29× | 11.2 | 17× |

Every setting produced about the same number of tokens (about 24k for 0.6B and
33k for 1.7B), so the times compare like with like. Each setting was timed once.
Against what the `sanity` stage did before (fp32, batch 8), vLLM end to end is
7.7× faster for 0.6B and 7.0× for 1.7B.

## Two settings in the code that come from this

- **`max_model_len` is capped at 4,096 tokens** (`MAX_MODEL_LEN` in
  `pipeline/generate.py`). vLLM shares the GPU with training and gets only part
  of its memory. Qwen3-1.7B advertises a 40,960-token context, and in that share
  the engine's cache for it did not fit, so the engine refused to start. 4,096
  is far above anything the pipeline generates.
- **One container image, pinned for vLLM.** TRL 0.29.1 accepts vLLM up to
  version 0.12.0, and vLLM 0.12.0 needs torch 2.9.0 and numpy 2.2.6. So
  `configs/requirements.txt` pins those for the one cloud image, rather than
  keeping a separate vLLM image whose versions would have to be kept matching.

## Why this decides the NLHF method

NLHF trains the model (the **policy**) by playing it against an opponent: both
answer the same prompt, and a pairwise judge, a model that reads two answers,
says which is better. NLHF methods differ in **who the opponent is**, and that
decides where the samples can come from.

- **Nash-MD** plays against a *geometric mixture* of the current policy π and
  the reference model π_ref (the model before training):
  π^(1−ηβ)·π_ref^(ηβ), normalised, where η is the step size and β the strength of the pull
  towards the reference. Sampling that mixture needs both models' next-token
  probabilities at every token, so it must run token by token in plain
  PyTorch, and no inference engine can speed it up.
- **NashMP**, the leg in this pipeline, plays against *an earlier version of the
  policy itself*. Training runs in rounds of a few steps (4 in the `real`
  profile), and the opponent for a round is the policy as it was at the
  round's start. That opponent is an ordinary model, and the round's prompts are
  known in advance, so all its answers for the round are drawn at the start,
  from the same vLLM engine.

## What to take away

1. **Benchmark end to end, not just the generation call.** The 29× figure
   becomes 3.8× once you pay for saving the model and starting the engine.
   Which one matters depends on how many times you sample per engine start.
2. **Online methods should gain the most.** A training loop that keeps one
   engine running, and copies the new weights into it each step, avoids the
   start-up cost after the first step. That is our expectation from the
   benchmark; we did not time it separately.
3. **Check whether a method can sample from an ordinary model.** If it needs
   several models' probabilities at every token, as a mixture does, it will
   not run on vLLM, and that can make it several times
   more expensive in practice.
