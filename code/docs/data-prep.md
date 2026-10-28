# Data preparation: what `data_prep` produces, and why

The pipeline runs as a series of **stages**, each a step that reads the outputs
of the ones before it. A **profile**, a config file in `configs/`, sets every
stage's options; a **run** pushes one profile through all the stages. Outputs go
to the run's artifact store: a local folder, or a storage volume on Modal, the
cloud GPU service the pipeline runs on.

`data_prep` is the first stage. It turns raw preference datasets into **one
record shape**, deterministically. Everything after it consumes that shape,
whatever the data came from: training, the judge (PairRM, a small model that
reads a prompt and two answers and says which is better) and the checks defined
in the README's [What we measure](../README.md#what-we-measure).

## The unit: a pairwise preference record

A training example is *"for this prompt, response A was preferred over
response B"*. Training never uses a scalar score as its signal, even where a
dataset has scores. That is the point of Nash learning from human feedback
(NLHF): it learns from preferences between pairs, not from a reward per
response. The judge in `pipeline/judge/` gives the same kind of signal, the
probability `P(y₁ ≻ y₂ | x)` that response y₁ is better than y₂ for prompt x.

```json
{"prompt": "How do I boil an egg?",
 "chosen": "Put the egg in boiling water for 7 minutes, then cool it in cold water.",
 "rejected": "Eggs.",
 "source": "ultrafeedback", "score_chosen": 8.0, "score_rejected": 3.0, "tie": false}
```

`prompt`, `chosen` and `rejected` are in every record. The other fields record
where the pair came from and the dataset's own scores (`tie` is true when the
two scores are equal); which ones appear depends on the source (see below). Rows where
`chosen == rejected`, or where any of the three fields is empty, are dropped.

The stage writes:

| Artifact | Contents |
| --- | --- |
| `data_prep/prefs.jsonl` | `n_pairs` training records as above, one per line |
| `data_prep/eval_prompts.jsonl` | `n_eval_prompts` prompts, `{"prompt": ...}`, drawn from rows not used for training |
| `data_prep/test_prefs.jsonl` | only with `n_test_pairs` set, and only for `ultrafeedback` and `hh_rlhf`: that many pairs from the dataset's own test split, never trained on. The dataset check (does training raise log p(chosen) − log p(rejected) above the untrained model's?) reads them. |
| `data_prep/manifest.json` | a hash of the stage's settings and its output list, so an unchanged profile skips the stage on a re-run |

`n_pairs`, `n_eval_prompts` and `n_test_pairs` are set in the profile under
`data_prep`. The profiles are `smoke` (offline test), `canary` (a short GPU run
that checks the setup) and `real` (the full run).

## How a sample is drawn

`data_prep.source` in the profile selects the dataset: `synthetic`,
`ultrafeedback` or `hh_rlhf`. For the two real datasets, each source has a
**loader** with two parts: a fetch that reads raw rows from Hugging Face, and a
**normalizer** that turns one raw row into one record, or rejects it. The
normalizers are plain functions, unit-tested offline on fixture rows; only the
fetch needs the network.

To draw `n` records, a loader:

1. **streams** `oversample × n` raw rows (4× by default), so a 1,000-pair sample
   does not download the whole dataset;
2. normalizes them, dropping rejected rows, ties, and (with `max_chars` set)
   pairs whose prompt plus longer answer is over that many characters;
3. **subsamples** `n` of what is left, with a seed.

The same source, split, `n`, seed, `oversample` and `max_chars` always give the
same records. `max_chars` keeps training sequences inside `train.max_length`,
which is counted in tokens, so the fit is approximate.

For training, the stage draws `n_pairs + n_eval_prompts` records and takes the
first `n_eval_prompts` of them as the eval prompts, so they pass the same
filters. The eval prompts and the training pairs are always different rows, but
prompt text is not deduplicated, so the same prompt could in principle appear in
both. Because `n` includes `n_pairs`, changing `n_pairs` changes which rows are
drawn, eval prompts included.

## The three sources

**`synthetic`** (the `smoke` profile, used by the offline end-to-end test). A
generator, with no network, whose vocabulary is limited to the words of the
tiny tokenizer the offline test builds. The preference it encodes is a toy one:
the chosen answer spells out its reasoning (a chain of thought), and the
rejected one answers bare. Each answer's "yes" or "no" is drawn independently,
so the preference is about showing reasoning, not about being right.

```json
{"prompt": "is the answer yes or no .",
 "chosen": "think step by step . first the question then the answer . the answer is yes .",
 "rejected": "the answer is no ."}
```

Its only job is to give the pipeline something shaped like real data, so the
wiring is exercised end to end in seconds on CPU.

**`ultrafeedback`** (`HuggingFaceH4/ultrafeedback_binarized`; the `canary` and
`real` profiles). UltraFeedback has four answers per prompt, each scored by
GPT-4. This binarized version keeps two of them: the
highest-scored as chosen, and one of the others, picked at random, as rejected.
Raw rows carry the prompt, the two answers as chat transcripts ending in an
assistant turn, and their scores:

```json
{"prompt": "How do I boil an egg?",
 "chosen":   [{"role": "user", "content": "How do I boil an egg?"},
              {"role": "assistant", "content": "Put the egg in boiling water for 7 minutes ..."}],
 "rejected": [{"role": "user", "content": "How do I boil an egg?"},
              {"role": "assistant", "content": "Eggs."}],
 "score_chosen": 8.0, "score_rejected": 3.0}
```

Normalization takes the **last assistant message** of each transcript as the
response, keeps the scores, and sets `tie` when they are equal. Ties carry no
preference, so the loader leaves them out of every sample.

**How often the judge agrees with these labels.** `scripts/judge_agreement.py`
asks PairRM about dataset pairs and counts how often it prefers the labelled
chosen answer; random guessing would agree half the time. On 1,000 pairs from
the binarized test split, drawn by the same loader (so without ties), PairRM
agreed on 0.832 of them. The pipeline uses it as the judge with that agreement
in mind; there is no fixed bar it had to clear.
On the pairs whose score margin (`score_chosen − score_rejected`) was at least
1, 2 and 4, it agreed on 0.858, 0.900 and 0.925.

**`hh_rlhf`** (`Anthropic/hh-rlhf`, human-labelled; no shipped profile uses it
yet). Raw rows are two flat transcripts in `\n\nHuman: ... \n\nAssistant: ...`
form, which normally share everything except the final assistant reply:

```
chosen:   "\n\nHuman: How do I boil an egg?\n\nAssistant: Put the egg in boiling water ..."
rejected: "\n\nHuman: How do I boil an egg?\n\nAssistant: Eggs."
```

Normalization splits each at the **final** `Assistant:` marker. Everything up
to and including the marker is the prompt, so the conversation markers and any
earlier turns stay in it; what follows is the response. If the two prompts
differ, the pair compares nothing and the row is dropped. HH-RLHF has no
scores, so its records have no score or `tie` fields.

## Where the network boundary is

The `smoke` profile is fully offline, so the end-to-end test never touches a
real dataset. The real sources import the `datasets` library only inside the
fetch, which offline tests never call. `tests/test_prefdata.py` replaces the
fetch with fixture rows and feeds them through the same normalizers the cloud
path uses.
