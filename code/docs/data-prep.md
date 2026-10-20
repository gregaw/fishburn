# Data preparation: what `data_prep` produces, and why

Everything downstream (training, the judge and the checks) consumes **one record shape**,
whatever the data came from. `data_prep` is the stage that gets raw corpora
into that shape, deterministically, and writes them to the artifact store.

## The unit: a pairwise preference record

A training example is *"for this prompt, response A was preferred over
response B"*. Nothing in the pipeline ever sees a scalar score as its
training signal. That is the point of NLHF (preference over pairs, not
reward over responses), and it is what the pairwise judge
`P(y₁ ≻ y₂ | x)` in `pipeline/judge/` is asked to reproduce.

```json
{"prompt": "How do I boil an egg?",
 "chosen": "Put the egg in boiling water for 7 minutes, then cool it in cold water.",
 "rejected": "Eggs.",
 "source": "ultrafeedback", "score_chosen": 8.0, "score_rejected": 3.0, "tie": false}
```

`prompt`, `chosen`, `rejected` are mandatory. The rest is provenance kept so
later stages can filter (for example, the judge-agreement check drops `tie: true` rows,
which carry no preference signal). Rows where `chosen == rejected`, or any
of the three fields is empty, are dropped during normalization.

For the pairwise sources the stage writes:

| Artifact | Contents |
| --- | --- |
| `data_prep/prefs.jsonl` | `n_pairs` records as above, one per line |
| `data_prep/eval_prompts.jsonl` | `n_eval_prompts` **held-out** prompts, `{"prompt": ...}`, never in `prefs.jsonl` |
| `data_prep/test_prefs.jsonl` | with `n_test_pairs` set, that many pairs from the dataset's test split (`ultrafeedback` and `hh_rlhf` only), never trained on; the preference check reads them |
| `data_prep/manifest.json` | params hash + output list, so an unchanged config skips the stage on re-run |

The held-out split is taken *first* from the deterministic sample and the
training pairs from the remainder, so eval prompts cannot leak into
training no matter how `n_pairs` changes.

## The three sources

`data_prep.source` in the profile selects where the records come from:
`synthetic`, `ultrafeedback` or `hh_rlhf`. The
loaders are pure normalizers (one raw row in, one record or `None` out),
unit-tested offline on fixture rows; only the fetch itself needs the network.

**`synthetic`** (the `smoke` profile, used by the offline end-to-end test). A generator, no
network, vocabulary limited to the tiny smoke tokenizer's words. The
"preference" it encodes is a toy CoT-vs-no-CoT axis: the chosen answer
verbalizes its reasoning, the rejected one answers bare.

```json
{"prompt": "is the answer yes or no .",
 "chosen": "think step by step . first the question then the answer . the answer is yes .",
 "rejected": "the answer is no ."}
```

Its only job is to give the pipeline something shaped like real data so the
wiring is exercised end to end in seconds on CPU.

**`ultrafeedback`** (`HuggingFaceH4/ultrafeedback_binarized`; canary and
real profiles). Raw rows carry the prompt plus two chat transcripts, each
ending in an assistant turn, and the two GPT-4 scores the binarization was
made from:

```json
{"prompt": "How do I boil an egg?",
 "chosen":   [{"role": "user", "content": "How do I boil an egg?"},
              {"role": "assistant", "content": "Put the egg in boiling water for 7 minutes ..."}],
 "rejected": [{"role": "user", "content": "How do I boil an egg?"},
              {"role": "assistant", "content": "Eggs."}],
 "score_chosen": 8.0, "score_rejected": 3.0}
```

Normalization takes the **last assistant message** of each transcript as the
response, keeps the scores, and marks `tie` when they are equal. Ties are
excluded from the training sample. The margin `score_chosen − score_rejected`
is what the judge-agreement report (`scripts/judge_agreement.py`) buckets by.
On a 1k sample of the binarized test split, PairRM agreed with the labels on 0.832 of pairs overall, rising to
0.858, 0.900 and 0.925 at margins of at least 1, 2 and 4.

**`hh_rlhf`** (`Anthropic/hh-rlhf`). Raw rows are two flat transcripts in
`\n\nHuman: ... \n\nAssistant: ...` form that share a prefix and differ only
in the final assistant reply:

```
chosen:   "\n\nHuman: How do I boil an egg?\n\nAssistant: Put the egg in boiling water ..."
rejected: "\n\nHuman: How do I boil an egg?\n\nAssistant: Eggs."
```

Normalization splits each at the **final** `Assistant:` marker: everything
up to and including the marker is the prompt (multi-turn context stays in
it), what follows is the response. If the two prompts differ, the pair is
not a comparison of anything and the row is dropped. No scores exist, so no
`tie` field.

Both real loaders **stream** `oversample × n` raw
rows (4× by default), normalize, drop unusable rows, then subsample `n` with a seed. Sampling is
therefore deterministic in `(source, n, seed, split)` and does not download
the full corpus for a 1k sample. With `max_chars` set, pairs whose prompt plus
longer answer exceed that many characters are dropped, so training sequences
fit the trainers' `max_length`.

## Where the network boundary is

The smoke profile is fully offline, so the end-to-end test never touches a
real corpus. The other sources fetch through a lazy
`datasets` import that offline tests never execute; `tests/test_prefdata.py`
monkeypatches the loader registry and feeds fixture
rows through the same normalizers the cloud path uses. The fetchers
themselves are cloud-only code paths.
