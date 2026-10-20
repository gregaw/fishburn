"""UltraFeedback and HH-RLHF loaders -> normalized (prompt, chosen, rejected).

Dataset schemas (as of the pinned dataset revisions):

- `HuggingFaceH4/ultrafeedback_binarized`: `prompt` is a plain string;
  `chosen` / `rejected` are chat transcripts (list of {role, content}) ending
  with the assistant reply; `score_chosen` / `score_rejected` are the GPT-4
  ratings the binarization used. Rows where the two scores tie carry no
  preference signal, so normalization records the tie and the agreement eval
  drops them.
- `Anthropic/hh-rlhf`: `chosen` / `rejected` are raw "\n\nHuman: ...
  \n\nAssistant: ..." transcripts sharing a prompt prefix; the response is
  whatever follows the final "Assistant:" marker.
"""

import random

UF_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
HH_DATASET = "Anthropic/hh-rlhf"
_HUMAN = "\n\nHuman:"
_ASSISTANT = "\n\nAssistant:"


def normalize_ultrafeedback(row: dict) -> dict | None:
    """One ultrafeedback_binarized row -> normalized record, or None if the
    row is unusable (missing/empty fields, identical responses)."""

    def last_assistant(messages) -> str | None:
        if not isinstance(messages, list):
            return None
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                content = m.get("content")
                return content.strip() if isinstance(content, str) else None
        return None

    prompt = (row.get("prompt") or "").strip()
    chosen = last_assistant(row.get("chosen"))
    rejected = last_assistant(row.get("rejected"))
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    rec = {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": "ultrafeedback"}
    sc, sr = row.get("score_chosen"), row.get("score_rejected")
    if isinstance(sc, (int, float)) and isinstance(sr, (int, float)):
        rec["score_chosen"] = float(sc)
        rec["score_rejected"] = float(sr)
        rec["tie"] = sc == sr
    return rec


def _split_transcript(transcript: str) -> tuple[str, str] | None:
    """'\n\nHuman: ...\n\nAssistant: reply' -> (prompt ending at the final
    Assistant marker, reply). None if there is no Assistant marker or the
    reply is empty."""
    if not isinstance(transcript, str):
        return None
    idx = transcript.rfind(_ASSISTANT)
    if idx < 0:
        return None
    prompt = transcript[: idx + len(_ASSISTANT)].strip()
    response = transcript[idx + len(_ASSISTANT) :].strip()
    if not prompt or not response:
        return None
    return prompt, response


def normalize_hh(row: dict) -> dict | None:
    """One hh-rlhf row -> normalized record, or None if unusable. The two
    transcripts must share the prompt prefix; a mismatch (rare in the raw
    data) makes the pair meaningless and drops the row."""
    c = _split_transcript(row.get("chosen"))
    r = _split_transcript(row.get("rejected"))
    if c is None or r is None:
        return None
    if c[0] != r[0] or c[1] == r[1]:
        return None
    return {"prompt": c[0], "chosen": c[1], "rejected": r[1], "source": "hh_rlhf"}


def within_chars(rec: dict, max_chars: int | None) -> bool:
    """Length filter: prompt plus the longer response, in characters. Keeps
    training sequences inside the trainers' max_length instead of letting
    truncation cut answers (experiment 0113)."""
    if max_chars is None:
        return True
    return len(rec["prompt"]) + max(len(rec["chosen"]), len(rec["rejected"])) <= max_chars


def subsample(rows: list, n: int, seed: int) -> list:
    """Deterministic sample of n rows (all of them, original order, if
    n >= len)."""
    if n >= len(rows):
        return list(rows)
    return random.Random(f"{seed}-subsample").sample(rows, n)


# -- network fetchers (lazy `datasets` import; exercised in-container) --------


def _load_hf_rows(dataset: str, split: str, take: int):
    """First `take` raw rows of a HF dataset split, streaming so a 1k sample
    does not download the full corpus."""
    from datasets import load_dataset  # lazy: offline tests never import this

    ds = load_dataset(dataset, split=split, streaming=True)
    rows = []
    for row in ds:
        rows.append(row)
        if len(rows) >= take:
            break
    return rows


def load_ultrafeedback(
    n: int, seed: int, split: str = "train_prefs", oversample: int = 4, max_chars: int | None = None
) -> list[dict]:
    """n normalized non-tie UltraFeedback records. Streams `oversample * n`
    raw rows, normalizes, drops ties and unusable or over-long rows, then
    subsamples, so the sample is deterministic in (n, seed, split, max_chars)."""
    raw = _load_hf_rows(UF_DATASET, split, take=oversample * n)
    recs = [
        r
        for r in (normalize_ultrafeedback(x) for x in raw)
        if r is not None and not r.get("tie") and within_chars(r, max_chars)
    ]
    return subsample(recs, n, seed)


def load_hh(
    n: int, seed: int, split: str = "train", oversample: int = 4, max_chars: int | None = None
) -> list[dict]:
    raw = _load_hf_rows(HH_DATASET, split, take=oversample * n)
    recs = [
        r for r in (normalize_hh(x) for x in raw) if r is not None and within_chars(r, max_chars)
    ]
    return subsample(recs, n, seed)


# held-out split per source, for the seen-versus-unseen judge comparison
# (experiments 0111 and 0112): PairRM's training data includes both corpora,
# so agreement on the training split may be inflated by memory
TRAIN_SPLITS = {"ultrafeedback": "train_prefs", "hh_rlhf": "train"}
TEST_SPLITS = {"ultrafeedback": "test_prefs", "hh_rlhf": "test"}


# data_prep's source registry; tests monkeypatch entries to keep the stage
# unit-testable offline.
LOADERS = {
    "ultrafeedback": load_ultrafeedback,
    "hh_rlhf": load_hh,
}


def load_records(source: str, n: int, seed: int, **kwargs) -> list[dict]:
    if source not in LOADERS:
        raise ValueError(f"unknown source {source!r}, expected one of {sorted(LOADERS)}")
    return LOADERS[source](n, seed, **kwargs)
