"""Judge interface, offline stub, and the cached-verdict wrapper.

A verdict is P(y1 > y2 | x) in [0, 1]. Backends implement one batch method;
everything else (single queries, caching, agreement scoring) is built on it.
"""

import hashlib
import json
from abc import ABC, abstractmethod

from pipeline.exec.base import ArtifactStore


class PreferenceJudge(ABC):
    """P(y1 > y2 | x). `name` feeds cache keys, so two differently-configured
    backends never share verdicts."""

    name: str = "judge"

    @abstractmethod
    def prob_prefer_batch(self, items: list[tuple[str, str, str]]) -> list[float]:
        """items = [(prompt, y1, y2), ...] -> [P(y1 > y2 | prompt), ...]."""

    def prob_prefer(self, prompt: str, y1: str, y2: str) -> float:
        return self.prob_prefer_batch([(prompt, y1, y2)])[0]


class StubJudge(PreferenceJudge):
    """Deterministic offline stand-in for tests and the smoke profile.

    Default heuristic prefers the longer response (a crude but *stable*
    signal), with a hash-derived probability margin so downstream code sees
    realistic non-{0,1} values. `fn(prompt, y1, y2) -> p` overrides it."""

    name = "stub"

    def __init__(self, fn=None):
        self._fn = fn

    def _default(self, prompt: str, y1: str, y2: str) -> float:
        if len(y1) == len(y2):
            h = hashlib.sha256(f"{prompt}\x00{y1}\x00{y2}".encode()).digest()
            return 0.5 + (h[0] / 255 - 0.5) * 0.2  # deterministic near-tie
        margin = min(0.45, abs(len(y1) - len(y2)) / 200)
        return 0.5 + margin if len(y1) > len(y2) else 0.5 - margin

    def prob_prefer_batch(self, items):
        fn = self._fn or self._default
        return [fn(*item) for item in items]


class CachedJudge(PreferenceJudge):
    """Caches verdicts in the ArtifactStore so repeat queries are free: a
    re-run, or a later stage judging the same pair, never re-runs the judge.

    Canonicalization: a verdict is stored once per unordered {y1, y2} (under
    the lexicographically-sorted order) and flipped on read, so P(y2>y1)
    never misses a cached P(y1>y2). Layout: one JSON per verdict under
    runs/<run_id>/judge/cache/<backend>/<sha[:2]>/<sha>.json, run-scoped
    like every other artifact."""

    def __init__(self, backend: PreferenceJudge, store: ArtifactStore):
        self.backend = backend
        self.store = store
        self.name = f"cached-{backend.name}"
        self.backend_calls = 0  # queries actually sent to the backend

    def _key(self, prompt: str, a: str, b: str) -> str:
        h = hashlib.sha256(json.dumps([self.backend.name, prompt, a, b]).encode()).hexdigest()
        return self.store.key("judge", "cache", self.backend.name, h[:2], f"{h}.json")

    def prob_prefer_batch(self, items):
        out: list[float | None] = [None] * len(items)
        misses: list[int] = []
        for i, (prompt, y1, y2) in enumerate(items):
            a, b = sorted((y1, y2))
            try:
                p_ab = self.store.get_json(self._key(prompt, a, b))["p"]
                out[i] = p_ab if (y1, y2) == (a, b) else 1.0 - p_ab
            except KeyError:
                misses.append(i)
        if misses:
            self.backend_calls += len(misses)
            fresh = self.backend.prob_prefer_batch([items[i] for i in misses])
            for i, p in zip(misses, fresh):
                prompt, y1, y2 = items[i]
                a, b = sorted((y1, y2))
                p_ab = p if (y1, y2) == (a, b) else 1.0 - p
                self.store.put_json(
                    self._key(prompt, a, b), {"p": p_ab, "backend": self.backend.name}
                )
                out[i] = p
        return out
