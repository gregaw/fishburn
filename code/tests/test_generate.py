"""Generation backends (experiment 0122). The vLLM path runs against a fake
`vllm` module: real vLLM exists only in the cloud image."""

import sys
import types

import pytest

from pipeline.generate import sample_models
from pipeline.shared import Sampling


class FakeModel:
    def __init__(self, tag):
        self.tag = tag

    def save_pretrained(self, d, safe_serialization=True):
        import os

        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "tag"), "w").write(self.tag)


class FakeTokenizer:
    def save_pretrained(self, d):
        pass


@pytest.fixture
def fake_vllm(monkeypatch):
    calls = []

    class LLM:
        def __init__(self, model, **kw):
            self.tag = open(f"{model}/tag").read()
            calls.append(("LLM", self.tag, kw))

        def generate(self, prompts, params, use_tqdm=False):
            calls.append(("generate", self.tag, params.kw))
            out = types.SimpleNamespace
            return [out(outputs=[out(text=f" {self.tag}:{p} ")]) for p in prompts]

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    mod = types.ModuleType("vllm")
    mod.LLM, mod.SamplingParams = LLM, SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return calls


def test_vllm_backend_samples_each_model_the_same_way(fake_vllm):
    models = [FakeModel("base"), FakeModel("trained")]
    s = Sampling(max_new_tokens=7, temperature=0.9, top_p=1.0)
    base, trained = sample_models(
        models, FakeTokenizer(), ["a", "b"], s, seed=3, device="cuda", backend="vllm"
    )
    assert base == ["base:a", "base:b"] and trained == ["trained:a", "trained:b"]
    assert models == []  # GPU copies released before the engines start
    gens = [c for c in fake_vllm if c[0] == "generate"]
    assert [g[2] for g in gens] == [dict(temperature=0.9, top_p=1.0, max_tokens=7, seed=3)] * 2


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown generation backend"):
        sample_models([], None, [], None, 0, "cpu", backend="nope")
