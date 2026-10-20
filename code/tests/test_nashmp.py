"""NashMP leg (experiment 0123): the Mirror Prox loop on the tiny model with
a stub judge, its pieces, and the vLLM sampler against a fake `vllm`."""

import os
import sys
import types

import pytest

from pipeline.config import load_profile
from pipeline.legs import nashmp
from pipeline.models import build_causal_lm, build_tokenizer
from pipeline.shared import Sampling, load_shared


def test_advantage_rewards_wins_and_penalises_both_log_ratios():
    a = nashmp.advantages([0.75, 0.5], [-1.0, -1.0], [-2.0, -1.0], [-1.0, -3.0], beta=0.1, eta=0.2)
    # (0.75 - 0.5) - 0.1 * 1 - 0.5 * 0 ; (0) - 0 - 0.5 * 2
    assert a == pytest.approx([0.15, -1.0])


def test_lr_factor_is_constant_then_decays_to_zero():
    f = [nashmp.lr_factor(s, 8, 0.25) for s in range(8)]
    assert f[:6] == [1.0] * 6 and f[6] == 1.0 and f[7] == 0.5


def test_cut_stops_after_eos_and_drops_padding():
    assert nashmp._cut([5, 6, 1, 7], eos=1, pad=0) == [5, 6, 1]
    assert nashmp._cut([5, 0, 0], eos=1, pad=0) == [5]
    assert nashmp._cut([5, 6], eos=1, pad=0) == [5, 6]


class PrefersLonger:
    name = "longer"

    def __init__(self):
        self.calls = 0

    def judge(self, prompts, completions, shuffle_order=True, return_scores=False):
        self.calls += 1
        return [1.0 if len(a) > len(b) else 0.0 if len(a) < len(b) else 0.5 for a, b in completions]


def test_win_rates_average_over_the_opponents_answers_in_one_call():
    j = PrefersLonger()
    o = nashmp.win_rates(j, ["p", "q"], [["aaa", "a"], ["bb"]], [["aa", "aaaa"], ["b"]])
    assert j.calls == 1
    assert o == [[0.5, 0.0], [1.0]]


class Recorder:
    def __init__(self):
        self.begin, self.steps = 0, []

    def on_train_begin(self, args, state, control, model=None, **kw):
        self.begin += 1

    def on_step_end(self, args, state, control, model=None, **kw):
        self.steps.append(state.global_step)


def test_nashmp_trains_both_phases_and_moves_the_policy(tmp_path):
    import copy

    import torch

    profile = load_profile("smoke")
    shared = load_shared(profile)
    tok = build_tokenizer(shared.base_model)
    torch.manual_seed(0)
    policy = build_causal_lm(shared.base_model, "float32", tok)
    ref = copy.deepcopy(policy).eval()
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    rows = [{"prompt": "is the answer yes or no ."}] * 4
    tcfg = {
        "steps": 4,
        "batch_size": 2,
        "learning_rate": 1e-2,
        "seed": 3,
        "inner_steps": 1,
        "max_length": 64,
    }
    rec = Recorder()
    hist = nashmp.train(policy, ref, tok, rows, PrefersLonger(), shared, tcfg, [rec], str(tmp_path))
    assert [h["phase"] for h in hist] == [1, 2, 1, 2]
    assert rec.begin == 1 and rec.steps == [1, 2, 3, 4]
    assert all(0.0 <= h["win_rate_vs_opponent"] <= 1.0 for h in hist)
    assert any(not torch.equal(before[k], v) for k, v in policy.state_dict().items())


def test_nashmp_refuses_resume_points(tmp_path):
    with pytest.raises(ValueError, match="resume"):
        nashmp.train(None, None, None, [], None, None, {"resume_every": 4}, [], str(tmp_path))


def test_unknown_sampler_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="sampler"):
        nashmp.build_sampler("nope", None, None, None, {}, str(tmp_path))


@pytest.fixture
def fake_vllm(monkeypatch):
    log = {"loaded": [], "params": []}

    class EngineModel:
        def load_weights(self, weights):
            log["loaded"].append([name for name, _ in weights])

    class LLM:
        def __init__(self, model, **kw):
            log["init"] = (model, kw)
            worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=EngineModel()))
            self.llm_engine = types.SimpleNamespace(
                model_executor=types.SimpleNamespace(driver_worker=worker)
            )

        def generate(self, prompts, params, use_tqdm=False):
            log["params"].append(params.kw)
            out = types.SimpleNamespace
            return [
                out(outputs=[out(token_ids=(i, j)) for j in range(params.kw["n"])])
                for i, _ in enumerate(prompts)
            ]

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    mod = types.ModuleType("vllm")
    mod.LLM, mod.SamplingParams = LLM, SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", mod)
    # VLLMSampler sets RANK, WORLD_SIZE and MASTER_* in os.environ; left behind,
    # they make every later trainer start a process group (and on a Mac fail
    # its barrier on MPS). monkeypatch.delenv records nothing for an absent
    # variable, so restore them by hand.
    saved = {k: os.environ.get(k) for k in nashmp.DIST_ENV}
    yield log
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_vllm_sampler_syncs_every_parameter_and_returns_token_ids(fake_vllm):
    s = Sampling(max_new_tokens=5, temperature=0.9, top_p=1.0)
    sampler = nashmp.VLLMSampler("/m", s, seed=1, gpu_memory_utilization=0.2, max_model_len=4096)
    assert fake_vllm["init"][1]["distributed_executor_backend"] == "external_launcher"

    class Model:
        def named_parameters(self):
            p = types.SimpleNamespace(data=0)
            return iter([("a.weight", p), ("b.weight", p)])

    sampler.sync(Model())
    assert fake_vllm["loaded"] == [["a.weight", "b.weight"]]
    out = sampler.sample(["x", "y"], n=2, seed=4)
    assert out == [[[0, 0], [0, 1]], [[1, 0], [1, 1]]]
    assert fake_vllm["params"][0] == {
        "n": 2,
        "temperature": 0.9,
        "top_p": 1.0,
        "max_tokens": 5,
        "seed": 4,
    }


def test_vllm_sampler_close_removes_the_env_it_added(fake_vllm, monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "1234")  # set by someone else: left alone
    s = Sampling(max_new_tokens=5, temperature=0.9, top_p=1.0)
    sampler = nashmp.VLLMSampler("/m", s, seed=1, gpu_memory_utilization=0.2, max_model_len=4096)
    assert os.environ["RANK"] == "0" and os.environ["MASTER_PORT"] == "1234"
    sampler.close()
    assert "RANK" not in os.environ and os.environ["MASTER_PORT"] == "1234"
    assert not hasattr(sampler, "llm")
