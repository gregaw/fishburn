"""KL-from-base estimator and milestone ladder (experiments 0110, 0113)."""

import copy

import torch

from pipeline.kl import crossed, make_kl_callback, sequence_kl
from pipeline.models import build_causal_lm, build_tokenizer
from pipeline.shared import KLLadder

PROMPTS = ["is the answer yes or no .", "think step by step .", "the answer is"]


def tiny():
    tok = build_tokenizer("tiny-random")
    torch.manual_seed(0)
    return tok, build_causal_lm("tiny-random", "float32", tok)


def test_crossed_reports_each_milestone_once_lowest_first():
    saved = set()
    assert crossed((4.0, 1.0, 2.0), saved, 2.5) == [1.0, 2.0]
    saved |= {1.0, 2.0}
    assert crossed((4.0, 1.0, 2.0), saved, 3.0) == []
    assert crossed((4.0, 1.0, 2.0), saved, 9.0) == [4.0]


def test_kl_of_a_model_against_itself_is_zero():
    tok, model = tiny()
    est = sequence_kl(model, copy.deepcopy(model), tok, PROMPTS, 6, "cpu", seed=1)
    assert est["n"] == len(PROMPTS)
    assert abs(est["kl"]) < 1e-5


def test_kl_grows_when_the_policy_moves_away():
    tok, model = tiny()
    ref = copy.deepcopy(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    est = sequence_kl(model, ref, tok, PROMPTS * 8, 6, "cpu", seed=1)
    assert est["kl"] > 0.1


def test_estimator_restores_padding_side_and_training_mode():
    tok, model = tiny()
    model.train()
    sequence_kl(model, copy.deepcopy(model), tok, PROMPTS, 4, "cpu")
    assert model.training and tok.padding_side == "right"


def test_callback_measures_at_start_every_k_and_end_and_saves_milestones():
    tok, model = tiny()
    evals, saved = [], []
    cb = make_kl_callback(
        ref=copy.deepcopy(model),
        tokenizer=tok,
        prompts=PROMPTS,
        ladder=KLLadder(milestones=(-1.0, 1e9), eval_every=2, n_prompts=3),
        max_new_tokens=4,
        device="cpu",
        seed=0,
        on_eval=lambda step, est: evals.append(step),
        on_milestone=lambda m, mdl, step, est: saved.append((m, step)),
    )

    class State:
        max_steps = 5
        global_step = 0

    state = State()
    cb.on_train_begin(None, state, None, model=model)
    for step in range(1, 6):
        state.global_step = step
        cb.on_step_end(None, state, None, model=model)
    assert evals == [0, 2, 4, 5]
    assert saved == [(-1.0, 0)]  # crossed once, at the first measurement


def test_completion_log_dists_match_a_full_log_softmax():
    from pipeline.kl import _completion_log_dists

    tok, model = tiny()
    enc = tok(["is the answer yes or no ."], return_tensors="pt")
    out = model.generate(**enc, do_sample=False, max_new_tokens=4, pad_token_id=tok.pad_token_id)
    attn = torch.ones_like(out)
    plen = enc["input_ids"].shape[1]
    got = _completion_log_dists(model, out, attn, plen)
    want = torch.log_softmax(model(input_ids=out).logits[:, plen - 1 : -1].float(), -1)
    assert torch.allclose(got, want, atol=1e-5)


def test_estimator_is_non_negative_per_sequence():
    # exact per-position KL is >= 0, unlike the plain log-ratio estimator
    tok, model = tiny()
    ref = copy.deepcopy(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    est = sequence_kl(model, ref, tok, PROMPTS, 6, "cpu", seed=3)
    assert est["kl"] >= 0
