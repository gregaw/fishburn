"""The frozen config every leg imports (experiment 0110)."""

import copy

import pytest

from pipeline import manifest
from pipeline.config import PROFILES, load_profile
from pipeline.exec.local_backend import LocalArtifactStore
from pipeline.judge import CachedJudge
from pipeline.legs import LEGS
from pipeline.shared import build_judge, load_shared, raw_prompt_lookup, render_prompt
from pipeline.stages import data_prep, relabel, train


class FakeChatTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return f"<user>{messages[0]['content']}</user><assistant>"


def test_every_profile_builds_a_shared_config():
    for name in PROFILES:
        shared = load_shared(load_profile(name))
        assert shared.base_model == load_profile(name)["model"]["name"]
        assert len(shared.hash) == 16
        assert list(shared.kl.milestones) == sorted(shared.kl.milestones)


def test_hash_is_stable_and_moves_with_any_shared_field():
    p = load_profile("smoke")
    assert load_shared(p).hash == load_shared(copy.deepcopy(p)).hash
    for mutate in (
        lambda q: q["shared"].update(beta=0.2),
        lambda q: q["shared"]["sampling"].update(temperature=0.5),
        lambda q: q["shared"]["kl"].update(milestones=[1.0]),
        lambda q: q["judge"].update(backend="pairrm"),
        lambda q: q["data_prep"].update(seed=8),
    ):
        q = copy.deepcopy(p)
        mutate(q)
        assert load_shared(q).hash != load_shared(p).hash


def chat_profile(template="chat"):
    p = load_profile("canary")
    p["shared"]["template"] = template
    return p


def test_render_and_raw_lookup_round_trip():
    shared = load_shared(chat_profile())
    tok = FakeChatTokenizer()
    assert render_prompt(shared, tok, "hi") == "<user>hi</user><assistant>"
    lookup = raw_prompt_lookup(shared, tok, ["hi", "there"])
    assert lookup("<user>there</user><assistant>") == "there"
    assert lookup("unknown") == "unknown"
    plain = load_shared(load_profile("smoke"))
    assert render_prompt(plain, None, "hi") == "hi"


def test_build_judge_is_cached_and_rejects_unknown_backends(tmp_path):
    store = LocalArtifactStore(tmp_path, run_id="t")
    assert isinstance(build_judge(load_shared(load_profile("smoke")), store), CachedJudge)
    p = load_profile("smoke")
    p["judge"]["backend"] = "nope"
    with pytest.raises(ValueError, match="unknown judge backend"):
        build_judge(load_shared(p), store)


def test_all_legs_get_the_identical_judge_template_prompts_and_hash(tmp_path, monkeypatch):
    """The P4 acceptance test: every leg is handed the same shared config,
    the same rendered prompts and a judge built from the same spec, and
    records the same shared_config_hash in its manifest."""
    profile = load_profile("smoke")
    store = LocalArtifactStore(tmp_path, run_id="t")
    data_prep.run(store, profile)
    relabel.run(store, profile)

    seen = {}
    for leg, module in LEGS.items():

        def fake_train(
            policy,
            ref,
            tokenizer,
            rows,
            judge,
            shared,
            tcfg,
            callbacks,
            output_dir,
            resume_from=None,
            leg=leg,
        ):
            seen[leg] = {
                "shared": shared,
                "prompts": [r["prompt"] for r in rows],
                "judge_name": judge and judge.name,
                "judge_verdict": judge
                and judge.judge(
                    [rows[0]["prompt"]],
                    [(rows[0]["chosen"], rows[0]["rejected"])],
                    return_scores=True,
                ),
            }
            return [{"loss": 1.0, "step": 1}]

        monkeypatch.setattr(module, "train", fake_train)
        train.make(leg)(store, profile)

    # DPO's judge spoke in relabel, so it is handed none (experiment 0122);
    # the online legs get the same one
    assert seen["dpo"]["judge_name"] is None
    online = [leg for leg in seen if leg != "dpo"]
    first = seen[online[0]]
    for leg, got in seen.items():
        assert got["shared"] == first["shared"] and got["prompts"] == first["prompts"], leg
        if leg != "dpo":
            assert got == first, leg
        assert manifest.load(store, f"train_{leg}")["shared_config_hash"] == first["shared"].hash


def test_chat_nothink_turns_qwen3_thinking_off():
    seen = {}

    class Qwen3Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
            seen.update(kw)
            return "rendered"

    shared = load_shared(chat_profile("chat-nothink"))
    assert render_prompt(shared, Qwen3Tok(), "hi") == "rendered"
    assert seen == {"enable_thinking": False}
    assert shared.hash != load_shared(chat_profile()).hash
