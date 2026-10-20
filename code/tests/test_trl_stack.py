"""The pinned training stack (experiment 0108): the TRL release
whose OnlineDPOTrainer still takes a pairwise `judge`, and
the adapter that turns our judge into one."""

import inspect
import re
from pathlib import Path

import pytest

from pipeline.judge import StubJudge
from pipeline.judge.trl_judge import make_trl_judge

REQS = Path(__file__).resolve().parent.parent / "configs" / "requirements.txt"


def pinned(package: str) -> str:
    m = re.search(rf"^{re.escape(package)}==(\S+)$", REQS.read_text(), re.M)
    assert m, f"{package} not pinned exactly in {REQS.name}"
    return m.group(1)


@pytest.mark.parametrize(
    "package", ["trl", "transformers", "torch", "peft", "accelerate", "datasets"]
)
def test_environment_matches_the_pins(package):
    import importlib.metadata

    assert importlib.metadata.version(package).split("+")[0] == pinned(package)


def test_online_dpo_trainer_accepts_a_pairwise_judge():
    from trl.experimental.online_dpo import OnlineDPOTrainer

    assert "judge" in inspect.signature(OnlineDPOTrainer.__init__).parameters


def test_trl_major_version_is_zero():
    # 1.x dropped the judge argument (experiment 0108); do not bump past 0.x
    # without re-reading that decision
    assert pinned("trl").startswith("0.")


def test_adapter_is_a_trl_pairwise_judge_with_scores_ranks_and_ties():
    from trl.experimental.judges import BasePairwiseJudge

    fixed = {"a": 0.9, "b": 0.2, "c": 0.5}
    stub = StubJudge(fn=lambda prompt, y1, y2: fixed[y1])
    judge = make_trl_judge(stub)
    assert isinstance(judge, BasePairwiseJudge)
    pairs = [("a", "x"), ("b", "x"), ("c", "x")]
    assert judge.judge(["p"] * 3, pairs, return_scores=True) == [0.9, 0.2, 0.5]
    assert judge.judge(["p"] * 3, pairs) == [0, 1, 0]  # tie: first wins


def test_adapter_maps_rendered_prompts_back_to_raw():
    seen = []
    stub = StubJudge(fn=lambda prompt, y1, y2: seen.append(prompt) or 0.5)
    judge = make_trl_judge(stub, raw_prompt={"<t>hi</t>": "hi"}.get)
    judge.judge(["<t>hi</t>"], [("a", "b")])
    assert seen == ["hi"]
