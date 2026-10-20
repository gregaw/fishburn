"""PreferenceJudge -> TRL pairwise judge (experiment 0108).

TRL's OnlineDPOTrainer takes a `BasePairwiseJudge` and asks it for the index
of the better completion; `return_scores=True` gives P(first > second)
instead. Both are answered from the same judge that the NashMP leg and the
checks query, so every leg is driven by one judge.

`shuffle_order` is accepted and ignored: PairRM already scores both orders and
averages them, and the stub is order-symmetric by construction, so shuffling
would only add noise.
"""

from pipeline.judge.base import PreferenceJudge


def make_trl_judge(preference: PreferenceJudge, raw_prompt=lambda p: p):
    """`raw_prompt` maps the prompt string TRL passes (already rendered with
    the chat template) back to the raw user prompt the judge should see."""
    from trl.experimental.judges import BasePairwiseJudge  # lazy: heavy import

    class TRLJudge(BasePairwiseJudge):
        name = f"trl-{preference.name}"

        def judge(self, prompts, completions, shuffle_order=True, return_scores=False):
            items = [(raw_prompt(p), a, b) for p, (a, b) in zip(prompts, completions, strict=True)]
            probs = preference.prob_prefer_batch(items)
            if return_scores:
                return [float(p) for p in probs]
            # tie rule (pipeline/shared.py): exactly 0.5 -> first wins
            return [0 if p >= 0.5 else 1 for p in probs]

    return TRLJudge()
