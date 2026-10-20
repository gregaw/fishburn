"""PairRM backend: llm-blender/PairRM (Jiang et al. 2023), a 0.4B pairwise
ranker, also the judge of SPPO (Wu et al. 2024).

Imports `llm_blender` lazily: the package (and its torch stack) exists only
in the cloud image or a venv that opted in, and the offline tests never load
it. P(y1 > y2) = sigmoid(comparison logit), with llm-blender's default
bidirectional mode (scores [A,B] and [B,A], averages) for order-bias
robustness.
"""

import math

from pipeline.judge.base import PreferenceJudge

DEFAULT_MODEL = "llm-blender/PairRM"


class PairRMJudge(PreferenceJudge):
    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 8):
        import llm_blender  # lazy: cloud/venv-only dependency

        self.name = f"pairrm:{model.split('/')[-1]}"
        self.batch_size = batch_size
        self._blender = llm_blender.Blender()
        self._blender.loadranker(model, device=device)

    def prob_prefer_batch(self, items):
        if not items:
            return []
        prompts = [p for p, _, _ in items]
        ys1 = [y1 for _, y1, _ in items]
        ys2 = [y2 for _, _, y2 in items]
        logits = self._blender.compare(
            prompts, ys1, ys2, batch_size=self.batch_size, return_logits=True, disable_tqdm=True
        )
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]
