"""The pairwise judge, P(y1 > y2 | x): NLHF's replacement for the scalar
reward model.

`base` defines the interface, the deterministic offline stub, and the
ArtifactStore-backed verdict cache; `pairrm` is the real backend
(llm-blender/PairRM, lazy import); `agreement` scores a judge against
labelled (prompt, chosen, rejected) records.
"""

from pipeline.judge.agreement import agreement_report  # noqa: F401
from pipeline.judge.base import (  # noqa: F401
    CachedJudge,
    PreferenceJudge,
    StubJudge,
)
