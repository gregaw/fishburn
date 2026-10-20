"""The training legs (experiments 0108, 0109, 0113, 0123).

| leg | trainer | preference signal | role |
| --- | --- | --- | --- |
| `online_dpo` | trl OnlineDPOTrainer | judge on own samples, BT loss | RLHF-style, Borda-like |
| `nashmp` | own loop (nashmp.py) | judge on own samples, Mirror Prox | NLHF, Maximal Lottery |
| `dpo` | trl DPOTrainer | dataset pairs, offline | cheap check that a setup trains |

Online DPO and NashMP sample from the policy and ask the same judge, so the
training rule is the only thing that differs between them.

Each leg module exposes `train(policy, ref, tokenizer, rows, judge, shared,
tcfg, callbacks) -> log_history`. Everything else (models, data, the judge,
KL milestones, checkpoints, manifests) is shared and lives in
pipeline/stages/train.py, so a leg cannot quietly differ from the others.
"""

from pipeline.legs import dpo, nashmp, online_dpo

LEGS = {
    "dpo": dpo,
    "online_dpo": online_dpo,
    "nashmp": nashmp,
}
