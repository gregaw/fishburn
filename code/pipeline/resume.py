"""Resume points for long training stages (experiment 0114).

A long training run can take hours, and without a resume point a stage that
dies (a timeout, a lost container, a hand stop) restarts from step 0: a
128-step run once lost 50 steps (~$1.40) that way. With `train.resume_every: N` in the
profile, TRL writes a full checkpoint every N optimizer steps (weights,
optimizer and scheduler state, RNG state, trainer state with its log history)
and this module copies it into the ArtifactStore, together with the stage's
own state (KL estimates, the milestone ladder). Re-dispatching the same run id
picks it up: the stage is not complete, its params hash matches, and training
continues from the saved step.

Two slots alternate, and `resume.json` names the slot holding the last
complete copy, written only after the copy finishes. A stage killed in the
middle of a copy therefore resumes from the previous checkpoint instead of a
half-written one. The resume files are deleted when the stage completes.
"""

import os
from pathlib import Path

RESUME = "resume"
INDEX = "resume.json"
SLOTS = ("slot0", "slot1")


class ResumePoint:
    def __init__(self, store, stage: str, params_hash: str):
        self.store = store
        self.stage = stage
        self.params_hash = params_hash

    def _key(self, *parts: str) -> str:
        return self.store.key(self.stage, RESUME, *parts)

    def load(self, workdir: str) -> tuple[str | None, dict]:
        """Restore the last resume point into `workdir`. Returns the TRL
        checkpoint directory to resume from (None to start fresh) and the
        stage state saved with it. A resume point from different params is
        ignored: it belongs to a different run of this stage."""
        index_key = self._key(INDEX)
        if not self.store.exists(index_key):
            return None, {}
        index = self.store.get_json(index_key)
        if index.get("params_hash") != self.params_hash:
            return None, {}
        prefix = self._key(index["slot"]) + "/"
        dest = Path(workdir) / f"checkpoint-{index['step']}"
        for key in self.store.list(prefix):
            self.store.get_file(key, dest / key[len(prefix) :])
        return str(dest), index.get("state", {})

    def save(self, checkpoint_dir: str, step: int, state: dict) -> None:
        index_key = self._key(INDEX)
        last = self.store.get_json(index_key).get("slot") if self.store.exists(index_key) else None
        slot = SLOTS[1] if last == SLOTS[0] else SLOTS[0]
        self.store.delete(self._key(slot))
        root = Path(checkpoint_dir)
        for path in sorted(root.rglob("*")):
            if path.is_file():
                self.store.put_file(self._key(slot, *path.relative_to(root).parts), path)
        self.store.flush()  # the files must be durable before the index names them
        self.store.put_json(
            index_key, {"params_hash": self.params_hash, "slot": slot, "step": step, "state": state}
        )
        self.store.flush()

    def clear(self) -> None:
        self.store.delete(self.store.key(self.stage, RESUME))
        self.store.flush()

    def callback(self, state_fn):
        """A TrainerCallback that copies each checkpoint TRL writes into the
        store, with `state_fn()` (the stage's own state at that step)."""
        from transformers import TrainerCallback

        point = self

        class ResumeCallback(TrainerCallback):
            def on_save(self, args, state, control, **kw):
                step = state.global_step
                ckpt = os.path.join(args.output_dir, f"checkpoint-{step}")
                point.save(ckpt, step, state_fn())

        return ResumeCallback()
