"""DPO leg (experiment 0113): trl DPOTrainer on the relabelled
dataset pairs. Offline, Bradley-Terry: the Borda-like baseline."""

from pipeline.legs.common import base_training_args

TRAINER = "trl.DPOTrainer"
# the judge already spoke in relabel; loading it here would only take GPU
# memory (experiment 0122: an 8B judge beside DPO ran an L4 out of memory)
USES_JUDGE = False


def train(
    policy, ref, tokenizer, rows, judge, shared, tcfg, callbacks, output_dir, resume_from=None
):
    """`rows`: [{"prompt" (rendered), "chosen", "rejected"}]. The judge is
    unused here: DPO's preference comes from the relabelled pairs, which the
    same judge produced in the relabel stage."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    args = DPOConfig(
        **base_training_args(tcfg, policy.device.type, output_dir),
        beta=shared.beta,
        max_length=tcfg["max_length"],
        loss_type="sigmoid",
    )
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=args,
        train_dataset=Dataset.from_list(
            [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows]
        ),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_from)
    return trainer.state.log_history
