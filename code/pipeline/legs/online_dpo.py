"""Online DPO leg (experiment 0109): trl OnlineDPOTrainer, the RLHF-style
leg NashMP is compared against. Same Bradley-Terry loss as DPO,
but on pairs of the policy's own samples labelled by the shared judge, so it
differs from NashMP only in the training rule, not in where the pairs come
from: experiment 0086 found that where the pairs come from, not the loss,
decided the aggregation rule for IPO."""

from pipeline.legs.common import base_training_args, online_args

TRAINER = "trl.experimental.online_dpo.OnlineDPOTrainer"


def sampler_args(tcfg: dict) -> dict:
    """`sampler: vllm` samples the policy in a vLLM engine colocated on the
    training GPU, with weights synced every step, as the NashMP leg does;
    `hf` (the default, and what the CPU tests run) uses transformers generate."""
    kind = tcfg.get("sampler", "hf")
    if kind == "hf":
        return {}
    if kind == "vllm":
        return dict(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=tcfg.get("vllm_gpu_memory_utilization", 0.2),
        )
    raise ValueError(f"unknown Online DPO sampler {kind!r}, expected 'hf' or 'vllm'")


def train(
    policy, ref, tokenizer, rows, judge, shared, tcfg, callbacks, output_dir, resume_from=None
):
    from datasets import Dataset
    from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer

    args = OnlineDPOConfig(
        **base_training_args(tcfg, policy.device.type, output_dir),
        **online_args(shared),
        **sampler_args(tcfg),
        max_length=tcfg["max_length"],
        loss_type="sigmoid",
    )
    trainer = OnlineDPOTrainer(
        model=policy,
        ref_model=ref,
        judge=judge,
        args=args,
        train_dataset=Dataset.from_list([{"prompt": r["prompt"]} for r in rows]),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_from)
    return trainer.state.log_history
