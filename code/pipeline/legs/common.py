"""What all legs share when handing work to TRL."""


def base_training_args(tcfg: dict, device: str, output_dir: str) -> dict:
    """TrainingArguments fields every leg sets identically. bf16 autocast on
    GPU over fp32 weights, so the reference model and the KL estimator see
    the same precision as the policy. `resume_every: N` makes TRL write a full
    checkpoint every N steps for the stage's resume point (pipeline/resume.py)."""
    every = tcfg.get("resume_every")
    return dict(
        output_dir=output_dir,
        max_steps=tcfg["steps"],
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg.get("grad_accum", 1),
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg.get("lr_scheduler", "constant"),
        warmup_steps=tcfg.get("warmup_steps", 0),
        seed=tcfg["seed"],
        logging_steps=1,
        save_strategy="steps" if every else "no",
        save_steps=every or 500,
        save_total_limit=1,  # the store keeps the copy; the scratch dir needs one
        report_to=[],
        use_cpu=device == "cpu",
        bf16=device == "cuda",
        # TRL's own default, set explicitly so no leg relies on a silent one
        gradient_checkpointing=True,
        disable_tqdm=True,
    )


def online_args(shared) -> dict:
    """Generation settings for the online legs, from the shared config."""
    return dict(
        max_new_tokens=shared.sampling.max_new_tokens,
        temperature=shared.sampling.temperature,
        top_p=shared.sampling.top_p,
        beta=shared.beta,
    )
