"""KL-from-base estimation and the KL-milestone checkpoint ladder
(experiments 0110 and 0113).

Optimization pressure is compared across legs by KL from the base model, not
by step count: a DPO step, an Online DPO step and a NashMP step are not the
same amount of movement. Every leg runs the same callback with the same
estimator on the same held-out prompts, and saves a checkpoint the first time
its KL crosses each milestone in the shared config.

The estimator samples y ~ policy at temperature 1 (the policy's own
distribution, whatever the training sampler uses) and, along each sample,
sums the exact next-token KL between the two models at every position:

    KL(policy || base) ~= mean over y of sum_t KL(p(. | y<t) || q(. | y<t))

in nats per sequence. It is unbiased for the sequence KL, like the plain
estimator sum_t [log p(y_t) - log q(y_t)], but with far lower variance
because the inner expectation is computed rather than sampled. The plain
estimator on 16 prompts gave a standard error of 7 nats on a mean of 3 in
experiment 0113's first canary, too noisy to place KL milestones.
"""

import math


def _completion_log_dists(model, input_ids, attention_mask, prompt_len: int):
    """Next-token log-distributions at every completion position of
    input_ids[:, prompt_len:] under `model`, shape (batch, completion, vocab).
    Position ids come from the attention mask so left padding does not shift
    positions. Only the completion positions' logits are materialised
    (`logits_to_keep`): with Qwen's 152k-token vocabulary, full-sequence fp32
    logits for a batch took 5 GB and ran an L4 out of memory (experiment 0113)."""
    import torch

    n_completion = input_ids.shape[1] - prompt_len
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logits_to_keep=n_completion + 1,
    ).logits
    return torch.log_softmax(logits[:, :-1].float(), dim=-1)


def sequence_kl(
    policy,
    ref,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: str,
    seed: int = 0,
    batch_size: int = 4,
) -> dict:
    """Estimate of KL(policy || ref) per sequence over `prompts` (already
    rendered). Returns {"kl", "sem", "n"}."""
    import torch

    was_training = policy.training
    policy.eval()
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    gen = torch.Generator(device="cpu").manual_seed(seed)
    per_seq: list[float] = []
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(
                device
            )
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
            with torch.no_grad():
                out = policy.generate(
                    **enc,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
                prompt_len = enc["input_ids"].shape[1]
                completion = out[:, prompt_len:]
                # tokens after the first eos are padding, not part of y
                is_eos = completion == tokenizer.eos_token_id
                after_eos = (is_eos.long().cumsum(-1) - is_eos.long()) > 0
                comp_mask = ~after_eos
                attn = torch.cat([enc["attention_mask"], comp_mask.long()], dim=1)
                lp = _completion_log_dists(policy, out, attn, prompt_len)
                lq = _completion_log_dists(ref, out, attn, prompt_len)
                per_token = (lp.exp() * (lp - lq)).sum(-1)  # exact KL per position
                per_seq += (per_token * comp_mask).sum(-1).tolist()
                del out, lp, lq, per_token
    finally:
        tokenizer.padding_side = side
        if device == "cuda":
            torch.cuda.empty_cache()
        if was_training:
            policy.train()
    n = len(per_seq)
    mean = sum(per_seq) / n
    var = sum((x - mean) ** 2 for x in per_seq) / max(1, n - 1)
    return {"kl": mean, "sem": math.sqrt(var / n), "n": n}


def crossed(milestones, saved: set, kl: float) -> list[float]:
    """Milestones reached for the first time at this KL, lowest first."""
    return [m for m in sorted(milestones) if m not in saved and kl >= m]


def make_kl_callback(
    *,
    ref,
    tokenizer,
    prompts: list[str],
    ladder,
    max_new_tokens: int,
    device: str,
    seed: int,
    on_eval,
    on_milestone,
    reached=(),
    measure_at_begin=True,
):
    """A transformers TrainerCallback that estimates KL every
    `ladder.eval_every` steps (and at steps 0 and the last one), calls
    `on_eval(step, est)` each time and `on_milestone(milestone, model, step,
    est)` once per milestone crossed. A resumed run passes the milestones
    already `reached` and skips the measurement at train begin, which would
    otherwise repeat the one taken at the resumed step."""
    from transformers import TrainerCallback

    class KLMilestoneCallback(TrainerCallback):
        def __init__(self):
            self.saved: set[float] = set(reached)

        def _measure(self, model, step):
            est = sequence_kl(
                model, ref, tokenizer, prompts, max_new_tokens, device, seed=seed + step
            )
            on_eval(step, est)
            for m in crossed(ladder.milestones, self.saved, est["kl"]):
                self.saved.add(m)
                on_milestone(m, model, step, est)

        def on_train_begin(self, args, state, control, model=None, **kw):
            if measure_at_begin:
                self._measure(model, 0)

        def on_step_end(self, args, state, control, model=None, **kw):
            step = state.global_step
            if step % ladder.eval_every == 0 or step >= state.max_steps:
                self._measure(model, step)

    return KLMilestoneCallback()
