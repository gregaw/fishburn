"""NashMP leg (experiment 0123): Nash Mirror Prox (Tiapkin et al., 2025),
sample-based, on the full policy: the Maximal-Lottery-like NLHF leg. Its
opponents are ordinary policies, not Nash-MD's token-by-token geometric
mixture with the base, so its sampling can run in vLLM. TRL has no NashMP
trainer; this is a plain PyTorch loop, the same algorithm as experiment 0119
(NashMP with different preference models, a LoRA policy on the Jackpot colour
task), here with full fine-tuning like the other legs.

Each iteration k has two phases of `inner_steps` (T) optimizer steps:

1. half step: play against the anchor pi_k (the policy at the start of k);
2. full step: play against the policy at the end of phase 1 (pi_{k+1/2}),
   continuing from where phase 1 ended, still penalised towards pi_k.

In each step, for each prompt, the policy draws `n_samples` answers and the
opponent `n_opponent`; answer y_i scores o_i, its mean judge preference over
the opponent's answers. The policy ascends the REINFORCE estimate with the
advantage

    A_i = (o_i - 1/2) - beta log(pi(y_i) / pi_ref(y_i)) - (beta/eta) log(pi(y_i) / pi_k(y_i))

held fixed, i.e. it descends -(1/B) sum_i A_i log pi(y_i). The opponent is
fixed within a phase and equals the current policy at the phase's start, so
its answers for all T steps are drawn then, in one batch.

AdamW without momentum (beta1 = 0: momentum makes the policy orbit a cycle,
experiment 0086, Online IPO on the Jackpot colour task); the learning rate
decays linearly to zero over the last `lr_decay_fraction` of steps. Training
settings come from the leg's train config:

| key | default | meaning |
| --- | --- | --- |
| `eta` | 2 * beta | proximal step size (Tiapkin et al.: last iterate converges for eta <= 2 beta) |
| `inner_steps` | 4 | T, optimizer steps per phase |
| `n_samples` / `n_opponent` | 2 / 2 | answers per prompt from policy / opponent |
| `sampler` | `hf` | `hf` (transformers generate) or `vllm` (colocated, synced every step) |
| `vllm_gpu_memory_utilization` | 0.2 | the engine's share of the GPU |
| `lr_decay_fraction` | 0.25 | |
| `micro_batch` | 2 | sequences per forward pass when scoring log-probabilities |
"""

import copy
import os
import random

TRAINER = "pipeline.legs.nashmp (NashMP, Tiapkin et al. 2025)"


# ---------------------------------------------------------------- samplers


class HFSampler:
    """transformers generate, the CPU path and the fallback."""

    def __init__(self, tokenizer, sampling, device: str, batch_size: int = 16):
        self.tok, self.s, self.device, self.bs = tokenizer, sampling, device, batch_size

    def sync(self, model):
        self.model = model

    def close(self):
        self.model = None

    def sample(self, prompts: list[str], n: int, seed: int) -> list[list[list[int]]]:
        """`n` completions (token ids, ending at EOS when one was produced)
        per prompt."""
        import torch

        torch.manual_seed(seed)
        model, tok = self.model, self.tok
        was_training = model.training
        model.eval()
        side, tok.padding_side = tok.padding_side, "left"
        out: list[list[list[int]]] = []
        try:
            per = max(1, self.bs // n)
            for i in range(0, len(prompts), per):
                enc = tok(
                    prompts[i : i + per],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(self.device)
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        do_sample=True,
                        temperature=self.s.temperature,
                        top_p=self.s.top_p,
                        max_new_tokens=self.s.max_new_tokens,
                        num_return_sequences=n,
                        pad_token_id=tok.pad_token_id,
                        use_cache=True,
                    )
                comp = gen[:, enc["input_ids"].shape[1] :].tolist()
                for j in range(0, len(comp), n):
                    out.append(
                        [_cut(c, tok.eos_token_id, tok.pad_token_id) for c in comp[j : j + n]]
                    )
        finally:
            tok.padding_side = side
            model.train(was_training)
        return out


DIST_ENV = {
    "RANK": "0",
    "LOCAL_RANK": "0",
    "WORLD_SIZE": "1",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29517",
}


class VLLMSampler:
    """A vLLM engine in this process, beside the policy. `sync` copies the
    policy's current weights into it (what TRL's colocate mode does), so every
    step samples the policy as it is now."""

    def __init__(
        self, model_dir: str, sampling, seed: int, gpu_memory_utilization: float, max_model_len: int
    ):
        # what external_launcher (an in-process engine) needs; removed again in
        # close(), since a warm container reuses this process for later stages
        self._env = {k: v for k, v in DIST_ENV.items() if k not in os.environ}
        os.environ.update(self._env)
        from vllm import LLM

        self.s = sampling
        self.llm = LLM(
            model=model_dir,
            dtype="bfloat16",
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            max_model_len=max_model_len,
            distributed_executor_backend="external_launcher",
            max_num_batched_tokens=4096,
        )

    def close(self):
        """Free the engine's GPU memory and its process group. Without this the
        next stage in the same container (the sanity check's own vLLM engines)
        found 16 of 22 GB still taken (experiment 0123, first canary)."""
        import gc

        import torch

        del self.llm
        gc.collect()
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        for k in self._env:
            os.environ.pop(k, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def sync(self, model):
        engine_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        engine_model.load_weights((name, p.data) for name, p in model.named_parameters())

    def sample(self, prompts: list[str], n: int, seed: int) -> list[list[list[int]]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=self.s.temperature,
            top_p=self.s.top_p,
            max_tokens=self.s.max_new_tokens,
            seed=seed,
        )
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        return [[list(o.token_ids) for o in r.outputs] for r in outs]


def _cut(ids: list[int], eos: int | None, pad: int | None) -> list[int]:
    """Completion ids up to and including the first EOS, without padding."""
    out = []
    for t in ids:
        if eos is not None and t == eos:
            out.append(t)
            break
        if pad is not None and t == pad and pad != eos:
            break
        out.append(t)
    return out


def build_sampler(kind: str, policy, tokenizer, shared, tcfg: dict, output_dir: str):
    if kind == "hf":
        return HFSampler(tokenizer, shared.sampling, policy.device.type)
    if kind == "vllm":
        from pipeline.generate import MAX_MODEL_LEN

        d = os.path.join(output_dir, "vllm-init")
        policy.save_pretrained(d, safe_serialization=True)
        tokenizer.save_pretrained(d)
        return VLLMSampler(
            d,
            shared.sampling,
            tcfg["seed"],
            tcfg.get("vllm_gpu_memory_utilization", 0.2),
            MAX_MODEL_LEN,
        )
    raise ValueError(f"unknown NashMP sampler {kind!r}, expected 'hf' or 'vllm'")


# ---------------------------------------------------------------- the update


def sequence_logprobs(
    model,
    prompt_ids: list[list[int]],
    completions: list[list[int]],
    device: str,
    micro_batch: int,
    grad: bool = False,
):
    """Summed log-probability of each completion after its prompt, as a 1-D
    tensor. With `grad`, each micro-batch keeps its graph (the caller sums
    them into one loss); otherwise everything runs without gradients."""
    import torch

    sums = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, len(completions), micro_batch):
            ps, cs = prompt_ids[i : i + micro_batch], completions[i : i + micro_batch]
            seqs = [p + c for p, c in zip(ps, cs)]
            width = max(len(s) for s in seqs)
            ids = torch.zeros(len(seqs), width, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            target_mask = torch.zeros(len(seqs), width, dtype=torch.bool, device=device)
            for j, (p, s) in enumerate(zip(ps, seqs)):
                ids[j, : len(s)] = torch.tensor(s, device=device)
                mask[j, : len(s)] = 1
                target_mask[j, len(p) : len(s)] = True
            with _autocast(device):
                logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
            lp = torch.log_softmax(logits.float(), dim=-1)
            tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            sums.append((tok_lp * target_mask[:, 1:]).sum(-1))
    return torch.cat(sums)


def _autocast(device: str):
    import contextlib

    import torch

    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def advantages(win_rate, lp, lp_ref, lp_anchor, beta: float, eta: float):
    """A_i = (o_i - 1/2) - beta log(pi/pi_ref) - (beta/eta) log(pi/pi_k)."""
    return [
        (o - 0.5) - beta * (a - r) - (beta / eta) * (a - k)
        for o, a, r, k in zip(win_rate, lp, lp_ref, lp_anchor, strict=True)
    ]


def lr_factor(step: int, total: int, decay_fraction: float) -> float:
    """Constant, then linear to zero over the last `decay_fraction` of steps."""
    start = int(total * (1 - decay_fraction))
    if step < start or total <= start:
        return 1.0
    return max(0.0, (total - step) / (total - start))


def win_rates(judge, prompts: list[str], ours: list[list[str]], theirs: list[list[str]]):
    """o_i per (prompt, own answer): mean judge P(own > opponent's) over the
    opponent's answers to the same prompt. One judge call for all pairs."""
    q_prompts, pairs, index = [], [], []
    for pi, (p, mine, opp) in enumerate(zip(prompts, ours, theirs)):
        for ai, a in enumerate(mine):
            for b in opp:
                q_prompts.append(p)
                pairs.append([a, b])
                index.append((pi, ai))
    probs = judge.judge(q_prompts, pairs, return_scores=True)
    acc: dict[tuple[int, int], list[float]] = {}
    for key, p in zip(index, probs):
        acc.setdefault(key, []).append(float(p))
    return [
        [sum(acc[(pi, ai)]) / len(acc[(pi, ai)]) for ai in range(len(mine))]
        for pi, mine in enumerate(ours)
    ]


def train(
    policy, ref, tokenizer, rows, judge, shared, tcfg, callbacks, output_dir, resume_from=None
):
    import torch
    from transformers import TrainerControl, TrainerState

    if tcfg.get("resume_every") or resume_from:
        raise ValueError("the NashMP leg has no resume points yet; keep runs under an hour")
    device = policy.device.type
    beta = shared.beta
    eta = tcfg.get("eta", 2 * beta)
    T = tcfg.get("inner_steps", 4)
    n_s, n_o = tcfg.get("n_samples", 2), tcfg.get("n_opponent", 2)
    per_step = tcfg["batch_size"] * tcfg.get("grad_accum", 1)  # prompts per step
    steps, micro = tcfg["steps"], tcfg.get("micro_batch", 2)

    prompts = [r["prompt"] for r in rows]
    rng = random.Random(tcfg["seed"])
    order: list[str] = []
    while len(order) < steps * per_step:
        block = prompts[:]
        rng.shuffle(block)
        order += block
    prompt_ids = {p: tokenizer(p, add_special_tokens=False).input_ids for p in set(order)}

    policy.train()
    if tcfg.get("gradient_checkpointing", True) and hasattr(
        policy, "gradient_checkpointing_enable"
    ):
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy.config.use_cache = False
    anchor = copy.deepcopy(policy).eval()
    if device == "cuda":  # only scores log-probabilities, under bf16 autocast anyway
        anchor.to(torch.bfloat16)
    for p in anchor.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(
        policy.parameters(), lr=tcfg["learning_rate"], betas=(0.0, 0.999), weight_decay=0.0
    )
    sampler = build_sampler(tcfg.get("sampler", "hf"), policy, tokenizer, shared, tcfg, output_dir)

    state, control = TrainerState(), TrainerControl()
    state.max_steps = steps
    for cb in callbacks:
        cb.on_train_begin(None, state, control, model=policy)

    history: list[dict] = []
    opponent: list[list[list[int]]] = []
    for step in range(steps):
        phase_pos = step % (2 * T)
        batch = order[step * per_step : (step + 1) * per_step]
        sampler.sync(policy)
        if phase_pos == 0:  # a new iteration: the anchor pi_k is the policy now
            anchor.load_state_dict(policy.state_dict())
        if phase_pos % T == 0:  # a new phase: its opponent is the policy now
            upcoming = order[step * per_step : (step + T) * per_step]
            opponent = sampler.sample(upcoming, n_o, seed=tcfg["seed"] * 100003 + step)
        opp = opponent[(phase_pos % T) * per_step : (phase_pos % T + 1) * per_step]
        ours = sampler.sample(batch, n_s, seed=tcfg["seed"] * 100003 + steps + step)

        dec = lambda c: tokenizer.decode(c, skip_special_tokens=True).strip()  # noqa: E731
        o = win_rates(
            judge, batch, [[dec(c) for c in g] for g in ours], [[dec(c) for c in g] for g in opp]
        )
        flat_p = [prompt_ids[p] for p, g in zip(batch, ours) for _ in g]
        flat_c = [c for g in ours for c in g]
        flat_o = [x for g in o for x in g]

        lp_ref = sequence_logprobs(ref, flat_p, flat_c, device, micro).tolist()
        lp_anchor = sequence_logprobs(anchor, flat_p, flat_c, device, micro).tolist()
        lp_now = sequence_logprobs(policy, flat_p, flat_c, device, micro).tolist()
        adv = advantages(flat_o, lp_now, lp_ref, lp_anchor, beta, eta)

        lr = tcfg["learning_rate"] * lr_factor(step, steps, tcfg.get("lr_decay_fraction", 0.25))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for i in range(0, len(flat_c), micro):
            lp = sequence_logprobs(
                policy, flat_p[i : i + micro], flat_c[i : i + micro], device, micro, grad=True
            )
            a = torch.tensor(adv[i : i + micro], device=lp.device)
            loss = -(a * lp).sum() / len(flat_c)
            loss.backward()
            loss_total += float(loss.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
        opt.step()

        n = len(flat_c)
        history.append(
            {
                "step": step + 1,
                "loss": loss_total,
                "phase": 1 if phase_pos < T else 2,
                "win_rate_vs_opponent": sum(flat_o) / n,
                "logratio_ref": sum(a - r for a, r in zip(lp_now, lp_ref)) / n,
                "logratio_anchor": sum(a - k for a, k in zip(lp_now, lp_anchor)) / n,
                "advantage_mean": sum(adv) / n,
                "completion_tokens": sum(len(c) for c in flat_c) / n,
                "grad_norm": grad_norm,
                "learning_rate": opt.param_groups[0]["lr"],
            }
        )
        state.global_step = step + 1
        state.log_history = history
        for cb in callbacks:
            cb.on_step_end(None, state, control, model=policy)

    if hasattr(policy, "gradient_checkpointing_disable"):
        policy.gradient_checkpointing_disable()
        policy.config.use_cache = True
    sampler.close()
    del anchor, opt
    return history
