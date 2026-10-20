"""The frozen config every Phase 1 leg imports (experiment 0110).

Same base model, same prompts, same judge, same sampling, same KL milestones:
the only thing allowed to differ between the DPO, Online DPO and NashMP legs
is the training rule itself. Every leg builds its config through
`load_shared(profile)` and its judge through `build_judge`, and writes
`shared_config_hash` into its manifest, so two runs whose hashes differ were
not compared under the same conditions.

What the judge sees (the call convention): the raw user prompt and the plain
completion text, never the chat-templated prompt. TRL renders conversational
prompts with its own template before calling a judge, so the legs hand TRL
prompts that are already rendered and the judge maps them back to the raw
prompt (`raw_prompt_lookup`).

Tie rule: a verdict of exactly 0.5 keeps the original label in relabelling,
counts as half a win in the sanity checks, and counts as a win for the first
completion when a trainer asks for a hard rank.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

CALL_CONVENTION = "raw-user-prompt, plain-completion, P(first > second)"
TIE_RULE = "p == 0.5: keep label (relabel), half a win (sanity), first wins (rank)"


@dataclass(frozen=True)
class JudgeSpec:
    backend: str
    model: str | None
    batch_size: int
    call_convention: str = CALL_CONVENTION
    tie_rule: str = TIE_RULE


@dataclass(frozen=True)
class Sampling:
    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class KLLadder:
    milestones: tuple[float, ...]
    eval_every: int
    n_prompts: int


@dataclass(frozen=True)
class SharedConfig:
    base_model: str
    dtype: str
    template: str  # "chat" (tokenizer's own template), "chat-nothink" or "plain"
    judge: JudgeSpec
    sampling: Sampling
    beta: float
    kl: KLLadder
    data_seed: int  # pins the prompt set together with data_prep's params
    data_source: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def hash(self) -> str:
        canon = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canon.encode()).hexdigest()[:16]


def load_shared(profile: dict) -> SharedConfig:
    s, o = profile["shared"], profile["judge"]
    return SharedConfig(
        base_model=profile["model"]["name"],
        dtype=profile["model"]["dtype"],
        template=s["template"],
        judge=JudgeSpec(
            backend=o["backend"], model=o.get("model"), batch_size=o.get("batch_size", 8)
        ),
        sampling=Sampling(**s["sampling"]),
        beta=float(s["beta"]),
        kl=KLLadder(
            milestones=tuple(float(m) for m in s["kl"]["milestones"]),
            eval_every=int(s["kl"]["eval_every"]),
            n_prompts=int(s["kl"]["n_prompts"]),
        ),
        data_seed=profile["data_prep"]["seed"],
        data_source=profile["data_prep"].get("source", "synthetic"),
    )


def with_judge(shared: SharedConfig, spec: dict) -> SharedConfig:
    """The same shared config with another judge, for second-opinion scoring
    (`sanity.extra_judges`); spec has the judge section's keys."""
    from dataclasses import replace

    return replace(
        shared,
        judge=JudgeSpec(
            backend=spec["backend"], model=spec.get("model"), batch_size=spec.get("batch_size", 8)
        ),
    )


def render_prompt(shared: SharedConfig, tokenizer, prompt: str) -> str:
    """The prompt string a leg's model is conditioned on."""
    if shared.template == "plain":
        return prompt
    if shared.template in ("chat", "chat-nothink"):
        # chat-nothink: Qwen3's template with its thinking mode off, which
        # prefills an empty <think></think> block (experiment 0122), so a
        # short sampling budget goes to the answer, not to a cut-off thought
        extra = {"enable_thinking": False} if shared.template == "chat-nothink" else {}
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **extra,
        )
    raise ValueError(f"unknown template {shared.template!r}")


def raw_prompt_lookup(shared: SharedConfig, tokenizer, prompts: list[str]):
    """rendered -> raw, so the judge always sees the raw prompt. Unknown
    strings pass through unchanged (the plain template renders to itself)."""
    table = {render_prompt(shared, tokenizer, p): p for p in prompts}
    return lambda rendered: table.get(rendered, rendered)


def build_judge(shared: SharedConfig, store, device: str | None = None):
    """The judge every leg and stage uses, with verdicts cached in the run's
    artifact store."""
    from pipeline.judge.base import CachedJudge, StubJudge

    if shared.judge.backend == "stub":
        backend = StubJudge()
    elif shared.judge.backend == "pairrm":
        from pipeline.judge.pairrm import PairRMJudge

        backend = PairRMJudge(
            model=shared.judge.model, device=device, batch_size=shared.judge.batch_size
        )
    else:
        raise ValueError(f"unknown judge backend {shared.judge.backend!r}")
    return CachedJudge(backend, store)
