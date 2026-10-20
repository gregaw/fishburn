"""Model and tokenizer construction, including the credential-free tiny smoke model.

The smoke profile uses a tiny randomly initialised Qwen2 architecture (the
family of the real profiles' Qwen3 models) with a WordLevel tokenizer built
in-process, so the local e2e test needs no credentials and no network at all.
Even a one-time tokenizer download is a network dependency, and an egress
allowlist without huggingface.co would block the whole test.
"""

import io

TINY_SENTINEL = "tiny-random"

# Small enough that even random logits emit parseable words, so eval stubs
# reliably have something to score. Out-of-vocabulary prompt words map to
# <unk>, which is fine for a wiring test.
TINY_VOCAB_WORDS = [
    "the",
    "answer",
    "is",
    "yes",
    "no",
    "think",
    "step",
    "by",
    "so",
    "therefore",
    "first",
    "then",
    "question",
    "or",
    ".",
]


def _tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        **{w: i + 3 for i, w in enumerate(TINY_VOCAB_WORDS)},
    }
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )


def build_tokenizer(model_name: str):
    if model_name == TINY_SENTINEL:
        return _tiny_tokenizer()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tiny_qwen2_config(vocab_size: int):
    from transformers import Qwen2Config

    return Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
    )


def build_causal_lm(model_name: str, dtype: str, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM

    if model_name == TINY_SENTINEL:
        return AutoModelForCausalLM.from_config(tiny_qwen2_config(len(tokenizer)))
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype))


def save_state(model, dtype: str | None = None) -> bytes:
    """Checkpoint as bytes, so it travels through ArtifactStore.put and the
    execution layer stays a pure key-value interface. `dtype` casts floating
    tensors on the way out, e.g. "bfloat16" halves a 0.5B checkpoint;
    load_state casts back into whatever dtype the receiving model has."""
    import torch

    state = model.state_dict()
    if dtype is not None:
        target = getattr(torch, dtype)
        state = {k: v.to(target) if v.is_floating_point() else v for k, v in state.items()}
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def load_state(model, data: bytes):
    import torch

    model.load_state_dict(torch.load(io.BytesIO(data), weights_only=True))
    return model
