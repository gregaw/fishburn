"""Sampling completions for a set of rendered prompts, with transformers or
with vLLM (experiment 0122).

The `hf` backend is `model.generate` in batches of 8, the default and what
the CPU tests run. The `vllm` backend writes the model to
a directory and samples it with vLLM, which batches the whole prompt set with
paged attention; it is the cloud path for the generation-heavy stages. Both
backends sample with the shared temperature, top-p and token budget, so the
two models a sanity check compares are always sampled the same way.

vLLM is imported lazily: it exists only in the cloud image
(configs/requirements-vllm.txt).
"""

import gc
import os
import tempfile

BACKENDS = ("hf", "vllm")


def sample_hf(
    model, tokenizer, rendered: list[str], sampling, seed: int, device: str, batch_size: int = 8
) -> list[str]:
    import torch

    torch.manual_seed(seed)
    texts = []
    for i in range(0, len(rendered), batch_size):
        enc = tokenizer(rendered[i : i + batch_size], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_new_tokens=sampling.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts += [
            t.strip()
            for t in tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
        ]
    return texts


# prompt + answer stay far below this (data_prep keeps prompts under 2,000
# characters); without it vLLM reserves KV cache for the model's full context
# (40,960 tokens for Qwen3) and a 1.7B model does not start on an L4
MAX_MODEL_LEN = 4096


def sample_vllm(
    model_dir: str, rendered: list[str], sampling, seed: int, gpu_memory_utilization: float = 0.4
) -> list[str]:
    """One vLLM engine over a saved model directory, released afterwards so a
    second model can be sampled in the same container."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
    )
    params = SamplingParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_new_tokens,
        seed=seed,
    )
    outs = llm.generate(rendered, params, use_tqdm=False)
    texts = [o.outputs[0].text.strip() for o in outs]
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return texts


def sample_models(
    models: list,
    tokenizer,
    rendered: list[str],
    sampling,
    seed: int,
    device: str,
    backend: str = "hf",
    gpu_memory_utilization: float = 0.4,
) -> list[list[str]]:
    """Completions from each model in `models`, all with the same seed. With
    vLLM every model is written to disk first and dropped from GPU memory, so
    the engines get the whole card."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown generation backend {backend!r}, expected {BACKENDS}")
    if backend == "hf":
        return [sample_hf(m, tokenizer, rendered, sampling, seed, device) for m in models]
    with tempfile.TemporaryDirectory() as tmp:
        dirs = []
        for i, m in enumerate(models):
            d = os.path.join(tmp, f"model{i}")
            m.save_pretrained(d, safe_serialization=True)
            tokenizer.save_pretrained(d)
            dirs.append(d)
        models.clear()  # the caller's list: drop the last references
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return [sample_vllm(d, rendered, sampling, seed, gpu_memory_utilization) for d in dirs]
