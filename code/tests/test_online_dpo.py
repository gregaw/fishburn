"""The Online DPO leg's sampler switch: transformers generate by default,
a colocated vLLM engine on request."""

import pytest

from pipeline.legs.online_dpo import sampler_args


def test_default_sampler_adds_nothing():
    assert sampler_args({}) == {}
    assert sampler_args({"sampler": "hf"}) == {}


def test_vllm_sampler_colocates_with_the_given_memory_share():
    assert sampler_args({"sampler": "vllm", "vllm_gpu_memory_utilization": 0.3}) == {
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": 0.3,
    }
    assert sampler_args({"sampler": "vllm"})["vllm_gpu_memory_utilization"] == 0.2


def test_vllm_args_are_fields_of_the_pinned_config():
    from trl.experimental.online_dpo import OnlineDPOConfig

    fields = OnlineDPOConfig.__dataclass_fields__
    assert all(k in fields for k in sampler_args({"sampler": "vllm"}))


def test_unknown_sampler_is_an_error():
    with pytest.raises(ValueError, match="sampler"):
        sampler_args({"sampler": "sglang"})
