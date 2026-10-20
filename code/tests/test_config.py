"""Profile loading, the seed flag and `--override` key paths."""

import pytest

from pipeline.config import PROFILES, load_profile


def test_all_profiles_load_and_share_shape():
    for name in PROFILES:
        p = load_profile(name)
        assert p["profile"] == name
        for section in (
            "model",
            "data_prep",
            "judge",
            "shared",
            "legs",
            "train",
            "sanity",
            "eval_cot",
            "eval_steering",
        ):
            assert section in p, (name, section)
        assert p["model"]["name"]


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown profile"):
        load_profile("nope")


def test_smoke_is_offline_cpu():
    p = load_profile("smoke")
    assert p["model"]["name"] == "tiny-random"
    assert p["model"]["device"] == "cpu"
    assert p["gpu"] == "none"


def test_seed_override_feeds_data_prep():
    assert load_profile("smoke", seed=99)["data_prep"]["seed"] == 99
    assert load_profile("smoke")["data_prep"]["seed"] != 99


def test_apply_override_sets_nested_keys():
    import pytest

    from pipeline.config import apply_override, load_profile

    p = apply_override(load_profile("real"), "train.overrides.dpo.learning_rate=2e-6")
    assert p["train"]["overrides"]["dpo"]["learning_rate"] == 2e-6
    assert apply_override({}, "a.b=[1, 2]") == {"a": {"b": [1, 2]}}
    with pytest.raises(ValueError):
        apply_override({}, "no-equals-sign")


@pytest.mark.parametrize("name", ["canary", "real"])
def test_cloud_profiles_compare_online_dpo_with_nashmp_under_pairrm(name):
    p = load_profile(name)
    assert p["legs"] == ["dpo", "online_dpo", "nashmp"]
    assert p["judge"]["backend"] == "pairrm"
    o = p["train"]["overrides"]
    # the two compared legs differ only in their training rule
    for key in ("steps", "learning_rate", "sampler", "vllm_gpu_memory_utilization"):
        assert o["online_dpo"][key] == o["nashmp"][key], key
