"""End-to-end pipeline on CPU with the smoke profile, through LocalExecutor:
tiny random Qwen2 + in-process tokenizer + stub judge, no GPU, no network, no
credentials. Every leg (DPO, Online DPO, NashMP) trains, is sanity-checked
and evaluated. It checks wiring, not results, and must pass before any cloud
run."""

import os

from pipeline import manifest
from pipeline.config import load_profile
from pipeline.exec.local_backend import LocalArtifactStore, LocalExecutor
from pipeline.legs import LEGS
from pipeline.orchestrate import format_status, run_pipeline, status
from pipeline.shared import load_shared
from pipeline.stages import stage_order

# Belt and braces: the smoke profile needs no network; make any accidental
# hub call fail fast instead of hanging on a blocked egress.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def test_e2e_smoke(tmp_path):
    profile = load_profile("smoke")
    order = stage_order(profile)
    assert set(profile["legs"]) == set(LEGS)  # the test covers every leg
    store = LocalArtifactStore(tmp_path, run_id="smoke")
    executor = LocalExecutor(store)

    results = run_pipeline(executor, profile)

    # every stage ran (not skipped) and left a complete manifest
    assert list(results) == list(order)
    for stage, res in results.items():
        assert res["skipped"] is False, stage
        assert manifest.load(store, stage) is not None, stage
    table = status(store, profile)
    assert all(r["done"] for r in table)
    assert format_status(table).endswith(f"{len(order)}/{len(order)} complete")

    # cost hook fired for every stage, all CPU (this test costs $0)
    for stage in order:
        cost = store.get_json(store.key(stage, "cost.json"))
        assert cost["gpu_seconds"] == 0.0, stage

    # relabel kept every pair and recorded the swapped fraction per split
    summary = store.get_json(store.key("relabel", "summary.json"))
    assert summary["splits"]["train"]["n_pairs"] == profile["data_prep"]["n_pairs"]
    assert summary["splits"]["test"]["n_pairs"] == profile["data_prep"]["n_test_pairs"]

    shared_hash = load_shared(profile).hash
    for leg in profile["legs"]:
        # each leg left a loadable checkpoint, TRL logs, KL estimates and
        # the same shared-config hash
        train = f"train_{leg}"
        assert store.exists(store.key(train, "checkpoint.pt"))
        rows = store.read_jsonl(store.key(train, "metrics.jsonl"))
        trl_steps = [r for r in rows if r["kind"] == "trl" and "loss" in r]
        assert len(trl_steps) == profile["train"]["steps"], leg
        assert [r["step"] for r in rows if r["kind"] == "kl"][0] == 0
        m = manifest.load(store, train)
        assert m["shared_config_hash"] == shared_hash, leg
        for ck in store.get_json(store.key(train, "checkpoints.json")):
            assert store.exists(ck["key"]), ck

        # sanity checks and the unjudged eval stubs wrote their artefacts
        sanity = store.get_json(store.key(f"sanity_{leg}", "sanity.json"))
        assert 0.0 <= sanity["checks"]["judge_preference"]["mean_p_trained_over_base"] <= 1.0
        samples = store.read_jsonl(store.key(f"sanity_{leg}", "samples.jsonl"))
        assert samples
        # the second-opinion judge is reported and kept per prompt
        assert "stub" in sanity["checks"]["other_judges"]
        assert all(0.0 <= r["p_stub"] <= 1.0 for r in samples)
        # the dataset check ran on the held-out labelled pairs (experiment 0123)
        pc = sanity["checks"]["preference_check"]
        assert pc["n_pairs"] == profile["sanity"]["pref_pairs"]
        assert 0.0 <= pc["accuracy_vs_dataset"] <= 1.0
        assert sanity["checks"]["onpolicy_check"]["n_pairs"] == profile["sanity"]["n_prompts"]
        cot = store.get_json(store.key(f"eval_cot_{leg}", "metrics.json"))
        assert cot["n"] == profile["eval_cot"]["n_samples"]
        steer = store.get_json(store.key(f"eval_steering_{leg}", "metrics.json"))
        assert -1.0 <= steer["steering_effect"] <= 1.0

    # report puts every leg side by side and records provenance
    report = store.get_json(store.key("report", "report.json"))
    assert report["run_id"] == "smoke"
    assert set(report["legs"]) == set(profile["legs"])
    assert set(report["upstream_params_hashes"]) == set(order) - {"report"}
    md = store.get(store.key("report", "report.md")).decode()
    assert all(f"`{leg}`" in md for leg in profile["legs"])

    # second run skips everything (params_hash skip logic)
    results2 = run_pipeline(executor, profile)
    for stage, res in results2.items():
        assert res["skipped"] is True, stage

    # --force re-executes
    res3 = run_pipeline(executor, profile, stage="data_prep", force=True)
    assert res3["data_prep"]["skipped"] is False
