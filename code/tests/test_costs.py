"""The stage timer writes cost.json, on success and on failure."""

from pipeline.costs import stage_timer
from pipeline.exec.local_backend import LocalArtifactStore


def test_stage_timer_writes_cost_json(tmp_path):
    store = LocalArtifactStore(tmp_path, run_id="t")
    with stage_timer(store, "s"):
        pass
    cost = store.get_json(store.key("s", "cost.json"))
    assert cost["stage"] == "s" and cost["run_id"] == "t"
    assert cost["gpu_type"] == "none" and cost["gpu_seconds"] == 0.0
    assert cost["container_seconds"] >= 0.0


def test_stage_timer_attributes_gpu_seconds(tmp_path):
    store = LocalArtifactStore(tmp_path, run_id="t")
    with stage_timer(store, "s", gpu_type="L4"):
        pass
    cost = store.get_json(store.key("s", "cost.json"))
    assert cost["gpu_type"] == "L4"
    assert cost["gpu_seconds"] == cost["container_seconds"]


def test_stage_timer_writes_even_on_failure(tmp_path):
    store = LocalArtifactStore(tmp_path, run_id="t")
    try:
        with stage_timer(store, "s"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert store.exists(store.key("s", "cost.json"))
