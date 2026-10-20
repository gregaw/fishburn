"""LocalExecutor: handles, failures reported through status(), the backend factory."""

import pytest

from pipeline.config import load_profile
from pipeline.exec.base import FAILED, SUCCEEDED, make_executor
from pipeline.exec.local_backend import LocalArtifactStore, LocalExecutor


@pytest.fixture
def executor(tmp_path):
    return LocalExecutor(LocalArtifactStore(tmp_path, run_id="t"))


def test_run_returns_terminal_succeeded_handle(executor):
    profile = load_profile("smoke")
    handle = executor.run("data_prep", profile)
    assert handle.backend == "local" and handle.stage == "data_prep"
    st = executor.status(handle)
    assert st.state == SUCCEEDED
    assert st.result["skipped"] is False
    assert executor.wait(handle).state == SUCCEEDED  # wait is a no-op when terminal


def test_unknown_stage_raises(executor):
    with pytest.raises(ValueError, match="unknown stage"):
        executor.run("nope", load_profile("smoke"))


def test_stage_failure_reported_via_status_not_raised(executor, monkeypatch):
    import pipeline.stages as stages

    def boom(store, profile, force=False):
        raise RuntimeError("boom")

    monkeypatch.setitem(stages.STAGES, "data_prep", boom)
    handle = executor.run("data_prep", load_profile("smoke"))
    st = executor.status(handle)
    assert st.state == FAILED and "boom" in st.error and st.result is None


def test_unknown_handle_raises(executor):
    from pipeline.exec.base import StageHandle

    with pytest.raises(KeyError):
        executor.status(StageHandle("nope", "data_prep", "local"))


def test_make_executor_factory(tmp_path):
    ex = make_executor("local", store=LocalArtifactStore(tmp_path, run_id="t"))
    assert isinstance(ex, LocalExecutor)
    with pytest.raises(ValueError, match="unknown backend"):
        make_executor("slurm")
