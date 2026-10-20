"""Pipeline wiring over the Executor interface, the same for every backend:
the e2e test runs this exact function through LocalExecutor, and a cloud
dispatch runs it through ModalExecutor. No Modal imports here.
"""

from pipeline import manifest
from pipeline.exec.base import SUCCEEDED, ArtifactStore, Executor
from pipeline.stages import ORDER, stage_order


def run_pipeline(
    executor: Executor,
    profile: dict,
    *,
    stage: str = "all",
    force: bool = False,
    timeout_s: float = 3600.0,
) -> dict:
    """Run the pipeline (or one stage) in stage_order(profile). Stages run one
    after another and the first failed stage raises, so a failure cannot hide
    behind stages that succeeded."""
    order = stage_order(profile)
    stages = order if stage == "all" else (stage,)
    unknown = set(stages) - set(order)
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)}, expected from {order}")

    results: dict = {}
    for s in stages:
        handle = executor.run(s, profile, force=force)
        st = executor.wait(handle, timeout_s=timeout_s)
        if st.state != SUCCEEDED:
            raise RuntimeError(f"stage {s} failed ({handle.handle_id}): {st.error}")
        results[s] = st.result
    return results


def status(store: ArtifactStore, profile: dict | None = None) -> list[dict]:
    """Completion table from manifests, for polling detached runs."""
    order = stage_order(profile) if profile is not None else ORDER
    return [{"stage": s, "done": manifest.load(store, s) is not None} for s in order]


def format_status(table: list[dict]) -> str:
    done = sum(r["done"] for r in table)
    lines = [f"  [{'x' if r['done'] else ' '}] {r['stage']}" for r in table]
    lines.append(f"{done}/{len(table)} complete")
    return "\n".join(lines)
