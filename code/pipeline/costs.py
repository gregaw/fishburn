"""Per-stage duration accounting -> cost.json in the stage's run tree.

The cost hook every stage runs under so every run leaves a record of what it
consumed. cost.json holds durations and the GPU type; pricing them is left to
the caller.
"""

import time
from contextlib import contextmanager

from pipeline.exec.base import ArtifactStore

COST = "cost.json"


@contextmanager
def stage_timer(store: ArtifactStore, stage: str, gpu_type: str = "none"):
    t0 = time.monotonic()
    try:
        yield
    finally:
        seconds = round(time.monotonic() - t0, 1)
        store.put_json(
            store.key(stage, COST),
            {
                "stage": stage,
                "run_id": store.run_id,
                "gpu_type": gpu_type,
                "container_seconds": seconds,
                "gpu_seconds": seconds if gpu_type != "none" else 0.0,
            },
        )
