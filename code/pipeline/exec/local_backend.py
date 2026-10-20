"""In-process backend: filesystem ArtifactStore + synchronous Executor.

What the CPU e2e test and unit tests use. run() executes the stage function
directly and records the outcome; status() replays it. Rooting the store at a
pytest tmp_path gives every test an isolated run tree with the same layout the
Modal volume uses (`runs/<run_id>/<stage>/...`).
"""

import shutil
from pathlib import Path

from pipeline.exec.base import (
    FAILED,
    SUCCEEDED,
    ArtifactStore,
    Executor,
    StageHandle,
    StageStatus,
)


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: str | Path, run_id: str = "default"):
        super().__init__(run_id)
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not p.is_relative_to(self.root.resolve()):
            raise ValueError(f"key {key!r} escapes the store root")
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key: str) -> bytes:
        p = self._path(key)
        if not p.is_file():
            raise KeyError(key)
        return p.read_bytes()

    def put_file(self, key: str, path) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, p)

    def get_file(self, key: str, dest) -> None:
        p = self._path(key)
        if not p.is_file():
            raise KeyError(key)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)

    def delete(self, prefix: str) -> None:
        p = self._path(prefix)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()

    def list(self, prefix: str = "") -> list[str]:
        base = self.root
        return sorted(
            str(p.relative_to(base))
            for p in base.rglob("*")
            if p.is_file() and str(p.relative_to(base)).startswith(prefix)
        )


class LocalExecutor(Executor):
    """Synchronous: the stage has already finished (or failed) by the time
    run() returns, so handles are terminal immediately."""

    def __init__(self, store: LocalArtifactStore):
        super().__init__(store)
        self._outcomes: dict[str, StageStatus] = {}
        self._counter = 0

    def run(self, stage: str, config: dict, force: bool = False) -> StageHandle:
        from pipeline.stages import STAGES

        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}, expected one of {tuple(STAGES)}")
        self._counter += 1
        handle = StageHandle(
            handle_id=f"local-{self._counter}-{stage}", stage=stage, backend="local"
        )
        try:
            result = STAGES[stage](self.store, config, force=force)
            self._outcomes[handle.handle_id] = StageStatus(SUCCEEDED, result=result)
        except Exception as e:  # surfaced via status(), matching remote semantics
            self._outcomes[handle.handle_id] = StageStatus(FAILED, error=repr(e))
        return handle

    def status(self, handle: StageHandle) -> StageStatus:
        if handle.handle_id not in self._outcomes:
            raise KeyError(f"unknown handle {handle.handle_id!r}")
        return self._outcomes[handle.handle_id]
