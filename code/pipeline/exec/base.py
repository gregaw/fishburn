"""Execution-layer interfaces: Executor and ArtifactStore.

Stages and orchestration depend only on what is defined in this module, so the
compute provider is swappable: `local_backend` runs everything in-process (what
the CPU e2e test and unit tests use), `modal_backend` wraps Modal remote
functions and volumes and is the only module in the package allowed to import
`modal` (test-enforced by tests/test_no_modal_imports.py). Switching to another
provider later means writing one new backend, not touching the stages.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# Terminal and non-terminal handle states, as plain strings so they serialize.
PENDING = "pending"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
TERMINAL = (SUCCEEDED, FAILED)


@dataclass(frozen=True)
class StageHandle:
    """Opaque reference to a dispatched stage. `handle_id` is backend-specific
    (a counter locally, a Modal function-call id remotely) and is what a human
    greps logs for."""

    handle_id: str
    stage: str
    backend: str


@dataclass
class StageStatus:
    state: str  # one of PENDING / RUNNING / SUCCEEDED / FAILED
    result: dict | None = None  # the stage's return dict, once SUCCEEDED
    error: str | None = None  # repr of the failure, once FAILED


class ArtifactStore(ABC):
    """Key-value artifact storage keyed `runs/<run_id>/<stage>/...`.

    Keys are forward-slash paths relative to the store root; `key()` builds
    them so stage code never assembles run paths by hand. JSON/JSONL helpers
    are implemented on top of put/get, so a backend only supplies bytes I/O.
    """

    def __init__(self, run_id: str = "default"):
        self.run_id = run_id

    def key(self, stage: str, *parts: str) -> str:
        return "/".join(["runs", self.run_id, stage, *parts])

    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Raise KeyError if the key is absent."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """All keys under `prefix`, sorted."""

    def exists(self, key: str) -> bool:
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    # -- files and durability (resume points, pipeline/resume.py) -----------

    def put_file(self, key: str, path) -> None:
        """Store a local file. Backends override this to avoid holding a
        multi-GB checkpoint in memory."""
        self.put(key, Path(path).read_bytes())

    def get_file(self, key: str, dest) -> None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.get(key))

    def delete(self, prefix: str) -> None:
        """Remove every key under `prefix` (a no-op if there are none)."""
        raise NotImplementedError

    def flush(self) -> None:
        """Make every write so far durable, for a backend that buffers them
        (Modal's Volume commits). A no-op by default."""

    # -- JSON/JSONL conveniences ------------------------------------------

    def put_json(self, key: str, obj) -> None:
        self.put(key, json.dumps(obj, indent=2, sort_keys=True).encode())

    def get_json(self, key: str):
        return json.loads(self.get(key))

    def append_jsonl(self, key: str, obj) -> None:
        # Read-modify-write: O(n^2) over a stage's lifetime, fine at pipeline
        # scale (tens of lines per stage); a backend may override with a real
        # append.
        try:
            existing = self.get(key)
        except KeyError:
            existing = b""
        self.put(key, existing + json.dumps(obj, sort_keys=True).encode() + b"\n")

    def read_jsonl(self, key: str) -> list:
        return [json.loads(line) for line in self.get(key).decode().splitlines() if line.strip()]


class Executor(ABC):
    """run(stage, config) -> handle; status(handle) -> StageStatus.

    `config` is the loaded profile dict (pipeline.config.load_profile). The
    executor owns an ArtifactStore; stages receive it and never touch the
    backend directly.
    """

    def __init__(self, store: ArtifactStore):
        self.store = store

    @abstractmethod
    def run(self, stage: str, config: dict, force: bool = False) -> StageHandle:
        """Dispatch one stage. Raises ValueError for an unknown stage name;
        a stage that starts and then fails is reported via status(), not
        raised here."""

    @abstractmethod
    def status(self, handle: StageHandle) -> StageStatus: ...

    def wait(
        self, handle: StageHandle, timeout_s: float = 3600.0, poll_s: float = 2.0
    ) -> StageStatus:
        """Poll status() until the handle reaches a terminal state."""
        import time

        deadline = time.monotonic() + timeout_s
        while True:
            st = self.status(handle)
            if st.state in TERMINAL:
                return st
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"stage {handle.stage} ({handle.handle_id}) not terminal after {timeout_s}s"
                )
            time.sleep(poll_s)


def make_executor(backend: str, **kwargs) -> Executor:
    """Lazy factory so importing the exec package never pulls `modal`."""
    if backend == "local":
        from pipeline.exec.local_backend import LocalExecutor

        return LocalExecutor(**kwargs)
    if backend == "modal":
        from pipeline.exec.modal_backend import ModalExecutor

        return ModalExecutor(**kwargs)
    raise ValueError(f"unknown backend {backend!r}, expected 'local' or 'modal'")
