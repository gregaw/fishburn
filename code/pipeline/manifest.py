"""Stage completion manifests and skip logic, over the ArtifactStore.

A stage is complete iff its manifest.json exists with status=complete and a
params_hash matching the current parameters. The manifest is written last, on
success, so a crashed stage is simply re-run. Change a hyperparameter and only
the affected stages re-run.
"""

import hashlib
import json
import os
import subprocess
import time

from pipeline.exec.base import ArtifactStore

MANIFEST = "manifest.json"


def params_hash(params: dict) -> str:
    canon = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def git_sha() -> str:
    """NCM_GIT_SHA wins: remote containers have no checkout, so the client
    resolves the sha at dispatch time and bakes it into the image env."""
    env_sha = os.environ.get("NCM_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_manifest(
    store: ArtifactStore, stage: str, phash: str, outputs: list[str], extra: dict | None = None
) -> None:
    store.put_json(
        store.key(stage, MANIFEST),
        {
            "status": "complete",
            "params_hash": phash,
            "git_sha": git_sha(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "outputs": outputs,
            **(extra or {}),
        },
    )


def is_complete(store: ArtifactStore, stage: str, phash: str) -> bool:
    try:
        m = store.get_json(store.key(stage, MANIFEST))
    except (KeyError, json.JSONDecodeError):
        return False
    return m.get("status") == "complete" and m.get("params_hash") == phash


def load(store: ArtifactStore, stage: str) -> dict | None:
    try:
        return store.get_json(store.key(stage, MANIFEST))
    except (KeyError, json.JSONDecodeError):
        return None
