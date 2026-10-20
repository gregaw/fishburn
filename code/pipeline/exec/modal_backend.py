"""Modal backend: the only module in this package allowed to import `modal`
(tests/test_no_modal_imports.py checks it). Wraps remote functions and the
artifact Volume behind the Executor / ArtifactStore interfaces from
pipeline.exec.base.

Inside a container the stage code runs against a LocalArtifactStore rooted at
the Volume mount, the same code path the CPU e2e test exercises, so the only
Modal-specific logic here is dispatch, polling, and Volume commit/reload.

The offline tests never import this module (no `modal` in the test
environment, no credentials, no network), so a cloud dispatch is its only test.

Environment traps in proxied sandboxes: Modal's gRPC cannot cross an HTTP
CONNECT proxy (run with `env -u HTTPS_PROXY -u https_proxy`), Modal trusts only
certifi's CA store (append the gateway's CA bundle to it), and remember
`modal run` without `--detach` kills the orchestrator when the client exits.
"""

import os
from pathlib import Path

import modal

from pipeline.exec.base import (
    FAILED,
    RUNNING,
    SUCCEEDED,
    ArtifactStore,
    Executor,
    StageHandle,
    StageStatus,
)

ROOT = Path(__file__).resolve().parent.parent.parent  # nlhf-cot-monitor/

APP_NAME = "nlhf-cot-monitor"
VOLUME_NAME = "ncm-artifacts"
MOUNT = "/artifacts"

app = modal.App(APP_NAME)

# Resolved on the dispatching client and baked into the image env: containers
# have no git checkout, so without it manifests written there would record
# `git_sha: unknown`.
try:
    from pipeline.manifest import git_sha

    GIT_SHA = git_sha()
except Exception:
    GIT_SHA = "unknown"

GPU_TRAIN = os.environ.get("NCM_GPU", "L4")

# The training legs need the PairRM judge in-container (NashMP and Online
# DPO query it every step), so the one image carries the judge's requirements too.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # requirements-judge installs llm-blender from GitHub
    .pip_install_from_requirements(str(ROOT / "configs" / "requirements.txt"))
    .pip_install_from_requirements(str(ROOT / "configs" / "requirements-judge.txt"))
    # vLLM for the generation stages (experiment 0122)
    .pip_install_from_requirements(str(ROOT / "configs" / "requirements-vllm.txt"))
    .env(
        {
            "HF_HOME": f"{MOUNT}/hf_cache",  # shared cache, outside runs/, never run-scoped
            "NCM_GIT_SHA": GIT_SHA,
            # hf_xet keeps a log file open under HF_HOME after a download, and an
            # open file on the Volume makes the next stage's vol.reload() fail in a
            # reused container (experiment 0120, first pull of Qwen2.5-0.5B)
            "HF_HUB_DISABLE_XET": "1",
            # fragmentation left 6 GB reserved but unusable when experiment 0113
            # ran out of memory
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCM_GPU": GPU_TRAIN,
        }
    )
    .add_local_dir(str(ROOT / "configs"), remote_path="/root/configs")
    .add_local_python_source("pipeline")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUMES = {MOUNT: vol}


def _container_store(run_id: str):
    from pipeline.exec.local_backend import LocalArtifactStore

    class VolumeStore(LocalArtifactStore):
        def flush(self) -> None:
            # a resume point must survive the container dying mid-stage
            vol.commit()

    return VolumeStore(MOUNT, run_id=run_id)


def _run_in_container(stage: str, config: dict, run_id: str, force: bool) -> dict:
    """One stage, in-container, against the Volume: the same stage functions
    the local tests run."""
    from pipeline.stages import STAGES

    vol.reload()
    result = STAGES[stage](_container_store(run_id), config, force=force)
    vol.commit()
    return result


# A long online run must not be killed mid-way by the stage limit: the slowest
# online trainer we measured took ~105 s per step on an L4, so 128 steps needed
# ~3.8 h (experiment 0114).
STAGE_TIMEOUT_S = 6 * 3600


# single_use_containers: every stage gets a fresh process. A warm container
# kept the NashMP leg's colocated vLLM engine (and its process group) alive,
# and the next stage's own vLLM engines found 16 of 22 GB taken (experiment
# 0123); a container start costs well under a minute.
@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=STAGE_TIMEOUT_S,
    gpu=GPU_TRAIN,
    max_containers=4,
    single_use_containers=True,
)
def run_stage(stage: str, config: dict, run_id: str, force: bool) -> dict:
    return _run_in_container(stage, config, run_id, force)


# Stages that touch no model: a GPU would only bill for idling.
CPU_STAGES = ("data_prep", "report")


@app.function(image=image, volumes=VOLUMES, timeout=3600, cpu=2.0)
def run_stage_cpu(stage: str, config: dict, run_id: str, force: bool) -> dict:
    return _run_in_container(stage, config, run_id, force)


@app.function(image=image, timeout=12 * 3600, cpu=0.25)
def orchestrate_remote(config: dict, run_id: str, stage: str, force: bool) -> dict:
    """The orchestrator runs in the cloud too, so `modal run --detach`
    keeps dispatching stages after the client exits (a client-side loop would
    die with the client and silently stop the pipeline)."""
    from pipeline.orchestrate import run_pipeline

    return run_pipeline(
        ModalExecutor(run_id=run_id), config, stage=stage, force=force, timeout_s=STAGE_TIMEOUT_S
    )


@app.local_entrypoint()
def main(
    profile: str = "canary",
    run_id: str = "default",
    stage: str = "all",
    legs: str = "",
    force: bool = False,
    seed: int = -1,
    override: str = "",
):
    """Dispatch the pipeline:

        env -u HTTPS_PROXY -u https_proxy .venv/bin/modal run --detach \
            pipeline/exec/modal_backend.py --profile canary \
            --run-id dpo-canary --legs dpo

    `--legs` (comma-separated) narrows the profile's legs, e.g. to run one
    canary leg first and watch it. `--override` (semicolon-separated
    key.path=value) changes profile settings for this dispatch; give it its
    own --run-id, since stage outputs are keyed by stage name."""
    import json

    from pipeline.config import apply_override, load_profile

    config = load_profile(profile, seed=None if seed < 0 else seed)
    for spec in filter(None, (s.strip() for s in override.split(";"))):
        apply_override(config, spec)
    if os.environ.get("NCM_GPU"):
        config["gpu"] = GPU_TRAIN  # cost records name the GPU actually used
    if legs:
        config["legs"] = [leg.strip() for leg in legs.split(",") if leg.strip()]
    call = orchestrate_remote.spawn(config, run_id, stage, force)
    print(
        json.dumps(
            {
                "orchestrator_call": call.object_id,
                "run_id": run_id,
                "profile": profile,
                "legs": config["legs"],
                "git_sha": GIT_SHA,
            }
        )
    )


@app.function(image=image, volumes=VOLUMES, timeout=600)
def store_put(key: str, data: bytes, run_id: str) -> None:
    _container_store(run_id).put(key, data)
    vol.commit()


@app.function(image=image, volumes=VOLUMES, timeout=600)
def store_get(key: str, run_id: str) -> bytes:
    # Returned as bytes from a remote function because direct `modal volume get`
    # downloads via *.modal-storage.com, which egress allowlists usually deny.
    vol.reload()
    return _container_store(run_id).get(key)


@app.function(image=image, volumes=VOLUMES, timeout=600)
def store_list(prefix: str, run_id: str) -> list[str]:
    vol.reload()
    return _container_store(run_id).list(prefix)


class ModalArtifactStore(ArtifactStore):
    """Client-side view of the Volume, tunnelled through remote functions."""

    def put(self, key: str, data: bytes) -> None:
        store_put.remote(key, data, self.run_id)

    def get(self, key: str) -> bytes:
        try:
            return store_get.remote(key, self.run_id)
        except Exception as e:
            if "KeyError" in repr(e):
                raise KeyError(key) from e
            raise

    def list(self, prefix: str = "") -> list[str]:
        return store_list.remote(prefix, self.run_id)


class ModalExecutor(Executor):
    """Dispatches run_stage and polls the spawned FunctionCall. handle_id is
    the Modal function-call id: log it, it is how a person finds the run."""

    def __init__(self, store: ModalArtifactStore | None = None, run_id: str = "default"):
        super().__init__(store or ModalArtifactStore(run_id))

    def run(self, stage: str, config: dict, force: bool = False) -> StageHandle:
        from pipeline.stages import STAGES

        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}, expected one of {tuple(STAGES)}")
        fn = run_stage_cpu if stage in CPU_STAGES else run_stage
        call = fn.spawn(stage, config, self.store.run_id, force)
        return StageHandle(handle_id=call.object_id, stage=stage, backend="modal")

    def status(self, handle: StageHandle) -> StageStatus:
        call = modal.FunctionCall.from_id(handle.handle_id)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            return StageStatus(RUNNING)
        except Exception as e:
            return StageStatus(FAILED, error=repr(e))
        return StageStatus(SUCCEEDED, result=result)
