"""Judge agreement run (experiment 0111): PairRM against the
labels of HH-RLHF (human crowdworkers) and UltraFeedback (GPT-4), on 1k pairs
from each held-out test split (unseen) and 1k from each training split (seen).

PairRM's model card lists both corpora among its training data without
saying which splits, so training-split agreement may be inflated by memory;
the seen-minus-unseen gap per dataset measures that. UltraFeedback's test
split may not be clean either: the binarized test prompts come from the same
original release PairRM trained on.

The four measurements run in parallel L4 containers. Verdicts are cached on
the ncm-artifacts volume under runs/<run_id>/judge/cache/, the per-set
reports under runs/<run_id>/judge_agreement/, and the combined report is
printed and written beside them.

    env -u HTTPS_PROXY -u https_proxy .venv/bin/modal run scripts/judge_agreement.py \\
        --n 1000 --seed 7 [--sets ultrafeedback:test,hh_rlhf:test] [--run-id judge-agreement]

Modal glue only: the science is pipeline.judge.agreement and
pipeline.prefdata (unit-tested offline). Plain `modal run` is fine (no
--detach): the client waits for the four calls and nothing is orphaned.
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent  # nlhf-cot-monitor/

APP_NAME = "ncm-judge-agreement"
VOLUME_NAME = "ncm-artifacts"  # shared with pipeline/exec/modal_backend.py
MOUNT = "/artifacts"
DEFAULT_RUN_ID = "judge-agreement"
ALL_SETS = "ultrafeedback:test,ultrafeedback:train,hh_rlhf:test,hh_rlhf:train"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(ROOT / "configs" / "requirements.txt"))
    .pip_install_from_requirements(str(ROOT / "configs" / "requirements-judge.txt"))
    .env({"HF_HOME": f"{MOUNT}/hf_cache"})  # shared cache, outside runs/
    .add_local_python_source("pipeline")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, volumes={MOUNT: vol}, timeout=3600, gpu="L4")
def measure(source: str, split_kind: str, n: int, seed: int, run_id: str) -> dict:
    import torch
    import transformers

    from pipeline.costs import stage_timer
    from pipeline.exec.local_backend import LocalArtifactStore
    from pipeline.judge import CachedJudge, agreement_report
    from pipeline.judge.agreement import by_score_margin
    from pipeline.judge.pairrm import PairRMJudge
    from pipeline.prefdata import TEST_SPLITS, TRAIN_SPLITS, load_records

    split = (TEST_SPLITS if split_kind == "test" else TRAIN_SPLITS)[source]
    vol.reload()
    store = LocalArtifactStore(MOUNT, run_id=run_id)
    stage = f"judge_agreement/{source}-{split_kind}"
    with stage_timer(store, stage, gpu_type="L4"):
        records = load_records(source, n, seed, split=split)
        judge = CachedJudge(PairRMJudge(device="cuda"), store)
        report = agreement_report(records, judge)
        report["by_score_margin"] = by_score_margin(records, judge)
    report.update(
        source=source,
        split=split,
        split_kind=split_kind,
        n_requested=n,
        seed=seed,
        backend_calls=judge.backend_calls,
        label_kind="human" if source == "hh_rlhf" else "gpt-4",
        versions={"torch": torch.__version__, "transformers": transformers.__version__},
    )
    report["cost"] = store.get_json(store.key(stage, "cost.json"))
    store.put_json(store.key(stage, f"report_n{n}_seed{seed}.json"), report)
    vol.commit()
    return report


@app.function(image=image, volumes={MOUNT: vol}, timeout=600)
def save_combined(combined: dict, run_id: str, name: str) -> None:
    from pipeline.exec.local_backend import LocalArtifactStore

    store = LocalArtifactStore(MOUNT, run_id=run_id)
    store.put_json(store.key("judge_agreement", name), combined)
    vol.commit()


@app.local_entrypoint()
def main(n: int = 1000, seed: int = 7, sets: str = ALL_SETS, run_id: str = DEFAULT_RUN_ID):
    from pipeline.judge.agreement import seen_unseen_summary

    wanted = [tuple(s.split(":")) for s in sets.split(",") if s]
    calls = {w: measure.spawn(w[0], w[1], n, seed, run_id) for w in wanted}
    reports = {w: c.get() for w, c in calls.items()}
    combined = {
        "n": n,
        "seed": seed,
        "run_id": run_id,
        "summary": seen_unseen_summary(reports),
        "reports": {f"{s}:{k}": r for (s, k), r in reports.items()},
    }
    save_combined.remote(combined, run_id, f"combined_n{n}_seed{seed}.json")
    print(json.dumps(combined, indent=2))
