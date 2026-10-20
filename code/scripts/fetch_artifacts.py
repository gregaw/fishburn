"""Pull run artifacts from the ncm-artifacts volume to a local directory.

`modal volume get` downloads via *.modal-storage.com, which egress allowlists
usually deny, so bytes come back through a remote function instead. Small
files only (under 10 MB, typically JSON and JSONL); relative paths are kept.

    env -u HTTPS_PROXY -u https_proxy .venv/bin/modal run scripts/fetch_artifacts.py \
        --run-id <run-id> --dest results/<run-id> \
        [--prefix judge_agreement | --prefix relabel/summary.json]
"""

from pathlib import Path

import modal

app = modal.App("ncm-fetch-artifacts")
MOUNT = "/artifacts"

image = modal.Image.debian_slim(python_version="3.11")
vol = modal.Volume.from_name("ncm-artifacts", create_if_missing=True)


@app.function(image=image, volumes={MOUNT: vol}, timeout=600)
def fetch(run_id: str, prefix: str) -> dict[str, bytes]:
    vol.reload()
    root = Path(MOUNT) / "runs" / run_id
    if prefix:
        root = root / prefix
    out = {}
    # a file prefix fetches that one file: large results come back through
    # blob storage on a host the allowlist may deny, so fetch only what you need
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for p in paths:
        if p.is_file() and p.stat().st_size < 10_000_000:
            out[str(p.relative_to(Path(MOUNT) / "runs" / run_id))] = p.read_bytes()
    return out


@app.local_entrypoint()
def main(run_id: str, dest: str = "results", prefix: str = ""):
    files = fetch.remote(run_id, prefix)
    for rel, data in files.items():
        target = Path(dest) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"{len(data):>10}  {target}")
    print(f"{len(files)} files -> {dest}")
