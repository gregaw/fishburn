"""Profile loading: smoke / canary / real from configs/*.yaml.

- smoke:  tiny random Qwen2 on CPU, offline: what the e2e test runs.
- canary: Qwen3-0.6B on a small UltraFeedback slice: every leg for cents.
- real:   Qwen3-0.6B, the Phase 1 readiness setup with all three legs.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"

PROFILES = ("smoke", "canary", "real")


def load_profile(name: str = "smoke", seed: int | None = None) -> dict:
    """`seed` overrides the profile's data seed, making multi-seed sweeps a
    flag instead of a fork; it feeds every stage's params hash."""
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}, expected one of {PROFILES}")
    profile = yaml.safe_load((CONFIGS_DIR / f"{name}.yaml").read_text())
    profile["profile"] = name
    if seed is not None:
        profile["data_prep"]["seed"] = seed
    return profile


def apply_override(profile: dict, spec: str) -> dict:
    """`"train.overrides.dpo.learning_rate=2e-6"` -> set that key, creating
    missing levels; the value is parsed as YAML (numbers, lists, booleans).
    Lets a dispatch try one setting without forking a profile (experiment
    0122). The value lands in the profile, so it feeds the params hashes."""
    path, _, raw = spec.partition("=")
    if not path or not _:
        raise ValueError(f"override {spec!r} is not key.path=value")
    keys = path.split(".")
    node = profile
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML 1.1 reads "2e-6" (no dot) as a string
        try:
            value = float(value)
        except ValueError:
            pass
    node[keys[-1]] = value
    return profile
