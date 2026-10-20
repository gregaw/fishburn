"""Swappable execution layer. Import interfaces from here; backends are loaded
lazily via make_executor so `modal` is never imported as a side effect."""

from pipeline.exec.base import (
    FAILED,
    PENDING,
    RUNNING,
    SUCCEEDED,
    TERMINAL,
    ArtifactStore,
    Executor,
    StageHandle,
    StageStatus,
    make_executor,
)

__all__ = [
    "ArtifactStore",
    "Executor",
    "StageHandle",
    "StageStatus",
    "make_executor",
    "PENDING",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "TERMINAL",
]
