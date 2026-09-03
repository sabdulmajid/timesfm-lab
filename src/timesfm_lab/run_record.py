"""Machine-readable experiment provenance.

Run records deliberately contain metadata rather than large model outputs. They are written
atomically so an interrupted experiment cannot leave a valid-looking partial record.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class RunStatus(StrEnum):
    """Terminal and non-terminal states retained in the experiment ledger."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def current_git_commit(repository: Path | None = None) -> str:
    """Return the current commit, or ``UNCOMMITTED`` before repository initialization."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


def utc_now() -> str:
    """Return an unambiguous, second-resolution UTC timestamp."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class RunRecord:
    """Serializable provenance shared by evaluation, training, and systems experiments."""

    run_id: str
    config_path: str
    seed: int
    git_commit: str
    model_revision: str
    dataset_revision: str
    hardware_snapshot: str
    started_at: str
    status: RunStatus = RunStatus.RUNNING
    ended_at: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    failure: str | None = None
    log_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        config_path: str,
        seed: int,
        model_revision: str,
        dataset_revision: str,
        hardware_snapshot: str,
        repository: Path | None = None,
        log_path: str | None = None,
    ) -> RunRecord:
        """Create a running record populated with current Git and wall-clock provenance."""

        return cls(
            run_id=run_id,
            config_path=config_path,
            seed=seed,
            git_commit=current_git_commit(repository),
            model_revision=model_revision,
            dataset_revision=dataset_revision,
            hardware_snapshot=hardware_snapshot,
            started_at=utc_now(),
            log_path=log_path,
        )

    def succeed(self, metrics: dict[str, float]) -> None:
        """Finish successfully while preserving supplied aggregate metrics."""

        if self.status is not RunStatus.RUNNING:
            raise RuntimeError(f"run {self.run_id!r} is already terminal")
        self.metrics = dict(metrics)
        self.status = RunStatus.SUCCEEDED
        self.ended_at = utc_now()

    def fail(self, message: str) -> None:
        """Finish unsuccessfully; failure records are first-class experiment artifacts."""

        if self.status is not RunStatus.RUNNING:
            raise RuntimeError(f"run {self.run_id!r} is already terminal")
        self.failure = message
        self.status = RunStatus.FAILED
        self.ended_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to JSON-compatible primitives."""

        value = asdict(self)
        value["status"] = self.status.value
        return value

    def write(self, path: Path) -> None:
        """Atomically serialize this record as deterministic, human-readable JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
