import json
from pathlib import Path

import pytest

from timesfm_lab.run_record import RunRecord, RunStatus


def make_record() -> RunRecord:
    return RunRecord(
        run_id="smoke-001",
        config_path="configs/reproduction/smoke.yaml",
        seed=7,
        git_commit="abc123",
        model_revision="teacher-revision",
        dataset_revision="dataset-revision",
        hardware_snapshot="results/environment/example.txt",
        started_at="2026-09-03T00:00:00+00:00",
    )


def test_failure_is_serialized_and_not_discarded(tmp_path: Path) -> None:
    record = make_record()
    record.fail("out of memory")
    destination = tmp_path / "nested" / "record.json"
    record.write(destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure"] == "out of memory"
    assert payload["ended_at"] is not None


def test_terminal_record_cannot_be_rewritten_as_success() -> None:
    record = make_record()
    record.fail("intentional test failure")

    with pytest.raises(RuntimeError, match="already terminal"):
        record.succeed({"mase": 1.0})


def test_success_copies_metrics() -> None:
    metrics = {"mase": 0.8}
    record = make_record()
    record.succeed(metrics)
    metrics["mase"] = 99.0

    assert record.status is RunStatus.SUCCEEDED
    assert record.metrics == {"mase": 0.8}
