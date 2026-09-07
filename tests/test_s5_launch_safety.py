from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manage_performance_recovery", ROOT / "scripts/manage_performance_recovery.py"
)
assert SPEC is not None and SPEC.loader is not None
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


def _registry() -> dict:
    return {
        "screening": {
            "measurement_safety_factor": 1.15,
            "estimated_fixed_overhead_seconds": 323.0,
            "maximum_examples_processed": 12_798_154,
            "maximum_steps": 50_000,
            "seed": 42,
            "split_seed": 42,
        }
    }


def _measurement(windows_per_second: float = 150.0) -> dict:
    registry = _registry()
    estimate, overhead, safety, examples = manager._derived_measurement_gpu_hours(
        windows_per_second, registry
    )
    return {
        "windows_per_second": windows_per_second,
        "maximum_examples_processed": examples,
        "estimated_fixed_overhead_seconds": overhead,
        "safety_factor": safety,
        "estimated_gpu_hours": estimate,
    }


def test_s5_budget_is_derived_and_capped() -> None:
    candidate = {"maximum_estimated_physical_gpu_hours": 29.1}
    measurement = _measurement()
    assert manager._validate_s5_measurement_budget(
        candidate, measurement, _registry()
    ) == pytest.approx(measurement["estimated_gpu_hours"])
    measurement["estimated_gpu_hours"] += 0.1
    with pytest.raises(manager.GateError, match="immutably derived"):
        manager._validate_s5_measurement_budget(candidate, measurement, _registry())

    too_slow = _measurement(100.0)
    assert too_slow["estimated_gpu_hours"] > 29.1
    with pytest.raises(manager.GateError, match="exceeds"):
        manager._validate_s5_measurement_budget(candidate, too_slow, _registry())


def test_cuda_visible_devices_resolves_physical_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager, "_gpu_inventory", lambda: {0: "GPU-aaaa", 1: "GPU-bbbb"})
    assert manager._resolve_visible_physical_gpu_uuid("1,0", 0) == "GPU-bbbb"
    assert manager._resolve_visible_physical_gpu_uuid("GPU-bbbb", 0) == "GPU-bbbb"
    with pytest.raises(manager.GateError):
        manager._resolve_visible_physical_gpu_uuid("0", 1)


def test_first_launch_and_nonadjacent_resume_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = ROOT / "checkpoints/performance-recovery/screens/S5/student-resume.pt"
    monkeypatch.setattr(manager, "_sha256", lambda _: "a" * 64)
    with pytest.raises(manager.GateError, match="first"):
        manager._verify_resume(resume, {"attempts": []}, _registry(), {})

    state = {
        "attempts": [
            {
                "artifacts_at_exit": {
                    "resume_checkpoint": {"path": manager._relative(resume), "sha256": "a" * 64}
                }
            },
            {"artifacts_at_exit": {}},
        ]
    }
    with pytest.raises(manager.GateError, match="immediately preceding"):
        manager._verify_resume(resume, state, _registry(), {})


def test_s5_resume_recipe_and_origin_are_both_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = ROOT / "checkpoints/performance-recovery/screens/S5/student-resume.pt"
    digest = "a" * 64
    state = {
        "attempts": [
            {
                "artifacts_at_exit": {
                    "resume_checkpoint": {"path": manager._relative(resume), "sha256": digest}
                }
            }
        ]
    }
    registry = _registry()
    candidate = {
        "maximum_estimated_physical_gpu_hours": 29.1,
        "initialization": {
            "kind": "seeded_random",
            "state_sha256": manager.S5_INITIALIZATION_STATE_SHA256,
        },
    }
    expected_origin, origin_digest = manager._s5_initialization_origin(candidate)
    payload = {
        "step": 1,
        "training_seed": 42,
        "split_seed": 42,
        "training_recipe_fingerprint": {"recipe": "tampered"},
        "training_recipe_sha256": "bad",
        "initialization_origin_fingerprint": expected_origin,
        "initialization_origin_sha256": origin_digest,
    }
    monkeypatch.setattr(manager, "_sha256", lambda _: digest)
    monkeypatch.setattr(manager.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(manager, "_s5_training_recipe", lambda *args: ({"recipe": "exact"}, "ok"))
    with pytest.raises(manager.GateError, match="training recipe"):
        manager._verify_resume(resume, state, registry, candidate)


def test_s5_autotune_accounting_is_exact() -> None:
    measurement = {
        "source": "results/probe.json",
        "source_sha256": "b" * 64,
        "external_job_id": "probe",
        "physical_gpu": 1,
        "physical_gpu_uuid": "GPU-bbbb",
        "autotune_elapsed_seconds": 72.0,
    }
    job = {
        "job_id": "probe",
        "status": "completed",
        "artifact": measurement["source"],
        "artifact_sha256": measurement["source_sha256"],
        "physical_gpu_count": 1,
        "physical_gpu": 1,
        "physical_gpu_uuid": "GPU-bbbb",
        "elapsed_seconds": 72.0,
        "actual_gpu_hours": 72.0 / 3600.0,
    }
    manager._verify_s5_autotune_accounting({"external_jobs": [job]}, measurement)
    job["actual_gpu_hours"] = math.nextafter(job["actual_gpu_hours"], math.inf)
    with pytest.raises(manager.GateError, match="accounting"):
        manager._verify_s5_autotune_accounting({"external_jobs": [job]}, measurement)


def test_measure_commands_use_one_artifact_snapshot() -> None:
    for function in (manager._command_measure, manager._command_remeasure):
        source = inspect.getsource(function)
        assert source.count("_load_json_snapshot(artifact)") == 1
        assert "_sha256(artifact)" not in source
