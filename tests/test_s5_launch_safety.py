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
    monkeypatch.setattr(
        manager, "_load_torch_snapshot", lambda *args, **kwargs: (payload, digest, b"snapshot")
    )
    monkeypatch.setattr(manager, "_s5_training_recipe", lambda *args: ({"recipe": "exact"}, "ok"))
    with pytest.raises(manager.GateError, match="training recipe"):
        manager._verify_resume(resume, state, registry, candidate)


def test_s5_autotune_accounting_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
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
        "category": manager.S5_AUTOTUNE_CATEGORY,
        "status": "completed",
        "outcome": "succeeded",
        "artifact": measurement["source"],
        "artifact_sha256": measurement["source_sha256"],
        "physical_gpu_count": 1,
        "physical_gpu": 1,
        "physical_gpu_uuid": "GPU-bbbb",
        "elapsed_seconds": 72.0,
        "actual_gpu_hours": 72.0 / 3600.0,
    }
    monkeypatch.setattr(manager, "_validate_s5_autotune_job", lambda _: None)
    manager._verify_s5_autotune_accounting({"external_jobs": [job]}, measurement)
    job["actual_gpu_hours"] = math.nextafter(job["actual_gpu_hours"], math.inf)
    with pytest.raises(manager.GateError, match="accounting"):
        manager._verify_s5_autotune_accounting({"external_jobs": [job]}, measurement)


def test_measure_commands_use_one_artifact_snapshot() -> None:
    for function in (manager._command_measure, manager._command_remeasure):
        source = inspect.getsource(function)
        assert source.count("_load_json_snapshot(artifact)") == 1
        assert "_sha256(artifact)" not in source


def test_external_accounting_rejects_nonpositive_values() -> None:
    for value in (0.0, -1.0, math.inf, math.nan):
        ledger = {
            "external_jobs": [
                {
                    "job_id": "bad",
                    "category": "legacy",
                    "status": "completed",
                    "estimated_gpu_hours": value,
                    "actual_gpu_hours": 1.0,
                }
            ]
        }
        with pytest.raises(manager.GateError, match="finite and positive"):
            manager._validate_external_jobs(ledger)


def test_autotune_is_preregistered_before_process_start() -> None:
    source = inspect.getsource(manager._command_autotune)
    assert source.index('"status": "launching"') < source.index("subprocess.Popen")
    assert source.index("_write_ledger(args.ledger, ledger, registry)") < source.index(
        "subprocess.Popen"
    )
    assert '_atomic_json_new(paths["launch"]' in source


def test_remeasure_is_explicitly_fail_closed() -> None:
    with pytest.raises(manager.GateError, match="disabled fail-closed"):
        manager._command_remeasure(object())


def test_resume_command_carries_adjacent_digest() -> None:
    source = inspect.getsource(manager._build_training_command)
    assert '"--resume", _relative(resume), "--expected-resume-sha256", resume_sha256' in source


def test_resume_verification_uses_one_byte_snapshot() -> None:
    source = inspect.getsource(manager._verify_resume)
    assert source.count("_load_torch_snapshot(resume)") == 1
    assert "_sha256(resume)" not in source
    assert "resume.read_bytes()" not in source
    trainer = (ROOT / "scripts/train_production_student.py").read_text()
    assert "resume_snapshot = args.resume.read_bytes()" in trainer
    assert "torch.load(io.BytesIO(resume_snapshot)" in trainer


def test_s5_launch_revalidates_exact_code_artifact() -> None:
    source = inspect.getsource(manager._prepare_launch)
    assert "_revalidate_s5_measurement_for_launch(" in source
    verifier = inspect.getsource(manager._verify_s5_throughput_artifact)
    assert "_verify_source_map_at_commit(expected_source_map, measured_commit)" in verifier
    assert "_require_paths_clean" in verifier


def test_measured_source_map_rejects_git_blob_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager, "_require_full_git_oid", lambda value, _: value)
    monkeypatch.setattr(manager, "_git_blob", lambda *args: ("oid", b"changed"))
    with pytest.raises(manager.GateError, match="differs from its Git blob"):
        manager._verify_source_map_at_commit({"src/timesfm_lab/config.py": "0" * 64}, "a" * 40)


def test_autotune_gpu_path_requires_live_manager_authorization() -> None:
    source = (ROOT / "scripts/autotune_s5_domain_balanced_training.py").read_text()
    assert "_validate_manager_authorization(" in source
    assert "GPU autotune is manager-only" in source
    assert "_write_json_new(output, record.to_dict())" in source
    assert "_assert_import_origins()" in source


def test_dead_autotune_without_worker_record_becomes_unreconciled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = {
        "job_id": "S5-exact-throughput-autotune-attempt01",
        "category": manager.S5_AUTOTUNE_CATEGORY,
        "status": "running",
        "worker_record": "results/reproduction/systems/missing-worker.json",
        "pid": 123,
        "process_start_ticks": "456",
    }
    monkeypatch.setattr(manager, "_process_alive", lambda *args: False)
    manager._reconcile_s5_autotune_jobs({"external_jobs": [job]})
    assert job["status"] == "unreconciled"
    assert "without immutable terminal evidence" in job["failure"]


def test_s5_terminal_cost_is_derived_from_worker_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = {
        "schema_version": 1,
        "kind": manager.S5_AUTOTUNE_CATEGORY,
        "job_id": "probe",
        "attempt": 1,
        "git_commit": "a" * 40,
        "output": "output.json",
        "worker_record": "worker.json",
        "physical_gpu": 0,
        "reserved_gpu_hours": 1.0,
    }
    worker = {
        "job_id": "probe",
        "attempt": 1,
        "launch_record": {"path": "launch.json", "sha256": "1" * 64},
        "physical_gpu": 0,
        "physical_gpu_uuid": "GPU-a",
        "elapsed_seconds": 72.0,
        "ended_at": "done",
        "exit_code": 0,
        "artifact_at_exit": {"path": "output.json", "sha256": "3" * 64},
    }
    snapshots = {"launch.json": (launch, "1" * 64), "worker.json": (worker, "2" * 64)}
    monkeypatch.setattr(manager, "_root_path", lambda value: Path(str(value)))
    monkeypatch.setattr(manager, "_load_json_snapshot", lambda path: snapshots[str(path)])
    monkeypatch.setattr(manager, "_verify_file", lambda *args: None)
    job = {
        "job_id": "probe",
        "category": manager.S5_AUTOTUNE_CATEGORY,
        "status": "completed",
        "outcome": "succeeded",
        "attempt": 1,
        "git_commit": "a" * 40,
        "estimated_gpu_hours": 1.0,
        "actual_gpu_hours": 72.0 / 3600.0,
        "elapsed_seconds": 72.0,
        "ended_at": "done",
        "exit_code": 0,
        "physical_gpu_count": 1,
        "physical_gpu": 0,
        "physical_gpu_uuid": "GPU-a",
        "launch_record": {"path": "launch.json", "sha256": "1" * 64},
        "worker_record": "worker.json",
        "terminal_record": {"path": "worker.json", "sha256": "2" * 64},
        "artifact": "output.json",
        "artifact_sha256": "3" * 64,
    }
    manager._validate_s5_autotune_job(job)
    job["actual_gpu_hours"] = math.nextafter(job["actual_gpu_hours"], math.inf)
    with pytest.raises(manager.GateError, match="derive from its worker"):
        manager._validate_s5_autotune_job(job)
