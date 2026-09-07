from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import math
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manage_performance_recovery", ROOT / "scripts/manage_performance_recovery.py"
)
assert SPEC is not None and SPEC.loader is not None
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)

TRAINER_SPEC = importlib.util.spec_from_file_location(
    "train_production_student_s5_safety", ROOT / "scripts/train_production_student.py"
)
assert TRAINER_SPEC is not None and TRAINER_SPEC.loader is not None
trainer = importlib.util.module_from_spec(TRAINER_SPEC)
sys.modules[TRAINER_SPEC.name] = trainer
TRAINER_SPEC.loader.exec_module(trainer)


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


def test_live_external_accounting_uses_larger_of_reservation_and_elapsed() -> None:
    started = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=2)).isoformat()
    ledger = {
        "external_jobs": [
            {
                "job_id": "live",
                "category": "legacy",
                "status": "running",
                "started_at": started,
                "estimated_gpu_hours": 1.0,
                "actual_gpu_hours": None,
            }
        ],
        "candidates": {},
        "hard_cap_physical_gpu_hours": 10.0,
    }
    accounting = manager._accounting(ledger, {"candidates": {}})
    assert accounting["committed_gpu_hours"] >= 2.0


def test_process_group_cleanup_kills_and_reaps_child() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        identity = manager._process_identity(child.pid)
        assert identity is not None
        manager._terminate_process_group_and_wait(
            pid=child.pid,
            identity=identity,
            process_group_id=child.pid,
            child=child,
        )
        assert child.poll() is not None
        assert not manager._process_alive(child.pid, identity)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_group_cleanup_never_signals_after_leader_identity_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager, "_process_alive", lambda *args: False)
    signalled = []
    monkeypatch.setattr(manager.os, "killpg", lambda *args: signalled.append(args))
    manager._terminate_process_group_and_wait(
        pid=123,
        identity="old-start-time",
        process_group_id=123,
    )
    assert signalled == []


def test_nonparent_cleanup_accepts_stopped_leader_without_group_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alive = iter((True, False))
    monkeypatch.setattr(manager, "_process_alive", lambda *args: next(alive))
    monkeypatch.setattr(manager, "_bound_pidfd", lambda *args: nullcontext(7))
    monkeypatch.setattr(manager.os, "getpgid", lambda _: 123)
    monkeypatch.setattr(
        manager,
        "_wait_for_process_identity_to_stop",
        lambda *args, **kwargs: True,
    )
    signals = []
    monkeypatch.setattr(
        manager.signal,
        "pidfd_send_signal",
        lambda descriptor, signum: signals.append((descriptor, signum)),
    )
    manager._terminate_process_group_and_wait(
        pid=123,
        identity="bound-start-time",
        process_group_id=123,
    )
    assert signals == [(7, manager.signal.SIGTERM)]


def test_autotune_is_preregistered_before_process_start() -> None:
    source = inspect.getsource(manager._command_autotune)
    assert source.index('"status": "launching"') < source.index("subprocess.Popen")
    assert source.index("_write_ledger(args.ledger, ledger, registry)") < source.index(
        "subprocess.Popen"
    )
    assert '_atomic_json_new(paths["launch"]' in source
    worker = inspect.getsource(manager._command_autotune_worker)
    assert "_set_parent_death_signal(wrapper_pid)" in worker
    assert "deadline - dt.datetime.now(dt.UTC)" in worker
    assert "_terminate_process_group_and_wait(" in worker
    assert "_bind_autotune_child_record(" in worker


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
    assert '"throughput_exact_code_authority": exact_code_authority' in source
    worker = inspect.getsource(manager._command_worker)
    assert "_verify_launch_exact_code_authority(launch)" in worker


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    launch_path = tmp_path / "launch.json"
    child_path = tmp_path / "child.json"
    child_path.touch()
    started_at = "2099-01-01T00:00:00Z"
    deadline_at = "2099-01-01T01:00:00Z"
    launch_binding = {"path": str(launch_path), "sha256": "1" * 64}
    launch = {
        "schema_version": 1,
        "kind": manager.S5_AUTOTUNE_CATEGORY,
        "job_id": "S5-exact-throughput-autotune-attempt01",
        "attempt": 1,
        "child_identity_record": str(child_path),
        "started_at": started_at,
        "deadline_at": deadline_at,
        "command_without_self_digest": [],
    }
    child = {
        "schema_version": 1,
        "kind": manager.S5_AUTOTUNE_CHILD_KIND,
        "job_id": launch["job_id"],
        "attempt": 1,
        "launch_record": launch_binding,
        "wrapper_pid": 123,
        "wrapper_process_start_ticks": "456",
        "child_pid": 789,
        "child_process_start_ticks": "999",
        "child_process_group_id": 789,
        "started_at": started_at,
        "deadline_at": deadline_at,
        "command_sha256": manager._canonical_sha256([]),
    }
    job = {
        "job_id": launch["job_id"],
        "category": manager.S5_AUTOTUNE_CATEGORY,
        "status": "running",
        "worker_record": str(tmp_path / "missing-worker.json"),
        "launch_record": launch_binding,
        "child_identity_record": str(child_path),
        "pid": 123,
        "process_start_ticks": "456",
        "deadline_at": deadline_at,
    }
    snapshots = {
        str(launch_path): (launch, "1" * 64),
        str(child_path): (child, "2" * 64),
    }
    monkeypatch.setattr(manager, "_root_path", lambda value: Path(str(value)))
    monkeypatch.setattr(manager, "_relative", lambda path: str(path))
    monkeypatch.setattr(manager, "_load_json_snapshot", lambda path: snapshots[str(path)])
    alive = iter((False, True))
    monkeypatch.setattr(manager, "_process_alive", lambda *args: next(alive))
    terminated = []
    monkeypatch.setattr(
        manager,
        "_terminate_process_group_and_wait",
        lambda **kwargs: terminated.append(kwargs),
    )
    manager._reconcile_s5_autotune_jobs({"external_jobs": [job]})
    assert job["status"] == "unreconciled"
    assert "without immutable terminal evidence" in job["failure"]
    assert job["child_record"] == {"path": str(child_path), "sha256": "2" * 64}
    assert terminated[0]["pid"] == 789


def test_reconcile_stops_live_wrapper_after_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    launch_path = tmp_path / "launch.json"
    child_path = tmp_path / "not-yet-persisted-child.json"
    launch_binding = {"path": str(launch_path), "sha256": "1" * 64}
    launch = {
        "schema_version": 1,
        "kind": manager.S5_AUTOTUNE_CATEGORY,
        "job_id": "S5-exact-throughput-autotune-attempt01",
        "attempt": 1,
        "child_identity_record": str(child_path),
        "started_at": "2020-01-01T00:00:00Z",
        "deadline_at": "2020-01-01T01:00:00Z",
        "command_without_self_digest": [],
    }
    job = {
        "job_id": launch["job_id"],
        "category": manager.S5_AUTOTUNE_CATEGORY,
        "status": "running",
        "worker_record": str(tmp_path / "missing-worker.json"),
        "launch_record": launch_binding,
        "child_identity_record": str(child_path),
        "pid": 123,
        "process_start_ticks": "456",
        "deadline_at": launch["deadline_at"],
    }
    monkeypatch.setattr(manager, "_root_path", lambda value: Path(str(value)))
    monkeypatch.setattr(
        manager,
        "_load_json_snapshot",
        lambda path: (launch, "1" * 64),
    )
    monkeypatch.setattr(manager, "_process_alive", lambda *args: True)
    stopped = []
    monkeypatch.setattr(
        manager,
        "_terminate_process_identity_and_wait",
        lambda pid, identity: stopped.append((pid, identity)),
    )
    manager._reconcile_s5_autotune_jobs({"external_jobs": [job]})
    assert stopped == [(123, "456")]
    assert job["status"] == "unreconciled"
    assert "deadline elapsed" in job["failure"]


def test_training_success_rejects_post_exit_artifact_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        "output": ROOT / "result.json",
        "best_checkpoint": ROOT / "best.pt",
        "final_checkpoint": ROOT / "final.pt",
    }
    monkeypatch.setattr(manager, "_candidate_artifact_paths", lambda _: paths)
    worker = {
        "artifacts_at_exit": {
            label: {"path": manager._relative(path), "sha256": label[0] * 64}
            for label, path in paths.items()
        }
    }
    validation = {
        "result_sha256": "x" * 64,
        "checkpoints": {
            label: worker["artifacts_at_exit"][label]
            for label in ("best_checkpoint", "final_checkpoint")
        },
    }
    with pytest.raises(manager.GateError, match="changed after"):
        manager._verify_success_artifacts_against_worker(
            worker=worker,
            candidate={},
            validation=validation,
        )
    reconcile = inspect.getsource(manager._reconcile_locked)
    assert "worker, worker_sha256 = _load_json_snapshot(worker_path)" in reconcile
    assert 'attempt["worker_record_sha256"] = worker_sha256' in reconcile
    launcher = inspect.getsource(manager._command_launch)
    assert 'attempt["status"] = "unreconciled"' in launcher
    assert 'state["status"] = "unreconciled"' in launcher


def test_launch_exact_code_authority_rejects_source_map_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_map = {"src/timesfm_lab/config.py": "a" * 64}
    launch = {
        "git_commit": "b" * 40,
        "throughput_measurement": {
            "source": "results/probe.json",
            "source_sha256": "c" * 64,
            "external_job_id": "probe",
        },
        "throughput_exact_code_authority": {
            "schema_version": 1,
            "measurement_artifact": {
                "path": "results/probe.json",
                "sha256": "c" * 64,
            },
            "external_job_id": "probe",
            "measured_git_commit": "b" * 40,
            "relevant_source_sha256": source_map,
            "relevant_source_map_sha256": manager._canonical_sha256(source_map),
        },
        "input_hashes": [
            {"path": "src/timesfm_lab/config.py", "sha256": "d" * 64},
            {"path": "results/probe.json", "sha256": "c" * 64},
        ],
    }
    monkeypatch.setattr(manager, "_verify_file", lambda *args: None)
    monkeypatch.setattr(manager, "_require_git_ancestor", lambda *args: None)
    monkeypatch.setattr(manager, "_verify_source_map_at_commit", lambda *args: None)
    with pytest.raises(manager.GateError, match="not measurement-bound"):
        manager._verify_launch_exact_code_authority(launch)


def test_terminal_verifier_rejects_protocol_downgrade_before_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_path = "results/reproduction/distillation/tampered-launch.json"
    launch = {"protocol_id": "timesfm3-performance-recovery-v1.1"}
    registry = {
        "protocol": {
            "id": "timesfm3-performance-recovery-v1.2",
            "supersedes": {"protocol_id": "timesfm3-performance-recovery-v1.1"},
            "grandfathered_launch_records": [],
        },
        "candidates": {"S5": {}},
    }
    ledger = {
        "candidates": {
            "S5": {
                "attempts": [
                    {
                        "attempt": 1,
                        "status": "failed",
                        "launch_record": launch_path,
                        "launch_record_sha256": "a" * 64,
                    }
                ]
            }
        }
    }
    monkeypatch.setattr(manager, "_load_json_snapshot", lambda _: (launch, "b" * 64))
    with pytest.raises(manager.GateError, match="launch record changed"):
        manager._verify_current_terminal_attempt_artifacts(ledger, registry)


def test_terminal_verifier_requires_explicit_grandfather_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_path = "results/reproduction/distillation/unlisted-predecessor-launch.json"
    launch = {"protocol_id": "timesfm3-performance-recovery-v1.1"}
    registry = {
        "protocol": {
            "id": "timesfm3-performance-recovery-v1.2",
            "supersedes": {"protocol_id": "timesfm3-performance-recovery-v1.1"},
            "grandfathered_launch_records": [],
        },
        "candidates": {"S5": {}},
    }
    ledger = {
        "candidates": {
            "S5": {
                "attempts": [
                    {
                        "attempt": 1,
                        "status": "failed",
                        "launch_record": launch_path,
                        "launch_record_sha256": "b" * 64,
                    }
                ]
            }
        }
    }
    monkeypatch.setattr(manager, "_load_json_snapshot", lambda _: (launch, "b" * 64))
    with pytest.raises(manager.GateError, match="unapproved non-current protocol"):
        manager._verify_current_terminal_attempt_artifacts(ledger, registry)


def test_sigterm_ignoring_autotune_child_is_reaped_before_hard_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); time.sleep(60)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        identity = manager._process_identity(child.pid)
        assert identity is not None
        monkeypatch.setattr(manager, "S5_AUTOTUNE_TERMINATION_GRACE_SECONDS", 0.1)
        monkeypatch.setattr(manager, "S5_AUTOTUNE_FINALIZATION_MARGIN_SECONDS", 0.2)
        deadline = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=1.0)
        manager._terminate_process_group_and_wait(
            pid=child.pid,
            identity=identity,
            process_group_id=child.pid,
            child=child,
            absolute_deadline=deadline,
        )
        assert child.poll() is not None
        assert dt.datetime.now(dt.UTC) < deadline
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=1.0)


def test_trainer_rejects_source_edit_between_wrapper_and_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer_path = tmp_path / "scripts/train_production_student.py"
    module_path = tmp_path / "src/timesfm_lab/config.py"
    trainer_path.parent.mkdir(parents=True)
    module_path.parent.mkdir(parents=True)
    trainer_path.write_text("# frozen trainer\n")
    module_path.write_text("# frozen module\n")
    source_map = {
        "scripts/train_production_student.py": hashlib.sha256(
            trainer_path.read_bytes()
        ).hexdigest(),
        "src/timesfm_lab/config.py": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    }
    launch = {
        "candidate_id": "S5",
        "throughput_exact_code_authority": {
            "schema_version": 1,
            "measured_git_commit": "a" * 40,
            "relevant_source_sha256": source_map,
            "relevant_source_map_sha256": trainer._canonical_sha256(source_map),
        },
        "input_hashes": [
            {"path": relative, "sha256": digest}
            for relative, digest in source_map.items()
        ],
    }
    launch_path = tmp_path / "launch.json"
    launch_bytes = json.dumps(launch, sort_keys=True).encode()
    launch_path.write_bytes(launch_bytes)
    required = {
        "timesfm_lab.config",
        "timesfm_lab.distill.data",
        "timesfm_lab.distill.losses",
        "timesfm_lab.models",
        "timesfm_lab.run_record",
    }

    def edit_after_first_pass(_source_map: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        module_path.write_text("# concurrently edited module\n")
        origins = {name: "src/timesfm_lab/config.py" for name in required}
        loaded = {"src/timesfm_lab/config.py": source_map["src/timesfm_lab/config.py"]}
        return origins, loaded

    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    monkeypatch.setattr(trainer, "__file__", str(trainer_path))
    monkeypatch.setattr(trainer, "_loaded_timesfm_module_authority", edit_after_first_pass)
    with pytest.raises(ValueError, match="changed during exact-code verification"):
        trainer._verify_manager_source_authority(
            launch_path,
            hashlib.sha256(launch_bytes).hexdigest(),
        )

    worker_source = inspect.getsource(manager._command_worker)
    validation_source = inspect.getsource(manager._validate_result)
    terminal_source = inspect.getsource(manager._verify_current_terminal_attempt_artifacts)
    assert '"--expected-manager-launch-sha256"' in worker_source
    assert "_verify_trainer_source_authority" in validation_source
    assert "_verify_trainer_source_authority" in terminal_source


def test_s5_terminal_cost_is_derived_from_worker_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_at = "2026-01-01T00:00:00Z"
    deadline_at = "2026-01-01T01:00:00Z"
    cleanup_budget_seconds = manager._s5_autotune_cleanup_budget_seconds()
    execution_deadline_at = "2026-01-01T00:59:35Z"
    launch = {
        "schema_version": 1,
        "kind": manager.S5_AUTOTUNE_CATEGORY,
        "job_id": "probe",
        "attempt": 1,
        "git_commit": "a" * 40,
        "output": "output.json",
        "worker_record": "worker.json",
        "child_identity_record": "child.json",
        "physical_gpu": 0,
        "reserved_gpu_hours": 1.0,
        "started_at": started_at,
        "deadline_at": deadline_at,
        "execution_deadline_at": execution_deadline_at,
        "cleanup_budget_seconds": cleanup_budget_seconds,
        "command_without_self_digest": [],
    }
    child_binding = {"path": "child.json", "sha256": "4" * 64}
    child = {
        "schema_version": 1,
        "kind": manager.S5_AUTOTUNE_CHILD_KIND,
        "job_id": "probe",
        "attempt": 1,
        "launch_record": {"path": "launch.json", "sha256": "1" * 64},
        "wrapper_pid": 111,
        "wrapper_process_start_ticks": "222",
        "child_pid": 333,
        "child_process_start_ticks": "444",
        "child_process_group_id": 333,
        "started_at": started_at,
        "deadline_at": deadline_at,
        "command_sha256": manager._canonical_sha256([]),
    }
    worker = {
        "job_id": "probe",
        "attempt": 1,
        "launch_record": {"path": "launch.json", "sha256": "1" * 64},
        "child_record": child_binding,
        "started_at": started_at,
        "deadline_at": deadline_at,
        "physical_gpu": 0,
        "physical_gpu_uuid": "GPU-a",
        "elapsed_seconds": 72.0,
        "child_stopped_at": "2026-01-01T00:01:11Z",
        "ended_at": "2026-01-01T00:01:12Z",
        "exit_code": 0,
        "artifact_at_exit": {"path": "output.json", "sha256": "3" * 64},
    }
    snapshots = {
        "launch.json": (launch, "1" * 64),
        "worker.json": (worker, "2" * 64),
        "child.json": (child, "4" * 64),
    }
    monkeypatch.setattr(manager, "_root_path", lambda value: Path(str(value)))
    monkeypatch.setattr(manager, "_load_json_snapshot", lambda path: snapshots[str(path)])
    monkeypatch.setattr(manager, "_verify_file", lambda *args: None)
    monkeypatch.setattr(manager, "_require_process_stopped", lambda *args: None)
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
        "ended_at": "2026-01-01T00:01:12Z",
        "exit_code": 0,
        "physical_gpu_count": 1,
        "physical_gpu": 0,
        "physical_gpu_uuid": "GPU-a",
        "launch_record": {"path": "launch.json", "sha256": "1" * 64},
        "worker_record": "worker.json",
        "child_identity_record": "child.json",
        "child_record": child_binding,
        "pid": 111,
        "process_start_ticks": "222",
        "terminal_record": {"path": "worker.json", "sha256": "2" * 64},
        "artifact": "output.json",
        "artifact_sha256": "3" * 64,
        "started_at": started_at,
        "deadline_at": deadline_at,
        "execution_deadline_at": execution_deadline_at,
        "cleanup_budget_seconds": cleanup_budget_seconds,
    }
    manager._validate_s5_autotune_job(job)
    job["actual_gpu_hours"] = math.nextafter(job["actual_gpu_hours"], math.inf)
    with pytest.raises(manager.GateError, match="derive from its worker"):
        manager._validate_s5_autotune_job(job)
    job["actual_gpu_hours"] = 72.0 / 3600.0
    worker["ended_at"] = "2026-01-01T02:00:00Z"
    job["ended_at"] = worker["ended_at"]
    worker["elapsed_seconds"] = 7200.0
    job["elapsed_seconds"] = 7200.0
    job["actual_gpu_hours"] = 2.0
    with pytest.raises(manager.GateError, match="absolute deadline"):
        manager._validate_s5_autotune_job(job)
