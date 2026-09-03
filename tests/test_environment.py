from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "capture_environment.py"
SPEC = importlib.util.spec_from_file_location("capture_environment", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
capture_environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_environment)


def test_capture_uses_required_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run(command: Sequence[str], *, cwd: Path | None = None) -> str:
        del cwd
        commands.append(tuple(str(part) for part in command))
        return "exit_code: 0\nstdout:\nprobe"

    monkeypatch.setattr(capture_environment, "_run", fake_run)
    arguments = argparse.Namespace(
        timesfm_model_revision=None,
        external_repo=[],
    )
    snapshot = capture_environment._build_snapshot(
        arguments,
        dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC),
    )

    assert ("nvidia-smi",) in commands
    assert ("nvidia-smi", "-q") in commands
    assert ("nvcc", "--version") in commands
    assert ("lscpu",) in commands
    assert ("free", "-b") in commands
    assert "PyTorch and CUDA probe" in snapshot
    assert "model_revision: NOT_YET_RESOLVED" in snapshot
    assert "environment variables are not captured" in snapshot


def test_main_exclusively_creates_read_only_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz: dt.tzinfo | None = None) -> FixedDateTime:
            return cls(2026, 1, 2, 3, 4, 5, 6789, tzinfo=tz)

    arguments = argparse.Namespace(
        output_dir=tmp_path,
        timesfm_model_revision=None,
        external_repo=[],
    )
    monkeypatch.setattr(capture_environment, "_arguments", lambda: arguments)
    monkeypatch.setattr(capture_environment.dt, "datetime", FixedDateTime)
    monkeypatch.setattr(
        capture_environment,
        "_build_snapshot",
        lambda args, captured_at: "snapshot\n",
    )

    assert capture_environment.main() == 0
    snapshots = list(tmp_path.glob("environment_*.txt"))
    assert len(snapshots) == 1
    assert snapshots[0].read_text(encoding="utf-8") == "snapshot\n"
    assert stat.S_IMODE(snapshots[0].stat().st_mode) == 0o444
    checksum_path = snapshots[0].with_suffix(".txt.sha256")
    expected_digest = hashlib.sha256(b"snapshot\n").hexdigest()
    assert checksum_path.read_text(encoding="utf-8") == (
        f"{expected_digest}  {snapshots[0].name}\n"
    )
    assert stat.S_IMODE(checksum_path.stat().st_mode) == 0o444

    with pytest.raises(FileExistsError):
        capture_environment.main()


def test_named_repo_requires_name_and_path() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        capture_environment._parse_named_repo("missing-separator")
    with pytest.raises(argparse.ArgumentTypeError):
        capture_environment._parse_named_repo("=./repo")


def test_privacy_filter_removes_machine_identifiers() -> None:
    raw = "\n".join(
        (
            f"Linux {capture_environment.HOSTNAME} kernel",
            f"path: {capture_environment.REPOSITORY_ROOT}/data",
            f"python: {sys.prefix}/bin/python",
            f"home: {Path.home()}/cache",
            "GPU UUID: GPU-01234567-89ab-cdef-0123-456789abcdef",
            "GPU PDI: 0x0123456789abcdef",
            "Serial Number: 123456789",
            "Board ID: 0x4900",
            "Chassis Serial Number: abc123",
            "Bus Id: 00000000:49:00.0",
        )
    )

    filtered = capture_environment._privacy_filter(raw)

    assert capture_environment.HOSTNAME not in filtered
    assert str(capture_environment.REPOSITORY_ROOT) not in filtered
    assert sys.prefix not in filtered
    assert str(Path.home()) not in filtered
    for identifier in (
        "GPU-01234567-89ab-cdef-0123-456789abcdef",
        "0x0123456789abcdef",
        "123456789",
        "0x4900",
        "abc123",
        "00000000:49:00.0",
    ):
        assert identifier not in filtered
