#!/usr/bin/env python3
"""Capture an append-only, human-readable machine environment snapshot.

The script deliberately runs a small allowlist of diagnostic commands instead
of dumping the process environment, which could expose credentials. Output is
created with exclusive-create semantics and made read-only after a successful
write. Command failures are recorded in the snapshot rather than aborting it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import shlex
import socket
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

FORMAT_VERSION = 2
COMMAND_TIMEOUT_SECONDS = 120
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "environment"
HOSTNAME = socket.gethostname()
PCI_BUS_ID_PATTERN = re.compile(
    r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{8}:)?[0-9A-Fa-f]{2}:"
    r"[0-9A-Fa-f]{2}\.[0-7](?![0-9A-Fa-f])"
)


def _privacy_filter(text: str) -> str:
    """Remove stable machine identifiers while retaining diagnostic values."""
    path_redactions = (
        (str(REPOSITORY_ROOT), "<REPOSITORY_ROOT>"),
        (sys.prefix, "<PYTHON_PREFIX>"),
        (str(Path.home()), "<USER_HOME>"),
    )
    for path, replacement in path_redactions:
        if path and path != os.sep:
            text = text.replace(path, replacement)
    if HOSTNAME:
        text = text.replace(HOSTNAME, "<REDACTED_HOSTNAME>")
    text = PCI_BUS_ID_PATTERN.sub("<REDACTED_PCI_BUS_ID>", text)

    redacted_lines = []
    for line in text.splitlines():
        key, separator, _value = line.partition(":")
        normalized_key = key.strip()
        is_sensitive_key = (
            "UUID" in normalized_key
            or "Serial Number" in normalized_key
            or normalized_key.startswith("Chassis ")
            or normalized_key in {"GPU PDI", "Board ID", "Host ID"}
        )
        if separator and is_sensitive_key:
            line = f"{key}: <REDACTED>"
        redacted_lines.append(line)
    return "\n".join(redacted_lines)


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    """Run one diagnostic command and retain stdout, stderr, and status."""
    rendered = shlex.join(str(part) for part in command)
    lines = [f"$ {rendered}"]
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        lines.extend(("exit_code: COMMAND_NOT_FOUND", f"stderr:\n{error}"))
    except subprocess.TimeoutExpired as error:
        lines.append(f"exit_code: TIMEOUT_AFTER_{COMMAND_TIMEOUT_SECONDS}_SECONDS")
        if error.stdout:
            stdout = (
                error.stdout.decode(errors="replace")
                if isinstance(error.stdout, bytes)
                else error.stdout
            )
            lines.append(f"stdout:\n{stdout.rstrip()}")
        if error.stderr:
            stderr = (
                error.stderr.decode(errors="replace")
                if isinstance(error.stderr, bytes)
                else error.stderr
            )
            lines.append(f"stderr:\n{stderr.rstrip()}")
    else:
        lines.append(f"exit_code: {completed.returncode}")
        lines.append(f"stdout:\n{completed.stdout.rstrip() or '<empty>'}")
        if completed.stderr:
            lines.append(f"stderr:\n{completed.stderr.rstrip()}")
    return _privacy_filter("\n".join(lines))


def _section(title: str, body: str) -> str:
    return f"\n{'=' * 80}\n{title}\n{'=' * 80}\n{body.rstrip()}\n"


def _parse_named_repo(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from error
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name.strip(), Path(path).expanduser().resolve()


def _repo_revision(name: str, path: Path) -> str:
    body = [f"name: {name}", f"path: <REPOSITORY:{name}>"]
    if not path.is_dir():
        body.append("status: NOT_FOUND")
        return "\n".join(body)
    body.append(_run(["git", "rev-parse", "HEAD"], cwd=path))
    body.append(_run(["git", "status", "--short"], cwd=path))
    return "\n".join(body)


def _torch_probe_command() -> list[str]:
    program = r"""
import json

try:
    import torch
except Exception as error:
    print(json.dumps({"import_error": repr(error)}, indent=2, sort_keys=True))
    raise SystemExit(1)

report = {
    "torch_version": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "torch_cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
}
devices = []
for index in range(torch.cuda.device_count()):
    try:
        properties = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        })
    except Exception as error:
        devices.append({"index": index, "probe_error": repr(error)})
report["devices"] = devices
print(json.dumps(report, indent=2, sort_keys=True))
""".strip()
    return [sys.executable, "-c", program]


def _timesfm_probe_command() -> list[str]:
    program = r"""
import importlib.metadata
import json

report = {}
for distribution in ("timesfm", "torch", "transformers", "huggingface-hub"):
    try:
        report[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        report[distribution] = "NOT_INSTALLED"
print(json.dumps(report, indent=2, sort_keys=True))
""".strip()
    return [sys.executable, "-c", program]


def _build_snapshot(args: argparse.Namespace, captured_at: dt.datetime) -> str:
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    header = "\n".join(
        (
            "timesfm-lab raw environment snapshot",
            f"format_version: {FORMAT_VERSION}",
            f"captured_at_utc: {captured_at.isoformat().replace('+00:00', 'Z')}",
            "capture_script: scripts/capture_environment.py",
            f"capture_script_sha256: {script_sha256}",
            "secrets_policy: allowlisted diagnostics only; environment variables are not captured",
            "privacy_safe: true",
            (
                "redactions: hostname, repository paths, PCI bus IDs, GPU UUID/PDI, "
                "serial, board ID, host ID, and chassis identity fields"
            ),
        )
    )

    sections = [
        _section("Operating system and kernel", _run(["uname", "-srmv"])),
        _section("OS release", _run(["cat", "/etc/os-release"])),
        _section("CPU", _run(["lscpu"])),
        _section("RAM (bytes)", _run(["free", "-b"])),
        _section("RAM (human-readable)", _run(["free", "-h"])),
        _section(
            "Disk at repository (bytes)",
            _run(["df", "-B1", "-P", str(REPOSITORY_ROOT)]),
        ),
        _section(
            "Disk at repository (human-readable)",
            _run(["df", "-h", "-P", str(REPOSITORY_ROOT)]),
        ),
        _section("NVIDIA summary", _run(["nvidia-smi"])),
        _section("NVIDIA full query", _run(["nvidia-smi", "-q"])),
        _section("CUDA compiler", _run(["nvcc", "--version"])),
        _section("Python", _run([sys.executable, "--version"])),
        _section("pip", _run([sys.executable, "-m", "pip", "--version"])),
        _section("PyTorch and CUDA probe", _run(_torch_probe_command())),
        _section("Core Python package versions", _run(_timesfm_probe_command())),
        _section(
            "timesfm-lab repository revision",
            _repo_revision("timesfm-lab", REPOSITORY_ROOT),
        ),
    ]

    revision_lines = [
        "model: google/timesfm-3.0-pytorch",
        f"model_revision: {args.timesfm_model_revision or 'NOT_YET_RESOLVED'}",
        (
            "note: Record the immutable Hugging Face commit hash with "
            "--timesfm-model-revision once the model is fetched."
        ),
    ]
    sections.append(_section("TimesFM model identity", "\n".join(revision_lines)))

    if args.external_repo:
        external_body = "\n\n".join(_repo_revision(name, path) for name, path in args.external_repo)
    else:
        external_body = (
            "status: NONE_SUPPLIED\n"
            "note: Use --external-repo NAME=PATH for TimesFM, GIFT-Eval, and "
            "other checked-out dependencies."
        )
    sections.append(_section("External repository revisions", external_body))
    return header + "".join(sections)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"snapshot directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--timesfm-model-revision",
        help="immutable Hugging Face commit hash for google/timesfm-3.0-pytorch",
    )
    parser.add_argument(
        "--external-repo",
        action="append",
        default=[],
        type=_parse_named_repo,
        metavar="NAME=PATH",
        help="record an external repository commit; may be supplied repeatedly",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    captured_at = dt.datetime.now(dt.UTC)
    timestamp = captured_at.strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"environment_{timestamp}.txt"
    snapshot = _build_snapshot(args, captured_at)

    # Mode "x" guarantees that a raw snapshot can never be overwritten.
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(snapshot)
        stream.flush()
        os.fsync(stream.fileno())
    output_path.chmod(0o444)

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    with checksum_path.open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {output_path.name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    checksum_path.chmod(0o444)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
