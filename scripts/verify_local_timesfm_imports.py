#!/usr/bin/env python3
"""Verify that every loaded TimesFM-Lab module came from this checkout."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

CONFIRMATION_LOCK_NAME = "timesfm-lab-performance-recovery-confirmation.lock"
CONFIRMATION_BURN_NAME = (
    "timesfm-lab-performance-recovery-confirmation-access-burn.json"
)
_RECOVERY_PROCESS_LOCK_FDS: list[int] = []


class ImportAuthorityError(RuntimeError):
    """The active Python import authority is not the declared repository."""


class ConfirmationAccessClosed(RuntimeError):
    """Performance-recovery training was attempted after confirmation access."""


def _release_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _open_lock(path: Path) -> int:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.set_inheritable(descriptor, False)
    return descriptor


def acquire_recovery_training_process_lock(git_common_dir: Path) -> dict[str, str]:
    """Hold a shared lock until process exit and atomically reject a prior burn."""

    common = git_common_dir.resolve()
    lock_path = common / CONFIRMATION_LOCK_NAME
    burn_path = common / CONFIRMATION_BURN_NAME
    descriptor = _open_lock(lock_path)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        if burn_path.exists():
            raise ConfirmationAccessClosed(
                "training is permanently closed after confirmation access is burned"
            )
    except BaseException:
        _release_lock(descriptor)
        raise
    # Intentionally retain the raw descriptor without registering Python-level
    # cleanup. The kernel releases it only when the complete process exits.
    _RECOVERY_PROCESS_LOCK_FDS.append(descriptor)
    return {
        "scope": "git_common_dir",
        "lock_name": CONFIRMATION_LOCK_NAME,
        "mode": "shared_until_process_exit",
        "burn_name": CONFIRMATION_BURN_NAME,
    }


@contextmanager
def hold_confirmation_evaluation_lock(
    git_common_dir: Path,
) -> Iterator[dict[str, str]]:
    """Hold the repository-global exclusive lock around the sole confirmation pass."""

    common = git_common_dir.resolve()
    lock_path = common / CONFIRMATION_LOCK_NAME
    descriptor = _open_lock(lock_path)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield {
            "scope": "git_common_dir",
            "lock_name": CONFIRMATION_LOCK_NAME,
            "mode": "exclusive_through_result_commit",
            "burn_name": CONFIRMATION_BURN_NAME,
        }
    finally:
        _release_lock(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expected_origins(package_root: Path, module_name: str) -> set[Path]:
    parts = module_name.split(".")[1:]
    stem = package_root.joinpath(*parts) if parts else package_root
    return {
        (stem / "__init__.py").resolve(),
        stem.with_suffix(".py").resolve(),
    }


def verify_local_timesfm_imports(repo_root: Path) -> dict[str, Any]:
    """Fail closed unless PYTHONPATH and loaded modules resolve to ``repo_root/src``."""

    root = repo_root.resolve()
    source_root = (root / "src").resolve()
    package_root = (source_root / "timesfm_lab").resolve()
    required_pythonpath = str(source_root)
    if os.environ.get("PYTHONPATH") != required_pythonpath:
        raise ImportAuthorityError(
            "PYTHONPATH must contain exactly this checkout's absolute src directory: "
            f"{required_pythonpath}"
        )
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ImportAuthorityError("PYTHONNOUSERSITE must equal 1")

    package = sys.modules.get("timesfm_lab")
    if not isinstance(package, ModuleType):
        raise ImportAuthorityError("timesfm_lab was not imported before authority validation")
    package_paths = [Path(value).resolve() for value in getattr(package, "__path__", ())]
    if package_paths != [package_root]:
        raise ImportAuthorityError(
            f"timesfm_lab package search path is not exact: {package_paths}"
        )

    bindings: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        if name != "timesfm_lab" and not name.startswith("timesfm_lab."):
            continue
        if not isinstance(module, ModuleType):
            raise ImportAuthorityError(f"loaded TimesFM module is invalid: {name}")
        origin_value = getattr(module, "__file__", None)
        spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if not isinstance(origin_value, str) or not isinstance(spec_origin, str):
            raise ImportAuthorityError(f"loaded TimesFM module lacks a file origin: {name}")
        origin = Path(origin_value).resolve()
        if origin != Path(spec_origin).resolve():
            raise ImportAuthorityError(f"TimesFM module file/spec origins differ: {name}")
        try:
            relative = origin.relative_to(root)
            origin.relative_to(package_root)
        except ValueError as error:
            raise ImportAuthorityError(
                f"TimesFM module came from outside this checkout: {name} -> {origin}"
            ) from error
        if origin not in _expected_origins(package_root, name) or not origin.is_file():
            raise ImportAuthorityError(
                f"TimesFM module has a noncanonical repository origin: {name} -> {origin}"
            )
        bindings.append(
            {"module": name, "path": str(relative), "sha256": _sha256(origin)}
        )
    if not bindings:
        raise ImportAuthorityError("no TimesFM modules were bound")
    authority = {
        "schema_version": 1,
        "status": "exact_local_import_authority",
        "pythonpath": required_pythonpath,
        "python_no_user_site": "1",
        "source_root": "src",
        "package_root": "src/timesfm_lab",
        "modules": bindings,
    }
    authority["authority_sha256"] = _canonical_sha256(authority)
    return authority
