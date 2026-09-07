#!/usr/bin/env python3
"""Freeze or verify byte-exact production source/cache artifact rosters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


class IntegrityError(RuntimeError):
    """A production artifact differs from its frozen byte roster."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise IntegrityError(f"artifact root is absent: {root}")
    symlinks = sorted(path for path in root.rglob("*") if path.is_symlink())
    if symlinks:
        raise IntegrityError(f"artifact root contains symlinks: {symlinks[:3]}")
    return sorted(path for path in root.rglob("*") if path.is_file())


def _hash_roster(root: Path, workers: int) -> list[dict[str, Any]]:
    paths = _files(root)

    def bind(path: Path) -> dict[str, Any]:
        return {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(bind, paths))


def _cache_audit_roster(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in audit["datasets"]:
        name = str(dataset["dataset"])
        for shard in dataset["shards"]:
            rows.append(
                {
                    "path": str(Path(name) / Path(str(shard["path"])).name),
                    "bytes": int(shard["bytes"]),
                    "sha256": str(shard["sha256"]),
                }
            )
    rows.sort(key=lambda row: row["path"])
    if len({row["path"] for row in rows}) != len(rows):
        raise IntegrityError("cache audit contains duplicate shard paths")
    return rows


def _section(root_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "declared_root": root_name,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "roster_sha256": canonical_sha256(rows),
        "files": rows,
    }


def freeze_manifest(
    *,
    source_root: Path,
    cache_root: Path,
    cache_audit: Path,
    output: Path,
    source_root_name: str,
    cache_root_name: str,
    cache_audit_name: str,
    workers: int,
) -> dict[str, Any]:
    if output.exists():
        raise IntegrityError(f"refusing to overwrite frozen manifest: {output}")
    audit = json.loads(cache_audit.read_text())
    source_rows = _hash_roster(source_root, workers)
    cache_rows = _hash_roster(cache_root, workers)
    expected_cache_rows = _cache_audit_roster(audit)
    cached_arrays = [row for row in cache_rows if Path(row["path"]).suffix == ".npz"]
    cached_metadata = [row for row in cache_rows if Path(row["path"]).suffix == ".json"]
    if (
        cached_arrays != expected_cache_rows
        or len(cached_metadata) != len(cached_arrays)
        or {Path(row["path"]).with_suffix("") for row in cached_metadata}
        != {Path(row["path"]).with_suffix("") for row in cached_arrays}
    ):
        raise IntegrityError("actual cache shard bytes/roster differ from the cache audit")
    payload = {
        "schema_version": 1,
        "status": "frozen_byte_exact",
        "dataset_revision": str(audit["dataset_revision"]),
        "source_data": _section(source_root_name, source_rows),
        "teacher_cache": {
            **_section(cache_root_name, cache_rows),
            "cache_audit": {
                "path": cache_audit_name,
                "sha256": sha256(cache_audit),
            },
        },
        "target_values_interpreted": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def verify_manifest(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    source_root: Path,
    cache_root: Path,
    cache_audit: Path,
    expected_source_root_name: str,
    expected_cache_root_name: str,
    expected_cache_audit_name: str,
    workers: int = 4,
) -> dict[str, Any]:
    if sha256(manifest_path) != expected_manifest_sha256:
        raise IntegrityError("production artifact manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest.pop("manifest_payload_sha256", None)
    if claimed != canonical_sha256(manifest):
        raise IntegrityError("production artifact manifest payload hash failed")
    manifest["manifest_payload_sha256"] = claimed
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "frozen_byte_exact"
        or manifest.get("target_values_interpreted") is not False
        or manifest["source_data"].get("declared_root") != expected_source_root_name
        or manifest["teacher_cache"].get("declared_root") != expected_cache_root_name
        or manifest["teacher_cache"].get("cache_audit")
        != {"path": expected_cache_audit_name, "sha256": sha256(cache_audit)}
    ):
        raise IntegrityError("production artifact manifest authority changed")
    actual_source = _section(
        expected_source_root_name, _hash_roster(source_root, workers)
    )
    actual_cache = _section(expected_cache_root_name, _hash_roster(cache_root, workers))
    if actual_source != manifest["source_data"]:
        raise IntegrityError("production source-data bytes/roster changed")
    expected_cache = {
        key: manifest["teacher_cache"][key]
        for key in ("declared_root", "file_count", "total_bytes", "roster_sha256", "files")
    }
    if actual_cache != expected_cache:
        raise IntegrityError("production teacher-cache bytes/roster changed")
    audit = json.loads(cache_audit.read_text())
    cached_arrays = [
        row for row in actual_cache["files"] if Path(row["path"]).suffix == ".npz"
    ]
    if cached_arrays != _cache_audit_roster(audit):
        raise IntegrityError("production teacher cache differs from its audit shard roster")
    return {
        "manifest_sha256": expected_manifest_sha256,
        "source_roster_sha256": actual_source["roster_sha256"],
        "source_file_count": actual_source["file_count"],
        "source_total_bytes": actual_source["total_bytes"],
        "cache_roster_sha256": actual_cache["roster_sha256"],
        "cache_file_count": actual_cache["file_count"],
        "cache_total_bytes": actual_cache["total_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    verify = subparsers.add_parser("verify")
    for command in (freeze, verify):
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--cache-root", type=Path, required=True)
        command.add_argument("--cache-audit", type=Path, required=True)
        command.add_argument("--source-root-name", required=True)
        command.add_argument("--cache-root-name", required=True)
        command.add_argument("--cache-audit-name", required=True)
        command.add_argument("--workers", type=int, default=4)
    freeze.add_argument("--output", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise IntegrityError("workers must be between 1 and 16")
    if args.command == "freeze":
        payload = freeze_manifest(
            source_root=args.source_root.resolve(),
            cache_root=args.cache_root.resolve(),
            cache_audit=args.cache_audit,
            output=args.output,
            source_root_name=args.source_root_name,
            cache_root_name=args.cache_root_name,
            cache_audit_name=args.cache_audit_name,
            workers=args.workers,
        )
        print(json.dumps(payload["manifest_payload_sha256"]))
    else:
        result = verify_manifest(
            manifest_path=args.manifest,
            expected_manifest_sha256=args.manifest_sha256,
            source_root=args.source_root.resolve(),
            cache_root=args.cache_root.resolve(),
            cache_audit=args.cache_audit,
            expected_source_root_name=args.source_root_name,
            expected_cache_root_name=args.cache_root_name,
            expected_cache_audit_name=args.cache_audit_name,
            workers=args.workers,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
