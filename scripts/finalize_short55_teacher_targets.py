#!/usr/bin/env python3
"""Validate the frozen short55 teacher result and finalize its quality thresholds.

The command is a dry run unless ``--write`` is supplied.  It intentionally has
no GPU or model dependencies: it consumes only the frozen YAML authorities and
the completed JSON run record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "configs/performance_recovery/targets.yaml"

MASE = "MASE[0.5]"
MWQL = "mean_weighted_sum_quantile_loss"
METRICS = (MASE, MWQL)
LIMIT_MULTIPLIER = 1.06
PENDING_READINESS = "awaiting_fingerprint_validated_teacher_artifact"
FINAL_READINESS = "fingerprint_validated_teacher_artifact"


class GateError(ValueError):
    """Raised when an input does not satisfy the frozen target contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise GateError(f"cannot load {kind} {path}: {error}") from error
    if not isinstance(value, dict):
        raise GateError(f"{kind} {path} must contain a mapping")
    return value


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a mapping")
    return value


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise GateError(f"{label} must be a list")
    return value


def _repo_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise GateError(f"{label} must be a repository-relative path: {value!r}")
    resolved = (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise GateError(f"{label} escapes the repository: {value!r}") from error
    return resolved


def _positive_metric(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise GateError(f"{label} must be finite and positive, got {number!r}")
    return number


def _same_float(actual: object, expected: float, *, label: str) -> None:
    number = _positive_metric(actual, label=label)
    if not math.isclose(number, expected, rel_tol=1e-13, abs_tol=1e-15):
        raise GateError(f"{label} is {number!r}, independently recomputed value is {expected!r}")


def _geometric_mean(values: list[float]) -> float:
    if not values:
        raise GateError("cannot aggregate an empty metric list")
    # fsum over logarithms is stable for the wide scale range in GIFT-Eval.
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _validate_artifact_git_config(
    artifact: dict[str, Any], *, config_path: str, expected_sha256: str, boundary: str
) -> str:
    commit = artifact.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise GateError(f"teacher artifact has invalid git_commit: {commit!r}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", boundary, commit],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        detail = ancestry.stderr.strip() or f"exit code {ancestry.returncode}"
        raise GateError(
            f"teacher artifact commit {commit} is not a valid descendant of {boundary}: {detail}"
        )
    historical = subprocess.run(
        ["git", "show", f"{commit}:{config_path}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if historical.returncode != 0:
        detail = historical.stderr.decode(errors="replace").strip()
        raise GateError(f"cannot read {config_path} at artifact commit {commit}: {detail}")
    historical_sha = hashlib.sha256(historical.stdout).hexdigest()
    if historical_sha != expected_sha256:
        raise GateError(
            "short55 config at the artifact's recorded commit does not match the frozen SHA: "
            f"artifact_commit_sha={historical_sha!r}, expected={expected_sha256!r}"
        )
    return commit


def _expected_configurations(evaluation_config: dict[str, Any]) -> list[str]:
    evaluation = _mapping(evaluation_config.get("evaluation"), label="evaluation config.evaluation")
    rows = _list(evaluation.get("datasets"), label="evaluation config.evaluation.datasets")
    names: list[str] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, label=f"evaluation config dataset row {index}")
        name = row.get("name")
        term = row.get("term")
        if not isinstance(name, str) or not name or not isinstance(term, str) or not term:
            raise GateError(f"evaluation config dataset row {index} has invalid name/term")
        names.append(f"{name}/{term}")
    if len(names) != len(set(names)):
        raise GateError("evaluation config contains duplicate configuration names")
    return names


def _validate_authorities(
    targets_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    targets = _load_mapping(targets_path, kind="target authority")
    if targets.get("protocol_id") != "timesfm3-performance-recovery-v1.1":
        raise GateError("target authority is not frozen recovery protocol v1.1")

    revisions = _mapping(targets.get("revisions"), label="targets.revisions")
    teacher_revision = _mapping(revisions.get("teacher"), label="targets.revisions.teacher")
    gift_revision = _mapping(revisions.get("gift_eval"), label="targets.revisions.gift_eval")
    quality = _mapping(targets.get("quality"), label="targets.quality")
    if quality.get("mode") != "multivariate":
        raise GateError("targets.quality.mode must remain 'multivariate'")
    if quality.get("aggregation") != "unweighted_geometric_mean_across_configurations":
        raise GateError("targets.quality.aggregation changed from the frozen protocol")
    if quality.get("maximum_student_to_teacher_ratio") != LIMIT_MULTIPLIER:
        raise GateError(f"target multiplier must remain exactly {LIMIT_MULTIPLIER}")
    scopes = _mapping(quality.get("scopes"), label="targets.quality.scopes")
    short55 = _mapping(scopes.get("short55"), label="targets.quality.scopes.short55")

    if short55.get("name") != "complete_short_horizon_gift_eval":
        raise GateError("short55 scope name changed from the frozen protocol")
    if short55.get("configuration_count") != 55:
        raise GateError("short55 configuration_count must be exactly 55")

    config_path = _repo_path(short55.get("config_path"), label="short55.config_path")
    artifact_path = _repo_path(short55.get("teacher_artifact"), label="short55.teacher_artifact")
    configured_sha = short55.get("config_sha256")
    actual_sha = _sha256(config_path)
    if configured_sha != actual_sha:
        raise GateError(
            f"short55 config SHA mismatch: authority={configured_sha!r}, actual={actual_sha!r}"
        )

    evaluation_config = _load_mapping(config_path, kind="short55 evaluation config")
    if evaluation_config.get("model_revision") != teacher_revision.get("revision"):
        raise GateError("teacher revision differs between targets and short55 config")
    if evaluation_config.get("dataset_revision") != gift_revision.get("revision"):
        raise GateError("GIFT-Eval revision differs between targets and short55 config")
    model = _mapping(evaluation_config.get("model"), label="evaluation config.model")
    if model.get("id") != teacher_revision.get("id"):
        raise GateError("teacher model id differs between targets and short55 config")
    evaluation = _mapping(evaluation_config.get("evaluation"), label="evaluation config.evaluation")
    if evaluation.get("reportable_scope") != short55.get("name"):
        raise GateError("evaluation config reportable scope differs from target scope")
    if evaluation.get("aggregation") != "geometric_mean_across_configurations":
        raise GateError("evaluation config aggregation differs from the frozen protocol")

    return targets, short55, evaluation_config, config_path, artifact_path


def _validate_artifact(
    artifact: dict[str, Any],
    *,
    short55: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    expected_mode = "multivariate"
    expected_scope = str(short55["name"])
    expected_count = int(short55["configuration_count"])
    expected_config_path = str(short55["config_path"])
    expected_names = _expected_configurations(evaluation_config)
    if len(expected_names) != expected_count:
        raise GateError(
            f"short55 config contains {len(expected_names)} names, expected {expected_count}"
        )

    if artifact.get("status") != "succeeded":
        raise GateError(f"teacher artifact status is not succeeded: {artifact.get('status')!r}")
    if artifact.get("failure") is not None:
        raise GateError(f"teacher artifact has top-level failure: {artifact.get('failure')!r}")
    if artifact.get("config_path") != expected_config_path:
        raise GateError(f"teacher artifact config_path mismatch: {artifact.get('config_path')!r}")
    if artifact.get("model_revision") != evaluation_config.get("model_revision"):
        raise GateError("teacher artifact model revision does not match pinned config")
    if artifact.get("dataset_revision") != evaluation_config.get("dataset_revision"):
        raise GateError("teacher artifact dataset revision does not match pinned config")
    if artifact.get("seed") != evaluation_config.get("seed"):
        raise GateError("teacher artifact seed does not match pinned config")
    expected_run_id = (
        f"{evaluation_config.get('run_id')}-{expected_mode}-seed{evaluation_config.get('seed')}"
    )
    if artifact.get("run_id") != expected_run_id:
        raise GateError(f"teacher artifact run_id mismatch: {artifact.get('run_id')!r}")

    extra = _mapping(artifact.get("extra"), label="teacher artifact.extra")
    if extra.get("mode") != expected_mode:
        raise GateError(f"teacher artifact mode must be {expected_mode!r}")
    if extra.get("scope") != expected_scope:
        raise GateError(f"teacher artifact scope must be {expected_scope!r}")
    failures = _list(extra.get("failures"), label="teacher artifact.extra.failures")
    if failures:
        raise GateError(f"teacher artifact contains {len(failures)} configuration failures")

    raw_results = _list(extra.get("results"), label="teacher artifact.extra.results")
    if len(raw_results) != expected_count:
        raise GateError(
            f"teacher artifact has {len(raw_results)} results, expected exactly {expected_count}"
        )
    results = [
        _mapping(raw, label=f"teacher artifact result row {index}")
        for index, raw in enumerate(raw_results)
    ]
    actual_names = [row.get("configuration") for row in results]
    if any(not isinstance(name, str) for name in actual_names):
        raise GateError("one or more teacher result rows have a non-string configuration name")
    if actual_names != expected_names:
        missing = sorted(set(expected_names).difference(actual_names))
        unexpected = sorted(set(actual_names).difference(expected_names))
        duplicates = sorted(
            {name for name in actual_names if actual_names.count(name) > 1}, key=str
        )
        raise GateError(
            "teacher artifact configuration sequence differs from frozen config: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
        )
    if any(row.get("mode") != expected_mode for row in results):
        raise GateError("one or more teacher result rows have the wrong mode")

    aggregate: dict[str, float] = {}
    for metric in METRICS:
        values = [
            _positive_metric(row.get(metric), label=f"{name}/{metric}")
            for name, row in zip(expected_names, results, strict=True)
        ]
        aggregate[metric] = _geometric_mean(values)

    embedded_aggregation = _mapping(
        extra.get("aggregation"), label="teacher artifact.extra.aggregation"
    )
    if embedded_aggregation.get("method") != "unweighted geometric mean across configurations":
        raise GateError("teacher artifact reports a different aggregation method")
    if embedded_aggregation.get("configuration_count") != expected_count:
        raise GateError("teacher artifact aggregation count is not exactly 55")
    embedded_metrics = _mapping(
        embedded_aggregation.get("metrics"), label="teacher artifact aggregation metrics"
    )
    top_metrics = _mapping(artifact.get("metrics"), label="teacher artifact.metrics")
    for metric, recomputed in aggregate.items():
        _same_float(embedded_metrics.get(metric), recomputed, label=f"embedded aggregate {metric}")
        _same_float(
            top_metrics.get(f"aggregate/geometric_mean/{metric}"),
            recomputed,
            label=f"top-level aggregate {metric}",
        )
    for name, row in zip(expected_names, results, strict=True):
        for metric in METRICS:
            _same_float(
                top_metrics.get(f"{name}/{metric}"),
                float(row[metric]),
                label=f"top-level per-configuration metric {name}/{metric}",
            )
    return aggregate, expected_names


def _desired_values(aggregate: dict[str, float], artifact_sha256: str) -> dict[str, object]:
    return {
        "teacher_artifact_sha256": artifact_sha256,
        "teacher_mase": aggregate[MASE],
        "teacher_mwql": aggregate[MWQL],
        "maximum_student_mase": aggregate[MASE] * LIMIT_MULTIPLIER,
        "maximum_student_mwql": aggregate[MWQL] * LIMIT_MULTIPLIER,
        "readiness": FINAL_READINESS,
    }


def _validate_existing(short55: dict[str, Any], desired: dict[str, object]) -> None:
    for key, expected in desired.items():
        actual = short55.get(key)
        if actual is None:
            continue
        if key == "readiness" and actual == PENDING_READINESS:
            continue
        if (
            isinstance(expected, float)
            and isinstance(actual, (int, float))
            and not isinstance(actual, bool)
        ):
            if math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=0.0):
                continue
        elif actual == expected:
            continue
        raise GateError(
            f"refusing to overwrite existing short55.{key}: "
            f"current={actual!r}, validated={expected!r}"
        )


def _yaml_scalar(value: object) -> str:
    if isinstance(value, float):
        return repr(value)
    if not isinstance(value, str):
        raise GateError(f"unsupported target scalar {value!r}")
    return value


def _patch_short55(text: str, desired: dict[str, object], existing: dict[str, Any]) -> str:
    lines = text.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.rstrip() == "    short55:"]
    if len(starts) != 1:
        raise GateError("target YAML must contain exactly one four-space short55 section")
    start = starts[0]
    stop = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.match(r"^    [A-Za-z0-9_-]+:\s*(?:#.*)?$", lines[index].rstrip("\n"))
        ),
        len(lines),
    )
    section = lines[start:stop]
    key_positions: dict[str, list[int]] = {}
    for offset, line in enumerate(section):
        match = re.match(r"^      ([A-Za-z0-9_-]+):(?:\s.*)?(?:\n)?$", line)
        if match:
            key_positions.setdefault(match.group(1), []).append(start + offset)
    for key, positions in key_positions.items():
        if len(positions) > 1:
            raise GateError(f"duplicate key in short55 target section: {key}")

    for key, value in desired.items():
        current = existing.get(key)
        already_final = current == value and not isinstance(value, float)
        if (
            isinstance(value, float)
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
        ):
            already_final = math.isclose(float(current), value, rel_tol=0.0, abs_tol=0.0)
        if already_final:
            continue
        rendered = f"      {key}: {_yaml_scalar(value)}\n"
        positions = key_positions.get(key, [])
        if positions:
            lines[positions[0]] = rendered
            continue
        if key != "teacher_artifact_sha256":
            raise GateError(f"pending short55 target key is missing: {key}")
        artifact_positions = key_positions.get("teacher_artifact", [])
        if len(artifact_positions) != 1:
            raise GateError("cannot place teacher_artifact_sha256 beside teacher_artifact")
        insert_at = artifact_positions[0] + 1
        lines.insert(insert_at, rendered)
        # Keep cached absolute positions correct for subsequent replacements.
        for positions_for_key in key_positions.values():
            for index, position in enumerate(positions_for_key):
                if position >= insert_at:
                    positions_for_key[index] = position + 1
        key_positions[key] = [insert_at]
        stop += 1
    return "".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    original_mode = path.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, original_mode)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="atomically fill the pending short55 values; without this flag, only audit",
    )
    return parser.parse_args()


def _run(args: argparse.Namespace) -> dict[str, object]:
    targets_path = DEFAULT_TARGETS.resolve()
    targets, short55, evaluation_config, config_path, artifact_path = _validate_authorities(
        targets_path
    )
    if not artifact_path.is_file():
        raise GateError(f"completed short55 teacher artifact does not exist: {artifact_path}")
    try:
        artifact_bytes = artifact_path.read_bytes()
        artifact_value = json.loads(artifact_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot load short55 teacher artifact {artifact_path}: {error}") from error
    artifact = _mapping(artifact_value, label="short55 teacher artifact")
    historical_boundary = _mapping(
        targets.get("historical_study_boundary"), label="targets.historical_study_boundary"
    )
    boundary_commit = historical_boundary.get("commit")
    if not isinstance(boundary_commit, str) or not boundary_commit:
        raise GateError("targets.historical_study_boundary.commit must be a non-empty string")
    artifact_commit = _validate_artifact_git_config(
        artifact,
        config_path=str(short55["config_path"]),
        expected_sha256=str(short55["config_sha256"]),
        boundary=boundary_commit,
    )
    aggregate, names = _validate_artifact(
        artifact, short55=short55, evaluation_config=evaluation_config
    )
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
    desired = _desired_values(aggregate, artifact_sha)
    _validate_existing(short55, desired)

    before_text = targets_path.read_text(encoding="utf-8")
    before_sha = hashlib.sha256(before_text.encode()).hexdigest()
    try:
        before_value = yaml.safe_load(before_text)
    except yaml.YAMLError as error:
        raise GateError(f"target authority changed into invalid YAML: {error}") from error
    if before_value != targets:
        raise GateError("target authority changed while it was being validated; retry")
    patched_text = _patch_short55(before_text, desired, short55)

    try:
        patched_value = yaml.safe_load(patched_text)
    except yaml.YAMLError as error:
        raise GateError(f"internal patch produced invalid target YAML: {error}") from error
    expected_targets = copy.deepcopy(targets)
    expected_short55 = expected_targets["quality"]["scopes"]["short55"]
    expected_short55.update(desired)
    if patched_value != expected_targets:
        raise GateError("internal patch changed content outside the pending short55 fields")
    action = "validated_dry_run"
    after_sha = before_sha
    if args.write:
        if patched_text == before_text:
            action = "already_finalized"
        else:
            _atomic_write(targets_path, patched_text)
            action = "updated"
            after_sha = _sha256(targets_path)
            written = _load_mapping(targets_path, kind="updated target authority")
            written_short55 = _mapping(
                _mapping(
                    _mapping(written.get("quality"), label="updated targets.quality").get("scopes"),
                    label="updated targets.quality.scopes",
                ).get("short55"),
                label="updated targets.quality.scopes.short55",
            )
            for key, expected in desired.items():
                if written_short55.get(key) != expected:
                    raise GateError(f"written short55.{key} failed post-write verification")

    names_sha = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
    return {
        "status": "succeeded",
        "action": action,
        "target_authority": str(targets_path.relative_to(ROOT)),
        "target_authority_sha256_before": before_sha,
        "target_authority_sha256_after": after_sha,
        "evaluation_config": str(config_path.relative_to(ROOT)),
        "evaluation_config_sha256": _sha256(config_path),
        "teacher_artifact": str(artifact_path.relative_to(ROOT)),
        "teacher_artifact_sha256": artifact_sha,
        "artifact_git_commit": artifact_commit,
        "artifact_git_config_sha256": short55["config_sha256"],
        "model_revision": artifact["model_revision"],
        "dataset_revision": artifact["dataset_revision"],
        "mode": artifact["extra"]["mode"],
        "scope": artifact["extra"]["scope"],
        "configuration_count": len(names),
        "configuration_sequence_sha256": names_sha,
        "aggregation": "unweighted geometric mean across configurations via fsum(log(x))",
        "teacher_mase": aggregate[MASE],
        "teacher_mwql": aggregate[MWQL],
        "limit_multiplier": LIMIT_MULTIPLIER,
        "maximum_student_mase": desired["maximum_student_mase"],
        "maximum_student_mwql": desired["maximum_student_mwql"],
    }


def main() -> int:
    args = _parse_args()
    try:
        evidence = _run(args)
    except (GateError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
