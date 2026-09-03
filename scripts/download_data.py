#!/usr/bin/env python3
"""Download pinned GIFT data through the official Hugging Face interface."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from timesfm_lab.revisions import (
    GIFT_EVAL_DATASET_ID,
    GIFT_EVAL_DATASET_REVISION,
    GIFT_EVAL_PRETRAIN_ID,
    GIFT_EVAL_PRETRAIN_REVISION,
)

DATASETS = {
    "eval": (GIFT_EVAL_DATASET_ID, GIFT_EVAL_DATASET_REVISION),
    "pretrain": (GIFT_EVAL_PRETRAIN_ID, GIFT_EVAL_PRETRAIN_REVISION),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("output", type=Path, help="local data directory (must not be in Git)")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--allow-pattern",
        action="append",
        help="download only a matching path; repeat for additional patterns",
    )
    scope.add_argument(
        "--full",
        action="store_true",
        help="explicitly download the full selected dataset",
    )
    return parser.parse_args()


def download(dataset: str, output: Path, patterns: list[str] | None) -> dict[str, Any]:
    """Download an immutable dataset snapshot and return its local provenance manifest."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("install the `teacher` extra to download Hugging Face data") from error

    repo_id, revision = DATASETS[dataset]
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        local_dir=output,
    )
    manifest = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "allow_patterns": patterns,
        "full_snapshot": patterns is None,
        "resolved_local_path": str(Path(resolved_path).resolve()),
        "completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    manifest_path = output / ".timesfm-lab-download.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = download(
        dataset=args.dataset,
        output=args.output,
        patterns=None if args.full else list(args.allow_pattern),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
