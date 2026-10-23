#!/usr/bin/env python3
"""Compare two independent deterministic builds with the locked split hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "validation", "test")


def hash_split(directory: Path, manifest: dict[str, Any], split: str) -> str:
    filename = manifest["splits"][split]["file"]
    return hashlib.sha256((directory / filename).read_bytes()).hexdigest()


def verify_rebuilds(locked: Path, first: Path, second: Path) -> dict[str, Any]:
    manifest = json.loads((locked / "manifest.json").read_text(encoding="utf-8"))
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source"]["revision"] != first_manifest["source"]["revision"]:
        raise AssertionError("First build has a different source revision")
    if manifest["source"]["revision"] != second_manifest["source"]["revision"]:
        raise AssertionError("Second build has a different source revision")
    split_results = {}
    for split in SPLITS:
        locked_hash = manifest["splits"][split]["sha256"]
        first_hash = hash_split(first, first_manifest, split)
        second_hash = hash_split(second, second_manifest, split)
        actual_locked_hash = hash_split(locked, manifest, split)
        result = {
            "locked_sha256": locked_hash,
            "build_a_sha256": first_hash,
            "build_b_sha256": second_hash,
            "locked_file_matches_manifest": actual_locked_hash == locked_hash,
            "both_rebuilds_match_locked": first_hash == second_hash == locked_hash,
        }
        split_results[split] = result
    passed = all(row["both_rebuilds_match_locked"] and row["locked_file_matches_manifest"] for row in split_results.values())
    return {
        "status": "passed" if passed else "failed",
        "source_revision": manifest["source"]["revision"],
        "python_hash_seed": manifest["generation"]["python_hash_seed"],
        "independent_rebuild_count": 2,
        "split_results": split_results,
        "limitations": [
            "Reproducibility is verified for the pinned source, Python/dependency versions, environment, and split seeds recorded in manifest.json.",
            "The upstream _composed generator leaves were excluded because their output ordering depends on identity-hashed object set iteration.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked", type=Path, default=ROOT / "data" / "efficient-rlvr-math-v1")
    parser.add_argument("--build-a", type=Path, required=True)
    parser.add_argument("--build-b", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=ROOT / "reports" / "efficient_rlvr_phase3_reproducibility.json")
    args = parser.parse_args()
    report = verify_rebuilds(args.locked.resolve(), args.build_a.resolve(), args.build_b.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit("Independent data rebuilds do not match the locked hashes")


if __name__ == "__main__":
    main()
