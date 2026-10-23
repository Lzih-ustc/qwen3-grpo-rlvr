#!/usr/bin/env python3
"""Hash the Efficient RLVR code, reports, locked data, and local smoke checkpoint."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "research" / "efficient_rlvr" / "artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_ignored(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def artifact(path: Path, category: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix(),
        "category": category,
        "exists": path.is_file(),
    }
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
        record["git_ignored"] = git_ignored(path)
    return record


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return None


def collect() -> dict[str, Any]:
    research_dir = ROOT / "research" / "efficient_rlvr"
    files: dict[str, tuple[Path, str]] = {}

    for path in research_dir.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path == MANIFEST_PATH:
            continue
        suffix = path.suffix.casefold()
        if suffix == ".py":
            category = "code_or_test"
        elif suffix == ".jsonl":
            category = "generated_data_local_only"
        elif suffix == ".json":
            category = "audit_or_manifest"
        else:
            category = "documentation_or_license"
        files[path.relative_to(ROOT).as_posix()] = (path, category)

    data_dir = ROOT / "data" / "efficient-rlvr-math-v1"
    for path in data_dir.iterdir() if data_dir.exists() else []:
        if path.is_file():
            category = "generated_data_local_only" if path.suffix == ".jsonl" else "data_provenance"
            files[path.relative_to(ROOT).as_posix()] = (path, category)

    for path in (ROOT / "reports").glob("efficient_rlvr_phase2_smoke*.json"):
        files[path.relative_to(ROOT).as_posix()] = (path, "smoke_report")
    for path in (ROOT / "reports").glob("efficient_rlvr_phase3*.json"):
        files[path.relative_to(ROOT).as_posix()] = (path, "data_audit_report")
    status_path = ROOT / "docs" / "STATUS.md"
    files[status_path.relative_to(ROOT).as_posix()] = (status_path, "project_status")

    run_dir = ROOT / "runs" / "efficient-rlvr-phase2-smoke-group-filter" / "attempt-01"
    local_run_paths = [
        run_dir / "config.json",
        run_dir / "trainer_state.json",
        run_dir / "trainer" / "checkpoint-1" / "adapter_model.safetensors",
        run_dir / "trainer" / "checkpoint-1" / "optimizer.pt",
        run_dir / "trainer" / "checkpoint-1" / "scheduler.pt",
        run_dir / "trainer" / "checkpoint-1" / "rng_state.pth",
        run_dir / "final_adapter" / "adapter_model.safetensors",
    ]

    return {
        "manifest_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "branch": git_value("branch", "--show-current"),
            "head_commit": git_value("rev-parse", "HEAD"),
        },
        "historical_archive_commit": "b98a22da844106544e1a26812099a4db3bb73e60",
        "source_data": {
            "upstream_repository": "https://github.com/google-deepmind/mathematics_dataset.git",
            "upstream_revision": "427f45075f84b8b9774950196ad63867ca20ffb3",
            "license_note": "Apache-2.0 for generator code; generated-output licensing is not separately stated upstream. JSONL splits are local and ignored by Git.",
            "split_hashes": json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))["splits"],
        },
        "files": [artifact(path, category) for _, (path, category) in sorted(files.items())],
        "local_only_checkpoint": [artifact(path, "local_smoke_checkpoint") for path in local_run_paths],
        "notes": [
            "This manifest hashes the data and smoke evidence for this branch; it does not alter Phase A-C artifacts.",
            "Generated JSONL split files and binary checkpoints are intentionally excluded from Git.",
        ],
    }


def main() -> None:
    manifest = collect()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "files": len(manifest["files"]),
        "local_checkpoint_files": sum(item["exists"] for item in manifest["local_only_checkpoint"]),
        "missing_local_checkpoint_files": [item["path"] for item in manifest["local_only_checkpoint"] if not item["exists"]],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
