#!/usr/bin/env python3
"""Verify SHA-256 and file sizes recorded by reports/artifact_manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(entries: list[dict[str, Any]], label: str) -> list[str]:
    errors = []
    for entry in entries:
        relative = Path(entry["path"])
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            errors.append(f"path escapes project root: {entry['path']}")
            continue
        if not path.is_file():
            errors.append(f"missing {label} file: {entry['path']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            errors.append(f"size mismatch: {entry['path']}")
            continue
        if sha256(path) != entry["sha256"]:
            errors.append(f"SHA-256 mismatch: {entry['path']}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-only", action="store_true", help="Skip model/checkpoint binaries not shipped in Git")
    args = parser.parse_args()
    report = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if report.get("schema_version") != 2:
        raise SystemExit("Unsupported artifact manifest schema; rebuild with scripts/build_manifest.py")
    portable = report.get("files", [])
    local = report.get("local_only_artifacts", [])
    errors = verify(portable, "portable")
    if not args.portable_only:
        errors.extend(verify(local, "local-only"))
    result = {
        "status": "failed" if errors else "passed",
        "portable_verified": len(portable) if not errors else None,
        "local_only_verified": None if args.portable_only else len(local) if not errors else None,
        "local_only_skipped": len(local) if args.portable_only else 0,
        "errors": errors[:50],
        "manifest": "reports/artifact_manifest.json",
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
