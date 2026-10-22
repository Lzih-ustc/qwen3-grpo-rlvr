#!/usr/bin/env python3
"""Build a SHA-256 inventory of the reproducibility artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files() -> list[Path]:
    paths = [ROOT / "README.md", ROOT / ".gitignore", ROOT / ".gitattributes", *ROOT.glob("requirements-wsl2*.txt")]
    for folder in ("docs", "scripts", "data", "reports", "runs"):
        paths.extend((ROOT / folder).rglob("*"))
    return sorted(
        path
        for path in set(paths)
        if path.is_file()
        and path != MANIFEST
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def main() -> None:
    files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in included_files()
    ]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "files": files,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "file_count": len(files), "manifest": str(MANIFEST)}))


if __name__ == "__main__":
    main()
