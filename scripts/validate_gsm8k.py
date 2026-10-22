#!/usr/bin/env python3
"""Validate a pinned GSM8K subset manifest, schema, and split isolation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_DATASET = "openai/gsm8k"
EXPECTED_REVISION = "740312add88f781978c0658806c59bc2815b9866"
ANSWER_RE = re.compile(r"^-?\d+$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--output", type=Path, default=Path("reports/gsm8k_data_validation.json"))
    parser.add_argument(
        "--exclude-source-data",
        action="append",
        type=Path,
        default=[],
        help="Existing JSONL source rows that must not overlap any new split.",
    )
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == EXPECTED_DATASET
    assert manifest["dataset_revision"] == EXPECTED_REVISION
    rows_by_split = {}
    prompts = {}
    checks = {}
    split_names = ["train", "validation", "test", "test2"]
    split_sources = manifest.get("split_source_splits", {"train": "train", "validation": "train", "test": "test", "test2": "test"})
    for split in split_names:
        path = args.data_dir / f"{split}.jsonl"
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]
        rows_by_split[split] = rows
        prompts[split] = {row["prompt"][0]["content"] for row in rows}
        assert len(rows) == manifest["counts"][split]
        assert hashlib.sha256(raw).hexdigest() == manifest["sha256"][split]
        assert len(prompts[split]) == len(rows), f"Duplicate prompts within {split}"
        for row in rows:
            assert ANSWER_RE.fullmatch(str(row["answer"])), row["answer"]
            meta = row["meta"]
            assert meta["dataset"] == EXPECTED_DATASET and meta["dataset_revision"] == EXPECTED_REVISION
            assert meta["source_split"] == split_sources[split], (split, meta["source_split"], split_sources[split])
            assert f"<answer>...</answer>" in row["prompt"][0]["content"]
        checks[split] = {"count": len(rows), "unique_prompts": len(prompts[split]), "sha256": manifest["sha256"][split]}
    overlaps = {
        f"{left}_{right}": len(prompts[left] & prompts[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    assert not any(overlaps.values()), overlaps
    excluded_indices = set()
    for path in args.exclude_source_data:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                meta = json.loads(line)["meta"]
                excluded_indices.add((meta["source_split"], int(meta["source_index"])))
    new_source_indices = {
        (row["meta"]["source_split"], int(row["meta"]["source_index"]))
        for split in split_names
        for row in rows_by_split[split]
    }
    assert not (excluded_indices & new_source_indices), "New split overlaps a previously used source row"
    result = {
        "status": "passed",
        "dataset_id": EXPECTED_DATASET,
        "dataset_revision": EXPECTED_REVISION,
        "license": manifest["license"],
        "splits": checks,
        "question_overlaps": overlaps,
        "excluded_source_rows_verified": len(excluded_indices),
        "answer_schema": "terminal integer in the pinned GSM8K source #### marker",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
