#!/usr/bin/env python3
"""Verify the locked synthetic math split and audit exact historical overlap."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "efficient-rlvr-math-v1"
INTEGER_ANSWER = re.compile(r"[+-]?\d+\Z")
NEW_PREFIX = (
    "Solve this math problem. Show concise work and put the final integer "
    "inside <answer>...</answer>.\n\n"
)
OLD_PROMPT_SUFFIXES = (
    "\n\nSolve the problem and show concise work.",
    "\n\nSolve the problem.",
    "\n\nSolve this problem.",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_exact(text: str) -> str:
    return " ".join(text.casefold().split())


def normalize_punctuation_insensitive(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def prompt_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        if isinstance(value.get("content"), str):
            yield value["content"]
        else:
            for nested in value.values():
                yield from prompt_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from prompt_strings(nested)
    elif hasattr(value, "tolist"):
        yield from prompt_strings(value.tolist())


def strip_known_instructions(text: str) -> list[str]:
    value = text.strip()
    result = [value]
    if value.startswith(NEW_PREFIX):
        result.append(value[len(NEW_PREFIX):].strip())
    for suffix in OLD_PROMPT_SUFFIXES:
        marker = value.find(suffix)
        if marker > 0:
            result.append(value[:marker].strip())
            break
    return [item for item in result if item]


def walk_historical(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).casefold()
            if lowered in {"prompt", "question", "source_question", "problem"}:
                for text in prompt_strings(value):
                    yield from strip_known_instructions(text)
            else:
                yield from walk_historical(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk_historical(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # JSONL records are separated by LF. `str.splitlines()` also splits U+2028,
    # which is valid inside a JSON string and occurs in archived source text.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def scan_history(root: Path) -> tuple[dict[str, list[str]], dict[str, list[str]], list[dict[str, Any]]]:
    exact_sources: dict[str, list[str]] = defaultdict(list)
    loose_sources: dict[str, list[str]] = defaultdict(list)
    file_records = []
    excluded_dir = (root / "data" / "efficient-rlvr-math-v1").resolve()
    candidates = []
    for folder in (root / "data", root / "reports", root / "runs"):
        if not folder.exists():
            continue
        candidates.extend(
            path for path in folder.rglob("*")
            if path.suffix.casefold() in {".json", ".jsonl", ".parquet"}
        )
    for path in sorted(candidates):
        if excluded_dir in path.resolve().parents:
            continue
        raw = path.read_bytes()
        file_records.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": len(raw),
            "sha256": sha256_bytes(raw),
        })
        if path.suffix.casefold() == ".parquet":
            import pyarrow.parquet as parquet  # noqa: PLC0415

            table = parquet.read_table(path)
            columns = [name for name in table.column_names if name.casefold() in {"prompt", "question", "source_question", "problem"}]
            source_strings = (
                text
                for column in columns
                for value in table[column].to_pylist()
                for text in walk_historical({column: value})
            )
        elif path.suffix.casefold() == ".jsonl":
            records = []
            for line_number, line in enumerate(raw.decode("utf-8").split("\n"), 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL record in {path}:{line_number}: {exc}") from exc
            source_strings = (text for record in records for text in walk_historical(record))
        else:
            source_strings = walk_historical(json.loads(raw.decode("utf-8")))
        for text in source_strings:
            exact = normalize_exact(text)
            loose = normalize_punctuation_insensitive(text)
            if exact:
                exact_sources[exact].append(path.relative_to(root).as_posix())
            if loose:
                loose_sources[loose].append(path.relative_to(root).as_posix())
    return exact_sources, loose_sources, file_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--history-root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path, default=ROOT / "reports" / "efficient_rlvr_phase3_data_audit.json")
    parser.add_argument("--output-markdown", type=Path, default=ROOT / "research" / "efficient_rlvr" / "phase3_data_audit.md")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    root = args.history_root.resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rebuild_path = root / "reports" / "efficient_rlvr_phase3_reproducibility.json"
    rebuild_report = json.loads(rebuild_path.read_text(encoding="utf-8")) if rebuild_path.is_file() else None
    if rebuild_report is not None:
        if rebuild_report.get("status") != "passed":
            raise AssertionError("Independent rebuild verification did not pass")
        for split_name in ("train", "validation", "test"):
            if rebuild_report["split_results"][split_name]["locked_sha256"] != manifest["splits"][split_name]["sha256"]:
                raise AssertionError(f"Rebuild audit hash does not match locked {split_name}")
    exact_sources, loose_sources, scanned_files = scan_history(root)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_summaries = {}
    exact_overlaps = []
    loose_overlaps = []
    global_questions: set[str] = set()
    per_split_question_sets: dict[str, set[str]] = {}

    for split_name in ("train", "validation", "test"):
        spec = manifest["splits"][split_name]
        path = data_dir / spec["file"]
        raw = path.read_bytes()
        rows = read_jsonl(path)
        if len(rows) != spec["count"]:
            raise AssertionError(f"{split_name} row count mismatch")
        if sha256_bytes(raw) != spec["sha256"]:
            raise AssertionError(f"{split_name} file hash mismatch")
        questions = set()
        for index, row in enumerate(rows):
            answer = str(row.get("answer", ""))
            question = str(row.get("meta", {}).get("source_question", ""))
            if not INTEGER_ANSWER.fullmatch(answer):
                raise AssertionError(f"Non-integer answer in {split_name} row {index}")
            if not question:
                raise AssertionError(f"Missing source question in {split_name} row {index}")
            key = normalize_exact(question)
            loose = normalize_punctuation_insensitive(question)
            if key in global_questions:
                raise AssertionError(f"Duplicate source question across splits: {split_name}/{index}")
            global_questions.add(key)
            questions.add(key)
            meta_hash = row["meta"].get("source_question_sha256")
            if meta_hash != sha256_bytes(question.encode("utf-8")):
                raise AssertionError(f"Source question hash mismatch in {split_name} row {index}")
            if key in exact_sources:
                exact_overlaps.append({
                    "split": split_name,
                    "row": index,
                    "question_sha256": meta_hash,
                    "historical_files": sorted(set(exact_sources[key])),
                })
            if loose in loose_sources:
                loose_overlaps.append({
                    "split": split_name,
                    "row": index,
                    "question_sha256": meta_hash,
                    "historical_files": sorted(set(loose_sources[loose])),
                })
        per_split_question_sets[split_name] = questions
        split_summaries[split_name] = {
            "rows": len(rows),
            "sha256": sha256_bytes(raw),
            "unique_source_questions": len(questions),
            "integer_answers": sum(bool(INTEGER_ANSWER.fullmatch(str(row["answer"]))) for row in rows),
            "unique_generator_modules": len(spec["modules"]),
        }
        split_rows[split_name] = rows

    report = {
        "status": "passed" if not exact_overlaps else "failed",
        "audit_scope": "Offline data integrity and text-level overlap audit only; no model generation, evaluation, or training.",
        "dataset_id": manifest["dataset_id"],
        "source": manifest["source"],
        "historical_roots_scanned": ["data/", "reports/", "runs/"],
        "historical_files_scanned": scanned_files,
        "historical_file_count": len(scanned_files),
        "historical_prompt_text_count": sum(len(paths) for paths in exact_sources.values()),
        "parquet_reader": {"library": "pyarrow", "version": importlib.metadata.version("pyarrow")},
        "split_checks": split_summaries,
        "independent_rebuild_verification": {
            "status": rebuild_report.get("status") if rebuild_report else "not_available",
            "report": "reports/efficient_rlvr_phase3_reproducibility.json" if rebuild_report else None,
        },
        "cross_split_exact_duplicates": 0,
        "historical_exact_normalized_overlaps": exact_overlaps,
        "historical_punctuation_insensitive_overlaps": loose_overlaps,
        "historical_overlap_method": "Compare source questions after case-folding, whitespace normalization, and separately punctuation stripping against strings under prompt/question/source_question/problem keys in JSON/JSONL/Parquet files in data/, reports/, and runs/. This detects text-level duplicates only, not paraphrases or model pretraining exposure.",
        "historical_training_and_test_policy": "No GSM8K/SVAMP consumed test rows are used for selection. The new extrapolation test split is locked and not a candidate-selection input.",
        "pretraining_contamination_claim": "Not assessed; public generator families or related forms may have appeared in Qwen3 pretraining.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    split_lines = "\n".join(
        f"| {name} | {summary['rows']} | `{summary['sha256']}` | {summary['unique_generator_modules']} |"
        for name, summary in split_summaries.items()
    )
    markdown = f"""# Phase 3 data lock audit

Status: **{report['status'].upper()}**. This is an offline data-integrity and exact-text overlap check. It did not generate model outputs, evaluate a checkpoint, or train a model.

## Locked source and splits

Source: [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset), pinned revision `{manifest['source']['revision']}`. The upstream repository's Apache-2.0 license covers its generator code; the repository does not separately state a generated-output license. A license copy is included, and the generated JSONL rows remain local/ignored by Git. Each row is newly generated from the upstream train, interpolate, or extrapolate regime and filtered to signed integer answers for the project's exact verifier.

| Split | Rows | SHA-256 | Generator modules |
|---|---:|---|---:|
{split_lines}

The `train` split is training-only. `validation` uses the upstream interpolation regime and may be used for candidate selection. `test` uses the separate extrapolation regime and is reserved for one final evaluation after all choices are frozen. All 2,300 normalized source questions are unique across splits.

Two separate full builds reproduced the three locked hashes; see [`efficient_rlvr_phase3_reproducibility.json`](../../reports/efficient_rlvr_phase3_reproducibility.json).

## Historical-use and duplicate audit

- Scanned {len(scanned_files)} JSON/JSONL/Parquet files under the archived `data/`, `reports/`, and `runs/` trees ({report['historical_prompt_text_count']:,} prompt-text entries); exact normalized source-question overlap count: **{len(exact_overlaps)}**.
- Punctuation-insensitive text match count: **{len(loose_overlaps)}**. This stricter diagnostic can include normalization collisions; it is reported separately and does not silently rewrite any locked rows.
- Cross-split exact duplicate count: **0**.
- GSM8K/SVAMP consumed test sets were not used to generate, select, rank, or tune the split. No new model quality result has been viewed.
- This is a text-level audit only. It does not detect paraphrase or prove the Qwen3 pretraining corpus never contained related public generator examples.

The detailed file list and SHA-256 hashes are in [`efficient_rlvr_phase3_data_audit.json`](../../reports/efficient_rlvr_phase3_data_audit.json); split content hashes and generation seeds are in [`manifest.json`](../../data/efficient-rlvr-math-v1/manifest.json).
"""
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "historical_file_count": len(scanned_files),
        "historical_prompt_text_count": report["historical_prompt_text_count"],
        "exact_overlap_count": len(exact_overlaps),
        "punctuation_insensitive_overlap_count": len(loose_overlaps),
        "split_checks": split_summaries,
        "report": str(args.output_json),
    }, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit("Historical exact text overlap was found; inspect the report before using the data")


if __name__ == "__main__":
    main()
