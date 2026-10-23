#!/usr/bin/env python3
"""Check a holdout against prior training data and saved model outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    # JSONL strings may contain Unicode line separator U+2028; split only on LF.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def source_key(row: dict) -> tuple[str, ...] | None:
    meta = row.get("meta", {})
    dataset = str(meta.get("dataset", ""))
    if dataset == "openai/gsm8k" and "source_index" in meta:
        return dataset, str(meta.get("source_split", "")), str(meta["source_index"])
    if dataset and "source_id" in meta:
        return dataset, str(meta.get("dataset_revision", "")), str(meta["source_id"])
    return None


def question(text: str) -> str:
    text = text.split("\n\nSolve the problem", 1)[0].strip()
    return " ".join(text.split())


def prompt_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return str(value[0].get("content", ""))
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/gsm8k-next-fresh"))
    parser.add_argument("--test-file", type=Path, help="Optional explicit holdout JSONL path.")
    parser.add_argument("--expected-count", type=int, help="Expected number of holdout rows.")
    parser.add_argument("--output", type=Path, default=Path("reports/gsm8k_next_fresh_holdout_audit.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--exclude-eval-report",
        action="append",
        type=Path,
        default=[],
        help="Current holdout evaluation reports to exclude while scanning prior predictions.",
    )
    args = parser.parse_args()

    test_path = args.test_file or (args.data_dir / "test.jsonl")
    test_rows = rows(test_path)
    test_keys = {key for row in test_rows if (key := source_key(row)) is not None}
    test_questions = {question(prompt_text(row["prompt"])) for row in test_rows}
    expected_count = args.expected_count if args.expected_count is not None else (1000 if args.test_file is None else len(test_rows))
    if len(test_rows) != expected_count or len(test_questions) != len(test_rows):
        raise SystemExit(
            f"Holdout row/question count mismatch: rows={len(test_rows)}, "
            f"unique_questions={len(test_questions)}, expected={expected_count}"
        )
    if len(test_keys) not in (0, len(test_rows)):
        raise SystemExit(f"Holdout source identity count mismatch: {len(test_keys)} of {len(test_rows)} rows have unique IDs.")

    training_files = sorted(Path("data").rglob("train.jsonl"))
    training_overlaps = {}
    training_question_overlaps = {}
    for path in training_files:
        training_rows = rows(path)
        keys = {key for row in training_rows if (key := source_key(row)) is not None}
        n = len(test_keys & keys)
        if n:
            training_overlaps[path.as_posix()] = n
        train_questions = {question(prompt_text(row.get("prompt", ""))) for row in training_rows}
        question_overlap = len(test_questions & train_questions)
        if question_overlap:
            training_question_overlaps[path.as_posix()] = question_overlap

    eval_files = []
    eval_overlap_details = []
    eval_rows_seen = 0
    excluded_eval_reports = {path.resolve() for path in args.exclude_eval_report}
    output_path = args.output.resolve()
    for path in sorted(args.reports_dir.rglob("*.json")):
        if path.resolve() == output_path or path.resolve() in excluded_eval_reports:
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        predictions = result.get("predictions") if isinstance(result, dict) else None
        if not isinstance(predictions, list):
            continue
        eval_files.append(path.as_posix())
        eval_rows_seen += len(predictions)
        found = sum(
            question(prompt_text(row.get("prompt", ""))) in test_questions
            for row in predictions
            if isinstance(row, dict)
        )
        if found:
            eval_overlap_details.append({"report": path.as_posix(), "overlapping_questions": found})

    intermediate_path = Path("data/gsm8k-next/test.jsonl")
    intermediate_overlap = 0
    is_gsm8k_holdout = bool(test_rows) and all(
        row.get("meta", {}).get("dataset") == "openai/gsm8k" for row in test_rows
    )
    if is_gsm8k_holdout and intermediate_path.is_file():
        intermediate_keys = {key for row in rows(intermediate_path) if (key := source_key(row)) is not None}
        intermediate_overlap = len(test_keys & intermediate_keys)

    manifest_path = args.data_dir / "manifest.json"
    holdout_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if args.test_file is None:
                holdout_sha256 = manifest.get("sha256", {}).get("test") or holdout_sha256
            elif test_path.name == "validation.jsonl":
                holdout_sha256 = manifest.get("sha256", {}).get("validation") or holdout_sha256
            else:
                holdout_sha256 = manifest.get("output_sha256") or holdout_sha256
        except (OSError, json.JSONDecodeError):
            pass

    report = {
        "status": "passed" if not training_overlaps and not training_question_overlaps and not eval_overlap_details else "failed",
        "holdout_file": test_path.as_posix(),
        "fresh_test_rows": len(test_rows),
        "expected_count": expected_count,
        "fresh_test_source_index_unique": len(test_keys),
        "historical_training_split_files_checked": [path.as_posix() for path in training_files],
        "fresh_test_overlaps_with_historical_training_splits": training_overlaps,
        "fresh_test_exact_question_overlaps_with_historical_training_splits": training_question_overlaps,
        "historical_model_evaluation_reports_checked": eval_files,
        "current_holdout_evaluation_reports_excluded_from_history_scan": sorted(
            path.as_posix() for path in excluded_eval_reports
        ),
        "historical_evaluation_prediction_rows_scanned": eval_rows_seen,
        "fresh_test_overlaps_with_historical_model_evaluation_outputs": eval_overlap_details,
        "prepared_but_superseded_intermediate_test_overlap": {
            "path": intermediate_path.as_posix(),
            "rows_overlapping": intermediate_overlap,
            "note": "This legacy GSM8K intermediate split had no model-evaluation output and was not used in a training split; its overlap is disclosed for provenance.",
        },
        "test_sha256": holdout_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "overlaps": {
        "training_sources": training_overlaps,
        "training_questions": training_question_overlaps,
        "historical_predictions": eval_overlap_details,
    }}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
