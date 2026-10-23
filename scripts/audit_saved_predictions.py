#!/usr/bin/env python3
"""Recheck paired evaluator outputs and export a deterministic manual sample."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from math_task import extract_answer


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Label=eval.json")
    parser.add_argument("--right", required=True, help="Label=eval.json")
    parser.add_argument("--samples-per-stratum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left_label, left_file = args.left.split("=", 1)
    right_label, right_file = args.right.split("=", 1)
    left, right = load(Path(left_file)), load(Path(right_file))
    assert left["eval_data_sha256"] == right["eval_data_sha256"]
    assert left["decode"] == right["decode"]
    assert [row["prompt"] for row in left["predictions"]] == [row["prompt"] for row in right["predictions"]]
    assert [row["expected"] for row in left["predictions"]] == [row["expected"] for row in right["predictions"]]

    evals = {left_label: left, right_label: right}
    parser_stats = {}
    for label, result in evals.items():
        parsed_mismatch = 0
        correct_mismatch = 0
        sources: Counter[str] = Counter()
        malformed_marker = 0
        visible_think = 0
        for row in result["predictions"]:
            value, source = extract_answer(row["completion"])
            parsed_mismatch += value != row["predicted"] or source != row["parse_source"]
            correct_mismatch += row["correct"] != (value == row["expected"])
            sources[source or "unparsed"] += 1
            malformed_marker += "<answer" in row["completion"].lower() and value is None
            visible_think += "<think>" in row["completion"] or "</think>" in row["completion"]
        assert parsed_mismatch == 0 and correct_mismatch == 0
        parser_stats[label] = {
            "tag": result["tag"],
            "adapter": result["adapter"],
            "correct_count": result["correct_count"],
            "eval_count": result["eval_count"],
            "accuracy": result["accuracy"],
            "parse_rate": result["parse_rate"],
            "parse_source_counts": dict(sources),
            "parser_mismatch_count": parsed_mismatch,
            "correctness_flag_mismatch_count": correct_mismatch,
            "unparsed_completion_with_answer_tag_markup": malformed_marker,
            "completions_with_visible_think_tags": visible_think,
            "decode": result["decode"],
            "elapsed_seconds": result["elapsed_seconds"],
            "generated_tokens": result["generated_tokens"],
            "tokens_per_second": result["tokens_per_second"],
            "torch_peak_allocated_mib": result["torch_peak_allocated_mib"],
            "whole_gpu_peak_memory_used_mib": result["gpu_monitor"]["whole_gpu_peak_memory_used_mib"],
        }

    strata: dict[str, list[int]] = {key: [] for key in ("both_correct", "left_only_correct", "right_only_correct", "both_wrong")}
    for index, (x, y) in enumerate(zip(left["predictions"], right["predictions"], strict=True)):
        if x["correct"] and y["correct"]:
            key = "both_correct"
        elif x["correct"]:
            key = "left_only_correct"
        elif y["correct"]:
            key = "right_only_correct"
        else:
            key = "both_wrong"
        strata[key].append(index)
    rng = random.Random(args.seed)
    sampled_indices = []
    for key, indices in strata.items():
        # Prioritize parse failures within each paired correctness stratum,
        # then fill the remainder with deterministic random items.
        failures = [i for i in indices if left["predictions"][i]["predicted"] is None or right["predictions"][i]["predicted"] is None]
        rng.shuffle(failures)
        chosen = failures[: args.samples_per_stratum]
        chosen_set = set(chosen)
        rest = [i for i in indices if i not in chosen_set]
        rng.shuffle(rest)
        chosen.extend(rest[: max(0, args.samples_per_stratum - len(chosen))])
        sampled_indices.extend(chosen)

    sample = []
    for index in sorted(set(sampled_indices)):
        row = {"index": index, "prompt": left["predictions"][index]["prompt"], "expected": left["predictions"][index]["expected"]}
        for label, result in evals.items():
            item = result["predictions"][index]
            row[label] = {
                "completion": item["completion"],
                "predicted": item["predicted"],
                "parse_source": item["parse_source"],
                "correct": item["correct"],
            }
        sample.append(row)

    report = {
        "status": "saved_prediction_audit_passed",
        "left_label": left_label,
        "right_label": right_label,
        "eval_data_sha256": left["eval_data_sha256"],
        "eval_count": left["eval_count"],
        "paired_outcomes": {key: len(values) for key, values in strata.items()},
        "parser_and_decode_audit": parser_stats,
        "manual_sample": {
            "seed": args.seed,
            "samples_per_stratum": args.samples_per_stratum,
            "sampled_question_count": len(sample),
            "strata": {key: [i for i in values if i in set(sampled_indices)] for key, values in strata.items()},
            "rows": sample,
            "note": "Offline sample of saved paired generations. Parser parity is exhaustively checked; semantic correctness is not independently re-derived by this script.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "eval_count": report["eval_count"], "sampled_question_count": len(sample), "parser_audit": {key: {"correct_count": value["correct_count"], "parse_rate": value["parse_rate"], "parser_mismatch_count": value["parser_mismatch_count"], "visible_think": value["completions_with_visible_think_tags"]} for key, value in parser_stats.items()}, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
