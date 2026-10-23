#!/usr/bin/env python3
"""Apply a pre-registered candidate rule to validation outputs only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [center - half, center + half]


def paired(left: dict, right: dict) -> dict:
    a = left["predictions"]
    b = right["predictions"]
    assert len(a) == len(b)
    both = left_only = right_only = neither = 0
    for x, y in zip(a, b):
        assert x["prompt"] == y["prompt"] and x["expected"] == y["expected"]
        if x["correct"] and y["correct"]:
            both += 1
        elif x["correct"]:
            left_only += 1
        elif y["correct"]:
            right_only += 1
        else:
            neither += 1
    n = left_only + right_only
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1)) / (2**n)) if n else 1.0
    return {
        "both_correct": both,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_wrong": neither,
        "exact_mcnemar_two_sided_p": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--candidate-validation", action="append", required=True, help="REWARD_MODE=EVAL_JSON; repeat for every locked candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = read(args.plan)
    assert plan["status"] == "locked_before_candidate_validation"
    assert not plan["test_used_for_selection"] and not plan["test_opened"]
    base = read(args.base_validation)
    expected_hash = plan["validation_sha256"]
    assert base["adapter"] is None and base["eval_data_sha256"] == expected_hash
    assert base["eval_count"] == 500
    locked = {row["reward_mode"]: row for row in plan["candidates"]}
    candidates = []
    outputs = {}
    for spec in args.candidate_validation:
        mode, raw_path = spec.split("=", 1)
        assert mode in locked and mode not in outputs, f"unexpected/duplicate reward mode: {mode}"
        result = read(Path(raw_path))
        lock = locked[mode]
        assert result["adapter"] == lock["adapter"]
        assert result["eval_data_sha256"] == expected_hash and result["eval_count"] == 500
        assert result["decode"]["do_sample"] is False and result["decode"]["max_new_tokens"] == 192
        assert result["seed"] == plan["decode"]["seed"]
        assert [r["prompt"] for r in result["predictions"]] == [r["prompt"] for r in base["predictions"]]
        assert [r["expected"] for r in result["predictions"]] == [r["expected"] for r in base["predictions"]]
        item = {
            "reward_mode": mode,
            "adapter": lock["adapter"],
            "optimizer_updates": lock["optimizer_updates"],
            "training_generated_tokens": lock["generated_tokens"],
            "validation_correct_count": result["correct_count"],
            "validation_count": result["eval_count"],
            "validation_accuracy": result["accuracy"],
            "validation_wilson_95_ci": wilson(result["correct_count"], result["eval_count"]),
            "parse_rate": result["parse_rate"],
            "generated_tokens": result["generated_tokens"],
            "tokens_per_second": result["tokens_per_second"],
            "elapsed_seconds": result["elapsed_seconds"],
            "torch_peak_allocated_mib": result["torch_peak_allocated_mib"],
            "torch_peak_reserved_mib": result["torch_peak_reserved_mib"],
            "whole_gpu_peak_memory_used_mib": result["gpu_monitor"]["whole_gpu_peak_memory_used_mib"],
            "validation_sha256": expected_hash,
        }
        candidates.append(item)
        outputs[mode] = result
    assert set(outputs) == set(locked), "all locked candidates must be evaluated before selection"
    selected = min(candidates, key=lambda row: (-row["validation_accuracy"], row["training_generated_tokens"], row["optimizer_updates"]))
    selected_mode = selected["reward_mode"]
    summary = {
        "status": "validation_selected_test_still_sealed",
        "plan": str(args.plan),
        "validation_sha256": expected_hash,
        "selection_metric": plan["candidate_selection_metric"],
        "selection_rule": plan["selection_rule"],
        "base_reference": {
            "correct_count": base["correct_count"],
            "eval_count": base["eval_count"],
            "accuracy": base["accuracy"],
            "parse_rate": base["parse_rate"],
            "wilson_95_ci": wilson(base["correct_count"], base["eval_count"]),
        },
        "candidates": sorted(candidates, key=lambda row: row["reward_mode"]),
        "paired_candidate_comparisons": {
            "distance_shaped_vs_exact_format": paired(outputs["distance_shaped"], outputs["exact_format"]),
            "base_vs_selected": paired(base, outputs[selected_mode]),
        },
        "selected": selected,
        "selection_used_test": False,
        "test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected_mode, "accuracy": selected["validation_accuracy"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
