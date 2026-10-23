#!/usr/bin/env python3
"""Audit paired base/adapted evaluation outputs for one held-out split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [center - half, center + half]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=Path("data/gsm8k-next-fresh/manifest.json"))
    args = parser.parse_args()
    selection, base, selected = load(args.selection), load(args.base), load(args.selected)
    expected_adapter = selection["selected"]["adapter"]
    test_manifest = load(args.test_manifest)
    expected_hash = test_manifest.get("sha256", {}).get("test", test_manifest.get("output_sha256"))
    if not expected_hash:
        raise ValueError("Test manifest must provide sha256.test or output_sha256")
    assert selection["test_opened"] is False and selection["selection_used_test"] is False
    assert base["adapter"] is None and selected["adapter"] == expected_adapter
    assert base["eval_data_sha256"] == selected["eval_data_sha256"] == expected_hash
    assert base["eval_count"] == selected["eval_count"] == 1000
    assert base["decode"] == selected["decode"]
    assert [row["prompt"] for row in base["predictions"]] == [row["prompt"] for row in selected["predictions"]]

    both = base_only = selected_only = neither = 0
    for x, y in zip(base["predictions"], selected["predictions"]):
        assert x["expected"] == y["expected"]
        if x["correct"] and y["correct"]:
            both += 1
        elif x["correct"]:
            base_only += 1
        elif y["correct"]:
            selected_only += 1
        else:
            neither += 1
    discordant = base_only + selected_only
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(base_only, selected_only) + 1)) / (2**discordant)) if discordant else 1.0
    report = {
        "status": "paired_test_audit_passed",
        "selection_json": str(args.selection),
        "selected_reward_mode": selection["selected"]["reward_mode"],
        "selected_candidate_id": selection["selected"].get("candidate_id"),
        "selected_group_size": selection["selected"].get("group_size"),
        "selected_adapter": expected_adapter,
        "test_sha256": expected_hash,
        "eval_count": base["eval_count"],
        "decode": base["decode"],
        "base": {
            "correct_count": base["correct_count"],
            "accuracy": base["accuracy"],
            "wilson_95_ci": wilson(base["correct_count"], base["eval_count"]),
            "parse_rate": base["parse_rate"],
            "generated_tokens": base["generated_tokens"],
            "tokens_per_second": base["tokens_per_second"],
            "elapsed_seconds": base["elapsed_seconds"],
            "torch_peak_allocated_mib": base["torch_peak_allocated_mib"],
            "torch_peak_reserved_mib": base["torch_peak_reserved_mib"],
            "whole_gpu_peak_memory_used_mib": base["gpu_monitor"]["whole_gpu_peak_memory_used_mib"],
        },
        "selected": {
            "correct_count": selected["correct_count"],
            "accuracy": selected["accuracy"],
            "wilson_95_ci": wilson(selected["correct_count"], selected["eval_count"]),
            "parse_rate": selected["parse_rate"],
            "generated_tokens": selected["generated_tokens"],
            "tokens_per_second": selected["tokens_per_second"],
            "elapsed_seconds": selected["elapsed_seconds"],
            "torch_peak_allocated_mib": selected["torch_peak_allocated_mib"],
            "torch_peak_reserved_mib": selected["torch_peak_reserved_mib"],
            "whole_gpu_peak_memory_used_mib": selected["gpu_monitor"]["whole_gpu_peak_memory_used_mib"],
        },
        "accuracy_delta_percentage_points": 100 * (selected["accuracy"] - base["accuracy"]),
        "paired_outcomes": {
            "both_correct": both,
            "base_only_correct": base_only,
            "selected_only_correct": selected_only,
            "both_wrong": neither,
        },
        "exact_mcnemar_two_sided_p": p,
        "selection_used_test": False,
        "test_opened_after_selection": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "accuracy_delta_pp": report["accuracy_delta_percentage_points"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
