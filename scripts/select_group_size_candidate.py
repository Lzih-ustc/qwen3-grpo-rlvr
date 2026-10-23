#!/usr/bin/env python3
"""Select a matched-token generation-group candidate on validation only."""

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
    both = left_only = right_only = neither = 0
    for x, y in zip(left["predictions"], right["predictions"]):
        assert x["prompt"] == y["prompt"] and x["expected"] == y["expected"]
        if x["correct"] and y["correct"]:
            both += 1
        elif x["correct"]:
            left_only += 1
        elif y["correct"]:
            right_only += 1
        else:
            neither += 1
    discordant = left_only + right_only
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / (2**discordant)) if discordant else 1.0
    return {"both_correct": both, "left_only_correct": left_only, "right_only_correct": right_only, "both_wrong": neither, "exact_mcnemar_two_sided_p": p}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--candidate-validation", action="append", required=True, help="CANDIDATE_ID=EVAL_JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = read(args.plan)
    assert plan["status"] == "locked_before_training_and_candidate_validation"
    assert not plan["test_used_for_selection"] and not plan["test_opened"]
    base = read(args.base_validation)
    assert base["adapter"] is None and base["eval_count"] == 500
    assert base["eval_data_sha256"] == plan["validation_sha256"]
    assert base["decode"]["do_sample"] is False and base["decode"]["max_new_tokens"] == 192
    assert base["seed"] == plan["seed"]
    by_id = {row["candidate_id"]: row for row in plan["candidates"]}
    outputs, candidates = {}, []
    for spec in args.candidate_validation:
        candidate_id, raw_path = spec.split("=", 1)
        assert candidate_id in by_id and candidate_id not in outputs
        lock = by_id[candidate_id]
        assert lock["adapter"] and lock["run_dir"] and lock["group_size"]
        training = read(Path(lock["run_dir"]) / "run_summary.json")
        assert training["status"] == "success" and training["seed"] == plan["seed"]
        assert training["reward_mode"] == plan["reward_mode"]
        assert training["dataset"]["sha256"] == plan["train_data_sha256"]
        assert training["attempt_config"]["num_generations"] == lock["group_size"]
        assert training["attempt_config"]["gradient_accumulation_steps"] == lock["group_size"]
        assert training["attempt_config"]["prompt_token_limit"] == plan["prompt_token_limit"]
        assert training["attempt_config"]["max_completion_length"] == plan["max_completion_length"]
        assert training["attempt_config"]["max_generated_tokens"] == plan["generated_token_budget_target"]
        assert training["oom_fallbacks"] == []
        assert training["optimizer_updates"] > 0 and training["observed_completion_tokens"] > 0
        result = read(Path(raw_path))
        assert result["adapter"] == lock["adapter"]
        assert result["eval_data_sha256"] == plan["validation_sha256"] and result["eval_count"] == 500
        assert result["decode"] == base["decode"] and result["seed"] == plan["seed"]
        assert [r["prompt"] for r in result["predictions"]] == [r["prompt"] for r in base["predictions"]]
        assert [r["expected"] for r in result["predictions"]] == [r["expected"] for r in base["predictions"]]
        item = {
            **lock,
            "optimizer_updates": training["optimizer_updates"],
            "generated_tokens": training["observed_completion_tokens"],
            "generation_completions": training["observed_generation_completion_count"],
            "training_completion_tokens_per_second": training["completion_tokens_per_second"],
            "training_completions_per_second": training["completions_per_second"],
            "training_steps_per_second": training["steps_per_second"],
            "training_elapsed_seconds": training["train_seconds"],
            "training_torch_peak_allocated_mib": training["torch_peak_allocated_mib"],
            "training_torch_peak_reserved_mib": training["torch_peak_reserved_mib"],
            "training_whole_gpu_peak_memory_used_mib": training["gpu_monitor"]["whole_gpu_peak_memory_used_mib"],
            "reward_mode": plan["reward_mode"],
            "validation_correct_count": result["correct_count"],
            "validation_count": result["eval_count"],
            "validation_accuracy": result["accuracy"],
            "validation_wilson_95_ci": wilson(result["correct_count"], result["eval_count"]),
            "parse_rate": result["parse_rate"],
            "eval_generated_tokens": result["generated_tokens"],
            "eval_tokens_per_second": result["tokens_per_second"],
            "eval_elapsed_seconds": result["elapsed_seconds"],
            "torch_peak_allocated_mib": result["torch_peak_allocated_mib"],
            "torch_peak_reserved_mib": result["torch_peak_reserved_mib"],
            "whole_gpu_peak_memory_used_mib": result["gpu_monitor"]["whole_gpu_peak_memory_used_mib"],
        }
        candidates.append(item)
        outputs[candidate_id] = result
    assert set(outputs) == set(by_id)
    selected = min(candidates, key=lambda row: (-row["validation_accuracy"], row["generated_tokens"], row["optimizer_updates"]))
    group4 = outputs["group4"]
    group8 = outputs["group8"]
    report = {
        "status": "validation_selected_test_still_sealed",
        "selection_plan": str(args.plan),
        "validation_sha256": plan["validation_sha256"],
        "selection_metric": plan["selection_metric"],
        "selection_rule": plan["selection_rule"],
        "matched_budget_target_tokens": plan["generated_token_budget_target"],
        "base_reference": {
            "correct_count": base["correct_count"], "eval_count": base["eval_count"],
            "accuracy": base["accuracy"], "parse_rate": base["parse_rate"],
            "wilson_95_ci": wilson(base["correct_count"], base["eval_count"]),
        },
        "candidates": sorted(candidates, key=lambda row: row["candidate_id"]),
        "paired_comparisons": {
            "group4_vs_group8": paired(group4, group8),
            "base_vs_selected": paired(base, outputs[selected["candidate_id"]]),
        },
        "selected": selected,
        "test_used_for_selection": False,
        "selection_used_test": False,
        "test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected["candidate_id"], "accuracy": selected["validation_accuracy"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
