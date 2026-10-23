#!/usr/bin/env python3
"""Audit completion-level reward, correctness, clipping, and budget for a GRPO run."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet

from math_task import extract_answer


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def analyze(run_dir: Path, data_path: Path) -> dict:
    summary = read_json(run_dir / "run_summary.json")
    group_size = int(summary["attempt_config"]["num_generations"])
    answers = {row["prompt"][0]["content"]: int(row["answer"]) for row in read_jsonl(data_path)}
    by_step: dict[int, list[dict]] = {}
    seen_prompts = set()
    for path in sorted((run_dir / "attempt-01" / "trainer" / "completions").glob("*.parquet")):
        for row in parquet.read_table(path).to_pylist():
            prompt = row["prompt"][0]["content"]
            if prompt not in answers:
                raise ValueError(f"Rollout prompt not found in training data: {path}")
            seen_prompts.add(prompt)
            completion = row["completion"][-1]["content"]
            predicted, source = extract_answer(completion)
            target = answers[prompt]
            by_step.setdefault(int(row["step"]), []).append({
                "reward": float(row["tracked_math_reward"]),
                "predicted": predicted,
                "source": source,
                "correct": predicted == target,
                "distance": abs(predicted - target) if predicted is not None else None,
            })

    groups = [by_step[step] for step in sorted(by_step)]
    if not groups or any(len(group) != group_size for group in groups):
        raise ValueError("Completion parquet does not contain complete generation groups")
    if len(groups) != int(summary["optimizer_updates"]):
        raise ValueError("Completion group count differs from optimizer update count")
    rows = [row for group in groups for row in group]
    logs = summary["reward_log_entries"]
    if len(logs) != len(groups):
        raise ValueError("Reward log count differs from completion group count")

    steps = sorted(by_step)
    correct_counts = [sum(row["correct"] for row in group) for group in groups]
    histogram = {str(value): correct_counts.count(value) for value in range(group_size + 1)}
    group_stds = [statistics.pstdev([row["reward"] for row in group]) for group in groups]
    trainer_log = run_dir / "attempt-01" / "trainer_log_history.json"
    trainer_rows = read_json(trainer_log) if trainer_log.is_file() else []
    logged_group_tokens = {
        int(row["step"]): int(round(float(row["completions/mean_length"]) * group_size))
        for row in trainer_rows
        if row.get("step") is not None and row.get("completions/mean_length") is not None
    }
    group_tokens = [logged_group_tokens.get(step) for step in steps]
    token_data_complete = all(value is not None for value in group_tokens)
    groups_with_correct = [group for group in groups if any(row["correct"] for row in group)]
    groups_without_correct = [group for group in groups if not any(row["correct"] for row in group)]
    zero_gradient_steps = 0
    if trainer_log.is_file():
        zero_gradient_steps = sum(
            float(row.get("grad_norm", 0.0)) == 0.0
            for row in read_json(trainer_log)
            if "grad_norm" in row
        )
    clipped_group_ratios = [float(row.get("completions/clipped_ratio", 0.0)) for row in logs]
    reward_variable = [std > 0.0 for std in group_stds]
    correctness_mixed = [0 < value < group_size for value in correct_counts]
    reward_variable_all_wrong = [std > 0.0 and count == 0 for std, count in zip(group_stds, correct_counts, strict=True)]
    reward_variable_all_correct = [std > 0.0 and count == group_size for std, count in zip(group_stds, correct_counts, strict=True)]
    tokens_by_kind = {}
    if token_data_complete:
        tokens_by_kind = {
            "all_groups": sum(group_tokens),
            "reward_variable_groups": sum(tokens for tokens, keep in zip(group_tokens, reward_variable, strict=True) if keep),
            "correctness_mixed_groups": sum(tokens for tokens, keep in zip(group_tokens, correctness_mixed, strict=True) if keep),
            "reward_variable_all_wrong_groups": sum(tokens for tokens, keep in zip(group_tokens, reward_variable_all_wrong, strict=True) if keep),
            "reward_variable_all_correct_groups": sum(tokens for tokens, keep in zip(group_tokens, reward_variable_all_correct, strict=True) if keep),
            "all_wrong_groups": sum(tokens for tokens, count in zip(group_tokens, correct_counts, strict=True) if count == 0),
            "all_correct_groups": sum(tokens for tokens, count in zip(group_tokens, correct_counts, strict=True) if count == group_size),
        }
    rollout = {
        "completions": len(rows),
        "generation_groups": len(groups),
        "group_size": group_size,
        "unique_prompts": len(seen_prompts),
        "parse_rate": sum(row["predicted"] is not None for row in rows) / len(rows),
        "answer_tag_rate": sum(row["source"] == "answer_tag" for row in rows) / len(rows),
        "exact_match_rate": sum(row["correct"] for row in rows) / len(rows),
        "correct_completions": sum(row["correct"] for row in rows),
        "correct_completions_per_group": histogram,
        "all_wrong_groups": histogram["0"],
        "all_correct_groups": histogram[str(group_size)],
        "groups_with_any_correct": len(groups_with_correct),
        "groups_with_mixed_correct_and_incorrect": sum(correctness_mixed),
        "nonzero_reward_variance_groups": sum(reward_variable),
        "nonzero_reward_variance_groups_without_any_correct": sum(reward_variable_all_wrong),
        "nonzero_reward_variance_groups_with_all_correct": sum(reward_variable_all_correct),
        "reward_variance_effective_group_rate": sum(reward_variable) / len(groups),
        "correctness_mixed_group_rate": sum(correctness_mixed) / len(groups),
        "mean_reward": statistics.fmean(row["reward"] for row in rows),
        "mean_reward_on_correct": statistics.fmean(row["reward"] for row in rows if row["correct"]) if any(row["correct"] for row in rows) else None,
        "mean_reward_on_incorrect": statistics.fmean(row["reward"] for row in rows if not row["correct"]) if any(not row["correct"] for row in rows) else None,
        "mean_within_group_reward_std": statistics.fmean(group_stds),
        "mean_group_reward_std_with_correct": statistics.fmean(statistics.pstdev([row["reward"] for row in group]) for group in groups_with_correct) if groups_with_correct else None,
        "mean_group_reward_std_without_correct": statistics.fmean(statistics.pstdev([row["reward"] for row in group]) for group in groups_without_correct) if groups_without_correct else None,
        "zero_variance_groups": sum(value == 0.0 for value in group_stds),
        "zero_gradient_steps": zero_gradient_steps,
        "logged_group_completion_tokens_complete": token_data_complete,
        "logged_completion_tokens_sum": sum(group_tokens) if token_data_complete else None,
        "run_summary_completion_tokens": int(summary["observed_completion_tokens"]),
        "group_tokens_by_kind": tokens_by_kind,
        "tokens_per_reward_variable_group": (tokens_by_kind["reward_variable_groups"] / sum(reward_variable)) if token_data_complete and sum(reward_variable) else None,
        "tokens_per_correctness_mixed_group": (tokens_by_kind["correctness_mixed_groups"] / sum(correctness_mixed)) if token_data_complete and sum(correctness_mixed) else None,
        "correctness_mixed_groups_per_1000_tokens": (sum(correctness_mixed) * 1000 / sum(group_tokens)) if token_data_complete and sum(group_tokens) else None,
        "clipped_updates": sum(value > 0 for value in clipped_group_ratios),
        "mean_completion_clipped_fraction": statistics.fmean(clipped_group_ratios),
        "estimated_clipped_completions": sum(round(value * group_size) for value in clipped_group_ratios),
        "mean_absolute_error_for_parsed_incorrect": statistics.fmean(row["distance"] for row in rows if row["distance"] is not None and not row["correct"]) if any(row["distance"] is not None and not row["correct"] for row in rows) else None,
    }
    return {
        "status": "passed",
        "run_dir": str(run_dir),
        "git": summary.get("git"),
        "reward_mode": summary.get("reward_mode"),
        "train_data": summary.get("dataset"),
        "optimizer_updates": summary["optimizer_updates"],
        "actual_generated_tokens": summary["observed_completion_tokens"],
        "training_budget": summary.get("training_budget"),
        "train_seconds": summary["train_seconds"],
        "completion_tokens_per_second": summary["completion_tokens_per_second"],
        "torch_peak_allocated_mib": summary["torch_peak_allocated_mib"],
        "torch_peak_reserved_mib": summary["torch_peak_reserved_mib"],
        "whole_gpu_peak_used_mib": summary["gpu_monitor"]["whole_gpu_peak_memory_used_mib"],
        "checkpoints": summary["checkpoints"],
        "oom_fallbacks": summary["oom_fallbacks"],
        "rollout_analysis": rollout,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.train_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "optimizer_updates": report["optimizer_updates"], "rollout_analysis": report["rollout_analysis"], "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
