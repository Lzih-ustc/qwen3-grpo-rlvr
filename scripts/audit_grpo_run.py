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

    correct_counts = [sum(row["correct"] for row in group) for group in groups]
    histogram = {str(value): correct_counts.count(value) for value in range(group_size + 1)}
    group_stds = [statistics.pstdev([row["reward"] for row in group]) for group in groups]
    groups_with_correct = [group for group in groups if any(row["correct"] for row in group)]
    groups_without_correct = [group for group in groups if not any(row["correct"] for row in group)]
    zero_gradient_steps = 0
    trainer_log = run_dir / "attempt-01" / "trainer_log_history.json"
    if trainer_log.is_file():
        zero_gradient_steps = sum(
            float(row.get("grad_norm", 0.0)) == 0.0
            for row in read_json(trainer_log)
            if "grad_norm" in row
        )
    clipped_group_ratios = [float(row.get("completions/clipped_ratio", 0.0)) for row in logs]
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
        "groups_with_any_correct": len(groups_with_correct),
        "groups_with_mixed_correct_and_incorrect": sum(0 < value < group_size for value in correct_counts),
        "mean_reward": statistics.fmean(row["reward"] for row in rows),
        "mean_reward_on_correct": statistics.fmean(row["reward"] for row in rows if row["correct"]) if any(row["correct"] for row in rows) else None,
        "mean_reward_on_incorrect": statistics.fmean(row["reward"] for row in rows if not row["correct"]) if any(not row["correct"] for row in rows) else None,
        "mean_within_group_reward_std": statistics.fmean(group_stds),
        "mean_group_reward_std_with_correct": statistics.fmean(statistics.pstdev([row["reward"] for row in group]) for group in groups_with_correct) if groups_with_correct else None,
        "mean_group_reward_std_without_correct": statistics.fmean(statistics.pstdev([row["reward"] for row in group]) for group in groups_without_correct) if groups_without_correct else None,
        "zero_variance_groups": sum(value == 0.0 for value in group_stds),
        "zero_gradient_steps": zero_gradient_steps,
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
