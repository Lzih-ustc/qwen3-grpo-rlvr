#!/usr/bin/env python3
"""Audit the 100-step GSM8K run, validation-only selection, and final test pair."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import pyarrow.parquet as parquet

from math_task import extract_answer


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def nonfinite_paths(value, prefix="") -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [] if math.isfinite(value) else [prefix or "<root>"]
    if isinstance(value, dict):
        return [bad for key, item in value.items() for bad in nonfinite_paths(item, f"{prefix}.{key}".strip("."))]
    if isinstance(value, list):
        return [bad for index, item in enumerate(value) for bad in nonfinite_paths(item, f"{prefix}[{index}]")]
    return []


def wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def checkpoint_audit(run_dir: Path, steps: list[int]) -> list[dict]:
    checks = []
    for step in steps:
        path = run_dir / "attempt-01" / "trainer" / f"checkpoint-{step}"
        required = ["adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
        missing = [name for name in required if not (path / name).is_file()]
        checks.append({"step": step, "path": str(path), "present": not missing, "missing_files": missing})
    return checks


def candidate(path: Path, expected_step: int, expected_hash: str) -> dict:
    result = read_json(path)
    adapter = Path(result["adapter"])
    step = int(adapter.name.split("-")[-1])
    assert step == expected_step, f"{path}: expected checkpoint step {expected_step}, got {step}"
    assert result["eval_data_sha256"] == expected_hash, f"{path}: validation hash mismatch"
    return {
        "step": step,
        "path": str(path),
        "adapter": str(adapter),
        "correct_count": result["correct_count"],
        "eval_count": result["eval_count"],
        "accuracy": result["accuracy"],
        "parse_rate": result["parse_rate"],
        "elapsed_seconds": result["elapsed_seconds"],
        "generated_tokens": result["generated_tokens"],
        "tokens_per_second": result["tokens_per_second"],
        "eval_data_sha256": result["eval_data_sha256"],
        "checkpoint_reloaded": True,
        "decode": result["decode"],
        "seed": result["seed"],
    }


def test_pair(base_path: Path, selected_path: Path, selection: dict, expected_hash: str) -> dict:
    base = read_json(base_path)
    selected = read_json(selected_path)
    assert base["adapter"] is None
    assert selected["adapter"] == selection["selected"]["adapter"]
    assert base["eval_data_sha256"] == selected["eval_data_sha256"] == expected_hash
    assert base["eval_count"] == selected["eval_count"] > 0
    assert base["decode"] == selected["decode"]
    assert [row["prompt"] for row in base["predictions"]] == [row["prompt"] for row in selected["predictions"]]
    both = base_only = selected_only = neither = 0
    for base_row, selected_row in zip(base["predictions"], selected["predictions"]):
        assert base_row["expected"] == selected_row["expected"]
        if base_row["correct"] and selected_row["correct"]:
            both += 1
        elif base_row["correct"]:
            base_only += 1
        elif selected_row["correct"]:
            selected_only += 1
        else:
            neither += 1
    discordant = base_only + selected_only
    mcnemar_p = min(
        1.0,
        2 * sum(math.comb(discordant, k) for k in range(max(base_only, selected_only), discordant + 1)) / (2**discordant),
    ) if discordant else 1.0
    return {
        "data_sha256": expected_hash,
        "eval_count": base["eval_count"],
        "base": {key: base[key] for key in ["correct_count", "accuracy", "parse_rate", "elapsed_seconds", "generated_tokens", "tokens_per_second"]},
        "selected": {key: selected[key] for key in ["correct_count", "accuracy", "parse_rate", "elapsed_seconds", "generated_tokens", "tokens_per_second"]},
        "accuracy_delta_percentage_points": 100 * (selected["accuracy"] - base["accuracy"]),
        "wilson_95_ci": {
            "base": wilson(base["correct_count"], base["eval_count"]),
            "selected": wilson(selected["correct_count"], selected["eval_count"]),
        },
        "paired_outcomes": {"both_correct": both, "base_only": base_only, "selected_only": selected_only, "both_wrong": neither},
        "exact_mcnemar_two_sided_p": mcnemar_p,
    }


def rollout_analysis(run_dir: Path, data_dir: Path, group_size: int) -> dict:
    train_rows = [
        json.loads(line)
        for line in (data_dir / "train.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    answers = {row["prompt"][0]["content"]: int(row["answer"]) for row in train_rows}
    by_step: dict[int, list[dict]] = {}
    seen_prompts = set()
    for path in sorted((run_dir / "attempt-01" / "trainer" / "completions").glob("*.parquet")):
        for row in parquet.read_table(path).to_pylist():
            prompt = row["prompt"][0]["content"]
            assert prompt in answers, f"Rollout prompt not in the pinned training split: {path}"
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
    rows = [row for step in sorted(by_step) for row in by_step[step]]
    assert all(len(group) == group_size for group in by_step.values())
    assert len(by_step) == 100
    rewards = [row["reward"] for row in rows]
    distances = [row["distance"] for row in rows if row["distance"] is not None]
    groups = list(by_step.values())
    correct_counts = [sum(row["correct"] for row in group) for group in groups]
    correct_count_histogram = {
        str(count): correct_counts.count(count) for count in range(group_size + 1)
    }
    group_reward_stds = [statistics.pstdev([row["reward"] for row in group]) for group in groups]
    zero_variance_groups = sum(value == 0 for value in group_reward_stds)
    groups_with_correct = [group for group in groups if any(row["correct"] for row in group)]
    groups_without_correct = [group for group in groups if not any(row["correct"] for row in group)]
    first_half = rows[: len(rows) // 2]
    second_half = rows[len(rows) // 2 :]
    return {
        "completions": len(rows),
        "unique_prompts": len(seen_prompts),
        "group_size": group_size,
        "parse_rate": sum(row["predicted"] is not None for row in rows) / len(rows),
        "answer_tag_rate": sum(row["source"] == "answer_tag" for row in rows) / len(rows),
        "exact_match_rate_on_rollouts": sum(row["correct"] for row in rows) / len(rows),
        "correct_rollouts": sum(row["correct"] for row in rows),
        "groups_with_at_least_one_correct": len(groups_with_correct),
        "groups_with_mixed_correct_and_incorrect": sum(0 < count < group_size for count in correct_counts),
        "correct_rollouts_per_group_histogram": correct_count_histogram,
        "mean_reward": statistics.fmean(rewards),
        "mean_reward_on_correct_rollouts": statistics.fmean(row["reward"] for row in rows if row["correct"]),
        "mean_reward_on_incorrect_rollouts": statistics.fmean(row["reward"] for row in rows if not row["correct"]),
        "mean_reward_std_within_group": statistics.fmean(group_reward_stds),
        "mean_reward_std_groups_with_correct": statistics.fmean(
            statistics.pstdev([row["reward"] for row in group]) for group in groups_with_correct
        ) if groups_with_correct else None,
        "mean_reward_std_groups_without_correct": statistics.fmean(
            statistics.pstdev([row["reward"] for row in group]) for group in groups_without_correct
        ) if groups_without_correct else None,
        "zero_reward_variance_groups_from_parquet": zero_variance_groups,
        "nonzero_distance_mean": statistics.fmean(distances) if distances else None,
        "first_half_exact_match_rate": sum(row["correct"] for row in first_half) / len(first_half),
        "second_half_exact_match_rate": sum(row["correct"] for row in second_half) / len(second_half),
        "first_half_mean_reward": statistics.fmean(row["reward"] for row in first_half),
        "second_half_mean_reward": statistics.fmean(row["reward"] for row in second_half),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/gsm8k-formal-g8-100"))
    parser.add_argument("--step50-validation", type=Path, default=Path("reports/gsm8k_formal_g8_step50_validation.json"))
    parser.add_argument("--step100-validation", type=Path, default=Path("reports/gsm8k_formal_g8_step100_validation.json"))
    parser.add_argument("--base-validation", type=Path, default=Path("reports/gsm8k_formal_base_validation.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/gsm8k-formal"))
    parser.add_argument("--base-test", type=Path)
    parser.add_argument("--selected-test", type=Path)
    parser.add_argument("--selection-output", type=Path, default=Path("reports/gsm8k_formal_checkpoint_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/gsm8k_formal_quality_gate.json"))
    args = parser.parse_args()
    if bool(args.base_test) != bool(args.selected_test):
        raise SystemExit("Provide both --base-test and --selected-test, or neither.")

    summary = read_json(args.run_dir / "run_summary.json")
    attempt = args.run_dir / "attempt-01"
    trainer_state = read_json(attempt / "trainer_state.json")
    history = trainer_state["log_history"]
    step_logs = [row for row in history if "reward_std" in row]
    finite_errors = nonfinite_paths(history)
    config = summary["attempt_config"]
    manifest = read_json(args.data_dir / "manifest.json")
    validation_hash = manifest["sha256"]["validation"]
    test_hash = manifest["sha256"]["test"]

    checkpoints = checkpoint_audit(args.run_dir, [50, 100])
    candidates = [
        candidate(args.step50_validation, 50, validation_hash),
        candidate(args.step100_validation, 100, validation_hash),
    ]
    assert candidates[0]["decode"] == candidates[1]["decode"]
    assert candidates[0]["seed"] == candidates[1]["seed"]
    candidates.sort(key=lambda row: (-row["accuracy"], row["step"]))
    selected = candidates[0]
    selection = {
        "criterion": "highest exact-match accuracy on the fixed 500-row formal validation split; ties choose the earlier checkpoint",
        "validation_sha256": validation_hash,
        "base_validation": {
            "correct_count": read_json(args.base_validation)["correct_count"],
            "eval_count": read_json(args.base_validation)["eval_count"],
            "accuracy": read_json(args.base_validation)["accuracy"],
            "parse_rate": read_json(args.base_validation)["parse_rate"],
            "validation_sha256": read_json(args.base_validation)["eval_data_sha256"],
        },
        "candidates": candidates,
        "selected": selected,
        "selection_used_final_test": False,
        "formal_test_sha256": test_hash,
        "formal_test_opened": bool(args.base_test),
    }
    assert selection["base_validation"]["validation_sha256"] == validation_hash
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rewards = summary["reward_log_entries"]
    rollouts = rollout_analysis(args.run_dir, args.data_dir, config["num_generations"])
    grad_logs = [row for row in step_logs if "grad_norm" in row]
    zero_var_steps = sum(float(row.get("frac_reward_zero_std", row.get("reward_std", 0))) > 0 for row in rewards)
    zero_grad_steps = sum(float(row.get("grad_norm", 0)) == 0 for row in grad_logs)
    checkpoints_ok = bool(checkpoints) and all(row["present"] for row in checkpoints)
    checks = {
        "trainer_completed_100_updates": summary["status"] == "success" and summary["optimizer_updates"] == trainer_state["global_step"] == 100,
        "group_and_gradient_accumulation_match": config["num_generations"] == config["gradient_accumulation_steps"] == 8,
        "observed_completions_match": summary["observed_generation_completion_count"] == 800,
        "adapter_weights_changed": summary["changed_lora_parameter_count"] > 0,
        "all_metrics_finite": not finite_errors,
        "positive_gradient_steps_exist": any(float(row.get("grad_norm", 0)) > 0 for row in grad_logs),
        "step50_and_step100_checkpoints_have_resume_state": checkpoints_ok,
        "no_oom_or_fallback": not summary.get("oom_fallbacks") and all(row["present"] for row in checkpoints),
        "run_commit_recorded_and_source_clean": bool(summary.get("git", {}).get("commit")) and summary["git"].get("working_tree_dirty") is False,
        "validation_candidates_reloaded_and_share_split": all(row["checkpoint_reloaded"] for row in candidates) and all(row["eval_count"] == 500 and row["eval_data_sha256"] == validation_hash for row in candidates),
        "selection_uses_validation_only": not selection["selection_used_final_test"] and selected["step"] == max(candidates, key=lambda row: (row["accuracy"], -row["step"]))["step"],
        "logged_rollouts_match_training_groups": rollouts["completions"] == 800 and rollouts["unique_prompts"] == 100,
        "reward_variance_matches_logged_steps": rollouts["zero_reward_variance_groups_from_parquet"] == zero_var_steps,
    }
    result = {
        "status": "validation_selection_passed" if all(checks.values()) and not args.base_test else "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "training": {
            "run_dir": str(args.run_dir),
            "git_commit": summary["git"]["commit"],
            "updates": summary["optimizer_updates"],
            "group_size": config["num_generations"],
            "completions": summary["observed_generation_completion_count"],
            "completion_tokens": summary["observed_completion_tokens"],
            "completion_tokens_per_second": summary["completion_tokens_per_second"],
            "train_seconds": summary["train_seconds"],
            "mean_reward": sum(float(row["reward"]) for row in rewards) / len(rewards),
            "zero_reward_variance_steps": zero_var_steps,
            "zero_gradient_steps": zero_grad_steps,
            "clipped_steps": sum(float(row.get("completions/clipped_ratio", 0)) > 0 for row in rewards),
            "changed_lora_parameters": summary["changed_lora_parameter_count"],
            "torch_peak_allocated_mib": summary["torch_peak_allocated_mib"],
            "torch_peak_reserved_mib": summary["torch_peak_reserved_mib"],
            "gpu_peak_used_mib": summary["gpu_monitor"]["whole_gpu_peak_memory_used_mib"],
            "checkpoints": checkpoints,
        },
        "selection": selection,
        "rollout_analysis": rollouts,
        "final_test_comparison": None,
    }
    if args.base_test and args.selected_test:
        result["final_test_comparison"] = test_pair(args.base_test, args.selected_test, selection, test_hash)
        assert result["final_test_comparison"]["eval_count"] == manifest["counts"]["test"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks, "selected_step": selected["step"], "final_test_evaluated": bool(args.base_test), "output": str(args.output)}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
