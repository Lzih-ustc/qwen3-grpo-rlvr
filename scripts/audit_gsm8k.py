#!/usr/bin/env python3
"""Audit the local GSM8K pilot, checkpoint selection, and untouched final slice."""
from __future__ import annotations

import json
import math
import argparse
from pathlib import Path


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite(value) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def run_summary(run_dir: str) -> tuple[dict, dict, dict]:
    root = Path(run_dir)
    summary = read(str(root / "run_summary.json"))
    trainer = read(str(root / "attempt-01/trainer_state.json"))
    logs = [row for row in trainer["log_history"] if "reward_std" in row]
    config = summary["attempt_config"]
    checkpoint_checks = []
    for checkpoint_name in summary["checkpoints"]:
        checkpoint = Path(checkpoint_name)
        required = ["adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
        missing = [name for name in required if not (checkpoint / name).is_file()]
        checkpoint_checks.append({"path": str(checkpoint), "missing_files": missing})
    checks = {
        "trainer_success": summary["status"] == "success",
        "finite_metrics": finite(trainer["log_history"]),
        "nonzero_gradient_observed": any(float(row.get("grad_norm", 0.0)) > 0 for row in logs),
        "reward_variance_observed": any(float(row.get("reward_std", 0.0)) > 0 for row in logs),
        "optimizer_and_state_steps_match": summary["optimizer_updates"] == trainer["global_step"] >= 1,
        "group_and_accumulation_match": config["num_generations"] == config["gradient_accumulation_steps"] >= 2,
        "completion_count_matches": summary["observed_generation_completion_count"] == summary["optimizer_updates"] * config["num_generations"],
        "adapter_changed": summary["changed_lora_parameter_count"] > 0,
        "checkpoints_have_resume_files": bool(checkpoint_checks) and all(not item["missing_files"] for item in checkpoint_checks),
        "no_oom_or_fallback": not summary.get("oom_fallbacks"),
    }
    return summary, {
        "run_dir": run_dir,
        "updates": summary["optimizer_updates"],
        "group_size": config["num_generations"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "completion_count": summary["observed_generation_completion_count"],
        "completion_tokens": summary["observed_completion_tokens"],
        "completion_tokens_per_second": summary["completion_tokens_per_second"],
        "train_seconds": summary["train_seconds"],
        "mean_reward": sum(float(row["reward"]) for row in logs) / len(logs),
        "mean_group_reward_std": sum(float(row["reward_std"]) for row in logs) / len(logs),
        "zero_reward_variance_steps": sum(float(row["reward_std"]) == 0 for row in logs),
        "clipped_steps": [row.get("step") for row in logs if row.get("completions/clipped_ratio", 0) > 0],
        "positive_gradient_steps": sum(float(row.get("grad_norm", 0)) > 0 for row in logs),
        "changed_lora_parameters": summary["changed_lora_parameter_count"],
        "torch_peak_allocated_mib": summary["torch_peak_allocated_mib"],
        "torch_peak_reserved_mib": summary["torch_peak_reserved_mib"],
        "gpu_peak_used_mib": summary["gpu_monitor"]["whole_gpu_peak_memory_used_mib"],
        "checkpoints": checkpoint_checks,
        "checks": checks,
    }, trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/gsm8k/manifest.json"))
    parser.add_argument("--data-validation", type=Path, default=Path("reports/gsm8k_data_validation.json"))
    parser.add_argument("--group2-run", type=Path, default=Path("runs/gsm8k-pilot"))
    parser.add_argument("--group4-run", type=Path, default=Path("runs/gsm8k-group4-pilot"))
    parser.add_argument("--group2-step4-validation", type=Path, default=Path("reports/gsm8k_validation_step4.json"))
    parser.add_argument("--group2-step8-validation", type=Path, default=Path("reports/gsm8k_validation_step8.json"))
    parser.add_argument("--group4-step2-validation", type=Path, default=Path("reports/gsm8k_group4_validation_step2.json"))
    parser.add_argument("--group4-step4-validation", type=Path, default=Path("reports/gsm8k_group4_validation_step4.json"))
    parser.add_argument("--base-test", type=Path, default=Path("reports/gsm8k_base_test2.json"))
    parser.add_argument("--selected-test", type=Path, default=Path("reports/gsm8k_selected_test2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    manifest = read(str(args.manifest))
    data_validation = read(str(args.data_validation))
    group2, group2_audit, _ = run_summary(str(args.group2_run))
    group4, group4_audit, _ = run_summary(str(args.group4_run))
    candidate_specs = [
        ("group2", 4, args.group2_run, args.group2_step4_validation),
        ("group2", 8, args.group2_run, args.group2_step8_validation),
        ("group4", 2, args.group4_run, args.group4_step2_validation),
        ("group4", 4, args.group4_run, args.group4_step4_validation),
    ]
    candidates = []
    for group, step, run_dir, path in candidate_specs:
        result = read(str(path))
        expected_adapter = (run_dir / f"attempt-01/trainer/checkpoint-{step}").as_posix()
        assert result["adapter"] == expected_adapter
        assert result["eval_data_sha256"] == manifest["sha256"]["validation"]
        candidates.append({
            "group_size": int(group.removeprefix("group")),
            "step": step,
            "adapter": result["adapter"],
            "correct_count": result["correct_count"],
            "eval_count": result["eval_count"],
            "accuracy": result["accuracy"],
            "parse_rate": result["parse_rate"],
            "elapsed_seconds": result["elapsed_seconds"],
            "tokens_per_second": result["tokens_per_second"],
            "generated_tokens": result["generated_tokens"],
            "validation_sha256": result["eval_data_sha256"],
            "checkpoint_reloaded": True,
        })
    # Select solely on validation exact match; ties prefer fewer optimizer updates.
    candidates.sort(key=lambda row: (-row["accuracy"], row["step"], row["group_size"]))
    selected = candidates[0]
    base_test = read(str(args.base_test))
    selected_test = read(str(args.selected_test))
    assert base_test["adapter"] is None
    assert selected_test["adapter"] == selected["adapter"]
    assert base_test["eval_data_sha256"] == selected_test["eval_data_sha256"] == manifest["sha256"]["test2"]
    selection = {
        "criterion": "highest validation exact-match accuracy; ties prefer fewer optimizer updates, then smaller group",
        "candidate_checkpoints": candidates,
        "selected": selected,
        "test2_sha256": manifest["sha256"]["test2"],
        "test2_used_for_selection": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gsm8k_checkpoint_selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    test_comparison = {
        "data_sha256": manifest["sha256"]["test2"],
        "count": selected_test["eval_count"],
        "base": {"correct": base_test["correct_count"], "accuracy": base_test["accuracy"], "parse_rate": base_test["parse_rate"]},
        "selected": {"correct": selected_test["correct_count"], "accuracy": selected_test["accuracy"], "parse_rate": selected_test["parse_rate"]},
        "accuracy_change_percentage_points": 100 * (selected_test["accuracy"] - base_test["accuracy"]),
        "base_elapsed_seconds": base_test["elapsed_seconds"],
        "selected_elapsed_seconds": selected_test["elapsed_seconds"],
        "base_tokens_per_second": base_test["tokens_per_second"],
        "selected_tokens_per_second": selected_test["tokens_per_second"],
    }
    group_budget = {
        "group2_total_completions": group2["observed_generation_completion_count"],
        "group4_total_completions": group4["observed_generation_completion_count"],
        "completion_count_matched": group2["observed_generation_completion_count"] == group4["observed_generation_completion_count"],
        "group2_completion_tokens": group2["observed_completion_tokens"],
        "group4_completion_tokens": group4["observed_completion_tokens"],
        "group2_train_seconds": group2["train_seconds"],
        "group4_train_seconds": group4["train_seconds"],
        "comparison_note": "Rollout count was matched; token count and wall time were not matched, so this is not a compute-equivalent efficiency claim.",
    }
    checks = {
        "dataset_validation_passed": data_validation["status"] == "passed" and not any(data_validation["question_overlaps"].values()),
        "group2_run": group2_audit,
        "group4_run": group4_audit,
        "group_rollout_count_matched": group_budget["completion_count_matched"],
        "all_candidates_use_same_validation_hash": len({row["validation_sha256"] for row in candidates}) == 1,
        "selected_by_validation_only": not selection["test2_used_for_selection"],
        "selected_checkpoint_matches_test2": selected_test["adapter"] == selected["adapter"],
        "base_and_selected_test2_hash_match": base_test["eval_data_sha256"] == selected_test["eval_data_sha256"],
    }
    passed = (
        checks["dataset_validation_passed"]
        and checks["group_rollout_count_matched"]
        and checks["all_candidates_use_same_validation_hash"]
        and checks["selected_by_validation_only"]
        and checks["selected_checkpoint_matches_test2"]
        and checks["base_and_selected_test2_hash_match"]
        and all(all(run["checks"].values()) for run in [group2_audit, group4_audit])
    )
    report = {
        "status": "passed" if passed else "failed",
        "model_revision": group4["model_revision"],
        "dataset": manifest,
        "group_budget_comparison": group_budget,
        "validation_selection": selection,
        "heldout_test2_comparison": test_comparison,
        "checks": checks,
    }
    (args.output_dir / "gsm8k_quality_gate.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
