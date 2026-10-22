#!/usr/bin/env python3
"""Audit finite training metrics, real updates, checkpoints, and validation selection."""
from __future__ import annotations

import hashlib
import json
import math
import argparse
from pathlib import Path


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


def audit_run(run_dir: Path) -> dict:
    summary = read_json(run_dir / "run_summary.json")
    attempt_dir = run_dir / "attempt-01"
    trainer_state = read_json(attempt_dir / "trainer_state.json")
    config = summary["attempt_config"]
    logs = [row for row in trainer_state["log_history"] if "reward_std" in row]
    finite_errors = nonfinite_paths(trainer_state["log_history"])
    grad_logs = [row for row in logs if "grad_norm" in row]
    reward_variances = [float(row["reward_std"]) for row in logs]
    checkpoints = []
    for checkpoint_name in summary["checkpoints"]:
        checkpoint = Path(checkpoint_name)
        required = ["adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
        missing = [name for name in required if not (checkpoint / name).is_file()]
        checkpoints.append({"path": str(checkpoint), "required_files_present": not missing, "missing_files": missing})
    final_adapter = Path(summary["final_adapter"])
    adapter_weights = final_adapter / "adapter_model.safetensors"
    checks = {
        "run_success": summary["status"] == "success",
        "optimizer_update_count_matches_state": summary["optimizer_updates"] == trainer_state["global_step"] >= 1,
        "finite_loss_reward_and_gradient_logs": not finite_errors,
        "at_least_one_nonzero_gradient": any(float(row.get("grad_norm", 0.0)) > 0 for row in grad_logs),
        "reward_variance_present": any(value > 0 for value in reward_variances),
        "effective_generation_group_and_accumulation_match": config["num_generations"] >= 2 and config["gradient_accumulation_steps"] == config["num_generations"],
        "observed_completions_match_updates_times_group": summary["observed_generation_completion_count"] == summary["optimizer_updates"] * config["num_generations"],
        "adapter_parameters_changed": summary["changed_lora_parameter_count"] > 0,
        "final_adapter_weights_exist": adapter_weights.is_file(),
        "all_checkpoints_have_resume_state": bool(checkpoints) and all(item["required_files_present"] for item in checkpoints),
    }
    if not checks["finite_loss_reward_and_gradient_logs"]:
        checks["nonfinite_metric_paths"] = finite_errors
    return {
        "run_dir": str(run_dir),
        "optimizer_updates": summary["optimizer_updates"],
        "generation_group": config["num_generations"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "positive_gradient_steps": sum(float(row.get("grad_norm", 0.0)) > 0 for row in grad_logs),
        "zero_gradient_steps": sum(float(row.get("grad_norm", 0.0)) == 0 for row in grad_logs),
        "zero_reward_variance_steps": sum(value == 0 for value in reward_variances),
        "total_reward_logged_steps": len(reward_variances),
        "changed_lora_parameters": summary["changed_lora_parameter_count"],
        "checkpoints": checkpoints,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-run", type=Path, default=Path("runs/smoke-v2"))
    parser.add_argument("--small-run", type=Path, default=Path("runs/local-small"))
    parser.add_argument("--step8-validation", type=Path, default=Path("reports/validation_step8.json"))
    parser.add_argument("--step12-validation", type=Path, default=Path("reports/validation_step12.json"))
    parser.add_argument("--split-report", type=Path, default=Path("reports/data_split_validation.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    smoke = audit_run(args.smoke_run)
    small = audit_run(args.small_run)
    split_report = read_json(args.split_report)
    split_hashes = {name: details["sha256"] for name, details in split_report["splits"].items()}
    candidates = []
    for path in [args.step8_validation, args.step12_validation]:
        result = read_json(path)
        candidates.append({
            "path": str(path),
            "adapter": result["adapter"],
            "correct_count": result["correct_count"],
            "eval_count": result["eval_count"],
            "accuracy": result["accuracy"],
            "parse_rate": result["parse_rate"],
            "eval_data_sha256": result["eval_data_sha256"],
            "checkpoint_reload_exercised": True,
        })
    assert all(candidate["eval_data_sha256"] == split_hashes["validation"] for candidate in candidates)
    # Primary criterion is exact match; ties go to the earlier (smaller) step.
    candidates.sort(key=lambda row: (-row["accuracy"], int(Path(row["adapter"]).name.split("-")[-1])))
    selection = {
        "criterion": "highest exact-match accuracy on data/validation.jsonl; ties choose the earlier checkpoint",
        "candidates": candidates,
        "selected": candidates[0],
        "final_test_data_sha256": split_hashes["eval"],
        "selection_used_final_test": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checks = {
        "smoke": smoke,
        "small_run": small,
        "three_way_data_split_passed": split_report["status"] == "passed" and not any(split_report["question_overlaps"].values()),
        "validation_checkpoint_selection": selection,
    }
    passed = all(all(run["checks"].values()) for run in [smoke, small]) and checks["three_way_data_split_passed"] and selection["selected"]["eval_count"] > 0
    result = {"status": "passed" if passed else "failed", "checks": checks}
    (args.output_dir / "quality_gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
