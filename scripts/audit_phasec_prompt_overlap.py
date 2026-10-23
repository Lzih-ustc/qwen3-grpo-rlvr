#!/usr/bin/env python3
"""Compare saved training-prompt coverage across matched-token GRPO arms.

This is an exploratory, read-only audit. It does not rerun rollouts or change
checkpoint selection; report only hashed prompt identifiers, not prompt text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import pyarrow.parquet as parquet

from math_task import extract_answer


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prompt_text(prompt: list[dict]) -> str:
    user_messages = [message["content"] for message in prompt if message.get("role") == "user"]
    if len(user_messages) != 1:
        raise ValueError("expected exactly one user message in each saved training prompt")
    return user_messages[0]


def _load_run(run_dir: Path, train_answers: dict[str, int]) -> dict:
    attempt_dir = run_dir / "attempt-01"
    summary = read_json(run_dir / "run_summary.json")
    group_size = int(summary["attempt_config"]["num_generations"])
    groups_by_step: dict[int, list[dict]] = {}
    for path in sorted((attempt_dir / "trainer" / "completions").glob("*.parquet")):
        for row in parquet.read_table(path).to_pylist():
            prompt = _prompt_text(row["prompt"])
            if prompt not in train_answers:
                raise ValueError(f"rollout prompt absent from training data: {path}")
            completion = row["completion"][-1]["content"]
            prediction, source = extract_answer(completion)
            groups_by_step.setdefault(int(row["step"]), []).append(
                {
                    "prompt": prompt,
                    "correct": prediction == train_answers[prompt],
                    "parsed": prediction is not None,
                    "reward": float(row["tracked_math_reward"]),
                    "answer_source": source,
                }
            )
    if len(groups_by_step) != int(summary["optimizer_updates"]):
        raise ValueError(f"completion groups do not match optimizer updates in {run_dir}")

    trainer_log_path = attempt_dir / "trainer_log_history.json"
    trainer_log = read_json(trainer_log_path) if trainer_log_path.is_file() else []
    tokens_by_step = {
        int(row["step"]): int(round(float(row["completions/mean_length"]) * group_size))
        for row in trainer_log
        if row.get("step") is not None and row.get("completions/mean_length") is not None
    }
    prompts: dict[str, list[dict]] = {}
    for step, rows in sorted(groups_by_step.items()):
        if len(rows) != group_size:
            raise ValueError(f"step {step} in {run_dir} has {len(rows)} completions, expected {group_size}")
        unique_at_step = {row["prompt"] for row in rows}
        if len(unique_at_step) != 1:
            raise ValueError(f"step {step} in {run_dir} contains more than one prompt")
        prompt = next(iter(unique_at_step))
        correct_count = sum(row["correct"] for row in rows)
        rewards = [row["reward"] for row in rows]
        groups = prompts.setdefault(prompt, [])
        groups.append(
            {
                "step": step,
                "correct": correct_count,
                "completions": len(rows),
                "accuracy": correct_count / len(rows),
                "correctness_mixed": 0 < correct_count < len(rows),
                "reward_std": statistics.pstdev(rewards),
                "reward_variance": statistics.pstdev(rewards) > 0,
                "tokens": tokens_by_step.get(step),
            }
        )
    return {
        "run_dir": str(run_dir),
        "group_size": group_size,
        "seed": summary.get("seed"),
        "dataset_sha256": summary["dataset"]["sha256"],
        "model_revision": summary.get("model_revision"),
        "reward_mode": summary.get("reward_mode"),
        "optimizer_updates": summary["optimizer_updates"],
        "generated_tokens": summary["observed_completion_tokens"],
        "prompts": prompts,
    }


def _run_pair(label: str, group4: dict, group8: dict) -> dict:
    for field in ("seed", "dataset_sha256", "model_revision", "reward_mode"):
        if group4[field] != group8[field]:
            raise ValueError(f"{label} pair mismatch for {field}: {group4[field]} != {group8[field]}")
    prompts4 = set(group4["prompts"])
    prompts8 = set(group8["prompts"])
    common = sorted(prompts4 & prompts8)

    def aggregate(run: dict, selected: list[str]) -> dict:
        selected_groups = [group for prompt in selected for group in run["prompts"][prompt]]
        token_values = [group["tokens"] for group in selected_groups]
        mixed = sum(group["correctness_mixed"] for group in selected_groups)
        variance = sum(group["reward_variance"] for group in selected_groups)
        return {
            "prompt_count": len(selected),
            "group_count": len(selected_groups),
            "completions": sum(group["completions"] for group in selected_groups),
            "mean_prompt_accuracy": (
                statistics.fmean(
                    statistics.fmean(group["accuracy"] for group in run["prompts"][prompt])
                    for prompt in selected
                )
                if selected
                else None
            ),
            "correctness_mixed_groups": mixed,
            "nonzero_reward_variance_groups": variance,
            "generated_tokens": sum(value for value in token_values if value is not None)
            if all(value is not None for value in token_values)
            else None,
            "tokens_per_correctness_mixed_group": (
                sum(token_values) / mixed
                if mixed and all(value is not None for value in token_values)
                else None
            ),
        }

    common4 = aggregate(group4, common)
    common8 = aggregate(group8, common)
    differences = []
    for prompt in common:
        accuracy4 = statistics.fmean(group["accuracy"] for group in group4["prompts"][prompt])
        accuracy8 = statistics.fmean(group["accuracy"] for group in group8["prompts"][prompt])
        differences.append(accuracy8 - accuracy4)
    union = prompts4 | prompts8
    return {
        "label": label,
        "seed": group4["seed"],
        "dataset_sha256": group4["dataset_sha256"],
        "group4": {
            "updates": group4["optimizer_updates"],
            "generated_tokens": group4["generated_tokens"],
            "unique_prompts": len(prompts4),
        },
        "group8": {
            "updates": group8["optimizer_updates"],
            "generated_tokens": group8["generated_tokens"],
            "unique_prompts": len(prompts8),
        },
        "prompt_coverage": {
            "intersection": len(common),
            "union": len(union),
            "group4_only": len(prompts4 - prompts8),
            "group8_only": len(prompts8 - prompts4),
            "jaccard": len(common) / len(union) if union else None,
            "common_prompt_sha256": [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in common],
        },
        "common_prompt_group_diagnostics": {
            "group4": common4,
            "group8": common8,
            "group8_minus_group4_mean_prompt_accuracy": statistics.fmean(differences) if differences else None,
            "prompts_group8_higher_accuracy": sum(value > 0 for value in differences),
            "prompts_tied_accuracy": sum(value == 0 for value in differences),
            "prompts_group4_higher_accuracy": sum(value < 0 for value in differences),
        },
    }


def analyze(
    train_data: Path,
    pairs: list[tuple[str, Path, Path]],
) -> dict:
    train_answers = {_prompt_text(row["prompt"]): int(row["answer"]) for row in read_jsonl(train_data)}
    results = []
    for label, group4_dir, group8_dir in pairs:
        results.append(_run_pair(label, _load_run(group4_dir, train_answers), _load_run(group8_dir, train_answers)))
    return {
        "status": "passed",
        "analysis_type": "exploratory_saved_prompt_coverage_audit",
        "train_data": str(train_data),
        "train_data_sha256": hashlib.sha256(train_data.read_bytes()).hexdigest(),
        "pairs": results,
        "limitations": [
            "Prompt overlap is descriptive; prompts were not assigned as a preregistered paired experimental design.",
            "The two arms sampled independent completions and had different update counts and prompt coverage.",
            "The paired subset may be small and selected by the observed sampling overlap; it cannot establish a causal group-size effect.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--seed1-g4", type=Path, required=True)
    parser.add_argument("--seed1-g8", type=Path, required=True)
    parser.add_argument("--seed2-g4", type=Path, required=True)
    parser.add_argument("--seed2-g8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.train_data,
        [
            ("seed1", args.seed1_g4, args.seed1_g8),
            ("seed2", args.seed2_g4, args.seed2_g8),
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "pairs": report["pairs"], "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
