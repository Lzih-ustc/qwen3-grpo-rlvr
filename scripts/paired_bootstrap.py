#!/usr/bin/env python3
"""Paired exact-match contrasts with deterministic item-bootstrap intervals."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    return min(1.0, 2 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant))


def paired(left_path: Path, right_path: Path, *, seed: int, replicates: int) -> dict:
    left, right = read(left_path), read(right_path)
    assert left["eval_data_sha256"] == right["eval_data_sha256"]
    assert left["decode"] == right["decode"]
    assert [row["prompt"] for row in left["predictions"]] == [row["prompt"] for row in right["predictions"]]
    assert [row["expected"] for row in left["predictions"]] == [row["expected"] for row in right["predictions"]]
    assert len(left["predictions"]) == len(right["predictions"]) > 0

    left_correct = np.asarray([row["correct"] for row in left["predictions"]], dtype=np.int8)
    right_correct = np.asarray([row["correct"] for row in right["predictions"]], dtype=np.int8)
    differences = left_correct - right_correct
    both = int(np.sum((left_correct == 1) & (right_correct == 1)))
    left_only = int(np.sum((left_correct == 1) & (right_correct == 0)))
    right_only = int(np.sum((left_correct == 0) & (right_correct == 1)))
    neither = int(np.sum((left_correct == 0) & (right_correct == 0)))

    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=np.float64)
    start = 0
    while start < replicates:
        batch = min(512, replicates - start)
        indices = rng.integers(0, len(differences), size=(batch, len(differences)))
        boot[start : start + batch] = differences[indices].mean(axis=1)
        start += batch
    point = float(differences.mean())
    lower, upper = np.quantile(boot, [0.025, 0.975]).tolist()
    return {
        "left_tag": left.get("tag"),
        "right_tag": right.get("tag"),
        "left_file": str(left_path),
        "right_file": str(right_path),
        "eval_data_sha256": left["eval_data_sha256"],
        "eval_count": len(differences),
        "left_accuracy": float(left_correct.mean()),
        "right_accuracy": float(right_correct.mean()),
        "left_minus_right_accuracy_percentage_points": 100 * point,
        "paired_item_bootstrap_95_ci_percentage_points": [100 * float(lower), 100 * float(upper)],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "paired_outcomes": {
            "both_correct": both,
            "left_only_correct": left_only,
            "right_only_correct": right_only,
            "both_wrong": neither,
        },
        "exact_mcnemar_two_sided_p": exact_mcnemar(left_only, right_only),
        "interpretation_scope": "item-level uncertainty conditional on these two saved model outputs; does not estimate training-seed variability",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Label=eval.json")
    parser.add_argument("--right", required=True, help="Label=eval.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        parser.error("--bootstrap-replicates must be at least 1000")
    left_label, left_file = args.left.split("=", 1)
    right_label, right_file = args.right.split("=", 1)
    report = {
        "status": "paired_bootstrap_complete",
        "contrast": f"{left_label}_minus_{right_label}",
        "metric": "exact_match_accuracy",
        "left": left_label,
        "right": right_label,
        "result": paired(Path(left_file), Path(right_file), seed=args.seed, replicates=args.bootstrap_replicates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "contrast": report["contrast"], "delta_pp": report["result"]["left_minus_right_accuracy_percentage_points"], "ci95_pp": report["result"]["paired_item_bootstrap_95_ci_percentage_points"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
