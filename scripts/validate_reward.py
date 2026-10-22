#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from math_task import build_examples, exact_math_reward, extract_answer


def main() -> None:
    cases = [
        ("work: 18 + 6 = 24. <answer>24</answer>", "24", 1.15),
        ("<answer>23</answer>", "24", 0.10),
        ("The result is 24.", "24", 0.0),
        ("<answer>24</answer> <answer>24</answer>", "24", 0.0),
        (r"\boxed{-7}", "-7", 1.10),
    ]
    for completion, target, expected in cases:
        actual = exact_math_reward([completion], [target])[0]
        assert abs(actual - expected) < 1e-9, (completion, expected, actual)
    assert extract_answer("The text mentions 999 but no marked result.") == (None, None)

    expected_train = build_examples(256, 20261022)
    expected_eval = build_examples(48, 20261022 + 1_000_003)
    excluded_questions = {
        row["prompt"][0]["content"]
        for row in expected_train + expected_eval
    }
    expected = {
        "train": expected_train,
        "validation": build_examples(
            96,
            20261022 + 2_000_003,
            forbidden_questions=excluded_questions,
        ),
        "eval": expected_eval,
    }
    paths = {
        "train": Path("data/train.jsonl"),
        "validation": Path("data/validation.jsonl"),
        "eval": Path("data/eval.jsonl"),
    }
    persisted = {
        name: [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
        for name, path in paths.items()
    }
    for name in expected:
        assert persisted[name] == expected[name], f"{name} split differs from deterministic generator"
    question_sets = {
        name: {row["prompt"][0]["content"] for row in rows}
        for name, rows in persisted.items()
    }
    overlaps = {
        "train_validation": len(question_sets["train"] & question_sets["validation"]),
        "train_eval": len(question_sets["train"] & question_sets["eval"]),
        "validation_eval": len(question_sets["validation"] & question_sets["eval"]),
    }
    assert not any(overlaps.values()), overlaps
    Path("reports").mkdir(exist_ok=True)
    report = {
        "status": "passed",
        "reward_cases": len(cases),
        "splits": {
            name: {"examples": len(rows), "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest()}
            for name, rows in persisted.items()
        },
        "question_overlaps": overlaps,
    }
    Path("reports/reward_correctness.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path("reports/data_split_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
