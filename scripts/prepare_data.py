#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from math_task import SEED, build_examples


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(contents, encoding="utf-8")
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=96)
    parser.add_argument("--eval-count", type=int, default=48)
    parser.add_argument("--train-output", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/validation.jsonl"))
    parser.add_argument("--eval-output", type=Path, default=Path("data/eval.jsonl"))
    args = parser.parse_args()
    # Distinct deterministic seeds keep training, validation, and final evaluation independent.
    train = build_examples(args.train_count, SEED)
    evaluation = build_examples(args.eval_count, SEED + 1_000_003)
    excluded_questions = {
        row["prompt"][0]["content"]
        for row in train + evaluation
    }
    validation = build_examples(
        args.validation_count,
        SEED + 2_000_003,
        forbidden_questions=excluded_questions,
    )
    train_hash = write_jsonl(args.train_output, train)
    validation_hash = write_jsonl(args.validation_output, validation)
    eval_hash = write_jsonl(args.eval_output, evaluation)
    print(json.dumps({
        "train_count": len(train), "train_sha256": train_hash,
        "validation_count": len(validation), "validation_sha256": validation_hash,
        "eval_count": len(evaluation), "eval_sha256": eval_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
