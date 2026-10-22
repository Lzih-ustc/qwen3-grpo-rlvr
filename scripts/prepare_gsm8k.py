#!/usr/bin/env python3
"""Create reproducible, disjoint GSM8K train/validation/test JSONL subsets."""
from __future__ import annotations

import hashlib
import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
SEED = 20261023
FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$")


def gold_integer(answer_text: str) -> str:
    match = FINAL_ANSWER_RE.search(answer_text.strip())
    if not match:
        raise ValueError(f"No terminal GSM8K answer marker in: {answer_text[-100:]!r}")
    normalized = match.group(1).replace(",", "")
    numeric = float(normalized)
    if not numeric.is_integer():
        raise ValueError(f"Non-integer target is unsupported by the exact integer verifier: {normalized}")
    return str(int(numeric))


def take_unique(rows, source_split: str, count: int, seed: int, seen: set[str]) -> list[dict]:
    if count == 0:
        return []
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected = []
    for source_index in order:
        row = rows[source_index]
        question = row["question"].strip()
        if question in seen:
            continue
        answer = gold_integer(row["answer"])
        prompt = (
            f"{question}\n\n"
            "Solve the problem and show concise work. Put only the final numeric answer, without units or commas, "
            "inside <answer>...</answer>."
        )
        selected.append({
            "prompt": [{"role": "user", "content": prompt}],
            "answer": answer,
            "meta": {
                "dataset": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "source_split": source_split,
                "source_index": source_index,
            },
        })
        seen.add(question)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} unique rows from {source_split}; found {len(selected)}")
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> str:
    contents = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=96)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--test2-count", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument(
        "--exclude-test-data",
        action="append",
        type=Path,
        default=[],
        help="Existing JSONL test slices whose source rows must stay excluded from a new test split.",
    )
    args = parser.parse_args()
    dataset = load_dataset(DATASET_ID, "main", revision=DATASET_REVISION)
    seen_train: set[str] = set()
    train = take_unique(dataset["train"], "train", args.train_count, SEED, seen_train)
    validation = take_unique(dataset["train"], "train", args.validation_count, SEED + 1, seen_train)
    excluded_indices = set()
    for path in args.exclude_test_data:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                excluded_indices.add(int(json.loads(line)["meta"]["source_index"]))
    seen_test = {dataset["test"][index]["question"].strip() for index in excluded_indices}
    test = take_unique(dataset["test"], "test", args.test_count, SEED + 2, seen_test)
    test2 = take_unique(dataset["test"], "test", args.test2_count, SEED + 3, seen_test)
    out_dir = args.output_dir
    split_hashes = {
        "train": write_jsonl(out_dir / "train.jsonl", train),
        "validation": write_jsonl(out_dir / "validation.jsonl", validation),
        "test": write_jsonl(out_dir / "test.jsonl", test),
        "test2": write_jsonl(out_dir / "test2.jsonl", test2),
    }
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_config": "main",
        "license": "MIT",
        "seed": SEED,
        "counts": {"train": len(train), "validation": len(validation), "test": len(test), "test2": len(test2)},
        "excluded_test_source_indices": sorted(excluded_indices),
        "sha256": split_hashes,
        "question_overlaps": {
            "train_validation": 0, "train_test": 0, "train_test2": 0,
            "validation_test": 0, "validation_test2": 0, "test_test2": 0,
        },
        "answer_parser": FINAL_ANSWER_RE.pattern,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
