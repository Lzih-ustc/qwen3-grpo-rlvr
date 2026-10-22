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


def take_unique(
    rows,
    source_split: str,
    count: int,
    seed: int,
    seen: set[str],
    excluded_indices: set[int] | None = None,
) -> list[dict]:
    if count == 0:
        return []
    excluded_indices = excluded_indices or set()
    order = [index for index in range(len(rows)) if index not in excluded_indices]
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


def read_source_indices(paths: list[Path], expected_split: str) -> set[int]:
    indices = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            meta = json.loads(line)["meta"]
            if meta["source_split"] != expected_split:
                raise ValueError(
                    f"Excluded row from source split {meta['source_split']!r}, "
                    f"expected {expected_split!r}: {path}"
                )
            indices.add(int(meta["source_index"]))
    return indices


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
    parser.add_argument("--test-source-split", choices=["train", "test"], default="test")
    parser.add_argument(
        "--exclude-train-data",
        action="append",
        type=Path,
        default=[],
        help="Existing JSONL train/validation rows to exclude from new train/validation/test sampling.",
    )
    parser.add_argument(
        "--exclude-test-data",
        action="append",
        type=Path,
        default=[],
        help="Existing JSONL test slices whose source rows must stay excluded from a new test split.",
    )
    args = parser.parse_args()
    dataset = load_dataset(DATASET_ID, "main", revision=DATASET_REVISION)
    previously_used_train_indices = read_source_indices(args.exclude_train_data, "train")
    seen_train = {
        dataset["train"][index]["question"].strip()
        for index in previously_used_train_indices
    }
    train = take_unique(
        dataset["train"], "train", args.train_count, SEED, seen_train,
        excluded_indices=previously_used_train_indices,
    )
    new_train_indices = {int(row["meta"]["source_index"]) for row in train}
    validation_excluded_indices = previously_used_train_indices | new_train_indices
    validation = take_unique(
        dataset["train"], "train", args.validation_count, SEED + 1, seen_train,
        excluded_indices=validation_excluded_indices,
    )
    new_validation_indices = {int(row["meta"]["source_index"]) for row in validation}

    excluded_test_indices = read_source_indices(args.exclude_test_data, args.test_source_split)
    seen_test = {
        dataset[args.test_source_split][index]["question"].strip()
        for index in excluded_test_indices
    }
    if args.test_source_split == "train":
        excluded_test_indices |= validation_excluded_indices | new_validation_indices
        seen_test |= {
            dataset["train"][index]["question"].strip()
            for index in validation_excluded_indices | new_validation_indices
        }
    else:
        seen_test |= {
            dataset["train"][index]["question"].strip()
            for index in validation_excluded_indices | new_validation_indices
        }
    test = take_unique(
        dataset[args.test_source_split], args.test_source_split, args.test_count, SEED + 2, seen_test,
        excluded_indices=excluded_test_indices,
    )
    test2 = take_unique(
        dataset[args.test_source_split], args.test_source_split, args.test2_count, SEED + 3, seen_test,
        excluded_indices=excluded_test_indices,
    )
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
        "split_source_splits": {
            "train": "train", "validation": "train",
            "test": args.test_source_split, "test2": args.test_source_split,
        },
        "excluded_test_source_split": args.test_source_split,
        "excluded_source_rows": {
            "train": {"source_split": "train", "source_indices": sorted(previously_used_train_indices)},
            "validation": {"source_split": "train", "source_indices": sorted(validation_excluded_indices)},
            "test": {"source_split": args.test_source_split, "source_indices": sorted(excluded_test_indices)},
            "test2": {"source_split": args.test_source_split, "source_indices": sorted(excluded_test_indices)},
        },
        "excluded_test_source_indices": sorted(excluded_test_indices),
        "sha256": split_hashes,
        "question_overlaps": {},
        "answer_parser": FINAL_ANSWER_RE.pattern,
    }
    prompt_sets = {
        name: {row["prompt"][0]["content"] for row in rows}
        for name, rows in {"train": train, "validation": validation, "test": test, "test2": test2}.items()
    }
    manifest["question_overlaps"] = {
        f"{left}_{right}": len(prompt_sets[left] & prompt_sets[right])
        for index, left in enumerate(["train", "validation", "test", "test2"])
        for right in ["train", "validation", "test", "test2"][index + 1 :]
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "seed": manifest["seed"],
        "counts": manifest["counts"],
        "split_source_splits": manifest["split_source_splits"],
        "sha256": manifest["sha256"],
        "question_overlaps": manifest["question_overlaps"],
        "excluded_source_row_counts": {
            name: len(value["source_indices"])
            for name, value in manifest["excluded_source_rows"].items()
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
