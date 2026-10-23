#!/usr/bin/env python3
"""Offline structure, difficulty-proxy, and truncation audit for locked data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "efficient-rlvr-math-v1"
TOKEN_REPORT = ROOT / "reports" / "efficient_rlvr_phase3_prompt_length_audit.json"
OUTPUT_JSON = ROOT / "reports" / "efficient_rlvr_phase4b_data_audit.json"
OUTPUT_MD = ROOT / "research" / "efficient_rlvr" / "phase4b_data_audit.md"
EXPANSION_SUFFIXES = ("_more_samples", "_longer", "_big", "_more")
WORD_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
OPERATOR_RE = re.compile(r"[+\-*/=^×÷]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, int((p * len(ordered) + 0.999999)) - 1)]


def module_family(name: str) -> str:
    for suffix in EXPANSION_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_rows(name: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    split = manifest["splits"][name]
    path = DATA_DIR / split["file"]
    actual_hash = sha256(path)
    if actual_hash != split["sha256"]:
        raise AssertionError(f"Locked {name} split hash mismatch")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize_lengths(values: list[int]) -> dict[str, int | float]:
    return {
        "min": min(values),
        "mean": round(mean(values), 3),
        "p50_nearest_rank": percentile(values, 0.50),
        "p90_nearest_rank": percentile(values, 0.90),
        "p95_nearest_rank": percentile(values, 0.95),
        "p99_nearest_rank": percentile(values, 0.99),
        "max": max(values),
    }


def row_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    question_chars: list[int] = []
    word_counts: list[int] = []
    number_counts: list[int] = []
    operator_counts: list[int] = []
    answer_digits: list[int] = []
    answer_magnitudes: list[int] = []
    module_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    domain_counts: Counter[str] = Counter()
    for row in rows:
        question = str(row["meta"]["source_question"])
        answer = str(row["answer"])
        module = str(row["meta"]["generator_module"])
        domain_counts[module.split("/", 1)[0]] += 1
        question_chars.append(len(question))
        word_counts.append(len(WORD_RE.findall(question)))
        number_counts.append(len(NUMBER_RE.findall(question)))
        operator_counts.append(len(OPERATOR_RE.findall(question)))
        answer_digits.append(len(answer.lstrip("+-")))
        answer_magnitudes.append(abs(int(answer)))
        module_rows[module].append(row)
    return {
        "rows": len(rows),
        "module_count": len(module_rows),
        "question_characters": summarize_lengths(question_chars),
        "question_word_or_number_tokens": summarize_lengths(word_counts),
        "numeric_literals_per_question": summarize_lengths(number_counts),
        "visible_operator_symbols_per_question": summarize_lengths(operator_counts),
        "answer_digits": summarize_lengths(answer_digits),
        "answer_magnitude": {
            "max_digits": max(answer_digits),
            "max_absolute_value": max(answer_magnitudes),
            "over_12_digits": sum(value > 12 for value in answer_digits),
            "over_20_digits": sum(value > 20 for value in answer_digits),
        },
        "module_rows": {
            module: {
                "count": len(items),
                "mean_question_chars": round(mean(len(str(row["meta"]["source_question"])) for row in items), 2),
                "mean_answer_digits": round(mean(len(str(row["answer"]).lstrip("+-")) for row in items), 2),
            }
            for module, items in sorted(module_rows.items())
        },
        "domain_rows": dict(sorted(domain_counts.items())),
    }


def module_overlap(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sets = {
        name: set(value["module_rows"])
        for name, value in splits.items()
    }
    family_sets = {
        name: {module_family(module) for module in modules}
        for name, modules in sets.items()
    }
    top_sets = {
        name: {module.split("/", 1)[0] for module in modules}
        for name, modules in sets.items()
    }
    result: dict[str, Any] = {}
    names = ("train", "validation", "test")
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = f"{left}_vs_{right}"
            result[key] = {
                "exact_leaf_overlap_count": len(sets[left] & sets[right]),
                "exact_leaf_overlap": sorted(sets[left] & sets[right]),
                "expansion_normalized_family_overlap_count": len(family_sets[left] & family_sets[right]),
                "expansion_normalized_family_overlap": sorted(family_sets[left] & family_sets[right]),
                "top_level_domain_overlap": sorted(top_sets[left] & top_sets[right]),
                "left_only_leaf_count": len(sets[left] - sets[right]),
                "right_only_leaf_count": len(sets[right] - sets[left]),
            }
    result["interpretation"] = (
        "Exact generator-leaf disjointness does not imply concept-level independence. "
        "The expansion-normalized family view strips only the upstream naming suffixes "
        "_big, _longer, _more, and _more_samples; it is a structural overlap diagnostic, "
        "not a calibrated math-difficulty score."
    )
    return result


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 4B locked-data audit",
        "",
        "Status: **PASSED for integrity and prompt-input limits; completion-length risk remains unmeasured**. This is an offline audit and does not change the locked JSONL files.",
        "",
        "## Split-level difficulty proxies",
        "",
        "Question character/word/number counts and answer digit/magnitude distributions are descriptive proxies only; they are not validated difficulty labels. The held-out test comes from the source extrapolation regime.",
        "",
        "| Split | Rows | Unique leaves | Question chars mean / p95 / max | Words/numbers mean / p95 / max | Numeric literals mean / p95 / max | Answer digits mean / p95 / max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, split in report["split_summaries"].items():
        chars = split["question_characters"]
        words = split["question_word_or_number_tokens"]
        nums = split["numeric_literals_per_question"]
        digits = split["answer_digits"]
        lines.append(
            f"| {name} | {split['rows']} | {split['module_count']} | "
            f"{chars['mean']} / {chars['p95_nearest_rank']} / {chars['max']} | "
            f"{words['mean']} / {words['p95_nearest_rank']} / {words['max']} | "
            f"{nums['mean']} / {nums['p95_nearest_rank']} / {nums['max']} | "
            f"{digits['mean']} / {digits['p95_nearest_rank']} / {digits['max']} |"
        )
    lines.extend(["", "## Module overlap", ""])
    for pair in ("train_vs_validation", "train_vs_test", "validation_vs_test"):
        item = report["module_overlap"][pair]
        lines.append(
            f"- **{pair}:** {item['exact_leaf_overlap_count']} exact leaf names; "
            f"{item['expansion_normalized_family_overlap_count']} expansion-normalized families; "
            f"shared top-level domains: {', '.join(item['top_level_domain_overlap']) or 'none'}."
        )
    lines.extend([
        "",
        report["module_overlap"]["interpretation"],
        "",
        "## Generator-domain mix",
        "",
        "| Split | Domain | Rows | Share |",
        "|---|---|---:|---:|",
    ])
    for name, split in report["split_summaries"].items():
        for domain, count in split["domain_rows"].items():
            lines.append(f"| {name} | {domain} | {count} | {100 * count / split['rows']:.1f}% |")
    lines.extend([
        "",
        "## Prompt and completion truncation",
        "",
        f"The pinned Qwen3 tokenizer audit at a 256-token prompt cap passed with no over-cap prompts. Max chat-template prompt lengths were train {report['prompt_token_audit']['train']['max']}, validation {report['prompt_token_audit']['validation']['max']}, and test {report['prompt_token_audit']['test']['max']} tokens. Thus the observed prompt-side cap margin is at least {min(256 - report['prompt_token_audit'][name]['max'] for name in ('train', 'validation', 'test'))} tokens.",
        "",
        "No outputs have been generated from these locked splits, so 256-token completion truncation rate, answer-at-cap rate, and CGLR reward effects at the cap are **unknown**. Do not infer them from target answer length or from the earlier smoke on different prompts. Measure them in validation; if the preregistered cap-hit threshold is exceeded, stop before opening the independent test.",
        "",
        "## Integrity",
        "",
        "All three source JSONL SHA-256 hashes match the immutable manifest. This report reads the files and does not rewrite them.",
        "",
    ])
    return "\n".join(lines)


def audit() -> dict[str, Any]:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    rows = {name: load_rows(name, manifest) for name in ("train", "validation", "test")}
    split_summaries = {name: row_features(value) for name, value in rows.items()}
    token_report = json.loads(TOKEN_REPORT.read_text(encoding="utf-8"))
    if token_report["max_prompt_length"] != 256 or token_report["status"] != "passed":
        raise AssertionError("Pinned prompt-token audit is missing or did not pass at the 256-token cap")
    return {
        "status": "passed",
        "audit_scope": "Offline data structure, descriptive difficulty proxies, module overlap, prompt cap, and split-hash verification; no generation or training.",
        "dataset_id": manifest["dataset_id"],
        "upstream_revision": manifest["source"]["revision"],
        "split_hashes_verified": {
            name: manifest["splits"][name]["sha256"] for name in rows
        },
        "difficulty_proxy_limitations": [
            "Question length, visible numbers/operators, answer digit count, and generator module are not calibrated difficulty labels.",
            "Train, interpolation-validation, and extrapolation-test regimes are structurally different; this audit does not claim equal difficulty or concept-level independence.",
            "Completion-length and reasoning truncation cannot be measured until a model generates on the training/validation data.",
        ],
        "module_overlap": module_overlap(split_summaries),
        "split_summaries": split_summaries,
        "prompt_token_audit": token_report["split_results"],
        "prompt_cap": {
            "tokens": token_report["max_prompt_length"],
            "over_cap_by_split": {name: token_report["split_results"][name]["over_cap"] for name in rows},
            "completion_cap_tokens": 256,
            "completion_truncation_risk": "unknown until generated outputs are collected; no new-split model evaluation has run",
        },
    }


def main() -> None:
    report = audit()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "split_hashes_verified": report["split_hashes_verified"],
        "module_overlap": {key: value for key, value in report["module_overlap"].items() if key != "interpretation"},
        "prompt_cap": report["prompt_cap"],
        "outputs": [str(OUTPUT_JSON), str(OUTPUT_MD)],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
