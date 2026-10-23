#!/usr/bin/env python3
"""Post-hoc paired analysis of saved Base/adapter prediction JSON files.

This script is deliberately offline. It never loads model weights or changes the
source prediction files or their original accuracy fields.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def quantile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return float(ordered[index])


def length_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(key: str) -> dict[str, float | int | None]:
        values = [int(row[key]) for row in records]
        return {
            "n": len(values),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p90_nearest_rank": quantile(values, 0.90),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    result = {
        "completion_characters": summarize("characters"),
        "completion_whitespace_words": summarize("whitespace_words"),
        "completion_lines": summarize("lines"),
    }
    if records and "posthoc_retokenized_tokens" in records[0]:
        result["completion_posthoc_retokenized_tokens"] = summarize("posthoc_retokenized_tokens")
    return result


def rows_by_length(predictions: list[dict[str, Any]], tokenizer: Any = None) -> list[dict[str, Any]]:
    records = []
    for index, item in enumerate(predictions):
        completion = item.get("completion")
        if not isinstance(completion, str):
            raise ValueError(f"Prediction {index} has no string completion")
        record = {
                "index": index,
                "correct": item.get("correct") is True,
                "parsed": item.get("predicted") is not None,
                "parse_source": item.get("parse_source"),
                "characters": len(completion),
                "whitespace_words": len(completion.split()),
                "lines": len(completion.splitlines()) or 1,
            }
        if tokenizer is not None:
            # Re-tokenized decoded text is not the original generated token sequence.
            record["posthoc_retokenized_tokens"] = len(tokenizer.encode(completion, add_special_tokens=False))
        records.append(record)
    return records


def model_summary(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    reported_count = int(payload.get("eval_count", len(rows)))
    correct_count = sum(row["correct"] for row in rows)
    parse_count = sum(row["parsed"] for row in rows)
    if reported_count != len(rows):
        raise ValueError("eval_count does not match the saved prediction row count")
    if payload.get("correct_count") is not None and int(payload["correct_count"]) != correct_count:
        raise ValueError("Saved row correctness does not match the original correct_count")

    by_correctness = {
        label: length_stats([row for row in rows if row["correct"] is correct])
        for label, correct in (("correct", True), ("incorrect", False))
    }
    by_parse = {
        label: length_stats([row for row in rows if row["parsed"] is parsed])
        for label, parsed in (("parsed", True), ("unparsed", False))
    }
    parse_sources = Counter(str(row["parse_source"]) for row in rows if row["parse_source"] is not None)
    n = len(rows)
    generated_tokens = payload.get("generated_tokens")
    elapsed_seconds = payload.get("elapsed_seconds")
    return {
        "tag": payload.get("tag"),
        "adapter": payload.get("adapter"),
        "eval_count": n,
        "correct_count_from_saved_rows": correct_count,
        "original_correct_count": payload.get("correct_count"),
        "original_accuracy": payload.get("accuracy"),
        "parse_count_from_saved_rows": parse_count,
        "parse_rate_from_saved_rows": parse_count / n if n else None,
        "original_parse_rate": payload.get("parse_rate"),
        "unparsed_count": n - parse_count,
        "parse_source_counts": dict(sorted(parse_sources.items())),
        "lengths_all": length_stats(rows),
        "lengths_by_correctness": by_correctness,
        "lengths_by_parse_status": by_parse,
        "posthoc_retokenized_completion_tokens": {
            "available": bool(rows and "posthoc_retokenized_tokens" in rows[0]),
            "sum_from_decoded_text": sum(row["posthoc_retokenized_tokens"] for row in rows)
            if rows and "posthoc_retokenized_tokens" in rows[0]
            else None,
            "reconstructed_from_original_token_ids": False,
        },
        "inference_cost": {
            "generated_tokens_total": generated_tokens,
            "elapsed_seconds_total": elapsed_seconds,
            "tokens_per_second": payload.get("tokens_per_second"),
            "whole_gpu_peak_memory_used_mib": payload.get("gpu_monitor", {}).get("peak_used_mib")
            if isinstance(payload.get("gpu_monitor"), dict)
            else None,
            "tokens_per_correct_answer": generated_tokens / correct_count
            if generated_tokens is not None and correct_count
            else None,
            "seconds_per_correct_answer": elapsed_seconds / correct_count
            if elapsed_seconds is not None and correct_count
            else None,
            "per_question_token_ids_saved": False,
            "per_question_latency_saved": False,
        },
    }


def analyze_case(name: str, base_path: Path, adapter_path: Path, tokenizer: Any = None) -> dict[str, Any]:
    base = read_json(base_path)
    adapter = read_json(adapter_path)
    base_predictions = base.get("predictions")
    adapter_predictions = adapter.get("predictions")
    if not isinstance(base_predictions, list) or not isinstance(adapter_predictions, list):
        raise ValueError(f"Both inputs for {name} must contain saved predictions")
    if len(base_predictions) != len(adapter_predictions):
        raise ValueError(f"Prediction counts differ for {name}")
    if base.get("eval_data_sha256") != adapter.get("eval_data_sha256"):
        raise ValueError(f"Evaluation data hashes differ for {name}")
    if base.get("decode") != adapter.get("decode"):
        raise ValueError(f"Decode settings differ for {name}")

    transitions: Counter[str] = Counter()
    transition_lengths: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"base": [], "adapter": []})
    paired_mismatches = 0
    for index, (base_item, adapter_item) in enumerate(zip(base_predictions, adapter_predictions)):
        if base_item.get("prompt") != adapter_item.get("prompt") or base_item.get("expected") != adapter_item.get("expected"):
            raise ValueError(f"Prompt/gold mismatch at row {index} for {name}")
        base_correct = base_item.get("correct") is True
        adapter_correct = adapter_item.get("correct") is True
        if base_correct and adapter_correct:
            transition = "both_correct"
        elif base_correct:
            transition = "base_only_correct"
        elif adapter_correct:
            transition = "adapter_only_correct"
        else:
            transition = "both_wrong"
        transitions[transition] += 1
        paired_mismatches += base_correct != adapter_correct

        for side, item in (("base", base_item), ("adapter", adapter_item)):
            completion = item.get("completion")
            if not isinstance(completion, str):
                raise ValueError(f"Missing completion at row {index} for {name}/{side}")
            record = {
                    "correct": item.get("correct") is True,
                    "parsed": item.get("predicted") is not None,
                    "characters": len(completion),
                    "whitespace_words": len(completion.split()),
                    "lines": len(completion.splitlines()) or 1,
                }
            if tokenizer is not None:
                record["posthoc_retokenized_tokens"] = len(tokenizer.encode(completion, add_special_tokens=False))
            transition_lengths[transition][side].append(record)

    base_rows = rows_by_length(base_predictions, tokenizer)
    adapter_rows = rows_by_length(adapter_predictions, tokenizer)
    base_summary = model_summary(base, base_rows)
    adapter_summary = model_summary(adapter, adapter_rows)
    if base_summary["parse_count_from_saved_rows"] != round(float(base.get("parse_rate", 0)) * len(base_rows)):
        # Recorded parse_rate is rounded to three decimals; tolerate that precision only.
        if abs(base_summary["parse_rate_from_saved_rows"] - float(base.get("parse_rate", 0))) > 0.0011:
            raise ValueError(f"Base saved parse status disagrees with the reported rate for {name}")
    if abs(adapter_summary["parse_rate_from_saved_rows"] - float(adapter.get("parse_rate", 0))) > 0.0011:
        raise ValueError(f"Adapter saved parse status disagrees with the reported rate for {name}")

    paired_cost = {}
    for key in ("generated_tokens_total", "elapsed_seconds_total", "tokens_per_second", "whole_gpu_peak_memory_used_mib"):
        base_value = base_summary["inference_cost"].get(key)
        adapter_value = adapter_summary["inference_cost"].get(key)
        paired_cost[key] = {
            "base": base_value,
            "adapter": adapter_value,
            "adapter_to_base_ratio": adapter_value / base_value if base_value not in (None, 0) and adapter_value is not None else None,
        }

    transitions_with_length_stats = {}
    for label, sides in transition_lengths.items():
        transitions_with_length_stats[label] = {
            side: length_stats(records) for side, records in sides.items()
        }

    return {
        "name": name,
        "analysis_label": "post-hoc; descriptive only; does not revise original accuracy",
        "n": len(base_predictions),
        "eval_data_sha256": base.get("eval_data_sha256"),
        "decode": base.get("decode"),
        "base": base_summary,
        "adapter": adapter_summary,
        "paired_correctness_transitions": dict(sorted(transitions.items())),
        "paired_discordant_questions": paired_mismatches,
        "completion_lengths_by_paired_transition": transitions_with_length_stats,
        "aggregate_inference_cost_comparison": paired_cost,
        "limitations": [
            "The saved prediction files contain decoded text but no per-question token IDs or latency; per-question lengths therefore use Unicode code points, whitespace words, and lines.",
            "If supplied, tokenizer lengths are reconstructed from decoded text and are not the original generated token IDs; recorded aggregate token totals remain authoritative.",
            "Aggregate generated-token counts and wall-clock times are the recorded inference measurements; dividing them by correct answers is descriptive, not a per-answer causal cost.",
            "Output length/correctness associations are post-hoc and do not imply that longer reasoning caused correctness.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Saved prediction output-length audit (post-hoc)",
        "",
        "> This offline analysis uses the saved per-question outputs. It does not modify source evaluation JSON, historical accuracy, candidate selection, or test standards.",
        "",
        "Character and whitespace-word counts are decoded-text measures. Optional Qwen token counts are re-tokenized from decoded strings, not recovered generation IDs. Original aggregate generated tokens, elapsed time, throughput, and GPU memory remain authoritative.",
        "",
    ]
    for case in report["cases"]:
        lines.extend([f"## {case['name']}", ""])
        lines.append("| Model | Accuracy | Parse rate | Mean chars (correct / incorrect) | Mean words (correct / incorrect) | Re-tokenized tokens (correct / incorrect) | Generated tokens | Wall time | tokens/s | tokens / correct |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label in ("base", "adapter"):
            model = case[label]
            lengths = model["lengths_by_correctness"]
            chars = lengths["correct"]["completion_characters"]["mean"], lengths["incorrect"]["completion_characters"]["mean"]
            words = lengths["correct"]["completion_whitespace_words"]["mean"], lengths["incorrect"]["completion_whitespace_words"]["mean"]
            token_lengths = lengths["correct"].get("completion_posthoc_retokenized_tokens", {}).get("mean"), lengths["incorrect"].get("completion_posthoc_retokenized_tokens", {}).get("mean")
            cost = model["inference_cost"]
            accuracy = f"{model['original_correct_count']}/{model['eval_count']} ({100*float(model['original_accuracy']):.1f}%)"
            token_text = "—" if token_lengths[0] is None or token_lengths[1] is None else f"{token_lengths[0]:.1f} / {token_lengths[1]:.1f}"
            lines.append(
                f"| {label} | {accuracy} | {100*float(model['parse_rate_from_saved_rows']):.1f}% ({model['unparsed_count']} unparsed) | "
                f"{chars[0]:.1f} / {chars[1]:.1f} | {words[0]:.1f} / {words[1]:.1f} | {token_text} | "
                f"{cost['generated_tokens_total']:,} | {float(cost['elapsed_seconds_total'])/60:.1f} min | "
                f"{float(cost['tokens_per_second']):.2f} | {float(cost['tokens_per_correct_answer']):.1f}"
            )
        trans = case["paired_correctness_transitions"]
        lines.extend(
            [
                "",
                f"Paired correctness changes: both correct {trans.get('both_correct',0)}, base-only correct {trans.get('base_only_correct',0)}, adapter-only correct {trans.get('adapter_only_correct',0)}, both wrong {trans.get('both_wrong',0)}.",
                "",
                "| Paired outcome | Base mean chars | Adapter mean chars | Base mean words | Adapter mean words | Base re-tokenized tokens | Adapter re-tokenized tokens |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for transition, label in (("both_correct", "Both correct"), ("base_only_correct", "Base only correct"), ("adapter_only_correct", "Adapter only correct"), ("both_wrong", "Both wrong")):
            stats = case["completion_lengths_by_paired_transition"].get(transition, {})
            vals = []
            for side in ("base", "adapter"):
                side_stats = stats.get(side, {})
                vals.append(side_stats.get("completion_characters", {}).get("mean"))
            words = []
            for side in ("base", "adapter"):
                words.append(stats.get(side, {}).get("completion_whitespace_words", {}).get("mean"))
            tokens = []
            for side in ("base", "adapter"):
                tokens.append(stats.get(side, {}).get("completion_posthoc_retokenized_tokens", {}).get("mean"))
            def fmt(x: Any) -> str:
                return "—" if x is None else f"{float(x):.1f}"
            lines.append(f"| {label} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(words[0])} | {fmt(words[1])} | {fmt(tokens[0])} | {fmt(tokens[1])} |")
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Unparsed means the saved verifier did not extract a final numeric answer (`predicted` is null); this is not silently repaired even when a human could infer the intended arithmetic.",
            "- The evaluation protocol used greedy decoding with the same saved decode metadata for each pair. Per-question token IDs and latency were not retained, so this report does not invent them from text lengths.",
            "- Length differences by correctness are descriptive post-hoc associations. Original exact-match values and all locked test decisions remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=3, action="append", required=True, metavar=("NAME", "BASE_JSON", "ADAPTER_JSON"))
    parser.add_argument("--output", type=Path, required=True, help="Combined JSON output")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown summary output")
    parser.add_argument("--tokenizer", help="Optional local tokenizer path or cached model id; loading is local-only")
    parser.add_argument("--tokenizer-revision", default=MODEL_REVISION, help="Pinned tokenizer revision for a cached model id")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            revision=args.tokenizer_revision,
            local_files_only=True,
            trust_remote_code=False,
        )
    cases = [analyze_case(name, Path(base), Path(adapter), tokenizer) for name, base, adapter in args.case]
    report = {
        "analysis": "post-hoc output length, saved verifier status, and inference cost",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_prediction_files_are_unchanged": True,
        "tokenizer": {
            "name_or_path": args.tokenizer,
            "revision": args.tokenizer_revision if args.tokenizer else None,
            "loaded_local_only": bool(args.tokenizer),
            "add_special_tokens": False,
            "meaning": "re-tokenization of decoded completion text, not original generated token IDs",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "passed", "cases": [case["name"] for case in cases], "output": str(args.output)}))


if __name__ == "__main__":
    main()
