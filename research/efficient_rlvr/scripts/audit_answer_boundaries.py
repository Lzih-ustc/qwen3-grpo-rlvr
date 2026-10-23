#!/usr/bin/env python3
"""Offline post-hoc answer-boundary and output-cost audit.

The source prediction records and their original accuracy/parser decisions are
read-only inputs. Token spans are aligned against decoded text and therefore
are reconstructed measurements, not original generation token IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.math_task import BOXED_RE as BOXED
from scripts.math_task import FORMAT_RE as ANSWER_TAG

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def read_payload(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def marker_span(text: str, parse_source: str | None) -> tuple[int, int] | None:
    """Return the accepted answer marker span under the archived parser result."""
    pattern = ANSWER_TAG if parse_source == "answer_tag" else BOXED if parse_source == "boxed" else None
    if pattern is None:
        return None
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    return match.start(), match.end()


def token_regions(text: str, tokenizer: Any, span: tuple[int, int] | None) -> dict[str, int | None]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    if span is None:
        return {"completion_tokens": len(offsets), "before_answer_tokens": None, "answer_span_tokens": None, "after_answer_tokens": None}
    start, end = span
    counts = Counter()
    for left, right in offsets:
        left, right = int(left), int(right)
        if right <= left:
            continue
        if right <= start:
            counts["before_answer_tokens"] += 1
        elif left >= end:
            counts["after_answer_tokens"] += 1
        else:
            # Tokens overlapping either marker boundary are assigned to the
            # answer span, so the three regions partition all normal tokens.
            counts["answer_span_tokens"] += 1
    result = {
        "completion_tokens": len(offsets),
        "before_answer_tokens": counts["before_answer_tokens"],
        "answer_span_tokens": counts["answer_span_tokens"],
        "after_answer_tokens": counts["after_answer_tokens"],
    }
    if sum(int(result[k] or 0) for k in ("before_answer_tokens", "answer_span_tokens", "after_answer_tokens")) != len(offsets):
        raise ValueError("Tokenizer returned zero-width offsets that prevent boundary partitioning")
    return result


def exact_repeated_sentence_tokens(text: str, tokenizer: Any, span: tuple[int, int] | None) -> dict[str, int]:
    """Count tokenized copies of repeated, normalized sentence-sized strings.

    This is a conservative lexical repetition proxy. It intentionally misses
    paraphrases and repeated math written in a different form.
    """
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    # Keep source offsets so repeated copies can be attributed to the parsed
    # answer boundary. Newlines and sentence punctuation both end a segment.
    pattern = re.compile(r"[^.!?\n]+(?:[.!?]+(?=\s|$)|(?=\n|$))")
    for match in pattern.finditer(text):
        raw = match.group().strip()
        normalized = re.sub(r"\s+", " ", raw).casefold()
        if len(normalized) < 30:
            continue
        if normalized in seen:
            token_count = len(tokenizer(raw, add_special_tokens=False)["input_ids"])
            if span is None:
                counts["boundary_unknown"] += token_count
            elif match.end() <= span[0]:
                counts["before_answer"] += token_count
            elif match.start() >= span[1]:
                counts["after_answer"] += token_count
            else:
                counts["overlapping_answer_span"] += token_count
            counts["duplicate_sentence_occurrences"] += 1
        else:
            seen.add(normalized)
    counts.setdefault("before_answer", 0)
    counts.setdefault("after_answer", 0)
    counts.setdefault("overlapping_answer_span", 0)
    counts.setdefault("boundary_unknown", 0)
    counts.setdefault("duplicate_sentence_occurrences", 0)
    counts["duplicate_sentence_tokens"] = sum(counts[k] for k in (
        "before_answer", "after_answer", "overlapping_answer_span", "boundary_unknown"
    ))
    return dict(counts)


def summarize(values: list[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90_nearest_rank": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90_nearest_rank": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in rows if row["parsed"]]
    return {
        "n": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else None,
        "parsed": len(parsed),
        "parse_rate": len(parsed) / len(rows) if rows else None,
        "unparsed": len(rows) - len(parsed),
        "completion_retokenized_tokens": summarize([row["completion_tokens"] for row in rows]),
        "completion_retokenized_tokens_by_correctness": {
            label: summarize([row["completion_tokens"] for row in rows if row["correct"] is correct])
            for label, correct in (("correct", True), ("incorrect", False))
        },
        "accepted_answer_regions": {
            field: summarize([int(row[field]) for row in parsed])
            for field in ("before_answer_tokens", "answer_span_tokens", "after_answer_tokens")
        },
        "nonempty_post_answer_tail": sum(int(row["after_answer_tokens"]) > 0 for row in parsed),
        "post_answer_tail_rate_among_parsed": sum(int(row["after_answer_tokens"]) > 0 for row in parsed) / len(parsed) if parsed else None,
        "accepted_answer_source_counts": dict(sorted(Counter(row["parse_source"] for row in rows if row["parsed"]).items())),
        "exact_repetition_proxy": {
            "outputs_with_repeated_sentence": sum(row["exact_repeated_sentence_tokens"]["duplicate_sentence_occurrences"] > 0 for row in rows),
            "duplicate_sentence_occurrences": sum(row["exact_repeated_sentence_tokens"]["duplicate_sentence_occurrences"] for row in rows),
            "duplicate_sentence_tokens": sum(row["exact_repeated_sentence_tokens"]["duplicate_sentence_tokens"] for row in rows),
            "duplicate_tokens_before_answer": sum(row["exact_repeated_sentence_tokens"]["before_answer"] for row in parsed),
            "duplicate_tokens_after_answer": sum(row["exact_repeated_sentence_tokens"]["after_answer"] for row in parsed),
            "duplicate_tokens_boundary_unknown": sum(row["exact_repeated_sentence_tokens"]["boundary_unknown"] for row in rows),
        },
    }


def length_rank_quartiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["completion_tokens"], row["index"]))
    result = []
    for quartile in range(4):
        start = quartile * len(ordered) // 4
        end = (quartile + 1) * len(ordered) // 4
        subset = ordered[start:end]
        parsed = sum(row["parsed"] for row in subset)
        correct = sum(row["correct"] for row in subset)
        result.append({
            "quartile_by_rank": quartile + 1,
            "n": len(subset),
            "completion_tokens": summarize([row["completion_tokens"] for row in subset]),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else None,
            "parsed": parsed,
            "parse_rate": parsed / len(subset) if subset else None,
        })
    return result


def analyze_case(name: str, base_path: Path, adapter_path: Path, tokenizer: Any) -> dict[str, Any]:
    base, base_sha = read_payload(base_path)
    adapter, adapter_sha = read_payload(adapter_path)
    for field in ("eval_data_sha256", "decode"):
        if base.get(field) != adapter.get(field):
            raise ValueError(f"Paired files differ in {field}: {name}")
    base_predictions, adapter_predictions = base.get("predictions"), adapter.get("predictions")
    if not isinstance(base_predictions, list) or not isinstance(adapter_predictions, list):
        raise ValueError(f"Missing prediction rows: {name}")
    if len(base_predictions) != len(adapter_predictions):
        raise ValueError(f"Paired row counts differ: {name}")

    sides: dict[str, list[dict[str, Any]]] = {"base": [], "adapter": []}
    transitions: Counter[str] = Counter()
    for index, pair in enumerate(zip(base_predictions, adapter_predictions, strict=True)):
        base_row, adapter_row = pair
        if base_row.get("prompt") != adapter_row.get("prompt") or base_row.get("expected") != adapter_row.get("expected"):
            raise ValueError(f"Prompt/gold mismatch at row {index}: {name}")
        base_correct, adapter_correct = base_row.get("correct") is True, adapter_row.get("correct") is True
        transition = ("both_correct" if base_correct and adapter_correct else
                      "base_only_correct" if base_correct else
                      "adapter_only_correct" if adapter_correct else "both_wrong")
        transitions[transition] += 1
        for side, item in (("base", base_row), ("adapter", adapter_row)):
            text = item.get("completion")
            if not isinstance(text, str):
                raise ValueError(f"Missing completion text: {name}/{side}/{index}")
            accepted_span = marker_span(text, item.get("parse_source"))
            regions = token_regions(text, tokenizer, accepted_span)
            repetition = exact_repeated_sentence_tokens(text, tokenizer, accepted_span)
            if item.get("predicted") is None and item.get("parse_source") is not None:
                raise ValueError(f"Parser metadata inconsistent at {name}/{side}/{index}")
            sides[side].append({
                "index": index,
                "correct": item.get("correct") is True,
                "parsed": item.get("predicted") is not None,
                "parse_source": item.get("parse_source"),
                "completion_characters": len(text),
                **regions,
                "exact_repeated_sentence_tokens": repetition,
                "paired_transition": transition,
            })

    for side, payload in (("base", base), ("adapter", adapter)):
        rows = sides[side]
        if int(payload.get("eval_count", len(rows))) != len(rows):
            raise ValueError(f"eval_count differs from rows: {name}/{side}")
        if payload.get("correct_count") is not None and int(payload["correct_count"]) != sum(r["correct"] for r in rows):
            raise ValueError(f"Saved correct_count differs from rows: {name}/{side}")

    result = {
        "name": name,
        "eval_data_sha256": base.get("eval_data_sha256"),
        "decode": base.get("decode"),
        "source_files": {
            "base": {"path": str(base_path), "sha256": base_sha},
            "adapter": {"path": str(adapter_path), "sha256": adapter_sha},
        },
        "paired_correctness_transitions": dict(sorted(transitions.items())),
        "models": {},
    }
    for side, payload in (("base", base), ("adapter", adapter)):
        rows = sides[side]
        aggregate_tokens = payload.get("generated_tokens")
        elapsed = payload.get("elapsed_seconds")
        summary = summarize_rows(rows)
        summary["length_rank_quartiles"] = length_rank_quartiles(rows)
        summary["recorded_inference_cost"] = {
            "generated_tokens": aggregate_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": payload.get("tokens_per_second"),
            "tokens_per_correct_solution": aggregate_tokens / summary["correct"] if aggregate_tokens is not None and summary["correct"] else None,
            "seconds_per_correct_solution": elapsed / summary["correct"] if elapsed is not None and summary["correct"] else None,
            "whole_gpu_peak_memory_used_mib": payload.get("gpu_monitor", {}).get("whole_gpu_peak_memory_used_mib") if isinstance(payload.get("gpu_monitor"), dict) else None,
        }
        result["models"][side] = summary
    result["limitations"] = [
        "This is a post-hoc description and does not alter archived accuracy, parser labels, candidate selection, or test standards.",
        "Completion token counts and answer-region token counts are re-tokenized from decoded strings with the pinned tokenizer; original per-question generation token IDs were not saved.",
        "Only the answer marker accepted by the historical parser defines the answer span. For unparsed outputs, answer boundaries are unknown and segment counts are null.",
        "Text after a parsed answer is a measurable post-answer tail, not proof of semantic redundancy; semantic redundancy requires human annotation or a separately validated method.",
        "Aggregate generated tokens and elapsed time are historical measurements. They cannot attribute latency or token cost to individual questions.",
        "Length/correctness correlations are descriptive and do not establish that extra reasoning caused a correct answer.",
    ]
    return result


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 1 saved-trajectory audit (post-hoc)", "",
        "> This offline analysis is explicitly post-hoc. It does not revise archived scores, parser outputs, model selection, or evaluation rules.", "",
        "All per-completion token counts below are reconstructed from decoded text using the pinned Qwen3 tokenizer. Original token IDs were not retained. The answer span is the unique marker accepted by the original verifier; unparsed rows have unknown answer boundaries. Post-answer text is a structural tail metric, not a semantic redundancy label.", "",
    ]
    for case in report["cases"]:
        lines += [f"## {case['name']}", "", f"Evaluation data SHA-256: `{case['eval_data_sha256']}`", "",
                  "| Model | Original exact match | Parsed | Mean decoded-text tokens, correct / incorrect | Mean tokens before answer | Mean answer span | Mean tokens after answer | Parsed outputs with nonempty tail | Recorded generated tokens | Wall time | tokens/s | tokens per correct |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for side in ("base", "adapter"):
            model = case["models"][side]
            cost = model["recorded_inference_cost"]
            ok = model["completion_retokenized_tokens_by_correctness"]
            region = model["accepted_answer_regions"]
            fmt = lambda value: "—" if value is None else f"{float(value):.1f}"
            correct = model["correct"]
            n = model["n"]
            accuracy = f"{correct}/{n} ({100*float(model['accuracy']):.1f}%)"
            parsed_text = f"{model['parsed']}/{n} ({100*float(model['parse_rate']):.1f}%)"
            lines.append(
                f"| {side} | {accuracy} | {parsed_text} | {fmt(ok['correct']['mean'])} / {fmt(ok['incorrect']['mean'])} | "
                f"{fmt(region['before_answer_tokens']['mean'])} | {fmt(region['answer_span_tokens']['mean'])} | "
                f"{fmt(region['after_answer_tokens']['mean'])} | {model['nonempty_post_answer_tail']}/{model['parsed']} | "
                f"{cost['generated_tokens']:,} | {float(cost['elapsed_seconds'])/60:.1f} min | {float(cost['tokens_per_second']):.2f} | "
                f"{fmt(cost['tokens_per_correct_solution'])}"
            )
        for side in ("base", "adapter"):
            model = case["models"][side]
            repetition = model["exact_repetition_proxy"]
            lines += ["", f"Exact repetition proxy for {side} (literal repeated sentence-sized text only): "
                      f"{repetition['outputs_with_repeated_sentence']}/{model['n']} outputs; "
                      f"{repetition['duplicate_sentence_tokens']:,} duplicate-copy tokens total, "
                      f"{repetition['duplicate_tokens_before_answer']:,} before accepted answer and "
                      f"{repetition['duplicate_tokens_after_answer']:,} after it, and "
                      f"{repetition['duplicate_tokens_boundary_unknown']:,} in unparsed outputs with unknown boundaries. "
                      "This misses paraphrase or repeated equations and is not a complete redundancy measure."]
        transitions = case["paired_correctness_transitions"]
        lines += ["", f"Paired changes (same items): adapter-only correct {transitions.get('adapter_only_correct',0)}, base-only correct {transitions.get('base_only_correct',0)}, both correct {transitions.get('both_correct',0)}, both incorrect {transitions.get('both_wrong',0)}.", "",
                  "### Completion-length rank quartiles", "", "| Model | Length quartile (short to long) | N | Mean re-tokenized tokens | Exact match | Parse rate |", "|---|---:|---:|---:|---:|---:|"]
        for side in ("base", "adapter"):
            for q in case["models"][side]["length_rank_quartiles"]:
                lines.append(f"| {side} | {q['quartile_by_rank']} | {q['n']} | {q['completion_tokens']['mean']:.1f} | {100*q['accuracy']:.1f}% | {100*q['parse_rate']:.1f}% |")
        lines += [""]
    lines += ["## Interpretation boundaries", "",
              "- The already published historical scores remain authoritative. This audit is additive and post-hoc.",
              "- A longer decoded output can contain useful reasoning, repetition, a delayed answer, or an answer-format failure. Length alone cannot distinguish these cases.",
              "- A nonempty post-answer tail is a candidate for manual review; the metric does not label it redundant.",
              "- Aggregate inference costs measure the full evaluation run, including batching/runtime effects; they are not per-question latency estimates.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=3, action="append", required=True, metavar=("NAME", "BASE_JSON", "ADAPTER_JSON"))
    parser.add_argument("--tokenizer-path", help="Optional local snapshot directory; otherwise resolve the pinned model from the HF cache in offline-only mode")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or MODEL_ID, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer with offset mappings is required; no network fallback is allowed")
    cases = [analyze_case(name, Path(base), Path(adapter), tokenizer) for name, base, adapter in args.case]
    report = {
        "analysis_label": "post-hoc; decoded-text answer-boundary and inference-cost audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "historical_results_modified": False,
        "tokenizer": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "loaded_local_only": True,
                      "local_snapshot_path_supplied": args.tokenizer_path is not None,
                      "fast_offsets": True, "token_counts_are_original_generated_ids": False},
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "passed", "cases": [case["name"] for case in cases], "output": str(args.output_json)}))


if __name__ == "__main__":
    main()
