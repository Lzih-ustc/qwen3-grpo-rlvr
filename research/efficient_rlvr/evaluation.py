"""Length-controlled, token-prefix evaluation helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from scripts.math_task import extract_answer


def evaluate_completion_prefixes(
    completion_ids: Sequence[int],
    expected: Any,
    tokenizer: Any,
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    if not budgets or any(int(budget) < 1 for budget in budgets):
        raise ValueError("budgets must contain positive integers")
    if list(budgets) != sorted(set(int(budget) for budget in budgets)):
        raise ValueError("budgets must be unique and in ascending order")
    results = []
    for budget in budgets:
        prefix_ids = list(completion_ids[: int(budget)])
        text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        predicted, source = extract_answer(text)
        results.append({
            "budget_tokens": int(budget),
            "generated_prefix_tokens": len(prefix_ids),
            "prefix_truncated": len(completion_ids) > int(budget),
            "parsed": predicted is not None,
            "correct": predicted is not None and predicted == int(expected),
            "parse_source": source,
        })
    return results


def summarize_length_curve(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    budgets = sorted({item["budget_tokens"] for row in rows for item in row["prefix_results"]})
    result = []
    for budget in budgets:
        observations = [next(x for x in row["prefix_results"] if x["budget_tokens"] == budget) for row in rows]
        n = len(observations)
        correct = sum(item["correct"] for item in observations)
        parsed = sum(item["parsed"] for item in observations)
        result.append({
            "budget_tokens": budget,
            "n": n,
            "correct": correct,
            "accuracy": correct / n,
            "parsed": parsed,
            "parse_rate": parsed / n,
            "truncated": sum(item["prefix_truncated"] for item in observations),
            "actual_prefix_tokens_mean": sum(item["generated_prefix_tokens"] for item in observations) / n,
        })
    return result


def group_filter_cost_summary(group_stats: dict[str, Any]) -> dict[str, float | int | None]:
    attempted = int(group_stats.get("attempted_completion_tokens", 0))
    retained = int(group_stats.get("retained_completion_tokens", 0))
    groups = int(group_stats.get("retained_groups", 0))
    return {
        "attempted_completion_tokens": attempted,
        "retained_completion_tokens": retained,
        "discarded_completion_tokens": attempted - retained,
        "tokens_per_retained_group": attempted / groups if groups else None,
    }
