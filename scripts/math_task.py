"""Deterministic toy math data, prompts, answer parsing, and reward."""
from __future__ import annotations

import random
import re
from typing import Any


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SEED = 20261022
FORMAT_RE = re.compile(r"<answer>\s*([-+]?\d+)\s*</answer>", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\s*\{\s*([-+]?\d+)\s*\}")


def make_prompt(a: int, operator: str, b: int, template_id: int) -> str:
    expression = f"{a} {operator} {b}"
    templates = [
        f"Compute {expression}. Show a short calculation and put the final integer exactly inside <answer>...</answer>.",
        f"Evaluate {expression}. Give one brief line of work, then write <answer>integer</answer> with the result.",
        f"What is {expression}? Return the final integer using this exact format: <answer>integer</answer>.",
    ]
    return templates[template_id % len(templates)]


def answer_for(a: int, operator: str, b: int) -> int:
    if operator == "+":
        return a + b
    if operator == "-":
        return a - b
    if operator == "*":
        return a * b
    raise ValueError(f"Unsupported operator: {operator}")


def build_examples(
    count: int,
    seed: int,
    *,
    low: int = 2,
    high: int = 99,
    forbidden_questions: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    seen = set(forbidden_questions) if forbidden_questions is not None else None
    index = 0
    while len(rows) < count:
        if index >= max(count * 100, 10_000):
            raise RuntimeError(f"Could not sample {count} unique questions from the configured range")
        operator = ["+", "-", "*"][index % 3]
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        if operator == "-" and b > a:
            a, b = b, a
        prompt_text = make_prompt(a, operator, b, index)
        if seen is not None and prompt_text in seen:
            index += 1
            continue
        if seen is not None:
            seen.add(prompt_text)
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "answer": str(answer_for(a, operator, b)),
                "meta": {"a": a, "operator": operator, "b": b, "split_index": index},
            }
        )
        index += 1
    return rows


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
    return str(completion)


def extract_answer(text: str) -> tuple[int | None, str | None]:
    """Parse exactly one marked integer; returns (value, source) or (None, None)."""
    marked = FORMAT_RE.findall(text)
    if len(marked) == 1:
        return int(marked[0]), "answer_tag"
    if len(marked) > 1:
        return None, None
    boxed = BOXED_RE.findall(text)
    if len(boxed) == 1:
        return int(boxed[0]), "boxed"
    return None, None


def exact_math_reward(completions: list[Any], answer: list[Any], **_: Any) -> list[float]:
    """Strict, deterministic reward: exact integer correctness plus small shaping.

    XML tag: +0.05 for requested format. XML or a single boxed integer receives
    up to +0.10 for numerical proximity. Exact integer: an additional +1.0.
    Unmarked/ambiguous text receives zero.
    """
    rewards: list[float] = []
    for completion, target in zip(completions, answer, strict=True):
        predicted, source = extract_answer(completion_text(completion))
        if predicted is None:
            rewards.append(0.0)
            continue
        target_int = int(target)
        distance = abs(predicted - target_int)
        format_bonus = 0.05 if source == "answer_tag" else 0.0
        rewards.append(format_bonus + 0.10 / (1 + distance) + (1.0 if distance == 0 else 0.0))
    return rewards


def exact_format_reward(completions: list[Any], answer: list[Any], **_: Any) -> list[float]:
    """Reward only verifier-confirmed answers, with a small requested-format bonus.

    Unlike ``exact_math_reward``, this has no numerical-distance shaping. Incorrect
    integer guesses therefore cannot earn relative advantage merely for being
    numerically close to the target.
    """
    rewards: list[float] = []
    for completion, target in zip(completions, answer, strict=True):
        predicted, source = extract_answer(completion_text(completion))
        if predicted is None:
            rewards.append(0.0)
            continue
        format_bonus = 0.05 if source == "answer_tag" else 0.0
        rewards.append(format_bonus + (1.0 if predicted == int(target) else 0.0))
    return rewards
