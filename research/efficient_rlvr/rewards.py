"""Small, explicit RLVR reward baselines for the Efficient RLVR branch."""

from __future__ import annotations

from typing import Any, Sequence

from scripts.math_task import completion_text, extract_answer


def correctness_gated_length_reward(
    completions: Sequence[Any],
    answer: Sequence[Any],
    completion_ids: Sequence[Sequence[int]] | None = None,
    *,
    max_completion_tokens: int,
    length_weight: float = 0.2,
    effective_group: Sequence[bool] | None = None,
    **_: Any,
) -> list[float | None]:
    """Give brevity credit only to verifier-correct completions.

    This is a testable baseline, not a novel algorithm. Wrong, malformed, and
    unparsed completions receive exactly zero regardless of length. Correct
    completions receive ``1 - lambda * min(tokens / cap, 1)``; requiring
    ``lambda < 1`` keeps every correct completion strictly above zero.
    """
    if max_completion_tokens < 1:
        raise ValueError("max_completion_tokens must be positive")
    if not 0.0 <= length_weight < 1.0:
        raise ValueError("length_weight must satisfy 0 <= length_weight < 1")
    if completion_ids is None:
        raise ValueError("completion_ids are required; character counts are not a token-length substitute")
    if not (len(completions) == len(answer) == len(completion_ids)):
        raise ValueError("completions, answer, and completion_ids must have equal lengths")
    if effective_group is not None and len(effective_group) != len(completions):
        raise ValueError("effective_group must have one flag per completion")

    rewards: list[float | None] = []
    for index, (completion, target, ids) in enumerate(zip(completions, answer, completion_ids, strict=True)):
        if effective_group is not None and not effective_group[index]:
            rewards.append(None)
            continue
        predicted, _source = extract_answer(completion_text(completion))
        if predicted is None or predicted != int(target):
            rewards.append(0.0)
            continue
        normalized_length = min(len(ids) / max_completion_tokens, 1.0)
        rewards.append(1.0 - length_weight * normalized_length)
    return rewards


def correctness_flags(completions: Sequence[Any], answers: Sequence[Any]) -> list[bool]:
    if len(completions) != len(answers):
        raise ValueError("completions and answers must have equal lengths")
    return [
        (predicted is not None and predicted == int(target))
        for completion, target in zip(completions, answers, strict=True)
        for predicted, _source in [extract_answer(completion_text(completion))]
    ]
