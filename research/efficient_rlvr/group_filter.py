"""Group-level correctness filtering, a DAPO-style baseline implementation.

This code filters entire prompt groups. It is distinct from GFPO's response-
level ranking/filtering and makes no algorithmic novelty claim.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from research.efficient_rlvr.rewards import correctness_flags


@dataclass
class GroupFilterStats:
    requested_groups: int = 0
    candidate_groups: int = 0
    retained_groups: int = 0
    exhausted_groups: int = 0
    attempted_completions: int = 0
    retained_completions: int = 0
    attempted_completion_tokens: int = 0
    retained_completion_tokens: int = 0
    attempted_group_outcomes: Counter[str] = field(default_factory=Counter)

    def record_candidate(self, correct: Sequence[bool], token_lengths: Sequence[int]) -> bool:
        if len(correct) != len(token_lengths) or not correct:
            raise ValueError("Each candidate group needs matching, nonempty correctness and length lists")
        self.candidate_groups += 1
        self.attempted_completions += len(correct)
        tokens = sum(int(length) for length in token_lengths)
        self.attempted_completion_tokens += tokens
        count = sum(correct)
        mixed = 0 < count < len(correct)
        self.attempted_group_outcomes["mixed" if mixed else "all_correct" if count else "all_incorrect"] += 1
        if mixed:
            self.retained_groups += 1
            self.retained_completions += len(correct)
            self.retained_completion_tokens += tokens
        return mixed

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["attempted_group_outcomes"] = dict(sorted(self.attempted_group_outcomes.items()))
        result["discarded_completion_tokens"] = self.attempted_completion_tokens - self.retained_completion_tokens
        result["effective_group_yield"] = self.retained_groups / self.candidate_groups if self.candidate_groups else None
        result["tokens_per_retained_group"] = self.attempted_completion_tokens / self.retained_groups if self.retained_groups else None
        return result


def retain_correctness_mixed_groups(
    completion_groups: Sequence[Sequence[Any]],
    answer_groups: Sequence[Sequence[Any]],
    token_length_groups: Sequence[Sequence[int]],
) -> tuple[list[int], GroupFilterStats]:
    if not (len(completion_groups) == len(answer_groups) == len(token_length_groups)):
        raise ValueError("Group collections must have equal lengths")
    stats = GroupFilterStats(requested_groups=len(completion_groups))
    kept = []
    for index, (completions, answers, lengths) in enumerate(zip(completion_groups, answer_groups, token_length_groups, strict=True)):
        if not (len(completions) == len(answers) == len(lengths)):
            raise ValueError(f"Group {index} has mismatched completion/answer/length counts")
        flags = correctness_flags(completions, answers)
        if stats.record_candidate(flags, lengths):
            kept.append(index)
    stats.exhausted_groups = len(completion_groups) - len(kept)
    return kept, stats


def canonical_prompt(prompt: Any) -> str:
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class DynamicMixedGroupRollout:
    """TRL `rollout_func` that oversamples prompt groups until correctness varies.

    It depends on TRL 1.16's experimental rollout API and internal generation
    methods. The version is pinned in this project; API changes must fail closed.
    Unfilled groups are returned with `effective_group=False`, causing their
    reward rows to be unscorable/zero-advantage and visible in the telemetry.
    """

    def __init__(self, train_rows: Sequence[dict[str, Any]], group_size: int, *, max_candidates_per_group: int = 12, seed: int = 0):
        if group_size < 2 or max_candidates_per_group < 1:
            raise ValueError("group_size must be >= 2 and max_candidates_per_group must be positive")
        self.rows = list(train_rows)
        self.group_size = group_size
        self.max_candidates_per_group = max_candidates_per_group
        self.rng = random.Random(seed)
        self.stats = GroupFilterStats()
        self._answers = {canonical_prompt(row["prompt"]): row["answer"] for row in self.rows}

    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        if not prompts or len(prompts) % self.group_size:
            raise ValueError("TRL generation batch must contain complete groups of repeated prompts")
        if not hasattr(trainer, "_tokenize_prompts") or not hasattr(trainer, "_generate_single_turn"):
            raise RuntimeError("The pinned TRL rollout API is unavailable")

        output_prompt_ids: list[list[int]] = []
        output_completion_ids: list[list[int]] = []
        output_answers: list[Any] = []
        output_effective: list[bool] = []
        prompt_groups = [prompts[i:i + self.group_size] for i in range(0, len(prompts), self.group_size)]
        self.stats.requested_groups += len(prompt_groups)

        for original_group in prompt_groups:
            original_key = canonical_prompt(original_group[0])
            if any(canonical_prompt(prompt) != original_key for prompt in original_group):
                raise ValueError("Prompt repetitions are not contiguous within a GRPO group")
            if original_key not in self._answers:
                raise ValueError("Input prompt has no verifier target in the locked smoke/train rows")

            accepted = False
            last_ids: list[list[int]] = []
            last_prompt_ids: list[list[int]] = []
            last_answer = self._answers[original_key]
            for candidate_index in range(self.max_candidates_per_group):
                if candidate_index == 0:
                    candidate_row = {"prompt": original_group[0], "answer": last_answer}
                else:
                    candidate_row = self.rng.choice(self.rows)
                candidate_prompt = candidate_row["prompt"]
                candidate_answer = candidate_row["answer"]
                repeated_prompt = [copy.deepcopy(candidate_prompt) for _ in range(self.group_size)]
                prompt_ids, images, multimodal = trainer._tokenize_prompts(repeated_prompt)
                if images is not None or multimodal:
                    raise NotImplementedError("DynamicMixedGroupRollout currently supports text-only tasks")
                completion_ids, _logprobs = trainer._generate_single_turn(prompt_ids, images, multimodal, 1)
                if len(completion_ids) != self.group_size:
                    raise RuntimeError("TRL returned an unexpected number of completion sequences")
                decoded = trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                flags = correctness_flags(decoded, [candidate_answer] * self.group_size)
                lengths = [len(ids) for ids in completion_ids]
                last_ids, last_prompt_ids, last_answer = completion_ids, prompt_ids, candidate_answer
                mixed = self.stats.record_candidate(flags, lengths)
                if mixed:
                    accepted = True
                    break

            if not accepted:
                self.stats.exhausted_groups += 1
            output_prompt_ids.extend(last_prompt_ids)
            output_completion_ids.extend(last_ids)
            output_answers.extend([last_answer] * self.group_size)
            output_effective.extend([accepted] * self.group_size)

        return {
            "prompt_ids": output_prompt_ids,
            "completion_ids": output_completion_ids,
            "logprobs": None,
            "answer": output_answers,
            "effective_group": output_effective,
        }
