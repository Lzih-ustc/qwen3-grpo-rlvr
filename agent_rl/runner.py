"""Policy-independent episode runner and Agent benchmark metrics."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from .environment import ToolEnvironment


@dataclass(frozen=True)
class PolicyResponse:
    """One policy generation. Token counts must come from the real tokenizer."""

    text: str
    output_tokens: int | None = None
    generation_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("PolicyResponse.text must be a string")
        if self.output_tokens is not None and (
            isinstance(self.output_tokens, bool) or not isinstance(self.output_tokens, int) or self.output_tokens < 0
        ):
            raise ValueError("output_tokens must be a non-negative integer or None")
        if self.generation_seconds is not None and (
            isinstance(self.generation_seconds, bool)
            or not isinstance(self.generation_seconds, (int, float))
            or not math.isfinite(self.generation_seconds)
            or self.generation_seconds < 0
        ):
            raise ValueError("generation_seconds must be a finite, non-negative number or None")


Policy = Callable[[Mapping[str, Any]], PolicyResponse]


def parse_action(text: str) -> dict[str, Any]:
    """Parse the prototype's one-JSON-object-per-turn action protocol."""
    try:
        action = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"type": "invalid_output"}
    return action if isinstance(action, dict) else {"type": "invalid_output"}


def run_episode(
    env: ToolEnvironment,
    policy: Policy,
    task_id: str,
    *,
    seed: int = 0,
    include_trace: bool = True,
) -> dict[str, Any]:
    """Run one JSON-action policy through an environment and capture costs."""
    started = time.perf_counter()
    observation, _ = env.reset(task_id, seed=seed)
    trace: list[dict[str, Any]] = []
    output_token_values: list[int | None] = []
    generation_time_values: list[float | None] = []
    reward = 0.0
    info: dict[str, Any] = {}

    for _ in range(env.max_steps):
        response = policy(observation)
        if not isinstance(response, PolicyResponse):
            raise TypeError("policy must return PolicyResponse")
        action = parse_action(response.text)
        observation, reward, terminated, truncated, info = env.step(action)
        output_token_values.append(response.output_tokens)
        generation_time_values.append(response.generation_seconds)
        trace.append(
            {
                "response_text": response.text,
                "action": action,
                "output_tokens": response.output_tokens,
                "generation_seconds": response.generation_seconds,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "observation": observation,
            }
        )
        if terminated or truncated:
            break

    elapsed = time.perf_counter() - started
    token_values = [value for value in output_token_values if value is not None]
    generation_values = [value for value in generation_time_values if value is not None]
    token_coverage = len(token_values) / len(output_token_values) if output_token_values else 0.0
    generation_coverage = len(generation_values) / len(generation_time_values) if generation_time_values else 0.0
    result = {
        "task_id": task_id,
        "seed": seed,
        "success": bool(info.get("success", False)),
        "reward": reward,
        "steps": info.get("steps", 0),
        "tool_calls": info.get("tool_calls", 0),
        "tool_call_attempts": info.get("tool_call_attempts", 0),
        "invalid_tool_calls": info.get("invalid_tool_calls", 0),
        "invalid_tool_call_rate": _rate(
            info.get("invalid_tool_calls", 0), info.get("tool_call_attempts", 0)
        ),
        "tool_errors": info.get("tool_errors", 0),
        "invalid_actions": info.get("invalid_actions", 0),
        "output_tokens": sum(token_values) if token_coverage == 1.0 else None,
        "output_tokens_known": sum(token_values),
        "output_token_coverage": token_coverage,
        "generation_seconds": sum(generation_values) if generation_coverage == 1.0 else None,
        "generation_time_coverage": generation_coverage,
        "end_to_end_seconds": elapsed,
        "terminated": info.get("terminated", False),
        "truncated": info.get("truncated", False),
    }
    if include_trace:
        result["trace"] = trace
    return result


def evaluate_policy(
    env: ToolEnvironment,
    policy: Policy,
    task_ids: Sequence[str],
    *,
    seed: int = 0,
    include_traces: bool = False,
) -> dict[str, Any]:
    """Evaluate a policy and report task success and measured interaction costs."""
    if not task_ids:
        raise ValueError("task_ids must contain at least one task")
    episodes = [
        run_episode(env, policy, task_id, seed=seed + index, include_trace=include_traces)
        for index, task_id in enumerate(task_ids)
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[episode["task_id"]].append(episode)
    return {
        "episodes": len(episodes),
        "task_success_rate": sum(episode["success"] for episode in episodes) / len(episodes),
        "successes": sum(episode["success"] for episode in episodes),
        "average_tool_calls_per_success": _average_tool_calls_per_success(episodes),
        "tool_call_attempts": sum(episode["tool_call_attempts"] for episode in episodes),
        "invalid_tool_calls": sum(episode["invalid_tool_calls"] for episode in episodes),
        "invalid_tool_call_rate": _rate(
            sum(episode["invalid_tool_calls"] for episode in episodes),
            sum(episode["tool_call_attempts"] for episode in episodes),
        ),
        "tool_errors": sum(episode["tool_errors"] for episode in episodes),
        "invalid_actions": sum(episode["invalid_actions"] for episode in episodes),
        "output_tokens_total": _complete_sum(episodes, "output_tokens"),
        "output_tokens_known_total": sum(episode["output_tokens_known"] for episode in episodes),
        "output_token_coverage": _response_coverage(episodes, "output_token_coverage"),
        "average_generation_seconds": _mean_available(episodes, "generation_seconds"),
        "generation_time_coverage": _response_coverage(episodes, "generation_time_coverage"),
        "average_end_to_end_latency_seconds": fmean(episode["end_to_end_seconds"] for episode in episodes),
        "per_task": {task_id: _summarize(group) for task_id, group in sorted(grouped.items())},
        "episode_records": episodes,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _average_tool_calls_per_success(episodes: Sequence[Mapping[str, Any]]) -> float | None:
    successful = [episode for episode in episodes if episode["success"]]
    if not successful:
        return None
    return fmean(episode["tool_call_attempts"] for episode in successful)


def _complete_sum(episodes: Sequence[Mapping[str, Any]], key: str) -> int | None:
    values = [episode[key] for episode in episodes]
    return sum(values) if all(value is not None for value in values) else None


def _response_coverage(episodes: Sequence[Mapping[str, Any]], key: str) -> float:
    denominator = sum(episode["steps"] for episode in episodes)
    numerator = sum(episode[key] * episode["steps"] for episode in episodes)
    return numerator / denominator if denominator else 0.0


def _mean_available(episodes: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [episode[key] for episode in episodes if episode[key] is not None]
    return fmean(values) if values else None


def _summarize(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(episodes),
        "task_success_rate": sum(episode["success"] for episode in episodes) / len(episodes),
        "average_tool_calls_per_success": _average_tool_calls_per_success(episodes),
        "tool_call_attempts": sum(episode["tool_call_attempts"] for episode in episodes),
        "invalid_tool_calls": sum(episode["invalid_tool_calls"] for episode in episodes),
        "invalid_tool_call_rate": _rate(
            sum(episode["invalid_tool_calls"] for episode in episodes),
            sum(episode["tool_call_attempts"] for episode in episodes),
        ),
        "output_tokens_total": _complete_sum(episodes, "output_tokens"),
        "output_tokens_known_total": sum(episode["output_tokens_known"] for episode in episodes),
        "output_token_coverage": _response_coverage(episodes, "output_token_coverage"),
        "average_end_to_end_latency_seconds": fmean(episode["end_to_end_seconds"] for episode in episodes),
    }
