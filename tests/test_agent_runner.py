from __future__ import annotations

import json
import unittest
from collections import deque
from typing import Any, Mapping

from agent_rl import PolicyResponse, ToolEnvironment, evaluate_policy, parse_action


class QueuePolicy:
    def __init__(self, responses: list[PolicyResponse]) -> None:
        self.responses = deque(responses)

    def __call__(self, _observation: Mapping[str, Any]) -> PolicyResponse:
        return self.responses.popleft()


def response(action: object, tokens: int | None = 3, seconds: float | None = 0.01) -> PolicyResponse:
    return PolicyResponse(json.dumps(action), output_tokens=tokens, generation_seconds=seconds)


class AgentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = ToolEnvironment()

    def tearDown(self) -> None:
        self.env.close()

    def test_metrics_include_invalid_calls_and_all_interaction_cost(self) -> None:
        policy = QueuePolicy(
            [
                response({"type": "tool_call", "tool": "not_a_tool", "input": {}}, tokens=5),
                response({"type": "tool_call", "tool": "python_calculate", "input": {"expression": "101"}}, tokens=4),
                response({"type": "submit", "answer": "101"}, tokens=2),
            ]
        )
        report = evaluate_policy(self.env, policy, ["arithmetic_01"], seed=7, include_traces=True)
        self.assertEqual(report["task_success_rate"], 1.0)
        self.assertEqual(report["average_tool_calls_per_success"], 2.0)
        self.assertEqual(report["tool_call_attempts"], 2)
        self.assertEqual(report["invalid_tool_calls"], 1)
        self.assertEqual(report["invalid_tool_call_rate"], 0.5)
        self.assertEqual(report["output_tokens_total"], 11)
        self.assertEqual(report["output_token_coverage"], 1.0)
        self.assertEqual(report["average_generation_seconds"], 0.03)
        self.assertEqual(len(report["episode_records"][0]["trace"]), 3)

    def test_missing_token_counts_are_reported_as_unavailable(self) -> None:
        policy = QueuePolicy(
            [
                response({"type": "tool_call", "tool": "python_calculate", "input": {"expression": "101"}}, tokens=None),
                response({"type": "submit", "answer": "101"}, tokens=2),
            ]
        )
        report = evaluate_policy(self.env, policy, ["arithmetic_01"], seed=0)
        self.assertIsNone(report["output_tokens_total"])
        self.assertEqual(report["output_tokens_known_total"], 2)
        self.assertEqual(report["output_token_coverage"], 0.5)

    def test_invalid_json_is_an_invalid_action_not_a_fabricated_call(self) -> None:
        self.assertEqual(parse_action("not json"), {"type": "invalid_output"})
        policy = QueuePolicy(
            [
                PolicyResponse("not json", output_tokens=2, generation_seconds=0.001),
                response({"type": "submit", "answer": "101"}),
            ]
        )
        report = evaluate_policy(self.env, policy, ["arithmetic_01"])
        self.assertEqual(report["invalid_actions"], 1)
        self.assertEqual(report["tool_call_attempts"], 0)
        self.assertIsNone(report["invalid_tool_call_rate"])

    def test_failed_tasks_do_not_define_success_cost_average(self) -> None:
        policy = QueuePolicy([response({"type": "submit", "answer": "0"})])
        report = evaluate_policy(self.env, policy, ["arithmetic_01"])
        self.assertEqual(report["task_success_rate"], 0.0)
        self.assertIsNone(report["average_tool_calls_per_success"])


if __name__ == "__main__":
    unittest.main()
