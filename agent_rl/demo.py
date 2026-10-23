"""Run scripted contract trajectories; this is not a model benchmark."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Mapping

from .environment import ToolEnvironment, task_ids
from .runner import PolicyResponse, evaluate_policy


class ScriptedPolicy:
    """Fixed action fixtures used only to exercise the runner contract."""

    def __init__(self) -> None:
        self._actions = {
            "arithmetic_01": [
                {"type": "tool_call", "tool": "python_calculate", "input": {"expression": "(18 * 7) - 25"}},
                {"type": "submit", "answer": "101"},
            ],
            "orders_tax_01": [
                {"type": "tool_call", "tool": "sqlite_query", "input": {"query": "SELECT SUM(amount_cents) AS cents FROM sales WHERE region = 'North' AND status = 'paid'"}},
                {"type": "tool_call", "tool": "python_calculate", "input": {"expression": "20000 * 1.1 / 100"}},
                {"type": "submit", "answer": "220.00"},
            ],
        }
        self._positions: dict[str, int] = defaultdict(int)

    def __call__(self, observation: Mapping[str, Any]) -> PolicyResponse:
        task_id = str(observation["task_id"])
        position = self._positions[task_id]
        action = self._actions[task_id][position]
        self._positions[task_id] += 1
        return PolicyResponse(text=json.dumps(action, sort_keys=True), output_tokens=None)


def main() -> None:
    env = ToolEnvironment()
    report = evaluate_policy(env, ScriptedPolicy(), task_ids(), seed=20261023, include_traces=True)
    env.close()
    report["run_kind"] = "scripted_contract_demo_not_model_performance"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
