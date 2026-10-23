from __future__ import annotations

import json
import unittest

from agent_rl import ToolEnvironment, task_ids


def tool(tool_name: str, **tool_input: str) -> dict[str, object]:
    return {"type": "tool_call", "tool": tool_name, "input": tool_input}


class ToolEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = ToolEnvironment()

    def tearDown(self) -> None:
        self.env.close()

    def test_task_catalog_and_fixed_seed_are_stable(self) -> None:
        self.assertEqual(task_ids(), ("arithmetic_01", "orders_tax_01"))
        first, first_info = self.env.reset("orders_tax_01", seed=41)
        second, second_info = self.env.reset("orders_tax_01", seed=41)
        self.assertEqual(first, second)
        self.assertEqual(first_info["seed"], second_info["seed"])

    def test_safe_calculator_and_strict_verifier(self) -> None:
        self.env.reset("arithmetic_01")
        _, reward, terminated, truncated, info = self.env.step(
            tool("python_calculate", expression="(18 * 7) - 25")
        )
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        self.assertEqual(info["tool_errors"], 0)
        _, reward, terminated, truncated, info = self.env.step({"type": "submit", "answer": "101.00"})
        self.assertEqual((reward, terminated, truncated), (1.0, True, False))
        self.assertTrue(info["success"])

    def test_calculator_rejects_code_and_bounds_exponents(self) -> None:
        self.env.reset("arithmetic_01")
        for expression in ("__import__('os').getcwd()", "2 ** 99", "1 / 0"):
            _, _, _, _, info = self.env.step(tool("python_calculate", expression=expression))
            self.assertEqual(info["tool_errors"], 1)
            payload = json.loads(self.env._messages[-1]["content"])
            self.assertFalse(payload["ok"])
            self.env.reset("arithmetic_01")

    def test_sql_task_requires_query_then_calculation_and_verifies(self) -> None:
        self.env.reset("orders_tax_01")
        _, _, _, _, query_info = self.env.step(
            tool("sqlite_query", query="SELECT SUM(amount_cents) AS cents FROM sales WHERE region='North' AND status='paid'")
        )
        query_result = json.loads(self.env._messages[-1]["content"])
        self.assertEqual(query_result["result"]["rows"], [[20000]])
        self.assertEqual(query_info["tool_errors"], 0)
        self.env.step(tool("python_calculate", expression="20000 * 1.1 / 100"))
        _, reward, terminated, truncated, info = self.env.step({"type": "submit", "answer": "220.00"})
        self.assertEqual((reward, terminated, truncated), (1.0, True, False))
        self.assertEqual(info["tool_calls"], 2)

    def test_sql_tool_is_read_only_and_rejects_multiple_statements(self) -> None:
        self.env = ToolEnvironment(max_tool_calls=5)
        self.env.reset("orders_tax_01")
        for query in (
            "DELETE FROM sales",
            "WITH x AS (SELECT 1) UPDATE sales SET amount_cents=0",
            "SELECT 1; SELECT 2",
            "SELECT random()",
        ):
            self.env.step(tool("sqlite_query", query=query))
            result = json.loads(self.env._messages[-1]["content"])
            self.assertFalse(result["ok"])
        self.assertEqual(self.env._tool_errors, 4)
        self.env.step(tool("sqlite_query", query="SELECT SUM(amount_cents) FROM sales WHERE region='North' AND status='paid'"))
        result = json.loads(self.env._messages[-1]["content"])
        self.assertEqual(result["result"]["rows"], [[20000]])

    def test_bad_answer_and_tool_limit_are_logged(self) -> None:
        env = ToolEnvironment(max_tool_calls=1, max_steps=5)
        try:
            env.reset("arithmetic_01")
            env.step(tool("python_calculate", expression="1+1"))
            env.step(tool("python_calculate", expression="1+2"))
            self.assertEqual(env._invalid_actions, 1)
            _, reward, terminated, truncated, info = env.step({"type": "submit", "answer": "the answer is 2"})
            self.assertEqual((reward, terminated, truncated), (0.0, True, False))
            self.assertEqual(info["tool_calls"], 1)
            self.assertEqual(info["invalid_actions"], 1)
        finally:
            env.close()

    def test_step_limit_truncates_without_reward(self) -> None:
        env = ToolEnvironment(max_steps=1)
        try:
            env.reset("arithmetic_01")
            _, reward, terminated, truncated, info = env.step(tool("python_calculate", expression="101"))
            self.assertEqual((reward, terminated, truncated), (0.0, False, True))
            self.assertEqual(info["success"], False)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
