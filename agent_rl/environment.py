"""Deterministic tool-use tasks with bounded calculator and read-only SQL tools.

This module intentionally has no model, trainer, network, or GPU dependency. It
provides a small environment surface that a future policy or GRPO data collector
can drive and that a separate verifier can score deterministically.
"""
from __future__ import annotations

import ast
import json
import operator
import re
import sqlite3
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    expected_answer: Decimal
    schema_sql: str | None = None
    setup_sql: str | None = None
    allowed_tables: frozenset[str] = frozenset()


_ORDER_SCHEMA = "CREATE TABLE sales (region TEXT NOT NULL, status TEXT NOT NULL, amount_cents INTEGER NOT NULL)"
_ORDER_FIXTURE = """
INSERT INTO sales (region, status, amount_cents) VALUES
    ('North', 'paid', 12050),
    ('North', 'paid', 7950),
    ('North', 'cancelled', 5000),
    ('South', 'paid', 20000)
"""

_TASKS = (
    TaskSpec(
        task_id="arithmetic_01",
        prompt=(
            "Use python_calculate to evaluate (18 * 7) - 25. "
            "Then submit the numerical result."
        ),
        expected_answer=Decimal("101"),
    ),
    TaskSpec(
        task_id="orders_tax_01",
        prompt=(
            "Use sqlite_query to find the total amount in cents for paid North sales. "
            "Use python_calculate to add 10% tax and convert the result to dollars. "
            "Submit the final amount. The database amount_cents column is integer cents."
        ),
        expected_answer=Decimal("220"),
        schema_sql=_ORDER_SCHEMA,
        setup_sql=_ORDER_FIXTURE,
        allowed_tables=frozenset({"sales"}),
    ),
)
_TASKS_BY_ID = {task.task_id: task for task in _TASKS}

_MAX_EXPRESSION_CHARS = 120
_MAX_SQL_CHARS = 1000
_MAX_SQL_ROWS = 100
_MAX_SQL_VM_OPS = 1_000_000
_MAX_ABS_VALUE = Decimal("1000000000000")
_MAX_EXPONENT = 12
_SAFE_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_SAFE_SQL_FUNCTIONS = frozenset(
    {"abs", "avg", "coalesce", "count", "ifnull", "length", "lower", "max", "min", "round", "substr", "sum", "upper"}
)
_FINAL_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def task_ids() -> tuple[str, ...]:
    """Return the stable, sorted identifiers of the bundled task fixtures."""
    return tuple(sorted(_TASKS_BY_ID))


def _eval_calculator_node(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only integer and decimal literals are allowed")
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_calculator_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_calculator_node(node.left)
        right = _eval_calculator_node(node.right)
        if isinstance(node.op, ast.Pow):
            if right != right.to_integral_value() or abs(right) > _MAX_EXPONENT:
                raise ValueError(f"exponent must be an integer between {-_MAX_EXPONENT} and {_MAX_EXPONENT}")
            value = left ** int(right)
        else:
            operation = _SAFE_BINARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("operator is not allowed")
            value = operation(left, right)
        if not value.is_finite() or abs(value) > _MAX_ABS_VALUE:
            raise ValueError("calculation result is outside the allowed range")
        return value
    raise ValueError("only numeric literals and basic arithmetic are allowed")


def calculate(expression: str) -> str:
    """Evaluate a short arithmetic expression without executing Python code."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression exceeds {_MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression, mode="eval")
        with localcontext() as context:
            context.prec = 28
            result = _eval_calculator_node(tree.body)
    except (SyntaxError, DecimalException, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("expression could not be evaluated") from exc
    rendered = format(result.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


class ToolEnvironment:
    """A resettable, deterministic environment with terminal numeric rewards.

    Actions are mappings of either ``{"type":"tool_call", "tool": ..., "input": ...}``
    or ``{"type":"submit", "answer": ...}``. A correct final answer earns 1.0;
    every other terminal result earns 0.0. Tool feedback never carries reward.
    """

    def __init__(self, *, max_tool_calls: int = 4, max_steps: int = 8) -> None:
        if max_tool_calls < 1 or max_steps < 1:
            raise ValueError("max_tool_calls and max_steps must be positive")
        self.max_tool_calls = max_tool_calls
        self.max_steps = max_steps
        self._task: TaskSpec | None = None
        self._seed: int | None = None
        self._connection: sqlite3.Connection | None = None
        self._messages: list[dict[str, str]] = []
        self._step_count = 0
        self._tool_calls = 0
        self._tool_call_attempts = 0
        self._invalid_tool_calls = 0
        self._tool_errors = 0
        self._invalid_actions = 0
        self._terminated = False
        self._truncated = False
        self._success = False

    def reset(self, task_id: str, *, seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a fresh fixture. The seed is recorded; bundled fixtures are fixed."""
        if task_id not in _TASKS_BY_ID:
            raise KeyError(f"unknown task_id {task_id!r}; available: {', '.join(task_ids())}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.close()
        self._task = _TASKS_BY_ID[task_id]
        self._seed = seed
        self._messages = []
        self._step_count = 0
        self._tool_calls = 0
        self._tool_call_attempts = 0
        self._invalid_tool_calls = 0
        self._tool_errors = 0
        self._invalid_actions = 0
        self._terminated = False
        self._truncated = False
        self._success = False
        if self._task.schema_sql is not None:
            self._connection = sqlite3.connect(":memory:")
            self._connection.row_factory = sqlite3.Row
            self._connection.executescript(self._task.schema_sql)
            if self._task.setup_sql:
                self._connection.executescript(self._task.setup_sql)
            vm_ops = 0

            def stop_long_query() -> int:
                nonlocal vm_ops
                vm_ops += 1000
                return int(vm_ops > _MAX_SQL_VM_OPS)

            self._connection.set_progress_handler(stop_long_query, 1000)
            self._connection.set_authorizer(self._authorize_sql)
        self._messages.append({"role": "user", "content": self._task.prompt})
        return self._observation(), self._info()

    def step(
        self, action: Mapping[str, Any] | Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._task is None:
            raise RuntimeError("call reset(task_id) before step(action)")
        if self._terminated or self._truncated:
            raise RuntimeError("episode has ended; call reset(task_id) before another step")
        self._step_count += 1
        reward = 0.0
        if not isinstance(action, Mapping):
            self._invalid_actions += 1
            self._messages.append({"role": "environment", "content": "invalid action: expected an object"})
        elif action.get("type") == "tool_call":
            self._tool_call_attempts += 1
            reward = self._handle_tool_call(action)
        elif action.get("type") == "submit":
            reward = self._handle_submit(action)
        else:
            self._invalid_actions += 1
            self._messages.append({"role": "environment", "content": "invalid action type"})

        if not self._terminated and self._step_count >= self.max_steps:
            self._truncated = True
            self._messages.append({"role": "environment", "content": "episode truncated: step limit reached"})
        return self._observation(), reward, self._terminated, self._truncated, self._info()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _handle_tool_call(self, action: Mapping[str, Any]) -> float:
        if set(action) != {"type", "tool", "input"}:
            self._invalid_actions += 1
            self._invalid_tool_calls += 1
            self._messages.append({"role": "environment", "content": "invalid tool action fields"})
            return 0.0
        tool = action.get("tool")
        tool_input = action.get("input")
        if tool not in self._available_tools():
            self._invalid_actions += 1
            self._invalid_tool_calls += 1
            self._messages.append({"role": "environment", "content": "tool is not available for this task"})
            return 0.0
        if self._tool_calls >= self.max_tool_calls:
            self._invalid_actions += 1
            self._invalid_tool_calls += 1
            self._messages.append({"role": "environment", "content": "tool-call limit reached"})
            return 0.0
        self._tool_calls += 1
        try:
            if tool == "python_calculate":
                if not isinstance(tool_input, Mapping) or set(tool_input) != {"expression"}:
                    raise ValueError("input must contain only expression")
                result: Any = calculate(tool_input["expression"])
            elif tool == "sqlite_query":
                if not isinstance(tool_input, Mapping) or set(tool_input) != {"query"}:
                    raise ValueError("input must contain only query")
                result = self._run_sql(tool_input["query"])
            else:  # Defensive: _available_tools is the source of truth.
                raise ValueError("unknown tool")
            payload = {"tool": tool, "ok": True, "result": result}
        except (ValueError, sqlite3.Error, sqlite3.Warning, DecimalException) as exc:
            self._tool_errors += 1
            self._invalid_tool_calls += 1
            payload = {"tool": tool, "ok": False, "error": str(exc)}
        self._messages.append(
            {"role": "tool", "content": json.dumps(payload, sort_keys=True, separators=(",", ":"))}
        )
        return 0.0

    def _handle_submit(self, action: Mapping[str, Any]) -> float:
        if set(action) != {"type", "answer"}:
            self._invalid_actions += 1
            self._messages.append({"role": "environment", "content": "invalid submit fields"})
            return 0.0
        answer = action.get("answer")
        correct = self._verify_answer(answer)
        self._success = correct
        self._terminated = True
        self._messages.append(
            {"role": "environment", "content": "answer accepted" if correct else "answer rejected"}
        )
        return 1.0 if correct else 0.0

    def _verify_answer(self, answer: Any) -> bool:
        if isinstance(answer, bool) or not isinstance(answer, (str, int, float, Decimal)):
            return False
        rendered = str(answer).strip()
        if len(rendered) > 64 or not _FINAL_NUMBER_RE.fullmatch(rendered):
            return False
        try:
            candidate = Decimal(rendered)
        except DecimalException:
            return False
        return candidate.is_finite() and candidate == self._task.expected_answer  # type: ignore[union-attr]

    def _available_tools(self) -> tuple[str, ...]:
        if self._task is None:
            return ()
        names = ["python_calculate"]
        if self._task.schema_sql is not None:
            names.append("sqlite_query")
        return tuple(names)

    def _run_sql(self, query: Any) -> dict[str, Any]:
        if self._connection is None:
            raise ValueError("sqlite_query is not configured for this task")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if len(query) > _MAX_SQL_CHARS:
            raise ValueError(f"query exceeds {_MAX_SQL_CHARS} characters")
        if query.lstrip().split(None, 1)[0].lower() not in {"select", "with"}:
            raise ValueError("only read-only SELECT queries are allowed")
        cursor = self._connection.execute(query)
        rows = cursor.fetchmany(_MAX_SQL_ROWS + 1)
        if len(rows) > _MAX_SQL_ROWS:
            raise ValueError(f"query returned more than {_MAX_SQL_ROWS} rows")
        columns = [description[0] for description in (cursor.description or ())]
        return {"columns": columns, "rows": [list(row) for row in rows]}

    def _authorize_sql(self, action: int, arg1: str | None, arg2: str | None, _db: str | None, _source: str | None) -> int:
        assert self._task is not None
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and arg1 in self._task.allowed_tables:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _SAFE_SQL_FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def _observation(self) -> dict[str, Any]:
        assert self._task is not None
        tool_descriptions = {
            "python_calculate": "Evaluate basic arithmetic from numeric literals; input: {expression: string}.",
        }
        if self._task.schema_sql is not None:
            tool_descriptions["sqlite_query"] = (
                "Run one read-only SELECT query; input: {query: string}. "
                "Available table schema: sales(region TEXT, status TEXT, amount_cents INTEGER)."
            )
        return {
            "task_id": self._task.task_id,
            "prompt": self._task.prompt,
            "tools": tool_descriptions,
            "messages": list(self._messages),
            "steps_remaining": max(0, self.max_steps - self._step_count),
            "tool_calls_remaining": max(0, self.max_tool_calls - self._tool_calls),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "task_id": self._task.task_id if self._task else None,
            "seed": self._seed,
            "steps": self._step_count,
            "tool_calls": self._tool_calls,
            "tool_call_attempts": self._tool_call_attempts,
            "invalid_tool_calls": self._invalid_tool_calls,
            "tool_errors": self._tool_errors,
            "invalid_actions": self._invalid_actions,
            "success": self._success if self._terminated or self._truncated else None,
            "terminated": self._terminated,
            "truncated": self._truncated,
        }
