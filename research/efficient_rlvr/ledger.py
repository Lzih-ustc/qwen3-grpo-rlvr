"""Durable attempted-token accounting for correctness-filtered rollouts.

Each generation reserves one complete worst-case group before calling the
model. Returned token IDs replace that reservation atomically. If a process
dies with an unresolved reservation, recovery charges the full reservation,
marks the actual count unknown, and refuses new generation for that run.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


class LedgerError(RuntimeError):
    """Base class for durable rollout accounting failures."""


class TokenBudgetExhausted(LedgerError):
    """A complete worst-case group no longer fits in the global token budget."""

    def __init__(self, remaining_tokens: int, required_tokens: int):
        self.remaining_tokens = int(remaining_tokens)
        self.required_tokens = int(required_tokens)
        super().__init__(
            f"budget boundary reached: {self.remaining_tokens} tokens remain, "
            f"but a full group requires a reservation of {self.required_tokens}"
        )


class LedgerRunInterrupted(LedgerError):
    """An unresolved attempt closed the run to new generation."""


class DuplicateAttemptConflict(LedgerError):
    """An idempotency key was reused with different data."""


class AttemptAlreadyRunning(LedgerError):
    """A duplicate caller encountered a reservation still owned by this process."""


@dataclass(frozen=True)
class CandidateAttempt:
    run_id: str
    group_id: str
    candidate_index: int
    status: str
    source_prompt_json: str
    answer_json: str
    reserved_tokens: int
    charged_tokens: int
    actual_tokens: int | None
    token_lengths: list[int] | None
    prompt_ids: list[list[int]] | None
    completion_ids: list[list[int]] | None
    correct_flags: list[bool] | None
    filter_outcome: str | None
    will_retry: bool
    is_retry: bool

    @property
    def answer(self) -> Any:
        return json.loads(self.answer_json)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_id_lists(value: Sequence[Sequence[int]]) -> list[list[int]]:
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").tolist()
    result = []
    for sequence in value:
        if hasattr(sequence, "detach"):
            sequence = sequence.detach().to(device="cpu").tolist()
        result.append([int(token) for token in sequence])
    return result


class AttemptedTokenLedger:
    """SQLite ledger and hard attempted-token gate for one immutable run ID."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        path: str | Path,
        run_id: str,
        *,
        target_tokens: int,
        group_size: int,
        max_completion_tokens: int,
        max_candidates_per_group: int,
    ):
        if not run_id:
            raise ValueError("run_id must be stable and nonempty")
        if target_tokens < 1 or group_size < 2 or max_completion_tokens < 1:
            raise ValueError("token budget, group size, and completion cap must be positive")
        if max_candidates_per_group < 1:
            raise ValueError("max_candidates_per_group must be positive")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.target_tokens = int(target_tokens)
        self.group_size = int(group_size)
        self.max_completion_tokens = int(max_completion_tokens)
        self.max_candidates_per_group = int(max_candidates_per_group)
        self.max_group_tokens = self.group_size * self.max_completion_tokens
        self._initialize()
        if self._claim_run_owner():
            self._recover_unfinished_reservations()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    target_tokens INTEGER NOT NULL,
                    group_size INTEGER NOT NULL,
                    max_completion_tokens INTEGER NOT NULL,
                    max_candidates_per_group INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    stop_reason TEXT,
                    owner_host TEXT,
                    owner_pid INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS groups (
                    run_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    source_prompt_json TEXT NOT NULL,
                    answer_json TEXT NOT NULL,
                    training_step INTEGER NOT NULL,
                    group_slot INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    accepted_candidate_index INTEGER,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, group_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    run_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    candidate_index INTEGER NOT NULL,
                    source_prompt_json TEXT NOT NULL,
                    answer_json TEXT NOT NULL,
                    is_retry INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    reserved_tokens INTEGER NOT NULL,
                    charged_tokens INTEGER NOT NULL,
                    actual_tokens INTEGER,
                    token_lengths_json TEXT,
                    prompt_ids_json TEXT,
                    completion_ids_json TEXT,
                    correct_flags_json TEXT,
                    filter_outcome TEXT,
                    will_retry INTEGER NOT NULL DEFAULT 0,
                    error_stage TEXT,
                    error_text TEXT,
                    created_at TEXT NOT NULL,
                    generated_at TEXT,
                    filtered_at TEXT,
                    PRIMARY KEY (run_id, group_id, candidate_index),
                    FOREIGN KEY (run_id, group_id) REFERENCES groups(run_id, group_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    group_id TEXT,
                    candidate_index INTEGER,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (run_id, event_key),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS candidates_run_status
                    ON candidates(run_id, status);
                CREATE INDEX IF NOT EXISTS events_run_seq
                    ON events(run_id, event_seq);
                """
            )
            now = _now()
            existing = connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (self.run_id,)
            ).fetchone()
            expected = (
                self.SCHEMA_VERSION,
                self.target_tokens,
                self.group_size,
                self.max_completion_tokens,
                self.max_candidates_per_group,
            )
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, schema_version, target_tokens, group_size,
                        max_completion_tokens, max_candidates_per_group,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (self.run_id, *expected, now, now),
                )
                self._insert_event(
                    connection,
                    event_key="run:created",
                    event_type="run_created",
                    payload={
                        "target_tokens": self.target_tokens,
                        "group_size": self.group_size,
                        "max_completion_tokens": self.max_completion_tokens,
                        "max_candidates_per_group": self.max_candidates_per_group,
                        "max_group_reservation_tokens": self.max_group_tokens,
                    },
                )
            else:
                actual = (
                    existing["schema_version"],
                    existing["target_tokens"],
                    existing["group_size"],
                    existing["max_completion_tokens"],
                    existing["max_candidates_per_group"],
                )
                if actual != expected:
                    raise DuplicateAttemptConflict(
                        f"run {self.run_id!r} already exists with a different immutable budget/config"
                    )

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _claim_run_owner(self) -> bool:
        """Claim an active run; return true when stale-owner recovery is needed.

        A second process cannot silently recover a live reservation. Multiple
        ledger objects in one process are allowed because SQLite reservations
        and idempotency keys serialize their actual writes.
        """
        host = socket.gethostname()
        pid = os.getpid()
        recover = False
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT status, owner_host, owner_pid FROM runs WHERE run_id=?",
                (self.run_id,),
            ).fetchone()
            if row["status"] != "active":
                return False
            owner_host, owner_pid = row["owner_host"], row["owner_pid"]
            if owner_pid is not None:
                same_process = owner_host == host and int(owner_pid) == pid
                owner_alive = owner_host != host or self._pid_is_alive(int(owner_pid))
                if owner_alive and not same_process:
                    raise AttemptAlreadyRunning(
                        f"run {self.run_id!r} is owned by live process {owner_host}:{owner_pid}"
                    )
                recover = not owner_alive
            else:
                recover = True
            connection.execute(
                "UPDATE runs SET owner_host=?, owner_pid=?, updated_at=? WHERE run_id=?",
                (host, pid, _now(), self.run_id),
            )
        return recover

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        *,
        event_key: str,
        event_type: str,
        payload: dict[str, Any],
        group_id: str | None = None,
        candidate_index: int | None = None,
    ) -> None:
        payload_json = _json(payload)
        existing = connection.execute(
            "SELECT group_id, candidate_index, event_type, payload_json FROM events WHERE run_id=? AND event_key=?",
            (self.run_id, event_key),
        ).fetchone()
        if existing is not None:
            expected = (group_id, candidate_index, event_type, payload_json)
            actual = (
                existing["group_id"], existing["candidate_index"],
                existing["event_type"], existing["payload_json"],
            )
            if actual != expected:
                raise DuplicateAttemptConflict(f"event idempotency key {event_key!r} changed payload")
            return
        connection.execute(
            """
            INSERT INTO events
                (run_id, event_key, group_id, candidate_index, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id, event_key, group_id, candidate_index,
                event_type, payload_json, _now(),
            ),
        )

    def _recover_unfinished_reservations(self) -> None:
        with self._transaction() as connection:
            pending = connection.execute(
                """
                SELECT group_id, candidate_index, reserved_tokens
                FROM candidates WHERE run_id=? AND status='reserved'
                """,
                (self.run_id,),
            ).fetchall()
            if not pending:
                return
            for row in pending:
                index = int(row["candidate_index"])
                connection.execute(
                    """
                    UPDATE candidates SET status='interrupted', error_stage='recovery',
                        error_text='process ended with unresolved reservation'
                    WHERE run_id=? AND group_id=? AND candidate_index=?
                    """,
                    (self.run_id, row["group_id"], index),
                )
                connection.execute(
                    "UPDATE groups SET status='interrupted' WHERE run_id=? AND group_id=?",
                    (self.run_id, row["group_id"]),
                )
                self._insert_event(
                    connection,
                    event_key=f"{row['group_id']}/candidate-{index:04d}:interrupted",
                    event_type="attempt_interrupted",
                    group_id=row["group_id"],
                    candidate_index=index,
                    payload={
                        "reason": "process ended before generated IDs were durably committed",
                        "actual_tokens": None,
                        "conservative_budget_charge": int(row["reserved_tokens"]),
                        "actual_count_unknown": True,
                    },
                )
            connection.execute(
                """
                UPDATE runs SET status='interrupted',
                    stop_reason='recovered unresolved candidate reservation',
                    updated_at=? WHERE run_id=?
                """,
                (_now(), self.run_id),
            )
            self._insert_event(
                connection,
                event_key="run:interrupted",
                event_type="run_interrupted",
                payload={
                    "unresolved_attempts": len(pending),
                    "policy": "retain full reservations and prohibit new generation",
                },
            )

    def _row_to_attempt(self, row: sqlite3.Row) -> CandidateAttempt:
        def decoded(column: str):
            value = row[column]
            return json.loads(value) if value is not None else None

        return CandidateAttempt(
            run_id=row["run_id"],
            group_id=row["group_id"],
            candidate_index=int(row["candidate_index"]),
            status=row["status"],
            source_prompt_json=row["source_prompt_json"],
            answer_json=row["answer_json"],
            reserved_tokens=int(row["reserved_tokens"]),
            charged_tokens=int(row["charged_tokens"]),
            actual_tokens=int(row["actual_tokens"]) if row["actual_tokens"] is not None else None,
            token_lengths=decoded("token_lengths_json"),
            prompt_ids=decoded("prompt_ids_json"),
            completion_ids=decoded("completion_ids_json"),
            correct_flags=decoded("correct_flags_json"),
            filter_outcome=row["filter_outcome"],
            will_retry=bool(row["will_retry"]),
            is_retry=bool(row["is_retry"]),
        )

    def register_group(
        self,
        group_id: str,
        *,
        source_prompt: Any,
        answer: Any,
        training_step: int,
        group_slot: int,
    ) -> None:
        prompt_json, answer_json = _json(source_prompt), _json(answer)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM groups WHERE run_id=? AND group_id=?",
                (self.run_id, group_id),
            ).fetchone()
            if row is not None:
                if (
                    row["source_prompt_json"] != prompt_json
                    or row["answer_json"] != answer_json
                    or row["training_step"] != int(training_step)
                    or row["group_slot"] != int(group_slot)
                ):
                    raise DuplicateAttemptConflict(f"group {group_id!r} changed inputs on replay")
                return
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id=?", (self.run_id,)
            ).fetchone()
            if run["status"] != "active":
                if run["status"] == "budget_exhausted":
                    raise TokenBudgetExhausted(self.remaining_tokens, self.max_group_tokens)
                raise LedgerRunInterrupted(f"run is {run['status']}; no new rollout group may start")
            connection.execute(
                """
                INSERT INTO groups (
                    run_id, group_id, source_prompt_json, answer_json,
                    training_step, group_slot, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'requested', ?)
                """,
                (
                    self.run_id, group_id, prompt_json, answer_json,
                    int(training_step), int(group_slot), _now(),
                ),
            )
            self._insert_event(
                connection,
                event_key=f"{group_id}:requested",
                event_type="group_requested",
                group_id=group_id,
                payload={"training_step": int(training_step), "group_slot": int(group_slot)},
            )

    def begin_candidate(
        self,
        group_id: str,
        candidate_index: int,
        *,
        source_prompt: Any,
        answer: Any,
    ) -> CandidateAttempt:
        index = int(candidate_index)
        if index < 0 or index >= self.max_candidates_per_group:
            raise ValueError("candidate_index is outside the registered candidate limit")
        prompt_json, answer_json = _json(source_prompt), _json(answer)
        required = self.max_group_tokens
        exhausted: tuple[int, int] | None = None
        result: CandidateAttempt | None = None
        with self._transaction() as connection:
            group = connection.execute(
                "SELECT * FROM groups WHERE run_id=? AND group_id=?",
                (self.run_id, group_id),
            ).fetchone()
            if group is None:
                raise LedgerError(f"group {group_id!r} must be registered before candidate generation")
            row = connection.execute(
                """
                SELECT * FROM candidates
                WHERE run_id=? AND group_id=? AND candidate_index=?
                """,
                (self.run_id, group_id, index),
            ).fetchone()
            is_retry = index > 0
            if row is not None:
                if (
                    row["source_prompt_json"] != prompt_json
                    or row["answer_json"] != answer_json
                    or bool(row["is_retry"]) != is_retry
                ):
                    raise DuplicateAttemptConflict(f"candidate {group_id}/{index} changed inputs on replay")
                if row["status"] in {"generated", "filtered"}:
                    return self._row_to_attempt(row)
                if row["status"] == "reserved":
                    raise AttemptAlreadyRunning(f"candidate {group_id}/{index} already has a live reservation")
                raise LedgerRunInterrupted(f"candidate {group_id}/{index} is {row['status']}; it will not rerun")
            if group["status"] in {"retained", "exhausted", "budget_stopped", "interrupted"}:
                raise LedgerRunInterrupted(f"group {group_id!r} is already terminal: {group['status']}")
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id=?", (self.run_id,)
            ).fetchone()
            if run["status"] != "active":
                if run["status"] == "budget_exhausted":
                    raise TokenBudgetExhausted(self.remaining_tokens, required)
                raise LedgerRunInterrupted(f"run is {run['status']}; no candidate may start")
            if index > 0:
                previous = connection.execute(
                    """
                    SELECT status, filter_outcome, will_retry FROM candidates
                    WHERE run_id=? AND group_id=? AND candidate_index=?
                    """,
                    (self.run_id, group_id, index - 1),
                ).fetchone()
                if (
                    previous is None
                    or previous["status"] != "filtered"
                    or previous["filter_outcome"] == "mixed"
                    or not previous["will_retry"]
                ):
                    raise LedgerError("a retry requires a durably discarded predecessor")
            charged = int(connection.execute(
                "SELECT COALESCE(SUM(charged_tokens), 0) FROM candidates WHERE run_id=?",
                (self.run_id,),
            ).fetchone()[0])
            remaining = self.target_tokens - charged
            if remaining < required:
                connection.execute(
                    "UPDATE groups SET status='budget_stopped' WHERE run_id=? AND group_id=?",
                    (self.run_id, group_id),
                )
                connection.execute(
                    """
                    UPDATE runs SET status='budget_exhausted',
                        stop_reason='next full group reservation does not fit',
                        updated_at=? WHERE run_id=?
                    """,
                    (_now(), self.run_id),
                )
                self._insert_event(
                    connection,
                    event_key=f"{group_id}:budget_stopped",
                    event_type="group_budget_stopped",
                    group_id=group_id,
                    payload={
                        "remaining_tokens": remaining,
                        "required_group_reservation": required,
                        "actual_budget_overshoot": 0,
                        "group_boundary_unused_budget_bound": required - 1,
                    },
                )
                self._insert_event(
                    connection,
                    event_key="run:budget_exhausted",
                    event_type="run_budget_exhausted",
                    group_id=group_id,
                    payload={
                        "accounted_tokens": charged,
                        "remaining_tokens": remaining,
                        "required_group_reservation": required,
                        "target_tokens": self.target_tokens,
                    },
                )
                exhausted = (remaining, required)
            else:
                now = _now()
                connection.execute(
                    """
                    INSERT INTO candidates (
                        run_id, group_id, candidate_index, source_prompt_json,
                        answer_json, is_retry, status, reserved_tokens, charged_tokens, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?)
                    """,
                    (
                        self.run_id, group_id, index, prompt_json, answer_json,
                        int(is_retry), required, required, now,
                    ),
                )
                connection.execute(
                    "UPDATE groups SET status='sampling' WHERE run_id=? AND group_id=?",
                    (self.run_id, group_id),
                )
                self._insert_event(
                    connection,
                    event_key=f"{group_id}/candidate-{index:04d}:reserved",
                    event_type="candidate_reserved",
                    group_id=group_id,
                    candidate_index=index,
                    payload={
                        "source_prompt": json.loads(prompt_json),
                        "is_retry": is_retry,
                        "reserved_tokens": required,
                    },
                )
                row = connection.execute(
                    """
                    SELECT * FROM candidates
                    WHERE run_id=? AND group_id=? AND candidate_index=?
                    """,
                    (self.run_id, group_id, index),
                ).fetchone()
                result = self._row_to_attempt(row)
        if exhausted is not None:
            raise TokenBudgetExhausted(*exhausted)
        assert result is not None
        return result

    def record_generation(
        self,
        group_id: str,
        candidate_index: int,
        *,
        prompt_ids: Sequence[Sequence[int]],
        completion_ids: Sequence[Sequence[int]],
    ) -> CandidateAttempt:
        pids = _token_id_lists(prompt_ids)
        cids = _token_id_lists(completion_ids)
        if len(pids) != self.group_size or len(cids) != self.group_size:
            raise ValueError("generation must return exactly one complete group")
        lengths = [len(sequence) for sequence in cids]
        if any(length < 1 for length in lengths):
            raise ValueError("completion IDs must be nonempty")
        actual = sum(lengths)
        cap_violation = any(length > self.max_completion_tokens for length in lengths)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM candidates
                WHERE run_id=? AND group_id=? AND candidate_index=?
                """,
                (self.run_id, group_id, int(candidate_index)),
            ).fetchone()
            if row is None:
                raise LedgerError("reserve the candidate before generation")
            if row["status"] in {"generated", "filtered"}:
                existing = self._row_to_attempt(row)
                if existing.prompt_ids != pids or existing.completion_ids != cids:
                    raise DuplicateAttemptConflict("saved generation output differs on replay")
                return existing
            if row["status"] != "reserved":
                raise LedgerRunInterrupted(f"cannot attach output to state {row['status']}")
            connection.execute(
                """
                UPDATE candidates SET status='generated', actual_tokens=?, charged_tokens=?,
                    token_lengths_json=?, prompt_ids_json=?, completion_ids_json=?, generated_at=?
                WHERE run_id=? AND group_id=? AND candidate_index=?
                """,
                (
                    actual, actual, _json(lengths), _json(pids), _json(cids), _now(),
                    self.run_id, group_id, int(candidate_index),
                ),
            )
            self._insert_event(
                connection,
                event_key=f"{group_id}/candidate-{int(candidate_index):04d}:generated",
                event_type="candidate_generated",
                group_id=group_id,
                candidate_index=int(candidate_index),
                payload={
                    "token_lengths": lengths,
                    "actual_tokens": actual,
                    "reserved_tokens": int(row["reserved_tokens"]),
                    "released_reservation_tokens": int(row["reserved_tokens"]) - actual,
                    "completion_cap_violation": cap_violation,
                    "completion_ids_sha256": hashlib.sha256(_json(cids).encode("utf-8")).hexdigest(),
                },
            )
            if cap_violation:
                connection.execute(
                    "UPDATE groups SET status='interrupted' WHERE run_id=? AND group_id=?",
                    (self.run_id, group_id),
                )
                connection.execute(
                    "UPDATE runs SET status='interrupted', stop_reason='generation exceeded completion cap', updated_at=? WHERE run_id=?",
                    (_now(), self.run_id),
                )
                self._insert_event(
                    connection,
                    event_key=f"{group_id}/candidate-{int(candidate_index):04d}:cap-violation",
                    event_type="candidate_cap_violation",
                    group_id=group_id,
                    candidate_index=int(candidate_index),
                    payload={
                        "token_lengths": lengths,
                        "cap": self.max_completion_tokens,
                        "actual_tokens": actual,
                        "budget_overrun_tokens": max(0, actual - self.target_tokens),
                    },
                )
            updated = connection.execute(
                """
                SELECT * FROM candidates
                WHERE run_id=? AND group_id=? AND candidate_index=?
                """,
                (self.run_id, group_id, int(candidate_index)),
            ).fetchone()
            attempt = self._row_to_attempt(updated)
        if cap_violation:
            raise ValueError(
                f"generation returned a sequence longer than the {self.max_completion_tokens}-token cap; "
                "actual IDs were retained in the ledger and the run was stopped"
            )
        return attempt

    def record_filter_decision(
        self,
        group_id: str,
        candidate_index: int,
        *,
        correct_flags: Sequence[bool],
        will_retry: bool,
    ) -> CandidateAttempt:
        flags = [bool(flag) for flag in correct_flags]
        if len(flags) != self.group_size:
            raise ValueError("filter decision must have one flag per completion")
        count = sum(flags)
        outcome = "mixed" if 0 < count < self.group_size else "all_correct" if count else "all_incorrect"
        index = int(candidate_index)
        expected_retry = outcome != "mixed" and index + 1 < self.max_candidates_per_group
        if bool(will_retry) != expected_retry:
            raise ValueError("will_retry does not match the fixed candidate limit and outcome")
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM candidates
                WHERE run_id=? AND group_id=? AND candidate_index=?
                """,
                (self.run_id, group_id, index),
            ).fetchone()
            if row is None:
                raise LedgerError("generation must be durable before filtering")
            if row["status"] == "filtered":
                existing = self._row_to_attempt(row)
                if existing.correct_flags != flags or existing.will_retry != bool(will_retry):
                    raise DuplicateAttemptConflict("filter decision differs on replay")
                return existing
            if row["status"] != "generated":
                raise LedgerRunInterrupted(f"cannot filter state {row['status']}")
            connection.execute(
                """
                UPDATE candidates SET status='filtered', correct_flags_json=?,
                    filter_outcome=?, will_retry=?, filtered_at=?
                WHERE run_id=? AND group_id=? AND candidate_index=?
                """,
                (
                    _json(flags), outcome, int(bool(will_retry)), _now(),
                    self.run_id, group_id, index,
                ),
            )
            if outcome == "mixed":
                connection.execute(
                    """
                    UPDATE groups SET status='retained', accepted_candidate_index=?
                    WHERE run_id=? AND group_id=?
                    """,
                    (index, self.run_id, group_id),
                )
            elif will_retry:
                connection.execute(
                    "UPDATE groups SET status='retry_pending' WHERE run_id=? AND group_id=?",
                    (self.run_id, group_id),
                )
            else:
                connection.execute(
                    "UPDATE groups SET status='exhausted' WHERE run_id=? AND group_id=?",
                    (self.run_id, group_id),
                )
            event_key = f"{group_id}/candidate-{index:04d}"
            self._insert_event(
                connection,
                event_key=f"{event_key}:filtered",
                event_type="candidate_filtered",
                group_id=group_id,
                candidate_index=index,
                payload={
                    "outcome": outcome,
                    "correct_flags": flags,
                    "retained": outcome == "mixed",
                    "will_retry": bool(will_retry),
                },
            )
            if outcome == "mixed":
                self._insert_event(
                    connection,
                    event_key=f"{event_key}:retained",
                    event_type="candidate_retained",
                    group_id=group_id,
                    candidate_index=index,
                    payload={"actual_tokens": int(row["actual_tokens"])},
                )
            else:
                self._insert_event(
                    connection,
                    event_key=f"{event_key}:discarded",
                    event_type="candidate_discarded",
                    group_id=group_id,
                    candidate_index=index,
                    payload={
                        "outcome": outcome,
                        "actual_tokens": int(row["actual_tokens"]),
                        "will_retry": bool(will_retry),
                    },
                )
                if will_retry:
                    self._insert_event(
                        connection,
                        event_key=f"{event_key}:retry",
                        event_type="retry_scheduled",
                        group_id=group_id,
                        candidate_index=index,
                        payload={"next_candidate_index": index + 1},
                    )
                else:
                    self._insert_event(
                        connection,
                        event_key=f"{group_id}:exhausted",
                        event_type="group_exhausted",
                        group_id=group_id,
                        payload={"candidate_limit": self.max_candidates_per_group},
                    )
            updated = connection.execute(
                """
                SELECT * FROM candidates
                WHERE run_id=? AND group_id=? AND candidate_index=?
                """,
                (self.run_id, group_id, index),
            ).fetchone()
            return self._row_to_attempt(updated)

    def record_exception(
        self,
        group_id: str,
        candidate_index: int,
        *,
        stage: str,
        error: BaseException,
    ) -> None:
        index = int(candidate_index)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM candidates
                WHERE run_id=? AND group_id=? AND candidate_index=?
                """,
                (self.run_id, group_id, index),
            ).fetchone()
            if row is None:
                return
            if row["status"] == "interrupted":
                return
            unknown = row["actual_tokens"] is None
            connection.execute(
                """
                UPDATE candidates SET status='interrupted', error_stage=?, error_text=?
                WHERE run_id=? AND group_id=? AND candidate_index=?
                """,
                (
                    str(stage), f"{type(error).__name__}: {error}"[:1000],
                    self.run_id, group_id, index,
                ),
            )
            connection.execute(
                "UPDATE groups SET status='interrupted' WHERE run_id=? AND group_id=?",
                (self.run_id, group_id),
            )
            connection.execute(
                """
                UPDATE runs SET status='interrupted', stop_reason=?, updated_at=? WHERE run_id=?
                """,
                (f"exception during {stage}", _now(), self.run_id),
            )
            base = f"{group_id}/candidate-{index:04d}"
            self._insert_event(
                connection,
                event_key=f"{base}:exception",
                event_type="candidate_exception",
                group_id=group_id,
                candidate_index=index,
                payload={
                    "stage": str(stage),
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                    "actual_count_unknown": unknown,
                    "conservative_budget_charge": int(row["charged_tokens"]),
                },
            )
            self._insert_event(
                connection,
                event_key="run:interrupted",
                event_type="run_interrupted",
                group_id=group_id,
                payload={"reason": f"exception during {stage}", "candidate_index": index},
            )

    def record_run_exception(self, *, stage: str, error: BaseException) -> None:
        """Durably stop a run when failure occurs outside candidate generation."""
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id=?", (self.run_id,)
            ).fetchone()
            if run["status"] != "active":
                return
            reason = f"exception during {stage}: {type(error).__name__}"
            connection.execute(
                "UPDATE runs SET status='interrupted', stop_reason=?, updated_at=? WHERE run_id=?",
                (reason, _now(), self.run_id),
            )
            self._insert_event(
                connection,
                event_key="run:interrupted",
                event_type="run_exception",
                payload={"stage": str(stage), "error_type": type(error).__name__, "error": str(error)[:1000]},
            )

    @property
    def remaining_tokens(self) -> int:
        with closing(self._connect()) as connection:
            charged = int(connection.execute(
                "SELECT COALESCE(SUM(charged_tokens), 0) FROM candidates WHERE run_id=?",
                (self.run_id,),
            ).fetchone()[0])
        return self.target_tokens - charged

    def event_log(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY event_seq",
                (self.run_id,),
            ).fetchall()
        return [
            {
                "event_seq": int(row["event_seq"]),
                "event_key": row["event_key"],
                "group_id": row["group_id"],
                "candidate_index": row["candidate_index"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def summary(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (self.run_id,)
            ).fetchone()
            candidates = connection.execute(
                "SELECT * FROM candidates WHERE run_id=? ORDER BY group_id, candidate_index",
                (self.run_id,),
            ).fetchall()
            groups = connection.execute(
                "SELECT status FROM groups WHERE run_id=?", (self.run_id,)
            ).fetchall()
            event_rows = connection.execute(
                "SELECT event_type, COUNT(*) AS count FROM events WHERE run_id=? GROUP BY event_type",
                (self.run_id,),
            ).fetchall()
        known_attempted = sum(int(row["actual_tokens"] or 0) for row in candidates)
        max_observed = max(
            (max(json.loads(row["token_lengths_json"])) for row in candidates if row["token_lengths_json"]),
            default=0,
        )
        charged = sum(int(row["charged_tokens"]) for row in candidates)
        unknown_reserved = sum(
            int(row["charged_tokens"]) for row in candidates if row["actual_tokens"] is None
        )
        retained = [row for row in candidates if row["filter_outcome"] == "mixed"]
        discarded = [
            row for row in candidates
            if row["filter_outcome"] in {"all_correct", "all_incorrect"}
        ]
        outcomes = {"all_correct": 0, "all_incorrect": 0, "mixed": 0}
        for row in candidates:
            if row["filter_outcome"] in outcomes:
                outcomes[row["filter_outcome"]] += 1
        statuses = [row["status"] for row in groups]
        return {
            "run_id": self.run_id,
            "status": run["status"],
            "stop_reason": run["stop_reason"],
            "target_tokens": int(run["target_tokens"]),
            "group_size": int(run["group_size"]),
            "max_completion_tokens": int(run["max_completion_tokens"]),
            "max_group_reservation_tokens": self.max_group_tokens,
            "group_boundary_max_budget_overshoot_tokens": 0,
            "requested_groups": len(groups),
            "candidate_groups": len(candidates),
            "retry_candidates": sum(int(row["is_retry"]) for row in candidates),
            "retained_groups": sum(status == "retained" for status in statuses),
            "exhausted_groups": sum(status == "exhausted" for status in statuses),
            "budget_stopped_groups": sum(status == "budget_stopped" for status in statuses),
            "interrupted_groups": sum(status == "interrupted" for status in statuses),
            "attempted_completions": sum(
                self.group_size for row in candidates if row["actual_tokens"] is not None
            ),
            "unknown_reserved_completions": sum(
                self.group_size for row in candidates if row["actual_tokens"] is None
            ),
            "attempted_completion_tokens": known_attempted,
            "max_observed_completion_tokens": max_observed,
            "retained_completions": len(retained) * self.group_size,
            "retained_completion_tokens": sum(int(row["actual_tokens"] or 0) for row in retained),
            "discarded_completion_tokens": sum(int(row["actual_tokens"] or 0) for row in discarded),
            "uncertain_reserved_tokens": unknown_reserved,
            "budget_charged_tokens": charged,
            "budget_remaining_tokens": int(run["target_tokens"]) - charged,
            "budget_overrun_tokens": max(0, charged - int(run["target_tokens"])),
            "group_boundary_max_unused_tokens_at_budget_stop": self.max_group_tokens - 1,
            "actual_count_unknown": unknown_reserved > 0,
            "attempted_group_outcomes": outcomes,
            "event_counts": {
                row["event_type"]: int(row["count"]) for row in event_rows
            },
        }

    def mark_budget_stop(self, reason: str = "next full group reservation does not fit") -> None:
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id=?", (self.run_id,)
            ).fetchone()
            if run["status"] != "active":
                return
            charged = int(connection.execute(
                "SELECT COALESCE(SUM(charged_tokens), 0) FROM candidates WHERE run_id=?",
                (self.run_id,),
            ).fetchone()[0])
            connection.execute(
                "UPDATE runs SET status='budget_exhausted', stop_reason=?, updated_at=? WHERE run_id=?",
                (str(reason), _now(), self.run_id),
            )
            self._insert_event(
                connection,
                event_key="run:budget_exhausted",
                event_type="run_budget_exhausted",
                payload={
                    "reason": str(reason),
                    "accounted_tokens": charged,
                    "remaining_tokens": self.target_tokens - charged,
                },
            )

    def mark_complete(self, reason: str = "trainer reached its registered step limit") -> None:
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id=?", (self.run_id,)
            ).fetchone()
            if run["status"] != "active":
                return
            connection.execute(
                "UPDATE runs SET status='complete', stop_reason=?, updated_at=? WHERE run_id=?",
                (str(reason), _now(), self.run_id),
            )
            self._insert_event(
                connection,
                event_key="run:complete",
                event_type="run_complete",
                payload={"reason": str(reason)},
            )
