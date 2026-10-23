from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from research.efficient_rlvr.group_filter import DynamicMixedGroupRollout
from research.efficient_rlvr.ledger import (
    AttemptedTokenLedger,
    DuplicateAttemptConflict,
    LedgerRunInterrupted,
    TokenBudgetExhausted,
)


ROOT = Path(__file__).resolve().parents[3]


def make_ledger(path: Path, run_id: str, *, target: int = 128, group_size: int = 2,
                cap: int = 8, candidates: int = 3) -> AttemptedTokenLedger:
    return AttemptedTokenLedger(
        path,
        run_id,
        target_tokens=target,
        group_size=group_size,
        max_completion_tokens=cap,
        max_candidates_per_group=candidates,
    )


class FakeProcessingClass:
    def batch_decode(self, sequences, **_):
        return ["".join(chr(int(token)) for token in sequence) for sequence in sequences]


class MixedFakeTrainer:
    def __init__(self):
        self.processing_class = FakeProcessingClass()
        self.calls = 0
        self.state = type("State", (), {"global_step": 0})()

    def _tokenize_prompts(self, prompts):
        return [[1] for _ in prompts], None, {}

    def _generate_single_turn(self, prompt_ids, _images, _multimodal, _count):
        self.calls += 1
        texts = ["<answer>7</answer>", "<answer>0</answer>"]
        return [[ord(char) for char in text] for text in texts[:len(prompt_ids)]], None


class FailingTrainer(MixedFakeTrainer):
    def _generate_single_turn(self, prompt_ids, images, multimodal, count):
        self.calls += 1
        raise RuntimeError("injected generation failure")


class AttemptedTokenLedgerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "ledger.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def test_records_discarded_retry_and_retained_candidate_tokens(self):
        ledger = make_ledger(self.db, "retry")
        prompt = [{"role": "user", "content": "one"}]
        retry_prompt = [{"role": "user", "content": "two"}]
        ledger.register_group("g0", source_prompt=prompt, answer="7", training_step=0, group_slot=0)
        ledger.begin_candidate("g0", 0, source_prompt=prompt, answer="7")
        ledger.record_generation("g0", 0, prompt_ids=[[1], [1]], completion_ids=[[1, 2], [3, 4, 5]])
        ledger.record_filter_decision("g0", 0, correct_flags=[False, False], will_retry=True)
        ledger.begin_candidate("g0", 1, source_prompt=retry_prompt, answer="8")
        ledger.record_generation("g0", 1, prompt_ids=[[2], [2]], completion_ids=[[1, 2, 3, 4], [5, 6, 7, 8]])
        ledger.record_filter_decision("g0", 1, correct_flags=[True, False], will_retry=False)

        summary = ledger.summary()
        self.assertEqual(summary["candidate_groups"], 2)
        self.assertEqual(summary["retry_candidates"], 1)
        self.assertEqual(summary["attempted_completion_tokens"], 13)
        self.assertEqual(summary["discarded_completion_tokens"], 5)
        self.assertEqual(summary["retained_completion_tokens"], 8)
        self.assertEqual(summary["budget_charged_tokens"], 13)
        self.assertEqual(summary["event_counts"]["retry_scheduled"], 1)
        self.assertEqual(summary["event_counts"]["candidate_discarded"], 1)
        self.assertEqual(summary["event_counts"]["candidate_retained"], 1)

    def test_hard_stopper_reserves_whole_group_and_never_overshoots_target(self):
        ledger = make_ledger(self.db, "budget", target=50, group_size=4, cap=8)
        first = [{"role": "user", "content": "a"}]
        second = [{"role": "user", "content": "b"}]
        ledger.register_group("g0", source_prompt=first, answer="1", training_step=0, group_slot=0)
        ledger.begin_candidate("g0", 0, source_prompt=first, answer="1")
        ledger.record_generation(
            "g0", 0, prompt_ids=[[1]] * 4,
            completion_ids=[[1] * n for n in (5, 6, 6, 6)],
        )
        ledger.record_filter_decision("g0", 0, correct_flags=[True, False, True, False], will_retry=False)
        ledger.register_group("g1", source_prompt=second, answer="2", training_step=1, group_slot=0)
        with self.assertRaises(TokenBudgetExhausted) as caught:
            ledger.begin_candidate("g1", 0, source_prompt=second, answer="2")

        summary = ledger.summary()
        self.assertEqual(caught.exception.remaining_tokens, 27)
        self.assertEqual(caught.exception.required_tokens, 32)
        self.assertEqual(summary["budget_charged_tokens"], 23)
        self.assertEqual(summary["budget_remaining_tokens"], 27)
        self.assertEqual(summary["budget_overrun_tokens"], 0)
        self.assertEqual(summary["group_boundary_max_budget_overshoot_tokens"], 0)
        self.assertEqual(summary["group_boundary_max_unused_tokens_at_budget_stop"], 31)
        self.assertEqual(summary["candidate_groups"], 1)
        self.assertEqual(summary["budget_stopped_groups"], 1)

    def test_process_crash_recovers_full_reservation_and_prevents_duplicate_generation(self):
        script = r'''
import os, sys
from research.efficient_rlvr.ledger import AttemptedTokenLedger
ledger = AttemptedTokenLedger(sys.argv[1], "crashed", target_tokens=64,
    group_size=2, max_completion_tokens=8, max_candidates_per_group=3)
prompt = [{"role":"user", "content":"crash"}]
ledger.register_group("g0", source_prompt=prompt, answer="1", training_step=0, group_slot=0)
ledger.begin_candidate("g0", 0, source_prompt=prompt, answer="1")
os._exit(73)
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.db)], env=env, check=False, capture_output=True
        )
        self.assertEqual(result.returncode, 73)

        ledger = make_ledger(self.db, "crashed", target=64)
        summary = ledger.summary()
        self.assertEqual(summary["status"], "interrupted")
        self.assertTrue(summary["actual_count_unknown"])
        self.assertEqual(summary["attempted_completion_tokens"], 0)
        self.assertEqual(summary["uncertain_reserved_tokens"], 16)
        self.assertEqual(summary["budget_charged_tokens"], 16)
        self.assertEqual(summary["budget_overrun_tokens"], 0)
        with self.assertRaises(LedgerRunInterrupted):
            ledger.begin_candidate(
                "g0", 0,
                source_prompt=[{"role": "user", "content": "crash"}], answer="1",
            )
        self.assertEqual(ledger.summary()["candidate_groups"], 1)

    def test_live_duplicate_process_cannot_recover_active_reservation(self):
        ledger = make_ledger(self.db, "live")
        prompt = [{"role": "user", "content": "live"}]
        ledger.register_group("g0", source_prompt=prompt, answer="1", training_step=0, group_slot=0)
        ledger.begin_candidate("g0", 0, source_prompt=prompt, answer="1")
        script = r'''
import sys
from research.efficient_rlvr.ledger import AttemptedTokenLedger
AttemptedTokenLedger(sys.argv[1], "live", target_tokens=128,
    group_size=2, max_completion_tokens=8, max_candidates_per_group=3)
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.db)], env=env, check=False, capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AttemptAlreadyRunning", result.stderr.decode("utf-8", errors="replace"))
        self.assertEqual(ledger.summary()["status"], "active")

    def test_crash_after_durable_generation_replays_output_without_regeneration(self):
        script = r'''
import os, sys
from research.efficient_rlvr.ledger import AttemptedTokenLedger
ledger = AttemptedTokenLedger(sys.argv[1], "generated", target_tokens=64,
    group_size=2, max_completion_tokens=8, max_candidates_per_group=3)
prompt = [{"role":"user", "content":"saved"}]
ledger.register_group("g0", source_prompt=prompt, answer="1", training_step=0, group_slot=0)
ledger.begin_candidate("g0", 0, source_prompt=prompt, answer="1")
ledger.record_generation("g0", 0, prompt_ids=[[1], [1]], completion_ids=[[2, 3], [4, 5, 6]])
os._exit(74)
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-c", script, str(self.db)], env=env, check=False)
        self.assertEqual(result.returncode, 74)
        ledger = make_ledger(self.db, "generated", target=64)
        prompt = [{"role": "user", "content": "saved"}]
        replay = ledger.begin_candidate("g0", 0, source_prompt=prompt, answer="1")
        self.assertEqual(replay.completion_ids, [[2, 3], [4, 5, 6]])
        ledger.record_filter_decision("g0", 0, correct_flags=[True, False], will_retry=False)
        self.assertEqual(ledger.summary()["attempted_completion_tokens"], 5)
        self.assertEqual(ledger.summary()["event_counts"]["candidate_generated"], 1)

    def test_duplicate_rollout_callback_reuses_cached_generation(self):
        rows = [{"prompt": [{"role": "user", "content": "Q"}], "answer": "7"}]
        ledger = make_ledger(self.db, "callback", group_size=2, cap=64, candidates=2)
        rollout = DynamicMixedGroupRollout(rows, 2, max_candidates_per_group=2, seed=1, ledger=ledger)
        trainer = MixedFakeTrainer()
        prompts = [rows[0]["prompt"], rows[0]["prompt"]]
        first = rollout(prompts, trainer)
        after_first = ledger.summary()
        second = rollout(prompts, trainer)
        after_second = ledger.summary()
        self.assertEqual(first["completion_ids"], second["completion_ids"])
        self.assertEqual(trainer.calls, 1)
        self.assertEqual(after_first["budget_charged_tokens"], after_second["budget_charged_tokens"])
        self.assertEqual(after_first["event_counts"], after_second["event_counts"])

    def test_exception_injection_is_durable_and_charged_conservatively(self):
        rows = [{"prompt": [{"role": "user", "content": "Q"}], "answer": "7"}]
        ledger = make_ledger(self.db, "exception", group_size=2, cap=64, candidates=2)
        rollout = DynamicMixedGroupRollout(rows, 2, max_candidates_per_group=2, seed=1, ledger=ledger)
        trainer = FailingTrainer()
        with self.assertRaisesRegex(RuntimeError, "injected generation failure"):
            rollout([rows[0]["prompt"], rows[0]["prompt"]], trainer)
        summary = ledger.summary()
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["uncertain_reserved_tokens"], 128)
        self.assertEqual(summary["budget_charged_tokens"], 128)
        self.assertEqual(summary["event_counts"]["candidate_exception"], 1)

    def test_generation_beyond_cap_is_recorded_and_stops_the_run(self):
        ledger = make_ledger(self.db, "over-cap", target=16, group_size=2, cap=8)
        prompt = [{"role": "user", "content": "cap"}]
        ledger.register_group("g0", source_prompt=prompt, answer="1", training_step=0, group_slot=0)
        ledger.begin_candidate("g0", 0, source_prompt=prompt, answer="1")
        try:
            ledger.record_generation(
                "g0", 0, prompt_ids=[[1], [1]], completion_ids=[[1] * 9, [1] * 8]
            )
        except ValueError as exc:
            ledger.record_exception("g0", 0, stage="cap_validation", error=exc)
        else:
            self.fail("generation longer than the hard completion cap was accepted")
        summary = ledger.summary()
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["budget_charged_tokens"], 17)
        self.assertEqual(summary["attempted_completion_tokens"], 17)
        self.assertEqual(summary["budget_overrun_tokens"], 1)
        self.assertFalse(summary["actual_count_unknown"])
        self.assertEqual(summary["event_counts"]["candidate_cap_violation"], 1)

    def test_concurrent_reservations_cannot_exceed_global_budget(self):
        ledger = make_ledger(self.db, "concurrent", target=16, group_size=2, cap=8)
        prompts = [[{"role": "user", "content": label}] for label in ("a", "b")]
        for index, prompt in enumerate(prompts):
            ledger.register_group(f"g{index}", source_prompt=prompt, answer="1", training_step=0, group_slot=index)
        barrier = threading.Barrier(2)
        outcomes: list[str] = []

        def reserve(index: int):
            local = make_ledger(self.db, "concurrent", target=16, group_size=2, cap=8)
            barrier.wait()
            try:
                local.begin_candidate(f"g{index}", 0, source_prompt=prompts[index], answer="1")
                outcomes.append("reserved")
            except TokenBudgetExhausted:
                outcomes.append("stopped")

        threads = [threading.Thread(target=reserve, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertCountEqual(outcomes, ["reserved", "stopped"])
        self.assertEqual(ledger.summary()["budget_charged_tokens"], 16)
        self.assertEqual(ledger.summary()["budget_overrun_tokens"], 0)

    def test_duplicate_key_with_changed_input_is_rejected(self):
        ledger = make_ledger(self.db, "conflict")
        prompt = [{"role": "user", "content": "original"}]
        ledger.register_group("g0", source_prompt=prompt, answer="1", training_step=0, group_slot=0)
        ledger.begin_candidate("g0", 0, source_prompt=prompt, answer="1")
        with self.assertRaises(DuplicateAttemptConflict):
            ledger.begin_candidate(
                "g0", 0, source_prompt=[{"role": "user", "content": "changed"}], answer="1"
            )

    def test_event_key_replay_must_preserve_its_payload(self):
        ledger = make_ledger(self.db, "event-conflict")
        with ledger._transaction() as connection:
            ledger._insert_event(
                connection, event_key="fixed-key", event_type="diagnostic", payload={"count": 1}
            )
        with ledger._transaction() as connection:
            with self.assertRaises(DuplicateAttemptConflict):
                ledger._insert_event(
                    connection, event_key="fixed-key", event_type="diagnostic", payload={"count": 2}
                )


if __name__ == "__main__":
    unittest.main()
