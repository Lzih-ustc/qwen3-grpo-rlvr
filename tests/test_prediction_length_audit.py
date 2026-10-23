from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_prediction_lengths import analyze_case


class PredictionLengthAuditTest(unittest.TestCase):
    def test_paired_outcomes_and_source_files_remain_unchanged(self) -> None:
        base_payload = {
            "tag": "base",
            "eval_count": 3,
            "correct_count": 1,
            "accuracy": 1 / 3,
            "parse_rate": 2 / 3,
            "eval_data_sha256": "f" * 64,
            "decode": {"do_sample": False, "max_new_tokens": 192, "thinking": False},
            "generated_tokens": 40,
            "elapsed_seconds": 2.0,
            "tokens_per_second": 20.0,
            "predictions": [
                {"prompt": "q1", "expected": 1, "completion": "<answer>1</answer>", "predicted": 1, "parse_source": "answer_tag", "correct": True},
                {"prompt": "q2", "expected": 2, "completion": "I think the answer is 5", "predicted": None, "parse_source": None, "correct": False},
                {"prompt": "q3", "expected": 3, "completion": "<answer>8</answer>", "predicted": 8, "parse_source": "answer_tag", "correct": False},
            ],
        }
        adapter_payload = {
            **base_payload,
            "tag": "adapter",
            "correct_count": 2,
            "accuracy": 2 / 3,
            "parse_rate": 1.0,
            "generated_tokens": 75,
            "elapsed_seconds": 4.0,
            "tokens_per_second": 18.75,
            "predictions": [
                {"prompt": "q1", "expected": 1, "completion": "show work\n<answer>1</answer>", "predicted": 1, "parse_source": "answer_tag", "correct": True},
                {"prompt": "q2", "expected": 2, "completion": "<answer>2</answer>", "predicted": 2, "parse_source": "answer_tag", "correct": True},
                {"prompt": "q3", "expected": 3, "completion": "<answer>8</answer>", "predicted": 8, "parse_source": "answer_tag", "correct": False},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            base_path = Path(temp) / "base.json"
            adapter_path = Path(temp) / "adapter.json"
            base_path.write_text(json.dumps(base_payload), encoding="utf-8")
            adapter_path.write_text(json.dumps(adapter_payload), encoding="utf-8")
            base_before = base_path.read_bytes()
            adapter_before = adapter_path.read_bytes()

            result = analyze_case("fixture", base_path, adapter_path)

            self.assertEqual(
                result["paired_correctness_transitions"],
                {"adapter_only_correct": 1, "both_correct": 1, "both_wrong": 1},
            )
            self.assertEqual(result["base"]["unparsed_count"], 1)
            self.assertEqual(result["adapter"]["unparsed_count"], 0)
            self.assertEqual(result["adapter"]["lengths_by_correctness"]["correct"]["completion_characters"]["n"], 2)
            self.assertEqual(base_path.read_bytes(), base_before)
            self.assertEqual(adapter_path.read_bytes(), adapter_before)


if __name__ == "__main__":
    unittest.main()
