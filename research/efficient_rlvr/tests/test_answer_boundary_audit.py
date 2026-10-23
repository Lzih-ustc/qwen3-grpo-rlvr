from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.efficient_rlvr.scripts.audit_answer_boundaries import analyze_case, marker_span, token_regions


class WordOffsetTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, object]:
        import re
        offsets = [match.span() for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


class AnswerBoundaryAuditTest(unittest.TestCase):
    def test_marker_uses_only_the_historical_unique_parser_result(self) -> None:
        self.assertEqual(marker_span("work <answer>42</answer> tail", "answer_tag"), (5, 24))
        self.assertIsNone(marker_span("<answer>1</answer> <answer>2</answer>", "answer_tag"))
        self.assertIsNone(marker_span("answer 42", None))

    def test_regions_partition_tokens_and_failed_parse_has_unknown_boundaries(self) -> None:
        tokenizer = WordOffsetTokenizer()
        result = token_regions("show work <answer>42</answer> trailing words", tokenizer, (11, 30))
        self.assertEqual(result, {
            "completion_tokens": 5,
            "before_answer_tokens": 2,
            "answer_span_tokens": 1,
            "after_answer_tokens": 2,
        })
        failed = token_regions("unmarked answer", tokenizer, None)
        self.assertEqual(failed["completion_tokens"], 2)
        self.assertIsNone(failed["after_answer_tokens"])

    def test_exact_repeated_sentence_copy_is_counted_after_first_occurrence(self) -> None:
        from research.efficient_rlvr.scripts.audit_answer_boundaries import exact_repeated_sentence_tokens
        text = "This exact reasoning sentence is sufficiently long to count. This exact reasoning sentence is sufficiently long to count. <answer>2</answer>"
        result = exact_repeated_sentence_tokens(text, WordOffsetTokenizer(), (text.index("<answer>"), len(text)))
        self.assertEqual(result["duplicate_sentence_occurrences"], 1)
        self.assertGreater(result["before_answer"], 0)
        self.assertEqual(result["after_answer"], 0)

    def test_paired_report_preserves_inputs_and_original_accuracy(self) -> None:
        shared = {
            "eval_count": 2, "eval_data_sha256": "a" * 64,
            "decode": {"do_sample": False}, "generated_tokens": 9,
            "elapsed_seconds": 1.0, "tokens_per_second": 9.0,
            "correct_count": 1,
            "predictions": [
                {"prompt": "q1", "expected": 2, "completion": "work <answer>2</answer>", "predicted": 2, "parse_source": "answer_tag", "correct": True},
                {"prompt": "q2", "expected": 3, "completion": "unparseable", "predicted": None, "parse_source": None, "correct": False},
            ],
        }
        adapter = {**shared, "correct_count": 2, "predictions": [
            shared["predictions"][0],
            {"prompt": "q2", "expected": 3, "completion": "<answer>3</answer> after", "predicted": 3, "parse_source": "answer_tag", "correct": True},
        ]}
        with tempfile.TemporaryDirectory() as temp:
            base_path, adapter_path = Path(temp) / "base.json", Path(temp) / "adapter.json"
            base_path.write_text(json.dumps(shared), encoding="utf-8")
            adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
            before = (base_path.read_bytes(), adapter_path.read_bytes())
            result = analyze_case("fixture", base_path, adapter_path, WordOffsetTokenizer())
            self.assertEqual(result["models"]["base"]["correct"], 1)
            self.assertEqual(result["models"]["base"]["unparsed"], 1)
            self.assertEqual(result["models"]["adapter"]["nonempty_post_answer_tail"], 1)
            self.assertEqual(result["paired_correctness_transitions"], {"adapter_only_correct": 1, "both_correct": 1})
            self.assertEqual((base_path.read_bytes(), adapter_path.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
