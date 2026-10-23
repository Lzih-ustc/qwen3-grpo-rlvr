from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from research.efficient_rlvr.scripts.prepare_locked_math_data import (
    INTEGER_ANSWER,
    flatten_modules,
    normalized_question,
)
from research.efficient_rlvr.scripts.verify_dataset_rebuilds import verify_rebuilds
from research.efficient_rlvr.scripts.audit_locked_math_data import walk_historical


class LockedDataBuilderTest(unittest.TestCase):
    def test_flatten_modules_ignores_empty_nested_regimes(self):
        def sample():
            return "q", "1"

        flattened = flatten_modules({"algebra": {}, "arithmetic": {"add": sample}})
        self.assertEqual(list(flattened), ["arithmetic/add"])
        self.assertIs(flattened["arithmetic/add"], sample)

    def test_question_normalization_collapses_case_and_whitespace(self):
        self.assertEqual(normalized_question("  Find  X\nNOW "), "find x now")

    def test_history_audit_extracts_prompt_from_array_like_parquet_value(self):
        class ArrayLike:
            def tolist(self):
                return [{"role": "user", "content": "A saved prompt"}]

        self.assertEqual(list(walk_historical({"prompt": ArrayLike()})), ["A saved prompt"])

    def test_integer_answer_filter_is_signed_base_ten_only(self):
        self.assertTrue(INTEGER_ANSWER.fullmatch("-120"))
        self.assertTrue(INTEGER_ANSWER.fullmatch("+0"))
        for non_integer in ("1/2", "2.0", "x+1", "", " 5 "):
            self.assertIsNone(INTEGER_ANSWER.fullmatch(non_integer))

    def test_rebuild_verifier_detects_different_split_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dirs = [root / name for name in ("locked", "build-a", "build-b")]
            for directory in dirs:
                directory.mkdir()
            splits = {}
            for split in ("train", "validation", "test"):
                payload = f"{split}\n".encode()
                digest = hashlib.sha256(payload).hexdigest()
                splits[split] = {"file": f"{split}.jsonl", "sha256": digest}
                for directory in dirs:
                    (directory / f"{split}.jsonl").write_bytes(payload)
            manifest = {
                "source": {"revision": "pinned"},
                "generation": {"python_hash_seed": 0},
                "splits": splits,
            }
            for directory in dirs:
                (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify_rebuilds(*dirs)["status"], "passed")
            (dirs[2] / "test.jsonl").write_text("changed\n", encoding="utf-8")
            self.assertEqual(verify_rebuilds(*dirs)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
