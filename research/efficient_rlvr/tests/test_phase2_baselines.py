from __future__ import annotations

import unittest

from research.efficient_rlvr.evaluation import evaluate_completion_prefixes, group_filter_cost_summary, summarize_length_curve
from research.efficient_rlvr.group_filter import DynamicMixedGroupRollout, retain_correctness_mixed_groups
from research.efficient_rlvr.rewards import correctness_gated_length_reward


class CharacterTokenizer:
    def decode(self, token_ids, **_):
        return "".join(chr(int(token)) for token in token_ids)


class FakeProcessingClass:
    def batch_decode(self, sequences, **_):
        return ["".join(chr(int(token)) for token in sequence) for sequence in sequences]


class FakeTrainer:
    def __init__(self):
        self.processing_class = FakeProcessingClass()
        self.calls = 0

    def _tokenize_prompts(self, prompts):
        ids = [[1 if prompt[0]["content"] == "A" else 2] for prompt in prompts]
        return ids, None, {}

    def _generate_single_turn(self, prompt_ids, images, multimodal, _num_generations):
        self.calls += 1
        if self.calls == 1:
            texts = ["<answer>0</answer>"] * len(prompt_ids)
        else:
            answer = 1 if prompt_ids[0][0] == 1 else 2
            texts = [f"<answer>{answer}</answer>", "unparsed"][:len(prompt_ids)]
        return [[ord(char) for char in text] for text in texts], None


class AllCorrectTrainer(FakeTrainer):
    def _generate_single_turn(self, prompt_ids, images, multimodal, _num_generations):
        texts = ["<answer>1</answer>"] * len(prompt_ids)
        return [[ord(char) for char in text] for text in texts], None


class Phase2BaselineTest(unittest.TestCase):
    def test_correctness_gated_length_reward_properties(self):
        completions = ["<answer>4</answer>", "<answer>4</answer>", "<answer>5</answer>", "no marker"]
        answers = [4, 4, 4, 4]
        ids = [[1, 2], list(range(10)), [1], [1]]
        rewards = correctness_gated_length_reward(
            completions, answers, ids, max_completion_tokens=10, length_weight=0.2
        )
        self.assertAlmostEqual(rewards[0], 0.96)
        self.assertAlmostEqual(rewards[1], 0.8)
        self.assertEqual(rewards[2:], [0.0, 0.0])
        masked = correctness_gated_length_reward(
            completions[:2], answers[:2], ids[:2], max_completion_tokens=10,
            length_weight=0.2, effective_group=[False, False]
        )
        self.assertEqual(masked, [None, None])

    def test_cglr_rejects_missing_token_ids_and_invalid_weights(self):
        with self.assertRaises(ValueError):
            correctness_gated_length_reward(["<answer>1</answer>"], [1], max_completion_tokens=10)
        with self.assertRaises(ValueError):
            correctness_gated_length_reward(["<answer>1</answer>"], [1], [[1]], max_completion_tokens=10, length_weight=1.0)

    def test_static_group_filter_retains_only_mixed_groups_and_counts_cost(self):
        completions = [
            ["<answer>1</answer>", "<answer>1</answer>"],
            ["<answer>0</answer>", "<answer>1</answer>"],
            ["unparsed", "<answer>0</answer>"],
        ]
        answers = [[1, 1], [1, 1], [1, 1]]
        lengths = [[4, 5], [6, 7], [8, 9]]
        kept, stats = retain_correctness_mixed_groups(completions, answers, lengths)
        self.assertEqual(kept, [1])
        summary = stats.as_dict()
        self.assertEqual(summary["retained_groups"], 1)
        self.assertEqual(summary["attempted_completion_tokens"], 39)
        self.assertEqual(summary["retained_completion_tokens"], 13)
        self.assertEqual(group_filter_cost_summary(summary)["discarded_completion_tokens"], 26)

    def test_dynamic_rollout_retries_and_returns_reward_aligned_targets(self):
        rows = [
            {"prompt": [{"role": "user", "content": "A"}], "answer": "1"},
            {"prompt": [{"role": "user", "content": "B"}], "answer": "2"},
        ]
        rollout = DynamicMixedGroupRollout(rows, group_size=2, max_candidates_per_group=3, seed=3)
        trainer = FakeTrainer()
        outputs = rollout([rows[0]["prompt"], rows[0]["prompt"]], trainer)
        self.assertEqual(len(outputs["completion_ids"]), 2)
        self.assertEqual(outputs["effective_group"], [True, True])
        expected_answer = "1" if outputs["prompt_ids"][0][0] == 1 else "2"
        self.assertEqual(outputs["answer"], [expected_answer, expected_answer])
        self.assertEqual(rollout.stats.candidate_groups, 2)
        self.assertEqual(rollout.stats.retained_groups, 1)

    def test_dynamic_rollout_marks_exhausted_group_ineffective(self):
        rows = [{"prompt": [{"role": "user", "content": "A"}], "answer": "1"}]
        rollout = DynamicMixedGroupRollout(rows, group_size=2, max_candidates_per_group=2, seed=1)
        outputs = rollout([rows[0]["prompt"], rows[0]["prompt"]], AllCorrectTrainer())
        self.assertEqual(outputs["effective_group"], [False, False])
        self.assertEqual(outputs["answer"], ["1", "1"])
        self.assertEqual(rollout.stats.exhausted_groups, 1)
        self.assertEqual(rollout.stats.retained_groups, 0)

    def test_length_controlled_evaluation_scores_prefixes_without_generation(self):
        text = "<answer>2</answer> tail"
        ids = [ord(char) for char in text]
        results = evaluate_completion_prefixes(ids, 2, CharacterTokenizer(), [8, 19, 32])
        self.assertEqual([row["correct"] for row in results], [False, True, True])
        self.assertEqual([row["prefix_truncated"] for row in results], [True, True, False])
        curve = summarize_length_curve([{"prefix_results": results}])
        self.assertEqual([row["accuracy"] for row in curve], [0.0, 1.0, 1.0])
        self.assertEqual(curve[-1]["actual_prefix_tokens_mean"], len(ids))


if __name__ == "__main__":
    unittest.main()
