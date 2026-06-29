"""Tests for preference-dataset prefix-sharing diagnostics."""

import unittest

# Import the real key constants from the existing (non-new) tokenization module
# to prove the diagnostic reads the same fields the preference tokenizers emit.
from open_instruct.dataset_transformation import CHOSEN_INPUT_IDS_KEY, REJECTED_INPUT_IDS_KEY
from open_instruct.prefix_sharing_stats import (
    compute_prefix_sharing_stats,
    log_prefix_sharing_stats,
    shared_prefix_length,
)


def make_row(prompt, chosen_completion, rejected_completion):
    """Build a tokenized preference row keyed like dataset_transformation output."""
    return {
        CHOSEN_INPUT_IDS_KEY: list(prompt) + list(chosen_completion),
        REJECTED_INPUT_IDS_KEY: list(prompt) + list(rejected_completion),
    }


class TestSharedPrefixLength(unittest.TestCase):
    def test_full_overlap_then_divergence(self):
        self.assertEqual(shared_prefix_length([1, 2, 3, 9], [1, 2, 3, 8]), 3)

    def test_no_overlap(self):
        self.assertEqual(shared_prefix_length([5, 6], [7, 8]), 0)

    def test_identical_sequences(self):
        self.assertEqual(shared_prefix_length([1, 2, 3], [1, 2, 3]), 3)


class TestComputePrefixSharingStats(unittest.TestCase):
    def test_stats_over_known_dataset(self):
        # Each example: 4-token shared prompt, 2-token chosen, 2-token rejected.
        dataset = [make_row([1, 2, 3, 4], [10, 11], [20, 21]) for _ in range(3)]
        stats = compute_prefix_sharing_stats(dataset)

        self.assertIsNotNone(stats)
        self.assertEqual(stats.num_examples, 3)
        self.assertEqual(stats.mean_shared_prefix_tokens, 4.0)
        self.assertEqual(stats.mean_chosen_tokens, 6.0)
        self.assertEqual(stats.mean_rejected_tokens, 6.0)
        # Naive forwards 12 tokens/example, 4 of which are redundant prefix.
        self.assertAlmostEqual(stats.redundant_token_fraction, 4.0 / 12.0)
        self.assertAlmostEqual(stats.estimated_token_speedup, 12.0 / 8.0)

    def test_no_shared_prefix_means_no_speedup(self):
        dataset = [make_row([], [1, 2], [3, 4])]
        stats = compute_prefix_sharing_stats(dataset)
        self.assertEqual(stats.redundant_token_fraction, 0.0)
        self.assertEqual(stats.estimated_token_speedup, 1.0)

    def test_max_samples_caps_scan(self):
        dataset = [make_row([1, 2, 3, 4], [10, 11], [20, 21]) for _ in range(100)]
        stats = compute_prefix_sharing_stats(dataset, max_samples=10)
        self.assertEqual(stats.num_examples, 10)

    def test_empty_dataset_returns_none(self):
        self.assertIsNone(compute_prefix_sharing_stats([]))

    def test_missing_fields_returns_none(self):
        self.assertIsNone(compute_prefix_sharing_stats([{"foo": [1, 2, 3]}]))

    def test_as_dict_namespaced_keys(self):
        dataset = [make_row([1, 2], [3], [4])]
        stats = compute_prefix_sharing_stats(dataset)
        keys = stats.as_dict()
        self.assertIn("prefix_sharing/estimated_token_speedup", keys)
        self.assertIn("prefix_sharing/redundant_token_fraction", keys)


class TestLogPrefixSharingStats(unittest.TestCase):
    def test_log_returns_stats(self):
        dataset = [make_row([1, 2, 3, 4], [10, 11], [20, 21])]
        stats = log_prefix_sharing_stats(dataset)
        self.assertIsNotNone(stats)
        self.assertEqual(stats.num_examples, 1)

    def test_log_handles_unavailable(self):
        self.assertIsNone(log_prefix_sharing_stats([]))


if __name__ == "__main__":
    unittest.main()
