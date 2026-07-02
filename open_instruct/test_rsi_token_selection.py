"""Unit tests for RSI-S entropy-adaptive token selection and its GRPO wiring.

Covers the standalone Relative Surprisal Index math plus the integration points:
the RSI keep-mask composing into ``grpo_utils.compute_grpo_loss`` (the gradient
path) and its metrics slots being pre-allocated by ``grpo_utils.create_loss_stats``
(the logging path).
"""

import unittest
from unittest.mock import MagicMock

import torch

from open_instruct import grpo_utils, rsi_token_selection


def _make_grpo_config(**kwargs) -> grpo_utils.GRPOExperimentConfig:
    defaults = {
        "clip_lower": 0.2,
        "clip_higher": 0.2,
        "beta": 0.05,
        "kl_estimator": 2,
        "loss_fn": grpo_utils.GRPOLossType.dapo,
        "load_ref_policy": False,
    }
    defaults.update(kwargs)
    config = MagicMock(spec=grpo_utils.GRPOExperimentConfig)
    for key, value in defaults.items():
        setattr(config, key, value)
    return config


class TestRelativeSurprisalIndex(unittest.TestCase):
    def test_rsi_couples_surprisal_and_entropy(self):
        # RSI = surprisal / entropy = (-logprob) / (entropy + eps).
        logprobs = torch.tensor([[-1.0, -4.0]])
        entropy = torch.tensor([[2.0, 2.0]])
        mask = torch.ones_like(logprobs, dtype=torch.bool)
        rsi = rsi_token_selection.relative_surprisal_index(logprobs, entropy, mask)
        torch.testing.assert_close(rsi, torch.tensor([[0.5, 2.0]]), atol=1e-4, rtol=1e-4)

    def test_non_response_positions_zeroed(self):
        logprobs = torch.tensor([[-3.0, -3.0]])
        entropy = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[True, False]])
        rsi = rsi_token_selection.relative_surprisal_index(logprobs, entropy, mask)
        self.assertEqual(rsi[0, 1].item(), 0.0)
        self.assertAlmostEqual(rsi[0, 0].item(), 3.0, places=3)


class TestComputeRsiSelection(unittest.TestCase):
    def test_keeps_stable_interval(self):
        # RSI values are 0.2 (redundant), 1.0 (stable), 5.0 (unstable tail).
        logprobs = torch.tensor([[-0.2, -1.0, -5.0]])
        entropy = torch.ones_like(logprobs)
        mask = torch.ones_like(logprobs, dtype=torch.bool)
        sel = rsi_token_selection.compute_rsi_selection(logprobs, entropy, mask, 0.5, 2.0)
        torch.testing.assert_close(sel.weights, torch.tensor([[0.0, 1.0, 0.0]]))
        self.assertEqual(sel.metrics[rsi_token_selection.RSI_DROP_LOW_FRAC_KEY][0, 0].item(), 1.0)
        self.assertEqual(sel.metrics[rsi_token_selection.RSI_DROP_HIGH_FRAC_KEY][0, 2].item(), 1.0)
        self.assertEqual(sel.metrics[rsi_token_selection.RSI_KEEP_FRAC_KEY][0, 1].item(), 1.0)

    def test_disabled_bounds_keep_all(self):
        logprobs = torch.tensor([[-0.2, -1.0, -5.0]])
        entropy = torch.ones_like(logprobs)
        mask = torch.ones_like(logprobs, dtype=torch.bool)
        sel = rsi_token_selection.compute_rsi_selection(logprobs, entropy, mask, 0.0, 0.0)
        torch.testing.assert_close(sel.weights, torch.ones_like(logprobs))

    def test_padding_tokens_never_kept(self):
        logprobs = torch.tensor([[-1.0, -1.0]])
        entropy = torch.ones_like(logprobs)
        mask = torch.tensor([[True, False]])
        sel = rsi_token_selection.compute_rsi_selection(logprobs, entropy, mask, 0.5, 2.0)
        self.assertEqual(sel.weights[0, 1].item(), 0.0)
        self.assertEqual(sel.weights[0, 0].item(), 1.0)


class TestRsiComposesIntoGrpoLoss(unittest.TestCase):
    """The keep-mask must zero the per-token policy loss for filtered tokens."""

    def test_dropped_tokens_zero_policy_loss(self):
        config = _make_grpo_config()
        logprobs = torch.tensor([[-0.2, -1.0, -5.0]])
        entropy = torch.ones_like(logprobs)
        mask = torch.ones_like(logprobs, dtype=torch.bool)
        sel = rsi_token_selection.compute_rsi_selection(logprobs, entropy, mask, 0.5, 2.0)

        # Mirror the grpo_fast.py call site: RSI mask composes on top of rho weights.
        rho_weights = torch.ones_like(logprobs)
        composed = rho_weights * sel.weights
        ratio = torch.ones_like(logprobs)
        advantages = torch.ones_like(logprobs)

        pg_loss, clipfrac, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=composed,
        )
        self.assertEqual(pg_loss[0, 0].item(), 0.0)  # low-RSI redundant token dropped
        self.assertEqual(pg_loss[0, 2].item(), 0.0)  # high-RSI tail token dropped
        self.assertNotEqual(pg_loss[0, 1].item(), 0.0)  # stable token retained
        self.assertEqual(clipfrac[0, 0].item(), 0.0)
        self.assertEqual(clipfrac[0, 2].item(), 0.0)


class TestRsiMetricsWiredIntoLogging(unittest.TestCase):
    def test_keys_registered_in_scalar_loss_stats(self):
        for key in rsi_token_selection.RSI_METRIC_KEYS:
            self.assertIn(key, grpo_utils._SCALAR_LOSS_STAT_KEYS)

    def test_create_loss_stats_allocates_rsi_slots(self):
        stats = grpo_utils.create_loss_stats(num_samples=3, device=torch.device("cpu"))
        for key in rsi_token_selection.RSI_METRIC_KEYS:
            self.assertIn(key, stats)
            self.assertEqual(stats[key].shape, (3,))


if __name__ == "__main__":
    unittest.main()
