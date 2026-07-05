"""Unit tests for the control-token probability-mass diagnostic (arXiv:2606.26027).

These exercise the wiring at the call site (``grpo_utils.forward_for_logprobs``,
``create_loss_stats``, ``populate_sample_loss_stats``, ``compute_metrics_from_loss_stats``)
plus the masking helper that averages the signal over the response. No GPU or real
model is required — a fake model returns controlled logits.
"""

import unittest
from dataclasses import dataclass

import torch

from open_instruct import control_token_monitor, grpo_utils
from open_instruct.rl_utils import masked_mean


class _FakeTokenizer:
    """Encodes a small fixed vocab so resolution can be tested deterministically."""

    def __init__(self):
        self.unk_token_id = 3
        self.pad_token_id = 0
        self._table = {
            "<tool_response>": [7],
            "</tool_response>": [8],
            "<think>": [9],
            "</think>": [10],
            "<unk_str>": [3],  # resolves to the unk id -> dropped
            "<|im_start|>": [100, 101],  # multi-subtoken -> dropped
        }

    def encode(self, text, add_special_tokens=False):
        return list(self._table.get(text, []))


class _EmptyTokenizer:
    """Every token splits into multiple subtokens -> nothing resolves."""

    unk_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [1, 2]


class _FakeModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids, attention_mask, position_ids, **kwargs):
        return _FakeModelOutput(self._logits)


class TestResolveControlTokenIds(unittest.TestCase):
    def test_filters_unknown_multisubtoken_and_dedupes(self):
        tokenizer = _FakeTokenizer()
        ids = control_token_monitor.resolve_control_token_ids(
            tokenizer, tokens=("<tool_response>", "<think>", "<|im_start|>", "<unk_str>", "<tool_response>")
        )
        self.assertEqual(ids, [7, 9])

    def test_default_tokens_empty_when_none_resolve(self):
        self.assertEqual(control_token_monitor.resolve_control_token_ids(_EmptyTokenizer()), [])


class TestControlTokenMass(unittest.TestCase):
    def test_off_by_default_returns_none(self):
        logits = torch.zeros(2, 3, 10)
        self.assertIsNone(control_token_monitor.control_token_mass(logits, []))
        self.assertIsNone(control_token_monitor.control_token_mass(logits, None))

    def test_spike_on_control_token(self):
        # Vocab of 5; control id 2. All mass on id 2 at every position -> mass ~1.0.
        logits = torch.full((1, 4, 5), -1e9)
        logits[..., 2] = 1e9
        mass = control_token_monitor.control_token_mass(logits, [2])
        self.assertEqual(mass.shape, (1, 4))
        self.assertTrue(torch.allclose(mass, torch.ones(1, 4), atol=1e-4))

    def test_mass_splits_across_control_ids(self):
        # Equal logits over 4 ids -> softmax 0.25 each; ids [0, 1] -> 0.5.
        logits = torch.zeros(1, 1, 4)
        mass = control_token_monitor.control_token_mass(logits, [0, 1])
        self.assertTrue(torch.allclose(mass, torch.tensor([[0.5]]), atol=1e-6))


class TestControlTokenMassMasking(unittest.TestCase):
    """Averaging over the response uses the existing masked_mean helper, like entropy."""

    def test_ignores_prompt_and_pad_positions(self):
        per_token = torch.tensor([[0.0, 1.0]])  # pos 0 = prompt, pos 1 = response
        response_mask = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(masked_mean(per_token, response_mask).item(), 1.0, places=6)
        # Selecting only the prompt position yields the prompt's mass, not the response's.
        prompt_mask = torch.tensor([[1.0, 0.0]])
        self.assertAlmostEqual(masked_mean(per_token, prompt_mask).item(), 0.0, places=6)


@dataclass
class _StubConfig:
    """Only the attributes populate_sample_loss_stats reads off the config."""

    load_ref_policy: bool = False
    kl_estimator: int = 2
    beta: float = 0.0


class TestControlTokenMassLoggedAsMetric(unittest.TestCase):
    def _make_stats(self, record):
        return grpo_utils.create_loss_stats(
            num_samples=2, device=torch.device("cpu"), record_control_token_mass=record
        )

    def test_key_present_only_when_recorded(self):
        self.assertIn("policy/control_token_mass_avg", self._make_stats(True))
        self.assertNotIn("policy/control_token_mass_avg", self._make_stats(False))

    def test_populate_and_compute_metric(self):
        loss_stats_B = self._make_stats(True)
        # Per-token mass with 1.0 at the single response position (index 1).
        control_mass = torch.tensor([[0.0, 1.0, 0.0]])
        response_mask = torch.tensor([[0.0, 1.0, 0.0]])
        zeros = torch.zeros(1, 3)
        grpo_utils.populate_sample_loss_stats(
            loss_stats_B=loss_stats_B,
            sample_idx=0,
            pg_loss=zeros,
            clipfrac=zeros,
            ratio=torch.ones(1, 3),
            loss=torch.tensor(0.0),
            response_mask=response_mask,
            new_logprobs=zeros,
            ref_logprobs=None,
            entropy=None,
            config=_StubConfig(),
            control_token_mass=control_mass,
        )
        metrics = grpo_utils.compute_metrics_from_loss_stats(loss_stats_B, torch.tensor([1.0, 0.0]))
        self.assertIn("policy/control_token_mass_avg", metrics)
        self.assertAlmostEqual(metrics["policy/control_token_mass_avg"], 1.0, places=6)


class TestForwardForLogprobsIntegration(unittest.TestCase):
    """Exercises the call-site wiring: forward_for_logprobs surfaces the mass."""

    def _call(self, **kwargs):
        query_responses = torch.tensor([[0, 1, 2]])
        position_ids = torch.tensor([[0, 1, 2]])
        return grpo_utils.forward_for_logprobs(
            _FakeModel(kwargs.pop("logits")),
            query_responses,
            None,
            position_ids,
            pad_token_id=0,
            temperature=1.0,
            **kwargs,
        )

    def test_default_returns_two_tuple(self):
        out = self._call(logits=torch.zeros(1, 3, 5))
        self.assertEqual(len(out), 2)  # historical behavior unchanged when monitor is off

    def test_returns_control_mass_when_enabled(self):
        logits = torch.full((1, 3, 5), -1e9)
        logits[..., 2] = 1e9  # all mass on control id 2
        _, _, control_mass = self._call(logits=logits, return_control_token_mass=True, control_token_ids=[2])
        self.assertEqual(control_mass.shape, (1, 2))  # shifted logits are (1, 2, 5)
        self.assertTrue(torch.allclose(control_mass, torch.ones(1, 2), atol=1e-4))

    def test_enabled_with_empty_ids_returns_none_mass(self):
        out = self._call(logits=torch.zeros(1, 3, 5), return_control_token_mass=True, control_token_ids=[])
        self.assertEqual(len(out), 3)
        self.assertIsNone(out[2])


if __name__ == "__main__":
    unittest.main()
