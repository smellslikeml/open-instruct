"""Entropy-adaptive token selection for RLVR via the Relative Surprisal Index (RSI).

Adapted from "Which Tokens Matter? Adaptive Token Selection for RLVR with the
Relative Surprisal Index" (https://arxiv.org/abs/2606.31575).

The Relative Surprisal Index couples a token's self-information -- the surprisal
of the *sampled* token, ``-log p(token)`` -- with the predictive entropy of that
token's distribution. Evaluating probability or entropy in isolation is
insufficient: high-entropy positions are usually (but not always) low-probability,
and both signals independently correlate with useful gradient. RSI normalises the
sampled token's surprisal by the distribution's entropy so the two are considered
jointly::

    RSI_t = surprisal_t / (entropy_t + eps) = (-log p_t) / (H_t + eps)

RSI Selection (RSI-S) keeps only tokens whose RSI falls inside a stable interval
``[lower, upper]``, dropping both redundant low-surprisal tokens (RSI below the
lower bound) and unstable high-surprisal tail tokens (RSI above the upper bound).
The result is a per-token keep-mask over the ``[B, T]`` response tokens that
composes multiplicatively with the existing ``rho`` / gradient weighting consumed
by ``compute_grpo_loss`` -- it filters *which tokens* contribute gradient rather
than replacing the importance-sampling correction.
"""

from dataclasses import dataclass, field

import torch

# wandb keys for the fraction of response tokens kept / dropped by RSI-S. These
# mirror the ``val/rho_*`` metrics and are logged through the same per-token
# ``rho_metrics`` channel in ``populate_sample_loss_stats``.
RSI_KEEP_FRAC_KEY = "val/rsi_keep_frac"
RSI_DROP_LOW_FRAC_KEY = "val/rsi_drop_low_frac"
RSI_DROP_HIGH_FRAC_KEY = "val/rsi_drop_high_frac"

# Every metric key this module can emit -- consumed by ``create_loss_stats`` so
# the slots are pre-allocated even when RSI-S is disabled (staying zero), exactly
# like the rho metrics.
RSI_METRIC_KEYS = (RSI_KEEP_FRAC_KEY, RSI_DROP_LOW_FRAC_KEY, RSI_DROP_HIGH_FRAC_KEY)


@dataclass
class RsiSelection:
    """Output of RSI-S over ``[B, T]`` response tokens.

    ``weights`` is a ``{0, 1}`` float keep-mask multiplied into the policy loss
    weighting (all-ones keeps every token). ``metrics`` maps wandb keys to
    per-token tensors reduced by ``masked_mean(., response_mask)`` at logging time.
    """

    weights: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


def relative_surprisal_index(
    logprobs: torch.Tensor, entropy: torch.Tensor, response_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Per-token RSI = surprisal / entropy over the response tokens.

    ``logprobs`` are the log-probabilities of the sampled tokens (``[B, T]``) and
    ``entropy`` the predictive entropy of each token's distribution. Non-response
    positions are zeroed so they never trip the selection bounds.
    """
    surprisal = -logprobs
    rsi = surprisal / (entropy + eps)
    return torch.where(response_mask, rsi, torch.zeros_like(rsi))


def compute_rsi_selection(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    lower_bound: float,
    upper_bound: float,
    eps: float = 1e-6,
) -> RsiSelection:
    """Build the RSI-S keep-mask and its logging metrics.

    A bound of ``0.0`` disables that side of the interval, matching the
    ``rho_mask_*_bound`` convention: with both bounds at ``0.0`` every response
    token is kept and the mask is a no-op that composes cleanly with rho.
    """
    rsi = relative_surprisal_index(logprobs, entropy, response_mask, eps=eps)
    dropped_low = (rsi < lower_bound) & response_mask if lower_bound > 0.0 else torch.zeros_like(response_mask)
    dropped_high = (rsi > upper_bound) & response_mask if upper_bound > 0.0 else torch.zeros_like(response_mask)
    keep = response_mask & ~dropped_low & ~dropped_high
    metrics = {
        RSI_KEEP_FRAC_KEY: keep.float(),
        RSI_DROP_LOW_FRAC_KEY: dropped_low.float(),
        RSI_DROP_HIGH_FRAC_KEY: dropped_high.float(),
    }
    return RsiSelection(weights=keep.float(), metrics=metrics)
