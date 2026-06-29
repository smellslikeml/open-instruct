"""Diagnostics for shared-prefix redundancy in preference (DPO) datasets.

In standard paired preference tuning, the chosen and rejected sequences are
forwarded independently even though they share a common prompt prefix. That
prefix is therefore encoded twice, which is pure redundant computation. The
``Accelerating Direct Preference Optimization with Prefix Sharing`` paper
(https://arxiv.org/abs/2410.20305) shows that forwarding the pair as a single
prefix-shared sequence removes this redundancy and yields 1.1-1.5x training
throughput, scaling with how much of each example is shared prefix.

This module does not change the forward pass (that requires a custom
block-sparse attention mask). It surfaces the *result* the paper motivates:
how much of a concrete preference dataset is redundant prompt prefix, and what
token-level speedup prefix sharing could buy on that data. That lets a user
decide whether the prefix-sharing / packing path is worth enabling before
spending compute.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from open_instruct import logger_utils
from open_instruct.dataset_transformation import CHOSEN_INPUT_IDS_KEY, REJECTED_INPUT_IDS_KEY

logger = logger_utils.setup_logger(__name__)

# Cap on how many examples we scan; a sample is enough for a stable estimate and
# keeps startup cheap on large datasets.
DEFAULT_MAX_SAMPLES = 2048


@dataclass
class PrefixSharingStats:
    """Aggregate shared-prefix statistics over a preference dataset sample."""

    num_examples: int
    mean_chosen_tokens: float
    mean_rejected_tokens: float
    mean_shared_prefix_tokens: float
    # Fraction of all forwarded tokens that are redundant shared prefix under
    # the naive (independent) encoding.
    redundant_token_fraction: float
    # Estimated throughput multiplier from removing the redundant prefix
    # encoding (naive_tokens / prefix_shared_tokens). 1.0 means no savings.
    estimated_token_speedup: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "prefix_sharing/num_examples": self.num_examples,
            "prefix_sharing/mean_chosen_tokens": self.mean_chosen_tokens,
            "prefix_sharing/mean_rejected_tokens": self.mean_rejected_tokens,
            "prefix_sharing/mean_shared_prefix_tokens": self.mean_shared_prefix_tokens,
            "prefix_sharing/redundant_token_fraction": self.redundant_token_fraction,
            "prefix_sharing/estimated_token_speedup": self.estimated_token_speedup,
        }


def shared_prefix_length(chosen_ids: Sequence[int], rejected_ids: Sequence[int]) -> int:
    """Return the number of leading token ids shared by both sequences."""
    count = 0
    for chosen_token, rejected_token in zip(chosen_ids, rejected_ids):
        if chosen_token != rejected_token:
            break
        count += 1
    return count


def compute_prefix_sharing_stats(
    dataset: Sequence[dict[str, Any]],
    chosen_key: str = CHOSEN_INPUT_IDS_KEY,
    rejected_key: str = REJECTED_INPUT_IDS_KEY,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> PrefixSharingStats | None:
    """Measure shared-prefix redundancy over a tokenized preference dataset.

    Expects rows carrying ``chosen_key`` / ``rejected_key`` token-id lists (the
    fields produced by the preference tokenizers in ``dataset_transformation``).
    Returns ``None`` when the dataset is empty or lacks those fields, so callers
    can treat the diagnostic as best-effort.
    """
    num_examples = min(len(dataset), max_samples)
    if num_examples == 0:
        return None

    total_chosen = 0
    total_rejected = 0
    total_shared = 0
    counted = 0
    for index in range(num_examples):
        row = dataset[index]
        if chosen_key not in row or rejected_key not in row:
            return None
        chosen_ids = row[chosen_key]
        rejected_ids = row[rejected_key]
        prefix = shared_prefix_length(chosen_ids, rejected_ids)
        total_chosen += len(chosen_ids)
        total_rejected += len(rejected_ids)
        total_shared += prefix
        counted += 1

    if counted == 0:
        return None

    naive_tokens = total_chosen + total_rejected
    # Prefix sharing encodes the shared prefix once instead of twice.
    prefix_shared_tokens = naive_tokens - total_shared
    redundant_fraction = total_shared / naive_tokens if naive_tokens else 0.0
    speedup = naive_tokens / prefix_shared_tokens if prefix_shared_tokens else 1.0

    return PrefixSharingStats(
        num_examples=counted,
        mean_chosen_tokens=total_chosen / counted,
        mean_rejected_tokens=total_rejected / counted,
        mean_shared_prefix_tokens=total_shared / counted,
        redundant_token_fraction=redundant_fraction,
        estimated_token_speedup=speedup,
    )


def log_prefix_sharing_stats(
    dataset: Sequence[dict[str, Any]],
    chosen_key: str = CHOSEN_INPUT_IDS_KEY,
    rejected_key: str = REJECTED_INPUT_IDS_KEY,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> PrefixSharingStats | None:
    """Compute and log shared-prefix stats for a preference dataset.

    Best-effort: returns (and logs) ``None`` if the stats cannot be computed.
    """
    stats = compute_prefix_sharing_stats(dataset, chosen_key, rejected_key, max_samples)
    if stats is None:
        logger.info("Prefix-sharing stats unavailable (dataset empty or missing chosen/rejected token ids).")
        return None

    logger.info(
        "Prefix-sharing potential over %d examples: shared prefix avg %.1f tokens "
        "(chosen avg %.1f, rejected avg %.1f); %.1f%% of forwarded tokens are redundant prefix; "
        "estimated token-throughput speedup from prefix sharing ~%.2fx.",
        stats.num_examples,
        stats.mean_shared_prefix_tokens,
        stats.mean_chosen_tokens,
        stats.mean_rejected_tokens,
        100.0 * stats.redundant_token_fraction,
        stats.estimated_token_speedup,
    )
    return stats
