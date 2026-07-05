"""Diagnostic for tool-use RL collapse: control-token probability mass.

Adapted from "Why Multi-Step Tool-Use Reinforcement Learning Collapses and How
Supervisory Signals Fix It" (arXiv:2606.26027), which attributes catastrophic
collapse in multi-step tool-use RL to unexpected spikes in the policy's
per-step probability on a small set of *control tokens* — the structural
markers (``<|im_start|>``, ``<tool_response>``, ``<think>``, ...) that delimit
tool-call formatting. The underlying tool-use capability stays intact; it is
merely obscured by the format breakdown.

This module ships only the diagnostic signal: how much of the policy's
next-token probability mass lands on those control tokens. It is computed from
the logits that :func:`open_instruct.grpo_utils.forward_for_logprobs` already
materializes, so enabling it costs one ``softmax`` — never an extra forward
pass. Masking and averaging over the response is done by the loss-stats path
with :func:`open_instruct.rl_utils.masked_mean`, mirroring the entropy stat.

The supervisory-signal training fixes from the paper (off-policy / hint /
erroneous-example SFT interleaving) are deliberately out of scope here — they
need data-generation pipelines this repo does not host. This module is the
observable that detects the collapse, not the fix for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Structural control tokens used by the tool-call parsers registered in
# ``open_instruct/environments/tools/parsers.py`` (Hermes/ChatML, Qwen3 XML,
# Olmo 3, and the legacy ``<tool_name>`` format). A spike on any of these
# during GRPO is the collapse signature the paper describes.
DEFAULT_CONTROL_TOKENS: tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<tool_name>",
    "</tool_name>",
)


def resolve_control_token_ids(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, tokens: tuple[str, ...] | list[str] | None = None
) -> list[int]:
    """Map control-token strings to single vocab ids for ``tokenizer``.

    Tokens absent from the vocab or split into multiple subtokens are skipped —
    the monitor tracks mass on whole control tokens, and a partial match would
    muddy the signal. The unknown-token id is also dropped. Returns an empty
    list when nothing resolves, which disables the monitor at the call site.
    """
    if tokens is None:
        tokens = DEFAULT_CONTROL_TOKENS
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    ids: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            continue
        token_id = encoded[0]
        if token_id == unk_token_id or token_id in seen:
            continue
        seen.add(token_id)
        ids.append(token_id)
    return ids


def control_token_mass(
    logits: torch.Tensor, control_token_ids: torch.Tensor | list[int] | tuple[int, ...] | None
) -> torch.Tensor | None:
    """Per-position probability mass the policy assigns to control tokens.

    ``logits`` is the shifted next-token logits tensor (``... × vocab``) that
    ``forward_for_logprobs`` already computes. Returns a tensor shaped like
    ``logits`` without the vocab dim — mask and average it over the response at
    the call site, exactly as the entropy stat does — or ``None`` when
    ``control_token_ids`` is empty (monitor off).
    """
    if not control_token_ids:
        return None
    probs = torch.softmax(logits, dim=-1)
    return probs[..., control_token_ids].sum(dim=-1)
