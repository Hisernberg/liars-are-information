"""Confidence surfaces.

Two are stored per prediction and they are not interchangeable:

``logprob_confidence``
    Mean token logprob over the *answer span only* (not the chain of thought),
    mapped to (0, 1].  An agent cannot inflate this by asserting confidence in
    text, which is exactly why the ``falsified_confidence`` attack cannot touch
    it while it does corrupt every self-report-reading aggregator.

``self_reported_confidence``
    Elicited by a separate prompt asking for an integer 0-100.  Stored so we can
    show the attack landing on baselines that read it.

The span-to-token mapping below is pure and unit-tested; the vLLM adapter that
supplies token offsets lands in Phase A alongside :mod:`aip.models.inference`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from aip.types import Extraction


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    """One generated token with its character offsets in the completion."""

    text: str
    logprob: float
    start: int
    end: int


def token_offsets(token_texts: list[str], start: int = 0) -> list[tuple[int, int]]:
    """Character spans for a list of token strings concatenated in order.

    vLLM returns detokenized token strings whose concatenation reproduces the
    completion, so cumulative lengths give exact offsets.
    """
    offsets: list[tuple[int, int]] = []
    cursor = start
    for text in token_texts:
        offsets.append((cursor, cursor + len(text)))
        cursor += len(text)
    return offsets


def tokens_overlapping_span(
    tokens: list[TokenLogprob], span_start: int, span_end: int
) -> list[TokenLogprob]:
    """Tokens whose character range intersects ``[span_start, span_end)``.

    Overlap rather than containment, because tokenizers routinely merge an
    answer with adjacent punctuation (``"42."`` as one token); requiring
    containment would drop the only token carrying the answer.
    """
    if span_end <= span_start:
        return []
    return [t for t in tokens if t.start < span_end and t.end > span_start]


def mean_logprob(tokens: list[TokenLogprob]) -> float | None:
    """Mean logprob over tokens; ``None`` when there are none."""
    if not tokens:
        return None
    return sum(t.logprob for t in tokens) / len(tokens)


def logprob_to_confidence(mean_lp: float | None) -> float | None:
    """Map a mean token logprob to a (0, 1] confidence via ``exp``.

    ``exp(mean logprob)`` is the geometric-mean per-token probability of the
    answer span: length-normalized, monotone in the model's own likelihood, and
    directly comparable across models with different answer lengths.

    **This is the only transform applied, and the result is what the cache
    stores.**  No rank-normalization, z-scoring, min-max rescaling, temperature
    calibration, or any other cross-row transform happens on the write path.
    Those belong in the Phase C aggregation layer, applied at consumption time.

    The reason is methodological, not stylistic: comparing normalization schemes
    under matched self-vote share *is* the normalization-parity control, one of
    the paper's contributions.  A cache with a transform baked in cannot support
    that comparison, and the damage would be invisible -- the numbers would still
    look like confidences.  ``tests/test_confidence_raw.py`` enforces this.
    """
    if mean_lp is None:
        return None
    if math.isnan(mean_lp) or math.isinf(mean_lp):
        return None
    return float(math.exp(min(0.0, mean_lp)))


def span_confidence(
    tokens: list[TokenLogprob], extraction: Extraction, fallback_to_all: bool = True
) -> tuple[float | None, int]:
    """Confidence for an extracted answer, plus the number of tokens used.

    Falls back to the whole completion when the extraction has no usable span
    (for example a malformed adversarial completion), so that a prediction
    always carries *some* confidence rather than a null that silently drops the
    agent from weighted aggregation.
    """
    selected: list[TokenLogprob] = []
    if extraction.has_span:
        selected = tokens_overlapping_span(tokens, int(extraction.start), int(extraction.end))
    if not selected and fallback_to_all:
        selected = tokens
    return logprob_to_confidence(mean_logprob(selected)), len(selected)
