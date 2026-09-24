"""Answer-span confidence extraction (pure logic; no GPU required)."""

from __future__ import annotations

import math

import pytest

from aip.models.confidence import (
    TokenLogprob,
    logprob_to_confidence,
    mean_logprob,
    span_confidence,
    token_offsets,
    tokens_overlapping_span,
)
from aip.types import Extraction


def _tokens(pairs: list[tuple[str, float]]) -> list[TokenLogprob]:
    texts = [t for t, _ in pairs]
    return [
        TokenLogprob(text=t, logprob=lp, start=s, end=e)
        for (t, lp), (s, e) in zip(pairs, token_offsets(texts), strict=True)
    ]


class TestOffsets:
    def test_offsets_are_contiguous(self) -> None:
        assert token_offsets(["ab", "cde", "f"]) == [(0, 2), (2, 5), (5, 6)]

    def test_offsets_reconstruct_text(self) -> None:
        texts = ["The", " answer", " is", " 42"]
        text = "".join(texts)
        for token, (start, end) in zip(texts, token_offsets(texts), strict=True):
            assert text[start:end] == token


class TestSpanSelection:
    def test_selects_only_answer_tokens(self) -> None:
        tokens = _tokens([("The answer is ", -0.1), ("42", -2.0), (".", -0.05)])
        selected = tokens_overlapping_span(tokens, 14, 16)
        assert [t.text for t in selected] == ["42"]

    def test_overlap_not_containment(self) -> None:
        """A token merging the answer with punctuation must still be counted."""
        tokens = _tokens([("Answer: ", -0.1), ("42.", -1.5)])
        selected = tokens_overlapping_span(tokens, 8, 10)
        assert [t.text for t in selected] == ["42."]

    def test_empty_span(self) -> None:
        tokens = _tokens([("a", -0.1)])
        assert tokens_overlapping_span(tokens, 5, 5) == []


class TestConfidenceMapping:
    def test_mean_logprob(self) -> None:
        assert mean_logprob(_tokens([("a", -1.0), ("b", -3.0)])) == pytest.approx(-2.0)

    def test_mean_of_nothing_is_none(self) -> None:
        assert mean_logprob([]) is None

    def test_exp_mapping(self) -> None:
        assert logprob_to_confidence(-1.0) == pytest.approx(math.exp(-1.0))
        assert logprob_to_confidence(0.0) == pytest.approx(1.0)

    def test_positive_logprob_clamped(self) -> None:
        assert logprob_to_confidence(0.5) == pytest.approx(1.0)

    def test_non_finite_is_none(self) -> None:
        assert logprob_to_confidence(float("nan")) is None
        assert logprob_to_confidence(float("-inf")) is None
        assert logprob_to_confidence(None) is None

    def test_confidence_is_monotone_in_logprob(self) -> None:
        assert logprob_to_confidence(-0.1) > logprob_to_confidence(-2.0)


class TestSpanConfidence:
    def test_uses_answer_span_not_whole_completion(self) -> None:
        """A confident chain of thought must not mask an unconfident answer."""
        tokens = _tokens([("Clearly the answer is ", -0.01), ("7", -3.0)])
        confident_answer, n_used = span_confidence(tokens, Extraction("7", 22, 23))
        whole, _ = span_confidence(tokens, Extraction("7"), fallback_to_all=True)
        assert n_used == 1
        assert confident_answer < whole

    def test_falls_back_when_no_span(self) -> None:
        tokens = _tokens([("junk", -2.0)])
        value, n_used = span_confidence(tokens, Extraction(None))
        assert value is not None and n_used == 1

    def test_no_fallback_yields_none(self) -> None:
        tokens = _tokens([("junk", -2.0)])
        value, n_used = span_confidence(tokens, Extraction(None), fallback_to_all=False)
        assert value is None and n_used == 0
