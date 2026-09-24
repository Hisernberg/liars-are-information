"""Regression: the cache stores RAW confidences, never a transformed value.

Comparing normalization schemes under matched self-vote share is the
normalization-parity control -- a headline methodological contribution. A cache
with a transform baked in silently destroys that comparison, and the corruption
is invisible because transformed values still look like confidences. These tests
fail loudly if anything ever normalizes on the write path.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from aip.harness.cache import load_predictions, write_predictions
from aip.models.confidence import (
    TokenLogprob,
    logprob_to_confidence,
    span_confidence,
    token_offsets,
)
from aip.types import Extraction, Prediction


def _tokens(pairs: list[tuple[str, float]]) -> list[TokenLogprob]:
    texts = [t for t, _ in pairs]
    return [
        TokenLogprob(text=t, logprob=lp, start=s, end=e)
        for (t, lp), (s, e) in zip(pairs, token_offsets(texts), strict=True)
    ]


def _prediction(task_id: str, confidence: float | None) -> Prediction:
    return Prediction(
        task_id=task_id,
        benchmark="gsm8k",
        model="m1",
        raw_completion="#### 42",
        extracted_answer="42",
        gold_answer="42",
        is_correct=True,
        logprob_confidence=confidence,
        self_reported_confidence=0.5,
        n_answer_tokens=1,
        n_completion_tokens=8,
        seed=0,
        temperature=0.0,
    )


class TestTransformIsExpOnly:
    def test_confidence_is_exactly_exp_of_mean_logprob(self) -> None:
        tokens = _tokens([("The answer is ", -0.5), ("42", -2.0)])
        value, _ = span_confidence(tokens, Extraction("42", 14, 16))
        assert value == pytest.approx(math.exp(-2.0), rel=1e-12)

    def test_mapping_is_pure(self) -> None:
        """Same input, same output, regardless of what else was computed."""
        assert logprob_to_confidence(-1.234) == logprob_to_confidence(-1.234)

    def test_no_dependence_on_other_values(self) -> None:
        """A rank or min-max transform would make one item's value depend on others."""
        lone, _ = span_confidence(_tokens([("42", -2.0)]), Extraction("42", 0, 2))
        with_neighbours, _ = span_confidence(
            _tokens([("x", -0.001), ("42", -2.0)]), Extraction("42", 1, 3)
        )
        assert lone == pytest.approx(with_neighbours, rel=1e-12)


class TestCacheRoundTripIsUntransformed:
    def test_values_survive_write_read_bit_exact(self, tmp_path: Path) -> None:
        values = [0.9999123456789, 0.5, 0.0001234, 1.0]
        write_predictions(
            [_prediction(f"t{i}", v) for i, v in enumerate(values)],
            "gsm8k",
            "m1",
            root=tmp_path,
        )
        frame = load_predictions("gsm8k", "m1", root=tmp_path).sort_values("task_id")
        for got, expected in zip(frame["logprob_confidence"].tolist(), values, strict=True):
            assert got == pytest.approx(expected, rel=1e-15)

    def test_write_does_not_rescale_to_unit_range(self, tmp_path: Path) -> None:
        """Min-max normalization would push the extremes to exactly 0 and 1."""
        values = [0.90, 0.95, 0.99]
        write_predictions(
            [_prediction(f"t{i}", v) for i, v in enumerate(values)],
            "gsm8k",
            "m1",
            root=tmp_path,
        )
        frame = load_predictions("gsm8k", "m1", root=tmp_path)
        assert frame["logprob_confidence"].min() == pytest.approx(0.90)
        assert frame["logprob_confidence"].max() == pytest.approx(0.99)

    def test_write_does_not_rank_transform(self, tmp_path: Path) -> None:
        """Rank-normalizing 3 rows would yield evenly spaced values."""
        values = [0.9990, 0.9991, 0.9999]
        write_predictions(
            [_prediction(f"t{i}", v) for i, v in enumerate(values)],
            "gsm8k",
            "m1",
            root=tmp_path,
        )
        got = sorted(load_predictions("gsm8k", "m1", root=tmp_path)["logprob_confidence"])
        gaps = [got[1] - got[0], got[2] - got[1]]
        assert gaps[0] != pytest.approx(gaps[1], abs=1e-6), "values look rank-transformed"

    def test_incremental_writes_do_not_renormalize_earlier_rows(self, tmp_path: Path) -> None:
        """Resume must not rewrite values already on disk."""
        write_predictions([_prediction("t0", 0.5)], "gsm8k", "m1", root=tmp_path)
        write_predictions([_prediction("t1", 0.999)], "gsm8k", "m1", root=tmp_path)
        frame = load_predictions("gsm8k", "m1", root=tmp_path).set_index("task_id")
        assert frame.loc["t0", "logprob_confidence"] == pytest.approx(0.5)
        assert frame.loc["t1", "logprob_confidence"] == pytest.approx(0.999)

    def test_self_reported_confidence_also_raw(self, tmp_path: Path) -> None:
        pred = _prediction("t0", 0.5)
        pred.self_reported_confidence = 0.83
        write_predictions([pred], "gsm8k", "m1", root=tmp_path)
        frame = load_predictions("gsm8k", "m1", root=tmp_path)
        assert frame["self_reported_confidence"].iloc[0] == pytest.approx(0.83)
