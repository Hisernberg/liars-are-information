"""Normalization and exact-match scoring."""

from __future__ import annotations

import pytest

from aip.tasks import get_benchmark
from aip.tasks.base import parse_self_reported_confidence
from aip.tasks.gsm8k import normalize_number, numeric_match
from aip.tasks.math500 import math_match, normalize_latex


class TestNumberNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("42", "42"),
            ("42.0", "42"),
            ("1,234", "1234"),
            ("$5.50", "5.5"),
            (" 7 ", "7"),
            ("-0", "0"),
            ("18%", "18"),
            ("3.", "3"),
            ("abc", ""),
            ("", ""),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_number(raw) == expected

    def test_numeric_match_tolerates_formatting(self) -> None:
        assert numeric_match("1,234", "1234")
        assert numeric_match("42.000", "42")
        assert not numeric_match("41", "42")
        assert not numeric_match(None, "42")


class TestLatexNormalization:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("\\dfrac{1}{2}", "\\frac{1}{2}"),
            ("\\frac12", "\\frac{1}{2}"),
            ("\\left(3\\right)", "(3)"),
            ("5^\\circ", "5"),
            ("x = 7", "7"),
            ("$3$", "3"),
            (".5", "0.5"),
            ("2.0", "2"),
            ("\\text{even}", "even"),
            ("3\\%", "3"),
            ("{42}", "42"),
        ],
    )
    def test_equivalent_surface_forms(self, a: str, b: str) -> None:
        assert normalize_latex(a) == normalize_latex(b)

    def test_distinct_answers_stay_distinct(self) -> None:
        assert normalize_latex("\\frac{1}{2}") != normalize_latex("\\frac{1}{3}")

    def test_math_match_symbolic_fallback(self) -> None:
        assert math_match("\\frac{2}{4}", "\\frac{1}{2}")
        assert not math_match("\\frac{2}{5}", "\\frac{1}{2}")

    def test_math_match_none(self) -> None:
        assert not math_match(None, "5")


class TestBenchmarkScorers:
    def test_gsm8k_end_to_end(self) -> None:
        benchmark = get_benchmark("gsm8k")
        extraction, correct = benchmark.score_completion("reasoning\n#### 1,234", "1234")
        assert extraction.value == "1234"
        assert correct

    def test_math500_end_to_end(self) -> None:
        benchmark = get_benchmark("math500")
        _, correct = benchmark.score_completion("thus \\boxed{\\dfrac{1}{2}}", "\\frac{1}{2}")
        assert correct

    def test_mmlu_end_to_end(self) -> None:
        benchmark = get_benchmark("mmlu")
        _, correct = benchmark.score_completion("Answer: C", "C")
        assert correct

    def test_failed_extraction_scores_incorrect(self) -> None:
        for name, gold in (("gsm8k", "42"), ("math500", "42"), ("mmlu", "A")):
            benchmark = get_benchmark(name)
            _, correct = benchmark.score_completion("", gold)
            assert not correct


class TestSelfReportedConfidence:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("85", 0.85),
            ("85%", 0.85),
            ("Confidence: 90", 0.90),
            ("0.75", 0.75),
            ("120", 1.0),
            ("-5", 0.0),
            ("", None),
            ("no idea", None),
        ],
    )
    def test_parse(self, raw: str, expected: float | None) -> None:
        result = parse_self_reported_confidence(raw)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)
