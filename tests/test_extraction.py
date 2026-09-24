"""Answer extraction on well-formed, unusual, and adversarial completions."""

from __future__ import annotations

import pytest

from aip.tasks import get_benchmark
from aip.tasks.gsm8k import extract_gsm8k_answer
from aip.tasks.math500 import extract_math_answer, find_boxed_spans
from aip.tasks.mmlu import extract_mc_answer


class TestGSM8K:
    @pytest.mark.parametrize(
        ("completion", "expected"),
        [
            ("Some reasoning.\n#### 42", "42"),
            ("#### 1,234", "1234"),
            ("#### $42", "42"),
            ("#### -7", "-7"),
            ("#### 42.0", "42"),
            ("#### 3.50", "3.5"),
            ("The final answer is 18.", "18"),
            ("Answer: 99", "99"),
            ("So the answer is $12,500.", "12500"),
            ("Therefore \\boxed{56} is right.", "56"),
            ("First 3 apples, then 4 more, giving 7", "7"),
        ],
    )
    def test_known_formats(self, completion: str, expected: str) -> None:
        assert extract_gsm8k_answer(completion).value == expected

    def test_hash_form_wins_over_earlier_numbers(self) -> None:
        completion = "We had 5 boxes and 6 crates, so 11 things.\n#### 11"
        extraction = extract_gsm8k_answer(completion)
        assert extraction.value == "11"
        assert completion[extraction.start : extraction.end] == "11"

    def test_last_hash_answer_wins(self) -> None:
        assert extract_gsm8k_answer("#### 1\nwait\n#### 2").value == "2"

    @pytest.mark.parametrize("completion", ["", "   ", "I cannot answer.", "no digits here"])
    def test_malformed_returns_none(self, completion: str) -> None:
        assert extract_gsm8k_answer(completion).value is None

    def test_span_is_recoverable(self) -> None:
        completion = "blah blah\n#### 512"
        extraction = extract_gsm8k_answer(completion)
        assert extraction.has_span
        assert completion[extraction.start : extraction.end] == "512"


class TestMATH500:
    @pytest.mark.parametrize(
        ("completion", "expected"),
        [
            ("Thus \\boxed{42}.", "42"),
            ("Thus \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),
            ("\\boxed{\\frac{\\sqrt{3}}{2}}", "\\frac{\\sqrt{3}}{2}"),
            ("\\fbox{7}", "7"),
            ("The answer is 15", "15"),
            ("answer: x^2 + 1", "x^2 + 1"),
        ],
    )
    def test_known_formats(self, completion: str, expected: str) -> None:
        assert extract_math_answer(completion).value == expected

    def test_nested_braces_are_brace_matched(self) -> None:
        completion = "so \\boxed{\\frac{a}{b} + \\sqrt{c}}"
        assert extract_math_answer(completion).value == "\\frac{a}{b} + \\sqrt{c}"

    def test_last_box_wins(self) -> None:
        assert extract_math_answer("\\boxed{1} then \\boxed{2}").value == "2"

    def test_unclosed_box_does_not_crash(self) -> None:
        extract_math_answer("\\boxed{1 + 2")  # must not raise

    def test_find_boxed_spans_returns_offsets(self) -> None:
        text = "x \\boxed{9} y"
        spans = find_boxed_spans(text)
        assert len(spans) == 1
        start, end, content = spans[0]
        assert text[start:end] == content == "9"

    @pytest.mark.parametrize("completion", ["", "   "])
    def test_empty_returns_none(self, completion: str) -> None:
        assert extract_math_answer(completion).value is None


class TestMMLU:
    @pytest.mark.parametrize(
        ("completion", "expected"),
        [
            ("Answer: B", "B"),
            ("answer is (C)", "C"),
            ("ANSWER: d", "D"),
            ("\\boxed{A}", "A"),
            ("**C**", "C"),
            ("B) because the enzyme denatures", "B"),
            ("Final answer: A.", "A"),
        ],
    )
    def test_known_formats(self, completion: str, expected: str) -> None:
        assert extract_mc_answer(completion).value == expected

    def test_out_of_range_letter_rejected(self) -> None:
        assert extract_mc_answer("Answer: G", n_choices=4).value is None

    def test_choice_text_fallback(self) -> None:
        choices = ("mitochondria", "ribosome", "nucleus", "golgi")
        assert extract_mc_answer("It is the nucleus.", choices=choices).value == "C"

    def test_last_answer_declaration_wins(self) -> None:
        assert extract_mc_answer("Answer: A\nOn reflection, Answer: D").value == "D"

    @pytest.mark.parametrize("completion", ["", "   ", "I refuse."])
    def test_malformed_returns_none(self, completion: str) -> None:
        assert extract_mc_answer(completion).value is None


class TestAdversarialRobustness:
    """Phase D adversaries emit junk on purpose; extractors must never raise."""

    JUNK = [
        "\\boxed{" * 50,
        "#### " * 200,
        "\x00\x01 binary �",
        "Answer: " + "A" * 5000,
        "}{}{}{}{",
        "\\frac{}{}",
        "-" * 1000,
        "####",
        "\\boxed{}",
    ]

    @pytest.mark.parametrize("benchmark_name", ["gsm8k", "math500", "mmlu"])
    @pytest.mark.parametrize("junk", JUNK)
    def test_no_exceptions_on_junk(self, benchmark_name: str, junk: str) -> None:
        benchmark = get_benchmark(benchmark_name)
        extraction = benchmark.extract_answer(junk)
        assert extraction.value is None or isinstance(extraction.value, str)
        # Whatever came back must be scoreable without raising.
        benchmark.score(extraction.value, "42")

    @pytest.mark.parametrize("benchmark_name", ["gsm8k", "math500", "mmlu"])
    def test_span_within_bounds_when_present(self, benchmark_name: str) -> None:
        benchmark = get_benchmark(benchmark_name)
        completions = ["Answer: B", "#### 42", "\\boxed{42}", "the answer is 42"]
        for completion in completions:
            extraction = benchmark.extract_answer(completion)
            if extraction.has_span:
                assert 0 <= extraction.start <= extraction.end <= len(completion)
