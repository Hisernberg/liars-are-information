"""Token/character alignment and vLLM output parsing (no GPU required).

``align_tokens`` is the load-bearing piece: if character offsets are wrong, the
answer-span confidence silently measures the wrong tokens and every Phase C
number built on it is quietly invalid. These tests pin the contract, including
the failure mode where alignment cannot be established.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aip.models.confidence import TokenLogprob
from aip.models.inference import GenerationResult, _sampled_logprobs, align_tokens


class FakeTokenizer:
    """Decodes ids to pieces, mimicking a tokenizer whose pieces concatenate."""

    def __init__(self, vocab: dict[int, str]) -> None:
        self.vocab = vocab

    def decode(self, ids, skip_special_tokens: bool = True) -> str:  # noqa: ANN001
        return "".join(self.vocab[i] for i in ids)


class BrokenTokenizer(FakeTokenizer):
    """Decodes to something that does not match the reported completion."""

    def decode(self, ids, skip_special_tokens: bool = True) -> str:  # noqa: ANN001
        return "".join(self.vocab[i] for i in ids).replace(" ", "")


@dataclass
class FakeLogprob:
    logprob: float
    rank: int = 1


@dataclass
class FakeCompletion:
    token_ids: list[int]
    logprobs: list[dict[int, FakeLogprob]] | None
    text: str = ""


VOCAB = {1: "The", 2: " answer", 3: " is", 4: " 42", 5: "."}


class TestAlignTokens:
    def _tok(self) -> FakeTokenizer:
        return FakeTokenizer(VOCAB)

    def test_offsets_reconstruct_the_completion(self) -> None:
        ids = [1, 2, 3, 4, 5]
        text = "The answer is 42."
        tokens = align_tokens(self._tok(), ids, [-0.1] * 5, text)
        assert "".join(t.text for t in tokens) == text
        for token in tokens:
            assert text[token.start : token.end] == token.text

    def test_spans_are_contiguous_and_ordered(self) -> None:
        tokens = align_tokens(self._tok(), [1, 2, 3, 4, 5], [-0.1] * 5, "The answer is 42.")
        assert tokens[0].start == 0
        for a, b in zip(tokens, tokens[1:], strict=False):
            assert a.end == b.start

    def test_logprobs_are_attached_in_order(self) -> None:
        lps = [-0.1, -0.2, -0.3, -0.4, -0.5]
        tokens = align_tokens(self._tok(), [1, 2, 3, 4, 5], lps, "The answer is 42.")
        assert [t.logprob for t in tokens] == lps

    def test_answer_token_is_locatable_by_span(self) -> None:
        """The whole point: find the tokens covering ' 42'."""
        text = "The answer is 42."
        tokens = align_tokens(self._tok(), [1, 2, 3, 4, 5], [-0.1, -0.2, -0.3, -3.0, -0.4], text)
        start = text.index("42")
        covering = [t for t in tokens if t.start < start + 2 and t.end > start]
        assert [t.logprob for t in covering] == [-3.0]

    def test_mismatch_falls_back_to_zero_spans(self) -> None:
        """Never report offsets that were not verified against the text."""
        tokens = align_tokens(BrokenTokenizer(VOCAB), [1, 2], [-0.1, -0.2], "The answer")
        assert all(t.start == 0 and t.end == 0 for t in tokens)
        assert [t.logprob for t in tokens] == [-0.1, -0.2]

    def test_empty_generation(self) -> None:
        assert align_tokens(self._tok(), [], [], "") == []

    def test_missing_logprobs_are_tolerated(self) -> None:
        tokens = align_tokens(self._tok(), [1, 2, 3], [-0.1], "The answer is")
        assert len(tokens) == 1


class TestSampledLogprobs:
    def test_extracts_sampled_token_logprob(self) -> None:
        completion = FakeCompletion(
            token_ids=[7, 8],
            logprobs=[
                {7: FakeLogprob(-0.5), 9: FakeLogprob(-2.0)},
                {8: FakeLogprob(-1.5)},
            ],
        )
        assert _sampled_logprobs(completion) == [-0.5, -1.5]

    def test_no_logprobs_returns_empty(self) -> None:
        assert _sampled_logprobs(FakeCompletion(token_ids=[1], logprobs=None)) == []

    def test_none_entry_becomes_nan(self) -> None:
        import math

        out = _sampled_logprobs(FakeCompletion(token_ids=[1], logprobs=[None]))
        assert len(out) == 1 and math.isnan(out[0])

    def test_falls_back_when_sampled_id_absent(self) -> None:
        completion = FakeCompletion(token_ids=[7], logprobs=[{9: FakeLogprob(-2.0, rank=1)}])
        assert _sampled_logprobs(completion) == [-2.0]


class TestGenerationResult:
    def test_truncation_flag(self) -> None:
        assert GenerationResult(text="x", finish_reason="length").truncated
        assert not GenerationResult(text="x", finish_reason="stop").truncated

    def test_n_tokens(self) -> None:
        result = GenerationResult(
            text="ab", tokens=[TokenLogprob("a", -0.1, 0, 1), TokenLogprob("b", -0.2, 1, 2)]
        )
        assert result.n_tokens == 2


class TestRealTokenizerAlignment:
    """Same contract against a real HF tokenizer, where BPE merges are real."""

    @pytest.mark.slow
    def test_real_tokenizer_offsets_reconstruct(self) -> None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
        text = "Janet has 16 eggs.\nThe answer is 18.\n#### 18"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        decoded = tok.decode(ids, skip_special_tokens=True)
        tokens = align_tokens(tok, ids, [-0.1] * len(ids), decoded)
        assert "".join(t.text for t in tokens) == decoded
        for token in tokens:
            assert decoded[token.start : token.end] == token.text

    @pytest.mark.slow
    def test_real_tokenizer_answer_span_isolation(self) -> None:
        from transformers import AutoTokenizer

        from aip.models.confidence import span_confidence
        from aip.tasks import get_benchmark

        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
        text = "Reasoning that is long and confident.\n#### 42"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        decoded = tok.decode(ids, skip_special_tokens=True)
        # Make the answer tokens far less likely than the reasoning tokens.
        extraction = get_benchmark("gsm8k").extract_answer(decoded)
        lps = [-0.01] * len(ids)
        tokens = align_tokens(tok, ids, lps, decoded)
        covering = [
            i for i, t in enumerate(tokens) if t.start < extraction.end and t.end > extraction.start
        ]
        assert covering, "answer span matched no tokens"
        for i in covering:
            lps[i] = -4.0
        tokens = align_tokens(tok, ids, lps, decoded)
        answer_conf, n_used = span_confidence(tokens, extraction)
        whole_conf, _ = span_confidence(tokens, type(extraction)(extraction.value))
        assert n_used == len(covering)
        assert answer_conf < whole_conf
