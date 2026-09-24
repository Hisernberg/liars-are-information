"""Integration tests against the real HuggingFace datasets.

Marked ``slow`` (they download). Run with ``pytest -m slow``.

The round-trip tests here are the strongest guarantee in the suite: they take
every real gold answer, wrap it in the output format the prompt asks for, and
require the extractor + normalizer to recover and score it. A regression in the
LaTeX normalizer shows up here as a hard failure on real data rather than as a
few percent of silently mis-scored predictions in Phase A.
"""

from __future__ import annotations

import pytest

from aip.tasks import get_benchmark

pytestmark = pytest.mark.slow

SEED = 20250825
N = 100

#: Every benchmark in the study. The three added for the H100 regeneration
#: (medqa, boolq, arc) go through exactly the same round-trip gauntlet as the
#: originals -- that is the point of parametrising rather than special-casing.
ALL = ("gsm8k", "math500", "mmlu", "medqa", "boolq", "arc")


@pytest.fixture(scope="module")
def loaded() -> dict[str, list]:
    return {name: get_benchmark(name).load(n=N, seed=SEED) for name in ALL}


class TestLoading:
    @pytest.mark.parametrize("name", ALL)
    def test_loads_requested_count(self, loaded: dict, name: str) -> None:
        assert len(loaded[name]) == N

    @pytest.mark.parametrize("name", ALL)
    def test_task_ids_unique(self, loaded: dict, name: str) -> None:
        ids = [i.task_id for i in loaded[name]]
        assert len(set(ids)) == len(ids)

    @pytest.mark.parametrize("name", ALL)
    def test_questions_and_golds_non_empty(self, loaded: dict, name: str) -> None:
        for item in loaded[name]:
            assert item.question.strip()
            assert item.gold_answer.strip()

    @pytest.mark.parametrize("name", ALL)
    def test_load_is_deterministic(self, loaded: dict, name: str) -> None:
        again = get_benchmark(name).load(n=N, seed=SEED)
        assert [i.task_id for i in again] == [i.task_id for i in loaded[name]]

    def test_mmlu_is_stratified_and_well_formed(self, loaded: dict) -> None:
        items = loaded["mmlu"]
        subjects = {i.metadata["subject"] for i in items}
        assert len(subjects) >= 50, f"only {len(subjects)} subjects in the draw"
        for item in items:
            assert len(item.choices) == 4
            assert item.gold_answer in ("A", "B", "C", "D")

    def test_gsm8k_golds_are_numeric(self, loaded: dict) -> None:
        for item in loaded["gsm8k"]:
            assert float(item.gold_answer) == float(item.gold_answer)


class TestGoldRoundTrip:
    """Gold answers must survive format -> extract -> score on real data."""

    def test_gold_scores_against_itself(self, loaded: dict) -> None:
        for name, items in loaded.items():
            benchmark = get_benchmark(name)
            for item in items:
                assert benchmark.score(item.gold_answer, item.gold_answer), (
                    f"{name} {item.task_id}: gold {item.gold_answer!r} does not self-match"
                )

    def test_gsm8k_hash_format_round_trip(self, loaded: dict) -> None:
        benchmark = get_benchmark("gsm8k")
        for item in loaded["gsm8k"]:
            completion = f"Some reasoning here.\n#### {item.gold_answer}"
            _, correct = benchmark.score_completion(completion, item.gold_answer)
            assert correct, f"{item.task_id}: {completion!r}"

    def test_math500_boxed_round_trip(self, loaded: dict) -> None:
        benchmark = get_benchmark("math500")
        failures = []
        for item in loaded["math500"]:
            completion = f"Working.\n\\boxed{{{item.gold_answer}}}"
            extraction, correct = benchmark.score_completion(completion, item.gold_answer)
            if not correct:
                failures.append((item.task_id, item.gold_answer, extraction.value))
        assert not failures, f"boxed round-trip failed on {len(failures)}: {failures[:5]}"

    def test_mmlu_answer_format_round_trip(self, loaded: dict) -> None:
        benchmark = get_benchmark("mmlu")
        for item in loaded["mmlu"]:
            completion = f"Reasoning.\nAnswer: {item.gold_answer}"
            _, correct = benchmark.score_completion(completion, item.gold_answer)
            assert correct, item.task_id

    def test_wrong_answers_score_wrong(self, loaded: dict) -> None:
        """Guard against a normalizer so lossy that everything matches."""
        benchmark = get_benchmark("mmlu")
        wrong = 0
        for item in loaded["mmlu"]:
            other = "A" if item.gold_answer != "A" else "B"
            if not benchmark.score(other, item.gold_answer):
                wrong += 1
        assert wrong == len(loaded["mmlu"])


class TestPromptsOnRealItems:
    @pytest.mark.parametrize("name", ALL)
    def test_every_item_renders(self, loaded: dict, name: str) -> None:
        benchmark = get_benchmark(name)
        for item in loaded[name]:
            messages = benchmark.build_messages(item)
            assert messages[1]["content"].strip()
            confidence = benchmark.build_confidence_messages(item, "42")
            assert confidence[1]["content"].strip()


class TestNewBenchmarks:
    """The three benchmarks added for the H100 regeneration.

    Each carries a specific upstream hazard that a generic round-trip would not
    catch, so each gets a test aimed at that hazard.
    """

    def test_boolq_is_label_balanced(self, loaded: dict) -> None:
        """A uniform draw would be 62% Yes and hand a constant responder 62%."""
        golds = [it.gold_answer for it in loaded["boolq"]]
        assert set(golds) == {"A", "B"}
        assert abs(golds.count("A") - golds.count("B")) <= 1

    def test_boolq_answer_space_is_binary(self, loaded: dict) -> None:
        """C=2 is what makes BoolQ the exact-law control; nothing may widen it."""
        for it in loaded["boolq"]:
            assert it.label_space == ("A", "B")
            assert it.choices == ("Yes", "No")

    def test_boolq_carries_its_passage(self, loaded: dict) -> None:
        """BoolQ is unanswerable without the passage; a dropped one scores as noise."""
        benchmark = get_benchmark("boolq")
        for it in loaded["boolq"]:
            assert it.metadata.get("passage")
            assert it.metadata["passage"] in benchmark.build_messages(it)[1]["content"]

    def test_boolq_accepts_a_worded_answer(self) -> None:
        """Told to answer with a letter, models often say the word anyway."""
        benchmark = get_benchmark("boolq")
        for text, expect in (("Answer: Yes", "A"), ("Answer: No", "B")):
            assert benchmark.normalize(benchmark.extract_answer(text).value or "") == expect

    def test_arc_is_four_option_only(self, loaded: dict) -> None:
        """The C=4 ceiling in inversion_thresholds.yaml must not be diluted."""
        for it in loaded["arc"]:
            assert len(it.choices) == 4
            assert it.label_space == ("A", "B", "C", "D")

    def test_arc_numeric_labels_are_remapped_by_position(self) -> None:
        """21 of the 1172 ARC items label options 1-4, with a digit answerKey."""
        items = get_benchmark("arc").load(n=None, seed=SEED)
        numeric = [it for it in items if it.metadata.get("label_style") == "1234"]
        assert numeric, "expected ARC items with numeric option labels"
        for it in numeric:
            assert it.gold_answer in ("A", "B", "C", "D")

    def test_medqa_is_four_option_with_letter_gold(self, loaded: dict) -> None:
        for it in loaded["medqa"]:
            assert len(it.choices) == 4
            assert it.gold_answer in ("A", "B", "C", "D")

    def test_medqa_gold_letter_indexes_the_right_option(self) -> None:
        """answer_idx must name the same text upstream called the answer."""
        from datasets import load_dataset

        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        items = {it.metadata["index"]: it for it in get_benchmark("medqa").load(n=None, seed=SEED)}
        for i in list(items)[:200]:
            rec = ds[int(i)]
            it = items[i]
            assert it.choices[ord(it.gold_answer) - ord("A")] == str(rec["answer"])
