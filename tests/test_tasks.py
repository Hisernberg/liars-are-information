"""Task layer: prompt construction, template hashing, task-set caching.

Dataset downloads are marked ``slow`` and exercised by ``tests/test_datasets.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aip.tasks import BENCHMARKS, get_benchmark
from aip.tasks.base import _read_task_cache, _write_task_cache
from aip.types import AnswerKind, TaskItem


@pytest.fixture
def mmlu_item() -> TaskItem:
    return TaskItem(
        task_id="mmlu-test-00001",
        benchmark="mmlu",
        question="Which organelle makes ATP?",
        gold_answer="B",
        answer_kind=AnswerKind.MULTIPLE_CHOICE,
        choices=("nucleus", "mitochondrion", "ribosome", "golgi"),
        metadata={"subject": "biology"},
    )


@pytest.fixture
def gsm8k_item() -> TaskItem:
    return TaskItem(
        task_id="gsm8k-test-00001",
        benchmark="gsm8k",
        question="Ann has 3 apples and buys 4 more. How many?",
        gold_answer="7",
        answer_kind=AnswerKind.OPEN,
    )


class TestRegistry:
    def test_all_benchmarks_instantiate(self) -> None:
        for name in BENCHMARKS:
            assert get_benchmark(name).name == name

    def test_aliases(self) -> None:
        assert get_benchmark("mmlu_subset").name == "mmlu"
        assert get_benchmark("MATH-500").name == "math500"

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_benchmark("imagenet")


class TestPrompts:
    @pytest.mark.parametrize("name", list(BENCHMARKS))
    def test_templates_render(self, name: str, mmlu_item: TaskItem, gsm8k_item: TaskItem) -> None:
        """Every template must survive str.format -- LaTeX braces included."""
        benchmark = get_benchmark(name)
        item = mmlu_item if name == "mmlu" else gsm8k_item
        messages = benchmark.build_messages(item)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert item.question in messages[1]["content"]

    def test_math_template_keeps_literal_braces(self, gsm8k_item: TaskItem) -> None:
        content = get_benchmark("math500").build_messages(gsm8k_item)[1]["content"]
        assert "\\boxed{<answer>}" in content

    def test_mmlu_prompt_lists_choices(self, mmlu_item: TaskItem) -> None:
        content = get_benchmark("mmlu").build_messages(mmlu_item)[1]["content"]
        assert "B. mitochondrion" in content
        assert "A, B, C, D" in content

    @pytest.mark.parametrize("name", list(BENCHMARKS))
    def test_confidence_prompt_includes_answer(self, name: str, mmlu_item, gsm8k_item) -> None:
        benchmark = get_benchmark(name)
        item = mmlu_item if name == "mmlu" else gsm8k_item
        content = benchmark.build_confidence_messages(item, "42")[1]["content"]
        assert "42" in content
        assert "0 to 100" in content

    def test_confidence_prompt_handles_missing_answer(self, gsm8k_item: TaskItem) -> None:
        content = get_benchmark("gsm8k").build_confidence_messages(gsm8k_item, None)[1]["content"]
        assert "no answer produced" in content


class TestTemplateHash:
    def test_stable(self) -> None:
        assert get_benchmark("gsm8k").prompt_template_hash() == (
            get_benchmark("gsm8k").prompt_template_hash()
        )

    def test_differs_across_benchmarks(self) -> None:
        hashes = {get_benchmark(n).prompt_template_hash() for n in BENCHMARKS}
        assert len(hashes) == len(BENCHMARKS)

    def test_sensitive_to_edits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        benchmark = get_benchmark("gsm8k")
        before = benchmark.prompt_template_hash()
        monkeypatch.setattr(benchmark, "system_prompt", "different", raising=False)
        assert benchmark.prompt_template_hash() != before


class TestTaskCache:
    def test_round_trip_preserves_fields(self, tmp_path: Path, mmlu_item: TaskItem) -> None:
        path = tmp_path / "tasks.json"
        _write_task_cache(path, [mmlu_item])
        (restored,) = _read_task_cache(path)
        assert restored == mmlu_item

    def test_label_space(self, mmlu_item: TaskItem, gsm8k_item: TaskItem) -> None:
        assert mmlu_item.label_space == ("A", "B", "C", "D")
        assert gsm8k_item.label_space == ()


class TestSubsampling:
    def _items(self, n: int, subjects: int = 5) -> list[TaskItem]:
        return [
            TaskItem(
                task_id=f"t{i}",
                benchmark="mmlu",
                question=f"q{i}",
                gold_answer="A",
                answer_kind=AnswerKind.MULTIPLE_CHOICE,
                choices=("a", "b", "c", "d"),
                metadata={"subject": f"s{i % subjects}"},
            )
            for i in range(n)
        ]

    def test_uniform_subsample_is_deterministic(self) -> None:
        benchmark = get_benchmark("gsm8k")
        items = self._items(50)
        a = benchmark.subsample(items, n=10, seed=7)
        b = benchmark.subsample(items, n=10, seed=7)
        assert [i.task_id for i in a] == [i.task_id for i in b]

    def test_different_seeds_differ(self) -> None:
        benchmark = get_benchmark("gsm8k")
        items = self._items(200)
        a = benchmark.subsample(items, n=20, seed=1)
        b = benchmark.subsample(items, n=20, seed=2)
        assert [i.task_id for i in a] != [i.task_id for i in b]

    def test_mmlu_stratifies_across_subjects(self) -> None:
        items = self._items(200, subjects=10)
        chosen = get_benchmark("mmlu").subsample(items, n=20, seed=0)
        subjects = {i.metadata["subject"] for i in chosen}
        assert len(chosen) == 20
        assert len(subjects) == 10, "stratified draw must cover every subject"

    def test_mmlu_stratification_is_deterministic(self) -> None:
        items = self._items(200, subjects=10)
        a = get_benchmark("mmlu").subsample(items, n=20, seed=3)
        b = get_benchmark("mmlu").subsample(items, n=20, seed=3)
        assert [i.task_id for i in a] == [i.task_id for i in b]

    def test_handles_n_larger_than_pool(self) -> None:
        items = self._items(10, subjects=2)
        chosen = get_benchmark("mmlu").subsample(items, n=50, seed=0)
        assert len(chosen) == 10
