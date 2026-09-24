"""Benchmark contract: load, prompt, extract, score.

A :class:`Benchmark` owns four things that must stay in lockstep:

1. the task set (seeded, cached to disk so every phase sees identical tasks),
2. the prompt templates (hashed into the Phase A manifest),
3. answer extraction, which returns a *character span* as well as a value so
   that :mod:`aip.models.confidence` can restrict token logprobs to the answer
   tokens, and
4. the exact-match scorer.

Extraction failure is a normal outcome, not an exception: a malformed completion
yields ``Extraction(None)`` and scores incorrect.  Adversaries in Phase D
deliberately produce malformed output, so every extractor is fuzz-tested against
junk in ``tests/test_extraction.py``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from aip.harness.logging import get_logger
from aip.harness.manifest import text_hash
from aip.types import AnswerKind, Extraction, TaskItem

log = get_logger(__name__)

ChatMessage = dict[str, str]

DEFAULT_TASK_CACHE = Path("data/benchmarks")


class Benchmark(ABC):
    """Abstract benchmark."""

    name: str
    answer_kind: AnswerKind
    hf_path: str
    hf_config: str | None = None
    hf_split: str = "test"

    system_prompt: str
    user_template: str
    confidence_system_prompt: str
    confidence_template: str

    # -- task set --------------------------------------------------------

    @abstractmethod
    def _load_raw(self) -> list[TaskItem]:
        """Load the full split as :class:`TaskItem` records."""

    def load(
        self,
        n: int | None = 100,
        seed: int = 0,
        cache_dir: Path | None = None,
        refresh: bool = False,
    ) -> list[TaskItem]:
        """Return a deterministic ``n``-task subset, cached to disk.

        The cache key includes ``n`` and ``seed``, so two phases that ask for
        the same ``(n, seed)`` are guaranteed the same tasks in the same order
        even if the upstream dataset is re-downloaded.
        """
        cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_TASK_CACHE
        cache_path = cache_dir / f"{self.name}_n{n if n is not None else 'all'}_seed{seed}.json"
        if cache_path.exists() and not refresh:
            items = _read_task_cache(cache_path)
            log.debug("task_cache.hit", benchmark=self.name, n=len(items), path=str(cache_path))
            return items

        items = self._load_raw()
        if n is not None and n < len(items):
            items = self.subsample(items, n=n, seed=seed)
        _write_task_cache(cache_path, items)
        log.info(
            "task_cache.write",
            benchmark=self.name,
            n=len(items),
            seed=seed,
            path=str(cache_path),
        )
        return items

    def subsample(self, items: list[TaskItem], n: int, seed: int) -> list[TaskItem]:
        """Uniform seeded subsample. Overridden where stratification matters."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(items), size=n, replace=False)
        return [items[int(i)] for i in sorted(idx)]

    # -- prompting -------------------------------------------------------

    def build_messages(self, item: TaskItem) -> list[ChatMessage]:
        """Chat messages eliciting an answer. Applied to each model's own template."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(**self._format_fields(item))},
        ]

    def build_confidence_messages(self, item: TaskItem, answer: str | None) -> list[ChatMessage]:
        """Chat messages eliciting a *self-reported* 0-100 confidence.

        Deliberately a second, separate call: the self-report is the surface the
        ``falsified_confidence`` attack manipulates, and it must be storable
        independently of the logprob-derived confidence.
        """
        fields = self._format_fields(item)
        fields["answer"] = answer if answer is not None else "(no answer produced)"
        return [
            {"role": "system", "content": self.confidence_system_prompt},
            {"role": "user", "content": self.confidence_template.format(**fields)},
        ]

    def _format_fields(self, item: TaskItem) -> dict[str, Any]:
        return {"question": item.question}

    def prompt_template_hash(self) -> str:
        """Hash over every template, so a prompt edit invalidates the cache."""
        joined = "\x00".join(
            [
                self.name,
                self.system_prompt,
                self.user_template,
                self.confidence_system_prompt,
                self.confidence_template,
            ]
        )
        return text_hash(joined)

    # -- extraction & scoring --------------------------------------------

    @abstractmethod
    def extract_answer(self, completion: str) -> Extraction:
        """Parse an answer plus its character span out of a raw completion."""

    @abstractmethod
    def normalize(self, answer: str) -> str:
        """Canonical form used for equality and for swarm vote bucketing."""

    @abstractmethod
    def score(self, extracted: str | None, gold: str) -> bool:
        """Exact-match correctness."""

    def score_completion(self, completion: str, gold: str) -> tuple[Extraction, bool]:
        extraction = self.extract_answer(completion)
        return extraction, self.score(extraction.value, gold)


# -- confidence self-report parsing (shared by all benchmarks) -------------


def parse_self_reported_confidence(completion: str) -> float | None:
    """Parse a 0-100 self-reported confidence into [0, 1]; ``None`` if absent.

    Accepts ``"85"``, ``"85%"``, ``"Confidence: 85"``, ``"0.85"``.  Values are
    clamped to [0, 100] rather than rejected -- a model answering ``"120"`` is
    reporting maximal confidence, and Phase D adversaries emit out-of-range
    values on purpose.
    """
    import re

    if not completion:
        return None
    text = completion.strip()

    # Prefer an explicitly labelled confidence, then fall back to the last bare
    # number. Reasoning models emit a whole thinking block before committing, so
    # the *first* number in the text is usually an intermediate quantity from the
    # reasoning, not the answer.
    labelled = re.findall(r"confidence\s*(?:is|:|=)?\s*(-?\d+(?:\.\d+)?)\s*%?", text, re.IGNORECASE)
    candidates = labelled or re.findall(r"(-?\d+(?:\.\d+)?)\s*%?", text)
    if not candidates:
        return None
    raw = candidates[-1]
    try:
        value = float(raw)
    except ValueError:
        return None
    # A bare fraction in [0, 1] is read as a probability, not as "1%".
    if 0.0 <= value <= 1.0 and "." in raw:
        return value
    return max(0.0, min(100.0, value)) / 100.0


# -- task-set cache serialization -----------------------------------------


def _write_task_cache(path: Path, items: list[TaskItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "task_id": it.task_id,
            "benchmark": it.benchmark,
            "question": it.question,
            "gold_answer": it.gold_answer,
            "answer_kind": str(it.answer_kind),
            "choices": list(it.choices),
            "metadata": it.metadata,
        }
        for it in items
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _read_task_cache(path: Path) -> list[TaskItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        TaskItem(
            task_id=rec["task_id"],
            benchmark=rec["benchmark"],
            question=rec["question"],
            gold_answer=rec["gold_answer"],
            answer_kind=AnswerKind(rec["answer_kind"]),
            choices=tuple(rec.get("choices", ())),
            metadata=rec.get("metadata", {}),
        )
        for rec in payload
    ]
