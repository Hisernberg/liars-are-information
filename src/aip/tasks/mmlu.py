"""MMLU (cais/mmlu, 'all', test split): stratified multiple-choice subset.

MMLU is the only closed-label-space benchmark here, which makes it the one where
AIP's multiclass decode and the Dawid-Skene baselines operate on their natural
domain.  The subset is stratified across subjects so that a 100-question draw is
not dominated by the largest subject.
"""

from __future__ import annotations

import re

import numpy as np

from aip.harness.logging import get_logger
from aip.tasks.base import Benchmark
from aip.types import AnswerKind, Extraction, TaskItem

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a knowledgeable expert answering multiple-choice questions. You "
    "think briefly, then commit to exactly one option letter."
)

USER_TEMPLATE = (
    "{question}\n\n"
    "{choices_block}\n\n"
    "Think briefly. Then give your final answer on its own last line in exactly "
    "this format:\n"
    "Answer: <letter>\n"
    "where <letter> is one of {letters}."
)

CONFIDENCE_SYSTEM_PROMPT = (
    "You judge how likely an answer is to be correct. You reply with a single "
    "integer and nothing else."
)

CONFIDENCE_TEMPLATE = (
    "Question:\n{question}\n\n"
    "{choices_block}\n\n"
    "Proposed answer: {answer}\n\n"
    "How confident are you that this answer is correct? Reply with a single "
    "integer from 0 to 100 and nothing else."
)

_ANSWER_LINE = re.compile(r"answer\s*(?:is|:|=)\s*\**\s*\(?\s*([A-Za-z])\s*\)?", re.IGNORECASE)
_BOXED_LETTER = re.compile(r"\\boxed\{\s*\(?\s*([A-Za-z])\s*\)?\s*\}")
_BOLD_LETTER = re.compile(r"\*\*\s*\(?([A-Za-z])\)?\s*\*\*")
_LEADING_LETTER = re.compile(r"^\s*\(?([A-Za-z])\)?\s*[).:\-]")
_STANDALONE_LETTER = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")


class MMLU(Benchmark):
    name = "mmlu"
    answer_kind = AnswerKind.MULTIPLE_CHOICE
    hf_path = "cais/mmlu"
    hf_config = "all"
    hf_split = "test"

    system_prompt = SYSTEM_PROMPT
    user_template = USER_TEMPLATE
    confidence_system_prompt = CONFIDENCE_SYSTEM_PROMPT
    confidence_template = CONFIDENCE_TEMPLATE

    def _load_raw(self) -> list[TaskItem]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_path, self.hf_config, split=self.hf_split)
        items: list[TaskItem] = []
        for i, rec in enumerate(ds):
            choices = tuple(str(c) for c in rec["choices"])
            gold_idx = int(rec["answer"])
            items.append(
                TaskItem(
                    task_id=f"mmlu-test-{i:05d}",
                    benchmark=self.name,
                    question=rec["question"].strip(),
                    gold_answer=chr(ord("A") + gold_idx),
                    answer_kind=self.answer_kind,
                    choices=choices,
                    metadata={"subject": rec.get("subject"), "index": i},
                )
            )
        log.info("dataset.loaded", benchmark=self.name, n=len(items))
        return items

    def subsample(self, items: list[TaskItem], n: int, seed: int) -> list[TaskItem]:
        """Subject-stratified draw: round-robin over subjects until ``n`` filled."""
        rng = np.random.default_rng(seed)
        by_subject: dict[str, list[int]] = {}
        for idx, item in enumerate(items):
            by_subject.setdefault(str(item.metadata.get("subject", "unknown")), []).append(idx)

        for subject in by_subject:
            order = rng.permutation(len(by_subject[subject]))
            by_subject[subject] = [by_subject[subject][int(i)] for i in order]

        subjects = sorted(by_subject)
        chosen: list[int] = []
        cursor = 0
        while len(chosen) < n:
            progressed = False
            for subject in subjects:
                if len(chosen) >= n:
                    break
                pool = by_subject[subject]
                if cursor < len(pool):
                    chosen.append(pool[cursor])
                    progressed = True
            if not progressed:
                break
            cursor += 1
        log.info(
            "mmlu.stratified",
            n=len(chosen),
            subjects=len({items[i].metadata.get("subject") for i in chosen}),
            seed=seed,
        )
        return [items[i] for i in sorted(chosen)]

    def _format_fields(self, item: TaskItem) -> dict[str, object]:
        letters = item.label_space
        return {
            "question": item.question,
            "choices_block": format_choices(item.choices),
            "letters": ", ".join(letters),
        }

    def extract_answer(self, completion: str) -> Extraction:
        return extract_mc_answer(completion)

    def normalize(self, answer: str) -> str:
        return str(answer).strip().upper()[:1] if answer else ""

    def score(self, extracted: str | None, gold: str) -> bool:
        if not extracted:
            return False
        return self.normalize(extracted) == self.normalize(gold)


def format_choices(choices: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))


def extract_mc_answer(
    completion: str, choices: tuple[str, ...] | None = None, n_choices: int = 4
) -> Extraction:
    """Extract a multiple-choice letter.

    Tries the instructed ``Answer: X`` form, then boxed / bold / leading-letter
    conventions, then -- if ``choices`` are supplied -- matching the verbatim
    choice text back to its letter, and finally a standalone letter anywhere in
    the last line.
    """
    if not completion:
        return Extraction(None)

    valid = {chr(ord("A") + i) for i in range(n_choices)}

    for pattern in (_ANSWER_LINE, _BOXED_LETTER, _BOLD_LETTER):
        for m in reversed(list(pattern.finditer(completion))):
            letter = m.group(1).upper()
            if letter in valid:
                return Extraction(letter, m.start(1), m.end(1))

    lines = [ln for ln in completion.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        m = _LEADING_LETTER.match(line)
        if m and m.group(1).upper() in valid:
            offset = completion.rfind(line) + m.start(1)
            return Extraction(m.group(1).upper(), offset, offset + 1)

    if choices:
        lowered = completion.lower()
        best: tuple[int, str] | None = None
        for i, choice in enumerate(choices):
            text = str(choice).strip().lower()
            if len(text) >= 3 and text in lowered:
                pos = lowered.rfind(text)
                if best is None or pos > best[0]:
                    best = (pos, chr(ord("A") + i))
        if best is not None:
            return Extraction(best[1], best[0], best[0] + 1)

    for line in reversed(lines):
        for m in reversed(list(_STANDALONE_LETTER.finditer(line))):
            letter = m.group(1).upper()
            if letter in valid:
                offset = completion.rfind(line) + m.start(1)
                return Extraction(letter, offset, offset + 1)
    return Extraction(None)
