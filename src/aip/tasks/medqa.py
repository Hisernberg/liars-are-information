"""MedQA (GBaker/MedQA-USMLE-4-options, test split): USMLE clinical vignettes.

The domain slot, and the benchmark AgentChain reports on, so the cost-of-defence
comparison in Phase 6 is anchored to a task both systems have actually run.

Four options throughout -- all 1273 test items -- so it sits in the same C=4
``multiple_choice`` calibration class as MMLU and ARC with no restriction needed.

Why a medical benchmark earns its place in a Byzantine-robustness study: the
questions are long clinical vignettes where a wrong answer is typically a
*plausible* differential rather than a random distractor. That is the regime
where honest models are most likely to share a wrong-answer attractor, which is
precisely the confound the calibrated honest-q ceiling exists to handle. If
inversion is going to misfire on honest agreement anywhere, it should misfire
here.
"""

from __future__ import annotations

from aip.harness.logging import get_logger
from aip.tasks.base import Benchmark
from aip.tasks.mmlu import extract_mc_answer, format_choices
from aip.types import AnswerKind, Extraction, TaskItem

log = get_logger(__name__)

N_OPTIONS = 4

SYSTEM_PROMPT = (
    "You are a medical expert answering USMLE-style multiple-choice questions. "
    "You think briefly, then commit to exactly one option letter."
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


class MedQA(Benchmark):
    name = "medqa"
    answer_kind = AnswerKind.MULTIPLE_CHOICE
    hf_path = "GBaker/MedQA-USMLE-4-options"
    hf_config = None
    hf_split = "test"

    system_prompt = SYSTEM_PROMPT
    user_template = USER_TEMPLATE
    confidence_system_prompt = CONFIDENCE_SYSTEM_PROMPT
    confidence_template = CONFIDENCE_TEMPLATE

    def _load_raw(self) -> list[TaskItem]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_path, split=self.hf_split)
        items: list[TaskItem] = []
        skipped = 0
        for i, rec in enumerate(ds):
            options = rec["options"]
            # `options` is a mapping letter -> text; order by letter so option k
            # is always the same option across every run.
            letters = sorted(str(k) for k in options)
            if len(letters) != N_OPTIONS:
                skipped += 1
                continue
            key = str(rec["answer_idx"]).strip().upper()
            if key not in letters:
                skipped += 1
                continue
            items.append(
                TaskItem(
                    task_id=f"medqa-test-{i:05d}",
                    benchmark=self.name,
                    question=str(rec["question"]).strip(),
                    gold_answer=chr(ord("A") + letters.index(key)),
                    answer_kind=self.answer_kind,
                    choices=tuple(str(options[k]) for k in letters),
                    metadata={"meta_info": rec.get("meta_info"), "index": i},
                )
            )
        log.info("dataset.loaded", benchmark=self.name, n=len(items), skipped=skipped)
        return items

    def _format_fields(self, item: TaskItem) -> dict[str, object]:
        return {
            "question": item.question,
            "choices_block": format_choices(item.choices),
            "letters": ", ".join(item.label_space),
        }

    def extract_answer(self, completion: str) -> Extraction:
        return extract_mc_answer(completion, n_choices=N_OPTIONS)

    def normalize(self, answer: str) -> str:
        return str(answer).strip().upper()[:1] if answer else ""

    def score(self, extracted: str | None, gold: str) -> bool:
        if not extracted:
            return False
        return self.normalize(extracted) == self.normalize(gold)
