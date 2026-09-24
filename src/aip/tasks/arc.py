"""ARC-Challenge (allenai/ai2_arc, ``ARC-Challenge``, test split): grade-school science.

The commonsense-reasoning slot, and the benchmark SAC-style filter-and-refine
baselines were built on, so including it is what makes the baseline comparison a
comparison rather than an extrapolation.

**Restricted to the four-option items**, which is 1165 of the 1172 test
questions. Four have three options and three have five. That is not tidiness:
``configs/inversion_thresholds.yaml`` calibrates the ``multiple_choice`` ceiling
at C=4 and says explicitly that for other option counts the chance level moves to
1/(C-1) and the ceiling must be re-measured rather than rescaled. Mixing 3- and
5-option items into a class calibrated at 4 would silently violate that, so the
seven odd items are dropped and the drop is recorded here.

**Label style is not uniform upstream.** 1144 items label their options
``A B C D`` while 21 label them ``1 2 3 4``, with ``answerKey`` following suit.
Both are normalised to letters by *position*, so the gold answer always names the
same option text it named upstream. A study that skipped this would mis-score
those 21 items rather than fail on them.
"""

from __future__ import annotations

from aip.harness.logging import get_logger
from aip.tasks.base import Benchmark
from aip.tasks.mmlu import extract_mc_answer, format_choices
from aip.types import AnswerKind, Extraction, TaskItem

log = get_logger(__name__)

#: The option count this benchmark is restricted to, matching the C=4 class the
#: inversion ceilings are calibrated on.
N_OPTIONS = 4

SYSTEM_PROMPT = (
    "You are a knowledgeable expert answering multiple-choice science questions. "
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


class ARC(Benchmark):
    name = "arc"
    answer_kind = AnswerKind.MULTIPLE_CHOICE
    hf_path = "allenai/ai2_arc"
    hf_config = "ARC-Challenge"
    hf_split = "test"

    system_prompt = SYSTEM_PROMPT
    user_template = USER_TEMPLATE
    confidence_system_prompt = CONFIDENCE_SYSTEM_PROMPT
    confidence_template = CONFIDENCE_TEMPLATE

    def _load_raw(self) -> list[TaskItem]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_path, self.hf_config, split=self.hf_split)
        items: list[TaskItem] = []
        skipped = 0
        for rec in ds:
            labels = [str(x) for x in rec["choices"]["label"]]
            texts = tuple(str(t) for t in rec["choices"]["text"])
            if len(texts) != N_OPTIONS:
                skipped += 1
                continue
            key = str(rec["answerKey"]).strip()
            if key not in labels:
                skipped += 1
                continue
            # Normalise by POSITION: whatever the upstream label style, option k
            # becomes the k-th letter, and the gold follows the same mapping.
            gold = chr(ord("A") + labels.index(key))
            items.append(
                TaskItem(
                    task_id=f"arc-test-{str(rec['id'])}",
                    benchmark=self.name,
                    question=str(rec["question"]).strip(),
                    gold_answer=gold,
                    answer_kind=self.answer_kind,
                    choices=texts,
                    metadata={"source_id": str(rec["id"]), "label_style": "".join(labels)},
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
