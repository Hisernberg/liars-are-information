"""BoolQ (google/boolq, validation split): passage-grounded yes/no questions.

**This is the theory's home turf.** BoolQ is the only benchmark here with a
binary answer space, and $C = 2$ is exactly where the attenuation law is not an
approximation: with one true answer and one false one, two agents who are both
wrong are wrong *identically*, so $q_{ij} = 1$ holds by construction rather than
by measurement (``docs/attenuation_law.md`` section 3). Everything the multiclass
law has to estimate, the binary law knows. Phase 2 uses it as the exact-law
control: the binary reduction must hold here within binomial noise, and if it
does not, the estimator is wrong rather than the data being hard.

It is also where inversion is *provably* uninformative -- the calibrated ceiling
for the ``binary`` class is 1.0, so the q-gate can never fire and AIP must fall
back to trust-only weighting. That is a prediction, not a limitation, and BoolQ
is the benchmark that tests it.

The split is ``validation``: BoolQ's test split ships without labels, and this
study scores every answer.

Modelled as a two-option multiple choice (``A. Yes`` / ``B. No``) rather than as
free text. That is deliberate reuse, not a workaround: the closed-label-space
machinery -- Dawid-Skene, the multiclass decode, and the attack classes that
enumerate ``item.label_space`` -- then operates unchanged, and the answer space
stays exactly the two labels the theory is about.
"""

from __future__ import annotations

import re

import numpy as np

from aip.harness.logging import get_logger
from aip.tasks.base import Benchmark
from aip.tasks.mmlu import extract_mc_answer, format_choices
from aip.types import AnswerKind, Extraction, TaskItem

log = get_logger(__name__)

#: A. Yes / B. No, fixed. The order is part of the prompt hash, so it must never
#: be permuted between runs.
CHOICES: tuple[str, ...] = ("Yes", "No")

SYSTEM_PROMPT = (
    "You answer yes/no questions about a passage. You read the passage, think "
    "briefly, then commit to exactly one option letter."
)

USER_TEMPLATE = (
    "Passage:\n{passage}\n\n"
    "Question: {question}\n\n"
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
    "Passage:\n{passage}\n\n"
    "Question: {question}\n\n"
    "{choices_block}\n\n"
    "Proposed answer: {answer}\n\n"
    "How confident are you that this answer is correct? Reply with a single "
    "integer from 0 to 100 and nothing else."
)


#: An instructed answer line naming the word rather than the letter.
_ANSWER_WORD_YES = re.compile(
    r"answer\s*(?:is|:|=)\s*\**\s*\(?\s*(yes|true)\b", re.IGNORECASE
)
_ANSWER_WORD_NO = re.compile(r"answer\s*(?:is|:|=)\s*\**\s*\(?\s*(no|false)\b", re.IGNORECASE)

#: A bare word anywhere, used only after every instructed form has failed.
_BARE_YES = re.compile(r"\b(yes|true)\b", re.IGNORECASE)
_BARE_NO = re.compile(r"\b(no|false)\b", re.IGNORECASE)


class BoolQ(Benchmark):
    name = "boolq"
    answer_kind = AnswerKind.MULTIPLE_CHOICE
    hf_path = "google/boolq"
    hf_config = None
    hf_split = "validation"

    system_prompt = SYSTEM_PROMPT
    user_template = USER_TEMPLATE
    confidence_system_prompt = CONFIDENCE_SYSTEM_PROMPT
    confidence_template = CONFIDENCE_TEMPLATE

    def _load_raw(self) -> list[TaskItem]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_path, split=self.hf_split)
        items: list[TaskItem] = []
        for i, rec in enumerate(ds):
            answer_yes = bool(rec["answer"])
            items.append(
                TaskItem(
                    task_id=f"boolq-validation-{i:05d}",
                    benchmark=self.name,
                    question=str(rec["question"]).strip(),
                    gold_answer="A" if answer_yes else "B",
                    answer_kind=self.answer_kind,
                    choices=CHOICES,
                    metadata={
                        "passage": str(rec["passage"]).strip(),
                        "answer_bool": answer_yes,
                        "index": i,
                    },
                )
            )
        log.info("dataset.loaded", benchmark=self.name, n=len(items))
        return items

    def subsample(self, items: list[TaskItem], n: int, seed: int) -> list[TaskItem]:
        """Label-balanced draw: equal Yes and No, rather than the natural 62/38.

        BoolQ's validation split is 62.2% Yes, so a uniform draw would hand a
        constant "Yes" responder 62% accuracy and make every swarm number on this
        benchmark hard to read against chance. Balancing puts chance at exactly
        50%, which is also the base rate the binary attenuation law assumes. The
        cost -- that this is not the benchmark's natural distribution -- is
        recorded in the frozen task list's ``selection_rule``.
        """
        rng = np.random.default_rng(seed)
        pools: dict[bool, list[int]] = {True: [], False: []}
        for idx, item in enumerate(items):
            pools[bool(item.metadata["answer_bool"])].append(idx)

        chosen: list[int] = []
        per_class = n // 2
        for label in (True, False):
            pool = pools[label]
            take = min(per_class, len(pool))
            picked = rng.choice(len(pool), size=take, replace=False)
            chosen.extend(pool[int(i)] for i in picked)
        # An odd n leaves one slot; fill it from whatever remains, deterministically.
        if len(chosen) < n:
            rest = [i for i in range(len(items)) if i not in set(chosen)]
            extra = rng.choice(len(rest), size=min(n - len(chosen), len(rest)), replace=False)
            chosen.extend(rest[int(i)] for i in extra)

        log.info(
            "boolq.balanced",
            n=len(chosen),
            n_yes=sum(1 for i in chosen if items[i].metadata["answer_bool"]),
            seed=seed,
        )
        return [items[i] for i in sorted(chosen)]

    def _format_fields(self, item: TaskItem) -> dict[str, object]:
        return {
            "question": item.question,
            "passage": item.metadata.get("passage", ""),
            "choices_block": format_choices(item.choices),
            "letters": ", ".join(item.label_space),
        }

    def extract_answer(self, completion: str) -> Extraction:
        """Letter or word, whichever the model actually produced.

        The shared multiple-choice extractor cannot carry this benchmark alone.
        Its verbatim-choice fallback only matches options of three characters or
        more -- a rule that stops two-letter distractors matching noise anywhere
        in a long completion -- so on a Yes/No space it recovers ``Yes`` and
        silently misses ``No``. That asymmetry would not have looked like a bug
        in the aggregate: it would have looked like every model having a Yes
        bias, on the one benchmark whose whole purpose is a clean binary answer
        space.

        So the word forms are matched here explicitly and on word boundaries,
        before delegating to the shared letter extractor. The returned span
        covers the word or letter itself, because that span is what restricts the
        logprob confidence to the answer tokens.
        """
        if not completion:
            return Extraction(None)

        # Latest instructed answer wins, across BOTH words: a reasoning model
        # that writes "Answer: Yes" and then revises to "Answer: No" must score
        # as No. Checking the words in a fixed order instead would return
        # whichever word was checked first, not whichever the model committed to.
        instructed: list[tuple[int, str, int, int]] = []
        for pattern, letter in ((_ANSWER_WORD_YES, "A"), (_ANSWER_WORD_NO, "B")):
            for m in pattern.finditer(completion):
                instructed.append((m.start(1), letter, m.start(1), m.end(1)))
        if instructed:
            _, letter, start, end = max(instructed)
            return Extraction(letter, start, end)

        letter_hit = extract_mc_answer(completion, n_choices=2)
        if letter_hit.ok:
            return letter_hit

        # Last resort: a bare Yes/No anywhere, latest wins.
        best: tuple[int, str, int, int] | None = None
        for pattern, letter in ((_BARE_YES, "A"), (_BARE_NO, "B")):
            for m in pattern.finditer(completion):
                if best is None or m.start(1) > best[0]:
                    best = (m.start(1), letter, m.start(1), m.end(1))
        if best is not None:
            return Extraction(best[1], best[2], best[3])
        return Extraction(None)

    def normalize(self, answer: str) -> str:
        """Canonicalise to A/B, accepting a bare yes/no as well as a letter.

        A model told to answer with a letter often answers with the word anyway;
        scoring that as an extraction failure would understate accuracy and,
        worse, would do so unevenly across models.
        """
        if not answer:
            return ""
        text = str(answer).strip().upper()
        if text.startswith("YES") or text.startswith("TRUE"):
            return "A"
        if text.startswith("NO") or text.startswith("FALSE"):
            return "B"
        return text[:1]

    def score(self, extracted: str | None, gold: str) -> bool:
        if not extracted:
            return False
        return self.normalize(extracted) == self.normalize(gold)
