"""GSM8K (main config, test split): grade-school word problems, numeric answers."""

from __future__ import annotations

import re

from aip.harness.logging import get_logger
from aip.tasks.base import Benchmark
from aip.types import AnswerKind, Extraction, TaskItem

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem step by step, then state "
    "the final numeric answer."
)

USER_TEMPLATE = (
    "{question}\n\n"
    "Reason step by step. Then give the final answer on its own last line in "
    "exactly this format:\n"
    "#### <number>\n"
    "The number must be a plain integer or decimal with no units, no commas, and "
    "no explanatory text."
)

CONFIDENCE_SYSTEM_PROMPT = (
    "You judge how likely an answer is to be correct. You reply with a single "
    "integer and nothing else."
)

CONFIDENCE_TEMPLATE = (
    "Problem:\n{question}\n\n"
    "Proposed final answer: {answer}\n\n"
    "How confident are you that this answer is correct? Reply with a single "
    "integer from 0 to 100 and nothing else."
)

# ``#### 42``, ``#### -1,234.5``, ``#### $42``
_HASH_ANSWER = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
_ANSWER_IS = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)",
    re.IGNORECASE,
)
_BOXED = re.compile(r"\\boxed\{\s*\$?\s*(-?[\d,]*\.?\d+)\s*\}")
_ANY_NUMBER = re.compile(r"-?[\d,]*\.?\d+")


class GSM8K(Benchmark):
    name = "gsm8k"
    answer_kind = AnswerKind.OPEN
    hf_path = "openai/gsm8k"
    hf_config = "main"
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
            gold_raw = rec["answer"].split("####")[-1].strip()
            items.append(
                TaskItem(
                    task_id=f"gsm8k-test-{i:05d}",
                    benchmark=self.name,
                    question=rec["question"].strip(),
                    gold_answer=normalize_number(gold_raw),
                    answer_kind=self.answer_kind,
                    metadata={"rationale": rec["answer"].strip(), "index": i},
                )
            )
        log.info("dataset.loaded", benchmark=self.name, n=len(items))
        return items

    def extract_answer(self, completion: str) -> Extraction:
        return extract_gsm8k_answer(completion)

    def normalize(self, answer: str) -> str:
        return normalize_number(answer)

    def score(self, extracted: str | None, gold: str) -> bool:
        return numeric_match(extracted, gold)


def extract_gsm8k_answer(completion: str) -> Extraction:
    """Extract the final numeric answer, preferring the requested ``####`` form.

    Fallback order is deliberate: the instructed format first, then explicit
    "the answer is" phrasing, then ``\\boxed{}`` (some instruct models default to
    it), and only then the last number anywhere in the text.  The last-number
    fallback is what keeps a verbose-but-correct completion from being scored
    wrong, but it is the weakest signal, so it comes last.
    """
    if not completion:
        return Extraction(None)

    for pattern in (_HASH_ANSWER, _ANSWER_IS, _BOXED):
        matches = list(pattern.finditer(completion))
        if matches:
            m = matches[-1]
            value = normalize_number(m.group(1))
            if value:
                return Extraction(value, m.start(1), m.end(1))

    matches = list(_ANY_NUMBER.finditer(completion))
    for m in reversed(matches):
        value = normalize_number(m.group(0))
        if value:
            return Extraction(value, m.start(0), m.end(0))
    return Extraction(None)


def normalize_number(text: str) -> str:
    """Canonicalize a numeric string: strip separators/units, drop trailing zeros.

    ``"1,234"`` -> ``"1234"``, ``"42.0"`` -> ``"42"``, ``"$5.50"`` -> ``"5.5"``,
    ``"-0"`` -> ``"0"``.  Returns ``""`` when nothing numeric survives, which
    callers treat as extraction failure.
    """
    if text is None:
        return ""
    s = str(text).strip()
    s = s.replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    s = s.rstrip(".")
    if not s:
        return ""
    try:
        value = float(s)
    except ValueError:
        return ""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    # Trim float noise without changing magnitude.
    return repr(round(value, 6)).rstrip("0").rstrip(".")


def numeric_match(extracted: str | None, gold: str, tol: float = 1e-6) -> bool:
    """Compare two numeric answers with a relative tolerance."""
    if extracted is None:
        return False
    a, b = normalize_number(extracted), normalize_number(gold)
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    return abs(fa - fb) <= tol * max(1.0, abs(fb))
