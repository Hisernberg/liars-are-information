"""MATH-500 (HuggingFaceH4/MATH-500): competition math with boxed answers."""

from __future__ import annotations

import re

from aip.harness.logging import get_logger
from aip.tasks.base import Benchmark
from aip.types import AnswerKind, Extraction, TaskItem

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an expert competition mathematician. Solve the problem step by "
    "step, then state the final answer in a LaTeX box."
)

# Literal braces are doubled: this string is rendered with str.format.
USER_TEMPLATE = (
    "{question}\n\n"
    "Reason step by step. Then give the final answer on its own last line in "
    "exactly this format:\n"
    "\\boxed{{<answer>}}\n"
    "Put only the final answer inside the box, in simplest form, with no units "
    "and no surrounding words."
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

_ANSWER_IS = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\**\s*(.+)", re.IGNORECASE)
_SAFE_FOR_SYMPY = re.compile(r"^[0-9a-zA-Z+\-*/^().,\[\]{}\\ _]*$")


class MATH500(Benchmark):
    name = "math500"
    answer_kind = AnswerKind.OPEN
    hf_path = "HuggingFaceH4/MATH-500"
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
        for i, rec in enumerate(ds):
            items.append(
                TaskItem(
                    task_id=f"math500-{rec.get('unique_id', i)}",
                    benchmark=self.name,
                    question=rec["problem"].strip(),
                    gold_answer=str(rec["answer"]).strip(),
                    answer_kind=self.answer_kind,
                    metadata={
                        "subject": rec.get("subject"),
                        "level": rec.get("level"),
                        "solution": rec.get("solution"),
                        "index": i,
                    },
                )
            )
        log.info("dataset.loaded", benchmark=self.name, n=len(items))
        return items

    def extract_answer(self, completion: str) -> Extraction:
        return extract_math_answer(completion)

    def normalize(self, answer: str) -> str:
        return normalize_latex(answer)

    def score(self, extracted: str | None, gold: str) -> bool:
        return math_match(extracted, gold)


def find_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """Find every ``\\boxed{...}`` / ``\\fbox{...}`` with correct brace matching.

    A regex cannot do this: MATH answers nest braces (``\\boxed{\\frac{1}{2}}``).
    Returns ``(content_start, content_end, content)`` triples in document order,
    so the caller keeps the character span needed for logprob confidence.
    """
    spans: list[tuple[int, int, str]] = []
    for command in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(command, start)
            if idx == -1:
                break
            brace = idx + len(command)
            while brace < len(text) and text[brace] in " \t":
                brace += 1
            if brace >= len(text) or text[brace] != "{":
                start = idx + len(command)
                continue
            depth = 0
            for j in range(brace, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        spans.append((brace + 1, j, text[brace + 1 : j]))
                        break
            start = brace + 1
    spans.sort(key=lambda s: s[0])
    return spans


def extract_math_answer(completion: str) -> Extraction:
    """Extract the final boxed answer, falling back to "the answer is ..."."""
    if not completion:
        return Extraction(None)

    spans = find_boxed_spans(completion)
    if spans:
        start, end, content = spans[-1]
        value = content.strip()
        if value:
            return Extraction(value, start, end)

    matches = list(_ANSWER_IS.finditer(completion))
    if matches:
        m = matches[-1]
        tail = m.group(1).strip()
        tail = tail.split("\n")[0].strip().rstrip(".").strip("$ ")
        if tail:
            offset = m.start(1) + m.group(1).find(tail)
            return Extraction(tail, offset, offset + len(tail))

    # Last non-empty line, as a last resort.
    for line in reversed(completion.strip().splitlines()):
        stripped = line.strip().strip("$ ").rstrip(".")
        if stripped:
            offset = completion.rfind(stripped)
            return Extraction(stripped, offset, offset + len(stripped))
    return Extraction(None)


def _unwrap_text_commands(s: str) -> str:
    """Replace ``\\text{foo}`` / ``\\mbox{foo}`` with ``foo`` (brace-matched)."""
    for command in ("\\text", "\\mbox", "\\textbf", "\\mathrm", "\\mathbf"):
        while True:
            idx = s.find(command + "{")
            if idx == -1:
                break
            brace = idx + len(command)
            depth = 0
            end = None
            for j in range(brace, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
            if end is None:
                break
            s = s[:idx] + s[brace + 1 : end] + s[end + 1 :]
    return s


def normalize_latex(answer: str) -> str:
    """Canonicalize a LaTeX answer for exact match.

    Handles the usual MATH-500 surface variation: ``\\dfrac`` vs ``\\frac``,
    ``\\frac12`` vs ``\\frac{1}{2}``, spacing macros, ``\\left``/``\\right``,
    degree and percent marks, ``x = 5`` vs ``5``, ``.5`` vs ``0.5``, and
    ``5.0`` vs ``5``.
    """
    if answer is None:
        return ""
    s = str(answer).strip()
    if not s:
        return ""

    s = s.replace("$", "").replace("\\$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = _unwrap_text_commands(s)
    for macro in ("\\!", "\\,", "\\;", "\\:", "\\ ", "~", "\\quad", "\\qquad"):
        s = s.replace(macro, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("^{\\circ}", "").replace("\\degree", "")
    s = re.sub(r"\\mbox\{.*?\}", "", s)

    # \frac12 -> \frac{1}{2}; \sqrt3 -> \sqrt{3}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)

    s = s.replace(" ", "")

    # Strip a single-variable left-hand side: "x=5" -> "5".
    m = re.match(r"^[a-zA-Z]\w*=(.+)$", s)
    if m:
        s = m.group(1)

    # Drop a fully-enclosing brace pair.
    while s.startswith("{") and s.endswith("}"):
        depth = 0
        enclosing = True
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    enclosing = False
                    break
        if not enclosing:
            break
        s = s[1:-1]

    s = s.rstrip(".")
    s = s.replace(",", "") if re.fullmatch(r"-?[\d,]+(\.\d+)?", s) else s

    if re.fullmatch(r"-?\.\d+", s):
        s = "0" + s if not s.startswith("-") else "-0" + s[1:]
    if re.fullmatch(r"-?\d+\.\d*0+", s) or re.fullmatch(r"-?\d+\.0*", s):
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", "-"):
        s = "0" if s == "-0" else s

    return s.lower()


def _latex_to_sympy(s: str) -> str:
    """Best-effort LaTeX -> sympy-parsable string for simple expressions."""
    out = s
    for _ in range(6):  # nested fractions
        new = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", out)
        if new == out:
            break
        out = new
    out = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", out)
    out = out.replace("\\pi", "pi").replace("\\infty", "oo")
    out = out.replace("^", "**")
    out = out.replace("\\", "")
    return out


def symbolic_equal(a: str, b: str) -> bool:
    """Symbolic equivalence for simple algebraic answers; ``False`` if unsure.

    Only attempted on short, safely-charactered strings -- this is a scoring
    convenience, not a CAS, and it must never hang a 100-task sweep.
    """
    if not a or not b or len(a) > 60 or len(b) > 60:
        return False
    if not (_SAFE_FOR_SYMPY.match(a) and _SAFE_FOR_SYMPY.match(b)):
        return False
    try:
        import sympy

        pa = sympy.sympify(_latex_to_sympy(a), rational=True)
        pb = sympy.sympify(_latex_to_sympy(b), rational=True)
        if pa.free_symbols or pb.free_symbols:
            return bool(sympy.simplify(pa - pb) == 0)
        return bool(abs(complex(pa) - complex(pb)) < 1e-9)
    except Exception:  # noqa: BLE001 - sympy raises a wide variety of errors
        return False


def math_match(extracted: str | None, gold: str) -> bool:
    """Normalized string match, with a symbolic fallback."""
    if extracted is None:
        return False
    na, nb = normalize_latex(extracted), normalize_latex(gold)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return symbolic_equal(na, nb)
