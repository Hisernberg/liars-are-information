"""Fault injection for MX: two coherent liars, both offline transforms.

Neither variant needs inference. They rewrite one replica's cached answer, the
same discipline as ``falsified_confidence`` and ``burst`` in the main study --
which is what makes the comparison clean: the honest replicas' answers are
byte-identical between the no-fault and faulted runs, so any difference in the
stage output is attributable to the injected replica and to nothing else.

**Threat model.** L1 is an INFORMED adversary: it reads the honest replicas'
answers on the same task before choosing its own. L2 is UNINFORMED: a fixed lie
chosen without reference to what the honest replicas said. Proposition 1
distinguishes exactly these two, and MX-H4 is the pipeline-scale test -- an
uninformed lie sits above the honest coherence ceiling and the gate fires on it;
an informed one can sit inside the ceiling where no function of per-question
coherence separates it from honest coincidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class LiarResult:
    answer: str
    fallback: bool


def attractor_rider(honest_answers: list[str | None], gold: str,
                    label_space: list[str], rng) -> LiarResult:
    """L1: endorse the honest replicas' most common WRONG answer.

    Where the honest replicas produced no shared wrong answer, there is no
    attractor to ride and the liar falls back to a uniform wrong choice. The
    fallback is reported, never hidden: the rate at which L1 degenerates into an
    uninformed liar bounds how much of any L1 result is actually about
    attractor-riding.
    """
    wrong = Counter(a for a in honest_answers if a is not None and a != gold)
    if wrong:
        top = max(wrong.values())
        # Deterministic tie-break, so the liar does not depend on replica order.
        return LiarResult(sorted(a for a, v in wrong.items() if v == top)[0], False)
    return LiarResult(uniform_wrong(gold, label_space, rng), True)


def fixed_lie(gold: str, label_space: list[str]) -> LiarResult:
    """L2: a designated wrong answer, constant across replicas and tasks.

    The rule is 'the first label that is not the gold answer', which is fixed
    given the label space and does not consult the honest replicas at all.
    """
    for lab in label_space:
        if lab != gold:
            return LiarResult(lab, False)
    raise ValueError("label space has no wrong answer")


def uniform_wrong(gold: str, label_space: list[str], rng) -> str:
    options = [lab for lab in label_space if lab != gold]
    if not options:
        raise ValueError("label space has no wrong answer")
    return str(rng.choice(options))


def inject(answers: list[str | None], slot: int, variant: str, gold: str,
           label_space: list[str], rng) -> tuple[list[str | None], LiarResult]:
    """Replace ``answers[slot]`` with the liar's answer.

    The liar reads only the *other* replicas -- an adversary does not observe
    the honest answer of the agent it replaced, because that agent did not run.
    """
    honest = [a for i, a in enumerate(answers) if i != slot]
    if variant == "L1":
        res = attractor_rider(honest, gold, label_space, rng)
    elif variant == "L2":
        res = fixed_lie(gold, label_space)
    else:
        raise ValueError(f"unknown liar variant {variant!r}")
    out = list(answers)
    out[slot] = res.answer
    return out, res
