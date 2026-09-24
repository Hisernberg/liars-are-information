"""Byzantine strategies.

Every symbolic attacker here is *gold-aware* (it knows the answer key) and most
are *rushing* (they see the honest broadcasts of the current task before
speaking). The defender knows neither the key nor who the attackers are.

The zoo is organised by what each strategy is designed to exploit:

==================  ==============================================================
``coherent``        the original AIP threat: one shared lie per task (f-bloc).
``independent``     every attacker lies on its own; never coherent.
``gate_aware``      shares the lie with probability ``p``, else lies alone; ``p`` is
                    swept so the bloc can park under a published coherence ceiling.
``partial``         lies (coherently) with probability ``e``, else tells the truth.
``uninformative``   answers uniformly at random, independent of the truth: the
                    minimax strategy against any consistent channel estimator.
``echo``            copies a random honest agent (conditionally independent of the
                    truth given the honest answers): the exact minimiser in
                    Theorem 1 -- adds no information but duplicates evidence.
``attractor``       always endorses the honest agents' most common *wrong* answer
                    (the MX pipeline's L1 "attractor-rider").
``camouflage``      tells the truth unless the honest vote is contested (plurality
                    share below ``theta``), then coherently backs the strongest wrong
                    honest answer: earns trust on easy tasks, spends it on pivotal
                    ones. Designed against RACE.
``sleeper``         replays an honest model on history, then lies coherently on
                    held-out tasks: violates the stationarity RACE relies on.
``llm:<prompt>``    replays answers an actual LLM wrote when *prompted to deceive*
                    (four prompts x two attacker models, from the artifact cache).
==================  ==============================================================
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

SYMBOLIC = (
    "coherent",
    "independent",
    "gate_aware",
    "partial",
    "uninformative",
    "echo",
    "attractor",
    "camouflage",
    "sleeper",
)


def _wrong_pool(
    gold: str,
    label_space: Sequence[str] | None,
    honest_answers: Sequence[str | None],
    is_correct,
) -> list[str]:
    if label_space:
        return [c for c in label_space if c != gold]
    pool = sorted({a for a in honest_answers if a is not None and not is_correct(a)})
    return pool or [f"{gold}__adv"]


def byzantine_answers(
    attack: str,
    param: float,
    *,
    gold: str,
    label_space: Sequence[str] | None,
    honest_answers: Sequence[str | None],
    n_byzantine: int,
    is_correct,
    rng: np.random.Generator,
    phase: str = "test",
    cover_answers: Sequence[str | None] | None = None,
    llm_answers: Sequence[str | None] | None = None,
) -> list[str | None]:
    """Answers for the ``n_byzantine`` attackers on one task.

    ``is_correct(answer) -> bool`` is the gold scorer (the attacker may use it).
    ``phase`` is ``history`` / ``validation`` / ``test`` (only ``sleeper`` reads it).
    ``cover_answers`` are the honest answers a sleeper impersonates on history.
    ``llm_answers`` are the cached LLM-attacker answers for ``llm:*`` attacks.
    """
    if n_byzantine == 0:
        return []
    pool = _wrong_pool(gold, label_space, honest_answers, is_correct)
    shared = pool[int(rng.integers(len(pool)))]

    def wrong() -> str:
        return pool[int(rng.integers(len(pool)))]

    if attack.startswith("llm:"):
        if llm_answers is None:
            raise ValueError("llm attacks need cached attacker answers")
        return [llm_answers[k % len(llm_answers)] for k in range(n_byzantine)]
    if attack == "coherent":
        return [shared] * n_byzantine
    if attack == "independent":
        return [wrong() for _ in range(n_byzantine)]
    if attack == "gate_aware":
        return [shared if rng.random() < param else wrong() for _ in range(n_byzantine)]
    if attack == "partial":
        return [shared if rng.random() < param else gold for _ in range(n_byzantine)]
    if attack == "uninformative":
        if label_space:
            return [label_space[int(rng.integers(len(label_space)))] for _ in range(n_byzantine)]
        cands = sorted({a for a in honest_answers if a is not None} | {f"{gold}__adv"})
        return [cands[int(rng.integers(len(cands)))] for _ in range(n_byzantine)]
    if attack == "echo":
        live = [a for a in honest_answers if a is not None]
        if not live:
            return [wrong() for _ in range(n_byzantine)]
        return [live[int(rng.integers(len(live)))] for _ in range(n_byzantine)]
    wrong_votes = Counter(a for a in honest_answers if a is not None and not is_correct(a))
    top_wrong = sorted(wrong_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if wrong_votes else shared
    if attack == "attractor":
        return [top_wrong] * n_byzantine
    if attack == "camouflage":
        votes = Counter(a for a in honest_answers if a is not None)
        total = sum(votes.values())
        share = max(votes.values()) / total if total else 0.0
        contested = share < param
        return [top_wrong if contested else gold] * n_byzantine
    if attack == "sleeper":
        if phase == "history":
            if cover_answers is None:
                raise ValueError("sleeper needs cover answers on history")
            return [cover_answers[k % len(cover_answers)] for k in range(n_byzantine)]
        return [shared] * n_byzantine
    raise ValueError(f"unknown attack {attack!r}")
