"""Attack classes. Four need inference; three are offline transforms."""

from __future__ import annotations

from aip.attacks import (
    adaptive_bandit,
    always_wrong,
    burst,
    falsified_confidence,
    rushing,
    semantic_hallucination,
    semantic_negation,
)
from aip.attacks.base import Attack, AttackContext

ATTACKS: dict[str, Attack] = {
    a.ATTACK.name: a.ATTACK
    for a in (
        always_wrong,
        semantic_negation,
        semantic_hallucination,
        rushing,
        falsified_confidence,
        burst,
        adaptive_bandit,
    )
}

INFERENCE_ATTACKS = [name for name, a in ATTACKS.items() if a.needs_inference]
OFFLINE_ATTACKS = [name for name, a in ATTACKS.items() if not a.needs_inference]


def get_attack(name: str) -> Attack:
    if name not in ATTACKS:
        raise KeyError(f"unknown attack {name!r}; known: {sorted(ATTACKS)}")
    return ATTACKS[name]


__all__ = [
    "ATTACKS",
    "INFERENCE_ATTACKS",
    "OFFLINE_ATTACKS",
    "Attack",
    "AttackContext",
    "get_attack",
]
