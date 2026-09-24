"""MX arms: how a stage's k replica answers become one stage output.

An arm is a (replica roster, aggregation rule) pair. P1/P2 share a roster and
differ only in the rule, as do P3/P4 -- which is why aggregation is offline and
the rule contrast costs no GPU time.

The aggregator also contributes its own independent answer to the stage task.
For plurality that answer is one vote among k+1. For AIP it is the **receiver
anchor**: the gate's two tests are computed against it rather than against the
plurality, which is the property that keeps them meaningful when the plurality
is adversarial.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from aip.aggregation.aip import DISCARD, INVERT, TRUST
from aip.types import Broadcast

# INVERT / TRUST / DISCARD are re-exported from the aggregator rather than
# redefined. A parallel vocabulary here would drift the moment the gate renamed
# one, and the telemetry would keep reporting the old name.
__all__ = ["INVERT", "TRUST", "DISCARD", "Arm", "StageDecision", "build_arms",
           "plurality", "to_broadcasts"]


@dataclass(frozen=True)
class Arm:
    """One experimental arm."""

    name: str
    #: Model name per replica slot. A same-model arm repeats one name.
    replicas: tuple[str, ...]
    #: Model that produces the receiver anchor.
    anchor_model: str
    rule: str  # "plurality" | "aip" | "none"

    @property
    def k(self) -> int:
        return len(self.replicas)

    @property
    def is_cross_family(self) -> bool:
        return len(set(self.replicas)) > 1


@dataclass
class StageDecision:
    """Telemetry for one stage of one task under one arm."""

    task_id: str
    stage: str
    arm: str
    output: str | None
    anchor: str | None
    replica_answers: tuple[str | None, ...]
    replica_models: tuple[str, ...]
    #: Per-replica gate decision; empty for plurality arms.
    decisions: tuple[str, ...] = ()
    liar_slot: int | None = None
    liar_variant: str | None = None
    liar_answer: str | None = None
    #: True when L1 could not find an attractor and fell back to a uniform lie.
    liar_fallback: bool = False
    extra: dict = field(default_factory=dict)

    def honest_slots(self) -> tuple[int, ...]:
        return tuple(i for i in range(len(self.replica_answers)) if i != self.liar_slot)

    def honest_inverts(self) -> int:
        """INVERT decisions landing on an honest channel -- the false-inversion count."""
        if not self.decisions:
            return 0
        return sum(1 for i in self.honest_slots()
                   if i < len(self.decisions) and self.decisions[i] == INVERT)


def plurality(answers: list[str | None], anchor: str | None,
              label_space: list[str]) -> str | None:
    """Plurality over replicas plus the aggregator's own answer.

    Ties break toward the lexicographically first label rather than toward
    insertion order, so the result cannot depend on which replica happened to be
    listed first.
    """
    votes: Counter[str] = Counter(a for a in [*answers, anchor] if a is not None)
    if not votes:
        return None
    top = max(votes.values())
    return sorted(a for a, v in votes.items() if v == top)[0]


def to_broadcasts(answers: list[str | None], models: tuple[str, ...],
                  anchor: str | None, task_id: str,
                  confidences: list[float] | None = None) -> list[Broadcast]:
    """Wrap stage replica answers as Broadcasts for the AIP aggregator.

    Slot ``len(answers)`` is the aggregator's own answer, i.e. the receiver. The
    AIP aggregator identifies the receiver by ``self_id``, so the anchor must be
    present in the list and must be the id passed as ``self_id``.
    """
    conf = confidences or [0.5] * len(answers)
    out = [
        Broadcast(agent_id=i, task_id=task_id, answer=a,
                  logprob_confidence=float(conf[i]), self_reported_confidence=0.5,
                  is_byzantine=False, source_model=models[i])
        for i, a in enumerate(answers)
    ]
    out.append(Broadcast(agent_id=len(answers), task_id=task_id, answer=anchor,
                         logprob_confidence=0.5, self_reported_confidence=0.5,
                         is_byzantine=False, source_model="anchor"))
    return out


def build_arms(same_model: str, cross_family: tuple[str, str, str],
               anchor_model: str) -> dict[str, Arm]:
    """The five pre-registered arms.

    ``anchor_model`` produces the receiver anchor in every arm so that the
    anchor is not itself a source of between-arm variation.
    """
    return {
        "P0": Arm("P0", (same_model,), anchor_model, "none"),
        "P1": Arm("P1", (same_model,) * 3, anchor_model, "plurality"),
        "P2": Arm("P2", (same_model,) * 3, anchor_model, "aip"),
        "P3": Arm("P3", cross_family, anchor_model, "aip"),
        "P4": Arm("P4", cross_family, anchor_model, "plurality"),
    }
