"""Adaptive adversary: an epsilon-greedy bandit choosing how to lie each epoch.

**Offline, no new inference.** Each epoch the adversary picks one of two arms --
``coherent_lie`` (replay the cached semantic_negation outputs) or ``noise``
(incoherent wrong answers) -- and receives as reward the *drop* in swarm accuracy
it caused. It is playing against the defence, so what it learns is a statement
about the defence.

This is the deterrence experiment. Against a discard-style aggregator, coherent
lying is strictly better: a coherent bloc captures the plurality. Against AIP,
coherence is what makes a channel invertible, so the same arm should become
*self-defeating* and the bandit should collapse onto noise. An adversary forced
into noise has been deterred rather than merely filtered, and noise is the weakest
attack available -- that is the claim this logs the evidence for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aip.attacks.base import Attack

ARMS = ("coherent_lie", "noise")


@dataclass
class BanditLog:
    """Per-epoch record, kept because the arm trajectory is a paper figure."""

    epoch: list[int] = field(default_factory=list)
    arm: list[str] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    swarm_accuracy: list[float] = field(default_factory=list)
    exploring: list[bool] = field(default_factory=list)
    value_coherent: list[float] = field(default_factory=list)
    value_noise: list[float] = field(default_factory=list)

    def to_records(self, defence: str) -> list[dict]:
        return [
            {
                "defence": defence,
                "epoch": e,
                "arm": a,
                "reward": r,
                "swarm_accuracy": s,
                "exploring": x,
                "value_coherent_lie": vc,
                "value_noise": vn,
            }
            for e, a, r, s, x, vc, vn in zip(
                self.epoch,
                self.arm,
                self.reward,
                self.swarm_accuracy,
                self.exploring,
                self.value_coherent,
                self.value_noise,
                strict=True,
            )
        ]


class EpsilonGreedyAdversary:
    """Epsilon-greedy over {coherent_lie, noise}, maximising swarm damage."""

    def __init__(self, epsilon: float = 0.15, seed: int = 0, optimistic: float = 1.0) -> None:
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        # Optimistic initialisation so both arms are tried before the greedy
        # policy locks in; without it a single unlucky first epoch can decide
        # everything.
        self.values = dict.fromkeys(ARMS, float(optimistic))
        self.counts = dict.fromkeys(ARMS, 0)
        self.log = BanditLog()

    def select(self) -> tuple[str, bool]:
        if self.rng.random() < self.epsilon:
            return str(self.rng.choice(ARMS)), True
        best = max(self.values.values())
        tied = [a for a in ARMS if np.isclose(self.values[a], best)]
        return str(self.rng.choice(tied)) if len(tied) > 1 else tied[0], False

    def update(self, arm: str, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def record(self, epoch: int, arm: str, reward: float, accuracy: float, exploring: bool) -> None:
        self.log.epoch.append(epoch)
        self.log.arm.append(arm)
        self.log.reward.append(float(reward))
        self.log.swarm_accuracy.append(float(accuracy))
        self.log.exploring.append(bool(exploring))
        self.log.value_coherent.append(float(self.values["coherent_lie"]))
        self.log.value_noise.append(float(self.values["noise"]))


ATTACK = Attack(
    name="adaptive_bandit",
    needs_inference=False,
    description="Epsilon-greedy bandit choosing coherent-lie vs noise per epoch.",
    build_messages=None,
    expected_invertible=True,
    notes="Deterrence test: does exploitation push the adversary onto noise?",
    metadata={"arms": list(ARMS), "default_epsilon": 0.15},
)
