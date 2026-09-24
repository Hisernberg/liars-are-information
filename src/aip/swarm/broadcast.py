"""Building what each agent puts on the wire, and what each agent hears.

Honest agents replay their cached Phase A answer -- this is the cache-once
principle in action, and it is what makes a 20,000-cell sweep affordable.
Byzantine agents in Phase C are *symbolic*: their answers are generated here
rather than by a model. Phase D replaces them with real LLM output.

The adversary's coordination probability is an experimental variable. A
**coherent** adversary emits the same wrong answer as every other adversary on
a given task. This can supply negative evidence if the channel is identified
correctly. A **noise** adversary independently draws from the wrong-answer
candidate pool; those draws can coincide, especially when the pool is small.
The sweep runs both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aip.swarm.assignment import SwarmAssignment
from aip.swarm.topology import Topology
from aip.types import Broadcast, Observation


@dataclass(frozen=True, slots=True)
class AdversaryConfig:
    """Symbolic adversary. ``kind`` selects the behaviour, ``error_rate`` its strength."""

    kind: str = "always_wrong"
    error_rate: float = 1.0
    """Wrong-answer probability for the non-gate-aware modes.

    ``gate_aware`` always emits a wrong answer. An error probability alone
    does not determine multiclass channel informativeness or coherence.
    """
    confidence: float = 0.99
    """Self-reported confidence the adversary claims, and the logprob confidence
    it fakes. Phase C's symbolic adversary asserts high confidence in both
    channels; Phase D's falsified_confidence attack separates them."""
    coherence: float = 1.0
    """For ``kind="gate_aware"``: probability each Byzantine agent follows the
    bloc's shared lie rather than lying independently.

    This is the knob a threshold-evading adversary would actually turn. AIP
    inverts a channel only when its measured coherence exceeds a calibrated
    honest-q ceiling, and that ceiling is published in
    ``configs/inversion_thresholds.yaml`` -- so an adversary who reads the
    defence can aim just below it: coordinated enough to move the vote, too
    incoherent to be inverted.

    Conditional on a task's uniform pool of K wrong candidates, the expected
    agreement of two Byzantine agents is ``q = p**2 + (1-p**2)/K``. Independent
    draws include the shared lie, so both mixed and independent branches can
    collide. For a closed space K=C-1; for an open space K is the number of
    distinct wrong honest answers, or one for the synthetic fallback. Open
    spaces therefore do not imply zero collision probability. This theoretical
    Byzantine-pair expectation is not the receiver-conditioned gate statistic.

    Historical gate-aware result files used an incorrect theoretical q axis;
    their attack probability p and sampled broadcasts remain the original ones.

    Every result in this study before this adversary assumed the attacker did not
    know AIP existed. This is the assumption removed."""


def _wrong_answer(
    gold: str,
    label_space: Sequence[str],
    honest_answers: Sequence[str | None],
    rng: np.random.Generator,
) -> str:
    """Pick a single wrong answer that all coherent adversaries will share.

    Preference order matters for realism. A closed label space gives a genuine
    alternative option. Otherwise a wrong answer some honest model actually
    produced is far more plausible -- and far harder for an aggregator -- than a
    synthetic token that could be spotted by its shape alone. Only if neither
    exists is a synthetic value minted.
    """
    if label_space:
        options = [c for c in label_space if c != gold]
        if options:
            return str(options[int(rng.integers(len(options)))])
    plausible = sorted({a for a in honest_answers if a is not None and a != gold})
    if plausible:
        return str(plausible[int(rng.integers(len(plausible)))])
    return f"{gold}__adv"


def build_broadcasts(
    task_id: str,
    gold: str,
    assignment: SwarmAssignment,
    honest_answers: dict[str, tuple[str | None, float, float]],
    label_space: Sequence[str],
    adversary: AdversaryConfig,
    rng: np.random.Generator,
) -> tuple[Broadcast, ...]:
    """One task's broadcasts from every agent.

    ``honest_answers`` maps a model name to its cached
    ``(answer, logprob_confidence, self_reported_confidence)``.
    """
    replayed = [honest_answers[m][0] for m in assignment.models if m is not None]
    shared_lie = _wrong_answer(gold, label_space, replayed, rng)

    out: list[Broadcast] = []
    for agent in range(assignment.n_agents):
        if agent in assignment.byzantine:
            if adversary.kind == "noise":
                # Incoherent: an independent draw per adversary, so nothing to invert.
                answer = _wrong_answer(gold, label_space, replayed, rng)
                if rng.random() >= adversary.error_rate:
                    answer = gold
            elif adversary.kind == "gate_aware":
                # Partial coordination, tuned. With probability `coherence` this
                # agent joins the bloc's shared lie; otherwise it lies on its own.
                # It never emits the gold answer -- an evading adversary has no
                # reason to be correct, only to be hard to characterise.
                answer = (
                    shared_lie
                    if rng.random() < adversary.coherence
                    else _wrong_answer(gold, label_space, replayed, rng)
                )
            else:
                answer = shared_lie if rng.random() < adversary.error_rate else gold
            out.append(
                Broadcast(
                    agent_id=agent,
                    task_id=task_id,
                    answer=answer,
                    logprob_confidence=adversary.confidence,
                    self_reported_confidence=adversary.confidence,
                    is_byzantine=True,
                    source_model=None,
                    attack=adversary.kind,
                )
            )
            continue

        model = assignment.models[agent]
        answer, logprob_conf, self_conf = honest_answers[model]
        out.append(
            Broadcast(
                agent_id=agent,
                task_id=task_id,
                answer=answer,
                logprob_confidence=float(logprob_conf),
                self_reported_confidence=float(self_conf),
                is_byzantine=False,
                source_model=model,
                attack=None,
            )
        )
    return tuple(out)


def observe(
    broadcasts: Sequence[Broadcast],
    topology: Topology,
    p_obs: float,
    rng: np.random.Generator,
) -> tuple[Observation, ...]:
    """Thin each agent's neighbourhood by ``p_obs`` and build its observation.

    An agent always observes itself. That self-vote is not a modelling
    convenience -- it is the mass that inflates trust-weighted schemes, and
    holding its share fixed across methods is the normalization-parity control.
    """
    observations = []
    for agent in range(topology.n_agents):
        heard = [broadcasts[agent]]
        for peer in sorted(topology.neighbours[agent]):
            if p_obs >= 1.0 or rng.random() < p_obs:
                heard.append(broadcasts[peer])
        observations.append(
            Observation(
                self_id=agent,
                task_id=broadcasts[agent].task_id,
                broadcasts=tuple(sorted(heard, key=lambda b: b.agent_id)),
            )
        )
    return tuple(observations)
