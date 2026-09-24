"""History-only evidence caps for dependent AIP channels.

The existing static AIP gate is fitted without alteration. This extension then
caps each learned peer group's total aggregation weight at its strongest member
before self-vote parity. Identical reports do not establish common provenance;
grouping is an operational dependence heuristic, not a Byzantine classifier.
Duplicates can still change the original gate's fitted statistics and its
multiple-testing threshold, so the complete estimator is not clone invariant.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from aip.aggregation.aip import DISCARD, INVERT, TRUST, AIPAggregator
from aip.aggregation.base import apply_self_vote_parity, observed_labels
from aip.types import Broadcast, Observation


@dataclass(frozen=True)
class PairSupport:
    left: int
    right: int
    co_observed: int
    agreements: int
    qualifies: bool


def learn_groups(
    observations: Sequence[Observation],
    *,
    agreement_threshold: float = 0.98,
    min_support: int = 20,
) -> tuple[tuple[tuple[int, ...], ...], tuple[PairSupport, ...]]:
    """Deterministic complete-link groups using aligned nonmissing reports.

    Every pair inside a group must meet both criteria. This avoids the
    transitive-chain failure where A resembles B and B resembles C although A
    and C disagree. Greedy group placement follows agent-ID order and is not
    guaranteed to recover a unique latent source partition.
    """
    if not 0 < agreement_threshold <= 1:
        raise ValueError("agreement_threshold must be in (0, 1]")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    rows: list[dict[int, str | None]] = []
    seen: set[str] = set()
    receiver = observations[0].self_id if observations else None
    for observation in sorted(observations, key=lambda value: value.task_id):
        if observation.self_id != receiver:
            raise ValueError("grouping must use one receiver's observations")
        if observation.task_id in seen:
            raise ValueError("duplicate training task")
        seen.add(observation.task_id)
        row: dict[int, str | None] = {}
        for broadcast in observation.broadcasts:
            if broadcast.task_id != observation.task_id or broadcast.agent_id in row:
                raise ValueError("mixed task or duplicated peer in an observation")
            row[broadcast.agent_id] = broadcast.answer
        rows.append(row)
    agents = sorted({agent for row in rows for agent in row})
    compatible: dict[tuple[int, int], bool] = {}
    support: list[PairSupport] = []
    for i, left in enumerate(agents):
        for right in agents[i + 1 :]:
            pairs = [(row.get(left), row.get(right)) for row in rows]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            agreements = sum(a == b for a, b in pairs)
            qualifies = len(pairs) >= min_support and agreements >= agreement_threshold * len(pairs)
            compatible[(left, right)] = qualifies
            support.append(PairSupport(left, right, len(pairs), agreements, qualifies))
    groups: list[list[int]] = []
    for agent in agents:
        for group in groups:
            if all(compatible[(min(other, agent), max(other, agent))] for other in group):
                group.append(agent)
                break
        else:
            groups.append([agent])
    return tuple(tuple(group) for group in groups), tuple(support)


def cap_group_weights(
    weights: dict[int, float], self_id: int, groups: Sequence[Sequence[int]]
) -> dict[int, float]:
    """Cap present peer mass; self-like peers add no second self channel.

    The self weight is preserved before parity. A group containing self has its
    other members zeroed, since the trusted self channel already represents it.
    Unseen/group-less peers are singleton channels. This cap concerns weights;
    fixed gate decisions and identical current reports are additionally needed
    for a fixed-statistics clone-invariance statement.
    """
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("weights must be finite and nonnegative")
    result = dict(weights)
    assigned: set[int] = set()
    for group in groups:
        if len(set(group)) != len(group) or assigned.intersection(group):
            raise ValueError("groups must form a disjoint partition")
        assigned.update(group)
        peers = [agent for agent in group if agent in result and agent != self_id]
        if self_id in group:
            for agent in peers:
                result[agent] = 0.0
            continue
        total = sum(result[agent] for agent in peers)
        strongest = max((result[agent] for agent in peers), default=0.0)
        if total > 0:
            for agent in peers:
                result[agent] *= strongest / total
    return result


class CloneAwareAIP(AIPAggregator):
    """Static AIP plus history-derived group caps; no privileged source labels."""

    def __init__(
        self,
        *args,
        agreement_threshold: float = 0.98,
        min_support: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.window:
            raise ValueError("CloneAwareAIP supports the static gate only")
        self._configure_groups(agreement_threshold, min_support)
        self.name = f"clone_cap_{self.name}"

    def _configure_groups(self, threshold: float, support: int) -> None:
        learn_groups([], agreement_threshold=threshold, min_support=support)
        self.agreement_threshold = threshold
        self.group_min_support = support
        self.groups_: dict[int, tuple[tuple[int, ...], ...]] = {}
        self.pair_support_: dict[int, tuple[PairSupport, ...]] = {}
        self.training_task_ids_: dict[int, tuple[str, ...]] = {}

    def _fit_groups(self, tasks: Sequence[Sequence[Observation]]) -> None:
        by_receiver: dict[int, list[Observation]] = {}
        for observations in tasks:
            for observation in observations:
                by_receiver.setdefault(observation.self_id, []).append(observation)
        self.groups_.clear()
        self.pair_support_.clear()
        self.training_task_ids_.clear()
        for receiver, history in by_receiver.items():
            self.groups_[receiver], self.pair_support_[receiver] = learn_groups(
                history,
                agreement_threshold=self.agreement_threshold,
                min_support=self.group_min_support,
            )
            self.training_task_ids_[receiver] = tuple(sorted(obs.task_id for obs in history))

    def fit(self, tasks: Sequence[Sequence[Observation]]) -> None:
        super().fit(tasks)
        self._fit_groups(tasks)

    @classmethod
    def from_fitted(
        cls,
        base: AIPAggregator,
        history: Sequence[Sequence[Observation]],
        *,
        agreement_threshold: float = 0.98,
        min_support: int = 20,
    ) -> CloneAwareAIP:
        """Reuse an identical static baseline fit to isolate the pooling change.

        ``history`` must be the same receiver history used for ``base.fit``.
        Copying the fitted state avoids redundant gate estimation in paired
        experiments; no validation/test observations may be supplied here.
        """
        if not isinstance(base, AIPAggregator) or base.window or not base._stats:
            raise ValueError("a fitted static AIP baseline is required")
        instance = cls.__new__(cls)
        instance.__dict__ = deepcopy(base.__dict__)
        instance._configure_groups(agreement_threshold, min_support)
        instance.name = f"clone_cap_{base.name}"
        instance._fit_groups(history)
        return instance

    def weight_diagnostics(self, observations: Sequence[Broadcast], self_id: int) -> dict:
        stats = self._stats.get(self_id, {})
        before = {
            b.agent_id: float(stats[b.agent_id].weight) if b.agent_id in stats else 0.0
            for b in observations
        }
        groups = self.groups_.get(self_id, ())
        capped = cap_group_weights(before, self_id, groups)
        return {
            "groups": groups,
            "before_cap": before,
            "after_cap": capped,
            "after_parity": apply_self_vote_parity(capped, self_id, self.parity),
        }

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        labels = self.label_space or observed_labels(observations)
        if not labels:
            return None
        weights = self.weight_diagnostics(observations, self_id)["after_parity"]
        stats = self._stats.get(self_id, {})
        scores = dict.fromkeys(labels, 0.0)
        for broadcast in observations:
            answer = broadcast.answer
            if answer is None or answer not in scores:
                continue
            channel = stats.get(broadcast.agent_id)
            decision = (
                TRUST
                if broadcast.agent_id == self_id
                else (channel.decision if channel else DISCARD)
            )
            weight = weights.get(broadcast.agent_id, 0.0)
            if weight <= 0 or decision == DISCARD:
                continue
            if decision == TRUST:
                scores[answer] += weight
            elif decision == INVERT and len(labels) > 1:
                scores[answer] -= weight
                for label in labels:
                    if label != answer:
                        scores[label] += weight / (len(labels) - 1)
        maximum = max(scores.values())
        return sorted(label for label, value in scores.items() if np.isclose(value, maximum))[0]
