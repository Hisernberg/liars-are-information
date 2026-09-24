"""The aggregator interface, plus the normalization-parity control.

Every method implements ``aggregate(observations, self_id) -> answer``, where
``observations`` are the broadcasts one agent actually received on one task.
Methods that need cross-task statistics (Dawid-Skene, SAC, AIP) declare
``needs_fit`` and get a pass over the whole task stream first -- which is
realistic, since an agent in a repeated setting accumulates history.

**Normalization parity.** Trust- and confidence-weighted schemes quietly differ
in how much weight an agent puts on its own answer. That self-vote mass is a
confound: a method can look strong simply because it leans on itself more, not
because its weighting of *peers* is better. :func:`apply_self_vote_parity`
rescales weights so the self-vote takes a fixed share of the total, letting every
weighted method be compared at matched share. Each result row records the parity
condition it ran under, and the sweep runs both matched and unmatched so the
confound's size is itself a reported number.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aip.types import Broadcast, Observation


@dataclass(frozen=True, slots=True)
class ParityConfig:
    """How much of the total weight the self-vote may take.

    ``share is None`` means unmatched: each method uses its native weighting.
    A float in (0, 1) rescales so the self-vote takes exactly that share.
    """

    share: float | None = None

    @property
    def label(self) -> str:
        return "unmatched" if self.share is None else f"matched@{self.share:g}"


def apply_self_vote_parity(
    weights: dict[int, float], self_id: int, parity: ParityConfig
) -> dict[int, float]:
    """Rescale so the self-vote holds ``parity.share`` of the total weight.

    With no peers the share is unachievable and weights pass through unchanged --
    an agent alone is entirely its own vote, and forcing a share there would mean
    dividing by zero rather than making a comparison fair.
    """
    if parity.share is None or self_id not in weights:
        return weights
    share = float(parity.share)
    peers = {k: v for k, v in weights.items() if k != self_id}
    peer_total = sum(abs(v) for v in peers.values())
    if peer_total <= 0 or not (0.0 < share < 1.0):
        return weights
    out = dict(peers)
    out[self_id] = share / (1.0 - share) * peer_total
    return out


def observed_labels(broadcasts: Sequence[Broadcast]) -> list[str]:
    """Sorted distinct non-missing answers -- the effective label space on a task."""
    return sorted({b.answer for b in broadcasts if b.answer is not None})


def onehot_matrix(broadcasts: Sequence[Broadcast], labels: Sequence[str]) -> np.ndarray:
    """Embed answers as one-hot rows so vector-space defences apply.

    A missing answer becomes the zero vector: an abstention that pulls toward no
    label, rather than a vote for an arbitrary one.
    """
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(broadcasts), len(labels)), dtype=float)
    for row, b in enumerate(broadcasts):
        if b.answer is not None and b.answer in index:
            matrix[row, index[b.answer]] = 1.0
    return matrix


def decode(scores: np.ndarray, labels: Sequence[str]) -> str | None:
    """Argmax decode with deterministic tie-breaking by label order."""
    if len(labels) == 0 or scores.size == 0:
        return None
    best = float(np.max(scores))
    if not np.isfinite(best):
        return None
    tied = [labels[i] for i in range(len(labels)) if np.isclose(scores[i], best)]
    return sorted(tied)[0] if tied else None


class Aggregator(ABC):
    """Common interface for every aggregation method in the sweep."""

    name: str = "base"
    #: True if the method needs a pass over the task stream before aggregating.
    needs_fit: bool = False
    #: True if the method is given information a deployed agent would not have
    #: (the true Byzantine fraction, anchor reliabilities). Phase E's
    #: minimax-regret table over deployable methods excludes these.
    is_oracle: bool = False

    def fit(self, tasks: Sequence[Sequence[Observation]]) -> None:  # noqa: B027
        """Optional pre-pass over all tasks, per receiving agent."""
        return None

    @abstractmethod
    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        """Return this agent's final answer for one task."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"
