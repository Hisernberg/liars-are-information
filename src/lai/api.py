"""TrustLayer: RACE as a drop-in component for any multi-agent LLM framework.

Each honest agent owns one ``TrustLayer``. After every round it calls
:meth:`TrustLayer.observe` with the answers it heard (its own included) and
asks :meth:`TrustLayer.decide` for its pooled answer. No labels, model names,
confidences or identities are ever needed, only who said what.

    layer = TrustLayer("planner", label_space=("A", "B", "C", "D"))
    for question in stream:
        answers = {name: agent.answer(question) for name, agent in swarm.items()}
        decision = layer.decide(answers)        # uses only earlier questions
        layer.observe(question.id, answers)     # then commits this one
        print(decision.answer, decision.explain())

Decisions are strictly causal: ``decide`` only uses questions previously passed
to ``observe``. ``forgetting`` (0 < gamma < 1) down-weights old questions so
the layer tracks attackers that change behaviour (sleepers); ``None`` keeps the
full history.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from aip.types import Broadcast
from lai.race import RACEAggregator


@dataclass
class PeerTrust:
    peer: Hashable
    accuracy: float
    """Label-free estimate of P(peer's answer = truth)."""
    weight: float
    """Log-odds contributed by one answer (positive: trust, negative: invert)."""
    decision: str
    """TRUST, DISCARD or INVERT."""
    group: int
    """Peers sharing a group are treated as one source (replicas, blocs)."""
    n_observed: int


@dataclass
class Decision:
    answer: str | None
    posterior: dict[str, float]
    peers: list[PeerTrust] = field(default_factory=list)
    n_history: int = 0

    def explain(self, top: int = 3) -> str:
        if not self.peers:
            return f"{self.answer} (no history yet: falling back to my own answer)"
        inverted = [p for p in self.peers if p.decision == "INVERT"]
        trusted = [p for p in self.peers if p.decision == "TRUST"]
        best = sorted(self.posterior.items(), key=lambda kv: -kv[1])[:top]
        parts = [f"answer {self.answer!r} after {self.n_history} unlabeled questions",
                 "posterior " + ", ".join(f"{k}={v:.2f}" for k, v in best),
                 f"trusting {len(trusted)} peer(s), inverting {len(inverted)}"]
        if inverted:
            parts.append("inverted: " + ", ".join(f"{p.peer} (â={p.accuracy:.2f})" for p in inverted[:5]))
        return "; ".join(parts)


class TrustLayer:
    """Receiver-anchored, label-free trust for one agent in a multi-agent system."""

    def __init__(
        self,
        agent_id: Hashable,
        label_space: Sequence[str] | None = None,
        *,
        forgetting: float | None = None,
        window: int | None = None,
        condition: str = "all",
        **race_kwargs,
    ) -> None:
        if forgetting is not None and not 0.0 < forgetting < 1.0:
            raise ValueError("forgetting must be in (0, 1) or None")
        self.agent_id = agent_id
        self.label_space = tuple(label_space) if label_space else None
        self.forgetting = forgetting
        self.window = window
        self._race_kwargs = dict(race_kwargs, condition=condition)
        self._ids: dict[Hashable, int] = {agent_id: 0}
        self._names: list[Hashable] = [agent_id]
        self._rows: list[dict[int, str | None]] = []
        self._question_ids: list[Hashable] = []
        self._model: RACEAggregator | None = None
        self._fitted_on = -1

    # ------------------------------------------------------------ plumbing

    def _index(self, agent: Hashable) -> int:
        if agent not in self._ids:
            self._ids[agent] = len(self._names)
            self._names.append(agent)
        return self._ids[agent]

    def _row(self, answers: Mapping[Hashable, str | None]) -> dict[int, str | None]:
        if self.agent_id not in answers:
            raise ValueError("the observing agent's own answer must be included")
        return {self._index(a): (None if v is None else str(v)) for a, v in answers.items()}

    def _broadcasts(self, row: dict[int, str | None]) -> tuple[Broadcast, ...]:
        return tuple(Broadcast(j, "q", a, 0.5, 0.5, False) for j, a in sorted(row.items()))

    def _fit(self) -> RACEAggregator | None:
        if not self._rows:
            return None
        if self._fitted_on == len(self._rows) and self._model is not None:
            return self._model
        rows = self._rows[-self.window:] if self.window else self._rows
        weights = None
        if self.forgetting is not None:
            weights = self.forgetting ** np.arange(len(rows) - 1, -1, -1, dtype=float)
        model = RACEAggregator(self.label_space, **self._race_kwargs)
        model.fit_receiver(0, rows, weights)
        self._model, self._fitted_on = model, len(self._rows)
        return model

    # ------------------------------------------------------------ public

    def observe(self, question_id: Hashable, answers: Mapping[Hashable, str | None]) -> None:
        """Commit one question's answers (the observing agent's own included) to history."""
        self._rows.append(self._row(answers))
        self._question_ids.append(question_id)

    def decide(self, answers: Mapping[Hashable, str | None]) -> Decision:
        """Pooled answer for the current question, using only committed history."""
        row = self._row(answers)
        model = self._fit()
        own = row.get(0)
        if model is None:
            return Decision(own, {own: 1.0} if own is not None else {}, [], 0)
        broadcasts = self._broadcasts(row)
        posterior = model.posterior(broadcasts, 0)
        answer = model.aggregate(broadcasts, 0)
        return Decision(answer, posterior, self.trust(), len(self._rows))

    def trust(self) -> list[PeerTrust]:
        """Current per-peer estimates, most distrusted first."""
        model = self._fit()
        if model is None:
            return []
        out = []
        for idx, est in model.diagnostics.channels.get(0, {}).items():
            if idx == 0:
                continue
            out.append(PeerTrust(self._names[idx], est.accuracy, est.weight_at_chance_k, est.decision.upper(),
                                 est.group, est.n_observed))
        return sorted(out, key=lambda p: p.weight)

    def __len__(self) -> int:
        return len(self._rows)
