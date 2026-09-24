"""LAACS-style reputation decay: the defence a sleeper adversary is built to beat.

Reputation systems are the standard answer to Byzantine peers in multi-agent
literature, and they are the right baseline for E1 precisely because their
failure mode is *structural* rather than incidental. A receiver scores each peer
by how often it has agreed with the swarm plurality, decays that score
exponentially so old evidence fades, and weights votes by the result.

The structure that matters: **reputation is earned before it is spent.** An
adversary that behaves honestly for k rounds accumulates the same standing as an
honest peer, and when it defects the decay is what governs how fast that standing
is lost. With decay ``lambda``, evidence from ``n`` rounds ago carries weight
``lambda**n``, so the half-life is ``ln(2)/ln(1/lambda)`` rounds -- and for the
first half-life after a defection the adversary is still being trusted on the
strength of its cover story. That is not a bug in the implementation; it is the
property E1 exists to measure, and it is why AgentChain-style systems list
long-term infiltration as an admitted limitation.

Contrast with AIP's gate, which conditions on *coherence with other dissenters*
rather than on accumulated agreement. A sleeper's honest phase buys it no
coherence credit, because coherence is recomputed from the current window rather
than integrated over history. The prediction under test in E1 is therefore that
reputation decay carries its trust across the betrayal and AIP-windowed does not.
Whether that holds is measured, not assumed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aip.aggregation.base import Aggregator, ParityConfig
from aip.aggregation.baselines import MajorityVote, _weighted_vote, rank_normalize
from aip.types import Broadcast, Observation


class ReputationDecay(Aggregator):
    """Exponentially decayed agreement reputation, per receiver, per peer.

    ``decay`` is the per-round multiplier on prior evidence. 1.0 would never
    forget (and could never recover from a defection); the default 0.9 gives a
    half-life of about 6.6 rounds, which is fast enough to react and slow enough
    that a sleeper's accumulated standing is worth something.

    ``floor_weight`` keeps a disgraced peer at a small non-zero weight rather
    than dropping it outright. That is deliberate: a reputation system that
    zeroes a peer can never observe it recovering, which would make
    rounds-to-recover unmeasurable by construction.
    """

    name = "reputation_decay"
    needs_fit = True

    def __init__(
        self,
        decay: float = 0.9,
        floor_weight: float = 0.05,
        parity: ParityConfig | None = None,
    ) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = float(decay)
        self.floor_weight = float(floor_weight)
        self.parity = parity or ParityConfig()
        #: (receiver, task_id) -> {peer: reputation in [0, 1]}
        self._reputation: dict[tuple[int, str], dict[int, float]] = {}
        #: Final standing, for reporting.
        self._final: dict[int, dict[int, float]] = {}

    @property
    def half_life_rounds(self) -> float:
        """Rounds for prior evidence to lose half its weight."""
        if self.decay >= 1.0:
            return float("inf")
        return float(np.log(2.0) / np.log(1.0 / self.decay))

    def fit(self, tasks: Sequence[Sequence[Observation]]) -> None:
        """Walk the task stream IN ORDER, updating reputation as it goes.

        Order is the whole point. A batch statistic over the same rounds would
        score a sleeper by its lifetime average and miss the defection entirely;
        what makes reputation interesting here is that it is a running quantity
        with state carried across the betrayal boundary.
        """
        # numerator / denominator of the decayed agreement rate, per (recv, peer)
        num: dict[int, dict[int, float]] = {}
        den: dict[int, dict[int, float]] = {}

        for observations in tasks:
            for obs in observations:
                plurality = MajorityVote().aggregate(obs.broadcasts, obs.self_id)
                n = num.setdefault(obs.self_id, {})
                d = den.setdefault(obs.self_id, {})
                snapshot: dict[int, float] = {}
                for b in obs.broadcasts:
                    agreed = 1.0 if (b.answer is not None and b.answer == plurality) else 0.0
                    n[b.agent_id] = n.get(b.agent_id, 0.0) * self.decay + agreed
                    d[b.agent_id] = d.get(b.agent_id, 0.0) * self.decay + 1.0
                    snapshot[b.agent_id] = n[b.agent_id] / d[b.agent_id] if d[b.agent_id] else 0.5
                # Record the reputation as it stood when this task was decided,
                # not as it ended up. Using the final value would let the
                # defender see the future, which is the whole question in E1.
                self._reputation[(obs.self_id, obs.broadcasts[0].task_id)] = snapshot

        for recv in num:
            self._final[recv] = {
                p: (num[recv][p] / den[recv][p] if den[recv][p] else 0.5) for p in num[recv]
            }

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        if not observations:
            return None
        task_id = observations[0].task_id
        rep = self._reputation.get((self_id, task_id)) or self._final.get(self_id) or {}
        weights = {
            b.agent_id: max(self.floor_weight, float(rep.get(b.agent_id, 0.5)))
            for b in observations
        }
        weights[self_id] = max(weights.get(self_id, 0.5), 1.0)  # a receiver trusts itself
        return _weighted_vote(observations, rank_normalize(weights), self_id, self.parity)
