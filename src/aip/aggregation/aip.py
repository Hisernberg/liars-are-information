"""AIP: signed informativeness weighting with threshold-gated inversion.

The claim under test is that a *coherent* adversary is not noise to be discarded
but a channel to be inverted: if a bloc reliably says X when the truth is not X,
"this bloc says X" is evidence against X. Prior art discards; AIP pools the
inverted signal.

**Why the gate exists.** Phase B2 measured that honest models share
wrong-answer attractors -- on MMLU, four models from four families pick the same
distractor 47.6% of the time against a 33.3% chance rate, CI entirely above
chance. So coherence alone is *not* evidence of adversarial coordination. A rule
that inverts on any above-chance coherence would invert honest agents on exactly
the questions they find hardest, which is the worst available failure mode.
Every channel therefore gets a three-way decision against a calibrated ceiling
(``configs/inversion_thresholds.yaml``):

===============  =========================================================
INVERT           coherence above the honest-q ceiling for this answer space
TRUST            coherence within the honest band and error estimate < 1/2
DISCARD          otherwise
===============  =========================================================

``AIPNaive`` inverts on any above-chance coherence and exists precisely to
demonstrate what the ceiling buys; ``AIPTrustOnly`` never inverts and isolates
how much of AIP's performance comes from inversion rather than from weighting.

**Why estimation is anchored on the receiver.** A receiver knows one thing no
label-free estimator can otherwise establish: *it is itself honest*. That matters
because at ``f > 0.5`` a coherent adversarial bloc *is* the plurality, and any
estimator that treats consensus as a proxy for truth -- plurality voting,
Dawid-Skene, spectral methods -- will converge on the lie and then classify the
honest minority as the coherent outliers. It would invert exactly the agents
worth keeping. Anchoring coherence on disagreement-with-self sidesteps that
inversion of roles entirely, and it is what lets AIP stay meaningful past the 50%
ceiling that the discard literature reports.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import binom

from aip.aggregation.base import (
    Aggregator,
    ParityConfig,
    apply_self_vote_parity,
    observed_labels,
)
from aip.aggregation.correlation import blind_marginal_error_rates
from aip.types import Broadcast, Observation

DEFAULT_THRESHOLDS = Path("configs/inversion_thresholds.yaml")

TRUST, INVERT, DISCARD = "trust", "invert", "discard"


@dataclass(frozen=True, slots=True)
class InversionThresholds:
    """Calibrated honest-q ceilings, loaded from config. Never tuned per run."""

    ceilings: dict[str, float]
    chances: dict[str, float]
    benchmark_classes: dict[str, str]
    max_blind_error_rate: float = 0.5
    min_observations: int = 5

    @classmethod
    def load(cls, path: Path | str = DEFAULT_THRESHOLDS) -> InversionThresholds:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            ceilings={k: float(v["ceiling"]) for k, v in raw["classes"].items()},
            chances={k: float(v["chance"]) for k, v in raw["classes"].items()},
            benchmark_classes=dict(raw["benchmark_classes"]),
            max_blind_error_rate=float(raw["trust"]["max_blind_error_rate"]),
            min_observations=int(raw["trust"]["min_observations"]),
        )

    def ceiling_for(self, benchmark: str) -> float:
        return self.ceilings[self.benchmark_classes[benchmark]]

    def min_support_for(self, benchmark: str, alpha: float) -> int:
        """Joint-dissent events needed before a channel is inversion-eligible.

        Derived, not chosen: the smallest ``n`` at which even the strongest
        possible evidence -- every joint dissent a coincidence -- clears the
        significance bar, i.e. the smallest ``n`` with ``q0**n <= alpha``. Below
        it no observation pattern is statistically distinguishable from honest
        behaviour, so inverting on one would be inverting on noise.

        For the calibrated ceilings this gives 10 events for 4-way multiple
        choice, 6 for open-numeric and 2 for open-text: the classes where honest
        agents coincide often need more evidence to overturn, which is exactly
        the right ordering.
        """
        q0 = self.ceiling_for(benchmark)
        if q0 >= 1.0 or q0 <= 0.0:
            return 10**9  # degenerate class: never eligible
        return max(1, int(np.ceil(np.log(alpha) / np.log(q0))))

    def chance_for(self, benchmark: str) -> float:
        return self.chances[self.benchmark_classes[benchmark]]


@dataclass
class ChannelStats:
    """What a receiver has learned about one peer."""

    n_observed: int = 0
    dissent_rate: float = 0.0
    agreement_with_self: float = 0.0
    expected_honest_agreement: float = float("nan")
    coherence: float = float("nan")
    coincidences: int = 0
    """Joint-dissent events on which the best-evidence partner coincided."""
    joint_dissents: int = 0
    """Joint-dissent events with that partner -- the test's sample size."""
    coherence_pvalue: float = float("nan")
    blind_error: float = float("nan")
    decision: str = DISCARD
    weight: float = 0.0


@dataclass
class AIPDiagnostics:
    """Per-receiver channel decisions, kept so the gate can be audited."""

    channels: dict[int, dict[int, ChannelStats]] = field(default_factory=dict)

    def decision_counts(self, honest: set[int]) -> dict[str, int]:
        counts = {TRUST: 0, INVERT: 0, DISCARD: 0}
        for peers in self.channels.values():
            for stats in peers.values():
                counts[stats.decision] += 1
        return counts

    def honest_inversions(self, byzantine: set[int]) -> tuple[int, int]:
        """(honest channels wrongly inverted, honest channels judged).

        Counts only what an **honest receiver** decided about an **honest peer**.
        Byzantine receivers must be excluded from both sides: an adversary's own
        aggregation is never scored, and its view of the swarm is inverted by
        construction, so including it reports the adversaries' opinions as if
        they were the defence's mistakes. Leaving them in made the gate look like
        it was inverting a third of all honest channels at f = 0.3 when the
        honest receivers had in fact inverted none of them.
        """
        inverted = total = 0
        for receiver, peers in self.channels.items():
            if receiver in byzantine:
                continue
            for peer, stats in peers.items():
                if peer == receiver or peer in byzantine:
                    continue
                total += 1
                inverted += int(stats.decision == INVERT)
        return inverted, total

    def adversary_inversions(self, byzantine: set[int]) -> tuple[int, int]:
        """(Byzantine channels correctly inverted, Byzantine channels judged).

        The complement of :meth:`honest_inversions`, again from honest receivers
        only. Together the two say whether the gate fires on the right channels.
        """
        inverted = total = 0
        for receiver, peers in self.channels.items():
            if receiver in byzantine:
                continue
            for peer, stats in peers.items():
                if peer == receiver or peer not in byzantine:
                    continue
                total += 1
                inverted += int(stats.decision == INVERT)
        return inverted, total


class AIPAggregator(Aggregator):
    """AIP with a configurable channel gate.

    ``mode`` is one of ``gated`` (threshold against the calibrated honest-q
    ceiling), ``naive`` (invert on any above-chance coherence), or ``trust_only``
    (never invert -- the ablation isolating the value of inversion).
    """

    needs_fit = True

    def __init__(
        self,
        benchmark: str,
        mode: str = "gated",
        thresholds: InversionThresholds | None = None,
        parity: ParityConfig | None = None,
        label_space: Sequence[str] | None = None,
        eb_prior_weight: float = 10.0,
        weight_clip: float | None = None,
        n_sigma: float = 3.0,
        coherence_test: str = "binomial",
        alpha: float = 0.05,
        n_peers_for_correction: int = 9,
        window: int | None = None,
        soft_gate: float | None = None,
        randomized_threshold: float | None = None,
        threshold_seed: int = 0,
    ) -> None:
        if mode not in (TRUST_ONLY := "trust_only", "gated", "naive"):
            raise ValueError(f"unknown AIP mode {mode!r}")
        self.benchmark = benchmark
        self.mode = mode
        self.thresholds = thresholds or InversionThresholds.load()
        self.parity = parity or ParityConfig()
        self.label_space = list(label_space) if label_space else None
        self.eb_prior_weight = float(eb_prior_weight)
        self.weight_clip = weight_clip
        self.n_sigma = float(n_sigma)
        if coherence_test not in ("binomial", "fixed_ceiling"):
            raise ValueError(f"unknown coherence_test {coherence_test!r}")
        self.coherence_test = coherence_test
        # Sliding-window channel statistics. None = static (pool all history),
        # which is what Phases C and D used. A window makes the gate track a
        # non-stationary adversary instead of averaging across its regimes.
        self.window = int(window) if window else None

        # -- M1: soft gate ------------------------------------------------
        # The hard ceiling is a cliff with a PUBLISHED location, which is exactly
        # what the gate-aware adversary parks under: below it the gate is silent,
        # above it inversion is decisive. `soft_gate` is the logistic steepness k
        # in sigma(k * (q/ceiling - 1)); the inversion weight then varies
        # continuously with coherence and there is no q at which behaviour
        # changes discontinuously. An adversary can still reduce its weight by
        # coordinating less, but it pays for that in coordination rather than
        # getting the reduction free at a known point.
        self.soft_gate = float(soft_gate) if soft_gate else None

        # -- M2: randomized threshold -------------------------------------
        # The ceiling is redrawn per fit from a private distribution of relative
        # width `randomized_threshold`. The attacker is assumed to know the
        # DISTRIBUTION -- it is published here -- but not the draw, so it cannot
        # park just below a number it can compute. This costs calibration
        # precision, which is the trade the experiment measures.
        self.randomized_threshold = (
            float(randomized_threshold) if randomized_threshold else None
        )
        self.threshold_seed = int(threshold_seed)
        self._ceiling_jitter = 1.0

        # Bonferroni over the channels one receiver judges: without it, testing
        # nine peers at 0.05 each puts the family-wise false-inversion rate near
        # 37%, and a false inversion is the expensive kind of mistake.
        self.alpha = float(alpha) / max(1, int(n_peers_for_correction))
        self.name = {"gated": "aip_gated", "naive": "aip_naive", TRUST_ONLY: "aip_trust_only"}[mode]
        if self.window:
            self.name = f"{self.name}_w{self.window}"
        # Mitigation tags go last so the base name and any window suffix are
        # already in place; a variant must be distinguishable in every table.
        tags = []
        if self.soft_gate:
            tags.append(f"soft{self.soft_gate:g}")
        if self.randomized_threshold:
            tags.append(f"rand{self.randomized_threshold:g}")
        if tags:
            self.name = self.name + "_" + "_".join(tags)
        self.diagnostics = AIPDiagnostics()
        self._stats: dict[int, dict[int, ChannelStats]] = {}
        #: (receiver, task_id) -> {peer: (decision, weight)}, only when windowed.
        self._task_decisions: dict[tuple[int, str], dict[int, tuple[str, float]]] = {}
        self.n_regime_flagged = 0

    # -- fit -------------------------------------------------------------

    def fit(self, tasks: Sequence[Sequence[Observation]]) -> None:
        """Accumulate receiver-anchored channel statistics over the task stream."""
        self._stats.clear()
        self._task_decisions.clear()
        self.n_regime_flagged = 0
        streams: dict[int, list[Observation]] = {}
        heard: dict[int, dict[int, list[tuple[str | None, str | None]]]] = {}
        for observations in tasks:
            for obs in observations:
                streams.setdefault(obs.self_id, []).append(obs)
                own = obs.own.answer
                bucket = heard.setdefault(obs.self_id, {})
                for b in obs.broadcasts:
                    bucket.setdefault(b.agent_id, []).append((b.answer, own))

        # Population mean dissent, for the empirical-Bayes shrinkage below.
        all_dissent = [
            float(np.mean([a != own for a, own in pairs]))
            for peers in heard.values()
            for peer, pairs in peers.items()
            for _ in (0,)
            if pairs
        ]
        population_dissent = float(np.mean(all_dissent)) if all_dissent else 0.5

        for receiver, peers in heard.items():
            # Missing observations must keep their task positions. Compacting
            # each peer independently compares answers to different questions.
            rows = [{b.agent_id: b.answer for b in obs.broadcasts} for obs in streams[receiver]]
            answers_by_peer = {peer: [row.get(peer) for row in rows] for peer in peers}
            own_answers = [obs.own.answer for obs in streams[receiver]]
            self._draw_ceiling_jitter(receiver)

            blind_errors = self._blind_errors(answers_by_peer, receiver)
            ceiling = self.thresholds.ceiling_for(self.benchmark)
            support_floor = (
                self.thresholds.min_support_for(self.benchmark, self.alpha)
                if self.coherence_test == "binomial"
                else 5
            )
            # The receiver's own error rate, needed to say what agreement an
            # honest peer *should* show. Estimated blind like any other channel,
            # with a population fallback: leaving it undefined would make the
            # anomaly test vacuously false and block inversion of exactly the
            # adversaries the gate exists to catch.
            e_pop = self._population_error(peers, ceiling)
            honest_reference = self._honest_agreement_reference(peers)
            e_self = blind_errors.get(receiver, float("nan"))
            if not np.isfinite(e_self):
                e_self = e_pop
            stats: dict[int, ChannelStats] = {}
            for peer, pairs in peers.items():
                n = len(pairs)
                dissent = float(np.mean([a != own for a, own in pairs])) if n else 0.0
                agree = 1.0 - dissent
                # EB shrinkage toward the population: a peer heard on few tasks
                # should not be judged on its handful of observations alone.
                shrunk = (n * dissent + self.eb_prior_weight * population_dissent) / (
                    n + self.eb_prior_weight
                )
                e_peer = blind_errors.get(peer, float("nan"))
                if not np.isfinite(e_peer):
                    e_peer = e_pop
                expected = honest_reference
                stats[peer] = ChannelStats(
                    n_observed=n,
                    dissent_rate=shrunk,
                    agreement_with_self=agree,
                    expected_honest_agreement=expected,
                    # Duplicates are NOT excluded from coherence evidence: a
                    # coordinated adversarial bloc is duplicated by construction,
                    # so excluding same-answer partners removes exactly the
                    # signal the gate exists to find. What separates an honest
                    # replica from a lying bloc is agreement with the receiver,
                    # handled below.
                    **self._coherence_fields(peer, answers_by_peer, own_answers, support_floor),
                    blind_error=e_peer,
                )
            for peer, s in stats.items():
                s.decision, s.weight = self._decide(peer, s, receiver)
            self._stats[receiver] = stats
        self.diagnostics.channels = self._stats
        if self.window:
            self._fit_windowed(tasks, heard)

    def _fit_windowed(
        self,
        tasks: Sequence[Sequence[Observation]],
        heard: dict[int, dict[int, list[tuple[str | None, str | None]]]],
    ) -> None:
        """Per-task channel decisions from a sliding window, with a regime flag.

        The static gate pools every task it has ever seen, which is the right
        thing to do against a stationary adversary and the wrong thing against
        one that alternates. A burst adversary that lies coherently for twenty
        tasks and emits noise for the next twenty presents, in aggregate, a
        channel whose coherence sits between the two regimes and clears neither
        threshold -- which is why the static gate never fired on burst at all.

        Two mechanisms here. The window recomputes coherence over the last ``W``
        tasks, so a coherent burst is visible while it is happening. The **regime
        flag** compares the window estimate against the pooled one and, when they
        diverge by more than sampling noise allows, DEMOTES the channel to
        discard rather than inverting it: a channel caught changing what it is
        should not be trusted with a sign, because by the time the evidence is in
        the regime may already have flipped. Discarding is the conservative
        response to instability; inverting is not.
        """
        order = [obs[0].task_id for obs in tasks] if tasks else []
        index = {task_id: i for i, task_id in enumerate(order)}
        window = int(self.window or 0)
        ceiling = self._effective_ceiling()
        support_floor = self.thresholds.min_support_for(self.benchmark, self.alpha)

        for receiver, peers in heard.items():
            answers_by_peer = {peer: [a for a, _ in pairs] for peer, pairs in peers.items()}
            own_answers = [own for _, own in next(iter(peers.values()))] if peers else []
            n_tasks = len(own_answers)
            pooled = {
                peer: self._coherence(peer, answers_by_peer, own_answers, min_support=1)
                for peer in peers
            }
            for task_id in order:
                t_index = index[task_id]
                lo = max(0, t_index - window + 1)
                sl = slice(lo, t_index + 1)
                sub_answers = {p: v[sl] for p, v in answers_by_peer.items()}
                sub_own = own_answers[sl]
                honest_reference = self._honest_agreement_reference(
                    {p: pairs[sl] for p, pairs in peers.items()}
                )
                decisions: dict[int, tuple[str, float]] = {}
                for peer, pairs in peers.items():
                    if peer == receiver:
                        decisions[peer] = (TRUST, 1.0)
                        continue
                    window_pairs = pairs[sl]
                    rate, k, n = self._coherence(peer, sub_answers, sub_own, min_support=1)
                    agree = (
                        float(np.mean([a != own for a, own in window_pairs]))
                        if window_pairs
                        else 0.0
                    )
                    stats = ChannelStats(
                        n_observed=len(window_pairs),
                        dissent_rate=agree,
                        agreement_with_self=1.0 - agree,
                        expected_honest_agreement=honest_reference,
                        coherence=rate,
                        coincidences=k,
                        joint_dissents=n,
                        coherence_pvalue=(
                            float(binom.sf(k - 1, n, ceiling))
                            if n >= support_floor and n > 0
                            else float("nan")
                        ),
                        blind_error=agree,
                    )
                    decision, weight = self._decide(peer, stats, receiver)
                    if decision == INVERT and self._regime_changed(rate, n, pooled[peer]):
                        # Non-stationary: demote rather than invert.
                        self.n_regime_flagged += 1
                        decision, weight = DISCARD, 0.0
                    decisions[peer] = (decision, weight)
                self._task_decisions[(receiver, task_id)] = decisions
            del n_tasks

    def _regime_changed(
        self, window_rate: float, window_n: int, pooled: tuple[float, int, int]
    ) -> bool:
        """Has this channel's coherence shifted beyond sampling noise?

        Compares the window estimate against the pooled one at three binomial
        standard errors of the window estimate -- the same convention used for
        every other tolerance in this project.
        """
        pooled_rate, _, pooled_n = pooled
        if not np.isfinite(window_rate) or not np.isfinite(pooled_rate):
            return False
        if window_n <= 0 or pooled_n <= 0:
            return False
        se = float(np.sqrt(max(pooled_rate * (1.0 - pooled_rate), 1e-9) / window_n))
        return abs(window_rate - pooled_rate) > 3.0 * se

    def _blind_errors(
        self, answers_by_peer: dict[int, list[str | None]], receiver: int
    ) -> dict[int, float]:
        """Marginal error rates from pairwise answer agreement, where identifiable.

        Uses the Phase B triplet identity. It returns nothing usable on
        open-ended answer spaces (agreement sits below 0.5 there), in which case
        the caller falls back to disagreement-with-self.
        """
        peers = sorted(answers_by_peer)
        if len(peers) < 3:
            return {}
        agreement: dict[tuple[str, str], float] = {}
        for i, a in enumerate(peers):
            for b in peers[i + 1 :]:
                xs, ys = answers_by_peer[a], answers_by_peer[b]
                same = [
                    x == y
                    for x, y in zip(xs, ys, strict=False)
                    if x is not None and y is not None
                ]
                if same:
                    agreement[(str(a), str(b))] = float(np.mean(same))
        recovered = blind_marginal_error_rates(agreement, [str(p) for p in peers])
        return {int(k): v for k, v in recovered.items() if np.isfinite(v) and 0.0 <= v <= 1.0}

    @staticmethod
    def _population_error(
        peers: dict[int, list[tuple[str | None, str | None]]], ceiling: float
    ) -> float:
        """Error rate implied by the swarm's mean agreement with the receiver.

        Inverts the attenuation law under the symmetric assumption
        ``e_i = e_j = e`` at the ceiling ``q``::

            m = (1-e)^2 + e^2 q   =>   e^2 (1+q) - 2e + (1-m) = 0

        taking the root in [0, 1/2] because agents are assumed better than
        chance. This gives a usable expectation from observables alone whenever
        the triplet identity cannot return one -- which on open-ended answer
        spaces is most of the time, since agreement there sits below 0.5.
        """
        rates = [
            float(np.mean([a == own for a, own in pairs])) for pairs in peers.values() if pairs
        ]
        if not rates:
            return 0.5
        m = float(np.clip(np.mean(rates), 1e-6, 1 - 1e-6))
        a = 1.0 + ceiling
        disc = 1.0 - a * (1.0 - m)
        if disc < 0 or a <= 0:
            return 0.5
        root = (1.0 - float(np.sqrt(disc))) / a
        return float(np.clip(root, 0.0, 0.5))

    @staticmethod
    def _honest_agreement_reference(
        peers: dict[int, list[tuple[str | None, str | None]]],
    ) -> float:
        """Agreement level that separates honest peers from a lying bloc.

        Estimated by finding the **largest gap** in the sorted agreements rather
        than by taking a fixed quantile, and the reason is that no fixed quantile
        survives the whole range of f. A median over all peers goes over to the
        adversaries once they are a majority; a median over the upper half
        survives to about f = 0.6 and then fails too, because at f = 0.7 seven of
        nine peers are adversarial and the upper half is no longer honest.

        Because the receiver is itself honest, a coherent adversary agrees with
        it far less than any honest peer does, so the agreements fall into two
        separated clusters and the split between them is the largest gap. The
        reference is the bottom of the honest cluster. A gap only counts if it is
        wide relative to the ordinary spacing within a cluster, so a swarm of
        purely honest peers -- which has no such gap -- yields no split and
        nothing is flagged.

        The law-derived alternative was tried and abandoned: same-model replicas
        push the triplet error estimator to implausibly low values, inflating the
        expected agreement to 0.72 on MATH-500 where honest models agree about
        0.40 of the time, so every honest peer looked anomalous.
        """
        rates = sorted(
            float(np.mean([a == own for a, own in pairs])) for pairs in peers.values() if pairs
        )
        # Drop the receiver's own channel, which agrees with itself perfectly.
        rates = [r for r in rates if r < 1.0 - 1e-9]
        if len(rates) < 3:
            return float("nan")

        gaps = [(rates[i + 1] - rates[i], i) for i in range(len(rates) - 1)]
        widest, index = max(gaps)
        others = sorted(g for g, _ in gaps)
        typical = float(np.median(others)) if others else 0.0
        # A split has to be clearly wider than ordinary within-cluster spacing,
        # and wide enough to matter at all.
        if widest < max(3.0 * typical, 0.15):
            return float("nan")
        return float(rates[index + 1])

    @staticmethod
    def _duplicate_groups(
        answers_by_peer: dict[int, list[str | None]], threshold: float = 0.99
    ) -> dict[int, frozenset[int]]:
        """Group channels that are the *same source*, not merely similar.

        Two agents replaying the same cached model emit byte-identical answers on
        every task. Their coincidence is exactly 1.0 on every joint dissent,
        which reads to the gate as a perfectly coherent bloc -- so a swarm of ten
        agents drawn from four models had every honest channel inverted, because
        each model appears two or three times.

        That is redundancy, not conspiracy. Phase B already measured that
        same-model replicas are epistemically one agent (N_eff ~ 1 for a
        homogeneous swarm); the gate must not read that as adversarial
        coordination. Channels agreeing above ``threshold`` are treated as one
        source and are excluded from informing on each other, since a source
        cannot corroborate itself.
        """
        peers = sorted(answers_by_peer)
        groups: dict[int, set[int]] = {p: {p} for p in peers}
        for i, a in enumerate(peers):
            for b in peers[i + 1 :]:
                xs, ys = answers_by_peer[a], answers_by_peer[b]
                pairs = [
                    (x, y) for x, y in zip(xs, ys, strict=False) if x is not None or y is not None
                ]
                if not pairs:
                    continue
                same = float(np.mean([x == y for x, y in pairs]))
                if same >= threshold:
                    groups[a].add(b)
                    groups[b].add(a)
        return {p: frozenset(g) for p, g in groups.items()}

    def _coherence(
        self,
        peer: int,
        answers_by_peer: dict[int, list[str | None]],
        own_answers: list[str | None],
        min_support: int = 5,
        duplicates: frozenset[int] | None = None,
    ) -> tuple[float, int, int]:
        """How tightly this peer coincides with its closest fellow dissenter.

        The statistic is a **maximum** over partners, not a mean, and the choice
        is forced by what it has to detect. Coherence is evidence of *bloc
        membership*: an adversary coincides perfectly with its co-conspirators
        and barely at all with everyone else. Averaging over the whole
        neighbourhood dilutes that signal in direct proportion to how many
        honest peers are present -- at f = 0.3 a lying channel's mean coincidence
        came out at 0.48, below the 0.589 honest ceiling, so the gate trusted
        every adversary in the swarm. The maximum is invariant to the size of the
        honest majority, which is precisely the property needed.

        Using a maximum does raise the false-positive rate on honest channels,
        since the largest of many noisy pairwise rates drifts upward. That is why
        inversion also requires the agreement test in :meth:`_decide`: honest
        peers agree with the receiver at the rate the law predicts and so fail
        the second condition even when their peak coincidence is high.

        Pairs with fewer than ``min_support`` jointly-dissenting tasks are
        skipped rather than counted, since a coincidence rate over two or three
        events is noise.
        """
        mine = answers_by_peer.get(peer)
        if mine is None or not own_answers:
            return float("nan"), 0, 0
        best: tuple[float, int, int] = (float("nan"), 0, 0)
        best_rate = -1.0
        for other, theirs in answers_by_peer.items():
            if other == peer or (duplicates is not None and other in duplicates):
                continue
            num = den = 0
            for a, b, own in zip(mine, theirs, own_answers, strict=False):
                if a is None or own is None or a == own:
                    continue
                if b is None or b == own:
                    continue
                den += 1
                num += int(a == b)
            if den >= min_support and den > 0:
                rate = num / den
                # Rank partners by coincidence rate, breaking ties toward the
                # partner with more evidence behind it.
                if (rate, den) > (best_rate, best[2]):
                    best_rate = rate
                    best = (rate, num, den)
        return best

    def _coherence_fields(
        self,
        peer: int,
        answers_by_peer: dict[int, list[str | None]],
        own_answers: list[str | None],
        support_floor: int,
        duplicates: frozenset[int] | None = None,
    ) -> dict[str, float | int]:
        rate, k, n = self._coherence(
            peer, answers_by_peer, own_answers, min_support=1, duplicates=duplicates
        )
        pvalue = float("nan")
        if n >= support_floor and n > 0:
            # One-sided exact binomial test of H0: q = ceiling against q > ceiling.
            # _effective_ceiling(), not the raw calibrated value: M2 randomizes the
            # threshold, and the binomial test is the DEFAULT rule, so a jitter
            # applied only to the fixed-ceiling branch leaves M2 inert in every
            # run that matters. That is what the first Task 3 pass measured.
            pvalue = float(binom.sf(k - 1, n, self._effective_ceiling()))
        return {
            "coherence": rate,
            "coincidences": k,
            "joint_dissents": n,
            "coherence_pvalue": pvalue,
        }

    def _effective_ceiling(self) -> float:
        """The ceiling this fit is actually using, after M2 jitter."""
        return float(np.clip(self.thresholds.ceiling_for(self.benchmark) * self._ceiling_jitter, 0.0, 1.0))

    def _draw_ceiling_jitter(self, epoch_key: object) -> None:
        """Redraw M2's private multiplier. Called once per fit, not per channel.

        Per-fit rather than per-channel is the honest version: a defender that
        rerolled the threshold for every peer would be averaging noise away and
        would not actually be unpredictable to an attacker facing the swarm.
        """
        if not self.randomized_threshold:
            self._ceiling_jitter = 1.0
            return
        payload = f"{self.threshold_seed}:{epoch_key}".encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        self._ceiling_jitter = float(
            1.0 + rng.uniform(-self.randomized_threshold, self.randomized_threshold)
        )

    def _decide(self, peer: int, s: ChannelStats, receiver: int) -> tuple[str, float]:
        """Three-way channel decision.

        Inversion needs **two** conditions, not one. Coherence alone is
        ambiguous from a receiver's viewpoint: a bloc that coherently disagrees
        with me is either a coherent adversary *or* a set of agents who happen to
        be right on the questions I get wrong. Conflating the two inverts honest
        peers whenever the receiver is the one in error.

        What breaks the symmetry is agreement with *me*, and the receiver knows
        it is itself honest. An honest peer agrees with an honest receiver at the
        rate the attenuation law predicts,
        ``m = (1-e_i)(1-e_j) + e_i e_j q``; a coherent liar agrees only on the
        rare tasks where the receiver's own error happens to coincide with the
        lie, which is far below that. So a channel is inverted only when it is
        *both* more coherent with other dissenters than an honest population ever
        was, *and* agrees with the receiver less than the law allows for an
        honest peer. This is the two-sided test, and it holds at any Byzantine
        fraction because both quantities are anchored on the receiver.
        """
        if peer == receiver:
            # A receiver always trusts itself; that self-vote is exactly the mass
            # the parity control holds fixed across methods.
            return TRUST, 1.0
        if s.n_observed < self.thresholds.min_observations:
            return DISCARD, 0.0

        error = s.blind_error if np.isfinite(s.blind_error) else s.dissent_rate
        capacity = abs(1.0 - 2.0 * float(np.clip(error, 0.0, 1.0)))
        if self.weight_clip is not None:
            capacity = min(capacity, self.weight_clip)

        if self.mode != "trust_only" and np.isfinite(s.coherence):
            if self.mode == "naive":
                # No ceiling and no test: invert on any above-chance coherence.
                coherent_enough = s.coherence > self.thresholds.chance_for(self.benchmark)
            elif self.coherence_test == "fixed_ceiling":
                # Phase C's rule, retained as an ablation. It compares a point
                # estimate to the ceiling with no notion of how much evidence
                # sits behind it, which is why it bled accuracy on MATH-500:
                # there the ceiling is 0.042, so a single lucky coincidence
                # between two honest peers cleared it.
                coherent_enough = s.coherence > self._effective_ceiling()
            else:
                # Binomial: is q_hat significantly above the calibrated ceiling,
                # given the number of joint-dissent events actually observed?
                # Channels below the derived support floor score NaN and are
                # never eligible.
                coherent_enough = (
                    np.isfinite(s.coherence_pvalue) and s.coherence_pvalue <= self.alpha
                )
            # AIP-naive deliberately omits the second condition as well as the
            # calibrated ceiling: it is the ablation showing what the gate buys.
            anomalous = True if self.mode == "naive" else self._agrees_less_than_honest(s)
            if self.soft_gate is not None and anomalous:
                # M1: no cliff. Weight rises smoothly through the ceiling, so
                # there is no coherence level at which the defence switches off.
                ceiling = self._effective_ceiling()
                if ceiling > 0 and np.isfinite(s.coherence):
                    z = self.soft_gate * (s.coherence / ceiling - 1.0)
                    soft = 1.0 / (1.0 + np.exp(-z))
                    if soft >= 0.5:
                        return INVERT, capacity * float(2.0 * (soft - 0.5))
                    # BELOW the soft midpoint the channel is not coherent enough
                    # to invert, and M1 must decide here rather than fall through.
                    # Falling through handed control to the hard gate, which
                    # inverted at full weight -- so M1 was inactive in exactly the
                    # sub-ceiling band it exists to defend, and its results were
                    # identical to the hard gate for that reason and not because
                    # smoothing cannot help. Trust it if its blind error is low
                    # enough to be worth pooling, discard it otherwise.
                    if error < self.thresholds.max_blind_error_rate:
                        return TRUST, capacity * float(2.0 * (0.5 - soft))
                    return DISCARD, 0.0
            if coherent_enough and anomalous:
                return INVERT, capacity
        if error < self.thresholds.max_blind_error_rate:
            return TRUST, capacity
        return DISCARD, 0.0

    def _agrees_less_than_honest(self, s: ChannelStats) -> bool:
        """Is this channel's agreement with the receiver *significantly* low?

        A one-sided test rather than a tuned margin: the law gives the agreement
        an honest peer should show, and a channel counts as anomalous only if the
        observed rate falls more than ``n_sigma`` binomial standard errors below
        it. That keeps the decision rule free of a constant fitted to the
        evaluation data -- the only calibrated numbers in AIP are the honest-q
        ceilings, measured once in Phase B2 against labelled data. Three sigma is
        the same band convention used for every binomial tolerance in Phases B
        and B2, chosen once for the project rather than per experiment.
        """
        expected = s.expected_honest_agreement
        if not np.isfinite(expected) or s.n_observed <= 0:
            return False
        expected = float(np.clip(expected, 1e-6, 1 - 1e-6))
        se = float(np.sqrt(expected * (1.0 - expected) / s.n_observed))
        return s.agreement_with_self < expected - self.n_sigma * se

    # -- aggregate -------------------------------------------------------

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        labels = self.label_space or observed_labels(observations)
        if not labels:
            return None
        stats = self._stats.get(self_id, {})
        task_id = observations[0].task_id if observations else None
        windowed = self._task_decisions.get((self_id, task_id)) if self.window and task_id else None

        def _decision_of(agent: int) -> str:
            if agent == self_id:
                return TRUST
            if windowed is not None:
                return windowed.get(agent, (DISCARD, 0.0))[0]
            s = stats.get(agent)
            return s.decision if s else DISCARD

        def _weight_of(agent: int) -> float:
            if windowed is not None:
                return windowed.get(agent, (DISCARD, 0.0))[1]
            return stats.get(agent, ChannelStats()).weight

        weights = {b.agent_id: _weight_of(b.agent_id) for b in observations}
        weights = apply_self_vote_parity(weights, self_id, self.parity)

        n_labels = len(labels)
        scores = dict.fromkeys(labels, 0.0)
        for b in observations:
            if b.answer is None or b.answer not in scores:
                continue
            decision = _decision_of(b.agent_id)
            w = float(weights.get(b.agent_id, 0.0))
            if w <= 0.0 or decision == DISCARD:
                continue
            if decision == TRUST:
                scores[b.answer] += w
            elif decision == INVERT and n_labels > 1:
                # Evidence against this answer, spread over the alternatives.
                scores[b.answer] -= w
                share = w / (n_labels - 1)
                for label in labels:
                    if label != b.answer:
                        scores[label] += share

        best = max(scores.values())
        return sorted(k for k, v in scores.items() if np.isclose(v, best))[0]


def make_aip(benchmark: str, mode: str, **kwargs: Any) -> AIPAggregator:
    return AIPAggregator(benchmark=benchmark, mode=mode, **kwargs)
