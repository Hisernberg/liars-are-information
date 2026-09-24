"""Baseline aggregators, and the Dawid-Skene machinery they share.

Phase C wraps these into the ``aggregate(observations, self_id) -> answer``
interface alongside majority, Krum, geometric median and the rest. Phase B2
needs the Dawid-Skene *estimator* rather than the aggregator, so the EM core
lands here first and is imported by :mod:`aip.aggregation.correlation`.

The model is the **one-coin** (homogeneous) Dawid-Skene variant: each agent has a
single competence parameter rather than a full ``C x C`` confusion matrix. That
choice is forced by the data, not by convenience. Two of the three benchmarks are
open-ended, so the label space differs from task to task and a fixed confusion
matrix has no meaning; and with four agents over a hundred tasks a full
per-agent confusion matrix would be estimated from a handful of observations per
cell. The one-coin model estimates one number per agent from all of them.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Answer = Hashable | None

#: A missing answer never coincides with anything, matching the Phase B policy:
#: two agents that both failed to produce a parseable answer have not reached the
#: same conclusion.
MISSING = None


@dataclass
class DawidSkeneResult:
    """Fitted one-coin Dawid-Skene model over a swarm's answers."""

    agents: list[str]
    posterior: list[dict[Any, float]]
    """Per task, a distribution over that task's candidate labels."""
    error_rates: dict[str, float]
    candidates: list[list[Any]]
    n_iter: int
    converged: bool
    log_likelihood: float
    history: list[float] = field(default_factory=list)

    def map_labels(self) -> list[Any]:
        """Posterior mode per task -- the DS aggregate answer."""
        out = []
        for post in self.posterior:
            out.append(max(post.items(), key=lambda kv: kv[1])[0] if post else None)
        return out


def _candidates_for_task(row: dict[str, Answer], label_space: Sequence[Any] | None) -> list[Any]:
    """Labels the latent truth may take on this task.

    With a closed answer space (MMLU) every option is a candidate whether or not
    anyone chose it -- that is what makes the chance level ``1/(C-1)`` meaningful.
    With open-ended answers the candidate set is what the swarm actually
    proposed, which encodes the standard assumption that the truth is somewhere
    among the broadcasts. When it is not, no label-free method could recover it.
    """
    if label_space is not None:
        return list(label_space)
    seen = [a for a in row.values() if a is not MISSING]
    # dict.fromkeys keeps first-seen order, so the fit is deterministic.
    return list(dict.fromkeys(seen))


def dawid_skene_em(
    answers: list[dict[str, Answer]],
    agents: Sequence[str] | None = None,
    label_space: Sequence[Any] | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
    init_error_rate: float = 0.3,
    floor: float = 1e-6,
) -> DawidSkeneResult:
    """Fit the one-coin Dawid-Skene model. **Uses no ground-truth labels.**

    ``answers[t][agent]`` is an agent's answer on task ``t``, or ``None``.

    Emission model, for a task with ``K`` candidate labels::

        P(a_i = k | y = c) = 1 - e_i           if k == c
                           = e_i / (K - 1)     otherwise
        P(a_i = None | y = c) = e_i            (constant in c: uninformative
                                                about y, but counted as an error)

    A missing answer is deliberately made uninformative rather than treated as a
    vote: it should not shift the posterior toward any label, but it must still
    cost the agent competence.
    """
    if agents is None:
        agents = sorted({a for row in answers for a in row})
    agents = list(agents)
    n_tasks = len(answers)
    if n_tasks == 0:
        return DawidSkeneResult(agents, [], dict.fromkeys(agents, float("nan")), [], 0, True, 0.0)

    candidates = [_candidates_for_task(row, label_space) for row in answers]
    error = {a: float(init_error_rate) for a in agents}

    posterior: list[dict[Any, float]] = []
    log_likelihood = -np.inf
    history: list[float] = []
    converged = False
    iteration = 0

    while iteration < max_iter:
        iteration += 1
        # -- E step -------------------------------------------------------
        posterior = []
        total_ll = 0.0
        for row, cands in zip(answers, candidates, strict=True):
            if not cands:
                posterior.append({})
                continue
            k = len(cands)
            log_post = np.zeros(k)
            for agent in agents:
                a = row.get(agent, MISSING)
                e = min(max(error[agent], floor), 1.0 - floor)
                if a is MISSING:
                    total_ll += np.log(e)  # constant in c
                    continue
                if k == 1:
                    log_post += np.log(1.0 - e)
                    continue
                wrong = np.log(e / (k - 1))
                right = np.log(1.0 - e)
                log_post += np.where(np.array([c == a for c in cands]), right, wrong)
            log_post -= log_post.max()
            probs = np.exp(log_post)
            total = probs.sum()
            if total <= 0:
                probs = np.full(k, 1.0 / k)
                total = 1.0
            probs = probs / total
            posterior.append(dict(zip(cands, probs.tolist(), strict=True)))
            total_ll += float(np.log(total) + log_post.max())

        # -- M step -------------------------------------------------------
        new_error: dict[str, float] = {}
        for agent in agents:
            expected_correct = 0.0
            seen = 0
            for row, post in zip(answers, posterior, strict=True):
                if agent not in row:
                    continue
                seen += 1
                a = row[agent]
                if a is MISSING:
                    continue
                expected_correct += float(post.get(a, 0.0))
            new_error[agent] = float("nan") if seen == 0 else 1.0 - expected_correct / seen
        shift = max(
            abs(new_error[a] - error[a])
            for a in agents
            if np.isfinite(new_error[a]) and np.isfinite(error[a])
        )
        error = new_error
        history.append(total_ll)
        if shift < tol:
            converged = True
            log_likelihood = total_ll
            break
        log_likelihood = total_ll

    return DawidSkeneResult(
        agents=agents,
        posterior=posterior,
        error_rates=error,
        candidates=candidates,
        n_iter=iteration,
        converged=converged,
        log_likelihood=float(log_likelihood),
        history=history,
    )


# =========================================================================
# Baseline aggregators (Phase C)
# =========================================================================
#
# Implemented from the specification; there is no legacy/ directory in this
# repository to port formulas from.

from collections import Counter  # noqa: E402

from aip.aggregation.base import (  # noqa: E402
    Aggregator,
    ParityConfig,
    apply_self_vote_parity,
    decode,
    observed_labels,
    onehot_matrix,
)
from aip.types import Broadcast, Observation  # noqa: E402


def _weighted_vote(
    broadcasts: Sequence[Broadcast],
    weights: dict[int, float],
    self_id: int,
    parity: ParityConfig,
) -> str | None:
    """Sum weights per answer under the parity condition and take the argmax."""
    weights = apply_self_vote_parity(weights, self_id, parity)
    scores: dict[str, float] = {}
    for b in broadcasts:
        if b.answer is None:
            continue
        scores[b.answer] = scores.get(b.answer, 0.0) + float(weights.get(b.agent_id, 0.0))
    if not scores:
        return None
    best = max(scores.values())
    return sorted(k for k, v in scores.items() if np.isclose(v, best))[0]


def rank_normalize(values: dict[int, float]) -> dict[int, float]:
    """Map values to within-task ranks in (0, 1].

    Required by the Phase A amendment: raw answer-span confidences are
    compressed into a band roughly 0.03 wide, so a confidence-weighted vote on
    raw values is numerically almost an unweighted vote. Ranking restores the
    ordering information that the compression destroys, and it is applied here
    at consumption time -- never baked into the cache.
    """
    if not values:
        return {}
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    return {k: (i + 1) / n for i, (k, _) in enumerate(items)}


class MajorityVote(Aggregator):
    """Plurality over broadcast answers. The unweighted reference point."""

    name = "majority"

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        counts = Counter(b.answer for b in observations if b.answer is not None)
        if not counts:
            return None
        top = max(counts.values())
        return sorted(k for k, v in counts.items() if v == top)[0]


class ConfidenceWeighted(Aggregator):
    """CP-WBFT-style weighting by confidence, rank-normalized within the task."""

    name = "confidence_weighted"

    def __init__(self, parity: ParityConfig | None = None, use_self_report: bool = False) -> None:
        self.parity = parity or ParityConfig()
        self.use_self_report = use_self_report
        if use_self_report:
            self.name = "confidence_weighted_selfreport"

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        raw = {
            b.agent_id: (
                b.self_reported_confidence if self.use_self_report else b.logprob_confidence
            )
            for b in observations
        }
        raw = {k: (0.0 if v is None or not np.isfinite(v) else float(v)) for k, v in raw.items()}
        return _weighted_vote(observations, rank_normalize(raw), self_id, self.parity)


class GeometricMedian(Aggregator):
    """Weiszfeld geometric median of the one-hot vectors, decoded by argmax."""

    name = "geometric_median"

    def __init__(self, max_iter: int = 128, tol: float = 1e-7) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        labels = observed_labels(observations)
        if not labels:
            return None
        points = onehot_matrix(observations, labels)
        estimate = points.mean(axis=0)
        for _ in range(self.max_iter):
            distances = np.linalg.norm(points - estimate, axis=1)
            near = distances < 1e-12
            if near.all():
                break
            inverse = np.where(near, 0.0, 1.0 / np.maximum(distances, 1e-12))
            if inverse.sum() <= 0:
                break
            update = (points * inverse[:, None]).sum(axis=0) / inverse.sum()
            if np.linalg.norm(update - estimate) < self.tol:
                estimate = update
                break
            estimate = update
        return decode(estimate, labels)


def _krum_scores(points: np.ndarray, n_byzantine: int) -> np.ndarray:
    """Krum score: distance to the ``n - f - 2`` nearest other points."""
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    sq = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(sq, np.inf)
    keep = max(1, n - n_byzantine - 2)
    keep = min(keep, max(1, n - 1))
    ordered = np.sort(sq, axis=1)[:, :keep]
    return ordered.sum(axis=1)


class Krum(Aggregator):
    """Select the single broadcast closest to its nearest neighbours.

    Oracle: Krum needs the Byzantine count. It is given the true one, which
    flatters it relative to any deployable setting, so it is flagged as oracle
    and excluded from the deployable minimax-regret comparison.
    """

    name = "krum"
    is_oracle = True

    def __init__(self, n_byzantine: int = 0) -> None:
        self.n_byzantine = n_byzantine

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        labels = observed_labels(observations)
        if not labels:
            return None
        points = onehot_matrix(observations, labels)
        if points.shape[0] < 3:
            return MajorityVote().aggregate(observations, self_id)
        scores = _krum_scores(points, self.n_byzantine)
        return observations[int(np.argmin(scores))].answer


class MultiKrum(Aggregator):
    """Average the ``m`` best-scoring broadcasts, then decode."""

    name = "multi_krum"
    is_oracle = True

    def __init__(self, n_byzantine: int = 0, m: int | None = None) -> None:
        self.n_byzantine = n_byzantine
        self.m = m

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        labels = observed_labels(observations)
        if not labels:
            return None
        points = onehot_matrix(observations, labels)
        n = points.shape[0]
        if n < 3:
            return MajorityVote().aggregate(observations, self_id)
        m = self.m if self.m is not None else max(1, n - self.n_byzantine)
        m = min(max(1, m), n)
        scores = _krum_scores(points, self.n_byzantine)
        chosen = np.argsort(scores)[:m]
        return decode(points[chosen].mean(axis=0), labels)


class CoordinateMedian(Aggregator):
    """Coordinate-wise median of the one-hot vectors."""

    name = "coord_median"

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        labels = observed_labels(observations)
        if not labels:
            return None
        return decode(np.median(onehot_matrix(observations, labels), axis=0), labels)


class TrimmedMean(Aggregator):
    """Coordinate-wise mean after trimming the extremes.

    Oracle for the same reason as Krum: the trim fraction is set from the true
    Byzantine fraction.
    """

    name = "trimmed_mean"
    is_oracle = True

    def __init__(self, trim: float = 0.1) -> None:
        self.trim = float(trim)

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        labels = observed_labels(observations)
        if not labels:
            return None
        points = onehot_matrix(observations, labels)
        n = points.shape[0]
        k = int(np.floor(self.trim * n))
        if 2 * k >= n:
            k = max(0, (n - 1) // 2)
        ordered = np.sort(points, axis=0)
        kept = ordered[k : n - k] if k > 0 else ordered
        return decode(kept.mean(axis=0), labels)


class DawidSkeneAggregator(Aggregator):
    """Dawid-Skene over the receiver's observation stream.

    ``use_anchors`` pins a subset of agents to their true error rates, which is
    oracle information; without anchors the fit is fully label-free.
    """

    name = "dawid_skene"
    needs_fit = True

    def __init__(
        self,
        label_space: Sequence[Any] | None = None,
        use_anchors: bool = False,
        anchor_error_rates: dict[int, float] | None = None,
    ) -> None:
        self.label_space = label_space
        self.use_anchors = use_anchors
        self.anchor_error_rates = anchor_error_rates or {}
        self.is_oracle = use_anchors
        if use_anchors:
            self.name = "dawid_skene_anchors"
        self._answer: dict[tuple[int, str], str | None] = {}

    def fit(self, tasks: Sequence[Sequence[Observation]]) -> None:
        by_agent: dict[int, list[dict[str, Answer]]] = {}
        task_ids: dict[int, list[str]] = {}
        for observations in tasks:
            for obs in observations:
                row = {str(b.agent_id): b.answer for b in obs.broadcasts}
                by_agent.setdefault(obs.self_id, []).append(row)
                task_ids.setdefault(obs.self_id, []).append(obs.task_id)

        for agent, rows in by_agent.items():
            init = 0.3
            if self.use_anchors and self.anchor_error_rates:
                init = float(np.mean(list(self.anchor_error_rates.values())))
            result = dawid_skene_em(rows, label_space=self.label_space, init_error_rate=init)
            for task_id, label in zip(task_ids[agent], result.map_labels(), strict=True):
                self._answer[(agent, task_id)] = label

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        if not observations:
            return None
        key = (self_id, observations[0].task_id)
        if key in self._answer:
            return self._answer[key]
        return MajorityVote().aggregate(observations, self_id)


class SACFilterRefine(Aggregator):
    """SAC-style filter-and-refine: score peers by history, drop outliers, re-vote.

    The receiver tracks how often each peer agreed with the swarm plurality, then
    filters the peers whose track record falls in the bottom ``drop_quantile``
    before taking a confidence-weighted vote over the survivors. This is the
    strongest *discard*-family baseline in the sweep -- it is the method AIP has
    to beat to justify inverting rather than discarding.
    """

    name = "sac_filter_refine"
    needs_fit = True

    def __init__(self, drop_quantile: float = 0.25, parity: ParityConfig | None = None) -> None:
        self.drop_quantile = float(drop_quantile)
        self.parity = parity or ParityConfig()
        self._keep: dict[int, set[int]] = {}

    def fit(self, tasks: Sequence[Sequence[Observation]]) -> None:
        agree: dict[int, dict[int, list[float]]] = {}
        for observations in tasks:
            for obs in observations:
                plurality = MajorityVote().aggregate(obs.broadcasts, obs.self_id)
                bucket = agree.setdefault(obs.self_id, {})
                for b in obs.broadcasts:
                    bucket.setdefault(b.agent_id, []).append(
                        1.0 if (b.answer is not None and b.answer == plurality) else 0.0
                    )
        for agent, peers in agree.items():
            rates = {p: float(np.mean(v)) for p, v in peers.items() if v}
            if not rates:
                self._keep[agent] = set()
                continue
            cutoff = float(np.quantile(list(rates.values()), self.drop_quantile))
            keep = {p for p, r in rates.items() if r >= cutoff}
            keep.add(agent)  # a receiver never filters itself out
            self._keep[agent] = keep

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        keep = self._keep.get(self_id)
        kept = [b for b in observations if keep is None or b.agent_id in keep]
        if not kept:
            kept = list(observations)
        raw = {
            b.agent_id: (
                0.0
                if b.logprob_confidence is None or not np.isfinite(b.logprob_confidence)
                else float(b.logprob_confidence)
            )
            for b in kept
        }
        return _weighted_vote(kept, rank_normalize(raw), self_id, self.parity)
