"""RACE: Receiver-Anchored Channel Estimation.

AIP asks a *coherence* question about each peer -- "does this channel coincide
with other dissenters more often than honest agents do?" -- and answers it with
a three-way gate (TRUST / DISCARD / INVERT). Proposition 1 of the original work
shows the limit of that question: a rule that is measurable with respect to
per-question coherence cannot separate an adversary that coordinates at or below
chance from honest independent disagreement, and the gate-aware adversary lives
exactly there.

RACE asks a different question: **how does this channel's report depend on the
truth?** A peer whose report is statistically dependent on the latent answer is
informative, whichever sign that dependence has; a peer whose report is
independent of the truth is not. Coherence is neither necessary nor sufficient
for that dependence. An adversary that never states the truth leaks the truth by
elimination even when it never coordinates, which is precisely the regime where
a coherence gate is blind.

Model (per receiver, per task ``t`` with candidate set ``S_t`` of size ``K_t``)::

    y_t  ~ Uniform(S_t)                                   latent truth
    P(r_tj = y_t)         = a_j                           agent j's accuracy
    P(r_tj = c | c != y_t) = (1 - a_j) / (K_t - 1)          symmetric errors

so the posterior score of candidate ``c`` is a *signed* weighted vote::

    score_t(c) = sum_j  w_tj * 1[r_tj = c] * lambda_tj,
    lambda_tj  = log( a_j (K_t - 1) / (1 - a_j) ).

``lambda > 0`` is TRUST, ``lambda < 0`` is INVERT (evidence against the
reported answer, spread over the alternatives), ``lambda = 0`` is DISCARD.
AIP's gate is a three-level quantisation of this weight; RACE keeps it
continuous, so there is no published cliff for an adversary to park under.

The parameters are fitted by EM on the receiver's own **unlabelled** history.
Label-free latent-class models are identified only up to a relabelling of the
latent truth, and with an adversarial plurality the likelihood prefers the
relabelling in which the liars are the experts: this is why Dawid--Skene and
majority-initialised EM break down at f >= 1/2. RACE removes the ambiguity with
the receiver's self-knowledge:

* **anchor initialisation** -- the first E-step is the receiver's own answer,
  not the plurality;
* **anchor prior and constraint** -- the receiver's own accuracy has a Beta
  prior centred above chance and is projected to stay above chance.

``w_tj`` is a **clone-aware** tempering weight: agents whose answer streams are
near-identical on history (complete-link groups) share one unit of evidence per
task, so ten replicas of one model -- or a perfectly coordinated lying bloc --
count once rather than ten times.

Nothing here reads gold labels, Byzantine identities, model names or
confidence. The only inputs are the answers the receiver itself observed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from aip.aggregation.base import Aggregator
from aip.types import Broadcast, Observation

TRUST, INVERT, DISCARD = "trust", "invert", "discard"


@dataclass
class ChannelEstimate:
    """What a receiver learned about one channel (possibly itself)."""

    accuracy: float
    n_observed: int
    group: int
    weight_at_chance_k: float
    """``lambda`` evaluated at the receiver's mean candidate-set size."""
    decision: str


@dataclass
class ReceiverFit:
    agents: tuple[int, ...]
    accuracy: np.ndarray
    groups: np.ndarray
    """``groups[j]`` is the group id of column ``j``; clones share an id."""
    n_observed: np.ndarray
    mean_k: float
    n_iter: int
    converged: bool
    log_likelihood: float
    confusion: np.ndarray | None = None
    class_prior: np.ndarray | None = None
    log_posterior: float = float("nan")
    init: str = "self"
    posterior: np.ndarray | None = None
    """History posteriors over candidates (T x K_max), for grouping and audit."""

    def column(self, agent: int) -> int | None:
        try:
            return self.agents.index(agent)
        except ValueError:
            return None


@dataclass
class RACEDiagnostics:
    channels: dict[int, dict[int, ChannelEstimate]] = field(default_factory=dict)

    def decision_counts(self, byzantine: set[int]) -> dict[str, dict[str, int]]:
        """Decision counts from honest receivers, split by true peer type."""
        out = {"honest": {TRUST: 0, INVERT: 0, DISCARD: 0}, "byzantine": {TRUST: 0, INVERT: 0, DISCARD: 0}}
        for receiver, peers in self.channels.items():
            if receiver in byzantine:
                continue
            for peer, est in peers.items():
                if peer == receiver:
                    continue
                out["byzantine" if peer in byzantine else "honest"][est.decision] += 1
        return out


def _rows_from_observations(
    tasks: Sequence[Sequence[Observation]],
) -> dict[int, list[tuple[str, dict[int, str | None]]]]:
    by_receiver: dict[int, list[tuple[str, dict[int, str | None]]]] = {}
    for observations in tasks:
        for obs in observations:
            row = {b.agent_id: b.answer for b in obs.broadcasts}
            by_receiver.setdefault(obs.self_id, []).append((obs.task_id, row))
    return by_receiver


def complete_link_groups(
    reports: np.ndarray,
    threshold: float,
    min_support: int,
) -> np.ndarray:
    """Group columns whose non-missing reports agree on >= ``threshold`` of tasks.

    ``reports`` is a (T, J) array of per-task candidate indices (-1 = missing).
    Complete linkage: two groups merge only if *every* cross pair is linked, so
    a chain of near-duplicates cannot join two dissimilar endpoints.
    """
    n_tasks, n_agents = reports.shape
    present = reports >= 0
    linked = np.zeros((n_agents, n_agents), dtype=bool)
    score = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            both = present[:, i] & present[:, j]
            n = int(both.sum())
            if n < min_support:
                continue
            rate = float(np.mean(reports[both, i] == reports[both, j]))
            score[i, j] = score[j, i] = rate
            linked[i, j] = linked[j, i] = rate >= threshold
    groups = list(range(n_agents))
    members = {g: {g} for g in groups}
    pairs = sorted(
        ((score[i, j], i, j) for i in range(n_agents) for j in range(i + 1, n_agents) if linked[i, j]),
        reverse=True,
    )
    for _, i, j in pairs:
        gi, gj = groups[i], groups[j]
        if gi == gj:
            continue
        if all(linked[a, b] for a in members[gi] for b in members[gj]):
            for a in members[gj]:
                groups[a] = gi
            members[gi] |= members.pop(gj)
    # Relabel to 0..G-1 in column order for stable diagnostics.
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(g, len(remap)) for g in groups], dtype=int)


def raw_links(reports: np.ndarray, threshold: float, min_support: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise raw agreement over co-observed tasks, and links above ``threshold``."""
    n_agents = reports.shape[1]
    present = reports >= 0
    score = np.zeros((n_agents, n_agents))
    linked = np.zeros((n_agents, n_agents), dtype=bool)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            both = present[:, i] & present[:, j]
            if int(both.sum()) < min_support:
                continue
            rate = float(np.mean(reports[both, i] == reports[both, j]))
            score[i, j] = score[j, i] = rate
            linked[i, j] = linked[j, i] = rate >= threshold
    return linked, score


def error_conditioned_links(
    reports: np.ndarray, truth: np.ndarray, threshold: float, min_support: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise agreement restricted to tasks where at least one member errs.

    Two distinct but accurate channels agree on almost every task simply because
    both are usually right; raw agreement therefore conflates *competence* with
    *dependence*. Replicas of one source -- and a coordinated bloc -- also agree
    when they are wrong. Conditioning on the (estimated) truth isolates exactly
    the conditional dependence that the latent-class model's independence
    assumption ignores.
    """
    n_agents = reports.shape[1]
    present = reports >= 0
    wrong = present & (reports != truth[:, None]) & (truth[:, None] >= 0)
    score = np.zeros((n_agents, n_agents))
    linked = np.zeros((n_agents, n_agents), dtype=bool)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            mask = present[:, i] & present[:, j] & (wrong[:, i] | wrong[:, j])
            n = int(mask.sum())
            if n < min_support:
                continue
            rate = float(np.mean(reports[mask, i] == reports[mask, j]))
            score[i, j] = score[j, i] = rate
            linked[i, j] = linked[j, i] = rate >= threshold
    return linked, score


def groups_from_links(linked: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Complete-link agglomeration of a boolean link matrix."""
    n_agents = linked.shape[0]
    groups = list(range(n_agents))
    members = {g: {g} for g in groups}
    pairs = sorted(
        ((score[i, j], i, j) for i in range(n_agents) for j in range(i + 1, n_agents) if linked[i, j]),
        reverse=True,
    )
    for _, i, j in pairs:
        gi, gj = groups[i], groups[j]
        if gi == gj:
            continue
        if all(linked[a, b] for a in members[gi] for b in members[gj]):
            for a in members[gj]:
                groups[a] = gi
            members[gi] |= members.pop(gj)
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(g, len(remap)) for g in groups], dtype=int)


def tempering_weights(reports: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """``w_tj = 1 / (# observed members of j's group on task t)``."""
    present = reports >= 0
    weights = np.zeros(reports.shape, dtype=float)
    for g in np.unique(groups):
        cols = np.flatnonzero(groups == g)
        count = present[:, cols].sum(axis=1)
        with np.errstate(divide="ignore"):
            share = np.where(count > 0, 1.0 / np.maximum(count, 1), 0.0)
        weights[:, cols] = present[:, cols] * share[:, None]
    return weights


class RACEAggregator(Aggregator):
    """Receiver-anchored, clone-aware latent-truth aggregation.

    Parameters
    ----------
    label_space:
        Closed label set, or ``None`` for open answers (the candidate set is then
        the distinct answers observed on each task).
    model:
        ``"auto"`` (the default, RACE v3.1): class-conditional channels for
        binary label spaces and the symmetric one-coin model otherwise.
        ``"onecoin"`` (one accuracy per channel; RACE v3.0 everywhere) or
        ``"full"`` (one class-conditional confusion matrix per channel; closed
        label spaces only). Binary questions get two parameters per channel,
        which short histories identify, and small LLMs answer them with an
        option bias ("always yes") that a symmetric channel cannot represent.
    anchored:
        ``True`` is RACE. ``False`` is the ablation: identical model, but EM is
        initialised from the plurality and the receiver gets the same prior as
        every peer -- i.e. ordinary label-free Dawid--Skene.
    clone_aware:
        Temper evidence of near-identical channels (see module docstring).
    cap:
        ``"self"`` caps every peer's accuracy estimate at the receiver's own, so
        no single peer can outweigh the anchor. Off by default; evaluated as a
        defence against the camouflage attack.
    """

    needs_fit = True
    is_oracle = False

    def __init__(
        self,
        label_space: Sequence[str] | None = None,
        *,
        model: str = "auto",
        anchored: bool = True,
        clone_aware: bool = True,
        clone_mode: str = "error",
        anchor_mean: float = 0.75,
        anchor_strength: float = 8.0,
        anchor_margin: float = 0.05,
        peer_strength: float = 2.0,
        clone_threshold: float = 0.95,
        same_source_threshold: float = 0.99,
        clone_min_support: int = 8,
        cap: str | None = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        eps: float = 1e-3,
        decision_band: float = 0.25,
        multistart: bool = False,
        condition: str = "all",
        name: str | None = None,
    ) -> None:
        if model not in ("auto", "onecoin", "full"):
            raise ValueError(f"unknown model {model!r}")
        if model == "full" and not label_space:
            raise ValueError("the full-confusion model needs a closed label space")
        self.model_rule = model
        if model == "auto":
            model = "full" if label_space and len(set(label_space)) == 2 else "onecoin"
        if cap not in (None, "self"):
            raise ValueError(f"unknown cap {cap!r}")
        self.label_space = tuple(sorted(label_space)) if label_space else None
        self.model = model
        self.anchored = bool(anchored)
        self.clone_aware = bool(clone_aware)
        if clone_mode not in ("error", "raw"):
            raise ValueError(f"unknown clone_mode {clone_mode!r}")
        self.clone_mode = clone_mode
        self.anchor_mean = float(anchor_mean)
        self.anchor_strength = float(anchor_strength)
        self.anchor_margin = float(anchor_margin)
        self.peer_strength = float(peer_strength)
        self.clone_threshold = float(clone_threshold)
        self.same_source_threshold = float(same_source_threshold)
        self.clone_min_support = int(clone_min_support)
        self.cap = cap
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.eps = float(eps)
        self.decision_band = float(decision_band)
        # Ablation (off by default): also run EM from the plurality and keep the
        # higher log-posterior. It does not rescue a near-chance receiver (the two
        # labellings are then likelihood-equivalent) and it is actively harmful
        # when replicated channels inflate the likelihood of the adversarial
        # labelling, which is why the anchor acts as initialisation + constraint
        # rather than through a likelihood comparison. See docs/RACE_NOTES.md.
        self.multistart = bool(multistart) and self.anchored
        # Extension (RACE-D): estimate channels only on history questions where
        # the heard answers disagree. Trust earned on unanimous questions is free
        # -- agreeing with everyone reveals nothing -- and is exactly what a
        # camouflage attacker banks before lying on contested questions.
        if condition not in ("all", "disagreement"):
            raise ValueError(f"unknown condition {condition!r}")
        self.condition = condition
        if name is None:
            name = "race" if self.anchored else "ds_onecoin"
            if self.model_rule != "auto" and self.anchored:
                name += f"_{self.model_rule}"
            if not self.clone_aware:
                name += "_noclone"
            elif self.clone_mode == "raw":
                name += "_rawclone"
            if self.cap:
                name += f"_cap{self.cap}"
            if self.multistart:
                name += "_ms"
            if self.condition == "disagreement":
                name += "_d"
        self.name = name
        self.fits: dict[int, ReceiverFit] = {}
        self.diagnostics = RACEDiagnostics()

    # ------------------------------------------------------------------ data

    def _candidates(self, row: dict[int, str | None]) -> list[str]:
        if self.label_space:
            return list(self.label_space)
        return sorted({a for a in row.values() if a is not None})

    def _encode(
        self, rows: list[dict[int, str | None]], agents: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        column = {a: j for j, a in enumerate(agents)}
        reports = np.full((len(rows), len(agents)), -1, dtype=int)
        n_cand = np.zeros(len(rows), dtype=int)
        for t, row in enumerate(rows):
            cands = self._candidates(row)
            index = {c: i for i, c in enumerate(cands)}
            n_cand[t] = len(cands)
            for agent, answer in row.items():
                if answer is None or agent not in column or answer not in index:
                    continue
                reports[t, column[agent]] = index[answer]
        return reports, n_cand

    # ------------------------------------------------------------------ fit

    def fit(self, tasks: Sequence[Sequence[Observation]]) -> None:
        self.fits.clear()
        self.diagnostics = RACEDiagnostics()
        for receiver, history in _rows_from_observations(tasks).items():
            rows = [row for _, row in history]
            self.fit_receiver(receiver, rows)

    def fit_receiver(
        self,
        receiver: int,
        rows: list[dict[int, str | None]],
        task_weights: np.ndarray | None = None,
    ) -> ReceiverFit:
        """Fit one receiver from its observed rows; ``task_weights`` enables forgetting."""
        agents = tuple(sorted({a for row in rows for a in row} | {receiver}))
        reports, n_cand = self._encode(rows, agents)
        own = agents.index(receiver)
        tw = np.ones(len(rows)) if task_weights is None else np.asarray(task_weights, dtype=float)
        if self.condition == "disagreement" and len(rows):
            informative = np.array([len({a for a in row.values() if a is not None}) > 1 for row in rows])
            if informative.sum() >= 5:  # otherwise fall back to every question
                tw = tw * informative
        singletons = np.arange(len(agents))
        if self.clone_aware and len(rows) and self.clone_mode == "raw":
            groups = complete_link_groups(reports, self.clone_threshold, self.clone_min_support)
        elif self.clone_aware and len(rows):
            # (1) Same-source links: (near-)identical answer streams. Applied before
            # any fit, so a receiver's own replica can never confirm the receiver.
            same, same_score = raw_links(reports, self.same_source_threshold, self.clone_min_support)
            first = self._fit_groups(reports, n_cand, tw, own, agents, groups_from_links(same, same_score))
            # (2) Dependent-error links: agreement on the tasks where at least one
            # member is wrong relative to the stage-1 truth estimate.
            truth = np.where(first.posterior.max(axis=1) > 0, first.posterior.argmax(axis=1), -1)
            err, err_score = error_conditioned_links(reports, truth, self.clone_threshold, self.clone_min_support)
            groups = groups_from_links(same | err, np.maximum(same_score * same, err_score * err))
        else:
            groups = singletons
        fit = self._fit_groups(reports, n_cand, tw, own, agents, groups)
        self.fits[receiver] = fit
        self.diagnostics.channels[receiver] = self._describe(fit, own)
        return fit

    def _fit_groups(self, reports, n_cand, tw, own, agents, groups) -> ReceiverFit:
        temper = tempering_weights(reports, groups)
        inits = ("self", "plurality") if self.multistart else (("self",) if self.anchored else ("plurality",))
        candidates = []
        for init in inits:
            if self.model == "full":
                candidates.append(self._em_full(reports, temper, tw, own, agents, groups, init))
            else:
                candidates.append(self._em_onecoin(reports, n_cand, temper, tw, own, agents, groups, init))
        return max(candidates, key=lambda f: (f.log_posterior, f.init == "self"))

    def _initial_posterior(
        self, reports: np.ndarray, n_cand: np.ndarray, temper: np.ndarray, own: int, k_max: int, init: str
    ) -> np.ndarray:
        n_tasks = reports.shape[0]
        q = np.zeros((n_tasks, k_max))
        valid = np.arange(k_max)[None, :] < n_cand[:, None]
        if init == "self":
            mine = reports[:, own]
            for t in range(n_tasks):
                k = n_cand[t]
                if k <= 0:
                    continue
                if mine[t] >= 0 and k > 1:
                    q[t, :k] = (1.0 - self.anchor_mean) / (k - 1)
                    q[t, mine[t]] = self.anchor_mean
                else:
                    q[t, :k] = 1.0 / k
        else:
            for t in range(n_tasks):
                k = n_cand[t]
                if k <= 0:
                    continue
                votes = np.zeros(k)
                for j in np.flatnonzero(reports[t] >= 0):
                    votes[reports[t, j]] += temper[t, j]
                votes = votes + 1e-3
                q[t, :k] = votes / votes.sum()
        return np.where(valid, q, 0.0)

    def _em_onecoin(
        self,
        reports: np.ndarray,
        n_cand: np.ndarray,
        temper: np.ndarray,
        tw: np.ndarray,
        own: int,
        agents: tuple[int, ...],
        groups: np.ndarray,
        init: str = "self",
    ) -> ReceiverFit:
        n_tasks, n_agents = reports.shape
        present = reports >= 0
        n_obs = present.sum(axis=0)
        if n_tasks == 0:
            return ReceiverFit(agents, np.full(n_agents, 0.5), groups, n_obs, 2.0, 0, True, 0.0, init=init,
                               posterior=np.zeros((0, 1)))
        k_max = max(1, int(n_cand.max()))
        valid = np.arange(k_max)[None, :] < n_cand[:, None]
        multi = n_cand >= 2
        # Chance level per agent: mean 1/K over the tasks it answered.
        inv_k = np.where(n_cand > 0, 1.0 / np.maximum(n_cand, 1), 0.0)
        chance = np.array(
            [float(np.mean(inv_k[present[:, j]])) if present[:, j].any() else 0.5 for j in range(n_agents)]
        )
        prior_mean = chance.copy()
        prior_strength = np.full(n_agents, self.peer_strength)
        if self.anchored:
            prior_mean[own] = max(self.anchor_mean, chance[own] + self.anchor_margin)
            prior_strength[own] = self.anchor_strength
        onehot = np.zeros((n_tasks, n_agents, k_max))
        tt, jj = np.nonzero(present)
        onehot[tt, jj, reports[tt, jj]] = 1.0
        log_km1 = np.log(np.maximum(n_cand - 1, 1)).astype(float)

        q = self._initial_posterior(reports, n_cand, temper, own, k_max, init)
        accuracy = np.full(n_agents, 0.5)
        converged = False
        loglik = float("nan")
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            # M-step: expected agreement of each channel with the latent truth.
            agree = np.einsum("tjk,tk->tj", onehot, q)  # P(y_t = r_tj)
            weight = tw[:, None] * present
            s = (agree * weight).sum(axis=0)
            n = weight.sum(axis=0)
            new = (s + prior_strength * prior_mean) / (n + prior_strength)
            new = np.clip(new, self.eps, 1.0 - self.eps)
            if self.anchored:
                new[own] = max(new[own], min(1.0 - self.eps, chance[own] + self.anchor_margin))
            if self.cap == "self":
                peers = np.arange(n_agents) != own
                new[peers] = np.minimum(new[peers], new[own])
            # E-step: signed log-odds votes.
            lam = np.log(new)[None, :] + log_km1[:, None] - np.log1p(-new)[None, :]
            logits = np.einsum("tjk,tj->tk", onehot, temper * lam)
            logits = np.where(valid, logits, -np.inf)
            logits[~multi] = np.where(valid[~multi], 0.0, -np.inf)
            m = logits.max(axis=1, keepdims=True)
            m = np.where(np.isfinite(m), m, 0.0)
            e = np.exp(logits - m)
            z = e.sum(axis=1, keepdims=True)
            q_new = np.where(z > 0, e / np.maximum(z, 1e-300), 0.0)
            change = float(np.max(np.abs(new - accuracy)))
            accuracy, q = new, q_new
            if change < self.tol:
                converged = True
                break
        # Full tempered observed-data log-likelihood (the c-independent terms were
        # dropped from the E-step logits but matter when comparing solutions).
        lam = np.log(accuracy)[None, :] + log_km1[:, None] - np.log1p(-accuracy)[None, :]
        logits = np.einsum("tjk,tj->tk", onehot, temper * lam)
        logits = np.where(valid, logits, -np.inf)
        m = logits.max(axis=1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        z = np.exp(logits - m).sum(axis=1)
        base = (temper * present * (np.log1p(-accuracy)[None, :] - log_km1[:, None])).sum(axis=1)
        per_task = np.where(n_cand > 0, m[:, 0] + np.log(np.maximum(z, 1e-300)) + base - np.log(np.maximum(n_cand, 1)), 0.0)
        per_task = np.where(n_cand >= 2, per_task, 0.0)
        loglik = float(np.sum(tw * per_task))
        # Beta(k*m + 1, k*(1-m) + 1) priors -> the MAP update used in the M-step.
        a = np.clip(accuracy, self.eps, 1 - self.eps)
        log_prior = float(np.sum(prior_strength * (prior_mean * np.log(a) + (1 - prior_mean) * np.log1p(-a))))
        mean_k = float(np.mean(n_cand[n_cand > 0])) if np.any(n_cand > 0) else 2.0
        return ReceiverFit(agents, accuracy, groups, n_obs, mean_k, iteration, converged, loglik,
                           log_posterior=loglik + log_prior, init=init, posterior=q)

    def _em_full(
        self,
        reports: np.ndarray,
        temper: np.ndarray,
        tw: np.ndarray,
        own: int,
        agents: tuple[int, ...],
        groups: np.ndarray,
        init: str = "self",
    ) -> ReceiverFit:
        n_tasks, n_agents = reports.shape
        c = len(self.label_space or ())
        present = reports >= 0
        n_obs = present.sum(axis=0)
        onehot = np.zeros((n_tasks, n_agents, c))
        tt, jj = np.nonzero(present)
        onehot[tt, jj, reports[tt, jj]] = 1.0
        n_cand = np.full(n_tasks, c)
        q = self._initial_posterior(reports, n_cand, temper, own, c, init)
        # Dirichlet priors: peers uninformative (uniform rows), receiver diagonal.
        prior = np.full((n_agents, c, c), self.peer_strength / c)
        if self.anchored:
            off = (1.0 - self.anchor_mean) / max(c - 1, 1)
            anchor = np.full((c, c), off)
            np.fill_diagonal(anchor, self.anchor_mean)
            prior[own] = self.anchor_strength * anchor
        confusion = np.full((n_agents, c, c), 1.0 / c)
        rho = np.full(c, 1.0 / c)
        converged = False
        loglik = float("nan")
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            counts = np.einsum("tk,tjl->jkl", q * tw[:, None], onehot)
            new = counts + prior
            new /= new.sum(axis=2, keepdims=True)
            new = np.clip(new, self.eps, 1.0)
            new /= new.sum(axis=2, keepdims=True)
            if self.anchored:
                diag = np.diag(new[own]).copy()
                floor = 1.0 / c + self.anchor_margin
                if np.any(diag < floor):
                    for k in range(c):
                        if new[own, k, k] < floor:
                            rest = 1.0 - floor
                            others = [x for x in range(c) if x != k]
                            tot = new[own, k, others].sum()
                            new[own, k, others] = new[own, k, others] / max(tot, 1e-12) * rest
                            new[own, k, k] = floor
            rho_new = (q * tw[:, None]).sum(axis=0) + 1.0
            rho_new /= rho_new.sum()
            log_conf = np.log(new)  # (J, C_true, C_reported)
            logits = np.log(rho_new)[None, :] + np.einsum("tjl,jkl,tj->tk", onehot, log_conf, temper)
            m = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - m)
            z = e.sum(axis=1, keepdims=True)
            q = e / z
            change = float(np.max(np.abs(new - confusion)))
            confusion, rho = new, rho_new
            loglik = float(np.sum(tw * (m[:, 0] + np.log(z[:, 0]))))
            if change < self.tol:
                converged = True
                break
        accuracy = np.einsum("k,jkk->j", rho, confusion)
        log_prior = float(np.sum(prior * np.log(confusion)))
        return ReceiverFit(
            agents, accuracy, groups, n_obs, float(c), iteration, converged, loglik,
            confusion=confusion, class_prior=rho, log_posterior=loglik + log_prior, init=init, posterior=q,
        )

    def _describe(self, fit: ReceiverFit, own: int) -> dict[int, ChannelEstimate]:
        out: dict[int, ChannelEstimate] = {}
        k = max(fit.mean_k, 2.0)
        binary_full = fit.confusion is not None and fit.confusion.shape[1] == 2
        for j, agent in enumerate(fit.agents):
            a = float(np.clip(fit.accuracy[j], self.eps, 1 - self.eps))
            if binary_full:
                # Class-conditional channel: its weight is half the log diagnostic odds
                # ratio, i.e. the mean log-likelihood ratio a report adds for the answer
                # it names. It is ~0 for "always yes" and negative for an inverter.
                p = np.clip(fit.confusion[j], self.eps, 1.0)
                lam = float(0.5 * (np.log(p[0, 0]) + np.log(p[1, 1]) - np.log(p[0, 1]) - np.log(p[1, 0])))
            else:
                lam = float(np.log(a * (k - 1) / (1 - a)))
            if j == own:
                decision = TRUST
            elif fit.n_observed[j] == 0:
                decision = DISCARD
            elif lam > self.decision_band:
                decision = TRUST
            elif lam < -self.decision_band:
                decision = INVERT
            else:
                decision = DISCARD
            out[agent] = ChannelEstimate(a, int(fit.n_observed[j]), int(fit.groups[j]), lam, decision)
        return out

    # ------------------------------------------------------------ aggregate

    def posterior(self, observations: Sequence[Broadcast], self_id: int) -> dict[str, float]:
        row = {b.agent_id: b.answer for b in observations}
        cands = self._candidates(row)
        if not cands:
            return {}
        fit = self.fits.get(self_id)
        index = {c: i for i, c in enumerate(cands)}
        k = len(cands)
        if fit is None or k == 1:
            own = row.get(self_id)
            probs = np.full(k, 1.0 / k)
            if own in index and k > 1:
                probs[:] = 0.0
                probs[index[own]] = 1.0
            return dict(zip(cands, probs.tolist(), strict=True))
        # Clone tempering among the members actually observed on this task.
        group_count: dict[int, int] = {}
        cols: dict[int, int] = {}
        for agent, answer in row.items():
            col = fit.column(agent)
            if col is None or answer is None or answer not in index:
                continue
            cols[agent] = col
            g = int(fit.groups[col])
            group_count[g] = group_count.get(g, 0) + 1
        if self.model == "full" and fit.confusion is not None:
            logits = np.log(fit.class_prior).copy()
            for agent, col in cols.items():
                w = 1.0 / group_count[int(fit.groups[col])]
                logits += w * np.log(fit.confusion[col][:, index[row[agent]]])
        else:
            logits = np.zeros(k)
            for agent, col in cols.items():
                a = float(np.clip(fit.accuracy[col], self.eps, 1 - self.eps))
                lam = np.log(a) + np.log(k - 1) - np.log1p(-a)
                w = 1.0 / group_count[int(fit.groups[col])]
                logits[index[row[agent]]] += w * lam
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return dict(zip(cands, probs.tolist(), strict=True))

    def evidence(self, observations: Sequence[Broadcast], self_id: int) -> dict[int, tuple[str, float]]:
        """Per-agent signed log-odds added to its answer: ``w * lambda`` (one-coin model).

        ``softmax`` of the per-candidate sums reproduces :meth:`posterior`; this is
        the decomposition the explainer film and ``TrustLayer`` display."""
        row = {b.agent_id: b.answer for b in observations}
        cands = self._candidates(row)
        fit = self.fits.get(self_id)
        if fit is None or len(cands) < 2:
            return {}
        if self.model == "full":
            if len(cands) != 2 or fit.confusion is None:
                raise NotImplementedError("per-agent evidence of the full model is defined for binary questions")
            return self._evidence_binary(row, cands, fit)
        k = len(cands)
        cols = {a: fit.column(a) for a, ans in row.items() if ans in cands and fit.column(a) is not None}
        group_count: dict[int, int] = {}
        for col in cols.values():
            group_count[int(fit.groups[col])] = group_count.get(int(fit.groups[col]), 0) + 1
        out = {}
        for agent, col in cols.items():
            a = float(np.clip(fit.accuracy[col], self.eps, 1 - self.eps))
            lam = np.log(a) + np.log(k - 1) - np.log1p(-a)
            out[agent] = (row[agent], float(lam / group_count[int(fit.groups[col])]))
        return out

    def _evidence_binary(self, row, cands, fit) -> dict[int, tuple[str, float]]:
        """Binary class-conditional decomposition. Each report adds
        ``w * log(pi[r, r] / pi[other, r])`` to the answer it names; key ``-1`` is
        the class prior's log-odds, credited to the more frequent class."""
        index = {c: i for i, c in enumerate(cands)}
        cols = {a: fit.column(a) for a, ans in row.items() if ans in index and fit.column(a) is not None}
        group_count: dict[int, int] = {}
        for col in cols.values():
            group_count[int(fit.groups[col])] = group_count.get(int(fit.groups[col]), 0) + 1
        out: dict[int, tuple[str, float]] = {}
        for agent, col in cols.items():
            r = index[row[agent]]
            conf = fit.confusion[col]
            out[agent] = (row[agent], float((np.log(conf[r, r]) - np.log(conf[1 - r, r])) / group_count[int(fit.groups[col])]))
        prior = np.log(fit.class_prior)
        top = int(np.argmax(prior))
        out[-1] = (cands[top], float(prior[top] - prior[1 - top]))
        return out

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        post = self.posterior(observations, self_id)
        if not post:
            return None
        best = max(post.values())
        tied = sorted(c for c, p in post.items() if np.isclose(p, best, rtol=1e-9, atol=1e-12))
        own = next((b.answer for b in observations if b.agent_id == self_id), None)
        if own in tied:
            return own
        return tied[0]
