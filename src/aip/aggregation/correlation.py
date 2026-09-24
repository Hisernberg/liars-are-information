"""Pairwise error-correlation estimators, including the blind estimator.

This module answers the headline Phase B question: *are homogeneous swarms
epistemically one agent?* It measures how much two agents' errors coincide, two
ways.

**With ground truth.** The phi (Matthews) coefficient on error indicator vectors
-- the Pearson correlation of two binary variables -- plus the raw co-error rate
and the answer-agreement rate.

**Without ground truth (the attenuation law).** Write ``s_i = +1`` when agent
``i`` is correct and ``-1`` when it is wrong, so ``E[s_i] = 1 - 2 e_i``. If the
answer space is binary and errors are independent, two agents agree exactly when
``s_i == s_j``::

    P(agree) = (1-e_i)(1-e_j) + e_i e_j
    2 P(agree) - 1 = 1 - 2e_i - 2e_j + 4 e_i e_j = (1-2e_i)(1-2e_j)

giving the estimator this project validates::

    (1-2e_i)(1-2e_j) / 2  =  P(agree) - 1/2

Agreement is observable from broadcasts alone, so an agent can estimate the
quality product of any pair it can hear without ever seeing a label.

The binary assumption is load-bearing and is *not* satisfied by these
benchmarks. With a ``K``-ary answer space and wrong answers spread uniformly,
``P(agree) = (1-e_i)(1-e_j) + e_i e_j / (K-1)``, so the two-way law understates
the co-error term -- badly for open-ended answers, where two wrong agents almost
never coincide. Quantifying that gap is the point of the validation, so the raw
agreement rate is always stored alongside the estimate and nothing is silently
corrected.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def error_vector(is_correct: np.ndarray) -> np.ndarray:
    """Error indicators (1 = wrong) from a correctness vector."""
    return 1 - np.asarray(is_correct, dtype=int)


def phi_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Matthews/phi correlation between two binary vectors.

    Returns NaN when either vector is constant: with no variation there is no
    correlation to estimate, and reporting 0.0 would falsely claim independence
    where the data are simply uninformative.
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    if x.size == 0 or x.size != y.size:
        return float("nan")

    n11 = float(np.sum((x == 1) & (y == 1)))
    n10 = float(np.sum((x == 1) & (y == 0)))
    n01 = float(np.sum((x == 0) & (y == 1)))
    n00 = float(np.sum((x == 0) & (y == 0)))

    denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if denom <= 0:
        return float("nan")
    return float((n11 * n00 - n10 * n01) / np.sqrt(denom))


def co_error_rate(e_i: np.ndarray, e_j: np.ndarray) -> float:
    """Fraction of tasks where both agents are wrong."""
    e_i = np.asarray(e_i, dtype=int)
    e_j = np.asarray(e_j, dtype=int)
    if e_i.size == 0:
        return float("nan")
    return float(np.mean((e_i == 1) & (e_j == 1)))


def answer_agreement(a_i: np.ndarray, a_j: np.ndarray) -> float:
    """Fraction of tasks where two agents broadcast the same answer.

    A missing answer (failed extraction) never agrees with anything, including
    another missing answer: two agents that both failed to produce a parseable
    answer have not reached the same conclusion.
    """
    a_i = np.asarray(a_i, dtype=object)
    a_j = np.asarray(a_j, dtype=object)
    if a_i.size == 0:
        return float("nan")
    both_present = np.array(
        [(x is not None) and (y is not None) for x, y in zip(a_i, a_j, strict=True)]
    )
    same = np.array([x == y for x, y in zip(a_i, a_j, strict=True)])
    return float(np.mean(same & both_present))


def correctness_agreement(c_i: np.ndarray, c_j: np.ndarray) -> float:
    """Fraction of tasks where two agents' *correctness* coincides.

    .. warning::
       This is **not** a blind observable and must never feed a blind estimator:
       computing it requires knowing who was right. It equals
       :func:`answer_agreement` only for a genuinely binary answer space, which
       is exactly why misusing it hid on GSM8K and showed up on MATH-500.

    It is reported as a diagnostic -- the gap between it and answer agreement is
    the co-error dispersion ``e_i e_j (1 - q_ij)`` -- and it is the right input
    when *simulating* the idealised binary channel in tests.
    """
    c_i = np.asarray(c_i, dtype=int)
    c_j = np.asarray(c_j, dtype=int)
    if c_i.size == 0:
        return float("nan")
    return float(np.mean(c_i == c_j))


# -- the blind estimator ---------------------------------------------------


def blind_product_estimate(agreement_rate: float) -> float:
    """Attenuation law: estimate ``(1-2e_i)(1-2e_j)/2`` from agreement alone.

    ``agreement_rate`` must be **answer** agreement -- what an agent can observe
    on the wire. Feeding it correctness agreement smuggles the labels back in.
    """
    return float(agreement_rate) - 0.5


def true_product(e_i_rate: float, e_j_rate: float) -> float:
    """The ground-truth quantity ``(1-2e_i)(1-2e_j)/2`` from known error rates."""
    return float((1.0 - 2.0 * e_i_rate) * (1.0 - 2.0 * e_j_rate) / 2.0)


def blind_marginal_error_rates(
    agreement: dict[tuple[str, str], float], names: list[str]
) -> dict[str, float]:
    """Recover each agent's marginal error rate from pairwise agreement only.

    With ``c_i = 1 - 2 e_i`` and ``m_ij = 2 P(agree_ij) - 1 = c_i c_j``, any
    triplet gives ``c_i^2 = m_ij m_ik / m_jk``. Averaging over every triplet that
    contains ``i`` damps the noise in individual pairs. This is the standard
    third-moment identity behind spectral crowdsourcing estimators, and it is
    what lets an agent estimate *its own* reliability with no labels at all.

    Agents are assumed better than chance, so the positive root is taken.
    Triplets whose implied square is negative or whose denominator is ~0 are
    skipped rather than clamped, and an agent with no usable triplet gets NaN.
    """

    def m(a: str, b: str) -> float:
        key = (a, b) if (a, b) in agreement else (b, a)
        rate = agreement.get(key)
        return float("nan") if rate is None else 2.0 * rate - 1.0

    out: dict[str, float] = {}
    for i in names:
        others = [n for n in names if n != i]
        estimates: list[float] = []
        for j, k in itertools.combinations(others, 2):
            m_ij, m_ik, m_jk = m(i, j), m(i, k), m(j, k)
            if not all(np.isfinite([m_ij, m_ik, m_jk])) or abs(m_jk) < 1e-9:
                continue
            squared = (m_ij * m_ik) / m_jk
            if squared <= 0:
                continue
            estimates.append(float(np.sqrt(squared)))
        if not estimates:
            out[i] = float("nan")
            continue
        c_i = float(np.clip(np.mean(estimates), -1.0, 1.0))
        out[i] = (1.0 - c_i) / 2.0
    return out


# -- effective swarm size --------------------------------------------------


def effective_swarm_size(n_agents: int, rho_bar: float) -> float:
    """``N_eff = N / (1 + (N-1) rho_bar)``.

    The number of *independent* agents a correlated swarm is worth. At
    ``rho_bar = 0`` it is ``N``; as ``rho_bar`` approaches 1 it approaches 1 --
    the swarm collapses to a single epistemic agent.
    """
    if not np.isfinite(rho_bar):
        return float("nan")
    denom = 1.0 + (n_agents - 1) * rho_bar
    if denom <= 0:
        return float("inf")
    return float(n_agents / denom)


def mixed_swarm_rho_bar(
    n_agents: int, rho_within_a: float, rho_within_b: float, rho_cross: float
) -> float:
    """Mean pairwise rho for an ``N``-agent swarm split evenly between two models.

    A 5+5 swarm is not simply ``rho_cross``: of its 45 pairs, 20 are same-model
    (and therefore more correlated) and only 25 are cross-model. Averaging over
    the real pair census is what keeps the ``N_eff`` table honest.
    """
    half = n_agents // 2
    other = n_agents - half
    n_aa = half * (half - 1) // 2
    n_bb = other * (other - 1) // 2
    n_ab = half * other
    total = n_aa + n_bb + n_ab
    if total == 0:
        return float("nan")
    return float((n_aa * rho_within_a + n_bb * rho_within_b + n_ab * rho_cross) / total)


def allocate_agents(n_agents: int, n_models: int) -> list[int]:
    """Split ``n_agents`` across ``n_models`` as evenly as possible.

    Ten agents over three models is 4/3/3, not 3.33 each. The remainder has to
    go somewhere, and where it goes changes the pair census -- so it is
    allocated deterministically (earlier models take the surplus) rather than
    being averaged away.
    """
    if n_models <= 0:
        raise ValueError("n_models must be positive")
    base, extra = divmod(n_agents, n_models)
    return [base + (1 if i < extra else 0) for i in range(n_models)]


def subset_swarm_rho_bar(
    counts: list[int],
    rho_within: list[float],
    rho_cross: dict[tuple[int, int], float],
    *,
    include_within: bool = True,
) -> float:
    """Mean pairwise rho for a swarm built from several models.

    Generalises :func:`mixed_swarm_rho_bar` to any number of models. The average
    is over the real pair census: ``C(n_m, 2)`` same-model pairs for each model
    and ``n_m * n_m'`` cross pairs for each model pair.

    ``include_within=False`` drops the same-model pairs from the average, which
    is the only honest option on the benchmarks where within-model rho was never
    measured. It yields an **upper bound** on ``N_eff``, because same-model pairs
    are the more correlated ones and omitting them understates rho.
    """
    k = len(counts)
    if k != len(rho_within):
        raise ValueError("counts and rho_within must have equal length")
    weight = 0.0
    total = 0.0
    for i in range(k):
        n_i = counts[i]
        if include_within and n_i >= 2:
            pairs = n_i * (n_i - 1) / 2
            rho = rho_within[i]
            if not np.isfinite(rho):
                return float("nan")
            total += pairs * rho
            weight += pairs
        for j in range(i + 1, k):
            pairs = n_i * counts[j]
            rho = rho_cross.get((i, j), rho_cross.get((j, i), float("nan")))
            if not np.isfinite(rho):
                return float("nan")
            total += pairs * rho
            weight += pairs
    if weight == 0:
        return float("nan")
    return float(total / weight)


@dataclass
class PairEstimate:
    """All Phase B quantities for one pair on one benchmark."""

    benchmark: str
    model_i: str
    model_j: str
    slot_i: int
    slot_j: int
    kind: str  # "within_model" | "cross_model"
    n_tasks: int
    error_rate_i: float
    error_rate_j: float
    extra: dict[str, Any] = field(default_factory=dict)


# -- the multiclass law (Phase B2) -----------------------------------------
#
# Everything above this line is the binary-law path from Phase B and is
# deliberately left untouched: it is the published baseline the multiclass
# estimator has to beat, and `tests/test_correlation.py` pins its behaviour.


def chance_q(n_classes: int) -> float:
    """Chance level for ``q`` with ``C`` options: ``1/(C-1)``.

    The null hypothesis for honest agents on a closed answer space. MMLU's
    ``C = 4`` gives ``1/3``; a measured ``q`` materially above it means two models
    are drawn to the *same* distractor, which is a shared error mechanism rather
    than bad luck.
    """
    if n_classes < 2:
        raise ValueError("need at least two answer options")
    return 1.0 / (n_classes - 1)


def generalized_agreement(e_i: float, e_j: float, q_ij: float) -> float:
    """The law itself: ``m = (1-e_i)(1-e_j) + e_i e_j q``."""
    return float((1.0 - e_i) * (1.0 - e_j) + e_i * e_j * q_ij)


def q_empirical(
    ans_i: np.ndarray, ans_j: np.ndarray, correct_i: np.ndarray, correct_j: np.ndarray
) -> float:
    """Ground-truth ``q_ij = P(same answer | both wrong)``.

    **Validation only.** This reads correctness, so no aggregator may ever call
    it; it exists to check what the blind estimator recovers. Returns NaN when
    the pair is never jointly wrong, since the conditional is then undefined --
    reporting 0.0 would assert that they never coincide when in fact they were
    never tested.
    """
    ans_i = np.asarray(ans_i, dtype=object)
    ans_j = np.asarray(ans_j, dtype=object)
    both_wrong = (np.asarray(correct_i, dtype=int) == 0) & (np.asarray(correct_j, dtype=int) == 0)
    if not both_wrong.any():
        return float("nan")
    same = np.array(
        [
            (x is not None) and (y is not None) and (x == y)
            for x, y in zip(ans_i, ans_j, strict=True)
        ]
    )
    return float(np.mean(same[both_wrong]))


@dataclass
class MulticlassBlindEstimate:
    """Blind (label-free) estimates of marginals, pairwise ``q``, and products."""

    agents: list[str]
    error_rates: dict[str, float]
    q: dict[tuple[str, str], float]
    products: dict[tuple[str, str], float]
    implied_agreement: dict[tuple[str, str], float]
    n_tasks: int
    converged: bool
    n_iter: int
    ds: Any = None

    def pair(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if (a, b) in self.q else (b, a)


def blind_estimate_mc(
    answers: list[dict[str, Any]],
    agents: list[str] | None = None,
    label_space: list[Any] | None = None,
    **em_kwargs: Any,
) -> MulticlassBlindEstimate:
    """Jointly estimate ``e_i`` and ``q_ij`` from answer patterns, without labels.

    Pairwise agreement alone cannot identify both (see
    ``docs/attenuation_law.md`` §5: the system is short by ``n`` equations for
    every ``n``). This routine escapes that by fitting the *full* answer-pattern
    distribution with a one-coin Dawid-Skene model, then reading both quantities
    off the resulting posterior over latent true labels as posterior-weighted
    empirical frequencies::

        e_i   = 1 - E[ 1(a_i = y) ]
        q_ij  = E[ 1(a_i = a_j, a_i != y, a_j != y) ] / E[ 1(a_i != y, a_j != y) ]

    with the expectation over the posterior rather than over gold labels. The
    identifiability this buys is bought with the conditional-independence
    assumption, which correlated errors violate -- so a blind ``q`` is biased
    *downward* exactly when agents share attractors. That is the conservative
    direction for a detector of shared mechanism, and it means a blind ``q`` that
    already looks high is, if anything, an underestimate.
    """
    from aip.aggregation.baselines import dawid_skene_em

    if agents is None:
        agents = sorted({a for row in answers for a in row})
    agents = list(agents)

    ds = dawid_skene_em(answers, agents=agents, label_space=label_space, **em_kwargs)

    # Posterior-weighted marginal error rates come straight from the M step.
    error_rates = dict(ds.error_rates)

    q: dict[tuple[str, str], float] = {}
    implied: dict[tuple[str, str], float] = {}
    products: dict[tuple[str, str], float] = {}

    for a, b in itertools.combinations(agents, 2):
        num = 0.0
        den = 0.0
        for row, post in zip(answers, ds.posterior, strict=True):
            ai, aj = row.get(a), row.get(b)
            if not post:
                continue
            same = (ai is not None) and (aj is not None) and (ai == aj)
            for label, weight in post.items():
                if weight <= 0.0:
                    continue
                i_wrong = ai != label
                j_wrong = aj != label
                if i_wrong and j_wrong:
                    den += weight
                    if same:
                        num += weight
        q[(a, b)] = float(num / den) if den > 0 else float("nan")
        e_a, e_b = error_rates.get(a, np.nan), error_rates.get(b, np.nan)
        products[(a, b)] = true_product(e_a, e_b)
        implied[(a, b)] = generalized_agreement(e_a, e_b, q[(a, b)])

    return MulticlassBlindEstimate(
        agents=agents,
        error_rates=error_rates,
        q=q,
        products=products,
        implied_agreement=implied,
        n_tasks=len(answers),
        converged=ds.converged,
        n_iter=ds.n_iter,
        ds=ds,
    )
