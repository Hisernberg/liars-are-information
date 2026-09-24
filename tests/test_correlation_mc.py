"""Multiclass attenuation law: reduction, recovery, and identifiability limits.

Tolerances are derived, not tuned. An error rate estimated from ``n`` tasks has
standard error at most ``sqrt(e(1-e)/n)``; a ``q`` estimated from the
``n_bw ~ n * e_i * e_j`` tasks where a pair is jointly wrong has standard error
``sqrt(q(1-q)/n_bw)``. Every band below is three of those.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from aip.aggregation.baselines import dawid_skene_em
from aip.aggregation.correlation import (
    blind_estimate_mc,
    blind_product_estimate,
    chance_q,
    correctness_agreement,
    generalized_agreement,
    q_empirical,
    true_product,
)

AGENTS = ["a", "b", "c", "d"]


def band_rate(rate: float, n: int, sigmas: float = 3.0) -> float:
    return sigmas * math.sqrt(max(rate * (1 - rate), 1e-9) / max(n, 1))


def synth_multiclass(
    n: int,
    n_classes: int,
    error_rates: dict[str, float],
    q_target: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, int]], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Multiclass channels with known error rates and a planted ``q``.

    ``q`` is planted with a shared attractor: on a fraction ``lambda`` of tasks
    every wrong agent picks the *same* wrong option, otherwise each picks
    uniformly. That gives ``q = lambda + (1-lambda)/(C-1)``, so the target is hit
    by construction rather than by tuning.
    """
    names = list(error_rates)
    floor = 1.0 / (n_classes - 1)
    lam = 0.0 if n_classes == 2 else max(0.0, min(1.0, (q_target - floor) / (1.0 - floor)))
    truth = rng.integers(0, n_classes, n)
    rows: list[dict[str, int]] = []
    correct = {a: np.zeros(n, dtype=int) for a in names}
    answers = {a: np.empty(n, dtype=object) for a in names}
    for t in range(n):
        wrong_options = [c for c in range(n_classes) if c != truth[t]]
        attractor = int(rng.choice(wrong_options))
        coordinated = rng.random() < lam
        row: dict[str, int] = {}
        for name in names:
            if rng.random() < error_rates[name]:
                value = attractor if coordinated else int(rng.choice(wrong_options))
            else:
                value = int(truth[t])
            row[name] = value
            answers[name][t] = value
            correct[name][t] = int(value == truth[t])
        rows.append(row)
    return rows, correct, answers, truth


class TestReductionToBinary:
    def test_q_is_identically_one_when_c_is_two(self) -> None:
        """With one wrong answer, two wrong agents cannot fail to agree."""
        rng = np.random.default_rng(0)
        rows, correct, answers, _ = synth_multiclass(
            4000, 2, {"a": 0.3, "b": 0.4}, q_target=1.0, rng=rng
        )
        q = q_empirical(answers["a"], answers["b"], correct["a"], correct["b"])
        assert q == pytest.approx(1.0)

    def test_law_reduces_to_the_binary_estimator(self) -> None:
        """m = (1-e_i)(1-e_j) + e_i e_j  =>  (1-2e_i)(1-2e_j)/2 = m - 1/2."""
        for e_i, e_j in [(0.1, 0.2), (0.3, 0.45), (0.05, 0.5)]:
            m = generalized_agreement(e_i, e_j, 1.0)
            assert blind_product_estimate(m) == pytest.approx(true_product(e_i, e_j))

    @pytest.mark.parametrize(
        ("e_i", "e_j", "q"), [(0.4, 0.5, 1 / 3), (0.3, 0.3, 0.0), (0.2, 0.6, 0.5)]
    )
    def test_binary_law_bias_is_exactly_e_i_e_j_times_one_minus_q(self, e_i, e_j, q) -> None:
        """The derived bias: applying the binary law when q < 1 overstates by e_i e_j (1-q)."""
        m = generalized_agreement(e_i, e_j, q)
        bias = blind_product_estimate(m) - true_product(e_i, e_j)
        assert bias == pytest.approx(-e_i * e_j * (1.0 - q), abs=1e-12)
        assert bias <= 0.0, "the binary law can only overstate the product"


class TestChanceLevel:
    def test_known_values(self) -> None:
        assert chance_q(2) == pytest.approx(1.0)
        assert chance_q(4) == pytest.approx(1 / 3)
        assert chance_q(11) == pytest.approx(0.1)

    def test_rejects_degenerate_space(self) -> None:
        with pytest.raises(ValueError):
            chance_q(1)

    def test_independent_wrong_answers_land_at_chance(self) -> None:
        rng = np.random.default_rng(4)
        n, c = 20000, 4
        rows, correct, answers, _ = synth_multiclass(
            n, c, {"a": 0.4, "b": 0.4}, q_target=chance_q(c), rng=rng
        )
        q = q_empirical(answers["a"], answers["b"], correct["a"], correct["b"])
        n_bw = int(np.sum((correct["a"] == 0) & (correct["b"] == 0)))
        assert q == pytest.approx(chance_q(c), abs=band_rate(chance_q(c), n_bw))


class TestBlindMulticlassRecovery:
    """The estimator must recover e, q and the product with no labels."""

    ERRORS = {"a": 0.20, "b": 0.35, "c": 0.30, "d": 0.25}
    N = 6000

    def _fit(self, n_classes: int, q_target: float, seed: int):
        rng = np.random.default_rng(seed)
        rows, correct, answers, _ = synth_multiclass(self.N, n_classes, self.ERRORS, q_target, rng)
        est = blind_estimate_mc(rows, agents=AGENTS, label_space=list(range(n_classes)))
        return est, correct, answers

    def test_recovers_error_rates_at_chance_q(self) -> None:
        est, _, _ = self._fit(4, chance_q(4), seed=1)
        assert est.converged
        for agent, true_e in self.ERRORS.items():
            assert est.error_rates[agent] == pytest.approx(true_e, abs=band_rate(true_e, self.N))

    def test_recovers_q_at_chance(self) -> None:
        est, correct, answers = self._fit(4, chance_q(4), seed=1)
        for a, b in itertools.combinations(AGENTS, 2):
            truth = q_empirical(answers[a], answers[b], correct[a], correct[b])
            n_bw = int(np.sum((correct[a] == 0) & (correct[b] == 0)))
            assert est.q[(a, b)] == pytest.approx(truth, abs=band_rate(truth, n_bw))

    def test_recovers_pairwise_product_at_chance_q(self) -> None:
        est, correct, _ = self._fit(4, chance_q(4), seed=1)
        for a, b in itertools.combinations(AGENTS, 2):
            truth = true_product(float(1 - np.mean(correct[a])), float(1 - np.mean(correct[b])))
            assert est.products[(a, b)] == pytest.approx(truth, abs=0.05)

    def test_recovers_near_zero_q_on_a_wide_answer_space(self) -> None:
        """The open-ended regime: many options, so coincidence is rare."""
        est, correct, answers = self._fit(10, 0.10, seed=2)
        for a, b in itertools.combinations(AGENTS, 2):
            truth = q_empirical(answers[a], answers[b], correct[a], correct[b])
            n_bw = int(np.sum((correct[a] == 0) & (correct[b] == 0)))
            assert est.q[(a, b)] == pytest.approx(truth, abs=band_rate(truth, n_bw) + 0.02)

    def test_detects_a_shared_attractor_well_above_chance(self) -> None:
        """The inversion signal: coordinated wrongness must be visible blind."""
        est, correct, answers = self._fit(4, 0.80, seed=3)
        for a, b in itertools.combinations(AGENTS, 2):
            truth = q_empirical(answers[a], answers[b], correct[a], correct[b])
            assert est.q[(a, b)] > chance_q(4) + 0.15, "shared attractor went undetected"
            # Documented direction: coordination violates the conditional
            # independence the fit assumes, so the blind q understates the truth.
            assert est.q[(a, b)] <= truth + 0.05

    def test_blind_estimate_never_sees_labels(self) -> None:
        """Permuting the latent truth must not change what the estimator is given."""
        rng = np.random.default_rng(9)
        rows, _, _, _ = synth_multiclass(400, 4, self.ERRORS, chance_q(4), rng)
        first = blind_estimate_mc(rows, agents=AGENTS, label_space=[0, 1, 2, 3])
        second = blind_estimate_mc(rows, agents=AGENTS, label_space=[0, 1, 2, 3])
        assert first.error_rates == second.error_rates


class TestDawidSkeneCore:
    def test_recovers_labels_better_than_majority_with_uneven_competence(self) -> None:
        """One strong agent outvoted by three weak ones is DS's canonical win."""
        rng = np.random.default_rng(12)
        errors = {"strong": 0.05, "w1": 0.6, "w2": 0.6, "w3": 0.6}
        rows, correct, answers, truth = synth_multiclass(3000, 4, errors, chance_q(4), rng)
        ds = dawid_skene_em(rows, agents=list(errors), label_space=[0, 1, 2, 3])
        ds_acc = float(np.mean([m == t for m, t in zip(ds.map_labels(), truth, strict=True)]))
        majority = []
        for row in rows:
            counts: dict[int, int] = {}
            for v in row.values():
                counts[v] = counts.get(v, 0) + 1
            majority.append(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])
        maj_acc = float(np.mean([m == t for m, t in zip(majority, truth, strict=True)]))
        assert ds_acc > maj_acc + 0.05

    def test_missing_answers_do_not_shift_the_posterior(self) -> None:
        """A None is uninformative about y, but still costs the agent competence."""
        rows = [{"a": 0, "b": 1, "c": None} for _ in range(50)]
        ds = dawid_skene_em(rows, agents=["a", "b", "c"], label_space=[0, 1])
        assert ds.error_rates["c"] == pytest.approx(1.0, abs=1e-6)
        post = ds.posterior[0]
        assert set(post) == {0, 1}

    def test_empty_input(self) -> None:
        ds = dawid_skene_em([], agents=["a"])
        assert ds.posterior == [] and ds.converged

    def test_unanimous_swarm_is_degenerate_but_stable(self) -> None:
        rows = [{"a": 1, "b": 1, "c": 1} for _ in range(30)]
        ds = dawid_skene_em(rows, agents=["a", "b", "c"])
        assert all(math.isclose(e, 0.0, abs_tol=1e-6) for e in ds.error_rates.values())


class TestQEmpirical:
    def test_basic(self) -> None:
        ans_i = np.array(["X", "X", "A"], dtype=object)
        ans_j = np.array(["X", "Y", "A"], dtype=object)
        correct = np.array([0, 0, 1])
        assert q_empirical(ans_i, ans_j, correct, correct) == pytest.approx(0.5)

    def test_undefined_when_never_jointly_wrong(self) -> None:
        ans = np.array(["A", "A"], dtype=object)
        assert math.isnan(q_empirical(ans, ans, np.array([1, 1]), np.array([1, 1])))

    def test_missing_answers_never_coincide(self) -> None:
        ans_i = np.array([None, None], dtype=object)
        ans_j = np.array([None, None], dtype=object)
        assert q_empirical(ans_i, ans_j, np.array([0, 0]), np.array([0, 0])) == 0.0


class TestCorrelatedCorrectnessBias:
    def test_blind_q_understates_when_correctness_is_correlated(self) -> None:
        """Documented limitation, measured rather than assumed."""
        rng = np.random.default_rng(31)
        n, c = 8000, 4
        truth = rng.integers(0, c, n)
        hard = rng.random(n) < 0.35  # tasks everyone tends to miss
        rows, correct, answers = [], {}, {}
        for name in AGENTS:
            correct[name] = np.zeros(n, dtype=int)
            answers[name] = np.empty(n, dtype=object)
        for t in range(n):
            wrong_options = [k for k in range(c) if k != truth[t]]
            attractor = int(rng.choice(wrong_options))
            row = {}
            for name in AGENTS:
                p_err = 0.75 if hard[t] else 0.05
                if rng.random() < p_err:
                    value = attractor if rng.random() < 0.7 else int(rng.choice(wrong_options))
                else:
                    value = int(truth[t])
                row[name] = value
                answers[name][t] = value
                correct[name][t] = int(value == truth[t])
            rows.append(row)
        est = blind_estimate_mc(rows, agents=AGENTS, label_space=list(range(c)))
        gaps = []
        for a, b in itertools.combinations(AGENTS, 2):
            gaps.append(q_empirical(answers[a], answers[b], correct[a], correct[b]) - est.q[(a, b)])
        assert np.mean(gaps) > 0, "blind q should understate under correlated errors"


class TestBinaryPathUnchanged:
    """The Phase B estimator must keep behaving exactly as published."""

    def test_binary_product_estimate_is_untouched(self) -> None:
        assert blind_product_estimate(0.75) == pytest.approx(0.25)
        assert blind_product_estimate(0.5) == pytest.approx(0.0)

    def test_correctness_agreement_untouched(self) -> None:
        assert correctness_agreement(
            np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])
        ) == pytest.approx(0.75)
