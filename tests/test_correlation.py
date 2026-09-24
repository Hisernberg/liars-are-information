"""Correlation estimators, validated against synthetic channels with known truth.

The blind estimator is the paper's instrument, so it is tested the only way an
instrument can be: on channels where the answer is known by construction. The
tolerances are derived from binomial sampling noise rather than tuned until the
tests pass -- at ``n`` tasks an agreement rate has standard error at most
``sqrt(0.25/n)``, so a 3-sigma band is ``3 * sqrt(0.25/n)`` (0.15 at n=100).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aip.aggregation.correlation import (
    answer_agreement,
    blind_marginal_error_rates,
    blind_product_estimate,
    co_error_rate,
    correctness_agreement,
    effective_swarm_size,
    error_vector,
    mixed_swarm_rho_bar,
    phi_coefficient,
    true_product,
)


def binomial_band(n: int, sigmas: float = 3.0) -> float:
    """Worst-case ``sigmas``-sigma band for a proportion estimated from n draws."""
    return sigmas * math.sqrt(0.25 / n)


def sample_correlated_errors(
    e_i: float, e_j: float, phi: float, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw two binary error vectors with exact marginals and planted phi.

    Builds the 2x2 joint directly from the identity
    ``cov = phi * sqrt(p_i (1-p_i) p_j (1-p_j))`` and samples from it, so the
    planted correlation is a property of the generating distribution rather than
    an artefact of a particular sampling scheme.
    """
    cov = phi * math.sqrt(e_i * (1 - e_i) * e_j * (1 - e_j))
    p11 = e_i * e_j + cov
    lo, hi = max(0.0, e_i + e_j - 1.0), min(e_i, e_j)
    if not (lo - 1e-12 <= p11 <= hi + 1e-12):
        raise ValueError(f"phi={phi} infeasible for marginals {e_i}, {e_j}")
    p10, p01 = e_i - p11, e_j - p11
    p00 = 1.0 - p11 - p10 - p01
    cells = rng.choice(4, size=n, p=[p11, p10, p01, p00])
    x = np.isin(cells, [0, 1]).astype(int)
    y = np.isin(cells, [0, 2]).astype(int)
    return x, y


def binary_answers(errors: np.ndarray) -> np.ndarray:
    """Binary answer space: correct agents say 'A', wrong agents say 'B'."""
    return np.array(["B" if e else "A" for e in errors], dtype=object)


class TestPhi:
    def test_recovers_planted_correlation(self) -> None:
        rng = np.random.default_rng(0)
        for planted in (0.0, 0.2, 0.5, 0.8):
            x, y = sample_correlated_errors(0.3, 0.4, planted, 20000, rng)
            assert phi_coefficient(x, y) == pytest.approx(planted, abs=0.03)

    def test_perfect_agreement(self) -> None:
        x = np.array([1, 0, 1, 0, 1])
        assert phi_coefficient(x, x) == pytest.approx(1.0)

    def test_perfect_disagreement(self) -> None:
        x = np.array([1, 0, 1, 0])
        assert phi_coefficient(x, 1 - x) == pytest.approx(-1.0)

    def test_constant_vector_is_undefined_not_zero(self) -> None:
        """No variation means no estimate; 0.0 would assert independence."""
        assert math.isnan(phi_coefficient(np.zeros(10, int), np.array([1, 0] * 5)))

    def test_empty(self) -> None:
        assert math.isnan(phi_coefficient(np.array([]), np.array([])))


class TestBasicRates:
    def test_error_vector_inverts_correctness(self) -> None:
        assert list(error_vector(np.array([1, 0, 1]))) == [0, 1, 0]

    def test_co_error_rate(self) -> None:
        assert co_error_rate(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])) == pytest.approx(0.25)

    def test_answer_agreement(self) -> None:
        a = np.array(["A", "B", "C"], dtype=object)
        b = np.array(["A", "B", "D"], dtype=object)
        assert answer_agreement(a, b) == pytest.approx(2 / 3)

    def test_missing_answers_never_agree(self) -> None:
        """Two failed extractions have not reached the same conclusion."""
        a = np.array([None, "A"], dtype=object)
        b = np.array([None, "A"], dtype=object)
        assert answer_agreement(a, b) == pytest.approx(0.5)

    def test_correctness_agreement(self) -> None:
        assert correctness_agreement(
            np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])
        ) == pytest.approx(0.75)


class TestBlindEstimatorOnBinaryChannels:
    """The law holds exactly for a binary answer space; these tests pin that."""

    @pytest.mark.parametrize(
        ("e_i", "e_j"),
        [(0.1, 0.1), (0.2, 0.4), (0.05, 0.5), (0.3, 0.3), (0.45, 0.15)],
    )
    def test_recovers_product_at_large_n(self, e_i: float, e_j: float) -> None:
        rng = np.random.default_rng(7)
        n = 40000
        x, y = sample_correlated_errors(e_i, e_j, 0.0, n, rng)
        agreement = correctness_agreement(1 - x, 1 - y)
        assert blind_product_estimate(agreement) == pytest.approx(
            true_product(e_i, e_j), abs=binomial_band(n)
        )

    @pytest.mark.parametrize(("e_i", "e_j"), [(0.15, 0.35), (0.25, 0.25)])
    def test_recovers_product_at_n100_within_binomial_band(self, e_i, e_j) -> None:
        """The real setting: 100 tasks, so the band is 0.15."""
        band = binomial_band(100)
        rng = np.random.default_rng(11)
        misses = 0
        for _ in range(200):
            x, y = sample_correlated_errors(e_i, e_j, 0.0, 100, rng)
            est = blind_product_estimate(correctness_agreement(1 - x, 1 - y))
            if abs(est - true_product(e_i, e_j)) > band:
                misses += 1
        assert misses <= 4, f"{misses}/200 draws outside a 3-sigma band"

    def test_estimate_uses_no_labels(self) -> None:
        """Sanity: the estimator is a function of agreement alone."""
        assert blind_product_estimate(0.75) == pytest.approx(0.25)
        assert blind_product_estimate(0.5) == pytest.approx(0.0)

    def test_identical_agents_give_max_product(self) -> None:
        rng = np.random.default_rng(3)
        x, _ = sample_correlated_errors(0.2, 0.2, 0.0, 20000, rng)
        assert blind_product_estimate(correctness_agreement(1 - x, 1 - x)) == pytest.approx(0.5)


class TestBlindMarginalRecovery:
    def test_recovers_each_error_rate_at_large_n(self) -> None:
        rng = np.random.default_rng(19)
        truth = {"a": 0.10, "b": 0.25, "c": 0.40, "d": 0.30}
        n = 60000
        errors = {k: (rng.random(n) < v).astype(int) for k, v in truth.items()}
        agreement = {
            (i, j): correctness_agreement(1 - errors[i], 1 - errors[j])
            for i, j in [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]
        }
        recovered = blind_marginal_error_rates(agreement, list(truth))
        for name, true_e in truth.items():
            assert recovered[name] == pytest.approx(true_e, abs=binomial_band(n) * 4)

    def test_n100_recovery_stays_inside_propagated_band(self) -> None:
        """At n=100 the marginal band is wider than the pairwise one, by theory.

        ``c_i = sqrt(m_ij m_ik / m_jk)`` propagates the noise in three agreement
        rates through a ratio and a square root, so the right tolerance is not
        the raw binomial band. With ``SE(m) = 2 sqrt(0.25/n)``, the relative
        error of ``c_i`` is about ``0.5 * sqrt(sum_k (SE(m)/m_k)^2)``, and
        ``SE(e_i) = SE(c_i)/2``. For the well-conditioned marginals below that
        works out near 0.11, so 0.12 is the theory-derived bound.
        """
        rng = np.random.default_rng(23)
        truth = {"a": 0.10, "b": 0.20, "c": 0.30, "d": 0.15}
        pairs = [
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        ]
        errs = []
        for _ in range(120):
            errors = {k: (rng.random(100) < v).astype(int) for k, v in truth.items()}
            agreement = {
                (i, j): correctness_agreement(1 - errors[i], 1 - errors[j]) for i, j in pairs
            }
            rec = blind_marginal_error_rates(agreement, list(truth))
            deltas = [abs(rec[k] - v) for k, v in truth.items() if np.isfinite(rec[k])]
            if deltas:
                errs.append(float(np.mean(deltas)))
        assert len(errs) > 100, "estimator returned NaN too often to assess"
        assert np.mean(errs) < 0.12, f"mean abs error {np.mean(errs):.3f}"

    def test_stays_usable_with_a_near_chance_agent(self) -> None:
        """Robustness that the averaging buys, measured rather than assumed.

        A near-chance agent has ``c = 1-2e`` close to 0, which makes any single
        triplet's denominator ill-conditioned. Averaging over every triplet that
        contains the agent, and skipping the ones whose implied square is
        negative, keeps the recovered marginal inside the same 0.12 band that
        the well-conditioned case meets.
        """
        rng = np.random.default_rng(29)
        truth = {"a": 0.15, "b": 0.30, "c": 0.48, "d": 0.20}  # 'c' is near chance
        pairs = [
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        ]
        errs = []
        for _ in range(120):
            errors = {k: (rng.random(100) < v).astype(int) for k, v in truth.items()}
            agreement = {
                (i, j): correctness_agreement(1 - errors[i], 1 - errors[j]) for i, j in pairs
            }
            rec = blind_marginal_error_rates(agreement, list(truth))
            deltas = [abs(rec[k] - v) for k, v in truth.items() if np.isfinite(rec[k])]
            if deltas:
                errs.append(float(np.mean(deltas)))
        assert len(errs) > 100
        assert np.mean(errs) < 0.12, f"mean abs error {np.mean(errs):.3f}"

    def test_insufficient_agents_yield_nan(self) -> None:
        rec = blind_marginal_error_rates({("a", "b"): 0.8}, ["a", "b"])
        assert all(math.isnan(v) for v in rec.values())


class TestKArySpaceBreaksTheBinaryLaw:
    """The assumption is load-bearing; this documents the direction of the bias."""

    def test_four_way_space_understates_the_product(self) -> None:
        rng = np.random.default_rng(5)
        n, e = 40000, 0.4
        wrong_i = rng.random(n) < e
        wrong_j = rng.random(n) < e
        # 4-way space: wrong agents pick uniformly among the 3 wrong options.
        ans_i = np.where(wrong_i, rng.integers(1, 4, n), 0)
        ans_j = np.where(wrong_j, rng.integers(1, 4, n), 0)
        agreement = float(np.mean(ans_i == ans_j))
        blind = blind_product_estimate(agreement)
        truth = true_product(e, e)
        assert blind < truth, "4-way agreement should sit below the binary law"
        assert abs(blind - truth) > 0.02


class TestEffectiveSwarmSize:
    def test_independent_swarm_keeps_all_agents(self) -> None:
        assert effective_swarm_size(10, 0.0) == pytest.approx(10.0)

    def test_perfect_correlation_collapses_to_one(self) -> None:
        assert effective_swarm_size(10, 1.0) == pytest.approx(1.0)

    def test_monotone_decreasing_in_rho(self) -> None:
        sizes = [effective_swarm_size(10, r) for r in (0.0, 0.1, 0.3, 0.6, 0.9)]
        assert all(a > b for a, b in zip(sizes, sizes[1:], strict=False))

    def test_known_value(self) -> None:
        # N=10, rho=0.5 -> 10 / (1 + 9*0.5) = 1.818...
        assert effective_swarm_size(10, 0.5) == pytest.approx(10 / 5.5)

    def test_nan_rho_propagates(self) -> None:
        assert math.isnan(effective_swarm_size(10, float("nan")))

    def test_mixed_swarm_counts_same_model_pairs(self) -> None:
        """A 5+5 swarm has 20 same-model pairs and 25 cross-model pairs."""
        rho = mixed_swarm_rho_bar(10, 0.8, 0.8, 0.2)
        assert rho == pytest.approx((10 * 0.8 + 10 * 0.8 + 25 * 0.2) / 45)
        assert rho > 0.2, "must exceed the cross-model rho alone"

    def test_mixed_equals_uniform_when_all_rho_equal(self) -> None:
        assert mixed_swarm_rho_bar(10, 0.4, 0.4, 0.4) == pytest.approx(0.4)


class TestSubsetSwarmRhoBar:
    """The k-model generalisation must agree with the 2-model helper it replaces."""

    def test_agrees_with_the_two_model_helper(self) -> None:
        from aip.aggregation.correlation import mixed_swarm_rho_bar, subset_swarm_rho_bar

        expected = mixed_swarm_rho_bar(10, 0.9, 0.5, 0.4)
        got = subset_swarm_rho_bar([5, 5], [0.9, 0.5], {(0, 1): 0.4})
        assert got == pytest.approx(expected)

    def test_homogeneous_reduces_to_within_rho(self) -> None:
        from aip.aggregation.correlation import subset_swarm_rho_bar

        assert subset_swarm_rho_bar([10], [0.65], {}) == pytest.approx(0.65)

    def test_dropping_within_pairs_raises_neff(self) -> None:
        """Omitting same-model pairs understates rho, so it is an upper bound."""
        from aip.aggregation.correlation import effective_swarm_size, subset_swarm_rho_bar

        counts, within, cross = [4, 3, 3], [0.9, 0.9, 0.9], {(0, 1): 0.3, (0, 2): 0.3, (1, 2): 0.3}
        full = subset_swarm_rho_bar(counts, within, cross)
        bound = subset_swarm_rho_bar(counts, within, cross, include_within=False)
        assert bound < full
        assert effective_swarm_size(10, bound) > effective_swarm_size(10, full)

    def test_pair_census_is_the_real_one(self) -> None:
        """4/3/3 has C(4,2)+C(3,2)+C(3,2) = 12 same-model pairs of 45 total."""
        from aip.aggregation.correlation import subset_swarm_rho_bar

        got = subset_swarm_rho_bar([4, 3, 3], [1.0, 1.0, 1.0],
                                   {(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0})
        assert got == pytest.approx(12 / 45)

    def test_allocation_gives_the_surplus_to_the_first_models(self) -> None:
        from aip.aggregation.correlation import allocate_agents

        assert allocate_agents(10, 3) == [4, 3, 3]
        assert allocate_agents(10, 5) == [2, 2, 2, 2, 2]
        assert sum(allocate_agents(10, 7)) == 10

    def test_missing_rho_propagates_as_nan(self) -> None:
        """A silently-dropped pair would flatter the composition."""
        import numpy as np

        from aip.aggregation.correlation import subset_swarm_rho_bar

        assert np.isnan(subset_swarm_rho_bar([5, 5], [0.5, 0.5], {}))
