"""Analytical and actual-sampler checks for corrected attack-pair metadata."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from phase_gate_aware import pairwise_q_metadata, realised_q  # noqa: E402

from aip.swarm.assignment import SwarmAssignment  # noqa: E402
from aip.swarm.broadcast import AdversaryConfig, build_broadcasts, observe  # noqa: E402
from aip.swarm.topology import build_topology  # noqa: E402


@pytest.mark.parametrize("p", [0.0, 0.2, 0.4, 0.75, 1.0])
@pytest.mark.parametrize("candidates", [1, 2, 3, 9])
def test_formula_matches_exhaustive_branch_and_label_enumeration(p, candidates):
    """Enumerate both mixed branches, which the historical expression omitted."""
    # Fix shared label zero; symmetry makes the value independent of its draw.
    branches = [(0, p)] + [(label, (1.0 - p) / candidates)
                           for label in range(candidates)]
    collision = sum(
        mass_left * mass_right
        for (left, mass_left), (right, mass_right) in itertools.product(branches, repeat=2)
        if left == right
    )
    assert realised_q(p, candidates + 1) == pytest.approx(collision)
    assert realised_q(p, None, n_wrong_candidates=candidates) == pytest.approx(collision)


def test_closed_four_class_axis_has_no_historical_interior_minimum():
    expected = [realised_q(float(p), 4) for p in np.linspace(0, 1, 51)]
    assert expected[0] == pytest.approx(1.0 / 3.0)
    assert expected[-1] == 1.0
    assert np.all(np.diff(expected) >= 0)
    assert realised_q(0.4, 4) == pytest.approx(0.44)
    assert realised_q(0.4, 4) != pytest.approx(0.28)


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_coordination_probability_is_rejected(p):
    with pytest.raises(ValueError, match="p must"):
        realised_q(p, 4)


@pytest.mark.parametrize("options", [0, 1, 2.5, True])
def test_invalid_closed_space_is_rejected(options):
    with pytest.raises(ValueError, match="n_options"):
        realised_q(0.4, options)


@pytest.mark.parametrize("candidates", [None, 0, -1, 1.5, True])
def test_open_space_requires_explicit_positive_candidate_count(candidates):
    with pytest.raises(ValueError, match="n_wrong_candidates"):
        realised_q(0.4, None, n_wrong_candidates=candidates)


def test_conflicting_candidate_specifications_are_rejected():
    with pytest.raises(ValueError, match="not both"):
        realised_q(0.4, 4, n_wrong_candidates=2)


def _assignment(honest_answers):
    models = tuple(f"h{i}" for i in range(len(honest_answers)))
    assignment = SwarmAssignment(
        n_agents=2 + len(models),
        byzantine=frozenset({0, 1}),
        models=(None, None, *models),
        composition="coherence_formula_fixture",
    )
    cache = {model: (answer, 0.8, 0.8)
             for model, answer in zip(models, honest_answers, strict=True)}
    return assignment, cache


@pytest.mark.parametrize(
    "labels,honest_answers,p,candidates",
    [
        (["truth", "b", "c", "d"], [], 0.4, 3),
        (["truth", "b"], [], 0.2, 1),
        ([], ["b", "c", "d"], 0.4, 3),
        ([], ["b", "b", "c"], 0.35, 2),
        ([], ["truth", None], 0.2, 1),
    ],
)
def test_actual_broadcast_sampler_matches_collision_expectation(
    labels, honest_answers, p, candidates
):
    """Use build_broadcasts, including open-pool deduplication and fallback."""
    assignment, cache = _assignment(honest_answers)
    rng = np.random.default_rng(20260912)
    config = AdversaryConfig(kind="gate_aware", coherence=p)
    repeats, collisions = 4000, 0
    for trial in range(repeats):
        broadcasts = build_broadcasts(
            str(trial), "truth", assignment, cache, labels, config, rng
        )
        assert broadcasts[0].answer != "truth"
        assert broadcasts[1].answer != "truth"
        collisions += broadcasts[0].answer == broadcasts[1].answer
    expected = realised_q(p, None, n_wrong_candidates=candidates)
    # Trials are independent; the loose six-sigma tolerance catches the old
    # axis error without relying on an accidental exact random count.
    tolerance = 6 * np.sqrt(expected * (1 - expected) / repeats) + 1 / repeats
    assert abs(collisions / repeats - expected) <= tolerance


def test_open_metadata_uses_actual_task_pools_and_measured_pairs():
    rng = np.random.default_rng(27)
    tasks = []
    expected_collisions = 0
    for i, answers in enumerate((["truth"], ["b", "c"], ["b", "c", "d"])):
        assignment, cache = _assignment(answers)
        broadcasts = build_broadcasts(
            str(i), "truth", assignment, cache, [],
            AdversaryConfig(kind="gate_aware", coherence=0.4), rng,
        )
        expected_collisions += int(broadcasts[0].answer == broadcasts[1].answer)
        tasks.append(observe(
            broadcasts, build_topology("complete", assignment.n_agents), 1.0, rng
        ))
    metadata = pairwise_q_metadata(0.4, None, tasks, ["truth"] * len(tasks))
    assert metadata["q_expected_pairwise"] == pytest.approx((1.0 + 0.58 + 0.44) / 3)
    assert metadata["q_empirical_pairwise"] == expected_collisions / 3
    assert metadata["q_pair_count"] == 3
    assert metadata["q_candidate_pool_min"] == 1
    assert metadata["q_candidate_pool_max"] == 3
    assert metadata["q_scope"] == "unconditioned_byzantine_pairs_not_receiver_gate"
    assert metadata["q_formula_version"] == "shared_uniform_wrong_v2"


def test_pairwise_metadata_rejects_partial_observation_pool():
    assignment, cache = _assignment(["b", "c"])
    rng = np.random.default_rng(27)
    broadcasts = build_broadcasts(
        "partial", "truth", assignment, cache, [],
        AdversaryConfig(kind="gate_aware", coherence=0.4), rng,
    )
    partial = observe(
        broadcasts, build_topology("complete", assignment.n_agents), 0.0, rng
    )
    with pytest.raises(ValueError, match="complete broadcast"):
        pairwise_q_metadata(0.4, None, [partial], ["truth"])
