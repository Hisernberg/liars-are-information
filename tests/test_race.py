"""Properties of RACE (receiver-anchored channel estimation)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aip.types import Broadcast, Observation
from lai.race import DISCARD, INVERT, TRUST, RACEAggregator, complete_link_groups, tempering_weights

LABELS = ("A", "B", "C", "D")


def _swarm(n_tasks, labels, honest_acc, liar_kind, n_liars, seed=0, p_obs=1.0):
    """Receiver 0 plus honest peers and liars; returns (tasks, gold, liar ids)."""
    rng = np.random.default_rng(seed)
    n_honest = len(honest_acc)
    liars = list(range(n_honest, n_honest + n_liars))
    tasks, gold = [], []
    for t in range(n_tasks):
        g = labels[int(rng.integers(len(labels)))]
        wrong = [x for x in labels if x != g]
        shared = wrong[int(rng.integers(len(wrong)))]
        row = []
        for j, acc in enumerate(honest_acc):
            a = g if rng.random() < acc else wrong[int(rng.integers(len(wrong)))]
            row.append(a)
        for _ in liars:
            if liar_kind == "coherent":
                row.append(shared)
            elif liar_kind == "independent":
                row.append(wrong[int(rng.integers(len(wrong)))])
            elif liar_kind == "random":
                row.append(labels[int(rng.integers(len(labels)))])
            else:
                raise ValueError(liar_kind)
        heard = [Broadcast(j, f"t{t}", a, 0.9, 0.9, j in liars) for j, a in enumerate(row)
                 if j == 0 or p_obs >= 1 or rng.random() < p_obs]
        tasks.append((Observation(0, f"t{t}", tuple(heard)),))
        gold.append(g)
    return tasks, gold, liars


def _accuracy(agg, tasks, gold):
    return float(np.mean([agg.aggregate(obs[0].broadcasts, 0) == g for obs, g in zip(tasks, gold, strict=True)]))


@pytest.mark.parametrize("labels", [("A", "B"), LABELS])
def test_anchor_survives_coherent_liar_majority(labels):
    tasks, gold, _ = _swarm(300, labels, [0.85, 0.85, 0.85], "coherent", 7, seed=1)
    hist, test = tasks[:150], tasks[150:]
    race = RACEAggregator(labels)
    race.fit(hist)
    unanchored = RACEAggregator(labels, anchored=False, clone_aware=False)
    unanchored.fit(hist)
    assert _accuracy(race, test, gold[150:]) > 0.9
    # Label-free EM initialised from the plurality adopts the liars' labelling.
    assert _accuracy(unanchored, test, gold[150:]) < 0.2


def test_liars_are_inverted_and_honest_trusted():
    tasks, _, liars = _swarm(200, LABELS, [0.8, 0.8, 0.8, 0.8], "coherent", 5, seed=2)
    race = RACEAggregator(LABELS)
    race.fit(tasks)
    ch = race.diagnostics.channels[0]
    assert all(ch[j].decision == INVERT for j in liars)
    assert all(ch[j].decision == TRUST for j in (1, 2, 3))


def test_incoherent_always_wrong_liars_are_still_informative():
    # The gate-aware regime: liars never coordinate, so coherence is at chance,
    # yet each one eliminates its answer. RACE must invert them.
    tasks, gold, liars = _swarm(300, LABELS, [0.7, 0.7], "independent", 7, seed=3)
    race = RACEAggregator(LABELS)
    race.fit(tasks[:150])
    ch = race.diagnostics.channels[0]
    assert all(ch[j].decision == INVERT for j in liars)
    self_acc = np.mean([o[0].own.answer == g for o, g in zip(tasks[150:], gold[150:], strict=True)])
    assert _accuracy(race, tasks[150:], gold[150:]) > self_acc + 0.1


def test_truth_independent_peer_is_discarded():
    tasks, _, liars = _swarm(400, LABELS, [0.85, 0.85, 0.85], "random", 3, seed=4)
    race = RACEAggregator(LABELS, clone_aware=False)
    race.fit(tasks)
    ch = race.diagnostics.channels[0]
    for j in liars:
        assert abs(ch[j].accuracy - 0.25) < 0.08
        assert ch[j].decision == DISCARD


def test_hidden_metadata_does_not_change_predictions():
    tasks, _, _ = _swarm(120, LABELS, [0.8, 0.8, 0.8], "coherent", 4, seed=5)
    a = RACEAggregator(LABELS)
    a.fit(tasks[:60])
    scrubbed = [
        (replace(o[0], broadcasts=tuple(replace(b, is_byzantine=False, source_model="x", attack="y",
                                                logprob_confidence=0.1) for b in o[0].broadcasts)),)
        for o in tasks
    ]
    b = RACEAggregator(LABELS)
    b.fit(scrubbed[:60])
    for o1, o2 in zip(tasks[60:], scrubbed[60:], strict=True):
        assert a.aggregate(o1[0].broadcasts, 0) == b.aggregate(o2[0].broadcasts, 0)


def test_label_permutation_equivariance():
    tasks, _, _ = _swarm(150, LABELS, [0.8, 0.7, 0.75], "coherent", 4, seed=6)
    perm = {"A": "C", "B": "D", "C": "A", "D": "B"}
    moved = [(replace(o[0], broadcasts=tuple(replace(b, answer=perm[b.answer]) for b in o[0].broadcasts)),)
             for o in tasks]
    a, b = RACEAggregator(LABELS), RACEAggregator(LABELS)
    a.fit(tasks[:100])
    b.fit(moved[:100])
    for o1, o2 in zip(tasks[100:], moved[100:], strict=True):
        assert perm[a.aggregate(o1[0].broadcasts, 0)] == b.aggregate(o2[0].broadcasts, 0)


def test_missing_observations_and_unseen_agents():
    tasks, gold, _ = _swarm(300, LABELS, [0.85, 0.85, 0.85, 0.85], "coherent", 5, seed=7, p_obs=0.5)
    race = RACEAggregator(LABELS)
    race.fit(tasks[:150])
    assert _accuracy(race, tasks[150:], gold[150:]) > 0.85
    # A never-seen agent carries zero weight: adding it cannot change a prediction.
    obs = tasks[200][0]
    extra = obs.broadcasts + (Broadcast(99, obs.task_id, "A", 0.9, 0.9, False),)
    assert race.aggregate(extra, 0) == race.aggregate(obs.broadcasts, 0)


def test_posterior_is_a_distribution_and_ties_go_to_self():
    race = RACEAggregator(LABELS)
    obs = [Broadcast(0, "t", "B", 0.9, 0.9, False), Broadcast(1, "t", "C", 0.9, 0.9, False)]
    # Unfitted receiver: falls back to its own answer.
    assert race.aggregate(obs, 0) == "B"
    tasks, _, _ = _swarm(80, LABELS, [0.8, 0.8], "coherent", 2, seed=8)
    race.fit(tasks)
    post = race.posterior(tasks[0][0].broadcasts, 0)
    assert pytest.approx(sum(post.values())) == 1.0


def test_open_answer_space():
    rng = np.random.default_rng(9)
    tasks, gold = [], []
    for t in range(200):
        g = str(int(rng.integers(1000)))
        honest = [g if rng.random() < acc else str(int(rng.integers(1000))) for acc in (0.7, 0.7, 0.7)]
        lie = f"{g}x"
        row = honest + [lie] * 5
        tasks.append((Observation(0, f"t{t}", tuple(Broadcast(j, f"t{t}", a, 0.9, 0.9, j >= 3) for j, a in enumerate(row))),))
        gold.append(g)
    race = RACEAggregator(None)
    race.fit(tasks[:100])
    assert _accuracy(race, tasks[100:], gold[100:]) > 0.8


def test_full_confusion_variant_anchored():
    tasks, gold, _ = _swarm(300, LABELS, [0.85, 0.85, 0.85], "coherent", 7, seed=10)
    race = RACEAggregator(LABELS, model="full")
    race.fit(tasks[:150])
    assert _accuracy(race, tasks[150:], gold[150:]) > 0.9


def test_complete_link_grouping():
    reports = np.array([[0, 0, 0, 1], [1, 1, 1, 2], [2, 2, 3, 0], [3, 3, 0, 1]] * 5)
    groups = complete_link_groups(reports, threshold=0.95, min_support=4)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0] and groups[3] != groups[0]
    w = tempering_weights(reports, groups)
    assert np.allclose(w[:, 0], 0.5) and np.allclose(w[:, 3], 1.0)


def test_capself_bounds_peer_accuracy():
    tasks, _, _ = _swarm(200, LABELS, [0.6, 0.95, 0.95], "coherent", 2, seed=11)
    race = RACEAggregator(LABELS, cap="self")
    race.fit(tasks)
    fit = race.fits[0]
    own = fit.agents.index(0)
    assert np.all(fit.accuracy <= fit.accuracy[own] + 1e-12)


def test_error_conditioned_links_separate_competence_from_dependence():
    from lai.race import error_conditioned_links, raw_links

    rng = np.random.default_rng(12)
    n = 400
    truth = rng.integers(0, 4, n)

    def noisy(acc):
        return np.where(rng.random(n) < acc, truth, (truth + rng.integers(1, 4, n)) % 4)

    strong_a, strong_b = noisy(0.97), noisy(0.97)
    weak = noisy(0.6)
    reports = np.stack([strong_a, strong_b, weak, weak.copy()], axis=1)
    raw, _ = raw_links(reports, 0.9, 8)
    err, _ = error_conditioned_links(reports, truth, 0.95, 8)
    assert raw[0, 1]            # raw agreement merges two distinct accurate models ...
    assert not err[0, 1]        # ... error-conditioned agreement does not
    assert err[2, 3] and raw[2, 3]  # replicas are linked either way


def test_receiver_replica_cannot_confirm_the_receiver():
    # A near-chance receiver plus its exact replica, strong honest peers, and a
    # liar echoing the receiver's errors. The replica must be grouped with the
    # receiver so it adds no independent confirmation.
    rng = np.random.default_rng(13)
    tasks = []
    for t in range(160):
        g = LABELS[int(rng.integers(4))]
        wrong = [x for x in LABELS if x != g]
        me = g if rng.random() < 0.5 else wrong[int(rng.integers(3))]
        row = [me, me] + [g if rng.random() < 0.9 else wrong[int(rng.integers(3))] for _ in range(5)]
        tasks.append((Observation(0, f"t{t}", tuple(Broadcast(j, f"t{t}", a, 0.9, 0.9, False) for j, a in enumerate(row))),))
    race = RACEAggregator(LABELS)
    race.fit(tasks)
    fit = race.fits[0]
    assert fit.groups[fit.agents.index(0)] == fit.groups[fit.agents.index(1)]
