"""Clone-cap validity without privileged labels or future observations."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from aip.aggregation.aip import INVERT, TRUST, AIPAggregator, ChannelStats
from aip.aggregation.base import ParityConfig, apply_self_vote_parity
from aip.extensions.clone_aware import CloneAwareAIP, cap_group_weights, learn_groups
from aip.types import Broadcast, Observation


def bc(agent, answer, task="t"):
    return Broadcast(agent, task, answer, 0.9, 0.9, False)


def history(n=40):
    result = []
    for t in range(n):
        answer = ("A", "B", "C", "D")[t % 4]
        other = ("B", "C", "D", "A")[t % 4]
        task = f"h{t:03d}"
        result.append(
            [
                Observation(
                    0,
                    task,
                    tuple(bc(i, a, task) for i, a in enumerate([answer, other, other, answer])),
                )
            ]
        )
    return result


def test_aligned_missingness_does_not_compact_different_tasks():
    observations = []
    for t in range(40):
        task = str(t)
        values = (
            [bc(0, "A", task), bc(1, "B", task)] if t % 2 else [bc(0, "A", task), bc(2, "B", task)]
        )
        observations.append(Observation(0, task, tuple(values)))
    groups, pairs = learn_groups(observations, min_support=10)
    assert all(not ({1, 2} <= set(group)) for group in groups)
    assert next(p for p in pairs if (p.left, p.right) == (1, 2)).co_observed == 0


def test_complete_link_prevents_transitive_grouping():
    observations = []
    for t in range(10):
        task = str(t)
        values = ["A", "B" if t == 0 else "A", "B" if t in {0, 1} else "A"]
        observations.append(
            Observation(0, task, tuple(bc(i, a, task) for i, a in enumerate(values)))
        )
    groups, _ = learn_groups(observations, min_support=10, agreement_threshold=0.9)
    assert groups == ((0, 1), (2,))


def test_cap_mass_and_self_group_before_parity():
    weights = {0: 1.0, 1: 0.8, 2: 0.4, 3: 1.0, 4: 0.6}
    capped = cap_group_weights(weights, 0, [(0, 3), (1, 2), (4,)])
    assert capped[0] == 1 and capped[3] == 0
    assert capped[1] + capped[2] == pytest.approx(0.8)
    assert capped[1] / capped[2] == pytest.approx(2.0)
    assert capped[4] == 0.6
    normalized = apply_self_vote_parity(capped, 0, ParityConfig(0.1))
    assert normalized[0] / sum(normalized.values()) == pytest.approx(0.1)
    assert weights[3] == 1  # input unchanged


def test_fixed_statistics_identical_clones_preserve_capped_mass_and_prediction():
    model = CloneAwareAIP("mmlu", label_space=["A", "B", "C", "D"], parity=ParityConfig(0.1))
    model._stats = {
        0: {0: ChannelStats(decision=TRUST, weight=1), 1: ChannelStats(decision=INVERT, weight=0.8)}
    }
    model.groups_ = {0: ((0,), (1,))}
    before = [bc(0, "A"), bc(1, "B")]
    expected = model.aggregate(before, 0)
    mass = sum(v for k, v in model.weight_diagnostics(before, 0)["after_cap"].items() if k != 0)
    for n in [5, 15]:
        augmented = deepcopy(model)
        for agent in range(2, n + 2):
            augmented._stats[0][agent] = deepcopy(model._stats[0][1])
        augmented.groups_ = {0: ((0,), tuple(range(1, n + 2)))}
        reports = before + [bc(agent, "B") for agent in range(2, n + 2)]
        assert augmented.aggregate(reports, 0) == expected
        assert sum(
            v for k, v in augmented.weight_diagnostics(reports, 0)["after_cap"].items() if k != 0
        ) == pytest.approx(mass)


def test_fits_only_history_and_does_not_learn_from_test_calls():
    model = CloneAwareAIP("mmlu", label_space=["A", "B", "C", "D"])
    model.fit(history())
    groups = deepcopy(model.groups_)
    fitted_ids = deepcopy(model.training_task_ids_)
    for t in range(10):
        model.aggregate([bc(0, "A", f"test{t}"), bc(1, "A", f"test{t}"), bc(2, "D", f"test{t}")], 0)
    assert groups == model.groups_ and fitted_ids == model.training_task_ids_
    assert all(task.startswith("h") for task in model.training_task_ids_[0])


def test_reused_original_gate_is_identical_to_independent_fit():
    base = AIPAggregator("mmlu", label_space=["A", "B", "C", "D"], n_peers_for_correction=9)
    base.fit(history())
    copied = CloneAwareAIP.from_fitted(base, history())
    fitted = CloneAwareAIP("mmlu", label_space=["A", "B", "C", "D"], n_peers_for_correction=9)
    fitted.fit(history())
    assert copied.groups_ == fitted.groups_
    for peer in base._stats[0]:
        assert copied._stats[0][peer].decision == base._stats[0][peer].decision
        assert copied._stats[0][peer].weight == base._stats[0][peer].weight
    assert copied.aggregate(
        [bc(0, "A"), bc(1, "B"), bc(2, "B"), bc(3, "A")], 0
    ) == fitted.aggregate([bc(0, "A"), bc(1, "B"), bc(2, "B"), bc(3, "A")], 0)


def test_hidden_metadata_cannot_change_groups_or_predictions():
    clean = history()
    poisoned = [
        [
            replace(
                obs,
                broadcasts=tuple(
                    replace(b, is_byzantine=True, source_model="oracle", attack="oracle")
                    for b in obs.broadcasts
                ),
            )
            for obs in task
        ]
        for task in clean
    ]
    left = CloneAwareAIP("mmlu", label_space=["A", "B", "C", "D"])
    right = CloneAwareAIP("mmlu", label_space=["A", "B", "C", "D"])
    left.fit(clean)
    right.fit(poisoned)
    assert left.groups_ == right.groups_
    assert left.aggregate(clean[0][0].broadcasts, 0) == right.aggregate(
        poisoned[0][0].broadcasts, 0
    )


class ObservableOnly:
    def __init__(self, b):
        self.agent_id, self.task_id, self.answer = b.agent_id, b.task_id, b.answer

    def __getattr__(self, key):
        raise AssertionError(f"Hidden field accessed: {key}")


def test_grouping_reads_only_observable_answers_and_ids():
    raw = [task[0] for task in history()]
    guarded = [
        replace(obs, broadcasts=tuple(ObservableOnly(b) for b in obs.broadcasts)) for obs in raw
    ]
    assert learn_groups(raw) == learn_groups(guarded)


def test_support_threshold_and_partition_validation():
    groups, _ = learn_groups([task[0] for task in history(5)], min_support=20)
    assert all(len(group) == 1 for group in groups)
    with pytest.raises(ValueError):
        learn_groups([], min_support=0)
    with pytest.raises(ValueError):
        cap_group_weights({0: 1.0, 1: 0.5}, 0, [(0, 1), (1,)])
    with pytest.raises(ValueError):
        cap_group_weights({0: np.nan}, 0, [(0,)])
    with pytest.raises(ValueError):
        learn_groups([], min_support=1.5)
    with pytest.raises(ValueError):
        cap_group_weights({0: 1.0, 1: 0.5}, 0, [(0,), (1, 1)])
    with pytest.raises(ValueError):
        CloneAwareAIP.from_fitted(AIPAggregator("mmlu"), history())


def test_added_copies_preserve_original_broadcasts_and_visibility():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import extension_clone as experiment

    models = ["a", "b"]
    cache = experiment.replay.pc.BenchmarkCache(
        "mmlu", ["t"], ["A"], {(m, 0): [("A", 0.8, 0.8)] for m in models}
    )
    world = dict(
        seed=42,
        benchmark="mmlu",
        composition="frozen_mix",
        attack="always_wrong",
        coherence_p=1.0,
        p_obs=1.0,
        f=0.3,
    )
    base, _, raw = experiment.replay.build_world(cache, models, world)
    zero, raw_zero, visible_zero = experiment.expand_world(
        base, raw, 0, "none", 42, "mmlu", 0.3, 0.5
    )
    for kind in ["honest", "byzantine"]:
        expanded, raw_copies, visible_copies = experiment.expand_world(
            base, raw, 15, kind, 42, "mmlu", 0.3, 0.5
        )
        assert raw_zero[0] == raw_copies[0][:10]
        assert set(visible_copies[0]) == set(base.honest)
        for receiver in base.honest:
            old = {b.agent_id for b in visible_zero[0][receiver].broadcasts}
            new = {b.agent_id for b in visible_copies[0][receiver].broadcasts if b.agent_id < 10}
            assert old == new
            assert all(
                not b.is_byzantine and b.source_model is None and b.attack is None
                for b in visible_copies[0][receiver].broadcasts
            )
        expected_byzantine = 3 if kind == "honest" else 18
        assert expanded["f_realized"] == expected_byzantine / 25
        assert zero["f_realized"] == 0.3
