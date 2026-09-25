"""Classical crowdsourcing baselines: recover reliable annotators with an honest
majority, and (like every label-free method without an anchor) adopt the liars'
labelling when a coherent bloc holds the majority."""

from __future__ import annotations

import numpy as np
import pytest

from aip.types import Broadcast, Observation
from lai.crowd import GLAD, IWMV, KOS, MACE
from lai.race import RACEAggregator


def _swarm(n, labels, n_honest, n_liars, seed=0, acc=0.75):
    rng = np.random.default_rng(seed)
    tasks, gold = [], []
    for t in range(n):
        g = labels[int(rng.integers(len(labels)))]
        wrong = [x for x in labels if x != g]
        lie = wrong[int(rng.integers(len(wrong)))]
        row = [g if rng.random() < acc else wrong[int(rng.integers(len(wrong)))] for _ in range(n_honest)]
        row += [lie] * n_liars
        heard = tuple(Broadcast(j, f"t{t}", a, 0.9, 0.9, False) for j, a in enumerate(row))
        tasks.append((Observation(0, f"t{t}", heard),))
        gold.append(g)
    return tasks, gold


def _acc(agg, tasks, gold):
    return float(np.mean([agg.aggregate(o[0].broadcasts, 0) == g for o, g in zip(tasks, gold, strict=True)]))


@pytest.mark.parametrize("cls", [IWMV, MACE, GLAD])
@pytest.mark.parametrize("labels", [("A", "B"), ("A", "B", "C", "D")])
def test_honest_majority_is_recovered(cls, labels):
    tasks, gold = _swarm(300, labels, 7, 3, seed=1)
    agg = cls(labels)
    agg.fit(tasks[:150])
    alone = float(np.mean([o[0].own.answer == g for o, g in zip(tasks[150:], gold[150:], strict=True)]))
    assert _acc(agg, tasks[150:], gold[150:]) > alone + 0.1


@pytest.mark.parametrize("cls", [IWMV, MACE, GLAD])
def test_liar_majority_defeats_unanchored_baselines_but_not_race(cls):
    labels = ("A", "B", "C", "D")
    tasks, gold = _swarm(300, labels, 3, 7, seed=2)
    agg = cls(labels)
    agg.fit(tasks[:150])
    race = RACEAggregator(labels)
    race.fit(tasks[:150])
    assert _acc(agg, tasks[150:], gold[150:]) < 0.4
    assert _acc(race, tasks[150:], gold[150:]) > 0.9


def test_kos_binary_only_and_honest_majority():
    with pytest.raises(ValueError):
        KOS(("A", "B", "C"))
    tasks, gold = _swarm(300, ("A", "B"), 8, 2, seed=3)
    kos = KOS(("A", "B"))
    kos.fit(tasks[:150])
    assert _acc(kos, tasks[150:], gold[150:]) > 0.85


def test_open_answers_supported():
    tasks, gold = _swarm(200, ("1", "2", "3"), 7, 2, seed=4)
    for cls in (IWMV, MACE, GLAD):
        agg = cls(None)
        agg.fit(tasks[:100])
        assert _acc(agg, tasks[100:], gold[100:]) > 0.8
