"""Strictly causal (predict-then-commit) evaluation under changing attackers.

RACE's guarantee is a stationarity guarantee: channels estimated on the past are
assumed to describe the present. A sleeper that behaves honestly while trust is
being estimated and lies afterwards breaks that assumption. This module measures
how badly, and how much exponential forgetting recovers.

Protocol: tasks arrive in a fixed chronological order. At each step every
honest receiver *predicts* from a state fitted on strictly earlier tasks, and
only then *commits* the current observation to its history. Refits happen every
``refit_every`` tasks. Changing a future broadcast cannot change a past
prediction (tested in ``tests/test_online.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from aip.aggregation.aip import AIPAggregator, InversionThresholds
from aip.aggregation.base import ParityConfig
from aip.aggregation.baselines import MajorityVote
from aip.types import Broadcast, Observation
from lai import attacks as atk
from lai.data import FROZEN_MODELS, load_benchmark, score
from lai.race import RACEAggregator
from lai.sim import THRESHOLDS_PATH, stable_seed

SCHEDULES = {
    # name: function(position, warmup) -> (attack, param, phase)
    "sleeper": lambda pos, w: ("sleeper", 1.0, "history" if pos < w + 60 else "test"),
    "coherent_to_independent": lambda pos, w: ("coherent", 1.0, "test") if pos < w + 80 else ("independent", 1.0, "test"),
    "independent_to_coherent": lambda pos, w: ("independent", 1.0, "test") if pos < w + 80 else ("coherent", 1.0, "test"),
    "toggle40": lambda pos, w: ("coherent", 1.0, "test") if (pos // 40) % 2 == 0 else ("uninformative", 1.0, "test"),
    "stationary_coherent": lambda pos, w: ("coherent", 1.0, "test"),
}


@dataclass(frozen=True)
class OnlineWorld:
    benchmark: str
    f: float
    schedule: str
    seed: int = 0
    n_agents: int = 10
    warmup: int = 40
    refit_every: int = 8


def _stream(world: OnlineWorld):
    data = load_benchmark(world.benchmark)
    n = world.n_agents
    rng_id = np.random.default_rng(stable_seed(world.seed, world.benchmark, "frozen", n, "identities"))
    comp = list(rng_id.permutation(list(FROZEN_MODELS)))
    base = [comp[i % len(comp)] for i in rng_id.permutation(n)]
    corrupted = np.random.default_rng(stable_seed(world.seed, world.benchmark, n, "corruption")).permutation(n)
    byz = sorted(int(i) for i in corrupted[: int(round(world.f * n))])
    models = [None if i in byz else base[i] for i in range(n)]
    honest = [i for i in range(n) if i not in byz]
    order = sorted(range(len(data.task_ids)), key=lambda i: stable_seed(world.seed, "chronology", data.task_ids[i]))
    cover = [m for m in models if m is not None]
    stream = []
    for pos, t in enumerate(order):
        tid, gold = data.task_ids[t], data.gold[t]
        attack, param, phase = SCHEDULES[world.schedule](pos, world.warmup)
        rng = np.random.default_rng(stable_seed(world.seed, world.benchmark, tid, "attack"))
        honest_answers = [data.honest[m][t] for m in cover]
        lies = atk.byzantine_answers(
            attack, param, gold=gold, label_space=data.label_space, honest_answers=honest_answers,
            n_byzantine=len(byz), is_correct=lambda a, _g=gold: score(world.benchmark, a, _g), rng=rng,
            phase=phase, cover_answers=honest_answers,
        )
        lie = dict(zip(byz, lies, strict=True))
        row = tuple(
            Broadcast(j, tid, data.canonical(t, lie[j]) if j in lie else data.honest[models[j]][t], 0.99, 0.99, False)
            for j in range(n)
        )
        stream.append((pos, t, tid, gold, attack, row))
    return data, honest, byz, stream


def _obs(row: tuple[Broadcast, ...], receiver: int) -> Observation:
    return Observation(receiver, row[0].task_id, tuple(replace(b) for b in row))


def online_methods(benchmark: str, labels) -> dict[str, Callable]:
    thresholds = InversionThresholds.load(THRESHOLDS_PATH)
    parity = ParityConfig(0.1)
    return {
        "aip_gated": (lambda: AIPAggregator(benchmark, "gated", thresholds, parity, labels)),
        "race": (lambda: RACEAggregator(labels)),
        "race_capself": (lambda: RACEAggregator(labels, cap="self")),
    }


RETENTION = {
    "fixed": None,            # warm-up history only
    "cumulative": "all",      # every committed task, equal weight
    "window64": 64,           # the last 64 committed tasks
    "decay0.97": 0.97,        # exponential forgetting (RACE only)
}


def run_online(world: OnlineWorld, max_receivers: int | None = 5) -> pd.DataFrame:
    data, honest, _, stream = _stream(world)
    return run_online_stream(world, data.label_space, honest[:max_receivers] if max_receivers else honest, stream)


def run_online_stream(world: OnlineWorld, label_space, honest, stream) -> pd.DataFrame:
    """Causal evaluation over an explicit stream of (pos, t, tid, gold, attack, row)."""
    labels = list(label_space) if label_space else None
    methods = online_methods(world.benchmark, labels)
    records = []
    for receiver in honest:
        committed: list[Observation] = []
        state: dict[str, object] = {}
        majority = MajorityVote()
        for pos, t, tid, gold, attack, row in stream:
            obs = _obs(row, receiver)
            if pos >= world.warmup:
                preds = {"self": obs.own.answer, "majority": majority.aggregate(obs.broadcasts, receiver)}
                for key, agg in state.items():
                    preds[key] = agg.aggregate(obs.broadcasts, receiver)
                for key, pred in preds.items():
                    records.append({"receiver": receiver, "pos": pos, "task": tid, "attack_now": attack,
                                    "method": key, "correct": float(score(world.benchmark, pred, gold)), "pred": pred})
            committed.append(obs)
            n_seen = len(committed)
            if n_seen == world.warmup or (n_seen > world.warmup and (n_seen - world.warmup) % world.refit_every == 0):
                for name, make in methods.items():
                    for retention, param in RETENTION.items():
                        if retention == "fixed" and n_seen != world.warmup:
                            continue
                        if retention.startswith("decay") and not name.startswith("race"):
                            continue
                        key = f"{name}_{retention}"
                        agg = make()
                        if retention == "window64":
                            hist = committed[-64:]
                        else:
                            hist = committed
                        if retention.startswith("decay") and isinstance(agg, RACEAggregator):
                            rows = [{b.agent_id: b.answer for b in o.broadcasts} for o in hist]
                            weights = param ** np.arange(len(hist) - 1, -1, -1, dtype=float)
                            agg.fit_receiver(receiver, rows, weights)
                        else:
                            agg.fit([(o,) for o in hist])
                        state[key] = agg
    frame = pd.DataFrame(records)
    for k, v in vars(world).items():
        frame[k] = v
    return frame
