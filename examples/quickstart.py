#!/usr/bin/env python3
"""Quick start: RACE in 40 lines.

1. A toy swarm: one honest receiver, three honest peers, six liars who always
   lie *independently* (the regime where a coherence gate is blind).
2. One real world from the repaired cache: MMLU, 70% gate-aware liars.

    PYTHONPATH=src python examples/quickstart.py
"""

import numpy as np

from aip.types import Broadcast, Observation
from lai.race import RACEAggregator
from lai.sim import World, evaluate_world

LABELS = ("A", "B", "C", "D")
rng = np.random.default_rng(0)


def toy_task(t: int) -> tuple[Observation, str]:
    gold = LABELS[rng.integers(4)]
    wrong = [x for x in LABELS if x != gold]
    honest = [gold if rng.random() < 0.7 else rng.choice(wrong) for _ in range(4)]  # agent 0 = receiver
    liars = [rng.choice(wrong) for _ in range(6)]  # never the truth, never coordinated
    row = tuple(Broadcast(j, f"t{t}", str(a), 0.9, 0.9, False) for j, a in enumerate(honest + liars))
    return Observation(0, f"t{t}", row), gold


history = [toy_task(t) for t in range(60)]
test = [toy_task(t) for t in range(60, 260)]
race = RACEAggregator(LABELS)
race.fit([(obs,) for obs, _ in history])  # unlabeled: gold is never passed
for peer, ch in sorted(race.diagnostics.channels[0].items()):
    print(f"peer {peer}: estimated accuracy {ch.accuracy:.2f} -> {ch.decision.upper()}")
acc = np.mean([race.aggregate(o.broadcasts, 0) == g for o, g in test])
own = np.mean([o.own.answer == g for o, g in test])
print(f"toy swarm, 60% liars: receiver alone {own:.1%}, RACE {acc:.1%}")

result = evaluate_world(World("mmlu", 0.7, "gate_aware", 0.25, seed=0),
                        methods=("self", "majority", "aip_gated", "race"))
by_method: dict[str, list[float]] = {}
for r in result["rows"]:
    if r["split"] == "test":
        by_method.setdefault(r["method"], []).append(r["accuracy"])
print("MMLU, 70% gate-aware liars:", {m: f"{np.mean(v):.1%}" for m, v in by_method.items()})
