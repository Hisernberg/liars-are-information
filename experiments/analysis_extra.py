#!/usr/bin/env python3
"""Per-channel and per-receiver diagnostics that the main runs do not store.

1. ``channel_estimates.parquet`` -- for every honest receiver and every peer it
   hears: RACE's estimated accuracy a_hat, the peer's true history accuracy
   (evaluator-side, from gold), the Bayes weight lambda, RACE's decision and
   AIP's decision. This is the direct test of "channel estimation".
2. ``receiver_gain.parquet`` -- per receiver: its own test accuracy and the
   test accuracy of RACE / AIP / majority, for the weak-anchor analysis.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.aggregation.baselines import MajorityVote  # noqa: E402
from lai.data import score  # noqa: E402
from lai.race import RACEAggregator  # noqa: E402
from lai.sim import THRESHOLDS_PATH, World, build_world  # noqa: E402

OUT = ROOT / "results" / "extra"


def worlds() -> list[World]:
    out = []
    attacks = [("coherent", 1.0), ("gate_aware", 0.25), ("independent", 1.0), ("llm:rushing", 1.0),
               ("llm:semantic_negation", 1.0), ("camouflage", 0.9), ("attractor", 1.0)]
    for b, f, (a, p), s in itertools.product(
        ("mmlu", "medqa", "arc", "boolq", "gsm8k", "math500"), (0.1, 0.3, 0.5, 0.7), attacks, (0, 1, 2)
    ):
        out.append(World(b, f, a, p, 1.0, s, study="extra"))
    return out


def one(world: World) -> tuple[list[dict], list[dict]]:
    warnings.filterwarnings("ignore")
    bw = build_world(world)
    b = world.benchmark
    labels = list(bw.data.label_space) if bw.data.label_space else None
    thresholds = InversionThresholds.load(THRESHOLDS_PATH)
    hist, test = bw.splits["history"], bw.splits["test"]
    true_acc = {
        j: float(np.mean([score(b, bw.raw[t][j].answer, bw.data.gold[t]) for t in hist]))
        for j in range(world.n_agents)
    }
    chans, recv = [], []
    meta = dict(benchmark=b, f=world.f, attack=world.attack, param=world.param, seed=world.seed)
    for r in bw.honest:
        history = [(bw.defense[t][r],) for t in hist]
        race = RACEAggregator(labels)
        race.fit(history)
        aip = AIPAggregator(b, "gated", thresholds, ParityConfig(0.1), labels)
        aip.fit(history)
        maj = MajorityVote()
        for peer, est in race.diagnostics.channels[r].items():
            if peer == r:
                continue
            aip_stats = aip.diagnostics.channels.get(r, {}).get(peer)
            chans.append(meta | dict(receiver=r, peer=peer, byzantine=peer in bw.byzantine,
                                     a_hat=est.accuracy, a_true=true_acc[peer], weight=est.weight_at_chance_k,
                                     race_decision=est.decision,
                                     aip_decision=aip_stats.decision if aip_stats else "discard",
                                     group=est.group))

        def acc(fn):
            return float(np.mean([score(b, fn(bw.defense[t][r]), bw.data.gold[t]) for t in test]))

        recv.append(meta | dict(
            receiver=r, model=str(bw.models[r]), self_acc=acc(lambda o: o.own.answer),
            receiver_hist_acc=true_acc[r],
            race_acc=acc(lambda o: race.aggregate(o.broadcasts, r)),
            aip_acc=acc(lambda o: aip.aggregate(o.broadcasts, r)),
            majority_acc=acc(lambda o: maj.aggregate(o.broadcasts, r)),
        ))
    return chans, recv


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chans, recv = [], []
    ws = worlds()
    with mp.get_context("fork").Pool(4) as pool:
        for i, (c, r) in enumerate(pool.imap_unordered(one, ws), 1):
            chans += c
            recv += r
            if i % 50 == 0:
                print(f"[{i}/{len(ws)}]", flush=True)
    pd.DataFrame(chans).to_parquet(OUT / "channel_estimates.parquet", index=False)
    pd.DataFrame(recv).to_parquet(OUT / "receiver_gain.parquet", index=False)


if __name__ == "__main__":
    main()
