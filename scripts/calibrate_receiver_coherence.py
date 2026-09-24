#!/usr/bin/env python3
"""Calibrate honest coherence as the GATE measures it, not as Phase B2 measured it.

Phase B2 measured ``q = P(same answer | both agents wrong)``, which needs gold
labels. AIP's gate cannot use that: a receiver has no labels, so it conditions on
what it can see -- ``P(same answer | both peers dissent from ME)``. The two
coincide only when the receiver is almost always right. On MATH-500, where models
are wrong 28-66% of the time, honest peers score 0.51-0.66 on the receiver-
conditioned statistic against the 0.020 Phase B2 measured for the gold-conditioned
one, so a ceiling calibrated on the latter is far too low and the gate fires on
honest channels.

This script measures the statistic the gate actually uses, from honest cached
answers only, pooling over every honest receiver. Cache-only; no labels are used
beyond selecting honest agents, and no model is loaded.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.analysis.bootstrap import bootstrap_statistic  # noqa: E402
from aip.tasks import roster  # noqa: E402

BENCHMARKS = roster.benchmarks()  # from configs/task_lists.json


def answers(frame: pd.DataFrame) -> np.ndarray:
    return np.array(
        [None if pd.isna(v) else str(v) for v in frame["extracted_answer"]], dtype=object
    )


def pooled_receiver_coherence(cols: dict[str, np.ndarray], idx: np.ndarray) -> float:
    """P(peer_a == peer_b | both differ from the receiver), pooled over receivers."""
    names = list(cols)
    num = den = 0
    for receiver in names:
        own = cols[receiver][idx]
        for a, b in itertools.combinations([n for n in names if n != receiver], 2):
            xa, xb = cols[a][idx], cols[b][idx]
            for i in range(len(idx)):
                if own[i] is None or xa[i] is None or xb[i] is None:
                    continue
                if xa[i] == own[i] or xb[i] == own[i]:
                    continue
                den += 1
                num += int(xa[i] == xb[i])
    return num / den if den else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20250825)
    args = parser.parse_args()

    models = sorted(p.stem for p in (args.cache / "gsm8k").glob("*.parquet"))
    print(
        f"models: {models}\nreceiver-conditioned honest coherence "
        f"(bootstrap {args.resamples}, seed {args.seed})\n"
    )
    print(f"{'benchmark':<10}{'q_recv [95% CI]':>30}{'gold-conditioned q (B2)':>26}")
    for benchmark in BENCHMARKS:
        frames = {
            m: pd.read_parquet(args.cache / benchmark / f"{m}.parquet")
            .set_index("task_id")
            .sort_index()
            for m in models
        }
        shared = None
        for f in frames.values():
            shared = f.index if shared is None else shared.intersection(f.index)
        cols = {m: answers(frames[m].loc[shared]) for m in models}
        n = len(shared)
        ci = bootstrap_statistic(
            lambda i, c=cols: pooled_receiver_coherence(c, i), n, args.resamples, seed=args.seed
        )
        # The L4-era gold-conditioned q used to be printed here for comparison.
        # Under RULE 1 those numbers are void and must not appear beside a
        # measurement from this regime, so the chance level for the benchmark's
        # answer space is shown instead -- which is the reference that actually
        # says whether the measured coherence is above accident.
        chance = roster.chance_coherence(benchmark)
        print(f"{benchmark:<10}{f'{ci.point:.3f} [{ci.low:.3f}, {ci.high:.3f}]':>30}{chance:>26.3f}")
        print(f"{'':<10}{'-> ceiling (upper CI bound) = ' + f'{ci.high:.3f}':>30}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
