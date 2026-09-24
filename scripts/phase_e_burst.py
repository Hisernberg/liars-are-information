#!/usr/bin/env python3
"""Bounded mechanism experiment: windowed channel statistics against burst.

Two versions only -- static (pooled history) and windowed (W in {5, 10, 20}) --
then stop. This is a characterisation of a known failure, not a new research
programme.

Success criterion, fixed before running: recover at least half of the -0.207
burst loss measured in Phase D, without degrading any stationary cell by more
than its bootstrap CI width.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import phase_c_aggregate as pc  # noqa: E402
import phase_d_sweep as pds  # noqa: E402

from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.aggregation.baselines import (  # noqa: E402
    CoordinateMedian,
    GeometricMedian,
    MajorityVote,
    SACFilterRefine,
)
from aip.analysis.bootstrap import bootstrap_statistic  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest, stable_seed  # noqa: E402

log = configure_logging(level="WARNING")
WINDOWS = [5, 10, 20]
F_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def evaluate(cache, assignment, tasks, benchmark, parity, thresholds, resamples, seed, windows):
    honest = [a for a in range(pc.N_AGENTS) if a not in assignment.byzantine]
    labels = pc.LABEL_SPACES[benchmark]
    aggs = [
        AIPAggregator(benchmark, "gated", thresholds, parity, labels),
        *[AIPAggregator(benchmark, "gated", thresholds, parity, labels, window=w) for w in windows],
        MajorityVote(),
        GeometricMedian(),
        CoordinateMedian(),
        SACFilterRefine(parity=parity),
    ]
    out = []
    for agg in aggs:
        if agg.needs_fit:
            agg.fit(tasks)
        per_task = np.array(
            [
                np.mean([agg.aggregate(tasks[t][a].broadcasts, a) == cache.gold[t] for a in honest])
                if honest
                else np.nan
                for t in range(len(tasks))
            ],
            dtype=float,
        )
        valid = per_task[np.isfinite(per_task)]
        ci = bootstrap_statistic(
            lambda i, v=valid: float(np.mean(v[i])) if i.size else float("nan"),
            valid.size,
            resamples,
            seed=seed,
        )
        row = {"method": agg.name, "window": getattr(agg, "window", None) or 0}
        row.update(ci.as_dict("accuracy_"))
        row["regime_flags"] = getattr(agg, "n_regime_flagged", 0)
        out.append(row)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--resamples", type=int, default=500)
    p.add_argument("--seed", type=int, default=20250825)
    p.add_argument("--out", type=Path, default=Path("results/adversarial"))
    args = p.parse_args()

    thresholds = InversionThresholds.load()
    models = sorted(q.stem for q in (Path("data/cache") / "gsm8k").glob("*.parquet"))
    cache = pc.load_cache(args.benchmark, models, Path("data/cache"), Path("data/cache_t07"))
    negation = pds.load_adversarial("semantic_negation", args.benchmark, cache.task_ids)
    if not negation:
        print("semantic_negation cache missing")
        return 1
    parity = ParityConfig(0.1)

    rows = []
    # burst is the target; semantic_negation and noise are the stationary controls
    # that the window must not damage.
    for attack in ("burst", "semantic_negation", "noise"):
        adv = negation if attack in ("burst", "semantic_negation") else {}
        for f in F_GRID:
            assignment, tasks = pds.build_tasks(
                cache,
                list(models),
                attack,
                f,
                1.0,
                adv,
                seed=args.seed + stable_seed((args.benchmark, attack, f)),
            )
            for row in evaluate(
                cache,
                assignment,
                tasks,
                args.benchmark,
                parity,
                thresholds,
                args.resamples,
                args.seed,
                WINDOWS,
            ):
                row.update(benchmark=args.benchmark, attack=attack, f=f)
                rows.append(row)

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "burst_windowed.parquet", index=False)

    bar = "=" * 92
    print(f"\n{bar}\nWINDOWED CHANNEL STATISTICS vs BURST ({args.benchmark})\n{bar}")
    for attack in ("burst", "semantic_negation", "noise"):
        sub = frame[frame.attack == attack]
        if sub.empty:
            continue
        print(f"\n{attack}")
        print(
            sub.pivot_table(index="method", columns="f", values="accuracy_point")
            .round(3)
            .to_string()
        )

    print(f"\n{bar}\nSUCCESS CRITERION\n{bar}")
    burst = frame[frame.attack == "burst"]
    disc = burst[
        burst.method.isin(["majority", "geometric_median", "coord_median", "sac_filter_refine"])
    ]
    best_disc = disc.groupby("f")["accuracy_point"].max().mean()
    static = burst[burst.method == "aip_gated"]["accuracy_point"].mean()
    print(f"  best discard baseline (mean over f): {best_disc:.3f}")
    print(f"  AIP static                          : {static:.3f}   gap {static - best_disc:+.3f}")
    target = 0.5 * abs(static - best_disc)
    for w in WINDOWS:
        acc = burst[burst.method == f"aip_gated_w{w}"]["accuracy_point"].mean()
        recovered = acc - static
        flags = int(burst[burst.method == f"aip_gated_w{w}"]["regime_flags"].sum())
        verdict = "MEETS" if recovered >= target else "misses"
        print(
            f"  AIP windowed W={w:<3}                  : {acc:.3f}   "
            f"recovered {recovered:+.3f} of {target:.3f} needed  [{verdict}]  "
            f"regime flags {flags}"
        )

    print("\n  stationary controls (windowed minus static; must stay within CI width)")
    for attack in ("semantic_negation", "noise"):
        sub = frame[frame.attack == attack]
        s = sub[sub.method == "aip_gated"]
        width = float((s["accuracy_ci_high"] - s["accuracy_ci_low"]).mean())
        base = float(s["accuracy_point"].mean())
        for w in WINDOWS:
            acc = float(sub[sub.method == f"aip_gated_w{w}"]["accuracy_point"].mean())
            # One-sided: the criterion is "does not DEGRADE by more than a CI
            # width". An improvement larger than a CI width is not a violation,
            # and flagging it as one would have reported the window's biggest
            # win on the noise control as a failure.
            degraded = (base - acc) > width
            print(
                f"    {attack:<20} W={w:<3} {acc:.3f} vs {base:.3f}  "
                f"delta {acc - base:+.3f}  CI width {width:.3f}  "
                f"[{'DEGRADED' if degraded else 'ok'}]"
            )

    m = Manifest.create(
        phase="phase_e_burst",
        config={
            "benchmark": args.benchmark,
            "windows": WINDOWS,
            "f_grid": F_GRID,
            "resamples": args.resamples,
        },
        seeds={"sweep": args.seed},
        notes={"inference_run": False},
    )
    m.finish(status="ok").write(Path("results/manifests") / f"{m.run_id}.json")
    print(f"\nwrote {args.out / 'burst_windowed.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
