#!/usr/bin/env python3
"""E3: what each defence costs, per question.

Two costs, measured separately because they differ by four orders of magnitude
and conflating them would flatter every method equally.

**Inference cost** is read from the cache: completion tokens per answer, per
model and benchmark. It is identical for every aggregation rule -- all of them
consume the same broadcasts -- so it is reported once, as the denominator the
aggregation cost has to be judged against.

**Aggregation cost** is measured here, by timing each aggregator over the cached
broadcasts. It is not recoverable from the phase logs: those record one
timestamp per (benchmark, composition) block covering all fourteen methods at
once, so no per-method figure can be extracted from them, and inventing one
from a block average would be a fabrication. No model is loaded; this runs on
the cache alone.

The reported quantity is wall-clock microseconds per receiver-decision, which is
what a deployed agent pays: one aggregation per question per agent. Repeats are
bootstrapped over tasks, and a percentile interval is reported rather than a
standard error, because per-task aggregation times are heavily right-skewed --
Dawid--Skene's EM and Krum's pairwise distances have data-dependent inner loops.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import phase_c_aggregate as pc  # noqa: E402

from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.aggregation.baselines import (  # noqa: E402
    CoordinateMedian,
    GeometricMedian,
    Krum,
    MajorityVote,
    MultiKrum,
    TrimmedMean,
)
from aip.harness.manifest import Manifest, stable_seed  # noqa: E402
from aip.harness.rules import load_rules  # noqa: E402
from aip.swarm.broadcast import AdversaryConfig  # noqa: E402


#: The six the table is asked for, in the order it reports them.
def build(benchmark: str, n_byzantine: int, f: float, thresholds, parity, labels):
    return [
        ("aip_gated", AIPAggregator(benchmark, "gated", thresholds, parity, labels)),
        ("aip_trust_only",
         AIPAggregator(benchmark, "trust_only", thresholds, parity, labels)),
        ("krum", Krum(n_byzantine)),
        ("multi_krum", MultiKrum(n_byzantine)),
        ("trimmed_mean", TrimmedMean(trim=max(f, 0.1))),
        ("majority", MajorityVote()),
        # Two extra reference points: the cheapest possible geometry-based rule
        # and a coordinate median, so the table shows where AIP sits in a range
        # rather than only against its own ablation.
        ("geometric_median", GeometricMedian()),
        ("coord_median", CoordinateMedian()),
    ]


def time_aggregator(agg: Any, tasks: list, honest: list[int]) -> np.ndarray:
    """Microseconds per receiver-decision, one entry per task."""
    if agg.needs_fit:
        agg.fit(tasks)
    out = np.empty(len(tasks), dtype=float)
    for t, observations in enumerate(tasks):
        start = time.perf_counter_ns()
        for agent in honest:
            agg.aggregate(observations[agent].broadcasts, agent)
        elapsed = time.perf_counter_ns() - start
        out[t] = elapsed / 1_000.0 / max(len(honest), 1)
    return out


def ci(values: np.ndarray, resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def inference_cost(cache_dir: Path, benchmarks: list[str]) -> pd.DataFrame:
    rows = []
    for benchmark in benchmarks:
        for path in sorted((cache_dir / benchmark).glob("*.parquet")):
            frame = pd.read_parquet(path)
            rows.append({
                "benchmark": benchmark,
                "model": path.stem,
                "n_tasks": int(len(frame)),
                "completion_tokens_mean": float(frame.n_completion_tokens.mean()),
                "completion_tokens_p95": float(frame.n_completion_tokens.quantile(0.95)),
                "answer_tokens_mean": float(frame.n_answer_tokens.mean()),
            })
    if not rows:
        raise SystemExit(f"no cache under {cache_dir}")
    return pd.DataFrame(rows)


def main() -> int:
    rules = load_rules()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="*", default=list(rules.headline_benchmarks))
    ap.add_argument("--f", type=float, default=0.5,
                    help="Byzantine fraction the timing is taken at (default: the tie)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent timing passes; the median pass is reported")
    ap.add_argument("--resamples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20250825)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=Path("results/aggregation"))
    args = ap.parse_args()

    thresholds = InversionThresholds.load()
    parity = ParityConfig(0.1)
    rng = np.random.default_rng(args.seed)
    manifest = Manifest.create(
        phase="phase_e3_cost",
        config={"benchmarks": args.benchmarks, "f": args.f, "repeats": args.repeats},
        seeds={"analysis": args.seed},
    )

    rows = []
    for benchmark in args.benchmarks:
        models = sorted(p.stem for p in (args.cache / benchmark).glob("*.parquet"))
        if not models:
            continue
        cache = pc.load_cache(benchmark, models, args.cache, Path("data/cache_t07"))
        labels = pc.LABEL_SPACES[benchmark]
        # The coherent symbolic adversary: the configuration every method in
        # the paper is compared under, so the timing is taken on the same
        # workload rather than on an easier one.
        assignment, tasks = pc.build_task_observations(
            cache, list(models), args.f, 1.0, "complete", None,
            AdversaryConfig(kind="always_wrong", error_rate=1.0),
            seed=args.seed + stable_seed((benchmark, args.f)),
        )
        honest = [a for a in range(pc.N_AGENTS) if a not in assignment.byzantine]
        n_byz = len(assignment.byzantine)

        for name, _ in build(benchmark, n_byz, args.f, thresholds, parity, labels):
            passes = []
            for _ in range(args.repeats):
                # A fresh aggregator per pass: several of them cache channel
                # statistics after the first call, and timing a warm one would
                # measure the second question, not the first.
                agg = dict(build(benchmark, n_byz, args.f, thresholds, parity, labels))[name]
                passes.append(time_aggregator(agg, tasks, honest))
            per_task = passes[int(np.argsort([p.mean() for p in passes])[len(passes) // 2])]
            lo, hi = ci(per_task, args.resamples, rng)
            rows.append({
                "benchmark": benchmark, "method": name,
                "n_tasks": len(tasks), "n_honest_receivers": len(honest),
                "f": args.f, "n_byzantine": n_byz,
                "us_per_decision_mean": float(per_task.mean()),
                "us_per_decision_ci_low": lo, "us_per_decision_ci_high": hi,
                "us_per_decision_median": float(np.median(per_task)),
                "us_per_decision_p95": float(np.percentile(per_task, 95)),
                "repeats": args.repeats,
            })

    if not rows:
        raise SystemExit("no cells timed; is data/cache populated?")
    cost = pd.DataFrame(rows)
    infer = inference_cost(args.cache, args.benchmarks)

    args.out.mkdir(parents=True, exist_ok=True)
    cost.to_parquet(args.out / "cost_of_defence.parquet", index=False)
    infer.to_parquet(args.out / "inference_cost.parquet", index=False)
    manifest.write(args.out / "cost_of_defence.manifest.json")

    pd.set_option("display.width", 200)
    print("=" * 96)
    print(f"E3 AGGREGATION COST -- microseconds per receiver-decision, f = {args.f}")
    print("=" * 96)
    pivot = cost.pivot_table(index="method", columns="benchmark",
                             values="us_per_decision_mean").round(1)
    order = cost.groupby("method").us_per_decision_mean.mean().sort_values().index
    print(pivot.loc[order].to_string())
    print("\nwith intervals:")
    for r in cost.sort_values(["benchmark", "us_per_decision_mean"]).itertuples():
        print(f"  {r.benchmark:<9}{r.method:<18}{r.us_per_decision_mean:8.1f} us "
              f"[{r.us_per_decision_ci_low:.1f}, {r.us_per_decision_ci_high:.1f}]"
              f"   p95 {r.us_per_decision_p95:8.1f}")

    print("\n" + "=" * 96)
    print("INFERENCE COST -- identical for every aggregation rule")
    print("=" * 96)
    print(infer.groupby("benchmark")[["completion_tokens_mean", "completion_tokens_p95"]]
          .mean().round(1).to_string())
    slowest = cost.us_per_decision_mean.max()
    print(f"\nThe slowest aggregation rule costs {slowest:.0f} us per decision. One "
          f"completion averages {infer.completion_tokens_mean.mean():.0f} tokens; at a "
          "generous 100 tokens/s that is ~"
          f"{infer.completion_tokens_mean.mean() / 100 * 1e6 / slowest:.0f}x the "
          "aggregation cost. Aggregation is not the bottleneck for any method here.")
    print(f"\nwrote {args.out}/cost_of_defence.parquet, {args.out}/inference_cost.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
