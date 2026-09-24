#!/usr/bin/env python3
"""N-scaling: does inversion's advantage grow with swarm size?

The Phase 3 sweep runs at N=10 only. This replays the SAME cached broadcasts at
N in {10, 20, 50} -- larger swarms draw more agents from the same cached model
answers, so this costs zero inference.

Why it is worth doing. The Phase 3 ablation shows inversion contributes nothing
at f=0 and about 0.27 at f=0.7, isolated by comparing aip_gated against
aip_trust_only. The open question is whether that gap is a fixed property of the
mechanism or grows with the number of agents. Discard-family methods have a
structural reason to saturate: beyond a point, throwing away more peers cannot
recover more signal, because what is discarded is gone. Inversion has no such
ceiling in principle -- every additional coherent liar is additional evidence.

If the inversion gain grows with N while the discard baselines flatten, that is a
qualitatively different claim from "AIP is more robust at high f", and a stronger
one. If it does not grow, that is worth knowing too and is reported as measured.
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

from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.aggregation.baselines import (  # noqa: E402
    ConfidenceWeighted,
    Krum,
    MajorityVote,
    SACFilterRefine,
)
from aip.analysis.bootstrap import bootstrap_statistic  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest, stable_seed  # noqa: E402
from aip.harness.rules import load_rules  # noqa: E402
from aip.swarm.broadcast import AdversaryConfig  # noqa: E402

log = configure_logging(level="WARNING")

N_GRID = [10, 20, 50]
F_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7]


def methods(benchmark: str, thresholds, parity, labels) -> list:
    """AIP with and without inversion, plus the strongest discard baselines.

    aip_trust_only is the load-bearing one: it is AIP with the inversion step
    removed and everything else identical, so the difference between it and
    aip_gated at a given (N, f) IS the contribution of inversion.
    """
    return [
        AIPAggregator(benchmark, "gated", thresholds, parity, labels),
        AIPAggregator(benchmark, "trust_only", thresholds, parity, labels),
        SACFilterRefine(parity=parity),
        MajorityVote(),
        ConfidenceWeighted(parity=parity),
        Krum(),
    ]


def main() -> int:
    rules = load_rules()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="*", default=list(rules.headline_benchmarks))
    ap.add_argument("--resamples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20250825)
    ap.add_argument("--out", type=Path, default=Path("results/aggregation"))
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "restrict the swarm to these models. REQUIRED to keep the arms apart: "
            "data/cache now holds both the frozen seven-model roster and the "
            "four-model weak-tier arm, so globbing the cache would silently pool "
            "two populations that must never be mixed."
        ),
    )
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        globals()["N_GRID"] = [10, 20]
        globals()["F_GRID"] = [0.0, 0.7]
        args.resamples, args.benchmarks = 20, args.benchmarks[:1]

    thresholds = InversionThresholds.load()
    parity = ParityConfig(0.1)
    manifest = Manifest.create(
        phase="phase_n_scaling",
        config={"benchmarks": args.benchmarks, "n_grid": N_GRID, "f_grid": F_GRID},
        seeds={"analysis": args.seed},
    )

    rows: list[dict] = []
    original_n = pc.N_AGENTS
    try:
        for benchmark in args.benchmarks:
            available = sorted(q.stem for q in (Path("data/cache") / benchmark).glob("*.parquet"))
            models = [m for m in (args.models or available) if m in available]
            if not models:
                continue
            cache = pc.load_cache(benchmark, models, Path("data/cache"), Path("data/cache_t07"))
            labels = pc.LABEL_SPACES[benchmark]
            for n_agents in N_GRID:
                # build_task_observations reads pc.N_AGENTS, so the swarm size is
                # set here rather than threaded through every call site.
                pc.N_AGENTS = n_agents
                for f in F_GRID:
                    assignment, tasks = pc.build_task_observations(
                        cache, list(models), f, 1.0, "complete", None,
                        AdversaryConfig(kind="always_wrong", error_rate=1.0),
                        seed=args.seed + stable_seed((benchmark, n_agents, f)),
                    )
                    honest = [a for a in range(n_agents) if a not in assignment.byzantine]
                    if not honest:
                        continue
                    for agg in methods(benchmark, thresholds, parity, labels):
                        if agg.needs_fit:
                            agg.fit(tasks)
                        per_task = np.array(
                            [
                                np.mean([
                                    agg.aggregate(tasks[t][a].broadcasts, a) == cache.gold[t]
                                    for a in honest
                                ])
                                for t in range(len(tasks))
                            ],
                            dtype=float,
                        )
                        ci = bootstrap_statistic(
                            lambda i, v=per_task: float(np.mean(v[i])),
                            per_task.size, args.resamples, seed=args.seed,
                        )
                        rows.append(
                            {
                                "benchmark": benchmark, "n_agents": n_agents, "f": f,
                                "method": agg.name, "n_honest": len(honest),
                                **ci.as_dict("accuracy_"),
                            }
                        )
                    log.warning("n_scaling.cell", benchmark=benchmark, n=n_agents, f=f)
    finally:
        pc.N_AGENTS = original_n

    if not rows:
        print("no rows produced", file=sys.stderr)
        return 1

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"n_scaling{args.tag}.parquet"
    frame.to_parquet(args.out / name, index=False)
    manifest.finish(status="ok", n_rows=len(frame)).write(
        Path("results/manifests") / f"{manifest.run_id}.json"
    )

    # The headline read: does the inversion gain grow with N?
    piv = frame.pivot_table(
        index=["benchmark", "f"], columns=["method", "n_agents"], values="accuracy_point"
    )
    print(f"\nwrote {args.out}/{name} ({len(frame)} rows)\n")
    gains = []
    for (b, f), _ in piv.iterrows():
        for n in N_GRID:
            try:
                g = (piv.loc[(b, f), ("aip_gated", n)]
                     - piv.loc[(b, f), ("aip_trust_only", n)])
                gains.append({"benchmark": b, "f": f, "n_agents": n, "inversion_gain": g})
            except KeyError:
                continue
    if gains:
        g = pd.DataFrame(gains)
        print("INVERSION GAIN (aip_gated - aip_trust_only) by swarm size:")
        print(g.pivot_table(index="f", columns="n_agents", values="inversion_gain").round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
