#!/usr/bin/env python3
"""Phase 5: E1 sleeper infiltration and E2 Sybil coherence amplification.

Both target *admitted* limitations of published systems rather than weaknesses we
invented for them, which is what makes the section worth writing.

E1 -- SLEEPER. An adversary behaves honestly for k rounds, accumulating standing,
then switches to a coherent lie. Reputation systems are the standard answer to
Byzantine peers, and their structure is what the attack exploits: reputation is
earned before it is spent, so a defector is still trusted on the strength of its
cover story for roughly one decay half-life after it turns. AIP's gate conditions
on coherence with other dissenters instead, recomputed from a window rather than
integrated over history, so an honest phase buys it no credit. The prediction is
that reputation carries trust across the betrayal and AIP-windowed does not.

E2 -- SYBIL. An adversary controls s of ten identities, all broadcasting the SAME
lie. Bloc-size defences degrade as s grows because the bloc becomes the plurality.
The prediction under test is the opposite for AIP: a bigger bloc is a MORE
coherent channel, so inversion should hold flat or improve in s. The
disagreeing-Sybil control is what makes that a real test -- s identities each
lying differently give no coherent channel, and AIP should then discard rather
than invert. Without the control, "AIP survives Sybils" could just mean "AIP
ignores everything".

Reported honestly either way: if reputation decay handles the sleeper well, or
if AIP degrades in s, that is the result.
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
    ConfidenceWeighted,
    MajorityVote,
    SACFilterRefine,
)
from aip.aggregation.reputation import ReputationDecay  # noqa: E402
from aip.analysis.bootstrap import bootstrap_statistic  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest, stable_seed  # noqa: E402
from aip.harness.rules import load_rules  # noqa: E402
from aip.tasks import roster  # noqa: E402

log = configure_logging(level="WARNING")

K_GRID = [10, 25, 50]           # rounds of cover before the sleeper defects
F_GRID_E1 = [0.1, 0.3, 0.5]
SYBIL_SIZES = [1, 2, 4, 8]      # identities in the bloc, out of N_AGENTS
WINDOW = 20                     # AIP-windowed, the defender E1 is aimed at


def defenders(benchmark: str, thresholds, parity, labels) -> list:
    """The four defences E1 and E2 compare, plus the two vote baselines."""
    return [
        AIPAggregator(benchmark, "gated", thresholds, parity, labels, window=WINDOW),
        AIPAggregator(benchmark, "gated", thresholds, parity, labels),  # static
        ReputationDecay(parity=parity),
        SACFilterRefine(parity=parity),
        MajorityVote(),
        ConfidenceWeighted(parity=parity),
    ]


def accuracy_over(agg, cache, tasks, honest, lo: int, hi: int) -> float:
    """Mean accuracy over rounds [lo, hi), averaged across honest receivers."""
    if not honest or hi <= lo:
        return float("nan")
    vals = [
        np.mean([agg.aggregate(tasks[t][a].broadcasts, a) == cache.gold[t] for a in honest])
        for t in range(lo, min(hi, len(tasks)))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def rounds_to_recover(agg, cache, tasks, honest, k: int, baseline: float, tol: float = 0.05) -> int:
    """Rounds after the switch until accuracy returns to within ``tol`` of cover-phase level.

    Returns -1 when it never recovers inside the horizon, which is itself a
    result and must not be silently reported as the horizon length.
    """
    window = 5
    for start in range(k, len(tasks) - window):
        if accuracy_over(agg, cache, tasks, honest, start, start + window) >= baseline - tol:
            return start - k
    return -1


def run_e1(cache, benchmark, thresholds, parity, labels, models, resamples, seed) -> list[dict]:
    rows = []
    adv = pds.load_adversarial("semantic_negation", benchmark, cache.task_ids)
    if not adv:
        log.warning("e1.no_adversarial_cache", benchmark=benchmark)
        return rows
    n = len(cache.task_ids)
    for k in K_GRID:
        if k >= n:
            continue
        # A single switch, not an alternating cycle: honest for k, then coherent.
        schedule = [t >= k for t in range(n)]
        for f in F_GRID_E1:
            assignment, tasks = pds.build_tasks(
                cache, list(models), "burst", f, 1.0, adv,
                seed=seed + stable_seed((benchmark, "e1", k, f)),
                coherent_schedule=schedule,
            )
            honest = [a for a in range(pc.N_AGENTS) if a not in assignment.byzantine]
            for agg in defenders(benchmark, thresholds, parity, labels):
                if agg.needs_fit:
                    agg.fit(tasks)
                cover = accuracy_over(agg, cache, tasks, honest, 0, k)
                betrayal = accuracy_over(agg, cache, tasks, honest, k, n)
                per_task = np.array(
                    [
                        np.mean(
                            [agg.aggregate(tasks[t][a].broadcasts, a) == cache.gold[t] for a in honest]
                        )
                        for t in range(n)
                    ],
                    dtype=float,
                )
                ci = bootstrap_statistic(
                    lambda i, v=per_task: float(np.mean(v[i])),
                    per_task.size, resamples, seed=seed,
                )
                rows.append(
                    {
                        "experiment": "E1_sleeper", "benchmark": benchmark, "method": agg.name,
                        "k": k, "f": f, "n_rounds": n,
                        "accuracy_cover": cover,
                        "accuracy_betrayal": betrayal,
                        "drop": cover - betrayal,
                        "rounds_to_recover": rounds_to_recover(
                            agg, cache, tasks, honest, k, cover
                        ),
                        **ci.as_dict("accuracy_total_"),
                    }
                )
    return rows


def run_e2(cache, benchmark, thresholds, parity, labels, models, resamples, seed) -> list[dict]:
    rows = []
    adv = pds.load_adversarial("semantic_negation", benchmark, cache.task_ids)
    if not adv:
        log.warning("e2.no_adversarial_cache", benchmark=benchmark)
        return rows
    one = sorted(adv)[0]  # the bloc's single shared generator
    n = len(cache.task_ids)
    for s in SYBIL_SIZES:
        f = s / pc.N_AGENTS
        for coherent in (True, False):
            assignment, tasks = pds.build_tasks(
                cache, list(models), "semantic_negation", f, 1.0, adv,
                seed=seed + stable_seed((benchmark, "e2", s, coherent)),
                # Coherent: every identity in the bloc tells the same lie.
                # Control: identities draw generators round-robin and disagree.
                single_adversary=one if coherent else None,
            )
            honest = [a for a in range(pc.N_AGENTS) if a not in assignment.byzantine]
            if not honest:
                continue
            for agg in defenders(benchmark, thresholds, parity, labels):
                if agg.needs_fit:
                    agg.fit(tasks)
                per_task = np.array(
                    [
                        np.mean(
                            [agg.aggregate(tasks[t][a].broadcasts, a) == cache.gold[t] for a in honest]
                        )
                        for t in range(n)
                    ],
                    dtype=float,
                )
                ci = bootstrap_statistic(
                    lambda i, v=per_task: float(np.mean(v[i])),
                    per_task.size, resamples, seed=seed,
                )
                rows.append(
                    {
                        "experiment": "E2_sybil", "benchmark": benchmark, "method": agg.name,
                        "sybil_size": s, "f": f,
                        "bloc": "coherent" if coherent else "disagreeing",
                        **ci.as_dict("accuracy_"),
                    }
                )
    return rows


def main() -> int:
    rules = load_rules()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="*", default=list(rules.headline_benchmarks))
    ap.add_argument("--resamples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20250825)
    ap.add_argument("--out", type=Path, default=Path("results/adversarial"))
    ap.add_argument("--smoke", action="store_true", help="tiny grids, wiring check only")
    args = ap.parse_args()

    if args.smoke:
        globals()["K_GRID"] = [5]
        globals()["F_GRID_E1"] = [0.3]
        globals()["SYBIL_SIZES"] = [2]
        args.resamples = 20
        args.benchmarks = args.benchmarks[:1]

    thresholds = InversionThresholds.load()
    parity = ParityConfig(0.1)
    manifest = Manifest.create(
        phase="phase_e_sleeper_sybil",
        config={"benchmarks": args.benchmarks, "k_grid": K_GRID,
                "sybil_sizes": SYBIL_SIZES, "window": WINDOW},
        seeds={"analysis": args.seed},
    )

    rows: list[dict] = []
    for benchmark in args.benchmarks:
        if benchmark not in roster.benchmarks():
            continue
        models = sorted(q.stem for q in (Path("data/cache") / benchmark).glob("*.parquet"))
        if not models:
            continue
        cache = pc.load_cache(benchmark, models, Path("data/cache"), Path("data/cache_t07"))
        labels = pc.LABEL_SPACES[benchmark]
        rows += run_e1(cache, benchmark, thresholds, parity, labels, models,
                       args.resamples, args.seed)
        rows += run_e2(cache, benchmark, thresholds, parity, labels, models,
                       args.resamples, args.seed)

    if not rows:
        print("no rows produced -- adversarial caches missing?", file=sys.stderr)
        return 1

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    e1 = frame[frame.experiment == "E1_sleeper"]
    e2 = frame[frame.experiment == "E2_sybil"]
    if len(e1):
        e1.to_parquet(args.out / "sleeper_e1.parquet", index=False)
    if len(e2):
        e2.to_parquet(args.out / "sybil_e2.parquet", index=False)
    manifest.finish(status="ok", n_rows=len(frame)).write(
        Path("results/manifests") / f"{manifest.run_id}.json"
    )
    print(f"wrote sleeper_e1.parquet ({len(e1)} rows) and sybil_e2.parquet ({len(e2)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
