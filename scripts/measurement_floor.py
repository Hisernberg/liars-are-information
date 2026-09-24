#!/usr/bin/env python3
"""The measurement floor: how small an effect this study can actually resolve.

Until this exists, no result here may be called "small" or "no effect" -- only
"undetectable at this scale". The distinction is not pedantry. Several statements
already in the paper ("inversion costs 0.001 at f=0", "gain is invariant in N")
are claims that a quantity is NEAR ZERO, and such a claim is meaningless without
knowing what zero looks like under this measurement.

Two independent noise sources are estimated:

**Resample noise.** The T=0.7 second pass re-answers the SAME tasks with the SAME
model. Any disagreement between it and the greedy pass is the model's own
sampling variability, which sets a floor on how precisely any single agent's
behaviour can be pinned down. For the reasoning models this is *path-level*
variation and is reported separately, because resampling a chain of thought is a
different quantity from resampling an answer.

**Aggregation noise.** A swarm accuracy over T tasks with N agents is a mean of
Bernoulli draws; its standard error is the binomial term. This is what bounds a
difference between two methods on the same cells.

The minimum detectable effect is the two-sided 95% interval on a *difference* of
two such means, sqrt(2) * 1.96 * sigma -- the quantity an observed delta must
exceed before it can be called real.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.tasks import roster  # noqa: E402

Z95 = 1.959963985


def resample_noise(benchmark: str) -> list[dict]:
    """Disagreement between the greedy and T=0.7 passes, per model."""
    rows = []
    for pq in sorted((Path("data/cache") / benchmark).glob("*.parquet")):
        model = pq.stem
        t07 = Path("data/cache_t07") / benchmark / f"{model}.parquet"
        if not t07.exists():
            continue
        a = pd.read_parquet(pq).set_index("task_id")
        b = pd.read_parquet(t07).set_index("task_id")
        shared = a.index.intersection(b.index)
        if len(shared) < 20:
            continue
        ca = a.loc[shared, "is_correct"].fillna(False).to_numpy(dtype=bool)
        cb = b.loc[shared, "is_correct"].fillna(False).to_numpy(dtype=bool)
        kind = a.loc[shared, "resample_kind"].iloc[0] if "resample_kind" in a else "?"
        rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "resample_kind": kind,
                "n_tasks": int(len(shared)),
                "accuracy_greedy": float(ca.mean()),
                "accuracy_t07": float(cb.mean()),
                "flip_rate": float((ca != cb).mean()),
                "abs_accuracy_delta": float(abs(ca.mean() - cb.mean())),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-grid", nargs="*", type=int, default=[10, 20, 50])
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    rows: list[dict] = []
    for b in roster.benchmarks():
        rows += resample_noise(b)
    if not rows:
        print("no T=0.7 cells found", file=sys.stderr)
        return 1
    frame = pd.DataFrame(rows)

    import json as _json
    frozen = _json.loads(Path("configs/task_lists.json").read_text())["benchmarks"]
    tasks_per = {b: int(frozen[b]["n"]) for b in roster.benchmarks()}

    # Only GSM8K carries a T=0.7 second pass, so it is the only benchmark with a
    # directly measured resample sigma. The others inherit it as a PROXY, flagged
    # as such: their floors would otherwise be built from aggregation noise alone
    # and would understate how much of an observed difference is resampling.
    measured = frame.groupby("resample_kind").abs_accuracy_delta.mean().to_dict()
    default_sigma = float(np.mean(list(measured.values()))) if measured else 0.01

    floors = []
    for b in roster.benchmarks():
        sub = frame[frame.benchmark == b]
        if not len(sub):
            # No T=0.7 pass here: aggregation noise measured, resample noise
            # borrowed from GSM8K and marked.
            n_tasks = tasks_per.get(b, 200)
            for n_agents in args.n_grid:
                sigma_agg = 0.5 / np.sqrt(n_tasks)
                sigma = float(np.hypot(default_sigma, sigma_agg))
                floors.append(
                    {
                        "benchmark": b, "resample_kind": "proxy(gsm8k)",
                        "n_agents": n_agents, "n_tasks": n_tasks,
                        "sigma_resample": default_sigma,
                        "sigma_aggregation": float(sigma_agg),
                        "sigma_total": sigma,
                        "min_detectable_effect": float(np.sqrt(2) * Z95 * sigma),
                    }
                )
            continue
        # Resample noise: the spread of accuracy between two samples of the same
        # model. Answer-level and path-level are kept apart -- pooling them would
        # let a reasoning model's chain-of-thought variance inflate the floor for
        # benchmarks measured on non-reasoning models.
        for kind in sorted(sub.resample_kind.unique()) if len(sub) else []:
            k = sub[sub.resample_kind == kind]
            sigma_resample = float(k.abs_accuracy_delta.mean()) if len(k) else float("nan")
            n_tasks = tasks_per.get(b, 200)
            for n_agents in args.n_grid:
                # Aggregation noise: SE of a mean over n_tasks Bernoulli outcomes,
                # at the worst-case p=0.5. Independent of N in the limit, since a
                # swarm decision is one outcome per task however many agents vote.
                sigma_agg = 0.5 / np.sqrt(n_tasks)
                sigma = float(np.hypot(sigma_resample, sigma_agg))
                floors.append(
                    {
                        "benchmark": b,
                        "resample_kind": kind,
                        "n_agents": n_agents,
                        "n_tasks": n_tasks,
                        "sigma_resample": sigma_resample,
                        "sigma_aggregation": float(sigma_agg),
                        "sigma_total": sigma,
                        # Two-sided 95% MDE on a DIFFERENCE of two means.
                        "min_detectable_effect": float(np.sqrt(2) * Z95 * sigma),
                    }
                )

    fl = pd.DataFrame(floors)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "resample_noise.parquet", index=False)
    fl.to_parquet(args.out / "measurement_floor.parquet", index=False)

    print("=== RESAMPLE NOISE (greedy vs T=0.7, same model, same tasks) ===")
    print(frame.groupby(["benchmark", "resample_kind"])[["flip_rate", "abs_accuracy_delta"]]
          .mean().round(4).to_string())
    print("\n=== MEASUREMENT FLOOR: minimum detectable effect (95%, difference of means) ===")
    piv = fl.pivot_table(index=["benchmark", "resample_kind"], columns="n_agents",
                         values="min_detectable_effect")
    print(piv.round(3).to_string())
    worst = fl.min_detectable_effect.max()
    print(f"\nWORST-CASE FLOOR ACROSS ALL CELLS: {worst:.3f}")
    print("Any effect whose CI crosses its cell's floor is UNDETECTABLE AT THIS SCALE.")
    print("It must not be described as 'small' or as 'no effect'.")

    (args.out / "measurement_floor.json").write_text(
        json.dumps(
            {
                "worst_case_floor": float(worst),
                "per_benchmark": {
                    f"{r.benchmark}:{r.resample_kind}:N{r.n_agents}": round(
                        r.min_detectable_effect, 4
                    )
                    for r in fl.itertuples()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}/measurement_floor.parquet and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
