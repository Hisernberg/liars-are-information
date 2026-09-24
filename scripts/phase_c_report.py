#!/usr/bin/env python3
"""Phase C tables: headline sweep, normalization-parity audit, and the gate audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aip.tasks import roster  # noqa: E402

BENCHMARKS = roster.benchmarks()  # from configs/task_lists.json
#: Methods whose behaviour depends on how much weight the self-vote carries.
WEIGHTED = [
    "confidence_weighted",
    "confidence_weighted_selfreport",
    "sac_filter_refine",
    "aip_gated",
    "aip_naive",
    "aip_trust_only",
]


def reference(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df.p_obs == 1.0)
        & (df.topology == "complete")
        & (df.adversary == "always_wrong")
        & (df.parity == "matched@0.1")
    ]


def headline(df: pd.DataFrame, composition: str) -> None:
    print(
        f"\n{'=' * 104}\n1. ACCURACY vs f  ({composition}, complete graph, p_obs=1, coherent adversary,"
    )
    print(f"   matched parity)\n{'=' * 104}")
    ref = reference(df)
    for benchmark in BENCHMARKS:
        sub = ref[(ref.benchmark == benchmark) & (ref.composition == composition)]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="method", columns="f", values="accuracy_point")
        print(f"\n  {benchmark}")
        print(pivot.round(3).to_string())


def crossover(df: pd.DataFrame, composition: str) -> None:
    print(f"\n{'=' * 104}\n2. WHERE INVERSION HELPS AND WHERE IT HURTS\n{'=' * 104}")
    ref = reference(df)
    print(f"{'benchmark':<9}{'f':>5}{'AIP-gated':>11}{'best baseline':>15}{'name':>22}{'delta':>9}")
    for benchmark in BENCHMARKS:
        sub = ref[(ref.benchmark == benchmark) & (ref.composition == composition)]
        if sub.empty:
            continue
        for f in sorted(sub.f.unique()):
            cell = sub[sub.f == f]
            aip = cell[cell.method == "aip_gated"]["accuracy_point"]
            base = cell[~cell.method.str.startswith("aip")]
            if aip.empty or base.empty:
                continue
            best = base.loc[base["accuracy_point"].idxmax()]
            delta = float(aip.iloc[0]) - float(best["accuracy_point"])
            mark = (
                "  <-- inversion wins"
                if delta > 0.02
                else ("  <-- inversion loses" if delta < -0.02 else "")
            )
            print(
                f"{benchmark:<9}{f:>5.1f}{float(aip.iloc[0]):>11.3f}"
                f"{float(best['accuracy_point']):>15.3f}{best['method']:>22}{delta:>+9.3f}{mark}"
            )


def gate_audit(df: pd.DataFrame, composition: str) -> None:
    print(f"\n{'=' * 104}\n3. DID THE HONEST-q THRESHOLD DO ITS JOB?\n{'=' * 104}")
    ref = reference(df)
    sub = ref[(ref.composition == composition) & (ref.method.str.startswith("aip"))]
    print("   honest channels inverted / honest channels judged")
    print(f"{'benchmark':<9}{'method':<18}" + "".join(f"{f:>9.1f}" for f in sorted(sub.f.unique())))
    for benchmark in BENCHMARKS:
        for method in ["aip_gated", "aip_naive"]:
            s = sub[(sub.benchmark == benchmark) & (sub.method == method)].sort_values("f")
            if s.empty:
                continue
            cells = "".join(
                f"{r.honest_inversion_rate:>9.2f}"
                if np.isfinite(r.honest_inversion_rate)
                else f"{'--':>9}"
                for r in s.itertuples()
            )
            print(f"{benchmark:<9}{method:<18}{cells}")

    print("\n   accuracy cost of naive inversion (gated minus naive)")
    print(f"{'benchmark':<9}" + "".join(f"{f:>9.1f}" for f in sorted(sub.f.unique())))
    for benchmark in BENCHMARKS:
        g = sub[(sub.benchmark == benchmark) & (sub.method == "aip_gated")].set_index("f")
        n = sub[(sub.benchmark == benchmark) & (sub.method == "aip_naive")].set_index("f")
        if g.empty or n.empty:
            continue
        cells = "".join(
            f"{g.loc[f, 'accuracy_point'] - n.loc[f, 'accuracy_point']:>+9.3f}"
            for f in sorted(sub.f.unique())
            if f in g.index and f in n.index
        )
        print(f"{benchmark:<9}{cells}")


def parity_audit(df: pd.DataFrame) -> pd.DataFrame:
    """The paper artifact: how much does self-vote share alone move each method?"""
    print(f"\n{'=' * 104}\n4. NORMALIZATION-PARITY AUDIT\n{'=' * 104}")
    sub = df[(df.p_obs == 1.0) & (df.topology == "complete") & (df.adversary == "always_wrong")]
    rows = []
    for (benchmark, composition, method, f), g in sub.groupby(
        ["benchmark", "composition", "method", "f"]
    ):
        matched = g[g.parity == "matched@0.1"]["accuracy_point"]
        unmatched = g[g.parity == "unmatched"]["accuracy_point"]
        if matched.empty or unmatched.empty:
            continue
        rows.append(
            {
                "benchmark": benchmark,
                "composition": composition,
                "method": method,
                "f": f,
                "accuracy_matched": float(matched.iloc[0]),
                "accuracy_unmatched": float(unmatched.iloc[0]),
                "parity_effect": float(unmatched.iloc[0]) - float(matched.iloc[0]),
            }
        )
    audit = pd.DataFrame(rows)
    if audit.empty:
        print("   (no paired parity cells)")
        return audit

    print(f"{'method':<32}{'mean |effect|':>14}{'max |effect|':>14}{'n cells':>9}  weighted?")
    summary = (
        audit.assign(abs_effect=audit.parity_effect.abs())
        .groupby("method")
        .agg(
            mean_abs=("abs_effect", "mean"), max_abs=("abs_effect", "max"), n=("abs_effect", "size")
        )
        .sort_values("mean_abs", ascending=False)
    )
    for method, r in summary.iterrows():
        print(
            f"{method:<32}{r.mean_abs:>14.4f}{r.max_abs:>14.4f}{int(r.n):>9}"
            f"  {'yes' if method in WEIGHTED else 'no'}"
        )
    print(
        "\n   Effect = unmatched minus matched accuracy. A method with a large effect is "
        "\n   partly being scored on how much it leans on its own vote rather than on how "
        "\n   well it weights peers, which is exactly the confound parity controls for."
    )
    return audit


def sensitivity(df: pd.DataFrame, composition: str) -> None:
    print(f"\n{'=' * 104}\n5. SENSITIVITY: topology and partial observability\n{'=' * 104}")
    base = df[
        (df.adversary == "always_wrong")
        & (df.parity == "matched@0.1")
        & (df.composition == composition)
    ]
    print("   AIP-gated accuracy by topology (p_obs=1)")
    t = base[base.p_obs == 1.0]
    t = t[t.method == "aip_gated"]
    if not t.empty:
        print(
            t.pivot_table(index=["benchmark", "topology"], columns="f", values="accuracy_point")
            .round(3)
            .to_string()
        )
    print("\n   AIP-gated accuracy by p_obs (complete graph)")
    p = base[(base.topology == "complete") & (base.method == "aip_gated")]
    if not p.empty:
        print(
            p.pivot_table(index=["benchmark", "p_obs"], columns="f", values="accuracy_point")
            .round(3)
            .to_string()
        )

    print("\n   Coherent vs noise adversary (AIP-gated, p_obs=1, complete)")
    adv = df[
        (df.parity == "matched@0.1")
        & (df.p_obs == 1.0)
        & (df.topology == "complete")
        & (df.composition == composition)
        & (df.method == "aip_gated")
    ]
    if not adv.empty:
        print(
            adv.pivot_table(index=["benchmark", "adversary"], columns="f", values="accuracy_point")
            .round(3)
            .to_string()
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("results/aggregation/sweep.parquet"))
    parser.add_argument("--composition", default="mix_all4")
    parser.add_argument("--out", type=Path, default=Path("results/aggregation"))
    args = parser.parse_args()

    df = pd.read_parquet(args.sweep)
    print(
        f"sweep rows: {len(df)}   methods: {df.method.nunique()}   "
        f"cells: {len(df.drop_duplicates(['benchmark', 'composition', 'f', 'p_obs', 'topology', 'adversary', 'parity']))}"
    )
    headline(df, args.composition)
    crossover(df, args.composition)
    gate_audit(df, args.composition)
    audit = parity_audit(df)
    if not audit.empty:
        audit.to_parquet(args.out / "parity_audit.parquet", index=False)
        print(f"\n   wrote {args.out / 'parity_audit.parquet'}")
    sensitivity(df, args.composition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
