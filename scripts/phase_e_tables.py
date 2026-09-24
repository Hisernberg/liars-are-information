#!/usr/bin/env python3
"""Phase E tables: v1/v2 delta, minimax regret, parity audit, limitations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.analysis.regret import minimax_summary, regret_table  # noqa: E402
from aip.tasks import roster  # noqa: E402

REF = dict(p_obs=1.0, topology="complete", adversary="always_wrong", parity="matched@0.1")
KEYS = ["benchmark", "composition", "f", "p_obs", "topology", "adversary", "parity", "method"]


def delta_report(v1: pd.DataFrame, v2: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Every cell where the gate version changed the number."""
    a = v1.set_index(KEYS)["accuracy_point"].rename("v1")
    b = v2.set_index(KEYS)["accuracy_point"].rename("v2")
    joined = pd.concat([a, b], axis=1).dropna().reset_index()
    joined["delta"] = joined.v2 - joined.v1
    aip = joined[joined.method.str.startswith("aip")]

    print(f"\n{'=' * 96}\n1. CONSISTENCY PASS — gate v1 vs v2 (symbolic sweep)\n{'=' * 96}")
    print(f"  cells compared: {len(joined)}   AIP cells: {len(aip)}")
    non_aip = joined[~joined.method.str.startswith("aip")]
    # The archived v1 sweep predates the seeding fix, when swarm draws came from
    # Python's randomised string hash. Methods the gate cannot touch therefore
    # differ between v1 and v2 purely by swarm draw, and that spread is the noise
    # floor against which any AIP delta has to be judged.
    noise_floor = float(non_aip.delta.abs().quantile(0.95))
    print(
        f"  SEED-NOISE FLOOR (non-AIP cells, which the gate cannot affect): "
        f"{(non_aip.delta.abs() > 1e-9).sum()} of {len(non_aip)} differ, "
        f"95th pct |delta| {noise_floor:.3f}, max {float(non_aip.delta.abs().max()):.3f}"
    )
    print(f"  Only AIP deltas above {noise_floor:.3f} are attributable to the mechanism.")
    print("  Seeds are deterministic from this run on (manifest.stable_seed).")
    print(f"  AIP cells changed:     {(aip.delta.abs() > 1e-9).sum()} of {len(aip)}")

    ref = aip[
        (aip.p_obs == 1.0)
        & (aip.topology == "complete")
        & (aip.adversary == "always_wrong")
        & (aip.parity == "matched@0.1")
        & (aip.composition == "mix_all4")
        & (aip.method == "aip_gated")
    ]
    print("\n  AIP-gated, mix_all4, reference condition — accuracy by f")
    print(f"  {'benchmark':<10}{'ver':<5}" + "".join(f"{f:>8.1f}" for f in sorted(ref.f.unique())))
    for benchmark in roster.benchmarks():
        s = ref[ref.benchmark == benchmark].sort_values("f")
        if s.empty:
            continue
        print(f"  {benchmark:<10}{'v1':<5}" + "".join(f"{x:>8.3f}" for x in s.v1))
        print(f"  {'':<10}{'v2':<5}" + "".join(f"{x:>8.3f}" for x in s.v2))
        print(f"  {'':<10}{'Δ':<5}" + "".join(f"{x:>+8.3f}" for x in s.delta))

    print("\n  headline check: is AIP-gated still flat in f?")
    for benchmark in roster.benchmarks():
        s = ref[ref.benchmark == benchmark]
        if s.empty:
            continue
        spread_v1 = float(s.v1.max() - s.v1.min())
        spread_v2 = float(s.v2.max() - s.v2.min())
        verdict = "FLAT" if spread_v2 < 0.15 else "NOT FLAT"
        biggest = float(s.delta.abs().max())
        source = "mechanism" if biggest > noise_floor else "within seed noise"
        print(
            f"    {benchmark:<10} v1 spread {spread_v1:.3f} -> v2 spread {spread_v2:.3f}  "
            f"[{verdict}]  max |Δ| {biggest:.3f} ({source})"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out, index=False)
    return joined


def regret(v2: pd.DataFrame) -> pd.DataFrame:
    ref = v2[
        (v2.p_obs == 1.0)
        & (v2.topology == "complete")
        & (v2.adversary == "always_wrong")
        & (v2.parity == "matched@0.1")
        & (v2.composition == "mix_all4")
    ]
    table = regret_table(ref, deployable_only=True)
    summary = minimax_summary(table)
    print(
        f"\n{'=' * 96}\n2. MINIMAX REGRET over f (deployable, non-oracle methods only)\n{'=' * 96}"
    )
    print(f"{'benchmark':<10}{'method':<34}{'minimax regret':>16}{'mean regret':>13}")
    for r in summary.itertuples():
        print(f"{r.benchmark:<10}{r.method:<34}{r.minimax_regret:>16.3f}{r.mean_regret:>13.3f}")
    print("\n  Oracle methods (Krum, Multi-Krum, trimmed mean) are excluded: they are given")
    print("  the true Byzantine count, which no deployed agent has.")
    return summary


def parity(v2: pd.DataFrame) -> pd.DataFrame:
    sub = v2[(v2.p_obs == 1.0) & (v2.topology == "complete") & (v2.adversary == "always_wrong")]
    rows = []
    for (b, c, m, f), g in sub.groupby(["benchmark", "composition", "method", "f"]):
        mt = g[g.parity == "matched@0.1"]["accuracy_point"]
        um = g[g.parity == "unmatched"]["accuracy_point"]
        if mt.empty or um.empty:
            continue
        rows.append(
            {
                "benchmark": b,
                "composition": c,
                "method": m,
                "f": f,
                "matched": float(mt.iloc[0]),
                "unmatched": float(um.iloc[0]),
                "effect": float(um.iloc[0]) - float(mt.iloc[0]),
            }
        )
    audit = pd.DataFrame(rows)
    print(f"\n{'=' * 96}\n3. NORMALIZATION-PARITY AUDIT (v2)\n{'=' * 96}")
    print(f"{'method':<34}{'mean |effect|':>15}{'max |effect|':>14}{'cells':>7}")
    summary = (
        audit.assign(a=audit.effect.abs())
        .groupby("method")
        .agg(mean_abs=("a", "mean"), max_abs=("a", "max"), n=("a", "size"))
        .sort_values("mean_abs", ascending=False)
    )
    for m, r in summary.iterrows():
        print(f"{m:<34}{r.mean_abs:>15.4f}{r.max_abs:>14.4f}{int(r.n):>7}")
    return audit


def accuracy_landscape() -> pd.DataFrame:
    import json

    report = json.load(open("results/phase_a_full_s0_report.json"))
    rows = [
        {
            "model": c["model"],
            "benchmark": c["benchmark"],
            "accuracy": c["accuracy"],
            "flag": c["headroom_flag"],
            "auc": c["confidence_auc"],
        }
        for c in report["cells"]
        if not c.get("error")
    ]
    frame = pd.DataFrame(rows)
    print(f"\n{'=' * 96}\n4. MODEL-ACCURACY LANDSCAPE (Phase A gate f)\n{'=' * 96}")
    print(
        frame.pivot_table(index="model", columns="benchmark", values="accuracy")
        .round(3)
        .to_string()
    )
    flagged = frame[frame.flag != "ok"]
    print(
        f"\n  cells above 0.95 or below 0.20: {len(flagged)} "
        f"({'none' if flagged.empty else list(flagged.itertuples())})"
    )
    return frame


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--v1", type=Path, default=Path("results/aggregation/v1_archive/sweep_v1.parquet")
    )
    p.add_argument("--v2", type=Path, default=Path("results/aggregation/sweep.parquet"))
    p.add_argument("--out", type=Path, default=Path("results/aggregation"))
    args = p.parse_args()

    v2 = pd.read_parquet(args.v2)

    # The gate v1-vs-v2 delta compared two MECHANISM versions inside one hardware
    # regime, and it existed to prove that a gate change had not quietly moved
    # cells the gate cannot touch. This regime has only ever run one gate
    # version, and the L4-era v1 archive is void under RULE 1 -- comparing
    # against it would mix regimes, which is exactly what that rule forbids. So
    # the delta is UNDEFINED here rather than missing, and its absence must not
    # take the three tables that follow down with it.
    if args.v1.exists():
        delta = delta_report(pd.read_parquet(args.v1), v2, args.out / "gate_version_delta.parquet")
        del delta
    else:
        print(f"gate delta skipped: no v1 archive at {args.v1} (single-gate regime)")

    reg = regret(v2)
    aud = parity(v2)
    land = accuracy_landscape()

    reg.to_parquet(args.out / "minimax_regret.parquet", index=False)
    aud.to_parquet(args.out / "parity_audit.parquet", index=False)
    land.to_parquet(args.out / "accuracy_landscape.parquet", index=False)
    written = ["minimax_regret.parquet", "parity_audit.parquet", "accuracy_landscape.parquet"]
    if args.v1.exists():
        written.insert(0, "gate_version_delta.parquet")
    print(f"\nwrote {args.out}/" + ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
