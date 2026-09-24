#!/usr/bin/env python3
"""Phase E figures: burst-windowed recovery and the p_obs constraint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.analysis.figures import PALETTE, apply_style  # noqa: E402
from aip.analysis.provenance import stamp  # noqa: E402
from aip.tasks import roster  # noqa: E402


def burst_windowed(frame: pd.DataFrame, out: Path) -> Path:
    """Static vs windowed AIP under burst, with the stationary controls beside it."""
    apply_style()
    attacks = [
        ("burst", "burst (non-stationary)"),
        ("semantic_negation", "negation (stationary control)"),
        ("noise", "noise (stationary control)"),
    ]
    windows = sorted(w for w in frame.window.unique() if w)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True, sharey=True)
    shades = [PALETTE["sky"], PALETTE["blue"], PALETTE["purple"]]
    for ax, (attack, title) in zip(axes, attacks, strict=True):
        sub = frame[frame.attack == attack]
        s = sub[sub.method == "aip_gated"].sort_values("f")
        if not s.empty:
            ax.plot(
                s.f,
                s.accuracy_point,
                color=PALETTE["vermillion"],
                linewidth=2.0,
                marker="o",
                markersize=3.4,
                label="AIP static",
                zorder=4,
            )
            ax.fill_between(
                s.f,
                s.accuracy_ci_low,
                s.accuracy_ci_high,
                color=PALETTE["vermillion"],
                alpha=0.13,
                linewidth=0,
            )
        for colour, w in zip(shades, windows, strict=False):
            sw = sub[sub.method == f"aip_gated_w{int(w)}"].sort_values("f")
            if sw.empty:
                continue
            ax.plot(
                sw.f,
                sw.accuracy_point,
                color=colour,
                linewidth=1.3,
                linestyle="--",
                marker="s",
                markersize=2.8,
                label=f"AIP windowed W={int(w)}",
                zorder=3,
            )
        best = (
            sub[
                sub.method.isin(
                    ["majority", "geometric_median", "coord_median", "sac_filter_refine"]
                )
            ]
            .groupby("f")["accuracy_point"]
            .max()
        )
        if not best.empty:
            ax.plot(
                best.index,
                best.values,
                color=PALETTE["black"],
                linewidth=1.1,
                linestyle=(0, (3, 1, 1, 1)),
                marker="v",
                markersize=2.8,
                label="best discard baseline",
                zorder=2,
            )
        ax.set_title(title, pad=5)
        ax.set_xlabel("Byzantine fraction  $f$")
        ax.set_ylim(-0.03, 1.0)
        ax.grid(axis="y", alpha=0.45)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("swarm accuracy")
    axes[0].legend(
        loc="lower left", fontsize=6.4, handlelength=2.0, borderpad=0.3, labelspacing=0.25
    )
    fig.suptitle(
        "Windowed channel statistics recover the burst failure without "
        "costing the stationary cells",
        fontsize=10,
        y=1.05,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def pobs_constraint(frame: pd.DataFrame, out: Path, composition: str = "mix_all4") -> Path:
    """AIP accuracy against f at each observability level -- the real limit."""
    apply_style()
    ref = frame[
        (frame.topology == "complete")
        & (frame.adversary == "always_wrong")
        & (frame.parity == "matched@0.1")
        & (frame.composition == composition)
        & (frame.method == "aip_gated")
    ]
    benchmarks = [b for b in roster.benchmarks() if b in set(ref.benchmark)]
    levels = sorted(ref.p_obs.unique(), reverse=True)
    colours = [PALETTE["vermillion"], PALETTE["orange"], PALETTE["blue"], PALETTE["sky"]]
    fig, axes = plt.subplots(
        1,
        len(benchmarks),
        figsize=(3.3 * len(benchmarks), 3.2),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, benchmark in zip(axes, benchmarks, strict=True):
        for colour, p in zip(colours, levels, strict=False):
            s = ref[(ref.benchmark == benchmark) & (ref.p_obs == p)].sort_values("f")
            if s.empty:
                continue
            ax.plot(
                s.f,
                s.accuracy_point,
                color=colour,
                linewidth=1.6,
                marker="o",
                markersize=3.0,
                label=f"$p_{{obs}}={p:g}$",
            )
            ax.fill_between(
                s.f, s.accuracy_ci_low, s.accuracy_ci_high, color=colour, alpha=0.10, linewidth=0
            )
        ax.set_title(benchmark, pad=5)
        ax.set_xlabel("Byzantine fraction  $f$")
        ax.set_ylim(-0.03, 1.0)
        ax.grid(axis="y", alpha=0.45)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("AIP-gated accuracy")
    axes[0].legend(loc="lower left", fontsize=6.8, handlelength=2.0)
    fig.suptitle(
        "Partial observability, not the Byzantine fraction, is the binding constraint on inversion",
        fontsize=10,
        y=1.05,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--burst", type=Path, default=Path("results/adversarial/burst_windowed.parquet"))
    p.add_argument("--sweep", type=Path, default=Path("results/aggregation/sweep.parquet"))
    p.add_argument("--figures", type=Path, default=Path("paper/figures"))
    args = p.parse_args()

    if args.burst.exists():
        out = burst_windowed(pd.read_parquet(args.burst),
                             args.figures / "burst_windowed.pdf")
        stamp(out, script="scripts/phase_e_figures.py", sources=[args.burst])
        print("wrote", out)
    if args.sweep.exists():
        out = pobs_constraint(pd.read_parquet(args.sweep),
                              args.figures / "pobs_constraint.pdf")
        stamp(out, script="scripts/phase_e_figures.py", sources=[args.sweep])
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
