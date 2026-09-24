#!/usr/bin/env python3
"""The evasion band, anchored at f = 0.5.

Why f = 0.5 and nothing else. At f = 0.7 the Byzantine bloc is a numerical
majority, so a swarm-accuracy collapse cannot be attributed to the gate: the
adversary wins by counting, not by evading. At f = 0.3 the honest side dominates
and the gate is barely load-bearing. Only at the exact tie does the outcome turn
on whether the gate fires, which is the quantity this figure is about. Every
number in the paper's evasion story is therefore read off the f = 0.5 rows.

Two things this figure must not do, both of which it was previously doing:

1. **No interpolation.** The coherence grid produces only three distinct values
   of realised q in [0.15, 0.40] -- the interval where the attack lives -- and q
   is not monotone in the coordination probability p. A line through those
   markers would draw a curve the experiment never measured, in exactly the
   region the paper's claim depends on. Markers only; the resolution-limited
   interval is shaded and labelled.
2. **No unqualified honest-coherence overlay.** The honest distribution is a
   per-pair estimate conditioned on both agents being wrong, and on MMLU only
   one of twenty-one pairs reaches the n >= 30 needed for it to be
   interpretable (rule R2). Powered and underpowered pairs are drawn
   differently and the counts are printed on the panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.analysis.figures import PALETTE, apply_style  # noqa: E402
from aip.analysis.paper_numbers import EVASION_ANCHOR_F, EVASION_BAND  # noqa: E402
from aip.analysis.provenance import stamp  # noqa: E402
from aip.harness.rules import load_rules  # noqa: E402
from aip.tasks import roster  # noqa: E402

#: Anchor and window come from the numbers module so the figure and the
#: appendix's resolution caveat cannot drift apart.
ANCHOR_F = EVASION_ANCHOR_F
RESOLUTION_BAND = EVASION_BAND

METHODS = [
    ("aip_gated", "AIP hard gate", PALETTE["vermillion"], "o"),
    ("aip_gated_soft6", "M1 soft gate", PALETTE["blue"], "s"),
    ("aip_gated_rand0.3", "M2 randomised threshold", PALETTE["purple"], "^"),
    ("aip_gated_soft6_rand0.3", "M3 soft + randomised", PALETTE["green"], "D"),
    ("majority", "majority vote", PALETTE["orange"], "x"),
]

MIN_PAIRS_FOR_POWER = 30


def honest_pairs(benchmark: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {path}; run the correlation phase first")
    frame = pd.read_parquet(path)
    sub = frame[frame.benchmark == benchmark]
    if sub.empty:
        raise SystemExit(f"{path.name} has no rows for {benchmark}")
    return sub


def draw(frame: pd.DataFrame, qpairs: Path, out: Path, benchmarks: list[str]) -> Path:
    apply_style()
    fig, axes = plt.subplots(
        1, len(benchmarks), figsize=(3.4 * len(benchmarks), 3.6),
        constrained_layout=True, sharey=True,
    )
    axes = [axes] if len(benchmarks) == 1 else list(axes)

    for ax, benchmark in zip(axes, benchmarks, strict=True):
        sub = frame[(frame.benchmark == benchmark) & (frame.f == ANCHOR_F)]
        if sub.empty:
            raise SystemExit(f"no f={ANCHOR_F} rows for {benchmark}")
        ceiling = float(sub.ceiling.iloc[0])
        n_opt = roster.n_options(benchmark)

        lo, hi = RESOLUTION_BAND
        grid = sorted(sub.q_realised.unique())
        in_band = [q for q in grid if lo <= q <= hi]
        ax.axvspan(lo, hi, color=PALETTE["yellow"], alpha=0.16, linewidth=0, zorder=0)

        # --- honest coherence, drawn as a rug so no density is invented -----
        pairs = honest_pairs(benchmark, qpairs)
        powered = pairs[pairs.n_both_wrong >= MIN_PAIRS_FOR_POWER]
        weak = pairs[pairs.n_both_wrong < MIN_PAIRS_FOR_POWER]
        for q in weak.q_true:
            ax.plot([q, q], [-0.055, -0.020], color=GREY, linewidth=0.7,
                    alpha=0.7, zorder=2, clip_on=False)
        for q in powered.q_true:
            ax.plot([q, q], [-0.055, -0.020], color=PALETTE["black"], linewidth=1.1,
                    zorder=3, clip_on=False)

        # --- the two lines the gate is defined against ----------------------
        ax.axvline(ceiling, color=PALETTE["black"], linestyle="--", linewidth=1.0,
                   zorder=1)
        ax.text(ceiling - 0.012, 0.74, f"honest ceiling {ceiling:.3f}",
                transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="top",
                fontsize=6.5)
        chance = 1.0 / (n_opt - 1) if n_opt and n_opt > 1 else None
        if chance is not None:
            ax.axvline(chance, color=PALETTE["sky"], linestyle=":", linewidth=1.2,
                       zorder=1)
            ax.text(chance + 0.012, 0.74, f"chance $1/(C-1)$ = {chance:.3f}",
                    transform=ax.get_xaxis_transform(), rotation=90, ha="left",
                    va="top", fontsize=6.5, color=PALETTE["sky"])

        # --- the measured grid, as points ----------------------------------
        for method, label, colour, marker in METHODS:
            s = sub[sub.method == method].sort_values("q_realised")
            if s.empty:
                continue
            ax.errorbar(
                s.q_realised, s.accuracy_point,
                yerr=[s.accuracy_point - s.accuracy_ci_low,
                      s.accuracy_ci_high - s.accuracy_point],
                fmt=marker, markersize=4.2, color=colour, ecolor=colour,
                elinewidth=0.7, capsize=1.6, linestyle="none", label=label, zorder=5,
            )

        # --- the point the paper's claim is about ---------------------------
        # The attacker's optimum is only an *evasion* if it sits below the
        # ceiling; above it the gate has fired and the adversary is winning by
        # coherence, not by hiding. Which of the two it is separates MedQA from
        # MATH-500, so both are reported rather than one being called "the"
        # optimum. Marked with a ring instead of an arrow: at this density an
        # arrow has to cross the data to reach a free patch of axes.
        hard = sub[sub.method == "aip_gated"]
        under = hard[hard.q_realised < ceiling].sort_values("accuracy_point")
        overall = hard.sort_values("accuracy_point").iloc[0]
        facts = [f"{len(in_band)} distinct $\\hat q$ in [{lo:g}, {hi:g}]",
                 f"{len(powered)}/{len(pairs)} honest pairs powered (R2)"]
        if not under.empty:
            best = under.iloc[0]
            ax.plot(best.q_realised, best.accuracy_point, marker="o", markersize=11,
                    markerfacecolor="none", markeredgecolor=PALETTE["vermillion"],
                    markeredgewidth=1.3, linestyle="none", zorder=6)
            facts.insert(0, f"best evasion (below ceiling): $\\hat q$ = "
                            f"{best.q_realised:.3f}, acc {best.accuracy_point:.3f}")
        if float(overall.q_realised) >= ceiling:
            facts.insert(1, f"but the global optimum is ABOVE the ceiling "
                            f"($\\hat q$ = {overall.q_realised:.3f}, "
                            f"acc {overall.accuracy_point:.3f}):\nthe gate is not "
                            f"what the adversary has to defeat here")

        ax.set_title(f"{benchmark}   ($C$ = {n_opt if n_opt else 'open'})", fontsize=9)
        ax.text(0.03, 0.97, "\n".join(facts), transform=ax.transAxes, fontsize=6.4,
                va="top", ha="left", color="#333333", linespacing=1.35,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none",
                      "boxstyle": "round,pad=0.35"}, zorder=7)
        ax.set_xlabel(r"realised adversarial coherence $\hat q$")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.06, 1.0)

    axes[0].set_ylabel("swarm accuracy")
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", markersize=4.2,
                      label=lab) for _, lab, c, m in METHODS]
    handles += [
        Line2D([], [], color=PALETTE["black"], linewidth=1.1,
               label=f"honest pair, $n_{{both wrong}} \\geq {MIN_PAIRS_FOR_POWER}$"),
        Line2D([], [], color=GREY, linewidth=0.7, label="honest pair, underpowered"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(
        f"Evasion band at $f = {ANCHOR_F}$ (exact tie): markers are the measured grid, "
        "not a curve", fontsize=9.5,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


GREY = "#8a8a8a"


def main() -> int:
    rules = load_rules()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path,
                    default=Path("results/adversarial/gate_aware_mitigations.parquet"))
    ap.add_argument("--qpairs", type=Path,
                    default=Path("results/correlation/q_pairwise.parquet"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures/evasion_band.pdf"))
    ap.add_argument("--benchmarks", nargs="*", default=list(rules.headline_benchmarks))
    args = ap.parse_args()

    if not args.sweep.exists():
        raise SystemExit(f"missing {args.sweep}; run scripts/phase_gate_aware.py first")
    frame = pd.read_parquet(args.sweep)
    if ANCHOR_F not in set(frame.f.unique()):
        raise SystemExit(f"{args.sweep.name} has no f={ANCHOR_F} rows")
    path = draw(frame, args.qpairs, args.out, args.benchmarks)
    stamp(path, script="scripts/fig_evasion_band.py",
          sources=[args.sweep, args.qpairs])
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
