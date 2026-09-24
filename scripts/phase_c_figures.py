#!/usr/bin/env python3
"""Phase C figures: the money figure draft and the gating ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.analysis.figures import PALETTE, apply_style  # noqa: E402
from aip.analysis.provenance import stamp  # noqa: E402
from aip.harness.rules import load_rules  # noqa: E402
from aip.tasks import roster  # noqa: E402

BENCHMARKS = roster.benchmarks()  # from configs/task_lists.json
PRETTY = {
    "aip_gated": "AIP (gated)",
    "aip_naive": "AIP (naive)",
    "aip_trust_only": "AIP (trust only)",
    "sac_filter_refine": "SAC filter-refine",
    "geometric_median": "geometric median",
    "majority": "majority",
    "dawid_skene": "Dawid–Skene",
}
STYLE = {
    "aip_gated": (PALETTE["vermillion"], "-", 2.0, "o"),
    "aip_naive": (PALETTE["purple"], "--", 1.4, "^"),
    "aip_trust_only": (PALETTE["orange"], ":", 1.4, "s"),
    "sac_filter_refine": (PALETTE["blue"], "-", 1.3, "s"),
    "geometric_median": (PALETTE["green"], "-.", 1.3, "D"),
    "majority": (PALETTE["black"], (0, (3, 1, 1, 1)), 1.2, "v"),
    "dawid_skene": (PALETTE["sky"], "-", 1.2, "P"),
}


def panel(ax, sub: pd.DataFrame, methods: list[str], title: str, show_legend: bool) -> None:
    for method in methods:
        s = sub[sub.method == method].sort_values("f")
        if s.empty:
            continue
        colour, dash, width, marker = STYLE[method]
        ax.plot(
            s.f,
            s.accuracy_point,
            color=colour,
            linestyle=dash,
            linewidth=width,
            marker=marker,
            markersize=3.4,
            label=PRETTY[method],
            zorder=3,
        )
        ax.fill_between(
            s.f,
            s.accuracy_ci_low,
            s.accuracy_ci_high,
            color=colour,
            alpha=0.13,
            linewidth=0,
            zorder=2,
        )
    ax.axvline(0.5, color="#999999", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    ax.text(
        0.505, 0.02, "f = 0.5", fontsize=6.5, color="#777777", rotation=90, va="bottom", ha="left"
    )
    ax.set_title(title, pad=5)
    ax.set_xlabel("Byzantine fraction  $f$")
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlim(-0.02, 0.72)
    ax.grid(axis="y", alpha=0.45)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(loc="lower left", fontsize=6.6, handlelength=2.0, borderpad=0.3, labelspacing=0.3)


def money_figure(df: pd.DataFrame, out: Path, composition: str) -> Path:
    apply_style()
    methods = ["aip_gated", "sac_filter_refine", "geometric_median", "majority"]
    # Panels come from the headline set, sized from it. This was
    # `subplots(1, 3)` against a six-benchmark BENCHMARKS list -- the tenth
    # place in this project where three benchmarks were assumed -- and it
    # aborted the script before the gating-ablation figure was written.
    benchmarks = list(load_rules().headline_benchmarks)
    fig, axes = plt.subplots(
        1, len(benchmarks), figsize=(3.2 * len(benchmarks), 3.1),
        constrained_layout=True, sharey=True,
    )
    axes = [axes] if len(benchmarks) == 1 else list(axes)
    for i, (ax, benchmark) in enumerate(zip(axes, benchmarks, strict=True)):
        sub = df[(df.benchmark == benchmark) & (df.composition == composition)]
        panel(ax, sub, methods, benchmark, show_legend=(i == 0))
        if i == 0:
            ax.set_ylabel("swarm accuracy")
    fig.suptitle(
        f"Accuracy vs Byzantine fraction — {composition}, complete graph, "
        "$p_{obs}=1$, coherent adversary",
        fontsize=9.5,
        y=1.06,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def gating_ablation(df: pd.DataFrame, out: Path, composition: str) -> Path:
    apply_style()
    methods = ["aip_gated", "aip_naive", "aip_trust_only"]
    order = ["mmlu", "gsm8k", "math500"]  # MMLU first: the shared-attractor case
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), constrained_layout=True, sharey=True)
    for i, (ax, benchmark) in enumerate(zip(axes, order, strict=True)):
        sub = df[(df.benchmark == benchmark) & (df.composition == composition)]
        label = f"{benchmark}  (honest q above chance)" if benchmark == "mmlu" else benchmark
        panel(ax, sub, methods, label, show_legend=(i == 0))
        if i == 0:
            ax.set_ylabel("swarm accuracy")
    fig.suptitle(
        "Why the honest-q threshold exists: gated vs naive inversion vs no inversion",
        fontsize=9.5,
        y=1.06,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("results/aggregation/sweep.parquet"))
    parser.add_argument("--composition", default="mix_all4")
    parser.add_argument("--figures", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    df = pd.read_parquet(args.sweep)
    ref = df[
        (df.p_obs == 1.0)
        & (df.topology == "complete")
        & (df.adversary == "always_wrong")
        & (df.parity == "matched@0.1")
    ]
    a = money_figure(ref, args.figures / "money_figure_draft.pdf", args.composition)
    b = gating_ablation(ref, args.figures / "gating_ablation.pdf", args.composition)
    for out in (a, b):
        stamp(out, script="scripts/phase_c_figures.py", sources=[args.sweep])
    print(f"wrote {a}\nwrote {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
