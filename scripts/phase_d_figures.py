#!/usr/bin/env python3
"""Phase D paper figures: money figure, invertibility spectrum, deterrence."""

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

METHODS = ["aip_gated", "sac_filter_refine", "geometric_median", "majority"]
PRETTY = {
    "aip_gated": "AIP (gated)",
    "sac_filter_refine": "SAC filter-refine",
    "geometric_median": "geometric median",
    "majority": "majority",
}
STYLE = {
    "aip_gated": (PALETTE["vermillion"], "-", 2.0, "o"),
    "sac_filter_refine": (PALETTE["blue"], "-", 1.3, "s"),
    "geometric_median": (PALETTE["green"], "-.", 1.3, "D"),
    "majority": (PALETTE["black"], (0, (3, 1, 1, 1)), 1.2, "v"),
}
ATTACK_LABEL = {
    "always_wrong": "always wrong",
    "semantic_negation": "semantic negation",
    "semantic_hallucination": "semantic hallucination",
    "rushing": "rushing",
    "falsified_confidence": "falsified confidence",
    "burst": "burst",
    "noise": "noise",
}


def money_figure(df: pd.DataFrame, out: Path, composition: str) -> Path:
    apply_style()
    attacks = [a for a in ATTACK_LABEL if a in set(df.attack)]
    ncol = 4
    nrow = int(np.ceil(len(attacks) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.0 * ncol, 2.75 * nrow),
        constrained_layout=True,
        sharey=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for i, attack in enumerate(attacks):
        ax = axes[i]
        sub = df[(df.attack == attack) & (df.composition == composition)]
        for method in METHODS:
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
                markersize=3.2,
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
        ax.set_title(ATTACK_LABEL[attack], pad=4)
        ax.set_ylim(-0.03, 1.0)
        ax.grid(axis="y", alpha=0.45)
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(
                loc="lower left", fontsize=6.3, handlelength=2.0, borderpad=0.3, labelspacing=0.25
            )
    for j in range(len(attacks), len(axes)):
        axes[j].set_visible(False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("Byzantine fraction  $f$")
    axes[0].set_ylabel("swarm accuracy")
    if nrow > 1:
        axes[ncol].set_ylabel("swarm accuracy")
    fig.suptitle(
        f"Swarm accuracy under real-LLM adversaries — {composition}, complete graph, $p_{{obs}}=1$",
        fontsize=10,
        y=1.03,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def invertibility_spectrum(
    df: pd.DataFrame, channels: pd.DataFrame, out: Path, composition: str
) -> Path:
    """Inversion gain vs measured channel coherence, one panel per benchmark.

    Faceted because the invertibility threshold is a property of the answer
    space: chance coincidence is 0 for open-ended answers and 1/3 for four-way
    multiple choice, and the calibrated honest ceiling differs accordingly. A
    single shared axis would put points from three different null models side by
    side and invite exactly the wrong comparison.
    """
    import yaml

    apply_style()
    cfg = yaml.safe_load(Path("configs/inversion_thresholds.yaml").read_text())
    classes = cfg["benchmark_classes"]

    rows = []
    for (benchmark, attack), g in df[df.composition == composition].groupby(
        ["benchmark", "attack"]
    ):
        high_f = g[g.f >= 0.4]
        aip = high_f[high_f.method == "aip_gated"]["accuracy_point"]
        disc = high_f[
            high_f.method.isin(
                [
                    "majority",
                    "geometric_median",
                    "sac_filter_refine",
                    "coord_median",
                    "trimmed_mean",
                    "krum",
                    "multi_krum",
                ]
            )
        ]
        if aip.empty or disc.empty:
            continue
        best_disc = disc.groupby("f")["accuracy_point"].max().mean()
        ch = channels[(channels.benchmark == benchmark) & (channels.attack == attack)]
        coherence = float(ch["coherence"].iloc[0]) if len(ch) else np.nan
        rows.append(
            {
                "benchmark": benchmark,
                "attack": attack,
                "gain": float(aip.mean()) - float(best_disc),
                "coherence": coherence,
            }
        )
    pts = pd.DataFrame(rows).dropna(subset=["coherence"])

    benchmarks = [b for b in roster.benchmarks() if b in set(pts.benchmark)]
    fig, axes = plt.subplots(
        1,
        len(benchmarks),
        figsize=(3.5 * len(benchmarks), 3.6),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    colours = {
        "semantic_negation": PALETTE["vermillion"],
        "always_wrong": PALETTE["orange"],
        "rushing": PALETTE["blue"],
        "semantic_hallucination": PALETTE["green"],
        "burst": PALETTE["purple"],
        "noise": PALETTE["sky"],
        "falsified_confidence": PALETTE["black"],
    }
    for ax, benchmark in zip(axes, benchmarks, strict=True):
        sub = pts[pts.benchmark == benchmark]
        cls = cfg["classes"][classes[benchmark]]
        chance, ceiling = float(cls["chance"]), float(cls["ceiling"])
        ax.axhline(0.0, color="#999999", linewidth=0.8, zorder=1)
        ax.axvspan(chance, ceiling, color="#d9d9d9", alpha=0.45, zorder=0, linewidth=0)
        ax.axvline(ceiling, color="#666666", linewidth=0.9, linestyle=(0, (4, 2)), zorder=2)
        ax.text(
            ceiling, 0.30, " honest-q\n ceiling", fontsize=6.2, color="#555555", va="top", ha="left"
        )
        for r in sub.itertuples():
            ax.scatter(
                r.coherence,
                r.gain,
                s=70,
                color=colours.get(r.attack, PALETTE["black"]),
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            ax.annotate(
                ATTACK_LABEL.get(r.attack, r.attack),
                (r.coherence, r.gain),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=6.0,
                color="#333333",
            )
        ax.set_title(benchmark, pad=5)
        ax.set_xlabel(r"adversary coherence  $\hat{q}$")
        ax.set_xlim(-0.04, 1.0)
        ax.grid(alpha=0.35)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("inversion gain\n(AIP-gated $-$ best discard, mean over $f\\geq0.4$)")
    fig.suptitle(
        "Invertibility spectrum: coherent channels can be inverted, "
        "idiosyncratic ones cannot\n"
        "(shaded band: coherence indistinguishable from an honest population)",
        fontsize=9.5,
        y=1.07,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def deterrence(log: pd.DataFrame, out: Path) -> Path:
    apply_style()
    defences = list(dict.fromkeys(log.defence))
    fig, axes = plt.subplots(
        2, len(defences), figsize=(4.6 * len(defences), 4.4), constrained_layout=True, sharex=True
    )
    axes = np.atleast_2d(axes)
    for j, defence in enumerate(defences):
        s = log[log.defence == defence].sort_values("epoch")
        top, bottom = axes[0, j], axes[1, j]
        coherent = (s.arm == "coherent_lie").astype(int)
        top.step(s.epoch, coherent, where="mid", color=PALETTE["vermillion"], linewidth=1.4)
        top.fill_between(
            s.epoch, 0, coherent, step="mid", color=PALETTE["vermillion"], alpha=0.18, linewidth=0
        )
        top.set_yticks([0, 1])
        top.set_yticklabels(["noise", "coherent lie"], fontsize=7.5)
        top.set_ylim(-0.15, 1.15)
        top.set_title(f"vs {defence}", pad=5)
        top.grid(axis="x", alpha=0.35)
        bottom.plot(
            s.epoch,
            s.value_coherent_lie,
            color=PALETTE["vermillion"],
            linewidth=1.4,
            label="value(coherent lie)",
        )
        bottom.plot(
            s.epoch,
            s.value_noise,
            color=PALETTE["blue"],
            linewidth=1.4,
            linestyle="--",
            label="value(noise)",
        )
        bottom.set_xlabel("epoch")
        bottom.grid(alpha=0.35)
        bottom.set_axisbelow(True)
        if j == 0:
            top.set_ylabel("arm chosen")
            bottom.set_ylabel("estimated damage\n(1 $-$ swarm accuracy)")
            bottom.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Deterrence: what an adaptive adversary learns against each defence", fontsize=10, y=1.04
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep", type=Path, default=Path("results/adversarial/adversarial_sweep.parquet")
    )
    p.add_argument(
        "--channels", type=Path, default=Path("results/adversarial/channel_stats.parquet")
    )
    p.add_argument("--bandit", type=Path, default=Path("results/adversarial/bandit_log.parquet"))
    p.add_argument("--composition", default="mix_all4")
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--figures", type=Path, default=Path("paper/figures"))
    args = p.parse_args()

    df = pd.read_parquet(args.sweep)
    ref = df[(df.p_obs == 1.0) & (df.parity == "matched@0.1") & (df.benchmark == args.benchmark)]
    out = money_figure(ref, args.figures / "money_figure.pdf", args.composition)
    stamp(out, script="scripts/phase_d_figures.py", sources=[args.sweep])
    print("wrote", out)
    if args.channels.exists():
        ch = pd.read_parquet(args.channels)
        full = df[(df.p_obs == 1.0) & (df.parity == "matched@0.1")]
        out = invertibility_spectrum(
            full, ch, args.figures / "invertibility_spectrum.pdf", args.composition
        )
        stamp(out, script="scripts/phase_d_figures.py",
              sources=[args.sweep, args.channels])
        print("wrote", out)
    if args.bandit.exists():
        out = deterrence(pd.read_parquet(args.bandit),
                         args.figures / "deterrence_timeseries.pdf")
        stamp(out, script="scripts/phase_d_figures.py", sources=[args.bandit])
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
