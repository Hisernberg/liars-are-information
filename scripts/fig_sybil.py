#!/usr/bin/env python3
"""Sybil bloc accuracy, coherent and disagreeing blocs on the same axes.

The two panels are the test of claim E2b -- that AIP's Sybil robustness comes
from coherence amplification. If that were the mechanism, the coherent panel
would separate from the disagreeing one. It does not, which is why they are
plotted side by side at a shared scale rather than as one averaged curve: an
average would hide the comparison the claim is about.
"""

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

SERIES = [
    ("aip_gated_w20", "AIP windowed", PALETTE["vermillion"], "o", "-"),
    ("aip_gated", "AIP pooled", PALETTE["orange"], "s", "-"),
    ("majority", "majority", PALETTE["blue"], "^", "--"),
    ("reputation_decay", "reputation decay", PALETTE["green"], "D", "--"),
    ("sac_filter_refine", "filter-and-refine", PALETTE["purple"], "v", "--"),
    ("confidence_weighted", "confidence weighted", PALETTE["sky"], "x", "--"),
]


def draw(frame: pd.DataFrame, out: Path) -> Path:
    apply_style()
    blocs = [("coherent", "coherent bloc"), ("disagreeing", "disagreeing bloc")]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), constrained_layout=True,
                             sharey=True)
    for ax, (bloc, title) in zip(axes, blocs, strict=True):
        sub = frame[frame.bloc == bloc]
        if sub.empty:
            raise SystemExit(f"no rows for bloc {bloc}")
        for method, label, colour, marker, style in SERIES:
            s = (sub[sub.method == method]
                 .groupby("sybil_size")["accuracy_point"].mean().sort_index())
            if s.empty:
                continue
            ax.plot(s.index, s.to_numpy(), color=colour, marker=marker, markersize=4,
                    linestyle=style, linewidth=1.4, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Sybil bloc size $s$")
        ax.set_xticks(sorted(sub.sybil_size.unique()))
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("swarm accuracy")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=Path("results/adversarial/sybil_e2.parquet"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures/sybil_accuracy.pdf"))
    args = ap.parse_args()
    if not args.data.exists():
        raise SystemExit(f"missing {args.data}; run scripts/phase_e_sleeper_sybil.py")
    path = draw(pd.read_parquet(args.data), args.out)
    stamp(path, script="scripts/fig_sybil.py", sources=[args.data])
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
