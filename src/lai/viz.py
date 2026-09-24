"""Shared figure style: one color and one marker per method, fixed everywhere.

Colours are the validated reference categorical palette (adjacent CVD dE >= 9.1,
normal-vision dE >= 22.9 on the light surface). Two slots sit below 3:1
contrast, so every line also carries a distinct marker and a legend, and the
important series are direct-labelled.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
# Diverging (blue <-> red, gray midpoint) and sequential blue ramps.
DIVERGING = ["#b2312f", "#e34948", "#f29c92", "#f0efec", "#86b6ef", "#2a78d6", "#1c5cab"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

METHOD_STYLE = {
    "race": dict(color=BLUE, marker="o", label="RACE (ours)", lw=2.2, zorder=6),
    "aip_gated": dict(color=ORANGE, marker="s", label="AIP gated", lw=1.6, zorder=5),
    "aip_soft": dict(color=ORANGE, marker="D", label="AIP soft gate (M1)", lw=1.2, zorder=4, alpha=0.7),
    "majority": dict(color=AQUA, marker="^", label="Majority vote", lw=1.6, zorder=4),
    "ds_onecoin": dict(color=YELLOW, marker="v", label="Dawid–Skene (label-free)", lw=1.6, zorder=4),
    "ds_full": dict(color=YELLOW, marker="P", label="Dawid–Skene full", lw=1.2, zorder=3),
    "self": dict(color=MUTED, marker="", label="Receiver alone", lw=1.4, zorder=2),
    "oracle_channel": dict(color=VIOLET, marker="*", label="Known-channel oracle", lw=1.2, zorder=3),
    "oracle_honest_majority": dict(color=VIOLET, marker="x", label="Honest-only oracle", lw=1.0, zorder=3),
    "race_noclone": dict(color=BLUE, marker="o", label="RACE w/o clone tempering", lw=1.0, zorder=3, alpha=0.6),
    "race_capself": dict(color=BLUE, marker="h", label="RACE + self cap", lw=1.0, zorder=3, alpha=0.6),
}

LABEL = {k: v["label"] for k, v in METHOD_STYLE.items()} | {
    "aip_trust_only": "AIP trust-only", "aip_naive": "AIP naive", "sac": "SAC filter-refine",
    "confidence": "Confidence-weighted", "race_full": "RACE full-confusion",
    "race_rawclone": "RACE raw-agreement clones", "race_ms": "RACE multi-start",
}

BENCH_LABEL = {"mmlu": "MMLU", "medqa": "MedQA", "arc": "ARC", "boolq": "BoolQ (binary)",
               "gsm8k": "GSM8K (open)", "math500": "MATH-500 (open)"}


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 8, "lines.markersize": 4.5,
        "font.family": "DejaVu Sans",
    })


def line(ax, x, y, method: str, **overrides):
    style = dict(METHOD_STYLE.get(method, dict(color=INK_2, marker=".", label=method, lw=1.0)))
    style.update(overrides)
    label = style.pop("label")
    return ax.plot(x, y, label=label, markeredgecolor=SURFACE, markeredgewidth=0.8, **style)[0]


def save(fig, path_stem) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
