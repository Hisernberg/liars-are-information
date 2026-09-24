"""Paper figures: style module, colorblind-safe palette, PDF output.

Every figure in the paper goes through :func:`apply_style` so the whole set
reads as one system. Nothing here relies on matplotlib's defaults -- the default
sans stack, the default `tab10` palette and the default `viridis`/`jet` maps are
all replaced.

Colour choices are constrained by accessibility, not taste:

* the categorical palette is Okabe-Ito, designed to stay distinguishable under
  deuteranopia, protanopia and tritanopia;
* the correlation map is a blue-white-orange divergence rather than red-green,
  since red-green is exactly the axis most colour-vision deficiencies collapse,
  and it is symmetric about zero so that "no correlation" reads as neutral.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

#: Okabe-Ito, colour-vision-deficiency safe.
PALETTE: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
CYCLE = [
    PALETTE["blue"],
    PALETTE["vermillion"],
    PALETTE["green"],
    PALETTE["orange"],
    PALETTE["purple"],
    PALETTE["sky"],
]

INK = "#1a1a1a"
GRID = "#c8c8c8"


def diverging_cmap() -> LinearSegmentedColormap:
    """Blue-white-orange divergence, symmetric about zero."""
    return LinearSegmentedColormap.from_list(
        "aip_diverging",
        ["#0B3C5D", "#3E7CA6", "#A8CBE0", "#F7F7F7", "#F3C48A", "#DE8A3E", "#8C4A0F"],
    )


def apply_style() -> None:
    """Install the project's matplotlib style. Call before creating any figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia", "serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.prop_cycle": plt.cycler(color=CYCLE),
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType so the PDF is editable/searchable
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "legend.frameon": False,
        }
    )


def _short(name: str) -> str:
    """Compact model labels so a 4x4 matrix stays readable."""
    return {
        "olmo2_7b": "OLMo-2 7B",
        "llama32_3b": "Llama-3.2 3B",
        "phi4_mini_reasoning": "Phi-4-mini-R",
        "ministral_8b": "Ministral 8B",
    }.get(name, name)


def correlation_heatmap(
    panels: list[tuple[str, list[str], np.ndarray]],
    out_path: Path,
    value_label: str = r"error correlation $\phi$",
    diagonal_note: str = "within-model",
) -> Path:
    """Render one NxN correlation matrix per benchmark panel.

    ``panels`` is ``(benchmark, model_names, matrix)``. NaN cells (an unmeasured
    within-model correlation, say) are drawn hatched rather than as a colour, so
    "not measured" can never be misread as "measured and near zero".
    """
    apply_style()
    cmap = diverging_cmap()
    cmap.set_bad("#ffffff")
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=1.0)

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.05 * n_panels + 0.9, 3.5), constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]

    mesh: Any = None
    for ax, (benchmark, names, matrix) in zip(axes, panels, strict=True):
        masked = np.ma.masked_invalid(matrix)
        mesh = ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal")

        # Hatch the unmeasured cells.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isfinite(matrix[i, j]):
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="///",
                            edgecolor="#b0b0b0",
                            linewidth=0.0,
                        )
                    )
                    continue
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}".replace("0.", "."),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if abs(value) > 0.62 else INK,
                )

        labels = [_short(n) for n in names]
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=38, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(benchmark, pad=6)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)

    cbar = fig.colorbar(mesh, ax=axes, fraction=0.021, pad=0.015, extend="both")
    cbar.set_label(value_label)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Pairwise error correlation  (diagonal: {diagonal_note})", y=1.04, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return out_path
