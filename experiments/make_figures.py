#!/usr/bin/env python3
"""Paper figures and tables from the saved study outputs (no recomputation).

Writes ``figures/*.png|pdf`` and ``results/tables/*.md|csv``. Every number in
the manuscript is taken from these tables.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from lai import viz  # noqa: E402
from lai.stats import compare, summary  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "figures"
TAB = RES / "tables"
BENCHES = ["mmlu", "medqa", "arc", "boolq", "gsm8k", "math500"]


def load(study: str) -> pd.DataFrame | None:
    path = RES / study / "per_task.parquet"
    return pd.read_parquet(path) if path.exists() else None


def pct(x):
    return 100 * x


def write_table(frame: pd.DataFrame, name: str, floatfmt: str = ".1f") -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TAB / f"{name}.csv")
    (TAB / f"{name}.md").write_text(frame.to_markdown(floatfmt=floatfmt) + "\n")
    try:
        digits = int(floatfmt.strip(".f") or 0)
        latex = frame.to_latex(float_format=lambda v: f"{v:.{digits}f}", escape=True, na_rep="--")
        (TAB / f"{name}.tex").write_text(latex)
    except Exception as exc:  # LaTeX export is a convenience, never fatal
        print("latex export failed", name, exc)


# ------------------------------------------------------------------ F1


def fig_main(main: pd.DataFrame) -> None:
    sub = main[main.p_obs == 1.0]
    s = summary(sub, ["benchmark", "f"])
    methods = ["self", "majority", "ds_onecoin", "aip_gated", "oracle_channel", "race"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True, sharey=True)
    for ax, b in zip(axes.flat, BENCHES, strict=True):
        tbl = s.loc[b]
        for m in methods:
            if m in tbl:
                viz.line(ax, tbl.index, pct(tbl[m]), m)
        ax.set_title(viz.BENCH_LABEL[b], loc="left")
        ax.set_ylim(-3, 103)
        ax.axvline(0.5, color=viz.GRID, lw=1.0, zorder=0)
        end = tbl.index.max()
        ax.annotate("RACE", (end, pct(tbl.loc[end, "race"])), xytext=(4, 0), textcoords="offset points",
                    color=viz.INK, fontsize=7.5, va="center", fontweight="bold")
    for ax in axes[1]:
        ax.set_xlabel("Byzantine fraction f")
    for ax in axes[:, 0]:
        ax.set_ylabel("Honest-receiver accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Coherent liars: plurality and label-free rules break at f = 1/2, AIP at 0.9 (BoolQ: 0.5); RACE holds", x=0.01,
                 ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    viz.save(fig, FIG / "fig1_main_sweep")
    cols = ["self", "majority", "confidence", "sac", "ds_full", "ds_onecoin", "aip_trust_only", "aip_naive",
            "aip_gated", "aip_soft", "race_noclone", "race_rawclone", "race_ms", "race_full", "race",
            "race_capself", "oracle_honest_majority", "oracle_channel"]
    for p in (1.0, 0.5):
        t = pct(summary(main[main.p_obs == p], ["benchmark", "f"]))
        write_table(t[[c for c in cols if c in t]].rename(columns=viz.LABEL), f"main_accuracy_pobs{p:g}")


# ------------------------------------------------------------------ F2


def fig_zoo(zoo: pd.DataFrame) -> None:
    zoo = zoo.assign(atk=zoo.attack + np.where(zoo.attack.isin(["gate_aware", "partial", "camouflage"]),
                                                "(" + zoo.param.map(lambda v: f"{v:g}") + ")", ""))
    s = summary(zoo, ["f", "atk"])
    methods = ["majority", "ds_onecoin", "aip_gated", "aip_soft", "race", "race_capself", "oracle_channel"]
    order = ["independent", "gate_aware(0)", "gate_aware(0.25)", "gate_aware(0.5)", "gate_aware(0.75)",
             "gate_aware(1)", "attractor", "partial(0.5)", "partial(0.8)", "uninformative", "echo",
             "camouflage(0.7)", "camouflage(0.9)", "sleeper"]
    cmap = LinearSegmentedColormap.from_list("div", viz.DIVERGING)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2), sharey=True)
    for ax, f in zip(axes, (0.3, 0.5, 0.7), strict=True):
        tbl = s.loc[f].reindex(order)
        delta = pct(tbl[methods].sub(tbl["self"], axis=0))
        im = ax.imshow(delta.values, cmap=cmap, norm=TwoSlopeNorm(0, -60, 20), aspect="auto")
        for (i, j), v in np.ndenumerate(delta.values):
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 35 else viz.INK)
        ax.set_xticks(range(len(methods)), [viz.LABEL[m].replace(" (ours)", "\n(ours)") for m in methods],
                      rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(order)), order, fontsize=7.5)
        ax.set_title(f"f = {f:g}", loc="left")
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cbar.set_label("Accuracy minus receiver alone (pp)")
    fig.suptitle("Attack zoo (6 benchmarks): blue = pooling helps, red = pooling hurts", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    viz.save(fig, FIG / "fig2_attack_zoo")
    write_table(pct(s[["self", *methods]]).rename(columns=viz.LABEL), "zoo_accuracy")


# ------------------------------------------------------------------ F3


def fig_gate_aware(zoo: pd.DataFrame) -> None:
    g = zoo[zoo.attack == "gate_aware"]
    s = summary(g, ["f", "param"])
    methods = ["self", "majority", "ds_onecoin", "aip_gated", "aip_soft", "race"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, f in zip(axes, (0.3, 0.5, 0.7), strict=True):
        tbl = s.loc[f]
        for m in methods:
            viz.line(ax, tbl.index, pct(tbl[m]), m)
        ax.set_title(f"f = {f:g}", loc="left")
        ax.set_xlabel("Coordination p (shared-lie probability)")
        top = ax.secondary_xaxis("top", functions=(lambda p: p ** 2 + (1 - p ** 2) / 3,
                                                 lambda q: np.sqrt(np.clip((3 * np.asarray(q) - 1) / 2, 0, None))))
        top.set_xticks([1 / 3, 0.5, 0.75, 1.0], ["1/3", ".5", ".75", "1"])
        top.set_xlabel("Byzantine pair coherence q(p), C=4", fontsize=7.5, color=viz.INK_2)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("The gate-aware adversary: AIP is defeated below its ceiling; RACE's weight never depended on coherence",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    viz.save(fig, FIG / "fig3_gate_aware")


# ------------------------------------------------------------------ F4


def fig_llm(llm: pd.DataFrame) -> None:
    llm = llm.assign(prompt=llm.attack.str.replace("llm:", "", regex=False))
    s = summary(llm, ["f", "prompt"])
    methods = ["self", "majority", "ds_onecoin", "aip_gated", "race"]
    prompts = ["always_wrong", "rushing", "semantic_hallucination", "semantic_negation"]
    fs = [f for f in (0.3, 0.5, 0.7) if f in s.index.get_level_values(0)]
    fig, axes = plt.subplots(1, len(fs), figsize=(11, 3.4), sharey=True)
    width = 0.16
    for ax, f in zip(np.atleast_1d(axes), fs, strict=True):
        tbl = s.loc[f].reindex(prompts)
        for k, m in enumerate(methods):
            st = viz.METHOD_STYLE[m]
            ax.bar(np.arange(len(prompts)) + (k - 2) * (width + 0.01), pct(tbl[m]), width, color=st["color"],
                   label=st["label"], edgecolor=viz.SURFACE, linewidth=1.0)
        ax.set_xticks(range(len(prompts)), [p.replace("_", "\n") for p in prompts], fontsize=7.5)
        ax.set_title(f"f = {f:g}", loc="left")
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Accuracy (%)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Replayed real-LLM deceivers (4 prompts × 2 attacker models × 6 benchmarks)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    viz.save(fig, FIG / "fig4_llm_liars")
    write_table(pct(summary(llm, ["f", "prompt", "benchmark"])[methods + ["oracle_channel"]]).rename(columns=viz.LABEL),
                "llm_accuracy")


# ------------------------------------------------------------------ F5 / F10 (extra)


def fig_channels() -> None:
    path = RES / "extra" / "channel_estimates.parquet"
    if not path.exists():
        return
    ch = pd.read_parquet(path)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    for byz, color, label in ((False, viz.INK_2, "Honest peer"), (True, viz.MAGENTA, "Byzantine peer")):
        d = ch[ch.byzantine == byz].sample(min(4000, (ch.byzantine == byz).sum()), random_state=0)
        ax.scatter(d.a_true, d.a_hat, s=6, color=color, alpha=0.35, label=label, linewidths=0)
    ax.plot([0, 1], [0, 1], color=viz.INK_2, lw=0.8)
    ax.set_xlabel("True channel accuracy on history (evaluator only)")
    ax.set_ylabel("RACE estimate (label-free)")
    corr = np.corrcoef(ch.a_true, ch.a_hat)[0, 1]
    mae = np.mean(np.abs(ch.a_true - ch.a_hat))
    ax.set_title(f"Channel estimation: r = {corr:.3f}, MAE = {mae:.3f}", loc="left")
    ax.legend(loc="upper left", markerscale=3)
    ax = axes[1]
    rows = []
    for method in ("aip_decision", "race_decision"):
        for byz in (False, True):
            d = ch[ch.byzantine == byz][method].value_counts(normalize=True)
            rows.append(dict(method="AIP gated" if method.startswith("aip") else "RACE",
                             peer="Byzantine" if byz else "Honest", **{k: d.get(k, 0.0) for k in ("trust", "discard", "invert")}))
    t = pd.DataFrame(rows)
    ylabels = [f"{r.method} · {r.peer}" for r in t.itertuples()]
    left = np.zeros(len(t))
    for key, color in (("trust", viz.BLUE), ("discard", "#c9c8c3"), ("invert", viz.RED)):
        ax.barh(ylabels, pct(t[key]), left=left, color=color, label=key.upper(), edgecolor=viz.SURFACE, linewidth=2)
        for i, v in enumerate(t[key]):
            if v > 0.07:
                ax.text(left[i] + pct(v) / 2, i, f"{pct(v):.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if key != "discard" else viz.INK)
        left += pct(t[key]).to_numpy()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of channel decisions (%)")
    ax.set_title("Who gets trusted, discarded, inverted", loc="left")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    viz.save(fig, FIG / "fig5_channel_estimation")
    summary_rows = ch.groupby(["attack", "byzantine"]).apply(
        lambda d: pd.Series(dict(mae=np.mean(np.abs(d.a_true - d.a_hat)), race_invert=np.mean(d.race_decision == "invert"),
                                 aip_invert=np.mean(d.aip_decision == "invert"), race_trust=np.mean(d.race_decision == "trust"),
                                 aip_trust=np.mean(d.aip_decision == "trust"), n=len(d))))
    write_table(summary_rows, "channel_decisions", ".3f")

    path = RES / "extra" / "receiver_gain.parquet"
    if not path.exists():
        return
    rg = pd.read_parquet(path)
    rg = rg[rg.f > 0]
    bins = np.array([0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
    rg["bin"] = pd.cut(rg.receiver_hist_acc, bins)
    g = rg.groupby("bin", observed=True)
    x = np.array([iv.mid for iv in g.groups])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    viz.line(ax, pct(x), pct(g.self_acc.mean()), "self", marker="o")
    for col, m in (("majority_acc", "majority"), ("aip_acc", "aip_gated"), ("race_acc", "race")):
        viz.line(ax, pct(x), pct(g[col].mean()), m)
    ax.set_xlabel("Receiver's own accuracy on history (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Mean accuracy by receiver competence", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    ax = axes[1]
    for col, m in (("majority_acc", "majority"), ("aip_acc", "aip_gated"), ("race_acc", "race")):
        viz.line(ax, pct(x), pct(g.apply(lambda d, c=col: np.mean(d[c] < d.self_acc - 0.05))), m)
    ax.axvspan(20, 50, color=viz.GRID, alpha=0.6, zorder=0)
    ax.text(22, 44, "weak anchor\n(≤ 50%)", fontsize=7.5, color=viz.INK_2)
    ax.set_xlabel("Receiver's own accuracy on history (%)")
    ax.set_ylabel("Receivers > 5 pp worse than\ntrusting themselves (%)")
    ax.set_title("Risk of pooling at all", loc="left")
    ax.set_ylim(0, 50)
    fig.suptitle("Anchor strength: RACE is almost never worse than self-trust once the receiver is above chance",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    viz.save(fig, FIG / "fig10_receiver_competence")
    tab = rg.groupby("bin", observed=True).apply(lambda d: pd.Series(dict(
        n=len(d), self=d.self_acc.mean(), race=d.race_acc.mean(), aip=d.aip_acc.mean(), majority=d.majority_acc.mean(),
        race_worse_than_self=np.mean(d.race_acc < d.self_acc - 0.05), aip_worse_than_self=np.mean(d.aip_acc < d.self_acc - 0.05),
        majority_worse_than_self=np.mean(d.majority_acc < d.self_acc - 0.05))))
    write_table(tab, "receiver_competence", ".3f")


# ------------------------------------------------------------------ F6


def fig_history(hist: pd.DataFrame) -> None:
    s = summary(hist.assign(atk=hist.attack + "(" + hist.param.map(lambda v: f"{v:g}") + ")"), ["atk", "f", "history_n"])
    methods = ["self", "majority", "ds_onecoin", "aip_gated", "race"]
    atks = list(dict.fromkeys(s.index.get_level_values(0)))
    fig, axes = plt.subplots(len(atks), 3, figsize=(10, 3.0 * len(atks)), sharex=True, sharey=True, squeeze=False)
    for i, a in enumerate(atks):
        for j, f in enumerate((0.3, 0.5, 0.7)):
            ax = axes[i, j]
            tbl = s.loc[(a, f)]
            for m in methods:
                viz.line(ax, tbl.index, pct(tbl[m]), m)
            ax.set_xscale("log", base=2)
            ax.set_xticks([5, 10, 20, 40, 80], ["5", "10", "20", "40", "80"])
            ax.minorticks_off()
            ax.set_title(f"{a}, f = {f:g}", loc="left")
            ax.set_ylim(-3, 103)
    for ax in axes[-1]:
        ax.set_xlabel("History tasks per receiver")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("How much history does inversion need? (MMLU, MedQA, BoolQ, MATH-500)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    viz.save(fig, FIG / "fig6_history_length")
    write_table(pct(s[methods]).rename(columns=viz.LABEL), "history_accuracy")


# ------------------------------------------------------------------ F7


def fig_swarm(sw: pd.DataFrame) -> None:
    comp = sw[(sw.n_agents == 10)]
    s = summary(comp, ["composition", "f"])
    methods = ["self", "majority", "aip_gated", "race_noclone", "race"]
    comps = ["frozen", "frozen+weak", "weak", "hom_qwen38_27b", "hom_llama32_3b"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    width = 0.15
    tbl = s.xs(0.5, level="f").reindex(comps)
    for k, m in enumerate(methods):
        st = viz.METHOD_STYLE[m]
        ax.bar(np.arange(len(comps)) + (k - 2) * (width + 0.01), pct(tbl[m]), width, color=st["color"],
               alpha=st.get("alpha", 1.0), label=st["label"], edgecolor=viz.SURFACE, linewidth=1.0)
    ax.set_xticks(range(len(comps)), ["7 frozen", "7 + weak arm", "weak roster", "10× Qwen-27B", "10× Llama-3B"], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy at f = 0.5 (%)")
    ax.set_title("Composition and replication (coherent liars)", loc="left")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    size = sw[sw.composition == "frozen"]
    s2 = summary(size, ["attack", "n_agents"])
    for a, ls in (("coherent", "-"), ("gate_aware", "--")):
        if a not in s2.index.get_level_values(0):
            continue
        tbl = s2.loc[a]
        for m in ("aip_gated", "race", "majority"):
            viz.line(ax, tbl.index, pct(tbl[m]), m, linestyle=ls, label=f"{viz.LABEL[m]} · {a.replace('_', '-')}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20, 40], ["5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_xlabel("Swarm size N")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Swarm size (f ∈ {.3,.5,.7} pooled)", loc="left")
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    viz.save(fig, FIG / "fig7_swarm")
    write_table(pct(summary(sw, ["composition", "n_agents", "attack", "f"])[methods + ["oracle_channel"]]).rename(columns=viz.LABEL),
                "swarm_accuracy")


# ------------------------------------------------------------------ F8


def fig_online() -> None:
    path = RES / "online" / "per_step.parquet"
    if not path.exists():
        return
    on = pd.read_parquet(path)
    on = on.groupby(["benchmark", "f", "schedule", "seed", "pos", "method"], as_index=False).correct.mean()
    show = {"race_fixed": ("race", dict(label="RACE, warm-up fit only", alpha=0.5, marker="")),
            "race_cumulative": ("race", dict(label="RACE, cumulative", marker="")),
            "race_decay0.97": ("race", dict(label="RACE, forgetting γ=0.97", color=viz.VIOLET, marker="")),
            "aip_gated_window64": ("aip_gated", dict(label="AIP gated, 64-task window", marker="")),
            "majority": ("majority", dict(marker="")),
            "self": ("self", dict(marker=""))}
    scheds = ["sleeper", "coherent_to_independent", "independent_to_coherent", "toggle40"]
    fig, axes = plt.subplots(1, len(scheds), figsize=(13, 3.3), sharey=True)
    for ax, sc in zip(axes, scheds, strict=True):
        d = on[(on.schedule == sc) & (on.f == 0.5)]
        piv = d.groupby(["pos", "method"]).correct.mean().unstack()
        for m, (style, kw) in show.items():
            if m in piv:
                viz.line(ax, piv.index, pct(piv[m].rolling(16, min_periods=4).mean()), style, **kw)
        titles = {"sleeper": "Sleeper: honest until task 100", "coherent_to_independent": "Coherent → independent at 120",
                  "independent_to_coherent": "Independent → coherent at 120",
                  "toggle40": "Coherent ↔ silent (shaded), every 40"}
        ax.set_title(titles[sc], loc="left")
        ax.set_xlabel("Task position (chronological)")
        switches = [100] if sc == "sleeper" else [80, 120, 160] if sc == "toggle40" else [120]
        for x in switches:
            ax.axvline(x, color=viz.INK_2, lw=0.8, ls="--", zorder=0)
        if sc == "toggle40":  # shaded: the bloc is silent (uninformative)
            ax.axvspan(40, 80, color=viz.GRID, alpha=0.6, lw=0, zorder=0)
            ax.axvspan(120, 160, color=viz.GRID, alpha=0.6, lw=0, zorder=0)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("Accuracy, 16-task rolling mean (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Non-stationary attackers (f = 0.5, predict-then-commit)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    viz.save(fig, FIG / "fig8_online")
    post = on[on.pos >= 100].groupby(["schedule", "f", "method"]).correct.mean().unstack()
    write_table(pct(post), "online_accuracy_after_switch")


# ------------------------------------------------------------------ F11 concept


def fig_concept() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax = axes[0]
    a = np.linspace(0.005, 0.995, 400)
    for k, color in ((2, viz.GREEN), (4, viz.BLUE)):
        ax.plot(a, np.log((k - 1) * a / (1 - a)), color=color, lw=2, label=f"RACE weight λ(a), K = {k}")
    ax.step([0, 0.2, 0.2, 0.33, 0.33, 1], [-2.5, -2.5, 0, 0, 2.5, 2.5], where="post", color=viz.ORANGE, lw=1.6,
            label="AIP: INVERT / DISCARD / TRUST (schematic)")
    ax.axhline(0, color=viz.INK_2, lw=0.8)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("Channel accuracy a  =  P(report = truth)")
    ax.set_ylabel("Log-odds contributed by one report")
    ax.set_title("A signed, continuous weight", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    p = np.linspace(0, 1, 200)
    ax.plot(p, p ** 2 + (1 - p ** 2) / 3, color=viz.ORANGE, lw=2, label="coherence q(p) — what AIP measures")
    ax.plot(p, np.zeros_like(p), color=viz.BLUE, lw=2, label="accuracy a(p) — what RACE measures")
    ax.axhline(0.677, color=viz.ORANGE, lw=0.8, alpha=0.6)
    ax.text(0.02, 0.69, "AIP's published MC ceiling 0.677", fontsize=7, color=viz.INK_2)
    ax.axhline(1 / 3, color=viz.MUTED, lw=0.8)
    ax.text(0.98, 0.3, "chance coherence 1/(K−1)", fontsize=7, color=viz.INK_2, ha="right", va="top")
    ax.set_xlabel("Gate-aware attacker's coordination p")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Coherence is tunable; truth-dependence is not", loc="left")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout()
    viz.save(fig, FIG / "fig11_concept")


# ------------------------------------------------------------------ live


def fig_live() -> None:
    """E8: what the live agents did in each role, and what pooling made of it."""
    path = RES / "live" / "per_task.parquet"
    if not path.exists():
        return
    live = pd.read_parquet(path)
    s = summary(live, ["source", "benchmark", "attack", "f"])
    write_table(pct(s), "live_accuracy")
    meta_path = RES / "live" / "live_summary.json"
    if meta_path.exists():
        (TAB / "live_summary.json").write_text(meta_path.read_text())
    roles = pd.read_csv(RES / "live" / "role_stats.csv")
    from lai.data import LIVE_MODELS
    names = {"qwen25_1p5b": "Qwen2.5-1.5B", "smollm2_1p7b": "SmolLM2-1.7B", "granite33_2b": "Granite-3.3-2B",
             "olmo2_1b": "OLMo-2-1B", "llama32_1b": "Llama-3.2-1B", "gemma3_1b": "Gemma-3-1B"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    role_style = {"honest": dict(color=viz.INK, marker="o", label="honest"),
                  "debate": dict(color=viz.BLUE, marker="D", label="honest after debate"),
                  "solo": dict(color=viz.RED, marker="v", label="saboteur (solo)"),
                  "rushing": dict(color=viz.ORANGE, marker="^", label="saboteur (sees honest votes)")}
    for ax, b in zip(axes[0], ("mmlu", "boolq"), strict=True):
        g = roles[roles.benchmark == b]
        ys = np.arange(len(LIVE_MODELS))[::-1]
        for y, m in zip(ys, LIVE_MODELS, strict=True):
            vals = g[g.model == m].set_index("role").accuracy
            ax.plot([pct(vals.min()), pct(vals.max())], [y, y], color=viz.GRID, lw=3, zorder=1, solid_capstyle="round")
            for role, st in role_style.items():
                if role in vals:
                    ax.scatter(pct(vals[role]), y, s=42, zorder=3, edgecolor=viz.SURFACE, lw=0.6,
                               **{k: v for k, v in st.items() if k != "label"})
        chance = 50 if b == "boolq" else 25
        ax.axvline(chance, color=viz.INK_2, lw=0.8, ls="--")
        ax.text(chance + 1, -0.45, "chance", fontsize=7.5, color=viz.INK_2, va="center")
        ax.set_ylim(-0.7, len(LIVE_MODELS) - 0.4)
        ax.set_yticks(ys, [names[m] for m in LIVE_MODELS])
        ax.set_xlim(0, 100)
        ax.set_xlabel("accuracy of the agent's answers (%)")
        ax.set_title(f"{viz.BENCH_LABEL[b]}: what each live agent did", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0][0].legend(handles=[plt.Line2D([], [], ls="", markersize=6, **st) for st in role_style.values()],
                      fontsize=7.5, loc="lower right")
    ind = live[(live.source == "live") & (live.split == "test")]
    ind = ind.assign(fam=np.where(ind.attack == "coherent", "coherent", "llm"))
    for ax, b in zip(axes[1], ("mmlu", "boolq"), strict=True):
        sub = ind[(ind.benchmark == b) & (ind.fam == "llm")]
        t = sub.groupby(["f", "method"]).accuracy.mean().unstack()
        for m in ("self", "majority", "ds_onecoin", "aip_gated", "race"):
            viz.line(ax, t.index, pct(t[m]), m)
        viz.line(ax, t.index, pct(t["race_onecoin"]), "race_onecoin")
        ax.set_xlabel("Byzantine fraction f (LLM saboteurs: solo, rushing, colluding)")
        ax.set_ylim(*((40, 90) if b == "boolq" else (20, 70)))
        ax.set_title(f"{viz.BENCH_LABEL[b]}: honest-agent accuracy after pooling", loc="left")
    axes[1][0].set_ylabel("accuracy (%)")
    axes[1][0].legend(fontsize=7.5, loc="lower left", ncol=2)
    fig.suptitle("E8: a live six-model swarm (fresh CPU inference). RACE v3.0 was frozen before this run; "
                 "v3.1's binary rule was adopted after it",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    viz.save(fig, FIG / "fig13_live_swarm")


# ------------------------------------------------------------------ best response


def best_response(zoo: pd.DataFrame) -> None:
    """Per defender: the attack chosen on VALIDATION to minimise it, scored on TEST.

    Two attack families: stationary (all but the sleeper) and all attacks."""
    zoo = zoo.assign(atk=zoo.attack + "@" + zoo.param.map(lambda v: f"{v:g}"))
    methods = ["self", "majority", "ds_onecoin", "aip_gated", "aip_soft", "race", "race_capself", "oracle_channel"]
    task = zoo[zoo.method.isin(methods)].groupby(["benchmark", "f", "seed", "method", "atk", "split"]).accuracy.mean().unstack("split")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), sharey=True)
    for ax, family in zip(axes, ("stationary", "all"), strict=True):
        sub = task if family == "all" else task[~task.index.get_level_values("atk").str.startswith("sleeper")]
        rows = []
        for (b, f, seed, m), g in sub.groupby(level=["benchmark", "f", "seed", "method"]):
            g = g.droplevel(["benchmark", "f", "seed", "method"])
            worst = g.validation.idxmin()
            rows.append(dict(benchmark=b, f=f, seed=seed, method=m, attack=worst, test=g.test[worst]))
        br = pd.DataFrame(rows)
        write_table(pct(br.groupby(["f", "method"]).test.mean().unstack()[methods]).rename(columns=viz.LABEL),
                    f"best_response_accuracy_{family}")
        write_table(br.groupby(["method", "attack"]).size().unstack(fill_value=0), f"best_response_choice_{family}", ".0f")
        t = br.groupby(["f", "method"]).test.mean().unstack()
        for m in ["self", "majority", "ds_onecoin", "aip_gated", "aip_soft", "race"]:
            viz.line(ax, t.index, pct(t[m]), m)
        ax.set_xlabel("Byzantine fraction f")
        ax.set_ylim(-3, 103)
        ax.set_title("stationary attacks (13 settings)" if family == "stationary" else "all attacks incl. sleeper (14)", loc="left")
    axes[0].set_ylabel("Test accuracy under the validation-\nselected worst attack (%)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Adaptive attacker: best response per defender, benchmark, f and seed", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    viz.save(fig, FIG / "fig9_best_response")


# ------------------------------------------------------------------ F12 information budget


EXT_ROWS = [("independent", 1.0, "Independent wrong answers"), ("gate_aware", 0.0, "Gate-aware, p = 0"),
            ("gate_aware", 0.5, "Gate-aware, p = 0.5"), ("gate_aware", 1.0, "Gate-aware, p = 1 (coherent)"),
            ("attractor", 1.0, "Attractor"), ("partial", 0.8, "Partial liar, e = 0.8"),
            ("partial", 0.5, "Partial liar, e = 0.5"), ("uninformative", 1.0, "Uninformative (silent)"),
            ("llm:always_wrong", 1.0, "LLM: always wrong"), ("llm:rushing", 1.0, "LLM: rushing"),
            ("llm:semantic_hallucination", 1.0, "LLM: hallucination"), ("llm:semantic_negation", 1.0, "LLM: negation"),
            ("echo", 1.0, "Echo (copies an honest agent)"), ("camouflage", 0.9, "Camouflage, θ = 0.9"),
            ("camouflage", 0.7, "Camouflage, θ = 0.7")]


def fig_budget(ext: pd.DataFrame) -> None:
    """How much information the liars carry, and how much of it RACE extracts.

    Zero is an oracle that knows who the liars are, removes them and decodes the
    honest agents with their true accuracies. The known-channel oracle keeps the
    liars; its lead over zero is the information the liars carry (Theorem 1).
    The sleeper is left out: a history-fitted oracle is invalid by construction."""
    test = ext[ext.split == "test"]
    cell = test.groupby(["attack", "param", "f", "benchmark", "seed", "method"]).accuracy.mean().unstack("method")
    m = pct(cell.groupby(["attack", "param", "f"]).mean())
    rel = m.sub(m.oracle_channel_honest, axis=0)
    table = rel[["oracle_channel", "race", "race_d", "aip_gated", "majority", "self"]].round(1)
    table.insert(0, "oracle_channel_honest", m.oracle_channel_honest.round(1))
    write_table(table, "information_budget")
    lines = [r"\begin{tabular}{@{}l" + "rr" * 3 + "@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$f={f:g}$}}" for f in (0.3, 0.5, 0.7)) + r"\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             "Attack & " + " & ".join([r"$B$ & $\Delta$"] * 3) + r"\\", r"\midrule"]
    for atk, prm, lab in [*EXT_ROWS, ("sleeper", 1.0, "Sleeper (non-stationary)")]:
        cells = []
        for f in (0.3, 0.5, 0.7):
            r = rel.loc[(atk, prm, f)]
            cells += [f"${r.oracle_channel:+.1f}$", f"${r.race:+.1f}$"]
        tex_label = re.sub(r"(p|e|θ) = ([0-9.]+)", lambda mt: f"${mt.group(1)}={mt.group(2)}$", lab).replace("θ", r"\theta")
        lines.append(tex_label + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "budget_compact.tex").write_text("\n".join(lines) + "\n")
    fs = [0.3, 0.5, 0.7]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 5.6), sharey=True)
    ys = np.arange(len(EXT_ROWS))[::-1]
    lim = (-16, 9)
    for ax, f in zip(axes, fs, strict=True):
        ax.axvline(0, color=viz.INK_2, lw=1)
        ax.axvspan(0, lim[1], color=viz.SEQ_BLUE[0], alpha=0.35, lw=0)
        for y, (atk, prm, _) in zip(ys, EXT_ROWS, strict=True):
            r = rel.loc[(atk, prm, f)]
            ax.plot([min(0, r.race), max(0, r.race)], [y, y], color=viz.GRID, lw=3, zorder=1, solid_capstyle="round")
            ax.scatter(np.clip(r.oracle_channel, *lim), y, marker="*", s=70, color=viz.VIOLET, zorder=3,
                       edgecolor=viz.SURFACE, lw=0.6)
            ax.scatter(np.clip(r.race_d, *lim), y, marker="o", s=34, facecolor="none", edgecolor=viz.BLUE, lw=1.1, zorder=4)
            ax.scatter(np.clip(r.race, *lim), y, marker="o", s=34, color=viz.BLUE, zorder=5, edgecolor=viz.SURFACE, lw=0.6)
            if r.race < lim[0]:
                ax.annotate(f"{r.race:.0f}", (lim[0], y), xytext=(3, 0), textcoords="offset points", fontsize=7,
                            va="center", color=viz.INK_2)
        ax.set_xlim(*lim)
        ax.set_title(f"f = {f:g}  ({round(10 * f)} of 10 lie)", loc="left")
        ax.set_xlabel("minus liar-removal oracle (pp)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [lab for *_, lab in EXT_ROWS])
    axes[0].set_ylim(-0.7, len(EXT_ROWS) - 0.3)
    axes[0].text(lim[1] - 0.3, 0, "better than\nremoving\nevery liar", ha="right", va="center",
                 fontsize=7.5, color=viz.BLUE)
    handles = [plt.Line2D([], [], marker="o", ls="", color=viz.BLUE, label="RACE (label-free)"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", color=viz.BLUE, label="RACE-D (extension)"),
               plt.Line2D([], [], marker="*", ls="", color=viz.VIOLET, markersize=9,
                          label="Known-channel oracle: information the liars carry")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("What the liars are worth: accuracy relative to an oracle that removes every liar",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    viz.save(fig, FIG / "fig12_information_budget")
    # RACE-D vs frozen RACE: paired per cell, Holm over all cells
    c = compare(ext, "race_d", ["race"], ["benchmark", "f", "attack", "param"])
    c.to_csv(TAB / "race_d_vs_race.csv", index=False)
    write_table(c.groupby(["attack", "param"]).verdict.value_counts().unstack(fill_value=0), "race_d_vs_race_wtl", ".0f")
    c = compare(ext, "race", ["oracle_channel_honest"], ["benchmark", "f", "attack", "param"])
    c.to_csv(TAB / "race_vs_removal_oracle.csv", index=False)
    write_table(c.groupby(["attack", "param"]).verdict.value_counts().unstack(fill_value=0), "race_vs_removal_oracle_wtl", ".0f")


CROWD_ROWS = ("oracle_channel", "race", "aip_gated", "iwmv", "mace", "glad", "kos", "ds_onecoin", "ds_full",
              "majority", "self")
CLASSICAL = ("iwmv", "mace", "glad", "ds_onecoin")


def fig_crowd(crowd: pd.DataFrame) -> None:
    """E11: classical crowdsourcing estimators have no anchor, so a liar majority flips them."""
    test = crowd[crowd.split == "test"]
    cell = test.groupby(["benchmark", "attack", "param", "f", "seed", "method"]).accuracy.mean().unstack("method")
    by_f = pct(cell.groupby("f").mean())
    by_bf = pct(cell.groupby(["benchmark", "f"]).mean())
    table = by_f[[m for m in CROWD_ROWS if m != "kos"]].T.rename(index=viz.LABEL)
    table.columns = [f"f={f:g}" for f in table.columns]
    write_table(table.round(1), "crowd_baselines_by_f")
    per_bench = by_bf.xs(0.7, level="f")[list(CROWD_ROWS)].T
    write_table(per_bench.rename(index=viz.LABEL).round(1), "crowd_baselines_f07")
    c = pd.read_csv(RES / "tables" / "race_vs_crowd.csv") if (RES / "tables" / "race_vs_crowd.csv").exists() else None
    if c is not None:
        wtl = c.groupby(["baseline", "f"]).verdict.value_counts().unstack(fill_value=0)
        write_table(wtl.rename(index=viz.LABEL, level=0), "race_vs_crowd_wtl", ".0f")

    fig = plt.figure(figsize=(11.2, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    fs = by_f.index.to_numpy()
    lo, hi = by_f[list(CLASSICAL)].min(axis=1), by_f[list(CLASSICAL)].max(axis=1)
    ax.fill_between(fs, lo, hi, color=viz.YELLOW, alpha=0.22, lw=0, zorder=1)
    ax.plot(fs, hi, color=viz.YELLOW, lw=1.6, ls="--", marker="v", markeredgecolor=viz.SURFACE, zorder=3,
            label="Best classical crowd estimator\n(band: IWMV, MACE, GLAD, Dawid–Skene)")
    for m in ("oracle_channel", "race", "aip_gated", "majority", "self"):
        viz.line(ax, fs, by_f[m], m)
    ax.axvline(0.5, color=viz.GRID, lw=1.0, zorder=0)
    ax.text(0.505, 8, "liars become\nthe majority", fontsize=7.5, color=viz.INK_2, va="bottom")
    for m, dy in (("race", -7), ("oracle_channel", 5), ("self", -9)):
        ax.annotate(f"{by_f[m].iloc[-1]:.1f}", (fs[-1], by_f[m].iloc[-1]), xytext=(5, dy), textcoords="offset points",
                    fontsize=7.5, color=viz.INK_2, va="center")
    ax.annotate(f"{hi.iloc[-1]:.1f}", (fs[-1], hi.iloc[-1]), xytext=(5, 0), textcoords="offset points", fontsize=7.5,
                color=viz.INK_2, va="center")
    ax.set_xticks(fs, [f"{f:g}" for f in fs])
    ax.set_xlim(fs[0] - 0.03, fs[-1] + 0.09)
    ax.set_ylim(0, 102)
    ax.set_xlabel("f: fraction of the 10 agents that lie")
    ax.set_ylabel("honest-agent accuracy (%)")
    ax.set_title("(a) Accuracy as the liar share grows", loc="left")
    ax.legend(loc="lower left", fontsize=7.2)

    ax = fig.add_subplot(gs[1])
    cols = [b for b in BENCHES if b in per_bench.columns]
    grid = per_bench[cols].copy()
    # a mean over fewer benchmarks is not comparable, so only methods defined everywhere get one
    grid["mean"] = by_f.loc[0.7, list(CROWD_ROWS)].where(grid[cols].notna().all(axis=1))
    vals = grid.to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("seq", viz.SEQ_BLUE)
    shown = np.where(np.isfinite(vals), vals, np.nan)
    ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 62 else viz.INK, fontweight="bold" if grid.index[i] == "race" else None)
            else:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=viz.GRID, lw=0))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color=viz.MUTED)
    ax.set_xticks(range(len(grid.columns)), [viz.BENCH_LABEL.get(b, b).split(" (")[0].replace("-", "-\n") for b in cols]
                  + ["Mean"])
    ax.set_yticks(range(len(grid.index)), [viz.LABEL.get(m, m) for m in grid.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(1.5, color=viz.SURFACE, lw=3)
    ax.axvline(len(cols) - 0.5, color=viz.SURFACE, lw=3)
    ax.set_title("(b) Per benchmark at f = 0.7 (7 of 10 agents lie)", loc="left")
    fig.suptitle("E11. Classical crowdsourcing estimators collapse when liars are the majority; the anchor does not",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.02, "Means over six benchmarks, six attack settings (coherent, gate-aware p = 0.25, four LLM prompts) "
             "and three seeds, TEST split. KOS is defined for binary questions only; full-confusion Dawid–Skene is not "
             "run on open-answer benchmarks (n/a).", fontsize=7, color=viz.INK_2)
    viz.save(fig, FIG / "fig14_crowd_baselines")


# (label, headline prefix pattern, f tag, oracle kind); the order is the paper's
OVERVIEW_ROWS = (
    ("E1  Coherent liar bloc", dict(race="mainRaceSeven", self="mainSelfSeven", oracle="mainOracleSeven",
                                    aip="mainAipSeven", maj="mainMajSeven", ds="mainDsSeven")),
    ("E2  Gate-aware adversary (built against AIP)", dict(race="zooGateRaceSeven", self="zooGateSelfSeven",
                                                          oracle="zooGateOracleSeven", aip="zooGateAipSeven",
                                                          maj="zooGateMajSeven", ds="zooGateDsSeven")),
    ("E2  Camouflage (built against RACE)", dict(race="zooCamoRaceSeven", self="zooCamoSelfSeven",
                                                 oracle="zooCamoOracleSeven", aip="zooCamoAipSeven",
                                                 maj="zooCamoMajSeven", ds="zooCamoDsSeven")),
    ("E3  LLMs prompted to deceive", dict(race="llmRaceSeven", self="llmSelfSeven", oracle="llmOracleSeven",
                                          aip="llmAipSeven", maj="llmMajSeven", ds="llmDsSeven")),
    ("E9  Independent liars", dict(race="extRaceIndepSeven", self="extSelfIndepSeven",
                                   removal="extRemovalIndepSeven", aip="extAipIndepSeven", maj="extMajIndepSeven")),
    ("E11 vs classical crowdsourcing", dict(race="crowdRaceSeven", self="crowdSelfSeven", oracle="crowdOracleSeven",
                                            aip="crowdAipSeven", maj="crowdMajSeven", ds="crowdBestClassicalSeven")),
    ("E8  Live swarm, MMLU (f = 0.5)", dict(race="liveMmluLlmRaceFive", self="liveMmluLlmSelfFive",
                                            oracle="liveMmluLlmOracleFive", aip="liveMmluLlmAipFive",
                                            maj="liveMmluLlmMajFive", ds="liveMmluLlmDsFive")),
    ("E8  Live swarm, BoolQ (f = 0.5)", dict(race="liveBoolqLlmRaceFive", self="liveBoolqLlmSelfFive",
                                             oracle="liveBoolqLlmOracleFive", aip="liveBoolqLlmAipFive",
                                             maj="liveBoolqLlmMajFive", ds="liveBoolqLlmDsFive")),
    ("E10 Fresh live run, ARC (f = 0.5)", dict(race="liveTwoArcLlmRaceFive", self="liveTwoArcLlmSelfFive",
                                               oracle="liveTwoArcLlmOracleFive", aip="liveTwoArcLlmAipFive",
                                               maj="liveTwoArcLlmMajFive", ds="liveTwoArcLlmDsFive")),
    ("E10 Fresh live run, BoolQ (f = 0.5)", dict(race="liveTwoBoolqLlmRaceFive", self="liveTwoBoolqLlmSelfFive",
                                                 oracle="liveTwoBoolqLlmOracleFive", aip="liveTwoBoolqLlmAipFive",
                                                 maj="liveTwoBoolqLlmMajFive", ds="liveTwoBoolqLlmDsFive")),
)


def fig_overview() -> None:
    """One row per study: RACE against every deployable baseline, the receiver alone and the oracle."""
    path = RES / "tables" / "headline.json"
    if not path.exists():
        return
    h = json.loads(path.read_text())

    def val(key):
        v = h.get(key.replace("Two", "Two"))
        try:
            return float(str(v).replace(",", ""))
        except (TypeError, ValueError):
            return np.nan

    rows = [(lab, {k: val(v) for k, v in keys.items()}) for lab, keys in OVERVIEW_ROWS]
    rows = [(lab, r) for lab, r in rows if np.isfinite(r["race"])]
    table = pd.DataFrame({lab: r for lab, r in rows}).T
    table = table.rename(columns={"race": "RACE", "self": "Receiver alone", "oracle": "Known-channel oracle",
                                  "removal": "Liar-removal oracle", "aip": "AIP gated", "maj": "Majority vote",
                                  "ds": "Dawid–Skene / best classical"})
    base_cols = ["AIP gated", "Majority vote", "Dawid–Skene / best classical"]
    table["RACE − receiver"] = table["RACE"] - table["Receiver alone"]
    table["RACE − best baseline"] = table["RACE"] - table[base_cols].max(axis=1)
    write_table(table.round(1), "overview")

    fig, ax = plt.subplots(figsize=(10.5, 0.52 * len(rows) + 1.6))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, r) in zip(ys, rows, strict=True):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=viz.GRID, alpha=0.35, lw=0, zorder=0)
        pts = [r[k] for k in ("race", "self", "aip", "maj", "ds", "oracle", "removal") if np.isfinite(r.get(k, np.nan))]
        ax.plot([min(pts), max(pts)], [y, y], color=viz.GRID, lw=1.2, zorder=1)
        ax.plot([r["self"]] * 2, [y - 0.28, y + 0.28], color=viz.MUTED, lw=2.4, zorder=2, solid_capstyle="butt")
        for k, style in (("aip", "aip_gated"), ("maj", "majority"), ("ds", "ds_onecoin")):
            if np.isfinite(r.get(k, np.nan)):
                st = viz.METHOD_STYLE[style]
                ax.scatter(r[k], y, marker=st["marker"], s=46, color=st["color"], zorder=3, edgecolor=viz.SURFACE,
                           lw=0.7)
        if np.isfinite(r.get("oracle", np.nan)):
            ax.scatter(r["oracle"], y, marker="*", s=95, color=viz.VIOLET, zorder=4, edgecolor=viz.SURFACE, lw=0.6)
        if np.isfinite(r.get("removal", np.nan)):
            ax.scatter(r["removal"], y, marker="*", s=95, facecolor="none", edgecolor=viz.VIOLET, lw=1.1, zorder=4)
        ax.scatter(r["race"], y, marker="o", s=80, color=viz.BLUE, zorder=6, edgecolor=viz.SURFACE, lw=0.9)
        d_self = r["race"] - r["self"]
        d_base = r["race"] - max(r.get(k, -np.inf) for k in ("aip", "maj", "ds") if np.isfinite(r.get(k, np.nan)))
        ax.text(101.5, y, f"{d_self:+.1f}", va="center", ha="left", fontsize=8,
                color=viz.INK if d_self >= 0 else viz.RED, fontweight="bold")
        ax.text(111.5, y, f"{d_base:+.1f}", va="center", ha="left", fontsize=8,
                color=viz.INK if d_base >= 0 else viz.RED, fontweight="bold")
    ax.text(101.5, len(rows) - 0.35, "vs receiver\nalone", fontsize=7, color=viz.INK_2, va="bottom")
    ax.text(111.5, len(rows) - 0.35, "vs best\nbaseline", fontsize=7, color=viz.INK_2, va="bottom")
    ax.set_yticks(ys, [lab for lab, _ in rows])
    ax.set_xlim(-2, 121)
    ax.set_xticks(range(0, 101, 20))
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("honest-agent accuracy (%), f = 0.7 unless stated (7 of 10 agents lie)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=viz.BLUE, markersize=8, label="RACE (ours)"),
               plt.Line2D([], [], marker="s", ls="", color=viz.ORANGE, label="AIP gated (prior work)"),
               plt.Line2D([], [], marker="^", ls="", color=viz.AQUA, label="Majority vote"),
               plt.Line2D([], [], marker="v", ls="", color=viz.YELLOW, label="Dawid–Skene / best classical"),
               plt.Line2D([], [], marker="|", ls="", color=viz.MUTED, markersize=10, markeredgewidth=2.4,
                          label="Receiver alone"),
               plt.Line2D([], [], marker="*", ls="", color=viz.VIOLET, markersize=10, label="Known-channel oracle"),
               plt.Line2D([], [], marker="*", ls="", markerfacecolor="none", color=viz.VIOLET, markersize=10,
                          label="Liar-removal oracle")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.08 - 0.45 / len(rows)), ncol=4, fontsize=7.5)
    ax.set_title("Every study at a glance: RACE against every deployable baseline", loc="left", fontsize=11)
    fig.tight_layout()
    viz.save(fig, FIG / "fig0_overview")


def fig_live2() -> None:
    """E10 (pre-registered): v3.1 on fresh answers (H1, H2) and RACE-informed debate (H3)."""
    base = RES / "live2"
    if not (base / "hypotheses.csv").exists():
        return
    hyp = pd.read_csv(base / "hypotheses.csv")
    t = hyp[["hypothesis", "statement", "target_acc", "baseline_acc", "delta", "wins", "ties", "losses", "verdict"]].copy()
    t.columns = ["Hypothesis", "Statement", "Target (%)", "Baseline (%)", "Δ (points)", "Wins", "Ties", "Losses", "Verdict"]
    write_table(t.set_index("Hypothesis"), "e10_hypotheses")
    lines = [r"\begin{tabular}{@{}lp{5.2cm}rrrcl@{}}", r"\toprule",
             r"& Pre-registered ordering & Target & Baseline & $\Delta$ & W/T/L & Verdict\\", r"\midrule"]
    for _, r in hyp.iterrows():
        stmt = (r.statement.replace(">=", r"$\ge$").replace(">", "$>$").replace("f$>$=0.5", r"$f\ge0.5$")
                .replace("f>=0.5", r"$f\ge0.5$"))
        lines.append(f"{r.hypothesis} & {stmt} & {r.target_acc:.1f} & {r.baseline_acc:.1f} & ${r.delta:+.1f}$ & "
                     f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)} & \\textit{{{r.verdict}}}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "e10_hypotheses_compact.tex").write_text("\n".join(lines) + "\n")
    d = pd.read_parquet(base / "per_task.parquet")
    test = d[(d.split == "test") & (d.source == "live2") & d.attack.str.startswith("llm:")]
    m = pct(test.groupby(["benchmark", "f", "method"]).accuracy.mean().unstack("method"))
    write_table(m[["self", "majority", "aip_gated", "ds_onecoin", "iwmv", "mace", "glad", "race_onecoin", "race",
                   "oracle_channel"]].rename(columns=viz.LABEL).round(1), "e10_accuracy")
    pm = pd.read_csv(base / "h3_per_model.csv")
    write_table(pct(pm.set_index(["benchmark", "model"])).round(1), "e10_informed_per_model")
    h3 = pd.read_csv(base / "h3_individual.csv").set_index(["scope", "subset"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for ax, b, keys, title in ((axes[0, 0], "boolq", ("race", "race_onecoin", "majority", "self", "oracle_channel"),
                                "(a) H1 · BoolQ (unseen items 120–199), LLM saboteurs"),
                               (axes[0, 1], "arc", ("race", "majority", "aip_gated", "self", "oracle_channel"),
                                "(b) H2 · ARC (new benchmark), LLM saboteurs")):
        mb = m.loc[b]
        for k in keys:
            viz.line(ax, mb.index, mb[k], k)
        ax.set_xticks(mb.index, [f"{f:g}" for f in mb.index])
        ax.set_xlabel("f: fraction of the 10 agents that lie")
        ax.set_ylabel("honest-agent accuracy (%)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(fontsize=7.2, loc="lower left")
    pretty = {"qwen25_1p5b": "Qwen2.5-1.5B", "smollm2_1p7b": "SmolLM2-1.7B", "granite33_2b": "Granite-3.3-2B",
              "olmo2_1b": "OLMo-2-1B", "llama32_1b": "Llama-3.2-1B", "gemma3_1b": "Gemma-3-1B"}
    dfr = pd.read_csv(base / "deference.csv") if (base / "deference.csv").exists() else None
    for ax, b, title in ((axes[1, 0], "arc", "(c) H3 · ARC"), (axes[1, 1], "boolq", "(d) H3 · BoolQ")):
        g = pm[pm.benchmark == b].sort_values("honest")
        ys = np.arange(len(g))
        for y, (_, r) in zip(ys, g.iterrows(), strict=True):
            ax.plot([pct(r.debate), pct(r.informed)], [y, y], color=viz.GRID, lw=2.5, zorder=1)
        if dfr is not None:
            fav = dfr[dfr.benchmark == b].groupby("model").favoured_right.mean()
            ax.scatter(pct(fav.reindex(g.model)), ys, marker="D", s=34, facecolor="none", edgecolor=viz.VIOLET, lw=1.3,
                       zorder=2, label="RACE's suggested option (in the informed prompt)")
        for col, color, marker, lab in (("honest", viz.MUTED, "|", "round 1 (independent)"),
                                        ("debate", viz.ORANGE, "s", "after plain debate"),
                                        ("informed", viz.BLUE, "o", "after RACE-informed debate")):
            ax.scatter(pct(g[col]), ys, color=color, marker=marker, s=60 if marker != "|" else 140, zorder=3,
                       lw=2 if marker == "|" else 0.6, edgecolor=viz.SURFACE if marker != "|" else None, label=lab)
        ax.set_yticks(ys, [pretty.get(mm, mm) for mm in g.model])
        ax.set_xlabel("individual accuracy of the honest model (%)")
        r = h3.loc[(b, "all")]
        ax.set_title(f"{title}: informed − plain = {pct(r.delta):+.1f} points "
                     f"[{pct(r.ci_low):+.1f}, {pct(r.ci_high):+.1f}]", loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("E10 (pre-registered). Fresh live answers: RACE v3.1, and debate with RACE's reliability notes",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    viz.save(fig, FIG / "fig15_live_confirmation")


def per_agent_table() -> None:
    """How much each honest agent gains by pooling, per model (E7 receiver-gain worlds, f >= 0.5)."""
    p = RES / "extra" / "receiver_gain.parquet"
    if not p.exists():
        return
    rg = pd.read_parquet(p)
    rg = rg[rg.f >= 0.5]
    rows = []
    for family, sub in (("all seven attacks", rg), ("LLM deceivers", rg[rg.attack.str.startswith("llm:")]),
                        ("camouflage", rg[rg.attack == "camouflage"])):
        t = pct(sub.groupby("model")[["self_acc", "majority_acc", "aip_acc", "race_acc"]].mean())
        t["gain"] = t.race_acc - t.self_acc
        t["family"] = family
        rows.append(t.reset_index())
    out = pd.concat(rows).set_index(["family", "model"]).rename(columns={
        "self_acc": "Alone", "majority_acc": "Majority vote", "aip_acc": "AIP", "race_acc": "RACE", "gain": "RACE gain"})
    write_table(out.round(1), "per_agent_gain")


# ------------------------------------------------------------------ comparisons


def comparisons(main, zoo, llm) -> None:
    rows = []
    baselines = ["self", "majority", "ds_onecoin", "aip_gated", "aip_soft"]
    if main is not None:
        c = compare(main[main.p_obs == 1.0], "race", baselines, ["benchmark", "f"])
        c["study"] = "main"
        rows.append(c)
    if zoo is not None:
        c = compare(zoo, "race", baselines, ["benchmark", "f", "attack", "param"])
        c["study"] = "zoo"
        rows.append(c)
    if llm is not None:
        c = compare(llm, "race", baselines, ["benchmark", "f", "attack"])
        c["study"] = "llm"
        rows.append(c)
    if not rows:
        return
    allc = pd.concat(rows, ignore_index=True)
    TAB.mkdir(parents=True, exist_ok=True)
    allc.to_csv(TAB / "paired_comparisons.csv", index=False)
    wtl = allc.groupby(["study", "baseline"]).verdict.value_counts().unstack(fill_value=0)
    for col in ("win", "tie", "loss"):
        if col not in wtl:
            wtl[col] = 0
    names = {"main": "E1 coherent sweep", "zoo": "E2 attack zoo", "llm": "E3 LLM deceivers", "live": "E8 live swarm"}
    wide = wtl.reset_index()
    wide["Study"] = wide.study.map(names)
    wide["Baseline"] = wide.baseline.map(lambda b: viz.LABEL.get(b, b))
    wide = wide.rename(columns={"win": "Wins", "tie": "Ties", "loss": "Losses"})
    wide["Cells"] = wide[["Wins", "Ties", "Losses"]].sum(axis=1)
    write_table(wide.set_index(["Study", "Baseline"])[["Wins", "Ties", "Losses", "Cells"]], "win_tie_loss", ".0f")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--late", action="store_true",
                    help="only the figures that read generated numbers (run after make_numbers.py)")
    args = ap.parse_args()
    viz.setup()
    FIG.mkdir(exist_ok=True)
    if args.late:
        fig_overview()
        crowd = load("crowd")
        if crowd is not None:
            fig_crowd(crowd)
        return
    main_df, zoo, llm, hist, sw, ext, crowd = (load(s) for s in ("main", "zoo", "llm", "history", "swarm", "ext",
                                                                 "crowd"))
    fig_concept()
    if main_df is not None:
        fig_main(main_df)
    if zoo is not None:
        fig_zoo(zoo)
        fig_gate_aware(zoo)
        best_response(zoo)
    if llm is not None:
        fig_llm(llm)
    if hist is not None:
        fig_history(hist)
    if sw is not None:
        fig_swarm(sw)
    if ext is not None:
        fig_budget(ext)
    if crowd is not None:
        fig_crowd(crowd)
    fig_channels()
    per_agent_table()
    fig_online()
    fig_live()
    fig_live2()
    comparisons(main_df, zoo, llm)
    fig_overview()
    print(json.dumps(sorted(p.name for p in FIG.glob("*.png")), indent=1))


if __name__ == "__main__":
    main()
