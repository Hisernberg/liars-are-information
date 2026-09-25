#!/usr/bin/env python3
"""Label switching, animated: why classical estimators fail when liars are the majority.

One honest receiver in a real replayed world (MMLU, coherent bloc, 7 of 10 agents
lie). The same EM machinery is run twice on the receiver's unlabelled history:

* **Dawid-Skene** (no anchor): starts from the majority vote, the textbook
  initialisation, and converges to the labelling in which the majority is right;
* **RACE**: identical updates, but anchored on the receiver's own answers
  (initialisation, prior and constraint) and with the bloc tempered as clones.

Both end at fixed points of the *same* likelihood, which cannot tell "seven agents
are right" from "three agents are right" (Proposition 2). Bars are each method's
estimated accuracy per agent after every EM iteration. Black ticks are the true
accuracies and the agent roles, shown to the viewer only; neither method sees a label.

    PYTHONPATH=src python experiments/make_switching.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import imageio_ffmpeg  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from aip.aggregation.baselines import MajorityVote  # noqa: E402
from lai import viz  # noqa: E402
from lai.data import score  # noqa: E402
from lai.race import RACEAggregator, _rows_from_observations, tempering_weights  # noqa: E402
from lai.sim import World, build_world  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
MEDIA = ROOT / "media"
WORLD = World("mmlu", 0.7, "coherent", 1.0, 1.0, 0)
N_ITER = 10
SUB = 8  # interpolated frames per EM iteration


def trajectories() -> dict:
    bw = build_world(WORLD)
    labels = bw.data.label_space
    r = bw.honest[0]
    hist = [bw.defense[t] for t in bw.splits["history"]]
    rows = [row for _, row in _rows_from_observations([(o[r],) for o in hist])[r]]
    gold_hist = [bw.data.gold[t] for t in bw.splits["history"]]

    race = RACEAggregator(labels)
    race.fit([(bw.defense[t][r],) for t in bw.splits["history"]])
    fit = race.fits[r]
    agents = fit.agents
    own = agents.index(r)
    reports, n_cand = race._encode(rows, agents)
    tw = np.ones(len(rows))
    ds = RACEAggregator(labels, model="onecoin", anchored=False, clone_aware=False)
    ds.fit([(bw.defense[t][r],) for t in bw.splits["history"]])

    def run(agg, groups, init):
        temper = tempering_weights(reports, groups)
        acc, dec = [], []
        for k in range(1, N_ITER + 1):
            agg.max_iter = k
            f = agg._em_onecoin(reports, n_cand, temper, tw, own, agents, groups, init)
            acc.append(f.accuracy.copy())
            # decoded truth on the history vs gold (viewer only)
            cand = [agg._candidates(row) for row in rows]
            guess = [c[int(np.argmax(f.posterior[t, :len(c)]))] if c else None for t, c in enumerate(cand)]
            dec.append(np.mean([score(WORLD.benchmark, g, y) for g, y in zip(guess, gold_hist, strict=True)]))
        return np.array(acc), np.array(dec)

    # the encoding orders candidates as RACE does; check that decoding agrees with it
    race_acc, race_dec = run(race, fit.groups, "self")
    ds_acc, ds_dec = run(ds, np.arange(len(agents)), "plurality")
    true_acc = np.array([np.mean([score(WORLD.benchmark, row.get(a), y) for row, y in zip(rows, gold_hist, strict=True)])
                         for a in agents])
    chance = float(np.mean([1 / max(k, 1) for k in n_cand if k > 0]))
    mean_k = float(np.mean(n_cand[n_cand > 0]))

    # test accuracy for the outro (the fitted models, full convergence)
    race.max_iter = 200
    test = {"race": [], "ds": [], "majority": [], "self": []}
    for t in bw.splits["test"]:
        o, g = bw.defense[t][r], bw.data.gold[t]
        test["race"].append(score(WORLD.benchmark, race.aggregate(o.broadcasts, r), g))
        test["ds"].append(score(WORLD.benchmark, ds.aggregate(o.broadcasts, r), g))
        test["majority"].append(score(WORLD.benchmark, MajorityVote().aggregate(o.broadcasts, r), g))
        test["self"].append(score(WORLD.benchmark, o.own.answer, g))
    roles = ["you" if a == r else ("honest" if a in bw.honest else "liar") for a in agents]
    order = sorted(range(len(agents)), key=lambda j: ({"you": 0, "honest": 1, "liar": 2}[roles[j]], j))
    bloc = sum(1 for j in range(len(agents)) if roles[j] == "liar")
    tempered = len(set(fit.groups[[j for j in range(len(agents)) if roles[j] == "liar"]]))
    return dict(race_acc=race_acc[:, order], ds_acc=ds_acc[:, order], race_dec=race_dec, ds_dec=ds_dec,
                true_acc=true_acc[order], roles=[roles[j] for j in order],
                models=[bw.models[agents[j]] for j in order], chance=chance, mean_k=mean_k, n_hist=len(rows),
                test={k: 100 * float(np.mean(v)) for k, v in test.items()}, n_test=len(bw.splits["test"]),
                bloc=bloc, bloc_groups=tempered)


def decision(a: np.ndarray, mean_k: float) -> list[str]:
    a = np.clip(a, 1e-6, 1 - 1e-6)
    lam = np.log(a) + np.log(max(mean_k - 1, 1)) - np.log1p(-a)
    return ["trust" if x > 0.25 else ("invert" if x < -0.25 else "discard") for x in lam]


def render(out: Path, fps: int = 12) -> Path:
    viz.setup()
    tr = trajectories()
    n = len(tr["roles"])
    intro, outro = 3 * fps, 6 * fps
    steps = (N_ITER - 1) * SUB + 1
    total = intro + steps + outro
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(viz.SURFACE)
    fig.text(0.03, 0.955, "Label switching: why classical estimators fail when liars are the majority",
             fontsize=17, fontweight="bold", color=viz.INK)
    fig.text(0.03, 0.915, f"One honest receiver, {tr['n_hist']} unlabelled past questions (MMLU). "
             f"{tr['bloc']} of {n} agents are a coordinated liar bloc.\nSame likelihood, same EM updates: "
             "only the starting point and the anchor differ.", fontsize=10, color=viz.INK_2, va="center")
    axes = [fig.add_axes([0.13, 0.41, 0.36, 0.40]), fig.add_axes([0.61, 0.41, 0.36, 0.40])]
    titles = ["Dawid–Skene (no anchor)\nstarts from the majority vote", "RACE (receiver-anchored)\nstarts from its own answers"]
    ys = np.arange(n)[::-1]
    names = []
    for j, role in enumerate(tr["roles"]):
        m = (tr["models"][j] or "").replace("_", "-")
        names.append({"you": "You (honest)", "honest": "Honest peer", "liar": "Liar"}[role] + f"\n{m}")
    bars = []
    for ax, title in zip(axes, titles, strict=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, n - 0.4)
        ax.axvline(tr["chance"], color=viz.MUTED, lw=1, ls=":")
        ax.text(tr["chance"], n - 0.45, "chance", fontsize=7.5, color=viz.MUTED, ha="center", va="bottom")
        ax.set_yticks(ys, names, fontsize=7.5)
        for tick, role in zip(ax.get_yticklabels(), tr["roles"], strict=True):
            tick.set_color(viz.RED if role == "liar" else viz.INK)
        ax.set_xlabel("estimated accuracy of each agent (no labels used)", fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=11, pad=14)
        ax.grid(axis="y", visible=False)
        ax.scatter(tr["true_acc"], ys, marker="|", s=260, color=viz.INK, lw=2.2, zorder=5)
        bars.append(ax.barh(ys, np.zeros(n), height=0.62, color=viz.GRID, zorder=3))
    iter_text = fig.text(0.535, 0.83, "", ha="center", fontsize=11, color=viz.INK, fontweight="bold")
    ax2 = fig.add_axes([0.13, 0.07, 0.36, 0.17])
    ax2.set_xlim(1, N_ITER)
    ax2.set_ylim(-4, 104)
    ax2.set_xlabel("EM iteration", fontsize=8.5)
    ax2.set_ylabel("history decoded\ncorrectly (%)", fontsize=8.5)
    ds_line, = ax2.plot([], [], color=viz.YELLOW, lw=2.2, label="Dawid–Skene")
    race_line, = ax2.plot([], [], color=viz.BLUE, lw=2.4, label="RACE")
    ax2.legend(loc="center right", fontsize=8)
    box = FancyBboxPatch((0.55, 0.04), 0.43, 0.21, boxstyle="round,pad=0.01,rounding_size=0.015",
                         transform=fig.transFigure, facecolor="white", edgecolor=viz.GRID, lw=1.2)
    fig.add_artist(box)
    caption = fig.text(0.565, 0.235, "", fontsize=9.5, color=viz.INK, va="top", linespacing=1.45)
    colors = {"trust": viz.BLUE, "invert": viz.RED, "discard": "#c9c8c3"}
    legend_items = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("trust", "discard", "invert")]
    legend_items.append(plt.Line2D([], [], marker="|", ls="", color=viz.INK, markersize=11, markeredgewidth=2.2))
    fig.legend(legend_items, ["TRUST (weight > 0)", "DISCARD (≈ 0)", "INVERT (weight < 0: read backwards)",
                              "true accuracy (viewer only)"],
               loc="upper center", bbox_to_anchor=(0.55, 0.335), ncol=4, fontsize=8.5, frameon=False)

    def state(frame):
        k = min(max(frame - intro, 0), steps - 1)
        i, frac = divmod(k, SUB)
        if i >= N_ITER - 1:
            return N_ITER - 1, N_ITER - 1, 0.0
        return i, i + 1, frac / SUB

    def text_for(frame, it):
        if frame < intro:
            return ("Start. Dawid–Skene initialises from the majority vote,\nwhich is the liars' shared answer. "
                    "RACE initialises from\nthe receiver's own answers and never lets the receiver's\n"
                    "own accuracy fall below chance.")
        if it < 4:
            return ("Each M-step re-scores every agent against the current\nguess of the truth; each E-step "
                    "re-votes with signed\nlog-odds weights. The two runs diverge from the first step.")
        if frame < intro + steps:
            return ("Both runs reach fixed points of the same likelihood, which\ncannot tell 'seven agents are right' "
                    "from 'three agents\nare right' (label switching, Prop. 2). Dawid–Skene trusts\nthe bloc and "
                    "inverts the honest agents; the anchor\nmakes RACE do the opposite.")
        t = tr["test"]
        return (f"Unseen TEST questions ({tr['n_test']}), this receiver:\n"
                f"Dawid–Skene {t['ds']:.0f}%  ·  majority {t['majority']:.0f}%  ·  alone {t['self']:.0f}%  ·  "
                f"RACE {t['race']:.0f}%\n"
                f"The liars are not ignored: RACE reads the bloc\nbackwards (INVERT) and counts it as "
                f"{tr['bloc_groups']} voice{'s' if tr['bloc_groups'] != 1 else ''}.")

    def update(frame):
        a, b, w = state(frame)
        for bset, acc in ((bars[0], tr["ds_acc"]), (bars[1], tr["race_acc"])):
            cur = acc[a] * (1 - w) + acc[b] * w
            if frame < intro:
                cur = cur * (frame / intro) ** 2
            for rect, v, dname in zip(bset, cur, decision(cur, tr["mean_k"]), strict=True):
                rect.set_width(v)
                rect.set_color(colors[dname])
        it = a + 1 + (1 if w > 0.5 else 0)
        iter_text.set_text("" if frame < intro else f"EM iteration {min(it, N_ITER)}")
        upto = min(it, N_ITER)
        xs = np.arange(1, upto + 1)
        ds_line.set_data(xs, 100 * tr["ds_dec"][:upto] if frame >= intro else [])
        race_line.set_data(xs, 100 * tr["race_dec"][:upto] if frame >= intro else [])
        caption.set_text(text_for(frame, upto))
        return []

    anim = animation.FuncAnimation(fig, update, frames=total, interval=1000 / fps, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2000, codec="libx264",
                                                 extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    plt.close(fig)
    # a still of the final state for the paper/README
    return out


if __name__ == "__main__":
    import json

    print(render(MEDIA / "label_switching.mp4"))
    t = trajectories()["test"]
    (ROOT / "results" / "tables" / "switching.json").write_text(json.dumps(
        {"switchDs": f"{t['ds']:.0f}", "switchRace": f"{t['race']:.0f}", "switchSelf": f"{t['self']:.0f}",
         "switchMaj": f"{t['majority']:.0f}"}, indent=1))
