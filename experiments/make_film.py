#!/usr/bin/env python3
"""Explainer film: how a multi-agent LLM swarm learns to use its liars.

Scenes (rendered separately, then concatenated with ffmpeg):

1. title        what the film shows
2. protocol     one round of the swarm: every agent thinks, broadcasts its
                answer as message packets to every honest agent, and each
                honest agent pools what it heard with its own RACE state
3. learning     the swarm's trust matrix (honest agents x all agents) evolving
                question by question, next to each honest agent's running
                accuracy; every honest agent learns *independently* and they
                converge on the same picture of who lies
4. elimination  one question in slow motion: raw votes vs RACE's signed evidence
5. results      headline numbers from the evaluation

All agents answer with real cached LLM outputs; the liars are real LLMs that
were prompted to mislead after seeing the honest answers. Nothing the viewer
sees about who is lying is available to the agents.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

from aip.aggregation.baselines import MajorityVote  # noqa: E402
from lai import viz  # noqa: E402
from lai.data import score  # noqa: E402
from lai.race import RACEAggregator  # noqa: E402
from lai.sim import World, build_world, stable_seed  # noqa: E402

FF = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams["animation.ffmpeg_path"] = FF
MEDIA = ROOT / "media"
FPS = 12
W, H = 16, 9
ANSWER_COLOR = {"A": viz.VIOLET, "B": viz.YELLOW, "C": viz.AQUA, "D": viz.MAGENTA}
DECISION_COLOR = {"trust": viz.BLUE, "invert": viz.RED, "discard": "#bdbcb7"}
DIV = LinearSegmentedColormap.from_list("div", viz.DIVERGING)


def writer():
    return animation.FFMpegWriter(fps=FPS, bitrate=2600, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def header(fig, title: str, sub: str) -> None:
    fig.text(0.03, 0.94, title, fontsize=20, fontweight="bold", color=viz.INK)
    fig.text(0.03, 0.905, sub, fontsize=12, color=viz.INK_2)


# ---------------------------------------------------------------- data


def prepare(world: World, n_steps: int) -> dict:
    """Online replay: at every step each honest agent fits RACE on earlier questions only."""
    bw = build_world(world)
    b = world.benchmark
    labels = list(bw.data.label_space)
    order = sorted(range(len(bw.data.task_ids)), key=lambda t: stable_seed("film", world.seed, bw.data.task_ids[t]))[:n_steps]
    honest = list(bw.honest)
    steps = []
    history = {r: [] for r in honest}
    tallies = {r: [] for r in honest}
    own_tally = {r: [] for r in honest}
    maj_tally = []
    for t in order:
        fits, preds, posts, evidence = {}, {}, {}, {}
        for r in honest:
            obs = bw.defense[t][r]
            if history[r]:
                agg = RACEAggregator(labels)
                agg.fit([(o,) for o in history[r]])
                fits[r] = {p: (c.decision, c.weight_at_chance_k, c.accuracy) for p, c in agg.diagnostics.channels[r].items()}
                preds[r] = agg.aggregate(obs.broadcasts, r)
                posts[r] = agg.posterior(obs.broadcasts, r)
                evidence[r] = agg.evidence(obs.broadcasts, r)
            else:
                fits[r], preds[r], posts[r], evidence[r] = {}, obs.own.answer, {}, {}
            tallies[r].append(float(score(b, preds[r], bw.data.gold[t])))
            own_tally[r].append(float(score(b, obs.own.answer, bw.data.gold[t])))
        maj = MajorityVote().aggregate(bw.defense[t][honest[0]].broadcasts, honest[0])
        maj_tally.append(float(score(b, maj, bw.data.gold[t])))
        steps.append(dict(task=bw.data.task_ids[t], gold=bw.data.gold[t],
                          answers={x.agent_id: x.answer for x in bw.raw[t]},
                          fits=fits, preds=preds, posts=posts, evidence=evidence, majority=maj,
                          self_curve=list(np.mean([np.cumsum(v) / np.arange(1, len(v) + 1) for v in own_tally.values()], axis=0)),
                          race_curve=list(np.mean([np.cumsum(v) / np.arange(1, len(v) + 1) for v in tallies.values()], axis=0)),
                          acc={r: float(np.mean(v)) for r, v in tallies.items()},
                          acc_curve={r: list(np.cumsum(v) / np.arange(1, len(v) + 1)) for r, v in tallies.items()},
                          maj_curve=list(np.cumsum(maj_tally) / np.arange(1, len(maj_tally) + 1))))
        for r in honest:
            history[r].append(bw.defense[t][r])
    items = {it["task_id"]: it for it in json.loads((ROOT / "data/benchmarks" / f"{b}_cached_items.json").read_text())}
    return dict(steps=steps, honest=honest, byz=sorted(bw.byzantine), models=[str(m) for m in bw.models],
                labels=labels, items=items, benchmark=b, n=world.n_agents)


def showcase_rounds(sim: dict, n_late: int = 3) -> tuple[int, ...]:
    """Two early rounds (no history yet), then late rounds where the vote is fooled but RACE is not."""
    steps, honest = sim["steps"], sim["honest"]

    def readable(k):
        item = sim["items"].get(steps[k]["task"])
        return item is not None and len(item["question"]) >= 60

    fooled = [k for k in range(len(steps) // 2, len(steps)) if readable(k) and steps[k]["majority"] != steps[k]["gold"]
              and np.mean([steps[k]["preds"][r] == steps[k]["gold"] for r in honest]) >= 0.8]
    rest = [k for k in range(len(steps) // 2, len(steps)) if readable(k) and k not in fooled]
    late = (fooled + rest)[:n_late]
    early = [k for k in range(len(steps)) if readable(k)][:2]
    return tuple(early + sorted(late))


def names(sim) -> dict[int, str]:
    out = {}
    for j, m in enumerate(sim["models"]):
        out[j] = "LLM liar" if j in sim["byz"] else m.replace("_", "-").replace("-reasoning", "").replace("-think", "")
    return out


# ---------------------------------------------------------------- scenes


def scene_title(path: Path, seconds: float = 5.0) -> None:
    viz.setup()
    fig = plt.figure(figsize=(W, H), dpi=100)

    def draw(i):
        fig.clear()
        a = min(1.0, i / (FPS * 1.2))
        fig.text(0.5, 0.62, "Liars Are Information", ha="center", fontsize=46, fontweight="bold", alpha=a)
        fig.text(0.5, 0.52, "How honest agents in a multi-agent LLM swarm learn, without labels,\n"
                            "to trust, discard, or invert what their peers tell them",
                 ha="center", fontsize=18, color=viz.INK_2, alpha=a)
        fig.text(0.5, 0.36, "RACE: Receiver-Anchored Channel Estimation", ha="center", fontsize=15, color=viz.BLUE,
                 fontweight="bold", alpha=a)
        fig.text(0.5, 0.1, "Real LLM answers (7 open-weight models) · liars are LLMs prompted to mislead · MMLU",
                 ha="center", fontsize=11, color=viz.MUTED, alpha=a)

    animation.FuncAnimation(fig, draw, frames=int(seconds * FPS)).save(path, writer=writer())
    plt.close(fig)


def ring(n: int, radius: float = 1.0) -> dict[int, tuple[float, float]]:
    return {j: (radius * np.cos(np.pi / 2 - 2 * np.pi * j / n), radius * np.sin(np.pi / 2 - 2 * np.pi * j / n))
            for j in range(n)}


def scene_protocol(sim: dict, path: Path, rounds: tuple[int, ...]) -> None:
    """One round in detail: think -> broadcast packets -> each honest agent decides."""
    viz.setup()
    fig = plt.figure(figsize=(W, H), dpi=100)
    pos = ring(sim["n"])
    nm = names(sim)
    think, fly, decide, hold = 6, 18, 14, 8
    per = think + fly + decide + hold
    frames = []
    for k in rounds:
        frames += [(k, "think", f / think) for f in range(think)]
        frames += [(k, "fly", f / (fly - 1)) for f in range(fly)]
        frames += [(k, "decide", f / decide) for f in range(decide)]
        frames += [(k, "hold", 1.0)] * hold
    del per

    def draw(i):
        k, phase, u = frames[i]
        st = sim["steps"][k]
        fig.clear()
        header(fig, "1 · How the swarm works: think, broadcast, decide",
               f"Question {k + 1}. Every agent answers alone, then broadcasts its answer to every other agent. "
               "Each honest agent pools what it heard using what it learned on earlier questions.")
        ax = fig.add_axes([0.02, 0.04, 0.6, 0.82])
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")
        # trust links (shown once the honest agents decide)
        if phase in ("decide", "hold"):
            alpha = min(1.0, u * 1.5) if phase == "decide" else 1.0
            for r in sim["honest"]:
                for p, (dec, lam, _) in st["fits"].get(r, {}).items():
                    if p == r:
                        continue
                    (x0, y0), (x1, y1) = pos[r], pos[p]
                    ax.plot([x0, x1], [y0, y1], color=DECISION_COLOR[dec], lw=0.6 + 0.5 * min(abs(lam), 4),
                            alpha=0.35 * alpha, zorder=1)
        # packets
        if phase == "fly":
            for s in range(sim["n"]):
                ans = st["answers"][s]
                for r in sim["honest"]:
                    if r == s:
                        continue
                    (x0, y0), (x1, y1) = pos[s], pos[r]
                    x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
                    ax.add_patch(Circle((x, y), 0.03, color=ANSWER_COLOR.get(ans, viz.MUTED), zorder=4, alpha=0.9))
        # agents
        for j in range(sim["n"]):
            x, y = pos[j]
            ans = st["answers"][j]
            liar = j in sim["byz"]
            pulse = 1.0 + (0.12 * np.sin(u * np.pi * 2) if phase == "think" else 0.0)
            ax.add_patch(Circle((x, y), 0.15 * pulse, facecolor="white", edgecolor=viz.RED if liar else viz.INK,
                                lw=2.4, zorder=5))
            if phase != "think":
                ax.add_patch(Circle((x, y), 0.105, facecolor=ANSWER_COLOR.get(ans, viz.MUTED), edgecolor="none", zorder=6))
                ax.text(x, y, ans or "–", ha="center", va="center", color="white", fontsize=15, fontweight="bold", zorder=7)
            else:
                ax.text(x, y, "…", ha="center", va="center", color=viz.INK_2, fontsize=18, zorder=7)
            ux, uy = x / np.hypot(x, y), y / np.hypot(x, y)
            ax.text(x + 0.2 * ux, y + 0.2 * uy, nm[j], ha="left" if ux > 0.2 else "right" if ux < -0.2 else "center",
                    va="bottom" if uy > 0.8 else "top" if uy < -0.8 else "center", fontsize=9.5,
                    color=viz.RED if liar else viz.INK_2, fontweight="bold" if liar else None, zorder=8)
            if phase in ("decide", "hold") and j in sim["honest"]:
                pred = st["preds"][j]
                ok = pred == st["gold"]
                ax.add_patch(FancyBboxPatch((x * 0.72 - 0.11, y * 0.72 - 0.07), 0.22, 0.14,
                                            boxstyle="round,pad=0.02", fc="white",
                                            ec=viz.GREEN if ok else viz.RED, lw=1.8, zorder=9))
                ax.text(x * 0.72, y * 0.72, f"→ {pred}", ha="center", va="center", fontsize=11, fontweight="bold", zorder=10)
        ax.text(0, 0, f"truth\n{st['gold']}", ha="center", va="center", fontsize=13, color=viz.GREEN, fontweight="bold")
        ax.text(-1.5, -1.33, "Black ring: honest agent.  Red ring: liar (shown to you, never to the agents).  "
                "Nobody is ever told the truth.", fontsize=9.5, color=viz.INK_2)
        if phase in ("decide", "hold") and k == 0:
            ax.text(0, -0.28, "no history yet:\neach honest agent keeps its own answer", ha="center", va="top",
                    fontsize=10, color=viz.INK_2)
        # side panel: the question and what each role does
        side = fig.add_axes([0.63, 0.04, 0.35, 0.82])
        side.axis("off")
        item = sim["items"].get(st["task"])
        if item:
            side.text(0, 0.98, "The question", fontsize=12, fontweight="bold", va="top")
            side.text(0, 0.92, "\n".join(textwrap.wrap(item["question"], 52)[:5]), fontsize=10.5, va="top")
            for n_, c in enumerate(item["choices"][:4]):
                side.add_patch(Circle((0.02, 0.62 - 0.055 * n_), 0.013, color=ANSWER_COLOR["ABCD"[n_]], transform=side.transAxes))
                side.text(0.05, 0.62 - 0.055 * n_, f"{'ABCD'[n_]}. {textwrap.shorten(c, 46)}", fontsize=10, va="center")
        steps_text = [("think", "1. Each agent answers alone (honest models answer; liars are LLMs told to mislead)."),
                      ("fly", "2. Everyone broadcasts: the coloured packets are answers travelling to each honest agent."),
                      ("decide", "3. Each honest agent pools the answers with its own RACE weights: blue link = trust, "
                                 "red link = invert (\"that answer is probably wrong\"). Box = its final answer.")]
        for n_, (ph, txt) in enumerate(steps_text):
            active = ph == phase or (phase == "hold" and ph == "decide")
            side.text(0, 0.34 - 0.11 * n_, "\n".join(textwrap.wrap(txt, 58)), fontsize=10.5, va="top",
                      color=viz.INK if active else viz.MUTED, fontweight="bold" if active else None)
        side.text(0, 0.0, f"Majority vote of all 10 agents: {st['majority']} "
                          f"{'✓' if st['majority'] == st['gold'] else '✗'}", fontsize=11,
                  color=viz.GREEN if st['majority'] == st['gold'] else viz.RED, fontweight="bold")

    animation.FuncAnimation(fig, draw, frames=len(frames)).save(path, writer=writer())
    plt.close(fig)


def scene_learning(sim: dict, path: Path, frames_per_step: int = 2) -> None:
    """Trust matrix of every honest agent evolving question by question."""
    viz.setup()
    fig = plt.figure(figsize=(W, H), dpi=100)
    nm = names(sim)
    honest, n = sim["honest"], sim["n"]
    order_cols = honest + sim["byz"]
    n_steps = len(sim["steps"])
    frames = [k for k in range(n_steps) for _ in range(frames_per_step)] + [n_steps - 1] * FPS * 3

    def draw(i):
        k = frames[i]
        st = sim["steps"][k]
        fig.clear()
        header(fig, "2 · The whole swarm learns, and every honest agent learns on its own",
               f"After {k} unlabeled questions. Row = one honest agent's private, label-free estimate of every peer "
               "(λ: log-odds one answer is worth). Nobody shares estimates.")
        ax = fig.add_axes([0.1, 0.17, 0.46, 0.64])
        mat = np.full((len(honest), n), np.nan)
        for a, r in enumerate(honest):
            for b_, p in enumerate(order_cols):
                if p == r:
                    continue
                dec = st["fits"].get(r, {}).get(p)
                mat[a, b_] = dec[1] if dec else 0.0
        im = ax.imshow(np.clip(mat, -4, 4), cmap=DIV, norm=TwoSlopeNorm(0, -4, 4), aspect="auto")
        for (a, b_), v in np.ndenumerate(mat):
            if np.isfinite(v):
                ax.text(b_, a, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 2.2 else viz.INK)
            else:
                ax.text(b_, a, "self", ha="center", va="center", fontsize=8.5, color=viz.MUTED)
        ax.set_xticks(range(n), [nm[p] for p in order_cols], rotation=35, ha="right", fontsize=9.5)
        for tick, p in zip(ax.get_xticklabels(), order_cols, strict=True):
            if p in sim["byz"]:
                tick.set_color(viz.RED)
                tick.set_fontweight("bold")
        ax.set_yticks(range(len(honest)), [nm[r] for r in honest], fontsize=10)
        ax.set_ylabel("honest agent (the one estimating)")
        ax.axvline(len(honest) - 0.5, color=viz.INK, lw=1.5)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("λ   (blue: trust · gray: discard · red: invert)")
        ax.set_title("peers: honest on the left, liars on the right (the red labels are for you; agents never see them)",
                     loc="left", fontsize=9.5, color=viz.INK_2, fontweight="normal")
        ax2 = fig.add_axes([0.68, 0.17, 0.29, 0.6])
        x = np.arange(1, k + 2)
        for r in honest:
            ax2.plot(x, 100 * np.array(st["acc_curve"][r]), lw=0.9, color=viz.BLUE, alpha=0.35)
        curves = [("race_curve", viz.BLUE, 2.4, "honest agents with RACE (mean)"),
                  ("self_curve", viz.MUTED, 1.8, "same agents answering alone"),
                  ("maj_curve", viz.AQUA, 1.8, "majority vote of all 10")]
        ends = []
        for key, color, lw, label in curves:
            c = 100 * np.array(st[key])
            ax2.plot(x, c, lw=lw, color=color, label=label)
            ends.append([c[-1], color])
        # direct labels at the line ends, nudged apart
        ends.sort(key=lambda e: e[0])
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 6)
        for (y, color), (key, *_rest) in zip(ends, sorted(curves, key=lambda c: 100 * st[c[0]][-1]), strict=True):
            ax2.text(k + 3, y, f"{100 * st[key][-1]:.0f}%", color=color, fontsize=11, fontweight="bold", va="center")
        ax2.set_xlim(1, n_steps + 12)
        ax2.set_ylim(0, 104)
        ax2.set_xlabel("questions seen")
        ax2.set_ylabel("running accuracy (%)")
        ax2.legend(loc="lower right", fontsize=9)
        ax2.set_title("what the learning buys", loc="left")

    animation.FuncAnimation(fig, draw, frames=len(frames)).save(path, writer=writer())
    plt.close(fig)


def scene_elimination(sim: dict, path: Path) -> None:
    """One question in slow motion: raw votes vs signed evidence, agent by agent."""
    viz.setup()
    steps = sim["steps"]
    r0 = sim["honest"][0]
    pick = next((k for k in range(len(steps) - 1, 20, -1)
                 if steps[k]["majority"] != steps[k]["gold"] and steps[k]["preds"][r0] == steps[k]["gold"]
                 and steps[k]["evidence"].get(r0)), len(steps) - 1)
    st = steps[pick]
    nm = names(sim)
    ev = st["evidence"][r0]
    labels = sim["labels"]
    fits = st["fits"][r0]
    order = [r0] + sorted((j for j in ev if j != r0), key=lambda j: -ev[j][1])
    fig = plt.figure(figsize=(W, H), dpi=100)
    n_frames = FPS * 16
    reveal = FPS * 9

    def draw(i):
        u = min(1.0, i / reveal)
        n_shown = max(1, int(np.ceil(u * len(order))))
        shown = order[:n_shown]
        done = n_shown == len(order)
        fig.clear()
        header(fig, "3 · One question in slow motion: counting votes vs. weighing evidence",
               f"Question {pick + 1}, seen by {nm[r0]}. Majority voting counts every answer as +1. RACE adds, for each "
               "answer, the log-odds that peer's answers carry about the truth.")
        # raw votes, one block per agent
        ax = fig.add_axes([0.05, 0.14, 0.24, 0.64])
        height = {l: 0 for l in labels}
        for j, a in st["answers"].items():
            if a not in height:
                continue
            ax.bar(labels.index(a), 1, bottom=height[a], color=ANSWER_COLOR[a], edgecolor=viz.SURFACE, lw=2,
                   hatch="//" if j in sim["byz"] else None)
            height[a] += 1
        ax.set_xticks(range(len(labels)), labels, fontsize=13)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_title(f"Raw votes: majority picks {st['majority']} ✗", loc="left", color=viz.RED)
        ax.set_ylabel("number of agents (hatched: liars)")
        ax.grid(axis="x", visible=False)
        # signed evidence, stacked up (for) and down (against)
        ax2 = fig.add_axes([0.36, 0.14, 0.3, 0.64])
        up = {l: 0.0 for l in labels}
        down = {l: 0.0 for l in labels}
        for j in shown:
            a, w = ev[j]
            dec = fits.get(j, ("discard",))[0]
            color = viz.INK if j == r0 else DECISION_COLOR[dec]
            if w >= 0:
                ax2.bar(labels.index(a), w, bottom=up[a], width=0.62, color=color, edgecolor=viz.SURFACE, lw=1.5)
                up[a] += w
            else:
                ax2.bar(labels.index(a), w, bottom=down[a], width=0.62, color=color, edgecolor=viz.SURFACE, lw=1.5)
                down[a] += w
        ax2.axhline(0, color=viz.INK_2, lw=1)
        total = {l: up[l] + down[l] for l in labels}
        for l in labels:
            if up[l] or down[l]:
                ax2.text(labels.index(l), up[l] + 0.25, f"{total[l]:+.1f}", ha="center", fontsize=11, fontweight="bold",
                         color=viz.GREEN if done and l == st["preds"][r0] else viz.INK)
        ax2.set_xticks(range(len(labels)), labels, fontsize=13)
        ax2.set_xlim(-0.6, len(labels) - 0.4)
        lo = min(0.0, min(down.values())) - 0.8
        hi = max(up.values()) + 1.2
        ax2.set_ylim(min(lo, -2), max(hi, 4))
        ax2.set_ylabel("signed evidence (log-odds)")
        ax2.set_title(f"RACE evidence: picks {st['preds'][r0]} {'✓' if st['preds'][r0] == st['gold'] else '✗'}"
                      if done else "RACE evidence (adding agents one by one)", loc="left",
                      color=viz.GREEN if done and st["preds"][r0] == st["gold"] else viz.INK)
        ax2.grid(axis="x", visible=False)
        # the ledger: who said what, and what it is worth
        ax3 = fig.add_axes([0.7, 0.14, 0.28, 0.64])
        ax3.axis("off")
        ax3.set_xlim(0, 1)
        ax3.set_ylim(len(order) + 0.8, -1.2)
        ax3.text(0.0, -0.8, "agent", fontsize=10, color=viz.INK_2, fontweight="bold")
        ax3.text(0.46, -0.8, "said", fontsize=10, color=viz.INK_2, fontweight="bold")
        ax3.text(0.6, -0.8, "â", fontsize=10, color=viz.INK_2, fontweight="bold")
        ax3.text(0.75, -0.8, "worth", fontsize=10, color=viz.INK_2, fontweight="bold")
        for row, j in enumerate(order):
            if j not in shown:
                continue
            a, w = ev[j]
            dec = fits.get(j, ("discard", 0.0, float("nan")))
            liar = j in sim["byz"]
            who = f"{nm[j]} (me)" if j == r0 else nm[j]
            ax3.text(0.0, row, who, fontsize=10, va="center", color=viz.RED if liar else viz.INK,
                     fontweight="bold" if liar else None)
            ax3.text(0.49, row, f" {a} ", fontsize=9.5, va="center", ha="center", color="white", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc=ANSWER_COLOR.get(a, viz.MUTED), ec="none"))
            ax3.text(0.6, row, f"{dec[2]:.2f}", fontsize=10, va="center")
            word = "own" if j == r0 else {"trust": "trust", "invert": "invert", "discard": "discard"}[dec[0]]
            ax3.text(0.75, row, f"{w:+.2f}  {word}", fontsize=10, va="center",
                     color=viz.INK if j == r0 else DECISION_COLOR[dec[0]] if dec[0] != "discard" else viz.MUTED)
        fig.text(0.05, 0.035, "Blue: evidence FOR an answer (trusted peer).  Red, below zero: evidence AGAINST it (inverted "
                              "peer).  Black: the agent's own answer.\nLiars that share a model are detected as clones and "
                              f"counted once. The truth is {st['gold']}; no agent was ever told any truth, or who the liars are.",
                 fontsize=10.5, color=viz.INK_2)

    animation.FuncAnimation(fig, draw, frames=n_frames).save(path, writer=writer())
    plt.close(fig)


def scene_results(path: Path, seconds: float = 9.0) -> None:
    head = json.loads((ROOT / "results/tables/headline.json").read_text())
    rows = [("Coherent liars, 90% of the swarm", "mainSelfNine", "mainMajNine", "mainAipNine", "mainRaceNine"),
            ("Gate-aware liars (the attack that beat AIP), 70%", "zooGateSelfSeven", "zooGateMajSeven", "zooGateAipSeven", "zooGateRaceSeven"),
            ("Real LLM deceivers, 50%", "llmSelfFive", "llmMajFive", "llmAipFive", "llmRaceFive"),
            ("Real LLM deceivers, 70%", "llmSelfSeven", "llmMajSeven", "llmAipSeven", "llmRaceSeven"),
            ("Attack mix vs crowdsourcing baselines, 70%", "crowdSelfSeven", "crowdMajSeven", "crowdAipSeven", "crowdRaceSeven")]
    viz.setup()
    fig = plt.figure(figsize=(W, H), dpi=100)

    def draw(i):
        fig.clear()
        a = min(1.0, i / (FPS * 1.0))
        header(fig, "4 · What this buys", "Mean accuracy of honest agents over six benchmarks, nine open-weight models "
               "(receiver alone / majority vote / AIP / RACE)")
        cols = ["Receiver alone", "Majority vote", "AIP (prior work)", "RACE"]
        for c_i, c in enumerate(cols):
            fig.text(0.47 + 0.13 * c_i, 0.74, c, ha="center", fontsize=13, color=viz.INK_2, alpha=a)
        for r_i, (lab, *keys) in enumerate(rows):
            y = 0.66 - 0.085 * r_i
            fig.text(0.05, y, lab, fontsize=13.5, alpha=a, va="center")
            for c_i, k in enumerate(keys):
                v = head.get(k, "??")
                fig.text(0.47 + 0.13 * c_i, y, f"{v}%", ha="center", va="center", alpha=a,
                         fontsize=20 if c_i == 3 else 16, fontweight="bold" if c_i == 3 else None,
                         color=viz.BLUE if c_i == 3 else viz.INK)
        fig.text(0.05, 0.24, f"Liars are information: with 7 of 10 agents lying independently, RACE reaches "
                             f"{head.get('extRaceIndepSeven', '??')}%, more than an oracle that knows who lies and removes "
                             f"them ({head.get('extRemovalIndepSeven', '??')}%).", fontsize=13, color=viz.INK, alpha=a)
        fig.text(0.05, 0.185, f"Classical crowdsourcing estimators (IWMV, MACE, GLAD, KOS, Dawid–Skene) fall to at most "
                              f"{head.get('crowdClassicalMaxSeven', '??')}% when 7 of 10 lie: the anchor is what matters.",
                 fontsize=13, color=viz.INK, alpha=a)
        fig.text(0.05, 0.13, f"Pre-registered fresh live run: {head.get('hypSupported', '??')} of {head.get('hypTotal', '??')} "
                             f"hypotheses supported (v3.1 {head.get('hypHOneATarget', '??')}% vs v3.0 "
                             f"{head.get('hypHOneABase', '??')}% on unseen yes/no questions).", fontsize=13, color=viz.INK,
                 alpha=a)
        fig.text(0.05, 0.05, "Where RACE fails, reported as prominently: camouflage attackers, sleepers, and receivers at chance.\n"
                             "Code, data, paper: github.com/Hisernberg/liars-are-information", fontsize=12, color=viz.INK_2,
                 alpha=a)

    animation.FuncAnimation(fig, draw, frames=int(seconds * FPS)).save(path, writer=writer())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--only", nargs="*", default=None,
                        help="re-render only these scenes (title protocol learning elimination results)")
    args = parser.parse_args()
    MEDIA.mkdir(exist_ok=True)
    parts = MEDIA / "film_parts"
    parts.mkdir(exist_ok=True)
    world = World("mmlu", 0.5, "llm:rushing", 1.0, 1.0, 0)
    sim = prepare(world, args.steps)
    todo = set(args.only or ("title", "protocol", "learning", "elimination", "results"))
    if "title" in todo:
        scene_title(parts / "0_title.mp4")
    if "protocol" in todo:
        scene_protocol(sim, parts / "1_protocol.mp4", rounds=showcase_rounds(sim))
    if "learning" in todo:
        scene_learning(sim, parts / "2_learning.mp4")
    if "elimination" in todo:
        scene_elimination(sim, parts / "3_elimination.mp4")
    if "results" in todo:
        scene_results(parts / "4_results.mp4")
    listing = parts / "concat.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in sorted(parts.glob("[0-9]_*.mp4"))))
    out = MEDIA / "film_liars_are_information.mp4"
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out)],
                   check=True, cwd=parts)
    print(out)


if __name__ == "__main__":
    main()
