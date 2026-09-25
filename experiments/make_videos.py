#!/usr/bin/env python3
"""Explainer videos of the multi-agent swarm (MP4 + GIF preview).

``swarm_<scenario>.mp4``  one honest receiver learning, task by task and with
    no labels, which peers to trust, discard or invert. Every method is refit
    online on strictly earlier tasks (predict-then-commit), so what the video
    shows is exactly what the receiver could have known at that moment.
``gate_aware_sweep.mp4``  the attack that defeats AIP, swept over coordination p.

Ground truth, peer identities and the liars' labels are drawn for the viewer
only; the receiver never sees them.
"""

from __future__ import annotations

import argparse
import json
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
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Patch  # noqa: E402

from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.aggregation.baselines import MajorityVote  # noqa: E402
from lai import viz  # noqa: E402
from lai.data import score  # noqa: E402
from lai.race import RACEAggregator  # noqa: E402
from lai.sim import THRESHOLDS_PATH, World, build_world, stable_seed  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
MEDIA = ROOT / "media"

SCENARIOS = {
    "boolq_coherent_f05": dict(world=World("boolq", 0.5, "coherent", 1.0, 1.0, 1),
                               title="BoolQ · 50% coordinated liars",
                               blurb="Binary questions. Half the swarm always says the opposite of the truth."),
    "mmlu_gateaware_f07": dict(world=World("mmlu", 0.7, "gate_aware", 0.25, 1.0, 0),
                               title="MMLU · 70% gate-aware liars",
                               blurb="The attack built to beat AIP: liars never tell the truth but barely coordinate."),
    "medqa_rushing_llm_f05": dict(world=World("medqa", 0.5, "llm:rushing", 1.0, 1.0, 2),
                                  title="MedQA · 50% real LLM deceivers",
                                  blurb="Liars are LLMs prompted to mislead after seeing the honest answers."),
}


def _questions(benchmark: str) -> dict[str, dict]:
    path = ROOT / "data" / "benchmarks" / f"{benchmark}_cached_items.json"
    if not path.exists():
        return {}
    return {it["task_id"]: it for it in json.loads(path.read_text())}


def simulate(world: World, n_steps: int, receiver_rank: int | None = None) -> dict:
    """Online replay for one honest receiver; returns per-step frame data."""
    bw = build_world(world)
    b = world.benchmark
    labels = list(bw.data.label_space) if bw.data.label_space else None
    # The median-accuracy honest receiver (fixed rule, not chosen by outcome).
    accs = {r: np.mean([score(b, bw.data.honest[bw.models[r]][t], bw.data.gold[t]) for t in range(len(bw.data.task_ids))])
            for r in bw.honest}
    rank = len(bw.honest) // 2 if receiver_rank is None else receiver_rank
    receiver = sorted(bw.honest, key=lambda r: -accs[r])[min(rank, len(bw.honest) - 1)]
    order = sorted(range(len(bw.data.task_ids)), key=lambda t: stable_seed("video", world.seed, bw.data.task_ids[t]))
    order = order[:n_steps]
    thresholds = InversionThresholds.load(THRESHOLDS_PATH)
    maj = MajorityVote()
    frames = []
    seen = []
    tallies = {k: [] for k in ("race", "aip_gated", "majority", "self")}
    for step, t in enumerate(order):
        obs = bw.defense[t][receiver]
        race = RACEAggregator(labels)
        aip = AIPAggregator(b, "gated", thresholds, ParityConfig(0.1), labels)
        if seen:
            race.fit([(o,) for o in seen])
            aip.fit([(o,) for o in seen])
        preds = {"race": race.aggregate(obs.broadcasts, receiver) if seen else obs.own.answer,
                 "aip_gated": aip.aggregate(obs.broadcasts, receiver) if seen else obs.own.answer,
                 "majority": maj.aggregate(obs.broadcasts, receiver), "self": obs.own.answer}
        gold = bw.data.gold[t]
        for k, v in preds.items():
            tallies[k].append(float(score(b, v, gold)))
        channels = race.diagnostics.channels.get(receiver, {}) if seen else {}
        aip_ch = aip.diagnostics.channels.get(receiver, {}) if seen else {}
        post = race.posterior(obs.broadcasts, receiver) if seen else {}
        frames.append(dict(
            step=step, task=bw.data.task_ids[t], gold=gold,
            answers={x.agent_id: x.answer for x in bw.raw[t]},
            correct={x.agent_id: bool(score(b, x.answer, gold)) for x in bw.raw[t]},
            decisions={p: (c.decision, c.weight_at_chance_k, c.accuracy) for p, c in channels.items()},
            aip_decisions={p: s.decision for p, s in aip_ch.items()},
            posterior=post, preds=preds,
            running={k: float(np.mean(v)) for k, v in tallies.items()},
            curve={k: list(np.cumsum(v) / np.arange(1, len(v) + 1)) for k, v in tallies.items()},
        ))
        seen.append(obs)
    return dict(frames=frames, receiver=receiver, byzantine=sorted(bw.byzantine), models=[str(m) for m in bw.models],
                labels=labels, benchmark=b)


DECISION_COLOR = {"trust": viz.BLUE, "invert": viz.RED, "discard": "#c9c8c3"}


def render_swarm(name: str, spec: dict, n_steps: int, fps: float, out_dir: Path) -> Path:
    sim = simulate(spec["world"], n_steps)
    questions = _questions(sim["benchmark"])
    frames = sim["frames"]
    receiver = sim["receiver"]
    peers = [j for j in range(len(sim["models"])) if j != receiver]
    angle = {p: np.pi / 2 - 2 * np.pi * k / len(peers) for k, p in enumerate(peers)}
    pos = {p: (np.cos(a) * 1.0, np.sin(a) * 1.0) for p, a in angle.items()}
    pos[receiver] = (0.0, 0.0)
    viz.setup()
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], height_ratios=[1.15, 0.8, 1.05], left=0.02, right=0.98,
                          top=0.9, bottom=0.07, wspace=0.08, hspace=0.45)
    ax_g = fig.add_subplot(gs[:, 0])
    ax_q = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[2, 1])
    intro = 10
    hold = 16

    def draw(i: int) -> None:
        k = min(max(i - intro, 0), len(frames) - 1)
        fr = frames[k]
        for ax in (ax_g, ax_q, ax_p, ax_c):
            ax.clear()
        fig.texts.clear()
        fig.text(0.02, 0.955, "Liars Are Information — one honest receiver learns whom to trust, discard or invert",
                 fontsize=17, fontweight="bold", color=viz.INK)
        fig.text(0.02, 0.925, f"{spec['title']}.  {spec['blurb']}  No labels are ever given to the receiver.",
                 fontsize=11, color=viz.INK_2)
        # ---- swarm graph
        ax_g.set_xlim(-1.45, 1.45)
        ax_g.set_ylim(-1.42, 1.38)
        ax_g.set_aspect("equal")
        ax_g.axis("off")
        for p in peers:
            dec, lam, a = fr["decisions"].get(p, ("discard", 0.0, 0.5))
            x, y = pos[p]
            lw = 0.8 + 1.4 * min(abs(lam), 5.0)
            ax_g.plot([0, x * 0.87], [0, y * 0.87], color=DECISION_COLOR[dec], lw=lw, alpha=0.85, zorder=1,
                      solid_capstyle="round")
            ans = fr["answers"][p]
            ok = fr["correct"][p]
            ax_g.add_patch(Circle((x, y), 0.13, facecolor="white", edgecolor=viz.INK if p not in sim["byzantine"] else viz.RED,
                                  lw=2.2, zorder=3))
            ax_g.text(x, y + 0.01, (ans or "–")[:6], ha="center", va="center", fontsize=15 if len(ans or "") < 3 else 9,
                      fontweight="bold", zorder=4, color=viz.INK)
            ax_g.text(x + 0.12, y + 0.1, "✓" if ok else "✗", color=viz.GREEN if ok else viz.RED, fontsize=13,
                      fontweight="bold", zorder=5)
            role = "LIAR" if p in sim["byzantine"] else sim["models"][p].replace("_", "-")
            ax_g.text(x, y - 0.19, role, ha="center", va="top", fontsize=8.5, zorder=6,
                      color=viz.RED if p in sim["byzantine"] else viz.INK_2, fontweight="bold" if p in sim["byzantine"] else None,
                      bbox=dict(boxstyle="round,pad=0.12", fc=viz.SURFACE, ec="none", alpha=0.9))
            if fr["decisions"]:
                ax_g.text(x * 0.68, y * 0.68, f"â={a:.2f}", ha="center", va="center", fontsize=7.5, color=viz.INK_2,
                          bbox=dict(boxstyle="round,pad=0.15", fc=viz.SURFACE, ec="none", alpha=0.8), zorder=2)
        ax_g.add_patch(Circle((0, 0), 0.19, facecolor=viz.BLUE, edgecolor=viz.SURFACE, lw=3, zorder=3))
        ax_g.text(0, 0.02, fr["answers"][receiver] or "–", ha="center", va="center", color="white", fontsize=18,
                  fontweight="bold", zorder=4)
        ax_g.text(0, -0.23, f"receiver: {sim['models'][receiver].replace('_', '-')} (median honest agent)", ha="center",
                  va="top", fontsize=9,
                  color=viz.INK, fontweight="bold", zorder=6,
                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=viz.GRID, alpha=0.95))
        legend = [("TRUST", viz.BLUE), ("DISCARD", "#a9a8a3"), ("INVERT", viz.RED)]
        for n, (lab, col) in enumerate(legend):
            ax_g.plot([-1.4, -1.28], [-1.22 - 0.07 * n] * 2, color=col, lw=4)
            ax_g.text(-1.25, -1.22 - 0.07 * n, lab, va="center", fontsize=9, color=viz.INK_2)
        ax_g.text(1.42, -1.33, "red ring / ✓✗ / names: shown to viewer only", ha="right", fontsize=8, color=viz.MUTED)
        ax_g.text(-1.42, 1.3, f"task {k + 1} of {len(frames)}   ·   history = {k} unlabeled tasks", fontsize=11,
                  color=viz.INK, fontweight="bold")
        # ---- question
        ax_q.axis("off")
        item = questions.get(fr["task"])
        qtext = item["question"] if item else fr["task"]
        if item and item.get("benchmark") == "boolq":
            qtext = qtext[0].upper() + qtext[1:] + "?"
        lines = textwrap.wrap(qtext, 70)[:4]
        ax_q.text(0, 1.0, "Question", fontsize=10, color=viz.INK_2, fontweight="bold", transform=ax_q.transAxes, va="top")
        ax_q.text(0, 0.86, "\n".join(lines), fontsize=11, color=viz.INK, transform=ax_q.transAxes, va="top")
        if item:
            opts = [f"{chr(65 + n)}. {textwrap.shorten(c, 34)}" for n, c in enumerate(item["choices"])]
            ax_q.text(0, 0.3, "    ".join(opts[:2]) + ("\n" + "    ".join(opts[2:]) if len(opts) > 2 else ""),
                      fontsize=9.5, color=viz.INK_2, transform=ax_q.transAxes, va="top")
        ax_q.text(0, -0.05, f"truth (hidden): {fr['gold']}", fontsize=10, color=viz.GREEN, fontweight="bold",
                  transform=ax_q.transAxes)
        # ---- posterior vs votes
        labels = sim["labels"] or sorted({a for a in fr["answers"].values() if a})
        votes = pd.Series([a for a in fr["answers"].values()]).value_counts()
        xs = np.arange(len(labels))
        vv = np.array([votes.get(l, 0) for l in labels], dtype=float)
        pp = np.array([fr["posterior"].get(l, 0.0) for l in labels]) if fr["posterior"] else np.zeros(len(labels))
        ax_p.bar(xs - 0.2, vv / max(vv.sum(), 1) * 100, 0.38, color=viz.AQUA, label="share of raw votes",
                 edgecolor=viz.SURFACE, lw=1.5)
        ax_p.bar(xs + 0.2, pp * 100, 0.38, color=viz.BLUE, label="RACE posterior", edgecolor=viz.SURFACE, lw=1.5)
        ax_p.set_xticks(xs, [l[:8] for l in labels])
        for n, l in enumerate(labels):
            if l == fr["gold"]:
                ax_p.get_xticklabels()[n].set_color(viz.GREEN)
                ax_p.get_xticklabels()[n].set_fontweight("bold")
        ax_p.set_ylim(0, 128)
        ax_p.set_yticks([0, 25, 50, 75, 100])
        ax_p.set_ylabel("%")
        ax_p.legend(handles=[Patch(facecolor=viz.AQUA, label="share of raw votes"),
                             Patch(facecolor=viz.BLUE, label="RACE posterior")], loc="upper right", ncol=2, fontsize=8)
        verdict = "  ".join(
            f"{viz.LABEL[m].replace(' (ours)', '')}: {fr['preds'][m] or '–'}{'✓' if score(sim['benchmark'], fr['preds'][m], fr['gold']) else '✗'}"
            for m in ("race", "aip_gated", "majority"))
        ax_p.set_title(verdict, loc="left", fontsize=9.5)
        ax_p.grid(axis="x", visible=False)
        # ---- running accuracy
        ends = []
        for m in ("self", "majority", "aip_gated", "race"):
            curve = fr["curve"][m]
            viz.line(ax_c, np.arange(1, len(curve) + 1), 100 * np.array(curve), m, marker="")
            ends.append([100 * curve[-1], 100 * curve[-1], m])
        ends.sort(key=lambda e: -e[0])
        ends[0][1] = min(ends[0][1], 100)
        for n in range(1, len(ends)):  # keep end labels >= 7 points apart, pushing downwards
            ends[n][1] = min(ends[n][1], ends[n - 1][1] - 7)
        for value, ypos, m in ends:
            ax_c.annotate(f"{value:.0f}%", (len(fr["curve"][m]), value), xytext=(len(fr["curve"][m]) + 2, ypos),
                          textcoords="data", fontsize=8.5, va="center", fontweight="bold",
                          color=viz.METHOD_STYLE.get(m, {}).get("color", viz.INK) if m != "self" else viz.INK_2)
        ax_c.set_xlim(1, len(frames) + 8)
        ax_c.set_ylim(-3, 103)
        ax_c.set_xlabel("tasks seen")
        ax_c.set_ylabel("running accuracy (%)")
        ax_c.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8)
        if i < intro:
            fig.patches.clear()
            fig.patches.append(FancyBboxPatch((0.2, 0.35), 0.6, 0.3, boxstyle="round,pad=0.02", transform=fig.transFigure,
                                              fc="white", ec=viz.GRID, lw=1.5, zorder=10, alpha=0.96))
            fig.text(0.5, 0.56, "A lie is still information", ha="center", fontsize=26, fontweight="bold", zorder=11)
            fig.text(0.5, 0.47, "If a peer's answer depends on the truth — even inversely — the receiver can learn it\n"
                                "from unlabeled history and use it. RACE: Receiver-Anchored Channel Estimation.",
                     ha="center", fontsize=13, color=viz.INK_2, zorder=11)
        else:
            fig.patches.clear()

    total = intro + len(frames) + hold
    anim = animation.FuncAnimation(fig, draw, frames=total, interval=1000 / fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"swarm_{name}.mp4"
    anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400, codec="libx264",
                                                  extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    plt.close(fig)  # README GIF previews are made by experiments/make_gifs.py
    (out_dir / f"swarm_{name}.json").write_text(json.dumps(
        {"receiver": sim["receiver"], "byzantine": sim["byzantine"], "models": sim["models"],
         "final_running_accuracy": frames[-1]["running"], "steps": len(frames), "world": vars(spec["world"])},
        indent=2, default=str))
    return path


def render_gate_sweep(out_dir: Path, fps: float = 8) -> Path | None:
    path = ROOT / "results" / "zoo" / "per_task.parquet"
    if not path.exists():
        return None
    zoo = pd.read_parquet(path)
    g = zoo[(zoo.attack == "gate_aware") & (zoo.split == "test")]
    s = g.groupby(["f", "param", "method"]).accuracy.mean().unstack()
    viz.setup()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), dpi=100)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.06, right=0.98, wspace=0.18)
    ps = np.linspace(0, 1, 121)
    methods = ["self", "majority", "aip_gated", "aip_soft", "race"]

    def interp(f, m, p):
        tbl = s.loc[f][m]
        return float(np.interp(p, tbl.index.to_numpy(), tbl.to_numpy()))

    def draw(i):
        p = ps[min(i, len(ps) - 1)]
        for ax in axes:
            ax.clear()
        fig.texts.clear()
        fig.text(0.02, 0.94, "The gate-aware adversary: tune coordination p to hide under AIP's ceiling",
                 fontsize=16, fontweight="bold")
        fig.text(0.02, 0.89, "Liars never state the truth. Low p = barely coherent, so a coherence gate cannot see them; "
                             "a truth-dependence estimator (RACE) still inverts them.", fontsize=10.5, color=viz.INK_2)
        ax = axes[0]
        pp = np.linspace(0, 1, 200)
        ax.plot(pp, pp ** 2 + (1 - pp ** 2) / 3, color=viz.ORANGE, lw=2.2, label="liar coherence q(p) — AIP's signal")
        ax.plot(pp, np.zeros_like(pp), color=viz.BLUE, lw=2.2, label="liar accuracy a(p) — RACE's signal")
        ax.axhline(0.677, color=viz.ORANGE, lw=1, alpha=0.6)
        ax.text(0.01, 0.69, "AIP inverts only above 0.677 (published)", fontsize=8.5, color=viz.INK_2)
        ax.scatter([p], [p ** 2 + (1 - p ** 2) / 3], s=120, color=viz.ORANGE, zorder=5, edgecolor="white", lw=2)
        ax.scatter([p], [0], s=120, color=viz.BLUE, zorder=5, edgecolor="white", lw=2)
        ax.axvline(p, color=viz.GRID, lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("attacker coordination p")
        ax.set_title(f"p = {p:.2f}", loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=9)
        ax = axes[1]
        xs = np.arange(len(methods))
        for n, f in enumerate((0.5, 0.7)):
            vals = [100 * interp(f, m, p) for m in methods]
            ax.bar(xs + (n - 0.5) * 0.4, vals, 0.38, color=[viz.METHOD_STYLE[m]["color"] for m in methods],
                   alpha=1.0 if f == 0.7 else 0.5, edgecolor=viz.SURFACE, lw=1.5)
            for x, v in zip(xs + (n - 0.5) * 0.4, vals, strict=True):
                ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5)
        ax.set_xticks(xs, [viz.LABEL[m] for m in methods], rotation=12, fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_ylabel("accuracy, mean of 6 benchmarks (%)")
        ax.set_title("light bars: f = 0.5    solid bars: f = 0.7", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)

    frames = list(range(len(ps))) + [len(ps) - 1] * 40
    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    out = out_dir / "gate_aware_sweep.mp4"
    anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=1800, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    for name, spec in SCENARIOS.items():
        if args.only and name not in args.only:
            continue
        print("rendering", name, flush=True)
        print(render_swarm(name, spec, args.steps, args.fps, MEDIA))
    if not args.only or "gate_sweep" in args.only:
        print(render_gate_sweep(MEDIA))
    if not args.only or "live" in args.only:
        for b in ("mmlu", "boolq"):
            print(render_live_debate(MEDIA, b))



def render_live_debate(out_dir: Path, benchmark: str = "mmlu", n_steps: int = 120, fps: float = 4.0) -> Path | None:
    """Live six-model panel: independent answers, covert saboteurs, and the debate round."""
    from lai.data import LIVE_MODELS

    raw_dir = ROOT / "data" / "live_cache" / "raw"
    paths = list((raw_dir / "stage1" / benchmark).glob("*.parquet")) + list((raw_dir / "stage2" / benchmark).glob("*.parquet"))
    if len(paths) < 2 * len(LIVE_MODELS):
        return None
    raw = pd.concat([pd.read_parquet(p) for p in paths])
    piv = raw.pivot_table(index="task_id", columns=["role", "model"], values="extracted_answer", aggfunc="first")
    gold = raw.drop_duplicates("task_id").set_index("task_id").gold_answer
    questions = _questions(benchmark)
    tasks = sorted(piv.index, key=lambda t: stable_seed("live-video", t))[:n_steps]
    labels = ["A", "B"] if benchmark == "boolq" else ["A", "B", "C", "D"]
    # Every honest model runs its own RACE, refit online on earlier questions only. The curves
    # average over all six receivers (no receiver is singled out); the cards show one receiver's
    # trust decisions, the most accurate honest model's.
    acc = {m: np.mean(piv[("honest", m)].loc[tasks] == gold.loc[tasks]) for m in LIVE_MODELS}
    receiver_model = max(acc, key=acc.get)
    agents = [("honest", m) for m in LIVE_MODELS] + [("solo", m) for m in LIVE_MODELS]
    rids = [agents.index(("honest", m)) for m in LIVE_MODELS]
    shown = agents.index(("honest", receiver_model))
    from aip.types import Broadcast, Observation

    def obs_for(t, rid):
        row = tuple(Broadcast(j, t, piv[a].get(t), 0.5, 0.5, False) for j, a in enumerate(agents))
        return Observation(rid, t, row)

    frames, seen = [], {rid: [] for rid in rids}
    tallies = {k: [] for k in ("debate_majority", "majority_all", "race", "race_onecoin", "self")}
    for t in tasks:
        per = {k: [] for k in tallies}
        shown_preds, ch = {}, {}
        for rid in rids:
            obs = obs_for(t, rid)
            preds = {"self": obs.own.answer, "majority_all": MajorityVote().aggregate(obs.broadcasts, rid)}
            for key, model in (("race", "auto"), ("race_onecoin", "onecoin")):
                if seen[rid]:
                    agg = RACEAggregator(labels, model=model)
                    agg.fit([(o,) for o in seen[rid]])
                    preds[key] = agg.aggregate(obs.broadcasts, rid)
                    if key == "race" and rid == shown:
                        ch = agg.diagnostics.channels.get(rid, {})
                else:
                    preds[key] = obs.own.answer
            preds["debate_majority"] = MajorityVote().aggregate(
                tuple(Broadcast(j, t, piv[("debate", m)].get(t), 0.5, 0.5, False) for j, m in enumerate(LIVE_MODELS)), 0)
            for k, v in preds.items():
                per[k].append(float(v == gold[t]))
            if rid == shown:
                shown_preds = preds
            seen[rid].append(obs)
        for k in tallies:
            tallies[k].append(float(np.mean(per[k])))
        frames.append(dict(task=t, gold=gold[t], preds=shown_preds,
                           curve={k: np.cumsum(v) / np.arange(1, len(v) + 1) for k, v in tallies.items()},
                           honest={m: piv[("honest", m)].get(t) for m in LIVE_MODELS},
                           solo={m: piv[("solo", m)].get(t) for m in LIVE_MODELS},
                           debate={m: piv[("debate", m)].get(t) for m in LIVE_MODELS},
                           decisions={agents[j][1] + "|" + agents[j][0]: c.decision for j, c in ch.items() if j != shown}))
    viz.setup()
    fig = plt.figure(figsize=(16, 9), dpi=100)
    names = {"debate_majority": ("Debate, then majority of honest agents", viz.ORANGE), "majority_all": ("Majority incl. liars", viz.AQUA),
             "race": ("RACE on independent answers", viz.BLUE),
             "race_onecoin": ("RACE v3.0 (one-coin everywhere)", viz.VIOLET), "self": ("Receiver alone", viz.MUTED)}

    def card(ax, x, y, title, ans, gold_, color_edge, note=None):
        ok = ans == gold_
        ax.add_patch(FancyBboxPatch((x, y), 0.135, 0.2, boxstyle="round,pad=0.01", fc="white", ec=color_edge, lw=2))
        ax.text(x + 0.0675, y + 0.155, title, ha="center", fontsize=8.5, color=viz.INK_2)
        ax.text(x + 0.0675, y + 0.07, ans or "–", ha="center", fontsize=22, fontweight="bold", color=viz.INK)
        ax.text(x + 0.12, y + 0.16, "✓" if ok else "✗", color=viz.GREEN if ok else viz.RED, fontsize=12, fontweight="bold")
        if note:
            ax.text(x + 0.0675, y + 0.015, note, ha="center", fontsize=7.5, color=viz.RED if "→" in note else viz.INK_2)

    def draw(i):
        fr = frames[min(i, len(frames) - 1)]
        fig.clear()
        fig.text(0.02, 0.955, "A live multi-agent LLM swarm: six small open models, six covert saboteurs",
                 fontsize=17, fontweight="bold")
        fig.text(0.02, 0.925, f"{benchmark.upper()} · fresh CPU inference · saboteurs are the same models prompted to mislead · "
                              f"question {min(i, len(frames) - 1) + 1} of {len(frames)}", fontsize=11, color=viz.INK_2)
        ax = fig.add_axes([0.02, 0.06, 0.62, 0.84])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        item = questions.get(fr["task"])
        q = textwrap.shorten(item["question"], 150) if item else fr["task"]
        if benchmark == "boolq" and item:
            q = q[0].upper() + q[1:] + "?"
        ax.text(0.0, 0.97, q, fontsize=11, va="top", wrap=True)
        ax.text(0.0, 0.9, f"truth (hidden from agents): {fr['gold']}", fontsize=10, color=viz.GREEN, fontweight="bold")
        rows = (("Round 1 — honest agents answer independently", fr["honest"], viz.INK, 0.62),
                ("Covert saboteurs (same models, deceptive prompt)", fr["solo"], viz.RED, 0.34),
                ("Round 2 — honest agents after seeing everyone's votes", fr["debate"], viz.ORANGE, 0.06))
        for title, answers, edge, y in rows:
            ax.text(0.0, y + 0.225, title, fontsize=10.5, fontweight="bold", color=viz.INK)
            for k, m in enumerate(LIVE_MODELS):
                note = None
                if answers is fr["debate"] and fr["debate"][m] != fr["honest"][m]:
                    note = f"{fr['honest'][m]} → {fr['debate'][m]}"
                if answers is fr["honest"]:
                    note = {"trust": "RACE: trust", "invert": "RACE: invert", "discard": "RACE: discard"}.get(
                        fr["decisions"].get(m + "|honest"), "receiver" if m == receiver_model else None)
                if answers is fr["solo"]:
                    note = {"trust": "RACE: trust", "invert": "RACE: invert", "discard": "RACE: discard"}.get(
                        fr["decisions"].get(m + "|solo"))
                card(ax, 0.005 + k * 0.165, y, m.replace("_", "-"), answers[m], fr["gold"],
                     viz.BLUE if (answers is fr["honest"] and m == receiver_model) else edge, note)
        ax2 = fig.add_axes([0.69, 0.5, 0.29, 0.36])
        for k, (lab, col) in names.items():
            c = fr["curve"][k]
            ax2.plot(np.arange(1, len(c) + 1), 100 * c, color=col, lw=2.2 if k == "race" else 1.6, label=lab)
        ax2.set_xlim(1, len(frames))
        ax2.set_ylim(0, 100)
        ax2.set_title("running accuracy, mean over all six honest agents\n(each runs its own RACE on earlier questions only)",
                      loc="left", fontsize=9.5)
        ax2.legend(fontsize=8, loc="lower left")
        ax3 = fig.add_axes([0.69, 0.08, 0.29, 0.3])
        ax3.axis("off")
        ax3.text(0, 1.0, f"This question, for {receiver_model.replace('_', '-')}", fontsize=11, fontweight="bold", va="top")
        for n, (k, (lab, col)) in enumerate(names.items()):
            p = fr["preds"][k]
            ax3.text(0, 0.82 - 0.15 * n, f"{lab}:", fontsize=10.5, color=viz.INK_2, va="top")
            ax3.text(0.9, 0.82 - 0.15 * n, f"{p or '–'} {'✓' if p == fr['gold'] else '✗'}", fontsize=12,
                     color=viz.GREEN if p == fr["gold"] else viz.RED, fontweight="bold", va="top")

    anim = animation.FuncAnimation(fig, draw, frames=len(frames) + int(4 * fps), interval=1000 / fps)
    out = out_dir / f"live_debate_{benchmark}.mp4"
    anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=1800, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
