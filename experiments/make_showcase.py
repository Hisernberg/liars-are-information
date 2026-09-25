#!/usr/bin/env python3
"""The flagship video: a team of live LLM agents solving questions while half of the votes are lies.

Everything on screen is real data from a live six-model swarm (E10 by default, E8 as
a fallback). Six small open models answer as honest agents, and the same six models
answer again as covert saboteurs, so every question carries a 12-vote panel in which
half the votes were written to mislead. The video has seven scenes:

1. the architecture: who talks to whom, and where RACE sits inside every honest agent;
2. round 1 on one question: twelve votes arrive and majority vote picks a lie;
3. inside one honest agent: the trust it learned from earlier questions (no labels),
   and how the twelve votes add up as signed evidence;
4. round 2: plain debate against RACE-informed debate on the same panel;
5. the whole question stream: running accuracy of every protocol;
6. the interaction analysis: switching, contagion, correction, and which part of the
   architecture is responsible for each gain (or loss);
7. credits.

Every honest agent runs its own RACE, refitted on strictly earlier questions only
(predict-then-commit). Roles and the truth are drawn for the viewer; the agents never
see them. The showcase question is selected by a documented rule (majority vote wrong
on it, enough history before it); every other number uses every question.

    PYTHONPATH=src python experiments/make_showcase.py                 # E10 ARC if available
    PYTHONPATH=src python experiments/make_showcase.py --root live_cache --benchmark mmlu
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
warnings.filterwarnings("ignore")

import imageio_ffmpeg  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

from aip.types import Broadcast  # noqa: E402
from lai import viz  # noqa: E402
from lai.data import LIVE_MODELS  # noqa: E402
from lai.race import RACEAggregator  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
MEDIA = ROOT / "media"
FPS = 10
W, H = 16.0, 9.0
PRETTY = {"qwen25_1p5b": "Qwen2.5-1.5B", "smollm2_1p7b": "SmolLM2-1.7B", "granite33_2b": "Granite-3.3-2B",
          "olmo2_1b": "OLMo-2-1B", "llama32_1b": "Llama-3.2-1B", "gemma3_1b": "Gemma-3-1B"}
DEC = {"trust": ("TRUST", viz.BLUE), "discard": ("DISCARD", "#a9a8a3"), "invert": ("INVERT", viz.RED)}
NOTE = {"trust": "reliable", "discard": "no better than chance", "invert": "usually wrong"}
WARMUP = 8
# scene name -> seconds
SCENES = [("title", 5.0), ("architecture", 25.0), ("round1", 15.0), ("inside", 21.0), ("debate", 17.0),
          ("stream", 16.0), ("analysis", 17.0), ("credits", 5.0)]


# ---------------------------------------------------------------------------- data


def prepare(root: str, bench: str) -> dict:
    raw_dir = ROOT / "data" / root / "raw"
    paths = sorted((raw_dir / "stage1" / bench).glob("*.parquet")) + sorted((raw_dir / "stage2" / bench).glob("*.parquet"))
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    piv = raw.pivot_table(index="task_id", columns=["role", "model"], values="extracted_answer", aggfunc="first")
    gold = raw.drop_duplicates("task_id").set_index("task_id").gold_answer.astype(str)
    items = {it["task_id"]: it for it in json.loads((ROOT / "data" / "benchmarks" / f"{bench}_cached_items.json").read_text())}
    first = raw[(raw.role == "honest") & (raw.model == LIVE_MODELS[0])]
    order = [t for t in pd.unique(first.task_id) if t in piv.index]
    roles = set(raw.role)
    informed = "informed" in roles
    agents = [("honest", m) for m in LIVE_MODELS] + [("solo", m) for m in LIVE_MODELS]
    n_honest = len(LIVE_MODELS)

    def ans(role, m, t):
        v = piv[(role, m)].get(t) if (role, m) in piv.columns else None
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    rows, per_q = [], []
    for i, t in enumerate(order):
        votes = [ans(r, m, t) for r, m in agents]
        labels = [chr(ord("A") + k) for k in range(len(items[t]["choices"]))] if t in items else sorted(
            {v for v in votes if v})
        obs = tuple(Broadcast(j, t, v, 0.5, 0.5, False) for j, v in enumerate(votes))
        race, fits = {}, {}
        for r in range(n_honest):
            if i >= WARMUP:
                agg = RACEAggregator(labels)
                agg.fit_receiver(r, rows)
                race[r] = agg.aggregate(obs, r)
                fits[r] = agg
            else:
                race[r] = votes[r]
        counts = Counter(v for v in votes if v)
        top = max(counts.values()) if counts else 0
        leaders = sorted(c for c, n in counts.items() if n == top)
        per_q.append(dict(task=t, gold=gold[t], votes=votes, labels=labels, race=race, fits=fits,
                          majority_leaders=leaders,
                          debate=[ans("debate", m, t) for m in LIVE_MODELS],
                          informed=[ans("informed", m, t) for m in LIVE_MODELS] if informed else None,
                          rushing=[ans("rushing", m, t) for m in LIVE_MODELS]))
        rows.append({j: v for j, v in enumerate(votes)})
    return dict(bench=bench, root=root, items=items, order=order, q=per_q, agents=agents, informed=informed,
                n_honest=n_honest, raw=raw, piv=piv, gold=gold)


def majority_for(q: dict, r: int) -> str | None:
    """Plurality of the 12 votes; ties go to the receiver's own answer, else the first option."""
    own = q["votes"][r]
    return own if own in q["majority_leaders"] else (q["majority_leaders"][0] if q["majority_leaders"] else None)


def analyse(d: dict) -> dict:
    """The interaction numbers shown in scene 6 (all questions, all six honest agents)."""
    qs, n = d["q"], d["n_honest"]
    k_chance = np.mean([1 / max(len(q["labels"]), 1) for q in qs])
    ok = lambda a, g: float(a == g)  # noqa: E731
    alone = np.array([[ok(q["votes"][r], q["gold"]) for r in range(n)] for q in qs])
    maj = np.array([[ok(majority_for(q, r), q["gold"]) for r in range(n)] for q in qs])
    race = np.array([[ok(q["race"][r], q["gold"]) for r in range(n)] for q in qs])
    deb = np.array([[ok(q["debate"][r], q["gold"]) for r in range(n)] for q in qs])
    inf = np.array([[ok(q["informed"][r], q["gold"]) for r in range(n)] for q in qs]) if d["informed"] else None
    sab = np.array([[ok(q["votes"][n + r], q["gold"]) for r in range(n)] for q in qs])
    after = qs[WARMUP:]
    out = dict(n_questions=len(qs), alone=alone.mean(), majority=maj.mean(), race=race.mean(),
               race_after_warmup=race[WARMUP:].mean(), alone_after_warmup=alone[WARMUP:].mean(),
               majority_after_warmup=maj[WARMUP:].mean(), debate=deb.mean(),
               informed=inf.mean() if inf is not None else float("nan"),
               saboteur_acc=sab.mean(), chance=k_chance,
               saboteurs_below_chance=float(np.mean(sab.mean(axis=0) < k_chance)),
               per_model_gain={LIVE_MODELS[r]: float(race[WARMUP:, r].mean() - alone[WARMUP:, r].mean())
                               for r in range(n)})

    def moves(key):
        rows = []
        for q in qs:
            sab_votes = [v for v in q["votes"][n:] if v]
            lie_top = Counter(sab_votes).most_common(1)[0][0] if sab_votes else None
            for r in range(n):
                a, b = q["votes"][r], q[key][r]
                if a is None or b is None:
                    continue
                rows.append((a != b, a == q["gold"] and b != q["gold"], a != q["gold"] and b == q["gold"],
                             a != b and b == lie_top and b != q["gold"]))
        m = np.array(rows, dtype=float)
        return dict(switch=m[:, 0].mean(), right_to_wrong=m[:, 1].mean(), wrong_to_right=m[:, 2].mean(),
                    to_lie=m[:, 3].mean())

    out["sab_repeats_honest"] = float(np.mean([q["votes"][n + r] == q["votes"][r] for q in qs for r in range(n)]))
    if d["informed"]:
        fol, fol_wrong, fav_ok = [], [], []
        for q in qs:
            for r, agg in q["fits"].items():
                post = agg.posterior(tuple(Broadcast(j, "q", v, 0.5, 0.5, False) for j, v in enumerate(q["votes"])), r)
                if not post:
                    continue
                fav = max(post, key=post.get)
                fol.append(q["informed"][r] == fav)
                fav_ok.append(fav == q["gold"])
                if fav != q["gold"]:
                    fol_wrong.append(q["informed"][r] == fav)
        out["follow"], out["follow_when_wrong"], out["favoured_right"] = (float(np.mean(fol)), float(np.mean(fol_wrong)),
                                                                          float(np.mean(fav_ok)))
    out["plain"] = moves("debate")
    if d["informed"]:
        out["inf"] = moves("informed")
    # what the honest agents' RACE decided about each kind of peer at the end of the stream
    dec = Counter()
    last_i = max((i for i, q in enumerate(qs) if q["fits"]), default=None)
    trusted_true = []
    if last_i is not None:
        hist = qs[:last_i]
        for r, agg in qs[last_i]["fits"].items():
            for j, ch in agg.diagnostics.channels[r].items():
                if j == r:
                    continue
                dec[("saboteur" if j >= n else "honest", ch.decision)] += 1
                if j >= n and ch.decision == "trust":
                    true = np.mean([h["votes"][j] == h["gold"] for h in hist])
                    trusted_true.append(true > np.mean([1 / len(h["labels"]) for h in hist]))
    out["sab_trusted_true"] = float(np.mean(trusted_true)) if trusted_true else float("nan")
    for kind in ("saboteur", "honest"):
        tot = sum(v for (k, _), v in dec.items() if k == kind) or 1
        for dname in ("trust", "discard", "invert"):
            out[f"{kind}_{dname}"] = dec[(kind, dname)] / tot
    out["curves"] = dict(alone=alone.mean(axis=1), majority=maj.mean(axis=1), race=race.mean(axis=1),
                         debate=deb.mean(axis=1), informed=inf.mean(axis=1) if inf is not None else None)
    return out


def pick_showcase(d: dict) -> tuple[int, int]:
    """Receiver: the median honest model by round-1 accuracy. Question: majority vote wrong, the receiver's
    RACE right, the receiver itself wrong if possible, informed debate at least as good as plain debate,
    at least 20 questions of history."""
    n, qs = d["n_honest"], d["q"]
    acc = [np.mean([q["votes"][r] == q["gold"] for q in qs]) for r in range(n)]
    receiver = int(np.argsort(acc)[len(acc) // 2])
    best, best_score = None, -1e9
    for i, q in enumerate(qs):
        if i < 20 or not q["fits"]:
            continue
        maj_wrong = majority_for(q, receiver) != q["gold"]
        race_right = q["race"][receiver] == q["gold"]
        own_wrong = q["votes"][receiver] != q["gold"]
        n_race = sum(q["race"][r] == q["gold"] for r in range(n))
        plain = sum(a == q["gold"] for a in q["debate"])
        inf = sum(a == q["gold"] for a in q["informed"]) if q["informed"] else plain
        qtext = d["items"].get(q["task"], {}).get("question", "")
        text = len(qtext)
        score = (8 * maj_wrong + 8 * race_right + 3 * own_wrong + 2 * (inf > plain) + n_race / n - (text > 260) * 2
                 - 4 * ("\\" in qtext or "$" in qtext))
        if score > best_score:
            best, best_score = i, score
    return receiver, best


# ---------------------------------------------------------------------------- drawing helpers


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def canvas(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return ax


def header(ax, kicker: str, title: str, sub: str | None = None, a: float = 1.0):
    ax.text(0.45, 8.55, kicker.upper(), fontsize=11, color=viz.BLUE, fontweight="bold", alpha=a)
    ax.text(0.45, 8.12, title, fontsize=21, fontweight="bold", color=viz.INK, alpha=a, va="center")
    if sub:
        ax.text(0.45, 7.68, sub, fontsize=11.5, color=viz.INK_2, alpha=a, va="center")


def caption(ax, text: str, a: float = 1.0, y: float = 0.42):
    ax.add_patch(FancyBboxPatch((0.45, y - 0.3), W - 0.9, 0.62, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=viz.GRID, lw=1.2, alpha=a))
    ax.text(W / 2, y + 0.01, text, ha="center", va="center", fontsize=12.5, color=viz.INK, alpha=a)


def box(ax, x, y, w, h, fc="white", ec=None, lw=1.4, a=1.0, r=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}", fc=fc,
                                ec=ec or viz.GRID, lw=lw, alpha=a))


def card(ax, x, y, w, h, top, answer, gold, edge, a=1.0, sub=None, sub_color=None, big=20):
    box(ax, x, y, w, h, ec=edge, lw=1.8, a=a)
    ax.text(x + w / 2, y + h - 0.2, top, ha="center", va="center", fontsize=8.6, color=viz.INK_2, alpha=a)
    ax.text(x + w / 2, y + h * 0.42, answer or "–", ha="center", va="center", fontsize=big, fontweight="bold",
            color=viz.INK, alpha=a)
    if answer is not None and gold is not None:
        good = answer == gold
        ax.text(x + w - 0.2, y + h * 0.42, "✓" if good else "✗", ha="center", va="center", fontsize=13,
                color=viz.GREEN if good else viz.RED, fontweight="bold", alpha=a)
    if sub:
        ax.text(x + w / 2, y - 0.2, sub, ha="center", va="center", fontsize=8.2, color=sub_color or viz.INK_2,
                alpha=a)


def pct(x):
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------- scenes


def scene_title(fig, t, dur, d, s):
    ax = canvas(fig)
    a = ease(t / 1.0)
    ax.text(W / 2, 5.6, "A team of LLM agents, half of whose votes are lies", ha="center", fontsize=30,
            fontweight="bold", color=viz.INK, alpha=a)
    ax.text(W / 2, 4.8, "How the agents interact, where they fail, and what the RACE trust layer changes",
            ha="center", fontsize=17, color=viz.INK_2, alpha=a)
    src = "E10, the pre-registered fresh live run" if d["root"] == "live_cache_v2" else "E8, the live swarm"
    ax.text(W / 2, 3.7, f"Real answers from six live open models ({src}) · {d['bench'].upper()} · "
                        f"{len(d['q'])} questions · CPU inference, no labels at any point", ha="center", fontsize=12.5,
            color=viz.INK_2, alpha=a)
    for k, m in enumerate(LIVE_MODELS):
        x = W / 2 - 5.25 + k * 1.75
        box(ax, x, 2.3, 1.55, 0.7, ec=viz.INK_2, a=ease((t - 0.3 - 0.15 * k) / 0.6))
        ax.text(x + 0.775, 2.65, PRETTY[m], ha="center", va="center", fontsize=9.5,
                alpha=ease((t - 0.3 - 0.15 * k) / 0.6))


ARCH_STEPS = [
    ("A question arrives. Every agent answers on its own (round 1). Six are honest; six are the same models told to "
     "mislead.\nNobody is told who is who.", "agents"),
    ("Every answer is broadcast to every agent: twelve votes per question, half of them lies.", "bus"),
    ("Each honest agent keeps its own unlabelled history: who said what on every earlier question. "
     "There is no answer key, ever.", "history"),
    ("RACE fits a latent-truth model to that history, anchored on the one fact the agent knows: that it is itself "
     "honest.\nWithout the anchor, the model would side with whichever group is larger (label switching).", "em"),
    ("Copies are counted once: replicas of one model, or a coordinated bloc, form a clone group (clone tempering).",
     "clone"),
    ("Every peer gets a signed weight λ. TRUST: its answer is evidence for itself. INVERT: its answer is evidence "
     "against itself.\nDISCARD: it carries no information. A reliable liar is therefore useful.", "weights"),
    ("The agent answers with the option that has the most weighted evidence. The same reliability notes can be "
     "written into\na second, debate round (RACE-informed debate).", "decision"),
]


def scene_architecture(fig, t, dur, d, s):
    ax = canvas(fig)
    header(ax, "1 · The architecture", "Who talks to whom, and where RACE sits",
           "A question stream, twelve agents, one broadcast channel, and a RACE trust layer inside every honest agent")
    step_len = dur / len(ARCH_STEPS)
    step = min(int(t / step_len), len(ARCH_STEPS) - 1)
    local = (t - step * step_len) / step_len
    active = ARCH_STEPS[step][1]
    # question stream
    box(ax, 0.45, 3.3, 2.1, 2.4, ec=viz.INK_2)
    ax.text(1.5, 5.35, "Question stream", ha="center", fontsize=11, fontweight="bold")
    for k in range(3):
        box(ax, 0.75 + 0.12 * k, 3.65 + 0.12 * k, 1.4, 0.95, ec=viz.GRID, fc="#f6f5f2")
    ax.text(1.62, 4.1 + 0.24, "Q", ha="center", va="center", fontsize=22, fontweight="bold", color=viz.INK_2)
    ax.text(1.5, 3.05, f"{d['bench'].upper()}, one at a time", ha="center", fontsize=9, color=viz.INK_2)
    # agents
    ys = np.linspace(7.05, 1.25, 12)
    for j, (role, m) in enumerate(d["agents"]):
        hi = active == "agents"
        col = viz.RED if role == "solo" else viz.INK_2
        box(ax, 3.3, ys[j] - 0.21, 2.6, 0.42, ec=col, lw=1.8 if hi else 1.1)
        ax.text(3.45, ys[j], ("Saboteur · " if role == "solo" else "Honest · ") + PRETTY[m], va="center",
                fontsize=8.8, color=col)
        ax.plot([2.6, 3.3], [4.5, ys[j]], color=viz.GRID, lw=0.8, zorder=0)
        ax.plot([5.9, 6.75], [ys[j], ys[j]], color=viz.GRID, lw=0.8, zorder=0)
    ax.text(4.6, 7.37, "12 agents (roles shown to you only)", ha="center", fontsize=9, color=viz.INK_2)
    # broadcast bus
    bus_hi = active == "bus"
    box(ax, 6.75, 1.0, 0.5, 6.3, fc=viz.SEQ_BLUE[0] if bus_hi else "#f3f2ee", ec=viz.BLUE if bus_hi else viz.GRID,
        lw=2 if bus_hi else 1.2)
    ax.text(7.0, 4.15, "broadcast: every answer to every agent", rotation=90, ha="center", va="center", fontsize=9.5,
            color=viz.INK_2)
    if active in ("agents", "bus"):
        for j in range(12):
            ph = (local * 2.2 + j * 0.07) % 1.0
            if active == "agents":
                x = 2.6 + ph * 0.7
                y = 4.5 + ph * (ys[j] - 4.5)
            else:
                x = 5.9 + ph * 0.85
                y = ys[j]
            ax.add_patch(Circle((x, y), 0.06, color=viz.RED if j >= 6 else viz.BLUE, zorder=5))
    # RACE box
    box(ax, 7.75, 0.95, 7.8, 6.35, ec=viz.BLUE, lw=2.0)
    ax.text(8.0, 6.95, "Inside every honest agent: the RACE trust layer", fontsize=12.5, fontweight="bold",
            color=viz.BLUE)
    blocks = [("history", "Unlabelled history", "who said what, on every earlier question"),
              ("em", "Anchored latent-truth EM", "'I am honest': start, prior and floor on its own accuracy"),
              ("clone", "Clone tempering", "near-identical agents count once"),
              ("weights", "Signed weights λ per peer", "TRUST  ·  DISCARD  ·  INVERT"),
              ("decision", "Decision", "argmax of the weighted evidence")]
    for k, (key, name, desc) in enumerate(blocks):
        y = 5.85 - k * 1.05
        hi = key == active
        box(ax, 8.1, y, 4.9, 0.82, fc=viz.SEQ_BLUE[0] if hi else "white", ec=viz.BLUE if hi else viz.GRID,
            lw=2.2 if hi else 1.1)
        ax.text(8.3, y + 0.55, f"{k + 1}. {name}", fontsize=11, fontweight="bold",
                color=viz.INK if hi else viz.INK_2)
        ax.text(8.3, y + 0.22, desc, fontsize=9.2, color=viz.INK_2)
        if k < len(blocks) - 1:
            ax.annotate("", (10.55, y - 0.2), (10.55, y + 0.02), arrowprops=dict(arrowstyle="->", color=viz.MUTED))
    ax.annotate("", (8.1, 6.25), (7.25, 6.25), arrowprops=dict(arrowstyle="->", color=viz.BLUE, lw=1.6))
    # weight illustration
    wx = 13.3
    ax.text(wx, 6.6, "what λ does", fontsize=9.5, color=viz.INK_2, fontweight="bold")
    for k, (dname, lab) in enumerate((("trust", "honest peer"), ("discard", "random peer"), ("invert", "reliable liar"))):
        y = 6.05 - k * 0.78
        name, col = DEC[dname]
        ax.text(wx, y + 0.2, lab, fontsize=8.6, va="center", color=viz.INK_2)
        ax.text(15.4, y + 0.2, name, fontsize=8.6, va="center", ha="right", color=col, fontweight="bold")
        length = (0.95, 0.08, -0.95)[k]
        if active == "weights":
            length *= ease(local * 2)
        x0 = 14.35
        ax.add_patch(plt.Rectangle((x0 if length >= 0 else x0 + length, y - 0.24), abs(length), 0.22, color=col))
        ax.plot([x0, x0], [y - 0.3, y + 0.04], color=viz.INK_2, lw=0.8)
    # debate loop
    hi = active == "decision"
    box(ax, 13.2, 1.2, 2.2, 1.55, fc=viz.SEQ_BLUE[0] if hi else "white", ec=viz.ORANGE if hi else viz.GRID)
    ax.text(14.3, 2.35, "optional round 2", ha="center", fontsize=9.5, fontweight="bold", color=viz.ORANGE)
    ax.text(14.3, 1.85, "debate prompt with\nreliability notes", ha="center", va="center", fontsize=8.8,
            color=viz.INK_2)
    ax.annotate("", (13.2, 1.95), (13.0, 1.95), arrowprops=dict(arrowstyle="<-", color=viz.ORANGE))
    ax.text(W - 0.45, 7.45, f"step {step + 1} / {len(ARCH_STEPS)}", ha="right", fontsize=10, color=viz.INK_2)
    caption(ax, ARCH_STEPS[step][0], a=ease(local * 4))


def question_block(ax, d, q, a=1.0, y=7.25, width=118):
    item = d["items"].get(q["task"], {})
    text = item.get("question", q["task"])
    if d["bench"] == "boolq":
        text = text[0].upper() + text[1:] + "?"
    wrapped = textwrap.wrap(text, width)
    ax.text(0.45, y, "\n".join(wrapped[:2]) + (" …" if len(wrapped) > 2 else ""), fontsize=12.5, va="top", alpha=a,
            color=viz.INK)
    opts = item.get("choices", [])
    for k, c in enumerate(opts[:6]):
        if len(opts) > 2:
            x, yy = 0.45 + (k % 2) * 7.6, y - 0.95 - (k // 2) * 0.36
        else:
            x, yy = 0.45 + k * 3.0, y - 0.95
        ax.text(x, yy, f"{chr(65 + k)}.  {textwrap.shorten(str(c), 70)}", fontsize=10.5, color=viz.INK_2, alpha=a,
                va="top")


def scene_round1(fig, t, dur, d, s):
    ax = canvas(fig)
    q = d["q"][s["i"]]
    header(ax, "2 · Round 1", f"Question {s['i'] + 1} of {len(d['q'])}: twelve independent answers",
           None)
    question_block(ax, d, q)
    ax.text(W - 0.45, 8.12, f"truth (hidden from the agents): {q['gold']}", ha="right", fontsize=12, color=viz.GREEN,
            fontweight="bold", va="center")
    reveal = t / (dur * 0.55)
    n = d["n_honest"]
    order = [j for pair in zip(range(n), range(n, 2 * n), strict=True) for j in pair]
    shown = order[: int(np.clip(reveal, 0, 1) * len(order) + 0.999)] if t > 0.3 else []
    ax.text(0.45, 4.75, "Honest agents", fontsize=11, fontweight="bold")
    ax.text(0.45, 2.75, "Saboteurs: the same models, told to mislead (hidden from the others)", fontsize=11,
            fontweight="bold", color=viz.RED)
    for j, (role, m) in enumerate(d["agents"]):
        k = j % n
        x, y = 0.45 + k * 1.72, (3.35 if role == "honest" else 1.35)
        vis = j in shown
        card(ax, x, y, 1.55, 1.25, PRETTY[m], q["votes"][j] if vis else " ", q["gold"] if vis else None,
             viz.RED if role == "solo" else (viz.BLUE if k == s["r"] else viz.INK_2), a=1.0 if vis else 0.35,
             sub="we follow this agent next" if (role == "honest" and k == s["r"]) else None, sub_color=viz.BLUE)
    # tally
    ax.text(11.2, 4.75, "Votes so far", fontsize=11, fontweight="bold")
    cnt = Counter(q["votes"][j] for j in shown if q["votes"][j])
    labels = q["labels"]
    for k, lab in enumerate(labels):
        y = 4.25 - k * 0.55
        ax.text(11.2, y, lab, fontsize=12, fontweight="bold", va="center",
                color=viz.GREEN if lab == q["gold"] else viz.INK)
        v = cnt.get(lab, 0)
        ax.add_patch(plt.Rectangle((11.55, y - 0.17), 0.33 * v, 0.34,
                                   color=viz.GREEN if lab == q["gold"] else viz.MUTED))
        ax.text(11.65 + 0.33 * v, y, str(v), fontsize=10, va="center", color=viz.INK_2)
    if reveal >= 1.0:
        mv = majority_for(q, s["r"])
        a = ease((t - dur * 0.55) / 1.0)
        box(ax, 11.1, 1.3, 4.45, 0.9, ec=viz.RED if mv != q["gold"] else viz.GREEN, lw=2, a=a)
        ax.text(13.33, 1.75, f"Majority vote picks {mv}  {'✓' if mv == q['gold'] else '✗'}", ha="center",
                va="center", fontsize=15, fontweight="bold", color=viz.RED if mv != q["gold"] else viz.GREEN, alpha=a)
        n_ok = sum(v == q["gold"] for v in q["votes"])
        caption(ax, f"Only {n_ok} of 12 votes are right. A team that simply counts votes answers {mv}. "
                    f"Can the honest agents do better without knowing who lies?", a=a)
    else:
        caption(ax, "Every agent answers without seeing the others. The votes are then broadcast to everyone.")


def scene_inside(fig, t, dur, d, s):
    ax = canvas(fig)
    q = d["q"][s["i"]]
    r = s["r"]
    agg = q["fits"][r]
    ch = agg.diagnostics.channels[r]
    header(ax, "3 · Inside one honest agent", f"What {PRETTY[LIVE_MODELS[r]]} learned, and how it weighs the twelve votes",
           f"Its RACE was fitted on the {s['i']} earlier questions only, from the panel's answers alone: no labels")
    # left: trust table
    ax.text(0.45, 7.1, "Learned reliability of every panelist (bars) vs the truth (ticks, shown to you only)",
            fontsize=10.5, fontweight="bold")
    n = d["n_honest"]
    earlier = d["q"][: s["i"]]
    x0, x1 = 3.55, 7.25
    ax.plot([x0 + (x1 - x0) * np.mean([1 / len(e["labels"]) for e in earlier])] * 2, [0.95, 6.8], color=viz.MUTED,
            ls=":", lw=1)
    ax.text(x0 + (x1 - x0) * np.mean([1 / len(e["labels"]) for e in earlier]), 6.85, "chance", fontsize=8,
            color=viz.MUTED, ha="center")
    grow = ease(t / 3.0)
    for j, (role, m) in enumerate(d["agents"]):
        y = 6.5 - j * 0.47
        lab = ("Saboteur · " if role == "solo" else "Honest · ") + PRETTY[m] + (" (itself)" if j == r else "")
        ax.text(0.45, y, lab, fontsize=8.8, va="center", color=viz.RED if role == "solo" else viz.INK)
        c = ch.get(j)
        if c is None:
            continue
        dname = c.decision if j != r else "trust"
        ax.add_patch(plt.Rectangle((x0, y - 0.15), (x1 - x0) * c.accuracy * grow, 0.3, color=DEC[dname][1]))
        true = np.mean([e["votes"][j] == e["gold"] for e in earlier])
        ax.plot([x0 + (x1 - x0) * true] * 2, [y - 0.2, y + 0.2], color=viz.INK, lw=2.2)
        ax.text(x1 + 0.12, y, "self" if j == r else DEC[dname][0], fontsize=8.3, va="center",
                color=viz.INK_2 if j == r else DEC[dname][1], fontweight="bold", alpha=grow)
    ax.text(x0, 0.72, "0", fontsize=8, color=viz.INK_2, ha="center")
    ax.text(x1, 0.72, "1", fontsize=8, color=viz.INK_2, ha="center")
    ax.text((x0 + x1) / 2, 0.72, "estimated accuracy", fontsize=8.5, color=viz.INK_2, ha="center")
    # right: evidence build-up
    obs = tuple(Broadcast(j, q["task"], v, 0.5, 0.5, False) for j, v in enumerate(q["votes"]))
    ev = agg.evidence(obs, r)
    cands = sorted({v for v in q["votes"] if v})
    ex0 = 11.45
    ax.text(8.75, 7.1, "The twelve votes as signed evidence (log-odds), added one by one", fontsize=10.5,
            fontweight="bold")
    items = sorted((k for k in ev if k >= 0), key=lambda k: -abs(ev[k][1]))
    extra = [k for k in ev if k < 0]
    shown_n = int(np.clip((t - 2.5) / (dur * 0.5), 0, 1) * len(items) + 0.999) if t > 2.5 else 0
    shown = items[:shown_n] + (extra if shown_n == len(items) else [])
    scale = 0.9 / max(1.0, max(abs(v) for _, v in ev.values()))
    tot = {c: 0.0 for c in cands}
    for k in shown:
        tot[ev[k][0]] += ev[k][1]
    for ci, c in enumerate(cands):
        y = 6.2 - ci * 1.1
        ax.text(8.75, y, c, fontsize=16, fontweight="bold", va="center", color=viz.GREEN if c == q["gold"] else viz.INK)
        ax.text(9.15, y, f"{sum(1 for v in q['votes'] if v == c)} votes", fontsize=8.5, va="center", color=viz.INK_2)
        ax.plot([ex0, ex0], [y - 0.35, y + 0.35], color=viz.INK_2, lw=1)
        pos, neg = ex0, ex0
        for k in shown:
            a_, v = ev[k]
            if a_ != c:
                continue
            wdt = v * scale
            col = viz.BLUE if v > 0 else viz.RED
            if k < 0:
                col = viz.MUTED
            if v >= 0:
                ax.add_patch(plt.Rectangle((pos, y - 0.2), wdt, 0.4, color=col, ec="white", lw=0.8))
                pos += wdt
            else:
                ax.add_patch(plt.Rectangle((neg + wdt, y - 0.2), -wdt, 0.4, color=col, ec="white", lw=0.8))
                neg += wdt
        ax.text(min(max(pos, ex0) + 0.1, 15.2), y, f"{tot[c]:+.1f}", fontsize=10, va="center", fontweight="bold",
                color=viz.INK)
    ax.text(ex0 - 0.05, 6.2 + 0.55, "← against", fontsize=8, color=viz.RED, ha="right")
    ax.text(ex0 + 0.05, 6.2 + 0.55, "for →", fontsize=8, color=viz.BLUE)
    done = t > 2.5 + dur * 0.5
    if done:
        a = ease((t - 2.5 - dur * 0.5) / 1.0)
        pick = q["race"][r]
        mv = majority_for(q, r)
        box(ax, 8.75, 1.35, 6.8, 0.95, ec=viz.GREEN if pick == q["gold"] else viz.RED, lw=2, a=a)
        ax.text(12.15, 1.82, f"RACE picks {pick} {'✓' if pick == q['gold'] else '✗'}     (majority: {mv} "
                             f"{'✓' if mv == q['gold'] else '✗'};  its own answer: {q['votes'][r]} "
                             f"{'✓' if q['votes'][r] == q['gold'] else '✗'})", ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=viz.INK, alpha=a)
        n_ok = sum(q["race"][k] == q["gold"] for k in range(n))
        caption(ax, f"Red segments are votes read backwards: a panelist that is usually wrong makes its answer less "
                    f"likely.\nEach of the six honest agents runs its own RACE; {n_ok} of 6 answer correctly here.",
                a=a)
    else:
        caption(ax, "Blue: TRUST (the vote supports its answer). Red: INVERT (the vote counts against its answer). "
                    "Grey: DISCARD.")


def scene_debate(fig, t, dur, d, s):
    ax = canvas(fig)
    q = d["q"][s["i"]]
    n = d["n_honest"]
    header(ax, "4 · Round 2: the agents talk", "Plain debate versus RACE-informed debate, on the same twelve votes",
           "Each honest model sees the round-1 panel (anonymised, shuffled) and answers again")
    ax.text(W - 0.45, 8.12, f"truth: {q['gold']}", ha="right", fontsize=12, color=viz.GREEN, fontweight="bold",
            va="center")
    r = s["r"]
    rng = np.random.default_rng(s["i"])
    order = rng.permutation(2 * n)
    ch = q["fits"][r].diagnostics.channels[r] if q["fits"] else {}
    panels = (("Plain debate", "debate", viz.ORANGE, 0.45), ("RACE-informed debate", "informed", viz.BLUE, 8.2))
    for title, key, col, x in panels:
        a = ease((t - (0 if key == "debate" else dur * 0.35)) / 1.0)
        box(ax, x, 1.2, 7.35, 6.1, ec=col, lw=2, a=a)
        ax.text(x + 0.25, 6.95, title, fontsize=14, fontweight="bold", color=col, alpha=a)
        ax.text(x + 0.25, 6.55, f"What {PRETTY[LIVE_MODELS[r]]} is shown:", fontsize=9.5, color=viz.INK_2, alpha=a)
        for k, j in enumerate(order[:6]):
            v = q["votes"][j]
            note = ""
            if key == "informed":
                note = f"  ({NOTE[ch[j].decision]})" if j in ch and j != r else "  (its own vote)" if j == r else ""
            ax.text(x + 0.35, 6.2 - k * 0.3, f"Panelist {k + 1}: {v}{note}", fontsize=9.3, family="DejaVu Sans Mono",
                    color=(DEC[ch[j].decision][1] if key == "informed" and j in ch and j != r else viz.INK), alpha=a)
        ax.text(x + 0.35, 6.2 - 6 * 0.3, "… and six more", fontsize=9, color=viz.INK_2, alpha=a)
        if key == "informed" and not d["informed"]:
            ax.text(x + 3.7, 3.2, "not run in this swarm (E10 only)", ha="center", fontsize=12, color=viz.MUTED,
                    alpha=a)
            continue
        if key == "informed":
            post = q["fits"][r].posterior(tuple(Broadcast(j, "q", v, 0.5, 0.5, False) for j, v in enumerate(q["votes"])), r)
            fav = max(post, key=post.get) if post else None
            ax.text(x + 0.35, 6.2 - 7 * 0.3, f"Weighing each vote by its reliability favours option {fav}.",
                    fontsize=9.3, family="DejaVu Sans Mono", color=viz.BLUE, alpha=a)
        after = q[key]
        ax.text(x + 0.25, 3.35, "Round 1 → round 2, the six honest agents:", fontsize=9.8, fontweight="bold", alpha=a)
        for k, m in enumerate(LIVE_MODELS):
            b0, b1 = q["votes"][k], after[k]
            cx = x + 0.25 + k * 1.18
            card(ax, cx, 1.55, 1.05, 1.45, PRETTY[m].split("-")[0], b1, q["gold"], col, a=a,
                 sub=f"{b0} → {b1}" if b0 != b1 else "kept", big=17)
        ok0 = sum(v == q["gold"] for v in q["votes"][:n])
        ok1 = sum(v == q["gold"] for v in after)
        ax.text(x + 7.1, 6.95, f"correct: {ok0}/6 → {ok1}/6", ha="right", fontsize=11.5, fontweight="bold",
                color=viz.GREEN if ok1 > ok0 else (viz.RED if ok1 < ok0 else viz.INK_2), alpha=a)
    caption(ax, "The reliability notes come from each agent's own RACE fit on earlier questions. No answer key is "
                "used, and the first 8 questions say 'no track record yet'.", a=ease((t - dur * 0.5) / 1.0))


def scene_stream(fig, t, dur, d, s):
    ax0 = canvas(fig)
    A = s["analysis"]
    header(ax0, "5 · The whole question stream", f"Running accuracy of the honest agents over all {len(d['q'])} questions",
           "Mean over the six honest models; RACE is refitted before every question on the earlier ones only")
    ax = fig.add_axes([0.07, 0.14, 0.56, 0.62])
    c = A["curves"]
    n_q = len(d["q"])
    upto = max(2, int(np.clip(t / (dur * 0.7), 0, 1) * n_q))
    xs = np.arange(1, n_q + 1)
    series = [("alone", "Each agent alone (round 1)", viz.MUTED, 1.8), ("majority", "Majority of the 12 votes", viz.AQUA, 1.8),
              ("debate", "After plain debate", viz.ORANGE, 1.8)]
    if c["informed"] is not None:
        series.append(("informed", "After RACE-informed debate", viz.VIOLET, 2.0))
    series.append(("race", "RACE on round-1 votes", viz.BLUE, 2.8))
    ends = []
    for key, lab, col, lw in series:
        run = np.cumsum(c[key]) / xs
        ax.plot(xs[:upto], 100 * run[:upto], color=col, lw=lw, label=lab)
        ends.append([100 * run[-1], 100 * run[-1], col])
    if upto == n_q:  # end labels, nudged apart so they never overlap
        ends.sort(key=lambda e: e[0])
        for k in range(1, len(ends)):
            ends[k][1] = max(ends[k][1], ends[k - 1][1] + 3.2)
        for val, ypos, col in ends:
            ax.text(n_q + 1.5, ypos, f"{val:.1f}%", fontsize=9.5, va="center", color=col, fontweight="bold")
    ax.axvspan(0.5, WARMUP + 0.5, color=viz.GRID, alpha=0.7, lw=0)
    ax.text(WARMUP / 2 + 0.5, 4, "warm-up", fontsize=8, ha="center", color=viz.INK_2)
    ax.set_xlim(1, n_q + 8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("question")
    ax.set_ylabel("running accuracy (%)")
    ax.legend(loc="lower right", fontsize=9)
    # per-model gains
    ax2 = fig.add_axes([0.72, 0.2, 0.25, 0.5])
    g = A["per_model_gain"]
    a = ease((t - dur * 0.7) / 1.5)
    models = list(g)
    vals = [100 * g[m] * a for m in models]
    ax2.barh(range(len(models)), vals, color=[viz.BLUE if v >= 0 else viz.RED for v in vals], height=0.6)
    ax2.axvline(0, color=viz.INK_2, lw=1)
    ax2.set_yticks(range(len(models)), [PRETTY[m] for m in models], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_title("RACE minus alone, per model\n(points, after warm-up)", fontsize=10, loc="left")
    lim = max(5, max(abs(100 * v) for v in g.values()) * 1.25)
    ax2.set_xlim(-lim, lim)
    for k, v in enumerate(vals):
        if a > 0.9:
            ax2.text(v + (0.5 if v >= 0 else -0.5), k, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
                     fontsize=9)
    ax2.grid(axis="y", visible=False)


def scene_analysis(fig, t, dur, d, s):
    ax = canvas(fig)
    A = s["analysis"]
    header(ax, "6 · Analysis", "What the interaction does, and which part of the architecture is responsible",
           f"All {A['n_questions']} questions, all six honest agents · {d['bench'].upper()}")
    a1, a2 = ease(t / 1.2), ease((t - dur * 0.35) / 1.2)
    box(ax, 0.45, 1.2, 7.3, 6.1, ec=viz.GRID, a=a1)
    ax.text(0.75, 6.9, "What the agents did", fontsize=14, fontweight="bold", alpha=a1)
    plain = A["plain"]
    lines = [
        ("Honest agent alone", pct(A["alone"])),
        ("Saboteur answers that are right", f"{pct(A['saboteur_acc'])}  (chance {pct(A['chance'])})"),
        ("Saboteur channels below chance", f"{100 * A['saboteurs_below_chance']:.0f}% of 6"),
        ("Saboteur repeats its own honest answer", pct(A["sab_repeats_honest"])),
        ("Majority of the 12 votes", pct(A["majority"])),
        ("Plain debate: answers changed", pct(plain["switch"])),
        ("   right → wrong  /  wrong → right", f"{pct(plain['right_to_wrong'])}  /  {pct(plain['wrong_to_right'])}"),
        ("   moved to the saboteurs' favourite lie", pct(plain["to_lie"])),
    ]
    if "inf" in A:
        inf = A["inf"]
        lines += [("Informed debate: answers changed", pct(inf["switch"])),
                  ("   right → wrong  /  wrong → right", f"{pct(inf['right_to_wrong'])}  /  {pct(inf['wrong_to_right'])}"),
                  ("   moved to the saboteurs' favourite lie", pct(inf["to_lie"]))]
    for k, (lab, val) in enumerate(lines):
        y = 6.4 - k * 0.445
        ax.text(0.75, y, lab, fontsize=10.5, color=viz.INK_2, alpha=a1, va="center")
        ax.text(7.45, y, val, fontsize=11, fontweight="bold", color=viz.INK, alpha=a1, va="center", ha="right")
    box(ax, 8.2, 1.2, 7.35, 6.1, ec=viz.BLUE, lw=1.8, a=a2)
    ax.text(8.5, 6.9, "How the architecture helps (and where it does not)", fontsize=14, fontweight="bold",
            color=viz.BLUE, alpha=a2)
    gain = A["race_after_warmup"] - A["alone_after_warmup"]
    gains = list(A["per_model_gain"].values())
    items = [
        ("Anchor: no honest-majority assumption",
         f"after the 8-question warm-up: RACE {pct(A['race_after_warmup'])}, majority {pct(A['majority_after_warmup'])},"
         f" alone {pct(A['alone_after_warmup'])}"),
        ("Signed weights: trust, ignore or read backwards",
         f"saboteur links: {100 * A['saboteur_invert']:.0f}% INVERT, {100 * A['saboteur_discard']:.0f}% DISCARD, "
         f"{100 * A['saboteur_trust']:.0f}% TRUST; honest links: {100 * A['honest_trust']:.0f}% TRUST"
         + (f"\nthe trusted saboteurs really are right more often than chance ({100 * A['sab_trusted_true']:.0f}% of them)"
            if A["saboteur_trust"] > 0 else "")),
        ("Each agent learns on its own, online",
         f"{gain * 100:+.1f} points over alone on average; per model {min(gains) * 100:+.1f} to {max(gains) * 100:+.1f}"),
    ]
    if "inf" in A:
        dlt = A["informed"] - A["debate"]
        verdict = "helps" if dlt > 0.005 else ("hurts" if dlt < -0.005 else "makes no difference")
        items.append(("Reliability notes in the debate prompt",
                      f"informed {pct(A['informed'])} vs plain {pct(A['debate'])}: it {verdict} ({dlt * 100:+.1f}). But agents "
                      f"mostly defer:\nthey adopt RACE's suggestion {100 * A['follow']:.0f}% of the time, "
                      f"{100 * A['follow_when_wrong']:.0f}% even when it is wrong"))
    best_debate = max(A["debate"], A["informed"] if "inf" in A else 0)
    items.append(("So: record independent answers, then pool them with RACE",
                  f"RACE on round-1 votes {pct(A['race_after_warmup'])} (after warm-up) vs the best debate protocol "
                  f"{pct(best_debate)}"))
    for k, (head, body) in enumerate(items):
        y = 6.25 - k * 1.05
        ax.text(8.5, y, f"{k + 1}. {head}", fontsize=11.2, fontweight="bold", alpha=a2)
        ax.text(8.8, y - 0.38, body, fontsize=9.8, color=viz.INK_2, alpha=a2)
    caption(ax, "Every number on this slide is recomputed from the released answers "
                f"(data/{d['root']}/) by experiments/make_showcase.py.", a=ease((t - dur * 0.6) / 1.0))


def scene_credits(fig, t, dur, d, s):
    ax = canvas(fig)
    a = ease(t / 1.0)
    ax.text(W / 2, 5.6, "Liars are information", ha="center", fontsize=30, fontweight="bold", alpha=a)
    ax.text(W / 2, 4.8, "RACE: Receiver-Anchored Channel Estimation for multi-agent LLM systems", ha="center",
            fontsize=15, color=viz.INK_2, alpha=a)
    ax.text(W / 2, 3.8, "Code, data, paper and every video:  github.com/Hisernberg/liars-are-information", ha="center",
            fontsize=13, color=viz.BLUE, alpha=a)
    ax.text(W / 2, 3.3, "Dataset and demo:  huggingface.co/datasets/Nabidnur/liars-are-information", ha="center",
            fontsize=13, color=viz.BLUE, alpha=a)
    ax.text(W / 2, 2.4, "Builds on the AIP study and artifact by Dhruv Jyoti Das (Dhruv1000/Liars_Are_Information).",
            ha="center", fontsize=11, color=viz.INK_2, alpha=a)


DRAW = {"title": scene_title, "architecture": scene_architecture, "round1": scene_round1, "inside": scene_inside,
        "debate": scene_debate, "stream": scene_stream, "analysis": scene_analysis, "credits": scene_credits}


def render(d: dict, out: Path, only: list[str] | None = None) -> dict:
    viz.setup()
    receiver, i = pick_showcase(d)
    A = analyse(d)
    s = dict(r=receiver, i=i, analysis=A)
    scenes = [(n, dur) for n, dur in SCENES if not only or n in only]
    starts = np.cumsum([0] + [dur for _, dur in scenes])
    total = int(starts[-1] * FPS)
    fig = plt.figure(figsize=(W, H), dpi=110)

    def update(frame):
        tt = frame / FPS
        k = int(np.searchsorted(starts, tt, side="right") - 1)
        k = min(k, len(scenes) - 1)
        fig.clear()
        fig.patch.set_facecolor(viz.SURFACE)
        name, dur = scenes[k]
        DRAW[name](fig, tt - starts[k], dur, d, s)
        return []

    anim = animation.FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=animation.FFMpegWriter(fps=FPS, bitrate=3000, codec="libx264",
                                                 extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    plt.close(fig)
    timing = {n: (float(starts[k]), float(dur)) for k, (n, dur) in enumerate(scenes)}
    return dict(receiver=LIVE_MODELS[receiver], question_index=i, task=d["q"][i]["task"], timing=timing, analysis=A)


def summary_json(d: dict, meta: dict) -> dict:
    A = meta["analysis"]
    keep = {k: v for k, v in A.items() if k not in ("curves", "per_model_gain", "plain", "inf")}
    keep |= {f"plain_{k}": v for k, v in A["plain"].items()}
    keep |= {f"informed_{k}": v for k, v in A.get("inf", {}).items()}
    keep["per_model_gain"] = A["per_model_gain"]
    return dict(source=d["root"], benchmark=d["bench"], receiver=meta["receiver"], task=meta["task"],
                question_index=meta["question_index"], timing=meta["timing"],
                numbers={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in keep.items()})


def macros(d: dict, meta: dict) -> dict:
    """README macros (``{{showX}}``) for the interaction analysis, as formatted strings."""
    A = meta["analysis"]
    f = lambda x: f"{100 * x:.1f}"  # noqa: E731
    out = {"showBench": {"arc": "ARC", "mmlu": "MMLU", "boolq": "BoolQ"}.get(d["bench"], d["bench"]),
           "showSource": "E10" if d["root"] == "live_cache_v2" else "E8", "showQuestions": str(A["n_questions"]),
           "showReceiver": PRETTY[meta["receiver"]], "showAlone": f(A["alone"]), "showMajority": f(A["majority"]),
           "showRace": f(A["race"]), "showAloneLate": f(A["alone_after_warmup"]),
           "showMajLate": f(A["majority_after_warmup"]), "showRaceLate": f(A["race_after_warmup"]),
           "showDebate": f(A["debate"]), "showSabAcc": f(A["saboteur_acc"]), "showChance": f(A["chance"]),
           "showSabBelow": f"{100 * A['saboteurs_below_chance']:.0f}",
           "showSabInvert": f"{100 * A['saboteur_invert']:.0f}", "showSabDiscard": f"{100 * A['saboteur_discard']:.0f}",
           "showSabTrust": f"{100 * A['saboteur_trust']:.0f}", "showHonestTrust": f"{100 * A['honest_trust']:.0f}",
           "showGainMin": f"{100 * min(A['per_model_gain'].values()):+.1f}",
           "showGainMax": f"{100 * max(A['per_model_gain'].values()):+.1f}",
           "showPlainSwitch": f(A["plain"]["switch"]), "showPlainRW": f(A["plain"]["right_to_wrong"]),
           "showPlainWR": f(A["plain"]["wrong_to_right"]), "showPlainToLie": f(A["plain"]["to_lie"])}
    out["showSabRepeats"] = f(A["sab_repeats_honest"])
    if "inf" in A:
        out |= {"showFollow": f"{100 * A['follow']:.0f}", "showFollowWrong": f"{100 * A['follow_when_wrong']:.0f}",
                "showFavRight": f(A["favoured_right"])}
        out |= {"showInformed": f(A["informed"]), "showInfSwitch": f(A["inf"]["switch"]),
                "showInfRW": f(A["inf"]["right_to_wrong"]), "showInfWR": f(A["inf"]["wrong_to_right"]),
                "showInfToLie": f(A["inf"]["to_lie"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="live_cache_v2 (E10, default if present) or live_cache (E8)")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="render only these scenes (for previews)")
    ap.add_argument("--out", default=str(MEDIA / "multi_agent_showcase.mp4"))
    args = ap.parse_args()
    v2 = (ROOT / "data" / "live_cache_v2" / "raw" / "stage2").exists() and any(
        "informed" in set(pd.read_parquet(p).role) for p in (ROOT / "data" / "live_cache_v2" / "raw" / "stage2").rglob("*.parquet"))
    root = args.root or ("live_cache_v2" if v2 else "live_cache")
    bench = args.benchmark or ("arc" if root == "live_cache_v2" else "mmlu")
    d = prepare(root, bench)
    meta = render(d, Path(args.out), args.only)
    if not args.only:
        info = summary_json(d, meta)
        (ROOT / "results" / "tables" / "showcase.json").write_text(json.dumps(info, indent=1))
        (ROOT / "results" / "tables" / "showcase_macros.json").write_text(json.dumps(macros(d, meta), indent=1))
        from make_gifs import gif

        for name, (start, dur) in meta["timing"].items():
            if name in ("architecture", "inside", "debate"):
                gif(Path(args.out), MEDIA / "gif" / f"showcase_{name}.gif", start, dur, 5, 800)
        print(json.dumps({k: v for k, v in info.items() if k != "numbers"}, indent=1))
    print(args.out)


if __name__ == "__main__":
    main()
