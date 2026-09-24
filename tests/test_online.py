"""Causality of the online protocol: the future cannot change the past."""

from __future__ import annotations

import numpy as np

from aip.types import Broadcast
from lai.online import OnlineWorld, run_online_stream

LABELS = ("A", "B", "C", "D")


def _stream(n_tasks, seed, perturb_after=None):
    rng = np.random.default_rng(seed)
    out = []
    for pos in range(n_tasks):
        g = LABELS[int(rng.integers(4))]
        wrong = [x for x in LABELS if x != g]
        row = [g if rng.random() < 0.8 else wrong[int(rng.integers(3))] for _ in range(4)]
        lie = wrong[int(rng.integers(3))]
        row += [lie] * 3
        if perturb_after is not None and pos >= perturb_after:
            row = [LABELS[(LABELS.index(a) + 1) % 4] for a in row]
            g = LABELS[(LABELS.index(g) + 1) % 4]
        tid = f"t{pos}"
        out.append((pos, pos, tid, g, "coherent", tuple(Broadcast(j, tid, a, 0.9, 0.9, False) for j, a in enumerate(row))))
    return out


def test_future_broadcasts_and_labels_cannot_change_past_predictions():
    world = OnlineWorld("mmlu", 0.3, "stationary_coherent", warmup=20, refit_every=4)
    base = run_online_stream(world, LABELS, [0, 1], _stream(70, 0))
    moved = run_online_stream(world, LABELS, [0, 1], _stream(70, 0, perturb_after=50))
    a = base[base.pos < 50].sort_values(["receiver", "pos", "method"]).reset_index(drop=True)
    b = moved[moved.pos < 50].sort_values(["receiver", "pos", "method"]).reset_index(drop=True)
    assert len(a) and a[["pred", "correct"]].equals(b[["pred", "correct"]])
    assert not base[base.pos >= 50].pred.reset_index(drop=True).equals(moved[moved.pos >= 50].pred.reset_index(drop=True))
