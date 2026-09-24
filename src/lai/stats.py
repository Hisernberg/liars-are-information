"""Paired inference with the task as the statistical unit.

Effect sizes carry a paired percentile-bootstrap 95% interval; p-values come
from the Wilcoxon signed-rank test on the same per-task differences.

Receivers, swarm seeds and peers inside a world are *not* independent samples:
they reuse the same cached answers. Every comparison therefore (1) averages a
method's accuracy over receivers (done in :mod:`lai.sim`) and over seeds within
each task, (2) pairs the two methods on the same tasks, and (3) resamples tasks.
Families of cells are corrected with Holm's step-down procedure.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def paired_bootstrap(diff: np.ndarray, resamples: int = 2000, seed: int = 0) -> tuple[float, float, float, float]:
    """(mean, 2.5%, 97.5%, two-sided bootstrap p) of a paired per-task difference."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(resamples, diff.size))
    means = diff[idx].mean(axis=1)
    point = float(diff.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    # Percentile-bootstrap p: how often the resampled mean falls on the other side of 0.
    if np.allclose(diff, 0):
        p = 1.0
    else:
        p = float(min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0))))
        p = max(p, 1.0 / resamples)
    return point, float(lo), float(hi), p


def signed_rank_p(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p (Pratt zeros). Continuous resolution, so
    Holm correction over hundreds of cells remains meaningful (a percentile
    bootstrap p cannot go below 1/resamples)."""
    from scipy.stats import wilcoxon

    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0 or np.allclose(diff, 0):
        return 1.0
    try:
        return float(wilcoxon(diff, zero_method="pratt", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values (family = the series)."""
    p = pvalues.astype(float).to_numpy()
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return pd.Series(adjusted, index=pvalues.index)


def compare(
    per_task: pd.DataFrame,
    target: str,
    baselines: list[str],
    by: list[str],
    split: str = "test",
    resamples: int = 2000,
) -> pd.DataFrame:
    """Paired target-minus-baseline differences per cell, Holm-corrected over all rows."""
    sub = per_task[(per_task.split == split) & per_task.method.isin([target, *baselines])]
    rows = []
    for key, group in sub.groupby(by):
        table = group.groupby(["task", "method"]).accuracy.mean().unstack()
        if target not in table:
            continue
        for base in baselines:
            if base not in table:
                continue
            pair = table[[target, base]].dropna()
            # Deterministic across processes (Python's str hash is salted).
            seed = int.from_bytes(hashlib.sha256(f"{key}|{base}|{target}".encode()).digest()[:4], "big")
            diff = (pair[target] - pair[base]).to_numpy()
            mean, lo, hi, _ = paired_bootstrap(diff, resamples, seed)
            p = signed_rank_p(diff)
            rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
                        | {"baseline": base, "target": target, "target_acc": pair[target].mean(),
                           "baseline_acc": pair[base].mean(), "delta": mean, "ci_low": lo, "ci_high": hi,
                           "p": p, "n_tasks": len(pair)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"])
        # A verdict needs both tests: the Holm-corrected signed-rank test (location)
        # and the paired bootstrap CI of the mean (which the tables report).
        significant = (out.p_holm < 0.05) & ((out.ci_low > 0) | (out.ci_high < 0))
        out["verdict"] = np.where(significant, np.where(out.delta > 0, "win", "loss"), "tie")
    return out


def summary(per_task: pd.DataFrame, by: list[str], split: str = "test") -> pd.DataFrame:
    """Mean accuracy per cell and method (tasks weighted equally, seeds averaged per task)."""
    sub = per_task[per_task.split == split]
    task_level = sub.groupby([*by, "method", "task"]).accuracy.mean().reset_index()
    return task_level.groupby([*by, "method"]).accuracy.mean().unstack("method")
