"""Bootstrap confidence intervals.

Design principle 3: no number in this project is reported without a CI, a logged
seed, and a stated sample size.

Resampling is over **tasks**, not over observations, because the paired
statistics here (agreement between two agents on the same question) are only
exchangeable at the task level. Resampling observations independently would
break the pairing and understate the interval.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 2000


@dataclass(frozen=True, slots=True)
class CI:
    """A point estimate with a percentile bootstrap interval."""

    point: float
    low: float
    high: float
    n: int
    n_resamples: int
    n_valid: int
    """Resamples that produced a finite statistic. A statistic that is undefined
    on some resamples (phi with a degenerate margin, say) is reported over the
    resamples where it existed, and this count makes that visible."""

    @property
    def width(self) -> float:
        return self.high - self.low

    def as_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {
            f"{prefix}point": self.point,
            f"{prefix}ci_low": self.low,
            f"{prefix}ci_high": self.high,
            f"{prefix}n": self.n,
            f"{prefix}n_resamples": self.n_resamples,
            f"{prefix}n_valid_resamples": self.n_valid,
        }


def bootstrap_statistic(
    stat_fn: Callable[[np.ndarray], float],
    n_items: int,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap of ``stat_fn`` over resampled task indices.

    ``stat_fn`` receives an index array (with replacement) and returns a scalar;
    returning NaN is allowed and means "undefined on this resample".
    """
    if n_items <= 0:
        return CI(float("nan"), float("nan"), float("nan"), 0, n_resamples, 0)

    rng = np.random.default_rng(seed)
    point = float(stat_fn(np.arange(n_items)))

    draws = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n_items, size=n_items)
        draws[b] = stat_fn(idx)

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return CI(point, float("nan"), float("nan"), n_items, n_resamples, 0)

    low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(point, float(low), float(high), n_items, n_resamples, int(finite.size))


def bootstrap_mean(
    values: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Bootstrap CI for a plain mean (used for MAE across pairs)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    def stat(idx: np.ndarray) -> float:
        return float(np.mean(arr[idx])) if idx.size else float("nan")

    return bootstrap_statistic(stat, arr.size, n_resamples, alpha, seed)
