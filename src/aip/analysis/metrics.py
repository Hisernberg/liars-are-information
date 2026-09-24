"""Small report-time metrics shared by phase scripts.

Kept separate from :mod:`aip.analysis.bootstrap` (Phase C) because these are
descriptive statistics used to *characterise* the cache, not inferential
statistics used to make claims about methods.
"""

from __future__ import annotations

from collections.abc import Sequence


def rank_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Probability that a random positive scores above a random negative.

    The Mann-Whitney U statistic normalised to [0, 1], with ties credited 0.5.
    ``0.5`` means no signal. Returns ``None`` when either class is empty, since
    AUC is undefined there -- reporting ``0.5`` would falsely suggest a measured
    null result rather than an unmeasurable one.
    """
    pos = [p for p in positive if p is not None]
    neg = [n for n in negative if n is not None]
    if not pos or not neg:
        return None

    merged = sorted([(v, 0) for v in pos] + [(v, 1) for v in neg])
    ranks: dict[int, float] = {}
    i = 0
    rank_sum_pos = 0.0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if merged[k][1] == 0:
                rank_sum_pos += average_rank
        i = j + 1
    del ranks

    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)
