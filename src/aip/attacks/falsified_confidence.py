"""Attack 2: invert the self-reported confidence, leave everything else alone.

**Offline transform, no new inference**, and that is essential to the experiment
rather than a saving. The prediction under test is that aggregators reading the
*self-reported* 0-100 confidence degrade while AIP's logprob-derived surface does
not. That comparison is only clean if the answer and the logprob confidence are
byte-identical to the honest cache and *only* the self-report moves. Generating
new outputs would confound the manipulation with ordinary sampling variation.
"""

from __future__ import annotations

import pandas as pd

from aip.attacks.base import Attack


def invert_self_report(value: float | None) -> float | None:
    """Map a self-reported confidence in [0, 1] to its complement."""
    if value is None or pd.isna(value):
        return value
    return float(min(1.0, max(0.0, 1.0 - float(value))))


def apply(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with only ``self_reported_confidence`` inverted."""
    out = frame.copy()
    out["self_reported_confidence"] = [
        invert_self_report(v) for v in out["self_reported_confidence"]
    ]
    return out


ATTACK = Attack(
    name="falsified_confidence",
    needs_inference=False,
    description="Invert the self-reported confidence; answer and logprob untouched.",
    build_messages=None,
    expected_invertible=False,
    notes=(
        "Attacks the surface a confident liar can trivially control. AIP reads "
        "the logprob channel instead, so it should be unmoved -- the whole point "
        "of caching two confidence surfaces separately in Phase A."
    ),
)
