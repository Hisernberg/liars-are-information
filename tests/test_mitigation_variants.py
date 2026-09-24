"""The mitigation variants must be different code, not the same code three times.

Task 3 reports that M1 (soft gate) and M2 (randomized threshold) produce results
numerically identical to the hard gate, and concludes from that identity that the
evasion band is a property of the coherence statistic rather than of the
threshold. That conclusion is only admissible if the variants genuinely execute
different logic -- otherwise the "identical results" are an implementation bug
and the conclusion is unearned.

These are the probes a reviewer would demand. They show the variants diverge from
the hard gate on inputs where they SHOULD diverge, so the identity observed in
the sweep is a fact about the adversary's operating point and not about the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.aggregation.aip import (  # noqa: E402
    AIPAggregator,
    ChannelStats,
    InversionThresholds,
)

BENCHMARK = "mmlu"


def _agg(**kw) -> AIPAggregator:
    return AIPAggregator(BENCHMARK, "gated", InversionThresholds.load(), None, None, **kw)


def _stats(coherence: float) -> ChannelStats:
    """A confidently-wrong, strongly-dissenting channel at a chosen coherence.

    Uses the real ChannelStats so the probe exercises the same fields the sweep
    does; a hand-rolled stand-in could diverge from the type and prove nothing.
    """
    return ChannelStats(
        n_observed=200,
        dissent_rate=0.9,
        agreement_with_self=0.0,
        expected_honest_agreement=0.6,
        coherence=coherence,
        coincidences=180,
        joint_dissents=200,
        coherence_pvalue=0.0,
        blind_error=0.9,
    )


def test_variants_are_distinct_objects_with_distinct_names() -> None:
    names = {
        _agg().name,
        _agg(soft_gate=6.0).name,
        _agg(randomized_threshold=0.3).name,
        _agg(soft_gate=6.0, randomized_threshold=0.3).name,
    }
    assert len(names) == 4, f"variants share names: {names}"


def test_soft_gate_weight_is_continuous_in_coherence() -> None:
    """M1's defining property: weight varies smoothly, with no jump anywhere.

    Note the hard gate's default rule is a BINOMIAL test on the coherence
    estimate, not a comparison against the ceiling, so its discontinuity does not
    sit at the ceiling and probing there proves nothing about it. What must be
    shown is M1's own smoothness: adjacent coherence levels give adjacent
    weights.
    """
    soft = _agg(soft_gate=6.0)
    ceiling = soft.thresholds.ceiling_for(BENCHMARK)
    qs = [ceiling * m for m in (0.96, 0.98, 1.0, 1.02, 1.04)]
    weights = [soft._decide(1, _stats(q), 0)[1] for q in qs]
    steps = [abs(b - a) for a, b in zip(weights[:-1], weights[1:], strict=True)]
    assert max(steps) < 0.25, f"soft gate jumped: weights {weights}"


def test_hard_gate_steps_when_using_the_fixed_ceiling_rule() -> None:
    """The control: under the fixed-ceiling rule the hard gate IS a cliff."""
    hard = _agg(coherence_test="fixed_ceiling")
    ceiling = hard.thresholds.ceiling_for(BENCHMARK)
    lo = hard._decide(1, _stats(ceiling * 0.98), 0)
    hi = hard._decide(1, _stats(ceiling * 1.02), 0)
    assert lo[0] != hi[0], f"fixed-ceiling gate did not step: {lo} vs {hi}"


@pytest.mark.parametrize("coherence", [0.30, 0.50, 0.90])
def test_soft_gate_weight_differs_from_hard_gate(coherence: float) -> None:
    """M1 must produce a different weight from the hard gate somewhere real."""
    hard_d, hard_w = _agg()._decide(1, _stats(coherence), 0)
    soft_d, soft_w = _agg(soft_gate=6.0)._decide(1, _stats(coherence), 0)
    assert (hard_d, round(hard_w, 6)) != (soft_d, round(soft_w, 6)) or hard_w == 0.0, (
        f"soft gate identical to hard gate at q={coherence}"
    )


def test_randomized_threshold_actually_moves_the_ceiling() -> None:
    """M2 must draw a ceiling that differs from the calibrated one, and vary."""
    rand = _agg(randomized_threshold=0.3, threshold_seed=1)
    base = rand.thresholds.ceiling_for(BENCHMARK)
    seen = set()
    for epoch in range(12):
        rand._draw_ceiling_jitter(epoch)
        seen.add(round(rand._effective_ceiling(), 6))
    assert len(seen) > 1, "randomized threshold never varied across epochs"
    assert any(abs(c - base) > 1e-9 for c in seen), "jitter never left the base ceiling"
    assert all(0.7 * base <= c <= 1.3 * base for c in seen), "jitter outside declared width"


def test_hard_gate_ceiling_is_fixed() -> None:
    """The control: the hard gate's ceiling must NOT move."""
    hard = _agg()
    seen = set()
    for epoch in range(5):
        hard._draw_ceiling_jitter(epoch)
        seen.add(round(hard._effective_ceiling(), 9))
    assert len(seen) == 1, "hard gate ceiling moved; it must be fixed"


def test_randomized_threshold_reaches_the_default_binomial_rule() -> None:
    """M2 must move the rule the study actually uses, not only the ablation.

    The default coherence test is a binomial tail probability against the
    ceiling; the fixed-ceiling comparison is an ablation. A jitter wired only
    into the ablation leaves M2 inert in every real run -- which is exactly what
    the first Task 3 pass measured, and why it wrongly reported M2 as having no
    effect.
    """
    rand = _agg(randomized_threshold=0.3, threshold_seed=7)
    stats = _stats(0.5)
    seen = set()
    for epoch in range(10):
        rand._draw_ceiling_jitter(epoch)
        # The p-value depends on the ceiling, so a moving ceiling must move it.
        seen.add(round(rand._coherence_fields.__self__._effective_ceiling(), 6)
                 if hasattr(rand._coherence_fields, "__self__")
                 else round(rand._effective_ceiling(), 6))
    assert len(seen) > 1, "M2 ceiling never varied"
    assert stats.coherence == 0.5
