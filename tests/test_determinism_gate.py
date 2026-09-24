"""Gate d's accepted-caveat logic.

``reproducibility.known_nonreproducing`` was written into every run manifest but
never consulted by the gate, so a documented caveat still failed. These tests pin
the corrected semantics. They exist as unit tests rather than as an observation
from a live run because the nondeterminism they describe is *intermittent*: the
one affected id reproduced fine on the very next dry run, so no live run can be
relied on to exercise the branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from phase_a_cache import classify_determinism  # noqa: E402

KNOWN = frozenset({"gsm8k-test-00000"})


def test_no_diffs_passes() -> None:
    ok, detail = classify_determinism([], 5, KNOWN)
    assert ok
    assert detail == "all 5 byte-identical"


def test_recorded_id_is_accepted_not_a_failure() -> None:
    """The whole point: a measured, documented caveat must not fail the gate."""
    ok, detail = classify_determinism(["gsm8k-test-00000"], 30, KNOWN)
    assert ok
    assert "previously recorded" in detail
    assert "gsm8k-test-00000" in detail


def test_new_id_fails() -> None:
    ok, detail = classify_determinism(["gsm8k-test-00042"], 30, KNOWN)
    assert not ok
    assert "NEWLY differ" in detail


def test_new_id_fails_even_alongside_a_recorded_one() -> None:
    """A recorded caveat must never mask an unrecorded regression."""
    ok, detail = classify_determinism(
        ["gsm8k-test-00000", "gsm8k-test-00042"], 30, KNOWN
    )
    assert not ok
    assert "gsm8k-test-00042" in detail
    assert "already recorded" in detail


def test_empty_known_set_means_any_diff_fails() -> None:
    ok, _ = classify_determinism(["gsm8k-test-00000"], 5, frozenset())
    assert not ok


@pytest.mark.parametrize("n", [1, 5, 30, 500])
def test_detail_reports_the_denominator(n: int) -> None:
    _, detail = classify_determinism([], n, KNOWN)
    assert f"all {n} " in detail
