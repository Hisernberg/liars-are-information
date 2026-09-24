"""Leakage, missingness, and structural controls for independent calibration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aip.extensions.calibration import (  # noqa: E402
    calibrate_partition, calibrate_receiver, public_observation,
    split_task_ids, thresholds_with_ceiling, validate_partitions,
)
from aip.aggregation.aip import InversionThresholds  # noqa: E402
from aip.types import Broadcast, Observation  # noqa: E402


def examples(n=40, binary=False):
    rows = []
    for t in range(n):
        answers = ["A", "B", "B" if t % 4 else ("A" if binary else "C"),
                   "B" if binary else "C"]
        broadcasts = tuple(Broadcast(i, f"t{t:03d}", a, .8, .8, i == 2, "hidden_model", "hidden_attack")
            for i, a in enumerate(answers))
        rows.append(Observation(0, f"t{t:03d}", broadcasts))
    return rows


def estimate(rows, **kwargs):
    return calibrate_receiver(rows, calibration_ids=[obs.task_id for obs in rows],
        receiver=0, resamples=80, seed=91, **kwargs)


def test_split_is_disjoint_complete_and_order_invariant():
    ids = [f"t{i:03d}" for i in range(100)]
    a, b = split_task_ids(ids), split_task_ids(ids[::-1])
    assert a == b
    assert [len(a[k]) for k in ["calibration", "history", "validation", "test"]] == [20, 30, 20, 30]
    assert set(sum(a.values(), [])) == set(ids)
    validate_partitions(a)


def test_calibration_invariant_to_test_labels_answers_and_hidden_annotations():
    rows = examples(100)
    partitions = split_task_ids([obs.task_id for obs in rows])
    by_id = {obs.task_id: obs for obs in rows}
    first = calibrate_partition(by_id, partitions, receiver=0, resamples=80, seed=91)
    # Test labels are never an estimator input; changing both labels and every
    # test answer leaves all fitted calibration fields exactly unchanged.
    labels = {tid: "A" for tid in by_id}
    for tid in partitions["test"]:
        labels[tid] = "changed_test_gold"
        by_id[tid] = replace(by_id[tid], broadcasts=tuple(
            replace(b, answer="changed_test_response", is_byzantine=True)
            for b in by_id[tid].broadcasts))
    second = calibrate_partition(by_id, partitions, receiver=0, resamples=80, seed=91)
    assert first == second
    assert all(labels[t] == "changed_test_gold" for t in partitions["test"])


def test_binary_receiver_conditioning_has_structural_ceiling_one():
    fitted = estimate(examples(binary=True), n_options=2)
    assert fitted.pooled_ceiling == fitted.partner_ceiling == 1.
    assert fitted.status == "binary_structural_ceiling"


def test_bootstrap_preserves_task_peer_dependence_and_selected_maximum():
    fitted = estimate(examples())
    assert fitted.joint_dissent_events == 120
    assert fitted.coincidence_events == 40
    assert fitted.pooled_point == pytest.approx(1 / 3)
    assert fitted.partner_max_point == .75
    assert fitted.partner_ceiling >= fitted.pooled_ceiling
    assert fitted.pooled_ci_low <= fitted.pooled_point <= fitted.pooled_ci_high
    assert fitted == estimate(examples()[::-1])


def test_no_joint_dissent_returns_explicit_conservative_fallback():
    rows = [replace(obs, broadcasts=tuple(replace(b, answer="A") for b in obs.broadcasts))
            for obs in examples()]
    fitted = estimate(rows)
    assert fitted.pooled_point is None
    assert fitted.pooled_ceiling == fitted.partner_ceiling == 1.
    assert fitted.status == "no_joint_dissent_conservative_ceiling"


def test_partial_observations_do_not_turn_missing_peers_into_agreement():
    rows = [replace(obs, broadcasts=obs.broadcasts[:2]) for obs in examples()]
    fitted = estimate(rows)
    assert fitted.n_pairs == fitted.joint_dissent_events == 0
    assert fitted.pooled_ceiling == 1.


@pytest.mark.parametrize("mutation", ["empty", "missing", "duplicate", "forbidden", "receiver"])
def test_calibration_membership_guards(mutation):
    rows = examples()
    ids = [obs.task_id for obs in rows]
    forbidden = []
    if mutation == "empty":
        rows = []
    elif mutation == "missing":
        rows = rows[:-1]
    elif mutation == "duplicate":
        rows = rows[:-1] + rows[:1]
    elif mutation == "forbidden":
        forbidden = [ids[0]]
    else:
        rows[0] = replace(rows[0], self_id=1)
    with pytest.raises(ValueError):
        calibrate_receiver(rows, calibration_ids=ids, receiver=0, forbidden_ids=forbidden, resamples=30)


def test_partition_helper_rejects_overlap_and_missing_calibration():
    rows = examples(100)
    partitions = split_task_ids([o.task_id for o in rows])
    by_id = {o.task_id: o for o in rows}
    missing = dict(by_id)
    missing.pop(partitions["calibration"][0])
    with pytest.raises(ValueError, match="Missing calibration"):
        calibrate_partition(missing, partitions, receiver=0)
    partitions["test"].append(partitions["calibration"][0])
    with pytest.raises(ValueError, match="overlapping"):
        calibrate_partition(by_id, partitions, receiver=0)


def test_metadata_strip_and_threshold_copy_preserve_originals():
    obs = examples()[0]
    sanitized = public_observation(obs)
    assert any(b.is_byzantine for b in obs.broadcasts)
    assert all(not b.is_byzantine and b.source_model is None and b.attack is None for b in sanitized.broadcasts)
    assert [b.answer for b in sanitized.broadcasts] == [b.answer for b in obs.broadcasts]
    legacy = InversionThresholds({"mc": .677}, {"mc": 1 / 3}, {"mmlu": "mc", "arc": "mc"})
    new = thresholds_with_ceiling(legacy, "mmlu", .85)
    assert legacy.ceiling_for("mmlu") == .677
    assert new.ceiling_for("mmlu") == .85
    assert new.ceiling_for("arc") == .677
    assert np.isclose(new.chance_for("mmlu"), 1 / 3)
