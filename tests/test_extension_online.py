import runpy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from aip.aggregation.aip import AIPAggregator, InversionThresholds
from aip.extensions.online import CausalAIPReceiver
from aip.types import Broadcast, Observation


def obs(t, answers=("A", "B", "B")):
    return Observation(0, f"t{t}", tuple(
        Broadcast(i, f"t{t}", answer, .8, .8, i > 0, f"model{i}", "hidden")
        for i, answer in enumerate(answers) if answer != "ABSENT"
    ))


def receiver(**kwargs):
    thresholds = InversionThresholds(
        {"mc": .677}, {"mc": 1 / 3}, {"mmlu": "mc"}, min_observations=2
    )
    return CausalAIPReceiver(0, "mmlu", warmup=2, refresh_interval=1,
        thresholds=thresholds, label_space=["A", "B", "C", "D"], **kwargs)


def replay(stream, labels=None):
    policy = receiver()
    result = []
    for i, current in enumerate(stream):
        prediction = policy.predict(current)
        # Labels are available to an evaluator only after the prediction.
        _score = prediction.answer == labels[i] if labels else None
        result.append(asdict(prediction))
        policy.update(current)
    return result


def test_predict_before_update_and_current_public_observation_is_immutable():
    policy = receiver()
    with pytest.raises(RuntimeError, match="predict must precede"):
        policy.update(obs(0))
    prediction = policy.predict(obs(0))
    assert prediction.answer == "A" and prediction.past_tasks_observed == 0
    with pytest.raises(RuntimeError, match="pending"):
        policy.predict(obs(1))
    with pytest.raises(ValueError, match="same public"):
        policy.update(obs(0, ("B", "B", "B")))
    assert policy.past_tasks_observed == 0
    policy.update(obs(0))
    with pytest.raises(ValueError, match="unique"):
        policy.predict(obs(0))


def test_no_fit_sees_current_or_future_task_and_window_path_is_never_used(monkeypatch):
    fitted_ids = []
    original = AIPAggregator.fit

    def capture(self, tasks):
        assert self.window is None
        fitted_ids.append(tuple(o.task_id for task in tasks for o in task))
        return original(self, tasks)

    monkeypatch.setattr(AIPAggregator, "fit", capture)
    policy = receiver()
    for t in range(7):
        prediction = policy.predict(obs(t))
        if t >= 2:
            assert fitted_ids[-1] == tuple(f"t{i}" for i in range(t))
            assert prediction.fit_last_task_id == f"t{t-1}"
        generation = policy.fit_generation
        policy.update(obs(t))
        assert policy.fit_generation == generation


def test_prefix_predictions_ignore_future_broadcasts_and_all_evaluator_labels():
    stream = [obs(t, ("A" if t % 3 else "C", "B", "B")) for t in range(10)]
    altered_future = stream[:6] + [obs(t, ("D", "D", "A")) for t in range(6, 10)]
    original = replay(stream, ["A"] * 10)
    changed = replay(altered_future, ["D"] * 10)
    assert original[:6] == changed[:6]
    assert original == replay(stream, ["B"] * 10)


def test_hidden_byzantine_model_and_attack_flags_never_affect_predictions_or_commits():
    stream = [obs(t) for t in range(8)]
    hidden_changed = [replace(o, broadcasts=tuple(
        replace(b, is_byzantine=not b.is_byzantine, source_model="secret_model", attack="oracle")
        for b in o.broadcasts
    )) for o in stream]
    assert replay(stream) == replay(hidden_changed)
    policy = receiver()
    policy.predict(stream[0])
    policy.update(hidden_changed[0])  # Publicly identical, so the commit is valid.
    assert all(not b.is_byzantine and b.source_model is None and b.attack is None
               for b in policy._history[0].broadcasts)


def test_missing_peer_positions_do_not_create_cross_task_coincidence():
    policy = receiver()
    for current in [obs(0, ("A", "B", "ABSENT")), obs(1, ("A", "ABSENT", "B"))]:
        policy.predict(current)
        policy.update(current)
    prediction = policy.predict(obs(2, (None, "B", "B")))
    snapshot = policy.fit_snapshot()
    assert prediction.fitted_valid_tasks == 2
    assert snapshot["channels"][1]["n_observed"] == 1
    assert snapshot["channels"][2]["n_observed"] == 1
    assert snapshot["channels"][1]["joint_dissents"] == 0
    assert snapshot["channels"][2]["joint_dissents"] == 0
    policy.update(obs(2, (None, "B", "B")))
    later = policy.predict(obs(3))
    assert later.fitted_history_tasks == 3 and later.fitted_valid_tasks == 2


def test_fixed_and_rolling_history_retention_uses_task_positions():
    fixed, rolling = receiver(retention="fixed"), receiver(retention="rolling", history_size=3)
    for t in range(8):
        for policy in [fixed, rolling]:
            policy.predict(obs(t))
            policy.update(obs(t))
    assert fixed.retained_task_ids == ("t0", "t1")
    assert fixed.fit_generation == 1
    assert rolling.retained_task_ids == ("t5", "t6", "t7")
    assert rolling.fit_snapshot()["history_task_ids"] == ["t4", "t5", "t6"]


def test_invalid_observation_does_not_advance_the_stream():
    policy = receiver()
    current = obs(0)
    with pytest.raises(ValueError, match="once per task"):
        policy.predict(replace(current, broadcasts=current.broadcasts + current.broadcasts[:1]))
    with pytest.raises(ValueError, match="different receiver"):
        policy.predict(replace(current, self_id=1))
    with pytest.raises(ValueError, match="current task"):
        policy.predict(replace(current, task_id="mismatch"))
    assert policy.past_tasks_observed == 0


def test_block_bootstrap_respects_paired_constant_difference():
    runner = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/extension_online.py"))
    interval = runner["block_interval"](np.full((3, 40), .125), ["before"] * 20 + ["after"] * 20,
        seed=8, resamples=20)
    assert interval == (.125, .125, .125)


def test_binary_independent_and_coherent_controls_are_behaviorally_identical():
    runner = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/extension_online.py"))
    models = ["m0", "m1", "m2"]
    v1 = runner["v1"]
    cache = v1.pc.BenchmarkCache("boolq", ["t0"], ["A"],
        {(model, 0): [("A", .8, .8)] for model in models})
    base, attacked = runner["base_identity"](models, "boolq", 17, .5)
    coherent = runner["broadcasts_at"](cache, 0, base, attacked, "coherent", 17)
    independent = runner["broadcasts_at"](cache, 0, base, attacked, "independent", 17)
    assert [b.answer for b in coherent] == [b.answer for b in independent]
    for receiver in attacked.honest:
        assert attacked.models[receiver] == base.models[receiver]
    assert runner["regime_for"]("sleeper", 39) == "honest"
    assert runner["regime_for"]("sleeper", 40) == "coherent"
    assert [runner["regime_for"]("toggle", t) for t in (39, 40, 80, 120)] == [
        "coherent", "independent", "coherent", "independent"]
