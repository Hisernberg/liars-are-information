"""Closed-label DS validity, unseen-task inference, and observable-only access."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aip.aggregation.baselines import MajorityVote
from aip.aggregation.dawid_skene_full import DawidSkeneFullAggregator
from aip.types import Broadcast, Observation

LABELS = ("A", "B", "C", "D")


def broadcast(agent: int, answer: str | None, task_id: str = "new-task") -> Broadcast:
    return Broadcast(agent, task_id, answer, 0.6, 0.6, False)


def training_history(n_tasks: int = 320) -> list[list[Observation]]:
    """Five competent conditionally independent workers and one cyclic anti-expert."""
    rng = np.random.default_rng(20260912)
    tasks = []
    for task in range(n_tasks):
        truth = task % len(LABELS)
        task_id = f"history-{task:04d}"
        answers = []
        for agent in range(5):
            answer = truth if rng.random() < 0.9 else (truth + rng.integers(1, 4)) % 4
            answers.append(broadcast(agent, LABELS[answer], task_id))
        answer = (truth + 1) % 4 if rng.random() < 0.99 else truth
        answers.append(broadcast(5, LABELS[answer], task_id))
        tasks.append([Observation(0, task_id, tuple(answers))])
    return tasks


@pytest.fixture(scope="module")
def fitted() -> DawidSkeneFullAggregator:
    model = DawidSkeneFullAggregator(LABELS)
    model.fit(training_history())
    return model


def test_recovers_class_specific_anti_expert_with_competent_majority(fitted) -> None:
    learned = fitted.models_[0]
    anti = learned.confusion_matrices[learned.agent_ids.index(5)]
    assert np.array_equal(anti.argmax(axis=1), np.array([1, 2, 3, 0]))
    assert np.all(anti[np.arange(4), [1, 2, 3, 0]] > 0.9)
    for agent in range(5):
        matrix = learned.confusion_matrices[learned.agent_ids.index(agent)]
        assert np.all(np.diag(matrix) > 0.75)
    assert np.allclose(learned.confusion_matrices.sum(axis=2), 1.0)
    assert learned.class_prior.sum() == pytest.approx(1.0)
    assert 1 <= learned.n_iter <= fitted.max_iter


def test_unseen_task_uses_learned_anti_channel_and_can_predict_unreported_label(fitted) -> None:
    observations = [broadcast(0, None), broadcast(5, "B")]
    assert "new-task" not in fitted.models_[0].training_task_ids
    assert MajorityVote().aggregate(observations, 0) == "B"
    assert fitted.aggregate(observations, 0) == "A"
    assert fitted.posterior(observations, 0)["A"] > 0.9
    for truth, report in zip(LABELS, ("B", "C", "D", "A"), strict=True):
        assert fitted.aggregate([broadcast(5, report, f"unseen-{truth}")], 0) == truth


def test_missing_and_unknown_workers_are_neutral(fitted) -> None:
    reference = fitted.posterior([broadcast(5, "B")], 0)
    extra_missing = [broadcast(5, "B"), broadcast(0, None), broadcast(44, None)]
    assert fitted.posterior(extra_missing, 0) == reference
    assert fitted.posterior(extra_missing + [broadcast(45, "D")], 0) == reference
    prior = fitted.posterior([broadcast(45, "D")], 0)
    assert np.allclose(list(prior.values()), fitted.models_[0].class_prior)
    assert fitted.aggregate([broadcast(0, None), broadcast(5, None)], 0) is None
    assert fitted.aggregate([], 0) is None
    assert fitted.aggregate([broadcast(5, "B")], 99) is None


def test_missing_only_training_tasks_do_not_change_fitted_parameters(fitted) -> None:
    history = training_history()
    history.append([Observation(0, "missing-only", (broadcast(22, None, "missing-only"),))])
    other = DawidSkeneFullAggregator(LABELS)
    other.fit(history)
    assert np.array_equal(fitted.models_[0].class_prior, other.models_[0].class_prior)
    assert np.array_equal(
        fitted.models_[0].confusion_matrices, other.models_[0].confusion_matrices
    )


def test_byzantine_flags_and_confidence_are_unused(fitted) -> None:
    history = []
    for observations in training_history():
        obs = observations[0]
        flipped = tuple(
            replace(b, is_byzantine=True, logprob_confidence=-1e12, self_reported_confidence=0.0)
            for b in obs.broadcasts
        )
        history.append([replace(obs, broadcasts=flipped)])
    other = DawidSkeneFullAggregator(LABELS)
    other.fit(history)
    assert np.array_equal(
        fitted.models_[0].confusion_matrices, other.models_[0].confusion_matrices
    )
    observed = [broadcast(5, "B")]
    modified = [replace(observed[0], is_byzantine=True, self_reported_confidence=1.0)]
    assert fitted.posterior(observed, 0) == other.posterior(modified, 0)


class ObservableOnly:
    def __init__(self, original: Broadcast) -> None:
        self.agent_id = original.agent_id
        self.answer = original.answer
        self.task_id = original.task_id

    def __getattr__(self, name: str):
        raise AssertionError(f"attempted to read hidden/oracle field {name}")


def test_fitting_and_prediction_require_no_gold_or_other_hidden_fields() -> None:
    tasks = []
    for observations in training_history(40):
        obs = observations[0]
        tasks.append([replace(obs, broadcasts=tuple(ObservableOnly(b) for b in obs.broadcasts))])
    model = DawidSkeneFullAggregator(LABELS)
    model.fit(tasks)
    assert model.aggregate([ObservableOnly(broadcast(5, "B"))], 0) == "A"


def test_receiver_models_use_only_their_own_observed_history() -> None:
    tasks = []
    for task in range(80):
        task_id = f"local-{task:03d}"
        truth = task % 4
        honest = [broadcast(i, LABELS[truth], task_id) for i in range(5)]
        tasks.append(
            [
                Observation(0, task_id, tuple(honest + [broadcast(5, LABELS[(truth + 1) % 4], task_id)])),
                Observation(1, task_id, tuple(honest + [broadcast(5, LABELS[(truth + 2) % 4], task_id)])),
            ]
        )
    model = DawidSkeneFullAggregator(LABELS)
    model.fit(tasks)
    assert model.aggregate([broadcast(5, "B")], 0) == "A"
    assert model.aggregate([broadcast(5, "B")], 1) == "D"


def test_fit_is_deterministic_bounded_and_replaces_previous_history() -> None:
    left = DawidSkeneFullAggregator(LABELS, max_iter=2, tol=1e-14)
    right = DawidSkeneFullAggregator(tuple(reversed(LABELS)), max_iter=2, tol=1e-14)
    left.fit(training_history(40))
    right.fit(list(reversed(training_history(40))))
    assert np.array_equal(left.models_[0].class_prior, right.models_[0].class_prior)
    assert np.array_equal(left.models_[0].confusion_matrices, right.models_[0].confusion_matrices)
    assert left.models_[0].n_iter == 2
    left.fit([])
    assert left.models_ == {}
    assert left.aggregate([broadcast(0, "A")], 0) is None


def test_stable_likelihood_with_many_concentrated_reports() -> None:
    model = DawidSkeneFullAggregator(("A", "B"), max_iter=3)
    tasks = []
    for task in range(20):
        task_id = str(task)
        broadcasts = tuple(broadcast(agent, ("A", "B")[task % 2], task_id) for agent in range(400))
        tasks.append([Observation(0, task_id, broadcasts)])
    model.fit(tasks)
    probability = model.posterior([broadcast(agent, "A") for agent in range(400)], 0)
    assert np.isfinite(model.models_[0].log_likelihood)
    assert np.isfinite(list(probability.values())).all()
    assert sum(probability.values()) == pytest.approx(1.0)
    assert probability["A"] > 0.999


@pytest.mark.parametrize("labels", [None, (), ("A",), ("A", "A"), ("A", ""), "AB"])
def test_requires_an_explicit_closed_label_space(labels) -> None:
    with pytest.raises(ValueError):
        DawidSkeneFullAggregator(labels)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 0},
        {"max_iter": True},
        {"tol": 0.0},
        {"smoothing": 0.0},
        {"smoothing": float("nan")},
        {"init_accuracy": 0.25},
        {"init_accuracy": 1.0},
    ],
)
def test_invalid_fit_settings_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        DawidSkeneFullAggregator(LABELS, **kwargs)


def test_rejects_invalid_labels_mixed_tasks_and_duplicate_evidence(fitted) -> None:
    with pytest.raises(ValueError, match="outside"):
        fitted.aggregate([broadcast(0, "E")], 0)
    with pytest.raises(ValueError, match="single task"):
        fitted.aggregate([broadcast(0, "A", "first"), broadcast(1, "A", "second")], 0)
    with pytest.raises(ValueError, match="duplicate worker"):
        fitted.aggregate([broadcast(0, "A"), broadcast(0, "A")], 0)
    fresh = DawidSkeneFullAggregator(LABELS)
    task = training_history(1)[0]
    with pytest.raises(ValueError, match="duplicate receiver/task"):
        fresh.fit([task, task])
