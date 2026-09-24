"""Full class-conditional Dawid–Skene for a fixed, closed answer space.

Each receiver fits its own latent-label model using only the answers it has
observed in history. A worker has a C by C confusion matrix, with rows indexing
the latent truth and columns indexing the reported answer. Unlike the legacy
one-coin aggregator, prediction on a new task uses the learned prior and these
matrices; it does not fall back to majority voting.

No ground truth, Byzantine identities, confidence scores, or oracle anchors are
read. Missing reports and workers absent from history contribute no likelihood.
This assumes workers' conditional response channels persist from history to
evaluation and that reports are conditionally independent given the true label.
Correlated/replicated workers violate that assumption. Diagonal initialization
selects a label orientation in which workers are initially better than chance;
it does not identify truth under an adversarial majority. Latent-label symmetry,
limited history, EM local optima, and nonstationary attacks remain limitations.
Open-answer tasks are intentionally unsupported: a fixed class-confusion matrix
cannot be learned when the meaning of a label changes between tasks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aip.aggregation.base import Aggregator
from aip.types import Broadcast, Observation


@dataclass(frozen=True)
class FullDawidSkeneModel:
    """Receiver-local learned parameters and bounded-fit diagnostics.

    ``confusion_matrices[i, true_label, reported_label]`` corresponds to
    ``agent_ids[i]``. Smoothing is a uniform positive pseudocount per class and
    per matrix cell, so every learned likelihood is strictly positive.
    """

    label_space: tuple[str, ...]
    agent_ids: tuple[int, ...]
    class_prior: np.ndarray
    confusion_matrices: np.ndarray
    training_task_ids: tuple[str, ...]
    n_iter: int
    converged: bool
    log_likelihood: float
    likelihood_history: tuple[float, ...]


class DawidSkeneFullAggregator(Aggregator):
    """Label-free EM with one full confusion matrix per worker and receiver.

    ``fit`` must receive historical/calibration tasks only for held-out
    evaluation. ``aggregate`` never updates model parameters or reads a cached
    task prediction. Unknown receivers abstain. A fitted receiver with no valid
    reports also abstains; an unseen worker with a valid report leaves the
    posterior equal to the learned class prior.
    """

    name = "dawid_skene_full"
    needs_fit = True
    is_oracle = False

    def __init__(
        self,
        label_space: Sequence[str],
        *,
        max_iter: int = 100,
        tol: float = 1e-7,
        smoothing: float = 0.5,
        init_accuracy: float = 0.7,
    ) -> None:
        if label_space is None or isinstance(label_space, (str, bytes)):
            raise ValueError("label_space must be an explicit fixed sequence of class labels")
        labels = tuple(label_space)
        if (
            len(labels) < 2
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("label_space must contain at least two distinct nonempty strings")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be finite and positive")
        if not np.isfinite(init_accuracy) or not 1.0 / len(labels) < init_accuracy < 1.0:
            raise ValueError("init_accuracy must be strictly above chance and below one")
        self.label_space = tuple(sorted(labels))
        self._label_index = {label: i for i, label in enumerate(self.label_space)}
        self.max_iter = max_iter
        self.tol = float(tol)
        self.smoothing = float(smoothing)
        self.init_accuracy = float(init_accuracy)
        self.models_: dict[int, FullDawidSkeneModel] = {}

    def _read_row(
        self, broadcasts: Sequence[Broadcast], task_id: str | None = None
    ) -> dict[int, int]:
        """Read only observable task/worker identifiers and reported labels."""
        row: dict[int, int] = {}
        seen: set[int] = set()
        for broadcast in broadcasts:
            if task_id is None:
                task_id = broadcast.task_id
            if broadcast.task_id != task_id:
                raise ValueError("one observation must contain broadcasts for a single task")
            if broadcast.agent_id in seen:
                raise ValueError("duplicate worker in one observation would double-count evidence")
            seen.add(broadcast.agent_id)
            if broadcast.answer is None:
                continue
            if broadcast.answer not in self._label_index:
                raise ValueError(f"answer {broadcast.answer!r} is outside the fixed label_space")
            row[broadcast.agent_id] = self._label_index[broadcast.answer]
        return row

    @staticmethod
    def _expectation(
        answers: np.ndarray, prior: np.ndarray, confusion: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Compute stable latent-label posteriors, omitting missing entries."""
        log_probability = np.broadcast_to(np.log(prior), (answers.shape[0], prior.size)).copy()
        for agent_index in range(answers.shape[1]):
            present = answers[:, agent_index] >= 0
            reported = answers[present, agent_index]
            log_probability[present] += np.log(confusion[agent_index][:, reported].T)
        maximum = log_probability.max(axis=1, keepdims=True)
        unnormalized = np.exp(log_probability - maximum)
        normalizer = unnormalized.sum(axis=1, keepdims=True)
        posterior = unnormalized / normalizer
        log_likelihood = float((maximum + np.log(normalizer)).sum())
        return posterior, log_likelihood

    def _fit_receiver(
        self, history: list[tuple[str, dict[int, int]]]
    ) -> FullDawidSkeneModel:
        history = sorted(history, key=lambda item: item[0])
        agents = tuple(sorted({agent for _, row in history for agent in row}))
        agent_index = {agent: i for i, agent in enumerate(agents)}
        n_classes = len(self.label_space)
        answers = np.full((len(history), len(agents)), -1, dtype=int)
        for task_index, (_, row) in enumerate(history):
            for agent, answer in row.items():
                answers[task_index, agent_index[agent]] = answer

        prior = np.full(n_classes, 1.0 / n_classes)
        initial = np.full(
            (n_classes, n_classes), (1.0 - self.init_accuracy) / (n_classes - 1)
        )
        np.fill_diagonal(initial, self.init_accuracy)
        confusion = np.repeat(initial[None, :, :], len(agents), axis=0)
        posterior, likelihood = self._expectation(answers, prior, confusion)
        likelihood_history = [likelihood]
        converged = False

        for iteration in range(1, self.max_iter + 1):  # noqa: B007 - saved in fit diagnostics
            new_prior = posterior.sum(axis=0) + self.smoothing
            new_prior /= new_prior.sum()
            new_confusion = np.full(confusion.shape, self.smoothing)
            for index in range(len(agents)):
                for reported in range(n_classes):
                    mask = answers[:, index] == reported
                    new_confusion[index, :, reported] += posterior[mask].sum(axis=0)
            new_confusion /= new_confusion.sum(axis=2, keepdims=True)
            change = max(
                float(np.max(np.abs(new_prior - prior))),
                float(np.max(np.abs(new_confusion - confusion))),
            )
            prior, confusion = new_prior, new_confusion
            posterior, likelihood = self._expectation(answers, prior, confusion)
            likelihood_history.append(likelihood)
            if change <= self.tol:
                converged = True
                break

        # Pseudocounts regularize the fit; unregularized observed likelihood is
        # diagnostic, not an asserted monotonically increasing objective.
        return FullDawidSkeneModel(
            label_space=self.label_space,
            agent_ids=agents,
            class_prior=prior,
            confusion_matrices=confusion,
            training_task_ids=tuple(task_id for task_id, _ in history),
            n_iter=iteration,
            converged=converged,
            log_likelihood=likelihood,
            likelihood_history=tuple(likelihood_history),
        )

    def fit(self, tasks: Sequence[Sequence[Observation]]) -> None:
        by_receiver: dict[int, list[tuple[str, dict[int, int]]]] = {}
        seen: set[tuple[int, str]] = set()
        for observations in tasks:
            for observation in observations:
                key = (observation.self_id, observation.task_id)
                if key in seen:
                    raise ValueError("duplicate receiver/task in training history")
                seen.add(key)
                row = self._read_row(observation.broadcasts, observation.task_id)
                # Entirely missing tasks have no observed-data likelihood and
                # must not strengthen an initialization-dependent class prior.
                if row:
                    by_receiver.setdefault(observation.self_id, []).append(
                        (observation.task_id, row)
                    )
        fitted = {
            receiver: self._fit_receiver(history)
            for receiver, history in sorted(by_receiver.items())
        }
        self.models_ = fitted

    def posterior(self, observations: Sequence[Broadcast], self_id: int) -> dict[str, float]:
        """Predict an unseen task from frozen parameters; empty means abstain."""
        row = self._read_row(observations)
        model = self.models_.get(self_id)
        if model is None or not row:
            return {}
        answers = np.array([[row.get(agent, -1) for agent in model.agent_ids]], dtype=int)
        probability, _ = self._expectation(
            answers, model.class_prior, model.confusion_matrices
        )
        return dict(zip(self.label_space, probability[0].tolist(), strict=True))

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        posterior = self.posterior(observations, self_id)
        # Insertion order follows sorted labels, providing exact, deterministic
        # tie breaking without treating merely close probabilities as equal.
        return max(posterior, key=posterior.__getitem__) if posterior else None
