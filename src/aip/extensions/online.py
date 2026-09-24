"""Strictly causal, receiver-local AIP with explicit predict/observe phases.

The retained history is a sequence of task positions, never independently
compacted peer streams. Neither labels nor hidden identities enter this API.
The historical AIP ``window`` implementation is deliberately never invoked.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import Literal

from aip.aggregation.aip import AIPAggregator, InversionThresholds
from aip.aggregation.base import ParityConfig
from aip.types import Observation


def public_observation(observation: Observation) -> Observation:
    """Validate task-local identity and erase evaluator-only metadata."""
    agents = [b.agent_id for b in observation.broadcasts]
    if len(set(agents)) != len(agents):
        raise ValueError("A peer may broadcast only once per task")
    if observation.self_id not in agents:
        raise ValueError("The receiver's own broadcast is required, even if its answer is missing")
    if any(b.task_id != observation.task_id for b in observation.broadcasts):
        raise ValueError("Every broadcast must belong to the current task")
    return replace(observation, broadcasts=tuple(
        replace(b, is_byzantine=False, source_model=None, attack=None)
        for b in sorted(observation.broadcasts, key=lambda b: b.agent_id)
    ))


def observation_fingerprint(observation: Observation) -> str:
    payload = [observation.self_id, observation.task_id, [
        (b.agent_id, b.answer, b.logprob_confidence, b.self_reported_confidence)
        for b in observation.broadcasts
    ]]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class OnlinePrediction:
    answer: str | None
    task_id: str
    receiver_id: int
    past_tasks_observed: int
    retained_history_tasks: int
    fit_generation: int
    fitted_history_tasks: int
    fitted_valid_tasks: int
    fit_last_task_id: str | None
    fit_task_ids_sha256: str
    refreshed: bool


class CausalAIPReceiver:
    """One receiver's immutable-past channel state.

    Call ``predict(current)`` exactly once, then ``update(current)``. Prediction
    can refresh a model using *only* already committed observations. Update
    commits the public observation but never fits it or revises the prediction.

    ``fixed`` retains the first ``warmup`` observations and fits once.
    ``cumulative`` retains the entire past; ``rolling`` retains the last
    ``history_size`` task positions. Both refresh every ``refresh_interval``
    committed observations, starting after the shared warmup count.
    """

    def __init__(
        self,
        receiver_id: int,
        benchmark: str,
        *,
        retention: Literal["fixed", "cumulative", "rolling"] = "cumulative",
        history_size: int | None = None,
        warmup: int = 40,
        refresh_interval: int = 8,
        thresholds: InversionThresholds | None = None,
        label_space: list[str] | None = None,
        parity_share: float | None = 0.1,
    ) -> None:
        if retention not in {"fixed", "cumulative", "rolling"}:
            raise ValueError("Unknown retention policy")
        if warmup < 1 or refresh_interval < 1:
            raise ValueError("Warmup and refresh interval must be positive")
        if retention == "rolling" and (history_size is None or history_size < 1):
            raise ValueError("Rolling history requires a positive history size")
        self.receiver_id = int(receiver_id)
        self.retention = retention
        self.warmup = int(warmup)
        self.refresh_interval = int(refresh_interval)
        self.history_size = history_size
        self._history: deque[Observation] = deque(
            maxlen=history_size if retention == "rolling" else None
        )
        self._seen_task_ids: set[str] = set()
        self._pending: Observation | None = None
        self._pending_fingerprint: str | None = None
        self._past_count = 0
        self._fit_generation = 0
        self._last_fit_at: int | None = None
        self._fit_history_ids: tuple[str, ...] = ()
        self._fit_valid_ids: tuple[str, ...] = ()
        self._fit_hash = hashlib.sha256(b"[]").hexdigest()
        self._model = AIPAggregator(
            benchmark=benchmark, thresholds=thresholds, parity=ParityConfig(parity_share),
            label_space=label_space, window=None,
        )

    @property
    def past_tasks_observed(self) -> int:
        return self._past_count

    @property
    def fit_generation(self) -> int:
        return self._fit_generation

    @property
    def retained_task_ids(self) -> tuple[str, ...]:
        return tuple(obs.task_id for obs in self._history)

    def _refresh_from_committed_past(self) -> bool:
        if self._past_count < self.warmup:
            return False
        if self._last_fit_at is not None:
            if self.retention == "fixed":
                return False
            if self._past_count - self._last_fit_at < self.refresh_interval:
                return False
        history = tuple(self._history)
        # A missing answer is unavailable evidence, never agreement on None.
        # Entire missing-self rows are omitted jointly for all peer series;
        # absent peer positions remain missing inside the base static fitter.
        valid = [replace(obs, broadcasts=tuple(b for b in obs.broadcasts if b.answer is not None))
                 for obs in history if obs.own.answer is not None]
        self._model.fit([(obs,) for obs in valid])
        self._fit_history_ids = tuple(obs.task_id for obs in history)
        self._fit_valid_ids = tuple(obs.task_id for obs in valid)
        self._fit_hash = hashlib.sha256(json.dumps(
            self._fit_history_ids, separators=(",", ":")
        ).encode()).hexdigest()
        self._last_fit_at = self._past_count
        self._fit_generation += 1
        return True

    def predict(self, observation: Observation) -> OnlinePrediction:
        if self._pending is not None:
            raise RuntimeError("Commit the pending observation with update before predicting again")
        public = public_observation(observation)
        if public.self_id != self.receiver_id:
            raise ValueError("Observation belongs to a different receiver")
        if public.task_id in self._seen_task_ids:
            raise ValueError("Task IDs must be unique in the chronological stream")
        refreshed = self._refresh_from_committed_past()
        # Before valid history exists, AIP's all-zero score tie must not become
        # an arbitrary alphabetical prediction. The predeclared cold start is self.
        answer = (
            self._model.aggregate(public.broadcasts, self.receiver_id)
            if self._fit_valid_ids else public.own.answer
        )
        self._pending = public
        self._pending_fingerprint = observation_fingerprint(public)
        return OnlinePrediction(
            answer=answer, task_id=public.task_id, receiver_id=self.receiver_id,
            past_tasks_observed=self._past_count, retained_history_tasks=len(self._history),
            fit_generation=self._fit_generation,
            fitted_history_tasks=len(self._fit_history_ids),
            fitted_valid_tasks=len(self._fit_valid_ids),
            fit_last_task_id=self._fit_history_ids[-1] if self._fit_history_ids else None,
            fit_task_ids_sha256=self._fit_hash, refreshed=refreshed,
        )

    def update(self, observation: Observation) -> None:
        if self._pending is None:
            raise RuntimeError("predict must precede update for every observation")
        public = public_observation(observation)
        if observation_fingerprint(public) != self._pending_fingerprint:
            raise ValueError("update must commit the same public observation that was predicted")
        if self.retention != "fixed" or len(self._history) < self.warmup:
            self._history.append(public)
        self._seen_task_ids.add(public.task_id)
        self._past_count += 1
        self._pending = None
        self._pending_fingerprint = None

    def fit_snapshot(self) -> dict:
        """Auditable fitted state, including exact strictly-past task support."""
        return {
            "receiver_id": self.receiver_id, "fit_generation": self._fit_generation,
            "fit_at_past_count": self._last_fit_at,
            "history_task_ids": list(self._fit_history_ids),
            "valid_fit_task_ids": list(self._fit_valid_ids),
            "fit_task_ids_sha256": self._fit_hash,
            "channels": {peer: asdict(stats) for peer, stats in
                         self._model.diagnostics.channels.get(self.receiver_id, {}).items()},
        }
