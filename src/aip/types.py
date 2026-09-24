"""Shared, typed records used across every phase.

These are deliberately plain dataclasses (not pydantic) because they sit on the
hot path of Phase C's aggregation sweeps and are round-tripped through pandas /
parquet.  Configuration objects, which are validated once at load time, use
pydantic instead (see :mod:`aip.models.registry` and :mod:`aip.harness.runner`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AnswerKind(StrEnum):
    """Shape of a benchmark's answer space.

    ``MULTIPLE_CHOICE`` has a closed, known label set, which is what AIP's
    multiclass decode needs.  ``OPEN`` answers (GSM8K numerals, MATH-500 boxed
    expressions) have an unbounded space; the effective label set for pooling is
    the set of distinct normalized answers observed in a swarm on that task.
    """

    OPEN = "open"
    MULTIPLE_CHOICE = "multiple_choice"


@dataclass(frozen=True, slots=True)
class TaskItem:
    """One benchmark question with its gold answer."""

    task_id: str
    benchmark: str
    question: str
    gold_answer: str
    answer_kind: AnswerKind
    choices: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label_space(self) -> tuple[str, ...]:
        """Closed label space, or ``()`` for open-ended answers."""
        if self.answer_kind is AnswerKind.MULTIPLE_CHOICE:
            return tuple(chr(ord("A") + i) for i in range(len(self.choices)))
        return ()


@dataclass(frozen=True, slots=True)
class Extraction:
    """An answer parsed out of a raw completion, with its character span.

    The span is what lets :mod:`aip.models.confidence` compute the *unfakeable*
    confidence surface: the mean token logprob restricted to the answer tokens
    rather than the whole chain of thought.  ``value is None`` means extraction
    failed, which is itself a recorded outcome (scored as incorrect) rather than
    an error.
    """

    value: str | None
    start: int | None = None
    end: int | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def has_span(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass(slots=True)
class Prediction:
    """One model's cached answer to one task.

    ``logprob_confidence`` is derived from token logprobs over the answer span
    and cannot be manipulated by a prompt-injected agent's *stated* confidence;
    ``self_reported_confidence`` is elicited by a separate prompt and is exactly
    the surface the ``falsified_confidence`` attack targets.  Both are stored so
    Phase C can compare aggregators that read one against aggregators that read
    the other.
    """

    task_id: str
    benchmark: str
    model: str
    raw_completion: str
    extracted_answer: str | None
    gold_answer: str
    is_correct: bool
    logprob_confidence: float | None
    self_reported_confidence: float | None
    n_answer_tokens: int
    n_completion_tokens: int
    seed: int
    temperature: float
    sample_index: int = 0
    resample_kind: str = "answer_level"
    """What varies when this model is resampled at T>0.

    ``answer_level`` -- the model commits directly, so a resample varies the
    answer. ``path_level`` -- the answer is the tail of a sampled chain of
    thought, so a resample varies the whole reasoning path and only then the
    answer. The two are NOT the same quantity: path-level resampling is what
    produced the finding that temperature is a weak decorrelator for reasoning
    models. Phase 2 must analyse them separately rather than pooling them, which
    is why this is recorded per row rather than re-derived downstream.
    """
    nonreproducing: bool = False
    """Recorded in ``reproducibility.known_nonreproducing``: this generation
    does not reproduce byte-identically across identical re-runs. Carried on
    the record so the caveat travels with the data into every downstream
    parquet instead of living only in a config file."""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Broadcast:
    """What one agent puts on the wire in one round."""

    agent_id: int
    task_id: str
    answer: str | None
    logprob_confidence: float
    self_reported_confidence: float
    is_byzantine: bool
    source_model: str | None = None
    attack: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """The subset of broadcasts one agent actually receives.

    Partial observability (``p_obs``) and topology both act by thinning this
    set.  ``self_id`` is always present in ``broadcasts`` -- an agent observes
    its own answer -- which is precisely the self-vote mass that the
    normalization-parity control has to hold fixed across methods.
    """

    self_id: int
    task_id: str
    broadcasts: tuple[Broadcast, ...]

    @property
    def own(self) -> Broadcast:
        for b in self.broadcasts:
            if b.agent_id == self.self_id:
                return b
        raise ValueError(f"agent {self.self_id} missing from its own observation")

    @property
    def peers(self) -> tuple[Broadcast, ...]:
        return tuple(b for b in self.broadcasts if b.agent_id != self.self_id)
