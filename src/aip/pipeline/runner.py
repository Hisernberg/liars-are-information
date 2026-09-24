"""MX pipeline executor: generate once, aggregate many times.

The expensive object is a *stage context* -- a (stage, model, prompt) triple.
Stage 1's context depends only on the task, so it is shared by every arm. Stage
2's depends on what stage 1's aggregator decided, and stage 3's on stage 2's, so
those contexts fan out across arms and fault configurations. Generation is
cached by a hash of the context, which means the fan-out costs GPU time only
where the contexts genuinely differ.

Aggregation never touches the GPU. A whole arm, or a whole liar variant, can be
re-scored from the cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from aip.pipeline.arms import Arm, StageDecision, plurality, to_broadcasts
from aip.pipeline.liars import inject
from aip.pipeline.stages import STAGE_ORDER, stage
from aip.types import Observation


def context_key(stage_name: str, model: str, user_text: str, sample_index: int) -> str:
    """Stable id for one generation. Hashing the rendered prompt is what lets a
    stage-2 context be reused across arms that happened to agree at stage 1."""
    payload = json.dumps([stage_name, model, user_text, sample_index], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class GenerationCache:
    """Prompt-hash -> generated answer, persisted as parquet."""

    path: Path
    rows: dict[str, dict] = field(default_factory=dict)

    def load(self) -> GenerationCache:
        """Load, dropping any row whose generation never completed.

        A row is written before generation and filled in after, so a run that
        dies mid-flight leaves rows with ``answer=None`` on disk. Served back,
        those read as an agent that declined to answer -- a legitimate
        abstention, scored as one, and indistinguishable in the output from a
        model that genuinely abstained. Dropping them costs a regeneration;
        keeping them silently corrupts every arm downstream.
        """
        if self.path.exists():
            dropped = 0
            for r in pd.read_parquet(self.path).to_dict("records"):
                if r.get("answer") is None or (isinstance(r.get("answer"), float)
                                               and pd.isna(r["answer"])):
                    dropped += 1
                    continue
                self.rows[r["key"]] = r
            if dropped:
                print(f"  cache: dropped {dropped} incomplete generation(s) "
                      "from a previous run")
        return self

    def get(self, key: str) -> dict | None:
        """Return a completed generation, or None.

        Never returns a row with a null answer: an incomplete generation is a
        cache miss, not an abstention.
        """
        row = self.rows.get(key)
        if row is None or row.get("answer") is None:
            return None
        return row

    def put(self, key: str, **fields) -> None:
        self.rows[key] = {"key": key, **fields}

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(list(self.rows.values())).to_parquet(self.path, index=False)
        return self.path

    def __len__(self) -> int:
        return len(self.rows)


class MissingGeneration(KeyError):
    """A context was needed for aggregation and is not in the cache.

    Raised rather than substituted, because a silently-absent stage output
    scores as an abstention and would look like a defended cascade.
    """


def record_generation(row: dict, result, bench) -> dict:
    """Fill a cache row from one GenerationResult.

    Pulled out of the run script so the field names are exercised by a test
    rather than by a GPU load. Three separate runs died here on assumed
    attributes -- ``Extraction.answer`` (it is ``.value``) and
    ``GenerationResult.token_ids`` (it is ``.tokens``) -- each costing a model
    load to discover. The dataclasses are the contract; this is the one place
    that touches them.
    """
    extraction = bench.extract_answer(result.text)
    value = getattr(extraction, "value", None)
    truncated = result.finish_reason == "length"
    committed = _has_committed_answer(result.text)

    # A completion cut off mid-reasoning has not chosen anything. The shared
    # extractor is right to fall back to boxed/bold/leading-letter forms for
    # models that answer without the instructed format -- but on a TRUNCATED
    # completion those fallbacks find a letter the model was merely *listing*.
    # Observed live: phi4 truncated at 512 tokens mid-sentence arguing about
    # option A, and the extractor confidently returned C, on 113 of 464
    # generations, at a 100% "parse rate" that hid it completely.
    row["truncated"] = truncated
    row["committed_answer"] = committed
    if truncated and not committed:
        row["answer"] = None
        row["answer_discarded"] = "truncated_without_commitment"
    else:
        row["answer"] = None if value is None else bench.normalize(value)
        row["answer_discarded"] = None
    row["raw"] = result.text[:2000]
    row["n_tokens"] = len(result.tokens)
    row["finish_reason"] = result.finish_reason
    return row


def _has_committed_answer(text: str) -> bool:
    """True when the completion contains the instructed ``Answer: X`` form.

    Deliberately narrow. This is not a second extractor -- it asks only whether
    the model reached the point of committing, which is the question truncation
    makes ambiguous.
    """
    import re

    return bool(re.search(r"(?im)^\s*answer\s*:\s*[A-Za-z]", text))


def stage_observations(arm: Arm, task_answers: dict[str, list[str | None]],
                       anchors: dict[str, str | None]) -> list[list[Observation]]:
    """Build the observation stream one AIP stage needs before it can be fitted.

    The AIP gate is ``needs_fit``: it accumulates receiver-anchored channel
    statistics across the whole task stream before it can decide anything. An
    unfitted aggregator still returns an answer, which is the dangerous part --
    it returns a *degenerate* one, with no channel statistics and no logged
    decisions, and nothing in the call signature says so. Every AIP stage is
    fitted over its own stream before a single task is aggregated.

    One observation per task, whose ``self_id`` is the anchor slot: the anchor
    is the receiver, which is what makes the gate's tests receiver-anchored.
    """
    out: list[list[Observation]] = []
    for task_id, answers in task_answers.items():
        bs = to_broadcasts(answers, arm.replicas, anchors.get(task_id), task_id)
        out.append([Observation(self_id=len(answers), task_id=task_id,
                                broadcasts=tuple(bs))])
    return out


def fit_stage(aggregator, arm: Arm, task_answers: dict[str, list[str | None]],
              anchors: dict[str, str | None]) -> None:
    """Fit an AIP stage aggregator over its whole task stream."""
    if getattr(aggregator, "needs_fit", False):
        aggregator.fit(stage_observations(arm, task_answers, anchors))


def aggregate_stage(
    arm: Arm,
    task_id: str,
    answers: list[str | None],
    anchor: str | None,
    label_space: list[str],
    aip_aggregator=None,
) -> StageDecision:
    """Apply the arm's rule to one stage's replica answers."""
    decisions: tuple[str, ...] = ()
    if arm.rule == "none":
        out = answers[0] if answers else None
    elif arm.rule == "plurality":
        out = plurality(answers, anchor, label_space)
    elif arm.rule == "aip":
        if aip_aggregator is None:
            raise ValueError(f"arm {arm.name} needs an AIP aggregator")
        bs = to_broadcasts(answers, arm.replicas, anchor, task_id)
        out = aip_aggregator.aggregate(bs, len(answers))
        decisions = tuple(_read_decisions(aip_aggregator, task_id, len(answers), arm.k))
    else:
        raise ValueError(f"unknown rule {arm.rule!r}")
    return StageDecision(
        task_id=task_id, stage="", arm=arm.name, output=out, anchor=anchor,
        replica_answers=tuple(answers), replica_models=arm.replicas,
        decisions=decisions,
    )


def _read_decisions(agg, task_id: str, self_id: int, k: int) -> list[str]:
    """Recover the gate's per-channel decision for telemetry.

    Read out of the aggregator's own records, never re-derived: re-deriving
    would be a second implementation of the gate's logic, and the telemetry
    would eventually stop describing what the aggregator actually did -- the
    failure the mitigation probes in the main study were written to catch.

    The gate keeps these in two places depending on its mode. The windowed gate
    writes ``_task_decisions[(receiver, task_id)]`` because its decision is
    per-task; the pooled gate writes a single ``ChannelStats.decision`` per peer
    into ``_stats[receiver]`` because its decision is not. Both are read here so
    the telemetry does not silently go blank if the mode changes.
    """
    windowed = getattr(agg, "_task_decisions", None) or {}
    per_task = windowed.get((self_id, task_id))
    if per_task:
        return [str(per_task[i][0]) if i in per_task else "" for i in range(k)]
    pooled = (getattr(agg, "_stats", None) or {}).get(self_id) or {}
    if not pooled:
        # No record for this receiver -- the gate was never fitted. Report
        # nothing rather than k blank strings: blanks would satisfy a
        # "decisions logged" count while carrying no decision at all, which is
        # precisely the kind of telemetry that lies quietly.
        return []
    return [str(pooled[i].decision) if i in pooled else "" for i in range(k)]


def apply_fault(
    answers: list[str | None], slot: int | None, variant: str | None,
    gold: str, label_space: list[str], rng: np.random.Generator,
) -> tuple[list[str | None], str | None, bool]:
    """Inject a liar into one replica slot, or pass through unchanged."""
    if slot is None or variant is None:
        return answers, None, False
    out, res = inject(answers, slot, variant, gold, label_space, rng)
    return out, res.answer, res.fallback


def stage_sequence() -> tuple[str, ...]:
    return STAGE_ORDER


def build_stage_prompt(stage_name: str, fields: dict, prior: dict[str, str]) -> str:
    return stage(stage_name).build_user(fields, prior)
