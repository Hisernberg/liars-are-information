"""Prediction cache: typed round trip, upsert semantics, resume support."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aip.harness.cache import (
    PREDICTION_SCHEMA,
    CacheMissError,
    cache_status,
    completed_task_ids,
    load_predictions,
    read_manifest,
    to_frame,
    write_predictions,
)
from aip.types import Prediction


def _prediction(task_id: str, model: str = "m1", correct: bool = True, **kw) -> Prediction:
    defaults = dict(
        benchmark="gsm8k",
        raw_completion="reasoning\n#### 42",
        extracted_answer="42",
        gold_answer="42",
        is_correct=correct,
        logprob_confidence=0.8,
        self_reported_confidence=0.9,
        n_answer_tokens=2,
        n_completion_tokens=25,
        seed=0,
        temperature=0.0,
        sample_index=0,
    )
    defaults.update(kw)
    return Prediction(task_id=task_id, model=model, **defaults)


class TestSchema:
    def test_to_frame_is_typed(self) -> None:
        frame = to_frame([_prediction("t1"), _prediction("t2")])
        for column, dtype in PREDICTION_SCHEMA.items():
            assert str(frame[column].dtype) == dtype, column

    def test_extra_is_serialized(self) -> None:
        frame = to_frame([_prediction("t1", extra={"attack": "none"})])
        assert "attack" in frame["extra_json"].iloc[0]


class TestRoundTrip:
    def test_write_then_load(self, tmp_path: Path) -> None:
        write_predictions([_prediction("t1"), _prediction("t2")], "gsm8k", "m1", root=tmp_path)
        frame = load_predictions("gsm8k", "m1", root=tmp_path)
        assert len(frame) == 2
        assert set(frame["task_id"]) == {"t1", "t2"}
        assert frame["is_correct"].dtype == "boolean"

    def test_manifest_written(self, tmp_path: Path) -> None:
        write_predictions(
            [_prediction("t1")],
            "gsm8k",
            "m1",
            root=tmp_path,
            manifest={"prompt_template_hash": "abc123"},
        )
        meta = read_manifest("gsm8k", "m1", root=tmp_path)
        assert meta["prompt_template_hash"] == "abc123"
        assert meta["n_tasks"] == 1

    def test_missing_cell_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CacheMissError, match="phase_a_cache"):
            load_predictions("gsm8k", "nope", root=tmp_path)

    def test_task_id_filter(self, tmp_path: Path) -> None:
        write_predictions([_prediction("t1"), _prediction("t2")], "gsm8k", "m1", root=tmp_path)
        frame = load_predictions("gsm8k", "m1", root=tmp_path, task_ids=["t2"])
        assert frame["task_id"].tolist() == ["t2"]


class TestUpsertAndResume:
    def test_merge_appends_new_tasks(self, tmp_path: Path) -> None:
        write_predictions([_prediction("t1")], "gsm8k", "m1", root=tmp_path)
        write_predictions([_prediction("t2")], "gsm8k", "m1", root=tmp_path)
        assert len(load_predictions("gsm8k", "m1", root=tmp_path)) == 2

    def test_merge_overwrites_same_key(self, tmp_path: Path) -> None:
        write_predictions([_prediction("t1", correct=True)], "gsm8k", "m1", root=tmp_path)
        write_predictions([_prediction("t1", correct=False)], "gsm8k", "m1", root=tmp_path)
        frame = load_predictions("gsm8k", "m1", root=tmp_path)
        assert len(frame) == 1
        assert not bool(frame["is_correct"].iloc[0])

    def test_samples_are_distinct_rows(self, tmp_path: Path) -> None:
        write_predictions(
            [_prediction("t1", sample_index=0), _prediction("t1", sample_index=1)],
            "gsm8k",
            "m1",
            root=tmp_path,
        )
        assert len(load_predictions("gsm8k", "m1", root=tmp_path)) == 2

    def test_completed_task_ids(self, tmp_path: Path) -> None:
        write_predictions([_prediction("t1"), _prediction("t2")], "gsm8k", "m1", root=tmp_path)
        assert completed_task_ids("gsm8k", "m1", root=tmp_path) == {"t1", "t2"}

    def test_completed_requires_confidence(self, tmp_path: Path) -> None:
        write_predictions(
            [_prediction("t1"), _prediction("t2", self_reported_confidence=None)],
            "gsm8k",
            "m1",
            root=tmp_path,
        )
        done = completed_task_ids("gsm8k", "m1", root=tmp_path, require_confidence=True)
        assert done == {"t1"}

    def test_completed_empty_when_absent(self, tmp_path: Path) -> None:
        assert completed_task_ids("gsm8k", "m1", root=tmp_path) == set()


class TestStatus:
    def test_grid_coverage(self, tmp_path: Path) -> None:
        write_predictions([_prediction("t1")], "gsm8k", "m1", root=tmp_path)
        status = cache_status(["gsm8k", "mmlu"], ["m1", "m2"], root=tmp_path)
        assert isinstance(status, pd.DataFrame)
        assert len(status) == 4
        assert int(status["exists"].sum()) == 1
