"""The inference cache: the one place LLM outputs live.

Design principle 1 (cache-once inference).  Phase A writes one parquet per
``(benchmark, model)`` cell plus a JSON manifest recording the prompt-template
hash, model revision, seed, and date.  Every later phase reads this cache and
must never trigger inference.

Resume safety: writes are atomic (temp file + rename) and
:func:`completed_task_ids` lets a re-run skip finished work.  Merging a partial
batch into an existing cell is an upsert keyed by
``(task_id, model, sample_index)``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from aip.harness.logging import get_logger
from aip.types import Prediction

log = get_logger(__name__)

DEFAULT_CACHE_ROOT = Path("data/cache")

#: Column -> pandas dtype for the prediction cache. Enforced on read and write
#: so that downstream phases can rely on the schema without defensive casting.
PREDICTION_SCHEMA: dict[str, str] = {
    "task_id": "string",
    "benchmark": "string",
    "model": "string",
    "raw_completion": "string",
    "extracted_answer": "string",
    "gold_answer": "string",
    "is_correct": "boolean",
    "logprob_confidence": "float64",
    "self_reported_confidence": "float64",
    "n_answer_tokens": "int64",
    "n_completion_tokens": "int64",
    "seed": "int64",
    "temperature": "float64",
    "sample_index": "int64",
    # Marks a row whose generation is known not to reproduce byte-identically
    # across identical re-runs, set from reproducibility.known_nonreproducing
    # at write time. Carrying the caveat in the data means every downstream
    # parquet inherits it, rather than it living only in a config file three
    # directories away. Rows written before this column existed read back as
    # False, which is correct: they passed gate d.
    "nonreproducing": "boolean",
    "resample_kind": "string",
}

KEY_COLUMNS = ["task_id", "model", "sample_index"]


class CacheMissError(FileNotFoundError):
    """Raised when a requested cache cell does not exist."""


def cache_path(benchmark: str, model: str, root: Path | str = DEFAULT_CACHE_ROOT) -> Path:
    return Path(root) / benchmark / f"{model}.parquet"


def manifest_path(benchmark: str, model: str, root: Path | str = DEFAULT_CACHE_ROOT) -> Path:
    return Path(root) / benchmark / f"{model}.manifest.json"


def to_frame(predictions: Iterable[Prediction]) -> pd.DataFrame:
    """Convert prediction records to a schema-typed DataFrame."""
    rows = []
    for p in predictions:
        rec = asdict(p)
        extra = rec.pop("extra", {}) or {}
        rec["extra_json"] = json.dumps(extra, default=str) if extra else None
        rows.append(rec)
    frame = pd.DataFrame(rows, columns=[*PREDICTION_SCHEMA.keys(), "extra_json"])
    return enforce_schema(frame)


def enforce_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast to :data:`PREDICTION_SCHEMA`, adding missing optional columns."""
    out = frame.copy()
    for column, dtype in PREDICTION_SCHEMA.items():
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = out[column].astype(dtype)
    # A missing flag means "not known to be nonreproducing", not "unknown".
    out["nonreproducing"] = out["nonreproducing"].fillna(False).astype("boolean")
    # Rows written before this column existed are answer-level: the only
    # reasoning models in the roster were cached after it was introduced.
    out["resample_kind"] = out["resample_kind"].fillna("answer_level").astype("string")
    if "extra_json" not in out.columns:
        out["extra_json"] = pd.NA
    out["extra_json"] = out["extra_json"].astype("string")
    return out[[*PREDICTION_SCHEMA.keys(), "extra_json"]]


def write_predictions(  # noqa: D417
    predictions: Iterable[Prediction] | pd.DataFrame,
    benchmark: str,
    model: str,
    root: Path | str = DEFAULT_CACHE_ROOT,
    manifest: dict[str, Any] | None = None,
    merge: bool = True,
) -> Path:
    """Write (or upsert) a cache cell atomically and refresh its manifest.

    Confidences are written **raw**: exactly what
    :func:`aip.models.confidence.logprob_to_confidence` produced, with no
    rank-normalization, rescaling, or calibration. Normalization is a Phase C
    consumption-time concern precisely so that competing schemes can be compared
    under matched parity. See ``tests/test_confidence_raw.py``.
    """
    frame = predictions if isinstance(predictions, pd.DataFrame) else to_frame(predictions)
    frame = enforce_schema(frame)

    path = cache_path(benchmark, model, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    if merge and path.exists():
        existing = enforce_schema(pd.read_parquet(path))
        combined = pd.concat([existing, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    else:
        combined = frame
    combined = combined.sort_values(KEY_COLUMNS).reset_index(drop=True)

    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(path)

    meta = {
        "benchmark": benchmark,
        "model": model,
        "n_rows": int(len(combined)),
        "n_tasks": int(combined["task_id"].nunique()),
        "updated_at": datetime.now(UTC).isoformat(),
        **(manifest or {}),
    }
    manifest_path(benchmark, model, root).write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    log.info("cache.write", benchmark=benchmark, model=model, rows=len(combined), path=str(path))
    return path


def load_predictions(
    benchmark: str,
    model: str,
    root: Path | str = DEFAULT_CACHE_ROOT,
    task_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load one cache cell as a typed DataFrame."""
    path = cache_path(benchmark, model, root)
    if not path.exists():
        raise CacheMissError(
            f"no cache for benchmark={benchmark!r} model={model!r} at {path}. "
            "Run scripts/phase_a_cache.py first."
        )
    frame = enforce_schema(pd.read_parquet(path))
    if task_ids is not None:
        wanted = set(task_ids)
        frame = frame[frame["task_id"].isin(wanted)].reset_index(drop=True)
    return frame


def load_many(
    benchmarks: Iterable[str],
    models: Iterable[str],
    root: Path | str = DEFAULT_CACHE_ROOT,
    strict: bool = True,
) -> pd.DataFrame:
    """Load several cells into one long-format DataFrame."""
    frames = []
    for benchmark in benchmarks:
        for model in models:
            try:
                frames.append(load_predictions(benchmark, model, root))
            except CacheMissError:
                if strict:
                    raise
                log.warning("cache.missing", benchmark=benchmark, model=model)
    if not frames:
        return enforce_schema(pd.DataFrame())
    return pd.concat(frames, ignore_index=True)


def completed_task_ids(
    benchmark: str,
    model: str,
    root: Path | str = DEFAULT_CACHE_ROOT,
    require_confidence: bool = False,
) -> set[str]:
    """Task ids already cached for this cell -- the basis for resume.

    With ``require_confidence`` a row only counts as complete once its
    self-reported confidence has also been elicited, so an interrupted run that
    finished generation but not the confidence pass resumes correctly.
    """
    path = cache_path(benchmark, model, root)
    if not path.exists():
        return set()
    frame = enforce_schema(pd.read_parquet(path))
    if require_confidence:
        frame = frame[frame["self_reported_confidence"].notna()]
    return set(frame["task_id"].dropna().astype(str).tolist())


def read_manifest(
    benchmark: str, model: str, root: Path | str = DEFAULT_CACHE_ROOT
) -> dict[str, Any] | None:
    path = manifest_path(benchmark, model, root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cache_status(
    benchmarks: Iterable[str], models: Iterable[str], root: Path | str = DEFAULT_CACHE_ROOT
) -> pd.DataFrame:
    """Coverage table over the benchmark x model grid (which cells exist)."""
    rows = []
    for benchmark in benchmarks:
        for model in models:
            path = cache_path(benchmark, model, root)
            meta = read_manifest(benchmark, model, root) or {}
            rows.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "exists": path.exists(),
                    "n_tasks": meta.get("n_tasks", 0),
                    "n_rows": meta.get("n_rows", 0),
                    "prompt_template_hash": meta.get("prompt_template_hash"),
                    "updated_at": meta.get("updated_at"),
                }
            )
    return pd.DataFrame(rows)
