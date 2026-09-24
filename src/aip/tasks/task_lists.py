"""Frozen task lists: the exact questions the study runs on.

``configs/task_lists.json`` is committed and is the *authority* on which tasks
enter the study. Phase A caches predictions for exactly these ids, and Phase D's
adversarial runs must reuse them -- otherwise honest and Byzantine broadcasts
would refer to different questions and no swarm could be assembled from the two.

Reading tasks through :func:`load_fixed_tasks` rather than re-running the sampler
means a change to the sampling code, the dataset revision, or the seed can never
silently shift which questions the paper reports on: an id that no longer exists
raises instead.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aip.harness.logging import get_logger
from aip.types import TaskItem

log = get_logger(__name__)

DEFAULT_PATH = Path("configs/task_lists.json")


class TaskListError(RuntimeError):
    """Raised when the frozen task list cannot be honoured exactly."""


def build_task_lists(
    specs: dict[str, int],
    seed: int,
    cache_dir: Path | None = None,
    preserve: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Materialise the task selection into a serialisable record.

    ``specs`` maps benchmark name to task count, because the benchmarks are not
    all drawn at the same size (GSM8K carries 500 to power the open-numeric
    honest-q calibration; the rest carry 200).

    ``preserve`` maps a benchmark to task ids that MUST appear in its list. This
    is what makes "extend the frozen 100" mean what it says. It is not
    decoration: only MMLU's sampler nests, because its subject-stratified
    round-robin walks a seeded per-subject permutation and simply walks further
    for a larger ``n``. MATH-500 draws with ``rng.choice(N, size=n)``, and that
    is *not* nested -- re-drawing at n=200 under the identical seed retains only
    33 of the frozen 100. Extending by preserving explicitly, then filling the
    remainder from the complement with the same seed, is therefore the only way
    to grow a list without silently rotating the study onto other questions.
    """
    from aip.tasks import get_benchmark

    preserve = preserve or {}
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "n_per_benchmark": dict(specs),
        "selection_rule": (
            "Benchmark.load(n, seed): seeded uniform subsample without replacement, "
            "returned in dataset order. MMLU and ARC-style multiple choice keep the "
            "subject-stratified round-robin draw where the benchmark defines one; "
            "BoolQ draws label-balanced (equal Yes/No) rather than at its natural "
            "62/38 rate. Where a benchmark has preserved ids, those are carried "
            "verbatim and only the remainder is drawn, from the complement, under "
            "the same seed."
        ),
        "benchmarks": {},
    }
    for name, n in specs.items():
        benchmark = get_benchmark(name)
        keep_ids = list(preserve.get(name, []))
        if keep_ids:
            all_items = benchmark.load(n=None, seed=seed, cache_dir=cache_dir)
            by_id = {it.task_id: it for it in all_items}
            missing = [t for t in keep_ids if t not in by_id]
            if missing:
                raise TaskListError(
                    f"{len(missing)} preserved ids for {name!r} are absent from "
                    f"{benchmark.hf_path} (first few: {missing[:3]}); refusing to "
                    "build a list that silently drops them."
                )
            kept = [by_id[t] for t in keep_ids]
            if len(kept) >= n:
                items = kept[:n]
            else:
                keep_set = set(keep_ids)
                complement = [it for it in all_items if it.task_id not in keep_set]
                extra = benchmark.subsample(complement, n=n - len(kept), seed=seed)
                order = {it.task_id: i for i, it in enumerate(all_items)}
                items = sorted(kept + extra, key=lambda it: order[it.task_id])
            log.info(
                "task_lists.extended",
                benchmark=name,
                preserved=len(kept),
                drawn=len(items) - len(kept),
                total=len(items),
            )
        else:
            items = benchmark.load(n=n, seed=seed, cache_dir=cache_dir)

        record: dict[str, Any] = {
            "hf_path": benchmark.hf_path,
            "hf_config": benchmark.hf_config,
            "hf_split": benchmark.hf_split,
            "prompt_template_hash": benchmark.prompt_template_hash(),
            "answer_kind": str(benchmark.answer_kind),
            "n": len(items),
            "n_preserved": len(keep_ids),
            "task_ids": [it.task_id for it in items],
        }
        subjects: dict[str, int] = {}
        for it in items:
            subject = it.metadata.get("subject")
            if subject:
                subjects[str(subject)] = subjects.get(str(subject), 0) + 1
        if subjects:
            record["n_subjects"] = len(subjects)
            record["subject_counts"] = dict(sorted(subjects.items()))
        payload["benchmarks"][benchmark.name] = record
    return payload


def write_task_lists(payload: dict[str, Any], path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "task_lists.write",
        path=str(path),
        benchmarks={k: v["n"] for k, v in payload["benchmarks"].items()},
    )
    return path


def read_task_lists(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    if not Path(path).exists():
        raise TaskListError(
            f"{path} not found. Generate it with scripts/make_task_lists.py before "
            "running any phase that caches predictions."
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_fixed_tasks(
    benchmark_name: str, path: Path = DEFAULT_PATH, limit: int | None = None
) -> list[TaskItem]:
    """Load exactly the tasks named in the frozen list, in recorded order.

    Raises :class:`TaskListError` if any recorded id is missing from the dataset,
    rather than quietly returning a smaller set.
    """
    from aip.tasks import get_benchmark

    payload = read_task_lists(path)
    benchmark = get_benchmark(benchmark_name)
    record = payload["benchmarks"].get(benchmark.name)
    if record is None:
        raise TaskListError(f"no frozen task list for benchmark {benchmark.name!r} in {path}")

    wanted = list(record["task_ids"])
    if limit is not None:
        wanted = wanted[:limit]

    by_id = {it.task_id: it for it in benchmark.load(n=None, seed=int(payload["seed"]))}
    missing = [tid for tid in wanted if tid not in by_id]
    if missing:
        raise TaskListError(
            f"{len(missing)} frozen task ids are absent from {benchmark.hf_path} "
            f"(first few: {missing[:3]}). The dataset revision may have changed; "
            "the study cannot silently run on different questions."
        )
    return [by_id[tid] for tid in wanted]
