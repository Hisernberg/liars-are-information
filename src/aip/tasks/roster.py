"""The benchmark roster, derived rather than hardcoded.

Four downstream scripts each carried their own ``BENCHMARKS = ["gsm8k",
"math500", "mmlu"]``. That was survivable when the study had three benchmarks and
one hardware regime; with six benchmarks it is four places to forget. Worse, the
same scripts special-cased MMLU as "the multiple-choice one" -- true then, false
now that medqa and arc are also C=4 and boolq is C=2.

So both facts are read from the files that already own them:

* which benchmarks exist -> ``configs/task_lists.json``, which is the committed
  authority on what the study runs on;
* what chance-level coherence is for each -> the answer-space class in
  ``configs/inversion_thresholds.yaml``, which is where the calibration lives.

Neither is re-derived from a literal anywhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

DEFAULT_TASK_LISTS = Path("configs/task_lists.json")
DEFAULT_THRESHOLDS = Path("configs/inversion_thresholds.yaml")


def benchmarks(path: Path | str = DEFAULT_TASK_LISTS) -> list[str]:
    """Every benchmark in the frozen task list, in recorded order."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["benchmarks"])


def answer_space_class(benchmark: str, path: Path | str = DEFAULT_THRESHOLDS) -> str:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return str(raw["benchmark_classes"][benchmark])


def chance_coherence(benchmark: str, path: Path | str = DEFAULT_THRESHOLDS) -> float:
    """Chance-level q for this benchmark's answer space.

    ``q`` is P(two agents give the same answer | both wrong), so chance is
    1/(C-1) on a closed space of size C and 0 on an open one. On a binary space
    it is 1.0 identically -- with one wrong answer available, two wrong agents
    always agree -- which is why the binary class is degenerate and its gate can
    never fire.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    klass = raw["benchmark_classes"][benchmark]
    return float(raw["classes"][klass]["chance"])


def n_options(benchmark: str, path: Path | str = DEFAULT_THRESHOLDS) -> int | None:
    """Closed label-space size, or ``None`` for an open answer space."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    klass = raw["benchmark_classes"][benchmark]
    if klass == "binary":
        return 2
    return raw["classes"][klass].get("n_options") if klass == "multiple_choice" else None


def label_space(benchmark: str, path: Path | str = DEFAULT_THRESHOLDS) -> list[str] | None:
    """Letters for a closed space, ``None`` for an open one."""
    n = n_options(benchmark, path)
    return None if n is None else [chr(ord("A") + i) for i in range(n)]
