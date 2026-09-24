"""Cached answers, gold-free canonicalisation, and scoring.

All inputs are the X3-repaired caches (post 2026-09-20 truncation repair):

* ``data/cache/<benchmark>/<model>.parquet`` -- honest greedy answers;
* ``data/cache_adversarial/<attack>/<benchmark>/<model>.parquet`` -- answers
  written by an LLM that was *prompted to deceive* (four attack prompts, two
  attacker models);
* ``data/cache_t07`` -- a temperature-0.7 second sample (GSM8K only).

Open-answer benchmarks are canonicalised **without gold**: two answers on the
same task are merged when the repository's equivalence checker says they denote
the same value (``math_match`` for MATH-500, numeric equality for GSM8K). This is
something a deployed receiver can do -- it compares answers with each other,
never with the key -- and it is applied identically to every aggregation method.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from aip.tasks.gsm8k import normalize_number, numeric_match
from aip.tasks.math500 import math_match

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

BENCHMARKS = ("mmlu", "medqa", "arc", "boolq", "gsm8k", "math500")
LABEL_SPACES: dict[str, tuple[str, ...] | None] = {
    "mmlu": ("A", "B", "C", "D"),
    "medqa": ("A", "B", "C", "D"),
    "arc": ("A", "B", "C", "D"),
    "boolq": ("A", "B"),
    "gsm8k": None,
    "math500": None,
}
FROZEN_MODELS = (
    "gemma4_31b",
    "granite42_30b",
    "llama32_3b",
    "ministral3_14b",
    "olmo3_32b_think",
    "phi4_mini_reasoning",
    "qwen38_27b",
)
WEAK_MODELS = ("ministral_8b", "olmo2_7b")
ATTACK_PROMPTS = ("always_wrong", "rushing", "semantic_hallucination", "semantic_negation")
ATTACKER_MODELS = ("llama32_3b", "ministral3_14b")


def is_open(benchmark: str) -> bool:
    return LABEL_SPACES[benchmark] is None


@lru_cache(maxsize=200_000)
def score(benchmark: str, answer: str | None, gold: str) -> bool:
    """Gold scoring, used only by the evaluator -- never by a defence."""
    if answer is None:
        return False
    if benchmark == "math500":
        return bool(math_match(answer, gold))
    if benchmark == "gsm8k":
        return bool(numeric_match(answer, gold))
    return answer == gold


def _equivalent(benchmark: str, a: str, b: str) -> bool:
    if a == b:
        return True
    if benchmark == "math500":
        return bool(math_match(a, b)) or bool(math_match(b, a))
    if benchmark == "gsm8k":
        return bool(numeric_match(a, b))
    return False


def canonical_map(benchmark: str, answers: set[str]) -> dict[str, str]:
    """Union-find over pairwise equivalence; representative = shortest, then lexicographic."""
    items = sorted(answers)
    if benchmark == "gsm8k":
        items = sorted({a for a in items})
    parent = {a: a for a in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            ra, rb = find(a), find(b)
            if ra != rb and _equivalent(benchmark, a, b):
                parent[rb] = ra
    groups: dict[str, list[str]] = {}
    for a in items:
        groups.setdefault(find(a), []).append(a)
    out: dict[str, str] = {}
    for members in groups.values():
        rep = sorted(members, key=lambda s: (len(s), s))[0]
        for m in members:
            out[m] = rep
    return out


@dataclass
class BenchmarkData:
    benchmark: str
    task_ids: list[str]
    gold: list[str]
    honest: dict[str, list[str | None]]
    """model -> canonical answer per task (greedy cache)."""
    logprob: dict[str, list[float]]
    adversarial: dict[tuple[str, str], list[str | None]] = field(default_factory=dict)
    """(attack prompt, attacker model) -> canonical answer per task."""
    canon: list[dict[str, str]] = field(default_factory=list)

    @property
    def label_space(self) -> tuple[str, ...] | None:
        return LABEL_SPACES[self.benchmark]

    def canonical(self, t: int, answer: str | None) -> str | None:
        if answer is None:
            return None
        if not is_open(self.benchmark):
            return answer
        return self.canon[t].get(answer, answer)

    def accuracy(self, model: str) -> float:
        return float(np.mean([score(self.benchmark, a, g) for a, g in zip(self.honest[model], self.gold, strict=True)]))


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.task_id.duplicated().any():
        frame = frame[frame.sample_index.astype(int) == 0] if "sample_index" in frame else frame
    return frame.set_index("task_id").sort_index()


def _canon_cache_path(benchmark: str) -> Path:
    return DATA / "derived" / f"canonical_{benchmark}.json"


@lru_cache(maxsize=16)
def load_benchmark(benchmark: str, models: tuple[str, ...] = FROZEN_MODELS + WEAK_MODELS) -> BenchmarkData:
    frames = {m: _read(DATA / "cache" / benchmark / f"{m}.parquet") for m in models}
    shared = None
    for f in frames.values():
        shared = f.index if shared is None else shared.intersection(f.index)
    task_ids = sorted(shared)
    gold = frames[models[0]].loc[task_ids, "gold_answer"].astype(str).tolist()
    for m, f in frames.items():
        if f.loc[task_ids, "gold_answer"].astype(str).tolist() != gold:
            raise ValueError(f"gold mismatch in {benchmark}/{m}")
    raw: dict[str, list[str | None]] = {}
    logprob: dict[str, list[float]] = {}
    for m, f in frames.items():
        sub = f.loc[task_ids]
        raw[m] = [None if pd.isna(a) else str(a) for a in sub.extracted_answer]
        logprob[m] = [0.0 if pd.isna(x) else float(x) for x in sub.logprob_confidence]
    adv_raw: dict[tuple[str, str], list[str | None]] = {}
    for attack in ATTACK_PROMPTS:
        for m in ATTACKER_MODELS:
            path = DATA / "cache_adversarial" / attack / benchmark / f"{m}.parquet"
            if not path.exists():
                continue
            f = _read(path)
            adv_raw[(attack, m)] = [
                None if (t not in f.index or pd.isna(f.at[t, "extracted_answer"])) else str(f.at[t, "extracted_answer"])
                for t in task_ids
            ]
    canon: list[dict[str, str]] = []
    if is_open(benchmark):
        cache_path = _canon_cache_path(benchmark)
        stored = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        dirty = False
        for t, tid in enumerate(task_ids):
            answers = {a for src in list(raw.values()) + list(adv_raw.values()) if (a := src[t]) is not None}
            key = tid
            if key in stored and set(stored[key]) >= answers:
                canon.append(stored[key])
                continue
            mapping = canonical_map(benchmark, answers)
            stored[key] = mapping
            canon.append(mapping)
            dirty = True
        if dirty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(stored, sort_keys=True))
    data = BenchmarkData(benchmark, task_ids, gold, {}, logprob, {}, canon)
    data.honest = {m: [data.canonical(t, a) for t, a in enumerate(v)] for m, v in raw.items()}
    data.adversarial = {k: [data.canonical(t, a) for t, a in enumerate(v)] for k, v in adv_raw.items()}
    return data


LIVE_MODELS = ("qwen25_1p5b", "smollm2_1p7b", "granite33_2b", "olmo2_1b", "llama32_1b", "gemma3_1b")
LIVE_ROLES = ("solo", "rushing", "collude")


@lru_cache(maxsize=16)
def load_live(benchmark: str, honest_dir: str = "cache") -> BenchmarkData:
    """Live-swarm caches written by ``experiments/live_swarm.py``.

    ``honest_dir`` is ``cache`` (independent round-1 answers) or ``cache_debate``
    (answers given after seeing a panel in which half the votes were lies).
    """
    root = DATA / "live_cache"
    frames = {m: _read(root / honest_dir / benchmark / f"{m}.parquet") for m in LIVE_MODELS}
    task_ids = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    gold = frames[LIVE_MODELS[0]].loc[task_ids, "gold_answer"].astype(str).tolist()
    honest = {m: [None if pd.isna(a) else str(a) for a in f.loc[task_ids, "extracted_answer"]] for m, f in frames.items()}
    logprob = {m: [float(x) for x in f.loc[task_ids, "logprob_confidence"]] for m, f in frames.items()}
    adversarial = {}
    for role in LIVE_ROLES:
        for m in LIVE_MODELS:
            path = root / "cache_adversarial" / role / benchmark / f"{m}.parquet"
            if path.exists():
                f = _read(path)
                adversarial[(role, m)] = [None if pd.isna(f.at[t, "extracted_answer"]) else str(f.at[t, "extracted_answer"])
                                          for t in task_ids]
    return BenchmarkData(benchmark, task_ids, gold, honest, logprob, adversarial, [])
