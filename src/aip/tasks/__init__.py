"""Benchmark loaders, prompt builders, answer extractors, and scorers."""

from __future__ import annotations

from aip.tasks.arc import ARC
from aip.tasks.base import Benchmark, parse_self_reported_confidence
from aip.tasks.boolq import BoolQ
from aip.tasks.gsm8k import GSM8K
from aip.tasks.math500 import MATH500
from aip.tasks.medqa import MedQA
from aip.tasks.mmlu import MMLU

BENCHMARKS: dict[str, type[Benchmark]] = {
    GSM8K.name: GSM8K,
    MATH500.name: MATH500,
    MMLU.name: MMLU,
    MedQA.name: MedQA,
    BoolQ.name: BoolQ,
    ARC.name: ARC,
}


def get_benchmark(name: str) -> Benchmark:
    """Instantiate a benchmark by registry name."""
    key = name.strip().lower()
    aliases = {
        "mmlu_subset": "mmlu",
        "math_500": "math500",
        "math-500": "math500",
        "med_qa": "medqa",
        "bool_q": "boolq",
        "arc_challenge": "arc",
        "arc-challenge": "arc",
    }
    key = aliases.get(key, key)
    if key not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}")
    return BENCHMARKS[key]()


__all__ = [
    "ARC",
    "BENCHMARKS",
    "Benchmark",
    "BoolQ",
    "GSM8K",
    "MATH500",
    "MMLU",
    "MedQA",
    "get_benchmark",
    "parse_self_reported_confidence",
]
