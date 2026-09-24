"""Pre-committed decision rules, read from ``configs/decision_rules.yaml``.

The rules exist so that an unattended batch run makes the same calls a careful
reader would, without a reader. Keeping them in a config that the code reads --
rather than in comments, or in an agent's head -- is what makes that true: a rule
that lives only in prose is a rule that gets applied inconsistently by a job
nobody is watching.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path("configs/decision_rules.yaml")


@dataclass(frozen=True, slots=True)
class Rules:
    raw: dict[str, Any]

    # -- R1 ---------------------------------------------------------------
    @property
    def ceiling(self) -> float:
        return float(self.raw["r1_ceiling"]["ceiling"])

    @property
    def headline_benchmarks(self) -> tuple[str, ...]:
        return tuple(self.raw["r1_ceiling"]["headline_benchmarks"])

    @property
    def appendix_benchmarks(self) -> tuple[str, ...]:
        return tuple(self.raw["r1_ceiling"]["appendix_benchmarks"])

    def is_headroom_low(self, accuracy: float) -> bool:
        return accuracy >= self.ceiling

    def headline_eligible(self, benchmark: str, accuracy: float) -> bool:
        """A cell may carry a headline figure.

        Headline benchmarks qualify on headroom alone. Appendix benchmarks --
        the ones current models saturate -- qualify only where the specific cell
        still has headroom, which is a property of the cell rather than of the
        tier that produced it.
        """
        if self.is_headroom_low(accuracy):
            return False
        if benchmark in self.headline_benchmarks:
            return True
        return bool(self.raw["r1_ceiling"].get("appendix_headline_requires_headroom", True))

    # -- R2 ---------------------------------------------------------------
    @property
    def min_wrong_for_auc(self) -> int:
        return int(self.raw["r2_auc_power"]["min_wrong"])

    def auc_is_interpretable(self, n_wrong: int) -> bool:
        return n_wrong >= self.min_wrong_for_auc

    # -- R4 ---------------------------------------------------------------
    @property
    def gpu_hours_cap(self) -> float:
        return float(self.raw["r4_budget"]["total_gpu_hours_remaining"])

    @property
    def t07_cap_seconds(self) -> int:
        return int(self.raw["r4_budget"]["per_model_t07_cap_seconds"])

    @property
    def cut_order(self) -> tuple[str, ...]:
        return tuple(self.raw["r4_budget"]["cut_order"])

    # -- R5 ---------------------------------------------------------------
    @property
    def attack_order(self) -> tuple[str, ...]:
        return tuple(self.raw["r5_attack_priority"]["order"])

    @property
    def attack_headroom_gated(self) -> tuple[str, ...]:
        return tuple(self.raw["r5_attack_priority"]["headroom_gated"])

    # -- R6 ---------------------------------------------------------------
    @property
    def runlog(self) -> Path:
        return Path(self.raw["r6_logging"]["runlog"])

    @property
    def status_file(self) -> Path:
        return Path(self.raw["r6_logging"]["status"])


@lru_cache(maxsize=4)
def load_rules(path: Path | str = DEFAULT_PATH) -> Rules:
    return Rules(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
