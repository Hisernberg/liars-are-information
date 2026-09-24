"""Task-separated, receiver-conditioned clean-channel calibration.

The upper bootstrap bounds are exploratory calibration heuristics. They do not
provide a family-wise error guarantee for the selected-partner AIP gate.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from aip.aggregation.aip import InversionThresholds
from aip.types import Observation


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], "big")


def split_task_ids(task_ids: Sequence[str], seed: int = 94027) -> dict[str, list[str]]:
    """Deterministic 20/30/20/30 CAL/HISTORY/VALIDATION/TEST, independent of labels."""
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task IDs must be unique")
    if len(task_ids) < 30:
        raise ValueError("At least 30 tasks are required for four nonempty partitions")
    ordered = sorted(task_ids, key=lambda task_id: stable_seed(seed, task_id))
    a, b, c = int(.2 * len(ordered)), int(.5 * len(ordered)), int(.7 * len(ordered))
    return dict(calibration=ordered[:a], history=ordered[a:b], validation=ordered[b:c], test=ordered[c:])


def validate_partitions(partitions: Mapping[str, Sequence[str]]) -> None:
    required = {"calibration", "history", "validation", "test"}
    if set(partitions) != required:
        raise ValueError("Require calibration/history/validation/test partitions")
    seen: set[str] = set()
    for name, task_ids in partitions.items():
        if not task_ids or len(task_ids) != len(set(task_ids)) or seen.intersection(task_ids):
            raise ValueError(f"Empty, duplicate, or overlapping partition: {name}")
        seen.update(task_ids)


def public_observation(observation: Observation) -> Observation:
    """Remove hidden attacker/model annotations before any defender sees data."""
    return replace(observation, broadcasts=tuple(
        replace(b, is_byzantine=False, source_model=None, attack=None)
        for b in observation.broadcasts
    ))


@dataclass(frozen=True)
class CalibrationEstimate:
    receiver: int
    n_calibration_tasks: int
    n_peers: int
    n_pairs: int
    n_eligible_pairs: int
    joint_dissent_events: int
    coincidence_events: int
    pooled_point: float | None
    pooled_ci_low: float | None
    pooled_ci_high: float | None
    partner_max_point: float | None
    partner_max_ci_low: float | None
    partner_max_ci_high: float | None
    pooled_ceiling: float
    partner_ceiling: float
    status: str
    min_pair_support: int
    bootstrap_resamples: int
    calibration_task_sha256: str
    heuristic: str = "task-bootstrap upper .975; no selection/FWER guarantee"

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate_receiver(
    observations: Sequence[Observation],
    *,
    calibration_ids: Sequence[str],
    receiver: int,
    forbidden_ids: Sequence[str] = (),
    n_options: int | None = None,
    resamples: int = 300,
    seed: int = 0,
    min_pair_support: int = 5,
) -> CalibrationEstimate:
    """Calibrate from complete CAL membership and a single receiver's view.

    Missing peer reports remain missing. Each resampled unit is an entire task,
    preserving every peer/partner event inside it. Pooled calibration uses the
    ratio of all coincidence and joint-dissent events. Partner calibration
    bootstraps the maximum eligible pair rate for this receiver; its operational
    ceiling is at least the pooled upper bound. No ground-truth label is used.
    """
    ids = [obs.task_id for obs in observations]
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(calibration_ids):
        raise ValueError("Calibration observations must cover every unique CAL task exactly once")
    if len(calibration_ids) != len(set(calibration_ids)) or set(ids).intersection(forbidden_ids):
        raise ValueError("Calibration IDs duplicate or overlap a forbidden partition")
    if any(obs.self_id != receiver for obs in observations):
        raise ValueError("Every calibration observation must belong to the specified receiver")
    if resamples < 20 or min_pair_support < 1:
        raise ValueError("Require at least 20 bootstrap resamples and positive pair support")
    observations = sorted(observations, key=lambda obs: obs.task_id)
    peers = sorted({b.agent_id for obs in observations for b in obs.peers})
    pairs = list(itertools.combinations(peers, 2))
    num = np.zeros((len(observations), len(pairs)), dtype=np.int64)
    den = np.zeros_like(num)
    for t, obs in enumerate(observations):
        own = obs.own.answer
        if own is None:
            continue
        answers = {b.agent_id: b.answer for b in obs.broadcasts}
        for j, (left, right) in enumerate(pairs):
            a, b = answers.get(left), answers.get(right)
            if a is not None and b is not None and a != own and b != own:
                den[t, j] = 1
                num[t, j] = int(a == b)
    totals_num, totals_den = num.sum(axis=0), den.sum(axis=0)
    eligible = totals_den >= min_pair_support
    total_num, total_den = int(totals_num.sum()), int(totals_den.sum())
    point = total_num / total_den if total_den else None
    maxima = float(np.max(totals_num[eligible] / totals_den[eligible])) if eligible.any() else None
    fields = dict(receiver=receiver, n_calibration_tasks=len(ids), n_peers=len(peers),
        n_pairs=len(pairs), n_eligible_pairs=int(eligible.sum()), joint_dissent_events=total_den,
        coincidence_events=total_num, pooled_point=point, partner_max_point=maxima,
        min_pair_support=min_pair_support, bootstrap_resamples=resamples,
        calibration_task_sha256=hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest())
    if n_options == 2:
        return CalibrationEstimate(**fields, pooled_ci_low=1., pooled_ci_high=1.,
            partner_max_ci_low=1., partner_max_ci_high=1., pooled_ceiling=1.,
            partner_ceiling=1., status="binary_structural_ceiling")
    if not total_den:
        return CalibrationEstimate(**fields, pooled_ci_low=None, pooled_ci_high=None,
            partner_max_ci_low=None, partner_max_ci_high=None, pooled_ceiling=1.,
            partner_ceiling=1., status="no_joint_dissent_conservative_ceiling")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=resamples)
    draws_num, draws_den = weights @ num, weights @ den
    pooled = np.divide(draws_num.sum(axis=1), draws_den.sum(axis=1),
        out=np.ones(resamples), where=draws_den.sum(axis=1) > 0)
    rates = np.divide(draws_num, draws_den, out=np.zeros_like(draws_num, dtype=float),
        where=draws_den >= min_pair_support)
    selected = rates.max(axis=1)
    selected[~(draws_den >= min_pair_support).any(axis=1)] = 1.
    low, high = map(float, np.quantile(pooled, [.025, .975]))
    selected_low, selected_high = map(float, np.quantile(selected, [.025, .975]))
    # Zero is a degenerate threshold in the inherited gate implementation.
    pooled_ceiling = float(np.clip(high, 1e-6, 1.))
    partner_ceiling = max(pooled_ceiling, selected_high) if eligible.any() else 1.
    return CalibrationEstimate(**fields, pooled_ci_low=low, pooled_ci_high=high,
        partner_max_ci_low=selected_low, partner_max_ci_high=selected_high,
        pooled_ceiling=pooled_ceiling, partner_ceiling=partner_ceiling,
        status="estimated" if eligible.any() else "no_eligible_partner_conservative_ceiling")


def calibrate_partition(
    observations_by_id: Mapping[str, Observation],
    partitions: Mapping[str, Sequence[str]],
    *,
    receiver: int,
    **kwargs,
) -> CalibrationEstimate:
    """Select CAL before estimation; HISTORY/VALIDATION/TEST never enter it."""
    validate_partitions(partitions)
    ids = partitions["calibration"]
    if not set(ids).issubset(observations_by_id):
        raise ValueError("Missing calibration observations")
    forbidden = [task_id for name, part in partitions.items() if name != "calibration" for task_id in part]
    return calibrate_receiver([public_observation(observations_by_id[t]) for t in ids],
        calibration_ids=ids, forbidden_ids=forbidden, receiver=receiver, **kwargs)


def thresholds_with_ceiling(
    legacy: InversionThresholds, benchmark: str, ceiling: float,
) -> InversionThresholds:
    """Create a separate benchmark calibration; never mutate frozen thresholds."""
    if not np.isfinite(ceiling) or not 0 < ceiling <= 1:
        raise ValueError("Operative ceiling must be finite and in (0,1]")
    klass = f"independent_calibration_{benchmark}"
    return replace(legacy, ceilings={**legacy.ceilings, klass: float(ceiling)},
        chances={**legacy.chances, klass: legacy.chance_for(benchmark)},
        benchmark_classes={**legacy.benchmark_classes, benchmark: klass})
