#!/usr/bin/env python3
"""Phase C: the offline aggregation sweep.

Cache-only. Honest agents replay Phase A answers; Byzantine agents are symbolic
(:mod:`aip.swarm.broadcast`). No model is loaded.

Sweep axes: benchmark x swarm composition x Byzantine fraction f x partial
observability p_obs x topology x adversary x normalization parity x method.
Every cell reports accuracy with a bootstrap CI over tasks.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.aggregation.baselines import (  # noqa: E402
    ConfidenceWeighted,
    CoordinateMedian,
    DawidSkeneAggregator,
    GeometricMedian,
    Krum,
    MajorityVote,
    MultiKrum,
    SACFilterRefine,
    TrimmedMean,
)
from aip.analysis.bootstrap import bootstrap_statistic  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest, stable_seed  # noqa: E402
from aip.swarm.assignment import assign_swarm  # noqa: E402
from aip.swarm.broadcast import AdversaryConfig, build_broadcasts, observe  # noqa: E402
from aip.swarm.topology import build_topology  # noqa: E402
from aip.tasks import roster  # noqa: E402

log = configure_logging(level="INFO")

BENCHMARKS = roster.benchmarks()  # from configs/task_lists.json
#: Closed answer spaces get their full option set; open ones get None and fall
#: back to the per-task observed candidates. Derived from each benchmark's
#: answer-space class rather than listed, because this dict is imported as
#: `pc.LABEL_SPACES` by phase_d_sweep, phase_d_bandit, phase_e_burst and
#: phase_e_decomposition -- a benchmark missing here is a KeyError four scripts
#: downstream, which is exactly how Phase 3 died on `medqa`.
LABEL_SPACES: dict[str, list[str] | None] = {b: roster.label_space(b) for b in BENCHMARKS}
N_AGENTS = 10
F_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
P_OBS_GRID = [1.0, 0.5, 0.25, 0.1]


@dataclass
class BenchmarkCache:
    """Cached answers per (model, sample slot), aligned on shared task ids."""

    benchmark: str
    task_ids: list[str]
    gold: list[str]
    # (model, slot) -> per-task (answer, logprob_conf, self_conf)
    answers: dict[tuple[str, int], list[tuple[str | None, float, float]]]

    def slots_for(self, model: str) -> list[int]:
        return sorted(s for (m, s) in self.answers if m == model)


def load_cache(benchmark: str, models: list[str], root: Path, root_t07: Path) -> BenchmarkCache:
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    shared: pd.Index | None = None
    for m in models:
        f = pd.read_parquet(root / benchmark / f"{m}.parquet").set_index("task_id").sort_index()
        frames[(m, 0)] = f
        shared = f.index if shared is None else shared.intersection(f.index)
        alt = root_t07 / benchmark / f"{m}.parquet"
        if alt.exists():
            frames[(m, 1)] = pd.read_parquet(alt).set_index("task_id").sort_index()

    task_ids = list(shared)
    gold = frames[(models[0], 0)].loc[task_ids, "gold_answer"].astype(str).tolist()
    answers: dict[tuple[str, int], list[tuple[str | None, float, float]]] = {}
    for key, frame in frames.items():
        if not set(task_ids).issubset(set(frame.index)):
            continue
        sub = frame.loc[task_ids]
        answers[key] = [
            (
                None if pd.isna(a) else str(a),
                0.0 if pd.isna(lc) else float(lc),
                0.0 if pd.isna(sc) else float(sc),
            )
            for a, lc, sc in zip(
                sub["extracted_answer"],
                sub["logprob_confidence"],
                sub["self_reported_confidence"],
                strict=True,
            )
        ]
    return BenchmarkCache(benchmark, task_ids, gold, answers)


def compositions_for(
    benchmark: str, models: list[str], accuracy: dict[str, float]
) -> dict[str, list[str]]:
    """Homogeneous swarms, the uniform 4-model mix, and a strong+weak pair."""
    ranked = sorted(models, key=lambda m: accuracy[m])
    out: dict[str, list[str]] = {f"hom_{m}": [m] for m in models}
    out["mix_all4"] = list(models)
    out["mix_strong_weak"] = [ranked[-1], ranked[0]]
    return out


def build_task_observations(
    cache: BenchmarkCache,
    composition: list[str],
    f: float,
    p_obs: float,
    topology_name: str,
    topology_k: int | None,
    adversary: AdversaryConfig,
    seed: int,
):
    """All tasks' observations for one swarm configuration."""
    rng = np.random.default_rng(seed)
    assignment = assign_swarm(N_AGENTS, f, composition, rng)
    topology = build_topology(topology_name, N_AGENTS, topology_k)
    label_space = LABEL_SPACES[cache.benchmark] or []

    # Homogeneous swarms replaying a single greedy cache are degenerate: every
    # honest agent gives the identical answer, so no aggregator can do better
    # than the model alone. Where a temperature-0.7 second sample exists, honest
    # agents alternate between slots so a homogeneous swarm has the within-model
    # diversity a real one would. Where it does not, the degeneracy is itself the
    # Phase B finding (N_eff ~ 1) and is reported rather than papered over.
    slot_of: dict[int, int] = {}
    for agent, model in enumerate(assignment.models):
        if model is None:
            continue
        slots = cache.slots_for(model)
        slot_of[agent] = slots[agent % len(slots)] if len(slots) > 1 else 0

    tasks = []
    for t, (task_id, gold) in enumerate(zip(cache.task_ids, cache.gold, strict=True)):
        per_agent_source: dict[str, tuple[str | None, float, float]] = {}
        agent_models: list[str | None] = []
        for agent, model in enumerate(assignment.models):
            if model is None:
                agent_models.append(None)
                continue
            key = f"{model}#{slot_of[agent]}"
            per_agent_source[key] = cache.answers[(model, slot_of[agent])][t]
            agent_models.append(key)
        pseudo = type(assignment)(
            n_agents=assignment.n_agents,
            byzantine=assignment.byzantine,
            models=tuple(agent_models),
            composition=assignment.composition,
        )
        broadcasts = build_broadcasts(
            task_id, gold, pseudo, per_agent_source, label_space, adversary, rng
        )
        tasks.append(observe(broadcasts, topology, p_obs, rng))
    return assignment, tasks


def make_aggregators(
    benchmark: str,
    n_byzantine: int,
    f: float,
    parity: ParityConfig,
    thresholds: InversionThresholds,
) -> list[Any]:
    labels = LABEL_SPACES[benchmark]
    return [
        MajorityVote(),
        ConfidenceWeighted(parity=parity),
        ConfidenceWeighted(parity=parity, use_self_report=True),
        GeometricMedian(),
        Krum(n_byzantine),
        MultiKrum(n_byzantine),
        CoordinateMedian(),
        TrimmedMean(trim=max(f, 0.1)),
        DawidSkeneAggregator(label_space=labels),
        DawidSkeneAggregator(label_space=labels, use_anchors=True),
        SACFilterRefine(parity=parity),
        AIPAggregator(benchmark, "gated", thresholds, parity, labels),
        AIPAggregator(benchmark, "naive", thresholds, parity, labels),
        AIPAggregator(benchmark, "trust_only", thresholds, parity, labels),
    ]


def evaluate_cell(
    cache: BenchmarkCache, assignment, tasks, aggregators, resamples: int, seed: int
) -> list[dict[str, Any]]:
    """Per-method accuracy over honest receivers, with a bootstrap CI over tasks."""
    honest = [a for a in range(N_AGENTS) if a not in assignment.byzantine]
    rows = []
    for agg in aggregators:
        if agg.needs_fit:
            agg.fit(tasks)
        per_task = np.zeros(len(tasks), dtype=float)
        for t, observations in enumerate(tasks):
            if not honest:
                per_task[t] = np.nan
                continue
            hits = 0
            for agent in honest:
                obs = observations[agent]
                hits += int(agg.aggregate(obs.broadcasts, agent) == cache.gold[t])
            per_task[t] = hits / len(honest)

        valid = per_task[np.isfinite(per_task)]

        def stat(idx: np.ndarray, v=valid) -> float:
            return float(np.mean(v[idx])) if idx.size else float("nan")

        ci = bootstrap_statistic(stat, valid.size, resamples, seed=seed)
        row: dict[str, Any] = {
            "method": agg.name,
            "is_oracle": bool(getattr(agg, "is_oracle", False)),
            "n_honest_receivers": len(honest),
        }
        row.update(ci.as_dict("accuracy_"))
        if isinstance(agg, AIPAggregator):
            byz = set(assignment.byzantine)
            inverted, total = agg.diagnostics.honest_inversions(byz)
            adv_inv, adv_tot = agg.diagnostics.adversary_inversions(byz)
            counts = agg.diagnostics.decision_counts(set(honest))
            row["honest_channels_inverted"] = inverted
            row["honest_channels_total"] = total
            row["honest_inversion_rate"] = (inverted / total) if total else np.nan
            row["adversary_channels_inverted"] = adv_inv
            row["adversary_channels_total"] = adv_tot
            row["adversary_inversion_rate"] = (adv_inv / adv_tot) if adv_tot else np.nan
            row["n_trust"] = counts["trust"]
            row["n_invert"] = counts["invert"]
            row["n_discard"] = counts["discard"]
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--cache-t07", type=Path, default=Path("data/cache_t07"))
    parser.add_argument("--out", type=Path, default=Path("results/aggregation"))
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20250825)
    parser.add_argument("--benchmarks", nargs="*", default=BENCHMARKS)
    parser.add_argument("--quick", action="store_true", help="small grid, for smoke testing")
    args = parser.parse_args()

    thresholds = InversionThresholds.load()
    models = sorted(p.stem for p in (args.cache / "gsm8k").glob("*.parquet"))
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / ("sweep_quick.parquet" if args.quick else "sweep.parquet")

    f_grid = [0.0, 0.3, 0.6] if args.quick else F_GRID
    p_grid = [1.0, 0.25] if args.quick else P_OBS_GRID
    adversaries = [
        AdversaryConfig(kind="always_wrong", error_rate=1.0),
        AdversaryConfig(kind="noise", error_rate=1.0),
    ]
    parities = [ParityConfig(share=0.1), ParityConfig(share=None)]

    done_keys: set[tuple] = set()
    existing: list[pd.DataFrame] = []
    if out_path.exists():
        prior = pd.read_parquet(out_path)
        existing.append(prior)
        done_keys = set(
            map(
                tuple,
                prior[["benchmark", "composition", "f", "p_obs", "topology", "adversary", "parity"]]
                .drop_duplicates()
                .to_numpy()
                .tolist(),
            )
        )
        log.info("phase_c.resume", cells_done=len(done_keys))

    rows: list[dict[str, Any]] = []
    for benchmark in args.benchmarks:
        cache = load_cache(benchmark, models, args.cache, args.cache_t07)
        accuracy = {
            m: float(
                np.mean(
                    [a == g for (a, _, _), g in zip(cache.answers[(m, 0)], cache.gold, strict=True)]
                )
            )
            for m in models
        }
        comps = compositions_for(benchmark, models, accuracy)
        log.info(
            "phase_c.benchmark",
            benchmark=benchmark,
            n_tasks=len(cache.task_ids),
            accuracy={k: round(v, 3) for k, v in accuracy.items()},
        )

        topologies = (
            [("complete", None)]
            if args.quick
            else [("complete", None), ("ring", None), ("k_nearest", 4)]
        )
        for comp_name, composition in comps.items():
            for f, p_obs, (topo_name, topo_k), adversary, parity in itertools.product(
                f_grid, p_grid, topologies, adversaries, parities
            ):
                # Topology and parity sensitivity only at the reference p_obs, to
                # keep the grid from multiplying without adding information.
                if topo_name != "complete" and p_obs != 1.0:
                    continue
                if parity.share is None and (p_obs != 1.0 or topo_name != "complete"):
                    continue
                if adversary.kind == "noise" and (p_obs != 1.0 or topo_name != "complete"):
                    continue

                key = (benchmark, comp_name, f, p_obs, topo_name, adversary.kind, parity.label)
                if key in done_keys:
                    continue
                # The swarm seed deliberately EXCLUDES parity. Parity is a property
                # of the aggregation rule, not of the swarm, so matched and unmatched
                # must be evaluated on the identical draw. Including it made the audit
                # compare two different swarms: unweighted methods such as Krum, which
                # cannot respond to parity at all, showed the largest apparent parity
                # effect in the table because they were measuring seed variance.
                swarm_key = (benchmark, comp_name, f, p_obs, topo_name, adversary.kind)
                assignment, tasks = build_task_observations(
                    cache,
                    composition,
                    f,
                    p_obs,
                    topo_name,
                    topo_k,
                    adversary,
                    seed=args.seed + stable_seed(swarm_key),
                )
                aggs = make_aggregators(benchmark, len(assignment.byzantine), f, parity, thresholds)
                for row in evaluate_cell(cache, assignment, tasks, aggs, args.resamples, args.seed):
                    row.update(
                        benchmark=benchmark,
                        composition=comp_name,
                        f=f,
                        p_obs=p_obs,
                        topology=topo_name,
                        topology_k=topo_k,
                        adversary=adversary.kind,
                        parity=parity.label,
                        n_agents=N_AGENTS,
                        n_byzantine=len(assignment.byzantine),
                        n_tasks=len(cache.task_ids),
                    )
                    rows.append(row)
            log.info(
                "phase_c.composition_done",
                benchmark=benchmark,
                composition=comp_name,
                rows=len(rows),
            )
        # Checkpoint after each benchmark so a long sweep is resume-safe.
        frame = pd.concat(existing + [pd.DataFrame(rows)], ignore_index=True)
        frame.to_parquet(out_path, index=False)

    frame = pd.concat(existing + [pd.DataFrame(rows)], ignore_index=True) if rows else existing[0]
    frame.to_parquet(out_path, index=False)

    manifest = Manifest.create(
        phase="phase_c",
        config={
            "resamples": args.resamples,
            "models": models,
            "n_agents": N_AGENTS,
            "f_grid": f_grid,
            "p_obs_grid": p_grid,
            "quick": args.quick,
        },
        seeds={"sweep": args.seed},
        notes={
            "inference_run": False,
            "thresholds": "configs/inversion_thresholds.yaml",
            "n_rows": len(frame),
        },
    )
    manifest.finish(status="ok").write(Path("results/manifests") / f"{manifest.run_id}.json")
    log.info("phase_c.done", rows=len(frame), path=str(out_path))
    print(f"\nwrote {out_path}  ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
