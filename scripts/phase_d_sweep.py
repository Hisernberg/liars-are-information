#!/usr/bin/env python3
"""Phase D sweep: the Phase C harness with real LLM adversaries.

Cache-only. Honest agents replay Phase A; Byzantine agents replay the Phase D
adversarial caches (or, for the three offline attacks, transforms and schedules
over cached outputs). No model is loaded.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import phase_c_aggregate as pc  # noqa: E402

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
from aip.attacks.burst import epoch_schedule  # noqa: E402
from aip.attacks.falsified_confidence import invert_self_report  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest, stable_seed  # noqa: E402
from aip.models.registry import load_registry  # noqa: E402
from aip.swarm.assignment import assign_swarm  # noqa: E402
from aip.swarm.broadcast import observe  # noqa: E402
from aip.swarm.topology import build_topology  # noqa: E402
from aip.types import Broadcast, Observation  # noqa: E402

log = configure_logging(level="INFO")

ADV_CACHE = Path("data/cache_adversarial")
#: Phase 4's adversarial generators, taken from the FROZEN roster rather than
#: named literally. The previous literal still said `ministral_8b`, a model
#: retired and disabled when the roster moved to Ministral 3 14B -- so Phase 4
#: was generating attacks with a checkpoint that is not part of this hardware
#: regime's population at all, which RULE 1 forbids outright.
#:
#: The weak/fast tier leads because compliance, not capability, is what makes a
#: good adversarial generator: llama32_3b followed the always_wrong injection
#: 100% of the time on the L4 regime where Ministral complied only 29%.
ADVERSARY_MODELS = [
    n
    for n in load_registry("configs/models.yaml").enabled_names
    if load_registry("configs/models.yaml").get(n).tier in ("weak_fast", "mid")
]
F_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
INFERENCE_ATTACKS = ["always_wrong", "semantic_negation", "semantic_hallucination", "rushing"]
OFFLINE_ATTACKS = ["falsified_confidence", "burst"]


def load_adversarial(
    attack: str, benchmark: str, task_ids: list[str]
) -> dict[str, list[tuple[str | None, float, float]]]:
    """Adversarial answers per model, aligned to the honest task order."""
    out: dict[str, list[tuple[str | None, float, float]]] = {}
    for model in ADVERSARY_MODELS:
        path = ADV_CACHE / attack / benchmark / f"{model}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path).set_index("task_id")
        if not set(task_ids).issubset(set(frame.index)):
            continue
        sub = frame.loc[task_ids]
        out[model] = [
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
    return out


def noise_answer(gold: str, pool: list[str], rng: np.random.Generator) -> str:
    """An incoherent wrong answer: an independent draw per adversary."""
    options = [p for p in pool if p is not None and p != gold]
    if options:
        return str(options[int(rng.integers(len(options)))])
    return f"{gold}__noise{int(rng.integers(1000))}"


def build_tasks(
    cache: pc.BenchmarkCache,
    composition: list[str],
    attack: str,
    f: float,
    p_obs: float,
    adv: dict[str, list[tuple[str | None, float, float]]],
    seed: int,
    epoch_length: int = 20,
    coherent_schedule: list[bool] | None = None,
    single_adversary: str | None = None,
) -> tuple[Any, list[tuple[Observation, ...]]]:
    """Build one swarm configuration's task stream.

    ``coherent_schedule`` overrides the alternating burst epochs with an explicit
    per-task flag. E1's sleeper is exactly this: False for the first k rounds and
    True thereafter, i.e. a single switch rather than a repeating cycle.

    ``single_adversary`` forces every Byzantine agent onto one generator instead
    of the usual round-robin. That is what makes an E2 Sybil bloc a bloc: s
    identities telling the SAME lie. Leaving it None gives the disagreeing-Sybil
    control, where each identity lies differently and there is no coherent
    channel to find.
    """
    rng = np.random.default_rng(seed)
    assignment = assign_swarm(pc.N_AGENTS, f, composition, rng)
    topology = build_topology("complete", pc.N_AGENTS)
    byz = sorted(assignment.byzantine)
    adv_models = [m for m in ADVERSARY_MODELS if m in adv] or ADVERSARY_MODELS
    if single_adversary is not None:
        adv_models = [single_adversary]
    # Byzantine agents draw their generator round-robin, so a swarm's adversaries
    # are a realistic mix rather than all one model.
    adv_of = {a: adv_models[i % len(adv_models)] for i, a in enumerate(byz)}

    slot_of = {}
    for agent, model in enumerate(assignment.models):
        if model is not None:
            slots = cache.slots_for(model)
            slot_of[agent] = slots[agent % len(slots)] if len(slots) > 1 else 0

    coherent_epochs = (
        coherent_schedule
        if coherent_schedule is not None
        else epoch_schedule(len(cache.task_ids), epoch_length)
    )
    tasks = []
    for t, (task_id, gold) in enumerate(zip(cache.task_ids, cache.gold, strict=True)):
        honest_pool = [cache.answers[(m, 0)][t][0] for m in {x for x in assignment.models if x}]
        broadcasts = []
        for agent in range(pc.N_AGENTS):
            if agent in assignment.byzantine:
                model = adv_of[agent]
                if attack == "falsified_confidence":
                    # The transform inverts ONLY the self-report, exactly as
                    # specified -- but it is applied to the *adversarial* cache,
                    # not the honest one. Applied to honest outputs the attack is
                    # toothless by construction: the Byzantine agents would be
                    # broadcasting correct answers, and no confidence weighting
                    # can be hurt by a correct answer. The attack that the
                    # prediction is about is a WRONG answer wearing a HIGH claimed
                    # confidence, which is what inverting a wrong answer's
                    # (low) self-report produces. The answer and the logprob
                    # confidence are passed through untouched, so any difference
                    # from plain semantic_negation is attributable to the
                    # self-report channel alone.
                    a, lc, sc = adv[model][t]
                    ans, lp, sr = a, lc, invert_self_report(sc)
                elif attack == "burst":
                    if coherent_epochs[t] and model in adv:
                        ans, lp, sr = adv[model][t]
                    else:
                        ans, lp, sr = noise_answer(gold, honest_pool, rng), 0.9, 0.9
                elif attack == "noise":
                    ans, lp, sr = noise_answer(gold, honest_pool, rng), 0.9, 0.9
                else:
                    ans, lp, sr = adv[model][t]
                broadcasts.append(
                    Broadcast(
                        agent_id=agent,
                        task_id=task_id,
                        answer=ans,
                        logprob_confidence=float(lp),
                        self_reported_confidence=float(sr),
                        is_byzantine=True,
                        source_model=model,
                        attack=attack,
                    )
                )
            else:
                model = assignment.models[agent]
                a, lc, sc = cache.answers[(model, slot_of[agent])][t]
                broadcasts.append(
                    Broadcast(
                        agent_id=agent,
                        task_id=task_id,
                        answer=a,
                        logprob_confidence=float(lc),
                        self_reported_confidence=float(sc),
                        is_byzantine=False,
                        source_model=model,
                        attack=None,
                    )
                )
        tasks.append(observe(tuple(broadcasts), topology, p_obs, rng))
    return assignment, tasks


def evaluate(cache, assignment, tasks, benchmark, f, parity, thresholds, resamples, seed):
    honest = [a for a in range(pc.N_AGENTS) if a not in assignment.byzantine]
    labels = pc.LABEL_SPACES[benchmark]
    n_byz = len(assignment.byzantine)
    aggs = [
        MajorityVote(),
        ConfidenceWeighted(parity=parity),
        ConfidenceWeighted(parity=parity, use_self_report=True),
        GeometricMedian(),
        Krum(n_byz),
        MultiKrum(n_byz),
        CoordinateMedian(),
        TrimmedMean(trim=max(f, 0.1)),
        DawidSkeneAggregator(label_space=labels),
        SACFilterRefine(parity=parity),
        AIPAggregator(benchmark, "gated", thresholds, parity, labels),
        AIPAggregator(benchmark, "naive", thresholds, parity, labels),
        AIPAggregator(benchmark, "trust_only", thresholds, parity, labels),
    ]
    rows = []
    for agg in aggs:
        if agg.needs_fit:
            agg.fit(tasks)
        per_task = np.array(
            [
                np.mean([agg.aggregate(tasks[t][a].broadcasts, a) == cache.gold[t] for a in honest])
                if honest
                else np.nan
                for t in range(len(tasks))
            ],
            dtype=float,
        )
        valid = per_task[np.isfinite(per_task)]
        ci = bootstrap_statistic(
            lambda i, v=valid: float(np.mean(v[i])) if i.size else float("nan"),
            valid.size,
            resamples,
            seed=seed,
        )
        row = {"method": agg.name, "is_oracle": bool(getattr(agg, "is_oracle", False))}
        row.update(ci.as_dict("accuracy_"))
        if isinstance(agg, AIPAggregator):
            byz = set(assignment.byzantine)
            hi, ht = agg.diagnostics.honest_inversions(byz)
            ai, at = agg.diagnostics.adversary_inversions(byz)
            row["honest_inversion_rate"] = hi / ht if ht else np.nan
            row["adversary_inversion_rate"] = ai / at if at else np.nan
        rows.append(row)
    return rows


def adversary_coherence(adv, cache, benchmark) -> dict[str, float]:
    """Measured coherence of the adversarial channel: how often two adversaries agree."""
    models = list(adv)
    if len(models) < 2:
        return {"coherence": np.nan, "error_rate": np.nan}
    a, b = adv[models[0]], adv[models[1]]
    same = [
        x[0] is not None and y[0] is not None and x[0] == y[0] for x, y in zip(a, b, strict=True)
    ]
    wrong = [x[0] != g for x, g in zip(a, cache.gold, strict=True)]
    return {"coherence": float(np.mean(same)), "error_rate": float(np.mean(wrong))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/adversarial"))
    p.add_argument("--resamples", type=int, default=500)
    p.add_argument("--seed", type=int, default=20250825)
    p.add_argument("--benchmarks", nargs="*", default=pc.BENCHMARKS)
    args = p.parse_args()

    thresholds = InversionThresholds.load()
    models = sorted(q.stem for q in (Path("data/cache") / "gsm8k").glob("*.parquet"))
    args.out.mkdir(parents=True, exist_ok=True)

    attacks = INFERENCE_ATTACKS + OFFLINE_ATTACKS + ["noise"]
    compositions = {"hom_llama32_3b": ["llama32_3b"], "mix_all4": list(models)}
    rows, chan = [], []

    for benchmark in args.benchmarks:
        cache = pc.load_cache(benchmark, models, Path("data/cache"), Path("data/cache_t07"))
        for attack in attacks:
            adv = (
                load_adversarial(attack, benchmark, cache.task_ids)
                if attack in INFERENCE_ATTACKS
                else {}
            )
            if attack in ("burst", "falsified_confidence"):
                # Both are transforms over the coherent-lie cache: burst schedules
                # it against noise, falsified_confidence rewrites its self-report.
                adv = load_adversarial("semantic_negation", benchmark, cache.task_ids)
            if attack in ("burst", "falsified_confidence") and not adv:
                log.warning("phase_d.missing_cache", attack=attack, benchmark=benchmark)
                continue
            if attack in INFERENCE_ATTACKS and not adv:
                log.warning("phase_d.missing_cache", attack=attack, benchmark=benchmark)
                continue
            if adv:
                c = adversary_coherence(adv, cache, benchmark)
                chan.append({"benchmark": benchmark, "attack": attack, **c})
            for comp_name, composition in compositions.items():
                for f, p_obs, parity in itertools.product(
                    F_GRID, [1.0, 0.5], [ParityConfig(0.1), ParityConfig(None)]
                ):
                    if parity.share is None and p_obs != 1.0:
                        continue
                    assignment, tasks = build_tasks(
                        cache,
                        composition,
                        attack,
                        f,
                        p_obs,
                        adv,
                        seed=args.seed + stable_seed((benchmark, comp_name, attack, f, p_obs)),
                    )
                    for row in evaluate(
                        cache,
                        assignment,
                        tasks,
                        benchmark,
                        f,
                        parity,
                        thresholds,
                        args.resamples,
                        args.seed,
                    ):
                        row.update(
                            benchmark=benchmark,
                            attack=attack,
                            composition=comp_name,
                            f=f,
                            p_obs=p_obs,
                            parity=parity.label,
                            n_byzantine=len(assignment.byzantine),
                            n_tasks=len(cache.task_ids),
                        )
                        rows.append(row)
            log.info("phase_d.attack_done", benchmark=benchmark, attack=attack, rows=len(rows))
        pd.DataFrame(rows).to_parquet(args.out / "adversarial_sweep.parquet", index=False)

    frame = pd.DataFrame(rows)
    frame.to_parquet(args.out / "adversarial_sweep.parquet", index=False)
    pd.DataFrame(chan).to_parquet(args.out / "channel_stats.parquet", index=False)
    m = Manifest.create(
        phase="phase_d_sweep",
        config={"resamples": args.resamples, "attacks": attacks},
        seeds={"sweep": args.seed},
        notes={"inference_run": False},
    )
    m.finish(status="ok", n_rows=len(frame)).write(Path("results/manifests") / f"{m.run_id}.json")
    print(f"\nwrote {args.out}/adversarial_sweep.parquet ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
