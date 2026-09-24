#!/usr/bin/env python3
"""Deterrence experiment: an adaptive adversary learns which attack works.

Offline. Each epoch the bandit picks ``coherent_lie`` (replay the cached
semantic_negation outputs) or ``noise`` (independent wrong answers), and is
rewarded by the swarm accuracy it destroys. It plays against two defences, so
what it learns is a statement about each defence.

The prediction: against a discard-style aggregator, coherent lying wins, because
a coherent bloc captures the plurality. Against AIP, coherence is exactly what
makes a channel invertible, so the same arm should become self-defeating and the
bandit should collapse onto noise. An adversary driven onto noise has been
deterred rather than filtered, and noise is the weakest attack it has.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import phase_c_aggregate as pc  # noqa: E402
import phase_d_sweep as pd_sweep  # noqa: E402

from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.aggregation.baselines import SACFilterRefine  # noqa: E402
from aip.attacks.adaptive_bandit import EpsilonGreedyAdversary  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest  # noqa: E402
from aip.swarm.assignment import assign_swarm  # noqa: E402
from aip.swarm.broadcast import observe  # noqa: E402
from aip.swarm.topology import build_topology  # noqa: E402
from aip.types import Broadcast  # noqa: E402

log = configure_logging(level="INFO")


def build_epoch(cache, assignment, adv, arm, task_idx, rng):
    """Broadcasts for one epoch's tasks under the chosen arm."""
    byz = sorted(assignment.byzantine)
    adv_models = [m for m in pd_sweep.ADVERSARY_MODELS if m in adv] or pd_sweep.ADVERSARY_MODELS
    adv_of = {a: adv_models[i % len(adv_models)] for i, a in enumerate(byz)}
    topology = build_topology("complete", pc.N_AGENTS)
    out = []
    for t in task_idx:
        gold = cache.gold[t]
        honest_pool = [cache.answers[(m, 0)][t][0] for m in {x for x in assignment.models if x}]
        bs = []
        for agent in range(pc.N_AGENTS):
            if agent in assignment.byzantine:
                model = adv_of[agent]
                if arm == "coherent_lie" and model in adv:
                    a, lp, sr = adv[model][t]
                else:
                    a, lp, sr = pd_sweep.noise_answer(gold, honest_pool, rng), 0.9, 0.9
                bs.append(
                    Broadcast(
                        agent_id=agent,
                        task_id=cache.task_ids[t],
                        answer=a,
                        logprob_confidence=float(lp),
                        self_reported_confidence=float(sr),
                        is_byzantine=True,
                        source_model=model,
                        attack=arm,
                    )
                )
            else:
                model = assignment.models[agent]
                slots = cache.slots_for(model)
                slot = slots[agent % len(slots)] if len(slots) > 1 else 0
                a, lc, sc = cache.answers[(model, slot)][t]
                bs.append(
                    Broadcast(
                        agent_id=agent,
                        task_id=cache.task_ids[t],
                        answer=a,
                        logprob_confidence=float(lc),
                        self_reported_confidence=float(sc),
                        is_byzantine=False,
                        source_model=model,
                        attack=None,
                    )
                )
        out.append(observe(tuple(bs), topology, 1.0, rng))
    return out


def run(defence_name, cache, adv, benchmark, f, epochs, epoch_len, thresholds, seed, composition):
    rng = np.random.default_rng(seed)
    assignment = assign_swarm(pc.N_AGENTS, f, composition, rng)
    honest = [a for a in range(pc.N_AGENTS) if a not in assignment.byzantine]
    bandit = EpsilonGreedyAdversary(epsilon=0.15, seed=seed)
    history: list = []
    for epoch in range(epochs):
        arm, exploring = bandit.select()
        idx = list(
            range(
                (epoch * epoch_len) % len(cache.task_ids),
                (epoch * epoch_len) % len(cache.task_ids) + epoch_len,
            )
        )
        idx = [i % len(cache.task_ids) for i in idx]
        tasks = build_epoch(cache, assignment, adv, arm, idx, rng)
        history.extend(tasks)
        if defence_name == "aip_gated":
            agg = AIPAggregator(
                benchmark, "gated", thresholds, ParityConfig(0.1), pc.LABEL_SPACES[benchmark]
            )
        else:
            agg = SACFilterRefine(parity=ParityConfig(0.1))
        # The defence learns from everything it has seen so far, as it would live.
        agg.fit(history)
        hits = [
            agg.aggregate(tasks[k][a].broadcasts, a) == cache.gold[idx[k]]
            for k in range(len(idx))
            for a in honest
        ]
        accuracy = float(np.mean(hits)) if hits else float("nan")
        reward = 1.0 - accuracy
        bandit.update(arm, reward)
        bandit.record(epoch, arm, reward, accuracy, exploring)
    return bandit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument(
        "--f",
        type=float,
        default=0.6,
        help="Byzantine fraction. Must be high enough that a coherent bloc can "
        "actually capture the plurality, or both defences behave identically "
        "and the bandit has nothing to learn from.",
    )
    p.add_argument(
        "--composition",
        nargs="*",
        default=None,
        help="honest models; defaults to the full heterogeneous mix",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--epoch-len", type=int, default=10)
    p.add_argument("--seed", type=int, default=20250825)
    p.add_argument("--out", type=Path, default=Path("results/adversarial"))
    args = p.parse_args()

    thresholds = InversionThresholds.load()
    models = sorted(q.stem for q in (Path("data/cache") / "gsm8k").glob("*.parquet"))
    cache = pc.load_cache(args.benchmark, models, Path("data/cache"), Path("data/cache_t07"))
    adv = pd_sweep.load_adversarial("semantic_negation", args.benchmark, cache.task_ids)
    if not adv:
        print("semantic_negation cache missing; run phase_d_adversarial.py first")
        return 1

    composition = args.composition or list(models)
    records = []
    print(
        f"\n{'=' * 78}\nDETERRENCE: adaptive adversary vs each defence "
        f"({args.benchmark}, f={args.f})\n{'=' * 78}"
    )
    for defence in ("aip_gated", "sac_filter_refine"):
        bandit = run(
            defence,
            cache,
            adv,
            args.benchmark,
            args.f,
            args.epochs,
            args.epoch_len,
            thresholds,
            args.seed,
            composition,
        )
        records.extend(bandit.log.to_records(defence))
        arms = bandit.log.arm
        second_half = arms[len(arms) // 2 :]
        noise_share = sum(a == "noise" for a in second_half) / len(second_half)
        print(f"\n  {defence}")
        print(
            f"    final arm values : coherent_lie={bandit.values['coherent_lie']:.3f}  "
            f"noise={bandit.values['noise']:.3f}"
        )
        print(f"    noise share (2nd half of run): {noise_share:.0%}")
        print(f"    mean swarm accuracy: {np.mean(bandit.log.swarm_accuracy):.3f}")
        print(f"    arm trace: {''.join('N' if a == 'noise' else 'C' for a in arms)}")

    frame = pd.DataFrame(records)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "bandit_log.parquet", index=False)
    m = Manifest.create(
        phase="phase_d_bandit",
        config={
            "benchmark": args.benchmark,
            "f": args.f,
            "epochs": args.epochs,
            "epoch_len": args.epoch_len,
        },
        seeds={"bandit": args.seed},
        notes={"inference_run": False},
    )
    m.finish(status="ok").write(Path("results/manifests") / f"{m.run_id}.json")
    print(f"\nwrote {args.out / 'bandit_log.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
