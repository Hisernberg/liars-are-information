"""World construction and the held-out evaluation harness.

The protocol follows the retained v1 replay so that numbers are comparable:

* a swarm of ``n_agents`` (default 10) identities, each honest identity replaying
  one cached model (round-robin over a shuffled roster, so some models repeat);
* a nested Byzantine set: the corrupted identities at fraction ``f`` are a prefix
  of one fixed permutation, so raising ``f`` only adds attackers;
* disjoint HISTORY / VALIDATION / TEST task partitions (40/20/40 by a stable
  hash); every receiver fits on HISTORY only and is scored on VALIDATION/TEST;
* per-task accuracy is the mean over **honest receivers** (Byzantine receivers
  are never scored); receivers, seeds and peers are not independent samples --
  the statistical unit is the task (see :mod:`lai.stats`).

The defence input is the list of answers each receiver actually heard; gold,
Byzantine flags, model names and attack labels are stripped before any method
sees them.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from aip.aggregation.aip import AIPAggregator, InversionThresholds
from aip.aggregation.base import Aggregator, ParityConfig
from aip.aggregation.baselines import ConfidenceWeighted, MajorityVote, SACFilterRefine
from aip.aggregation.dawid_skene_full import DawidSkeneFullAggregator
from aip.types import Broadcast, Observation
from lai import attacks as atk
from lai.data import (
    ATTACKER_MODELS,
    FROZEN_MODELS,
    ROOT,
    WEAK_MODELS,
    BenchmarkData,
    load_benchmark,
    score,
)
from lai.race import RACEAggregator, ReceiverFit, complete_link_groups

THRESHOLDS_PATH = ROOT / "configs" / "inversion_thresholds.yaml"


def stable_seed(*parts: object) -> int:
    payload = json.dumps([str(p) for p in parts]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def split_indices(task_ids: list[str], seed: int = 73019) -> dict[str, list[int]]:
    order = sorted(range(len(task_ids)), key=lambda i: stable_seed(seed, task_ids[i]))
    a, b = int(0.4 * len(order)), int(0.6 * len(order))
    return {"history": order[:a], "validation": order[a:b], "test": order[b:]}


@dataclass(frozen=True)
class World:
    benchmark: str
    f: float
    attack: str = "coherent"
    param: float = 1.0
    p_obs: float = 1.0
    seed: int = 0
    n_agents: int = 10
    composition: str = "frozen"
    history_n: int | None = None
    study: str = "main"
    source: str = "artifact"
    """``artifact`` (repaired cache), ``live`` or ``live_debate`` (live swarm)."""

    @property
    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class BuiltWorld:
    world: World
    data: BenchmarkData
    models: tuple[str | None, ...]
    byzantine: frozenset[int]
    honest: tuple[int, ...]
    splits: dict[str, list[int]]
    raw: list[tuple[Broadcast, ...]]
    defense: list[tuple[Observation, ...]]


def roster(composition: str) -> list[str]:
    if composition == "live":
        from lai.data import LIVE_MODELS

        return list(LIVE_MODELS)
    if composition == "frozen":
        return list(FROZEN_MODELS)
    if composition == "frozen+weak":
        return list(FROZEN_MODELS + WEAK_MODELS)
    if composition == "weak":
        return ["llama32_3b", "ministral_8b", "olmo2_7b", "phi4_mini_reasoning"]
    if composition.startswith("hom_"):
        return [composition[4:]]
    raise ValueError(composition)


def _data_for(world: World) -> BenchmarkData:
    if world.source == "artifact":
        return load_benchmark(world.benchmark)
    from lai.data import load_live

    return load_live(world.benchmark, "cache" if world.source == "live" else "cache_debate")


def build_world(world: World) -> BuiltWorld:
    data = _data_for(world)
    n = world.n_agents
    identity_rng = np.random.default_rng(stable_seed(world.seed, world.benchmark, world.composition, n, "identities"))
    comp = list(identity_rng.permutation(roster(world.composition)))
    base = [comp[i % len(comp)] for i in identity_rng.permutation(n)]
    corrupted = np.random.default_rng(stable_seed(world.seed, world.benchmark, n, "corruption")).permutation(n)
    byz = frozenset(int(i) for i in corrupted[: int(round(world.f * n))])
    models = tuple(None if i in byz else base[i] for i in range(n))
    honest = tuple(i for i in range(n) if i not in byz)
    splits = split_indices(data.task_ids)
    if world.history_n is not None:
        splits = dict(splits, history=splits["history"][: world.history_n])
    phase_of = {t: p for p, ts in splits.items() for t in ts}
    labels = data.label_space
    llm_sources: list[tuple[str, str]] = []
    if world.attack.startswith("llm:"):
        prompt = world.attack.split(":", 1)[1]
        attackers = ATTACKER_MODELS if world.source == "artifact" else sorted({m for (_, m) in data.adversarial})
        llm_sources = [(prompt, m) for m in attackers if (prompt, m) in data.adversarial]
        if not llm_sources:
            raise ValueError(f"no cached LLM attacker for {world.attack} on {world.benchmark}")
    # A sleeper impersonates the honest models present in the swarm.
    cover_models = [m for m in models if m is not None] or [base[0]]
    raw: list[tuple[Broadcast, ...]] = []
    defense: list[tuple[Observation, ...]] = []
    byz_sorted = sorted(byz)
    for t, tid in enumerate(data.task_ids):
        gold = data.gold[t]
        honest_answers = [data.honest[m][t] for m in models if m is not None]
        rng = np.random.default_rng(stable_seed(world.seed, world.benchmark, tid, "attack"))
        lies = atk.byzantine_answers(
            world.attack,
            world.param,
            gold=gold,
            label_space=labels,
            honest_answers=honest_answers,
            n_byzantine=len(byz_sorted),
            is_correct=lambda a, _g=gold: score(world.benchmark, a, _g),
            rng=rng,
            phase=phase_of.get(t, "unused"),
            cover_answers=[data.honest[m][t] for m in cover_models],
            llm_answers=[data.adversarial[s][t] for s in llm_sources] if llm_sources else None,
        )
        lie_of = dict(zip(byz_sorted, lies, strict=True))
        row = []
        for agent in range(n):
            if agent in byz:
                row.append(Broadcast(agent, tid, data.canonical(t, lie_of[agent]), 0.99, 0.99, True, None, world.attack))
            else:
                m = models[agent]
                row.append(Broadcast(agent, tid, data.honest[m][t], data.logprob[m][t], data.logprob[m][t], False, m, None))
        raw.append(tuple(row))
        obs_rng = np.random.default_rng(stable_seed(world.seed, world.benchmark, tid, "visibility"))
        observations = []
        for agent in range(n):
            heard = [row[agent]] + [
                row[p] for p in range(n) if p != agent and (world.p_obs >= 1.0 or obs_rng.random() < world.p_obs)
            ]
            stripped = tuple(
                replace(b, is_byzantine=False, source_model=None, attack=None)
                for b in sorted(heard, key=lambda b: b.agent_id)
            )
            observations.append(Observation(agent, tid, stripped))
        defense.append(tuple(observations))
    return BuiltWorld(world, data, models, byz, honest, splits, raw, defense)


# ---------------------------------------------------------------- oracles


class OracleHonestMajority(Aggregator):
    """Plurality over the *honest* agents a receiver heard. Knows identities."""

    is_oracle = True

    def __init__(self, honest: Iterable[int]):
        self.honest = set(honest)
        self.name = "oracle_honest_majority"

    def aggregate(self, observations, self_id):
        votes = Counter(b.answer for b in observations if b.agent_id in self.honest and b.answer is not None)
        if not votes:
            return None
        top = max(votes.values())
        tied = sorted(a for a, v in votes.items() if v == top)
        own = next((b.answer for b in observations if b.agent_id == self_id), None)
        return own if own in tied else tied[0]


def oracle_channel(built: BuiltWorld, receiver: int) -> RACEAggregator:
    """Known-channel reference: RACE's decoder with accuracies measured against gold
    on HISTORY (for every agent, Byzantine included) and true-source grouping."""
    agg = RACEAggregator(built.data.label_space, name="oracle_channel")
    agg.is_oracle = True
    b = built.world.benchmark
    agents = tuple(range(built.world.n_agents))
    hist = built.splits["history"]
    acc = []
    n_obs = []
    for j in agents:
        seen = [
            score(b, built.raw[t][j].answer, built.data.gold[t])
            for t in hist
            if any(x.agent_id == j for x in built.defense[t][receiver].broadcasts)
        ]
        n_obs.append(len(seen))
        acc.append((sum(seen) + 0.5) / (len(seen) + 1.0) if seen else 0.5)
    # True-source groups: replicas of one cached model, and the Byzantine bloc if
    # its members are identical on history.
    reports = np.full((len(hist), len(agents)), -1)
    for r, t in enumerate(hist):
        cands = {a: i for i, a in enumerate(sorted({x.answer for x in built.raw[t] if x.answer is not None}))}
        for j in agents:
            a = built.raw[t][j].answer
            if a is not None:
                reports[r, j] = cands[a]
    groups = complete_link_groups(reports, 0.999, 1)
    for j in agents:
        m = built.models[j]
        if m is not None:
            first = next(i for i in agents if built.models[i] == m)
            groups[j] = groups[first]
    k = float(len(built.data.label_space)) if built.data.label_space else 3.0
    agg.fits[receiver] = ReceiverFit(
        agents, np.array(acc), groups, np.array(n_obs), k, 0, True, 0.0
    )
    return agg


# ---------------------------------------------------------------- methods

CORE_METHODS = (
    "self",
    "majority",
    "confidence",
    "sac",
    "aip_gated",
    "aip_trust_only",
    "aip_naive",
    "aip_soft",
    "ds_full",
    "ds_onecoin",
    "race_noclone",
    "race_rawclone",
    "race_ms",
    "race",
    "race_full",
    "race_capself",
    "oracle_honest_majority",
    "oracle_channel",
)


def method_factories(built: BuiltWorld, methods: Iterable[str]) -> dict[str, Callable[[int], Aggregator | None]]:
    b = built.world.benchmark
    labels = list(built.data.label_space) if built.data.label_space else None
    thresholds = InversionThresholds.load(THRESHOLDS_PATH)
    parity = ParityConfig(0.1)
    table: dict[str, Callable[[int], Aggregator | None]] = {
        "self": lambda r: None,
        "majority": lambda r: MajorityVote(),
        "confidence": lambda r: ConfidenceWeighted(parity=parity),
        "sac": lambda r: SACFilterRefine(parity=parity),
        "aip_gated": lambda r: AIPAggregator(b, "gated", thresholds, parity, labels),
        "aip_trust_only": lambda r: AIPAggregator(b, "trust_only", thresholds, parity, labels),
        "aip_naive": lambda r: AIPAggregator(b, "naive", thresholds, parity, labels),
        "aip_soft": lambda r: AIPAggregator(b, "gated", thresholds, parity, labels, soft_gate=6.0),
        "ds_onecoin": lambda r: RACEAggregator(labels, anchored=False, clone_aware=False),
        "race_noclone": lambda r: RACEAggregator(labels, clone_aware=False),
        "race_rawclone": lambda r: RACEAggregator(labels, clone_mode="raw"),
        "race_ms": lambda r: RACEAggregator(labels, multistart=True),
        "race": lambda r: RACEAggregator(labels),
        "race_capself": lambda r: RACEAggregator(labels, cap="self"),
        "oracle_honest_majority": lambda r: OracleHonestMajority(built.honest),
        "oracle_channel": lambda r: oracle_channel(built, r),
    }
    if labels:
        table["ds_full"] = lambda r: DawidSkeneFullAggregator(labels, max_iter=60)
        table["race_full"] = lambda r: RACEAggregator(labels, model="full")
    return {m: table[m] for m in methods if m in table}


def evaluate_world(world: World, methods: Iterable[str] = CORE_METHODS, keep_diagnostics: bool = True) -> dict:
    built = build_world(world)
    b = world.benchmark
    rows: list[dict] = []
    diag: list[dict] = []
    factories = method_factories(built, methods)
    parts = ("validation", "test")
    for method, make in factories.items():
        scores = {p: {t: [] for t in built.splits[p]} for p in parts}
        for receiver in built.honest:
            agg = make(receiver)
            if agg is not None and agg.needs_fit and method != "oracle_channel":
                agg.fit([(built.defense[t][receiver],) for t in built.splits["history"]])
            if keep_diagnostics and agg is not None and method in ("aip_gated", "aip_soft", "race", "race_noclone", "ds_onecoin"):
                diag.append(_channel_summary(method, agg, receiver, built))
            for p in parts:
                for t in built.splits[p]:
                    obs = built.defense[t][receiver]
                    pred = obs.own.answer if agg is None else agg.aggregate(obs.broadcasts, receiver)
                    scores[p][t].append(float(score(b, pred, built.data.gold[t])))
        for p in parts:
            for t, vals in scores[p].items():
                rows.append({"method": method, "split": p, "task": built.data.task_ids[t], "accuracy": float(np.mean(vals))})
    q = None
    byz = sorted(built.byzantine)
    if len(byz) >= 2:
        q = float(np.mean([built.raw[t][i].answer == built.raw[t][j].answer for t in built.splits["test"] for i in byz for j in byz if i < j]))
    meta = asdict(world) | {
        "n_byzantine": len(built.byzantine),
        "n_honest": len(built.honest),
        "honest_models": [m for m in built.models if m],
        "byz_pair_coincidence_test": q,
    }
    return {"meta": meta, "rows": rows, "diagnostics": diag}


def _channel_summary(method: str, agg, receiver: int, built: BuiltWorld) -> dict:
    out = {"method": method, "receiver": receiver, "honest_inverted": 0, "honest_total": 0, "byz_inverted": 0,
           "byz_trusted": 0, "byz_total": 0, "honest_trusted": 0}
    channels = agg.diagnostics.channels.get(receiver, {})
    for peer, stats in channels.items():
        if peer == receiver:
            continue
        decision = stats.decision
        if peer in built.byzantine:
            out["byz_total"] += 1
            out["byz_inverted"] += decision == "invert"
            out["byz_trusted"] += decision == "trust"
        else:
            out["honest_total"] += 1
            out["honest_inverted"] += decision == "invert"
            out["honest_trusted"] += decision == "trust"
    return out


def _run_one(args: tuple[World, tuple[str, ...]]) -> dict:
    world, methods = args
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    result = evaluate_world(world, methods)
    return result


def run_worlds(
    worlds: list[World],
    methods: Iterable[str],
    out_dir: Path,
    processes: int = 4,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every world (in parallel), return (per_task, worlds, diagnostics)."""
    import multiprocessing as mp
    import time

    out_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(methods)
    per_task: list[pd.DataFrame] = []
    metas: list[dict] = []
    diags: list[dict] = []
    started = time.monotonic()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_one, [(w, methods) for w in worlds], chunksize=1), 1):
            meta = res["meta"]
            wid = stable_seed(json.dumps(meta, sort_keys=True, default=str)) % 10**12
            meta["world_id"] = wid
            frame = pd.DataFrame(res["rows"])
            for k in ("benchmark", "f", "attack", "param", "p_obs", "seed", "n_agents", "composition", "history_n", "study", "source"):
                frame[k] = meta[k]
            frame["world_id"] = wid
            per_task.append(frame)
            metas.append(meta)
            for d in res["diagnostics"]:
                diags.append(d | {k: meta[k] for k in ("benchmark", "f", "attack", "param", "p_obs", "seed", "study")} | {"world_id": wid})
            if progress and (i % 10 == 0 or i == len(worlds)):
                print(f"[{i}/{len(worlds)}] {time.monotonic() - started:.0f}s", flush=True)
    frame = pd.concat(per_task, ignore_index=True)
    worlds_frame = pd.DataFrame(metas)
    diag_frame = pd.DataFrame(diags)
    frame.to_parquet(out_dir / "per_task.parquet", index=False)
    worlds_frame.to_json(out_dir / "worlds.json", orient="records", indent=1)
    if len(diag_frame):
        diag_frame.to_parquet(out_dir / "channel_diagnostics.parquet", index=False)
    return frame, worlds_frame, diag_frame
