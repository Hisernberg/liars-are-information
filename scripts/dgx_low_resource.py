#!/usr/bin/env python3
"""Held-out receiver-local cache replay; no inference and no network calls."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import platform
import resource
import sys
import time
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import pandas as pd
import phase_c_aggregate as pc
import yaml

from aip.aggregation.aip import AIPAggregator, InversionThresholds
from aip.aggregation.base import ParityConfig
from aip.aggregation.baselines import ConfidenceWeighted, MajorityVote, SACFilterRefine
from aip.aggregation.dawid_skene_full import DawidSkeneFullAggregator
from aip.swarm.assignment import assign_swarm
from aip.swarm.broadcast import AdversaryConfig, build_broadcasts, observe
from aip.swarm.topology import build_topology
from aip.tasks.math500 import math_match
from aip.types import Broadcast


@lru_cache(maxsize=20000)
def score_answer(benchmark, answer, gold):
    if benchmark == "math500":
        return bool(math_match(answer, gold))
    return answer is not None and answer == gold


def math_broadcasts(task_id, gold, assignment, answers, adversary, rng):
    """Semantic-wrong open pool; gold is available only to symbolic attacker."""
    pool = sorted({answers[m][0] for m in assignment.models if m is not None
        and answers[m][0] is not None and not score_answer("math500", answers[m][0], gold)})
    if not pool:
        pool = [f"{gold}__adv"]
        if score_answer("math500", pool[0], gold):
            raise ValueError("Synthetic fallback unexpectedly scores correct")
    shared = pool[int(rng.integers(len(pool)))]
    result = []
    for agent, model in enumerate(assignment.models):
        if agent in assignment.byzantine:
            answer = shared
            if adversary.kind == "gate_aware" and rng.random() >= adversary.coherence:
                answer = pool[int(rng.integers(len(pool)))]
            result.append(Broadcast(agent, task_id, answer, adversary.confidence,
                adversary.confidence, True, None, adversary.kind))
        else:
            answer, confidence, reported = answers[model]
            result.append(Broadcast(agent, task_id, answer, confidence, reported, False, model, None))
    return tuple(result)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], "big")


def split_indices(task_ids, seed=73019):
    """Same disjoint 40/20/40 split across swarm seeds and methods."""
    ids = sorted(range(len(task_ids)), key=lambda i: stable_seed(seed, task_ids[i]))
    a, b = int(.4 * len(ids)), int(.6 * len(ids))
    if a < 5 or len(ids) - b < 2:
        raise ValueError("Need at least 15 tasks.")
    return {"history": ids[:a], "validation": ids[a:b], "test": ids[b:]}


def load_verified_cache(benchmark, models):
    reference = None
    for model in models:
        path = ROOT / "data/cache" / benchmark / f"{model}.parquet"
        rows = pd.read_parquet(path, columns=["task_id", "gold_answer", "sample_index"])
        if rows.task_id.duplicated().any() or not (rows.sample_index == 0).all():
            raise ValueError(f"Non-unique greedy cache: {path}")
        gold = rows.set_index("task_id").gold_answer.astype(str).sort_index()
        if reference is None:
            reference = gold
        elif not gold.equals(reference):
            raise ValueError(f"Task coverage or gold mismatch: {path}")
    return pc.load_cache(benchmark, models, ROOT / "data/cache", ROOT / "data/cache_t07")


def worlds(profile):
    seeds = [20260912] if profile == "smoke" else [20260912, 20260913, 20260914]
    f_grid = [0., .5] if profile == "smoke" else [0., .3, .5, .7]
    p_grid = [1.] if profile == "smoke" else [1., .5]
    for b, f, p, seed in itertools.product(["mmlu", "boolq", "math500"], f_grid, p_grid, seeds):
        yield dict(study="mechanism", benchmark=b, f=f, p_obs=p, seed=seed, composition="frozen_mix", attack="always_wrong", coherence_p=1.)
    if profile != "smoke":
        for b, f, seed in itertools.product(["mmlu", "math500"], [0., .5], seeds):
            yield dict(study="diversity", benchmark=b, f=f, p_obs=1., seed=seed, composition="hom_qwen38_27b", attack="always_wrong", coherence_p=1.)
    bs = ["mmlu"] if profile == "smoke" else ["mmlu", "math500"]
    fs = [.5] if profile == "smoke" else [.5, .7]
    ps = [0., 1.] if profile == "smoke" else [0., .25, .5, .75, 1.]
    for b, f, seed, p in itertools.product(bs, fs, seeds, ps):
        yield dict(study="adaptive", benchmark=b, f=f, p_obs=1., seed=seed, composition="frozen_mix", attack="gate_aware", coherence_p=p)


def build_world(cache, models, world):
    composition = ["qwen38_27b"] if world["composition"].startswith("hom_") else models
    identity_rng = np.random.default_rng(stable_seed(world["seed"], world["benchmark"], world["composition"], "identities"))
    composition = list(identity_rng.permutation(composition))
    base = assign_swarm(10, 0., composition, identity_rng)
    # Freeze identities BEFORE corruption, with nested corrupted sets across f.
    # Otherwise truncating the alphabetically sorted roster changes competence
    # systematically as f grows.
    corrupted_order = np.random.default_rng(stable_seed(world["seed"], world["benchmark"], "corruption")).permutation(10)
    byzantine = frozenset(int(i) for i in corrupted_order[:round(10 * world["f"])])
    assignment = replace(base, byzantine=byzantine,
        models=tuple(None if i in byzantine else m for i, m in enumerate(base.models)))
    topology = build_topology("complete", 10)
    adversary = AdversaryConfig(kind=world["attack"], coherence=world["coherence_p"])
    tasks, raw = [], []
    for t, (tid, gold) in enumerate(zip(cache.task_ids, cache.gold, strict=True)):
        answers = {m: cache.answers[(m, 0)][t] for m in composition}
        # Independent visibility stream prevents p_obs from changing attacks.
        attack_rng = np.random.default_rng(stable_seed(world["seed"], tid, "attack"))
        obs_rng = np.random.default_rng(stable_seed(world["seed"], tid, "visibility"))
        if cache.benchmark == "math500":
            broadcasts = math_broadcasts(tid, gold, assignment, answers, adversary, attack_rng)
        else:
            broadcasts = build_broadcasts(tid, gold, assignment, answers,
                pc.LABEL_SPACES[cache.benchmark] or [], adversary, attack_rng)
        raw.append(broadcasts)
        tasks.append(observe(broadcasts, topology, world["p_obs"], obs_rng))
    return assignment, tasks, raw


def factories(benchmark, thresholds, adaptive, seed):
    labels, parity = pc.LABEL_SPACES[benchmark], ParityConfig(.1)
    result = {
        "self_only": lambda: None,
        "majority": MajorityVote,
        "confidence": lambda: ConfidenceWeighted(parity=parity),
        "sac": lambda: SACFilterRefine(parity=parity),
        "aip_gated": lambda: AIPAggregator(benchmark, "gated", thresholds, parity, labels),
        "aip_trust_only": lambda: AIPAggregator(benchmark, "trust_only", thresholds, parity, labels),
        "aip_naive": lambda: AIPAggregator(benchmark, "naive", thresholds, parity, labels),
    }
    if labels:
        result["dawid_skene_full"] = lambda: DawidSkeneFullAggregator(labels, max_iter=60)
    if adaptive:
        result["aip_soft"] = lambda: AIPAggregator(benchmark, "gated", thresholds, parity, labels, soft_gate=6.)
        result["aip_randomized"] = lambda: AIPAggregator(benchmark, "gated", thresholds, parity, labels, randomized_threshold=.3, threshold_seed=seed)
        result["aip_soft_randomized"] = lambda: AIPAggregator(benchmark, "gated", thresholds, parity, labels, soft_gate=6., randomized_threshold=.3, threshold_seed=seed)
    return result


def paired_interval(values, seed, resamples):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [values[rng.integers(len(values), size=len(values))].mean() for _ in range(resamples)]
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def comparisons(frame, resamples):
    """Task cluster bootstrap, after averaging seeds; receivers are not N."""
    rows = []
    subset = frame[(frame.split == "test") & (frame.study != "adaptive")]
    axes = ["study", "benchmark", "f", "p_obs", "composition", "attack", "coherence_p"]
    for key, group in subset.groupby(axes):
        table = group.groupby(["task_id", "method"]).accuracy.mean().unstack()
        for baseline in table.columns:
            if baseline == "aip_gated":
                continue
            paired = table[["aip_gated", baseline]].dropna()
            point, low, high = paired_interval(paired.aip_gated - paired[baseline], stable_seed(key, baseline), resamples)
            rows.append(dict(zip(axes, key, strict=True), baseline=baseline, delta=point, ci_low=low, ci_high=high,
                n_unique_test_tasks=len(paired), n_swarm_seeds=group.seed.nunique(),
                ci_kind="task-cluster percentile; seeds averaged; pointwise exploratory"))
    return pd.DataFrame(rows)


def select_attacks(frame):
    adaptive = frame[frame.study == "adaptive"]
    validation = adaptive[adaptive.split == "validation"].groupby(
        ["benchmark", "f", "seed", "method", "coherence_p"]).accuracy.mean().reset_index()
    rows, predictions = [], []
    for key, group in validation.groupby(["benchmark", "f", "seed", "method"]):
        chosen = group.sort_values(["accuracy", "coherence_p"]).iloc[0]
        b, f, seed, method = key
        test = adaptive[(adaptive.benchmark == b) & (adaptive.f == f) & (adaptive.seed == seed)
            & (adaptive.coherence_p == chosen.coherence_p) & (adaptive.split == "test")]
        scores = test.groupby("method").accuracy.mean()
        rows.append(dict(benchmark=b, f=f, seed=seed, target_method=method,
            selected_p=float(chosen.coherence_p), validation_accuracy=float(chosen.accuracy),
            test_accuracy=float(scores[method]), self_accuracy=float(scores.self_only),
            trust_only_accuracy=float(scores.aip_trust_only),
            n_test_tasks=test.task_id.nunique(), selected_using="validation only; p ascending tie break"))
        if method.startswith("aip_"):
            predictions.append(test.assign(target_method=method))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def report(out, frame, selected, meta):
    import matplotlib.pyplot as plt
    subset = frame[(frame.split == "test") & (frame.study == "mechanism") & (frame.p_obs == 1)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, b in zip(axs, ["mmlu", "boolq", "math500"], strict=True):
        for method in ["self_only", "majority", "aip_gated", "aip_trust_only", "dawid_skene_full"]:
            data = subset[(subset.benchmark == b) & (subset.method == method)].groupby("f").accuracy.mean()
            if len(data):
                ax.plot(data.index, data.values, marker="o", label=method)
        ax.set(title=b, xlabel="Byzantine fraction", ylim=(0, 1))
        ax.grid(alpha=.2)
    axs[0].set_ylabel("Held-out accuracy")
    axs[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Cached-answer replay: fixed history, honest-receiver mean per task")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"mechanism.{ext}", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    data = selected[selected.target_method.isin(["aip_gated", "aip_soft", "aip_randomized", "aip_soft_randomized"])]
    data.groupby(["benchmark", "f", "target_method"]).test_accuracy.mean().unstack().plot.bar(ax=ax)
    ax.set(ylabel="Test accuracy", xlabel="Benchmark / Byzantine fraction", ylim=(0, 1),
        title="Each defender attacked using its own validation-selected p")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"adaptive.{ext}", dpi=170)
    plt.close(fig)
    lines = ["# DGX replay results", "",
        f"Completed {meta['worlds_completed']} worlds, {len(frame):,} task/method/split records.",
        "No new inference: honest agents replay frozen cached answers; attackers are symbolic and know gold.",
        "MATH-500 uses the repository math_match scorer and excludes semantically correct answers from the attack pool. Exact-string accuracy is retained as a sensitivity column.",
        "", "| Benchmark | f | AIP | Self | Majority | Trust only |",
        "|---|---:|---:|---:|---:|---:|"]
    for (b, f), row in subset.groupby(["benchmark", "f", "method"]).accuracy.mean().unstack().iterrows():
        lines.append(f"| {b} | {f:g} | {row.aip_gated:.3f} | {row.self_only:.3f} | {row.majority:.3f} | {row.aip_trust_only:.3f} |")
    lines += ["", "Full results: per_task.parquet, summary.csv, paired_comparisons.csv, selected_attacks.csv.",
        "Evidence: traces.jsonl.gz, channel_diagnostics.parquet, worlds.json, splits.json, run_manifest.json.",
        "", "History/validation/test are disjoint 40/20/40 task sets. Test never enters new channel fits or attack selection.",
        "Historical threshold calibration may overlap the caches; these are not wholly calibration-independent test results.",
        "CIs average seeds within each task before task bootstrap. They are pointwise exploratory intervals, without multiplicity correction.",
        "The finite-grid adaptive experiment is not a robustness or impossibility proof. See docs/DGX_CONTRIBUTIONS.md for scope and limitations.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def run(profile, out, resamples):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    thresholds = InversionThresholds.load(ROOT / "configs/inversion_thresholds.yaml")
    registry = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    models = sorted(n for n, entry in registry["models"].items() if entry.get("arm") == "frozen")
    caches, splits = {}, {}
    for b in ["mmlu", "boolq", "math500"]:
        cache = load_verified_cache(b, models)
        if profile == "smoke":
            cache = pc.BenchmarkCache(b, cache.task_ids[:60], cache.gold[:60], {k: v[:60] for k, v in cache.answers.items()})
        caches[b], splits[b] = cache, split_indices(cache.task_ids)
    (out / "splits.json").write_text(json.dumps({b: {k: [caches[b].task_ids[i] for i in v] for k, v in s.items()} for b, s in splits.items()}, indent=2))
    all_rows, channels, world_rows = [], [], []
    with gzip.open(out / "traces.jsonl.gz", "wt") as trace:
        for number, world in enumerate(worlds(profile), 1):
            b = world["benchmark"]
            cache, split = caches[b], splits[b]
            assignment, tasks, raw = build_world(cache, models, world)
            # Defense gets neither ground truth nor hidden identity/attack metadata.
            defense = [tuple(replace(obs, broadcasts=tuple(replace(x, is_byzantine=False,
                source_model=None, attack=None) for x in obs.broadcasts)) for obs in task) for task in tasks]
            context = dict(world, world_id=f"w{number:04d}")
            pairs = list(itertools.combinations(sorted(assignment.byzantine), 2))
            q = float(np.mean([raw[t][a].answer == raw[t][bb].answer for t in split["test"] for a, bb in pairs])) if pairs else None
            world_rows.append(dict(context, assignment=asdict(assignment), q_malicious_empirical_test=q,
                effective_sources=len(set(m for m in assignment.models if m))))
            trace_preds = {i: {} for i in split["test"][:3]}
            for method, make in factories(b, thresholds, world["study"] == "adaptive", world["seed"]).items():
                predictions = {part: {t: [] for t in split[part]} for part in ["validation", "test"]}
                for receiver in assignment.honest:
                    agg = make()
                    if agg is not None and agg.needs_fit:
                        agg.fit([(defense[t][receiver],) for t in split["history"]])
                    if isinstance(agg, AIPAggregator):
                        for peer, stats in agg.diagnostics.channels[receiver].items():
                            channels.append(dict(context, method=method, receiver=receiver, peer=peer,
                                peer_is_byzantine=peer in assignment.byzantine, **asdict(stats)))
                    for part, bucket in predictions.items():
                        for t in bucket:
                            obs = defense[t][receiver]
                            prediction = obs.own.answer if agg is None else agg.aggregate(obs.broadcasts, receiver)
                            bucket[t].append((float(score_answer(b, prediction, cache.gold[t])),
                                float(prediction == cache.gold[t])))
                            if part == "test" and t in trace_preds:
                                trace_preds[t].setdefault(method, {})[receiver] = prediction
                for part, bucket in predictions.items():
                    for t, values in bucket.items():
                        all_rows.append(dict(context, method=method, split=part, task_id=cache.task_ids[t],
                            accuracy=float(np.mean([v[0] for v in values])),
                            exact_match_accuracy=float(np.mean([v[1] for v in values])),
                            n_honest=len(assignment.honest)))
            for t, predictions in trace_preds.items():
                trace.write(json.dumps(dict(context, task_id=cache.task_ids[t], gold=cache.gold[t],
                    raw_broadcasts=[asdict(x) for x in raw[t]],
                    visibility={obs.self_id: [x.agent_id for x in obs.broadcasts] for obs in tasks[t]},
                    predictions=predictions), default=str) + "\n")
            print(f"{number:03d} {world['study']} {b} f={world['f']} obs={world['p_obs']} coh={world['coherence_p']} seed={world['seed']} elapsed={time.monotonic()-started:.1f}s", flush=True)
            if number % 12 == 0:
                pd.DataFrame(all_rows).to_parquet(out / "per_task.partial.parquet", index=False)
    frame = pd.DataFrame(all_rows)
    frame.to_parquet(out / "per_task.parquet", index=False)
    pd.DataFrame(channels).to_parquet(out / "channel_diagnostics.parquet", index=False)
    (out / "worlds.json").write_text(json.dumps(world_rows, indent=2, default=lambda x: sorted(x) if isinstance(x, frozenset) else str(x)))
    axes = ["study", "benchmark", "f", "p_obs", "composition", "coherence_p", "seed", "method", "split"]
    frame.groupby(axes).accuracy.agg(["mean", "count"]).reset_index().to_csv(out / "summary.csv", index=False)
    comparisons(frame, resamples).to_csv(out / "paired_comparisons.csv", index=False)
    selected, selected_predictions = select_attacks(frame)
    selected.to_csv(out / "selected_attacks.csv", index=False)
    selected_predictions.to_parquet(out / "selected_attack_predictions.parquet", index=False)
    meta = dict(profile=profile, worlds_completed=len(world_rows), models=models, inference_run=False,
        symbolic_attack=True, scoring="closed exact; MATH500 repository math_match",
        open_attack_pool="distinct assigned honest answers that fail math_match, or synthetic fallback",
        resamples=resamples, python=sys.version, machine=platform.machine(),
        elapsed_seconds=time.monotonic()-started, max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        cpu_affinity=sorted(os.sched_getaffinity(0)), input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/"configs/inversion_thresholds.yaml", ROOT/"configs/models.yaml"]
            + [ROOT/"data/cache"/b/f"{m}.parquet" for b in caches for m in models]},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), status="complete")
    report(out, frame, selected, meta)
    (out / "run_manifest.json").write_text(json.dumps(meta, indent=2))
    (out / "per_task.partial.parquet").unlink(missing_ok=True)
    print(f"COMPLETE {out} worlds={len(world_rows)} elapsed={meta['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    os.chdir(ROOT)
    run(args.profile, args.out.resolve(), args.resamples)
