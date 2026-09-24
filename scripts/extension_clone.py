#!/usr/bin/env python3
"""Bounded, serialized experiment for history-derived clone evidence caps."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import dgx_low_resource as replay
import numpy as np
import pandas as pd
import psutil
import yaml

from aip.aggregation.aip import AIPAggregator, InversionThresholds
from aip.aggregation.base import ParityConfig
from aip.aggregation.baselines import MajorityVote
from aip.extensions.clone_aware import CloneAwareAIP
from aip.types import Observation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand_world(
    base_assignment, base_raw, clone_count, source_kind, seed, benchmark, base_f, p_obs
):
    """Preserve base broadcasts and original-edge visibility when adding identities."""
    eligible = (
        base_assignment.honest if source_kind == "honest" else sorted(base_assignment.byzantine)
    )
    source = (
        eligible[
            replay.stable_seed(seed, benchmark, base_f, source_kind, "clone-source") % len(eligible)
        ]
        if clone_count
        else None
    )
    origin = {i: i for i in range(10)}
    for agent in range(10, 10 + clone_count):
        origin[agent] = source
    byzantine = set(base_assignment.byzantine)
    if source in base_assignment.byzantine:
        byzantine.update(range(10, 10 + clone_count))
    raw, observations = [], []
    for original in base_raw:
        task_id = original[0].task_id
        broadcasts = original + tuple(
            replace(original[source], agent_id=agent) for agent in range(10, 10 + clone_count)
        )
        raw.append(broadcasts)
        receiver_observations = {}
        for receiver in base_assignment.honest:
            heard = [
                b
                for b in broadcasts
                if b.agent_id == receiver
                or p_obs == 1.0
                or replay.stable_seed(seed, task_id, receiver, b.agent_id, "clone-visibility")
                / 2**64
                < p_obs
            ]
            # Experimental origin/model/adversary metadata is not available to a defense.
            clean = tuple(
                replace(b, is_byzantine=False, source_model=None, attack=None) for b in heard
            )
            receiver_observations[receiver] = Observation(receiver, task_id, clean)
        observations.append(receiver_observations)
    return (
        dict(
            source=source,
            origin=origin,
            byzantine=sorted(byzantine),
            actual_n=10 + clone_count,
            f_realized=len(byzantine) / (10 + clone_count),
        ),
        raw,
        observations,
    )


def recovery(groups, origin, source_family):
    """Ground-truth source fields are used here, in the scorer only."""
    group_of = {agent: i for i, group in enumerate(groups) for agent in group}
    result = []
    for target, labels in [
        ("copy_origin", origin),
        ("cached_model_or_attack_source", source_family),
    ]:
        counts = Counter(tp=0, fp=0, fn=0, tn=0)
        for left, right in itertools.combinations(sorted(group_of), 2):
            predicted, actual = group_of[left] == group_of[right], labels[left] == labels[right]
            counts[
                "tp" if predicted and actual else "fp" if predicted else "fn" if actual else "tn"
            ] += 1
        precision = (
            counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None
        )
        recall = (
            counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
        )
        result.append(dict(target=target, **counts, precision=precision, recall=recall))
    return result


def interval(values, seed, resamples=1000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(len(values), size=(resamples, len(values)))].mean(axis=1)
    return dict(
        delta=float(values.mean()),
        ci_low=float(np.quantile(sampled, 0.025)),
        ci_high=float(np.quantile(sampled, 0.975)),
        n_tasks=len(values),
        bootstrap_seed=seed,
    )


def paired_tables(frame, resamples):
    axes = ["benchmark", "base_f", "clone_count", "source_kind", "p_obs"]
    contrasts = [
        ("clone_cap_aip_gated", "aip_gated"),
        ("clone_cap_aip_trust_only", "aip_trust_only"),
        ("clone_cap_aip_gated", "clone_cap_aip_trust_only"),
    ]
    rows = []
    for key, group in frame.groupby(axes):
        table = group.groupby(["task_id", "method"]).accuracy.mean().unstack()
        for treatment, control in contrasts:
            result = interval(
                table[treatment] - table[control],
                replay.stable_seed("clone-contrast", key, treatment, control),
                resamples,
            )
            rows.append(
                dict(
                    zip(axes, key, strict=True),
                    treatment=treatment,
                    control=control,
                    f_realized=float(group.f_realized.iloc[0]),
                    n_seeds=group.seed.nunique(),
                    **result,
                )
            )
    clone_effects = []
    zero = frame[frame.clone_count == 0]
    for key, group in frame[frame.clone_count > 0].groupby(axes + ["method"]):
        b, f, count, kind, p, method = key
        reference = zero[
            (zero.benchmark == b) & (zero.base_f == f) & (zero.p_obs == p) & (zero.method == method)
        ]
        paired = group.merge(
            reference[["seed", "task_id", "accuracy"]],
            on=["seed", "task_id"],
            suffixes=("_cloned", "_base"),
            validate="one_to_one",
        )
        values = (paired.accuracy_cloned - paired.accuracy_base).groupby(paired.task_id).mean()
        result = interval(values, replay.stable_seed("clone-effect", key), resamples)
        clone_effects.append(
            dict(zip(axes + ["method"], key, strict=True), n_seeds=group.seed.nunique(), **result)
        )
    return pd.DataFrame(rows), pd.DataFrame(clone_effects)


def write_report(out, frame, paired, grouping, metadata):
    import matplotlib.pyplot as plt

    colors = {
        "self_only": "gray",
        "majority": "#999100",
        "aip_gated": "#1f77b4",
        "aip_trust_only": "#ff7f0e",
        "clone_cap_aip_gated": "#2ca02c",
        "clone_cap_aip_trust_only": "#d62728",
    }
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
    for i, benchmark in enumerate(["mmlu", "boolq", "math500"]):
        for j, source_kind in enumerate(["honest", "byzantine"]):
            ax = axes[i, j]
            subset = frame[
                (frame.benchmark == benchmark)
                & (frame.base_f == 0.5)
                & (frame.p_obs == 1.0)
                & ((frame.source_kind == source_kind) | (frame.clone_count == 0))
            ]
            for method, group in subset.groupby("method"):
                data = group.groupby("clone_count").accuracy.mean()
                ax.plot(data.index, data.values, marker="o", label=method, color=colors[method])
            ax.set(
                title=f"{benchmark}: {source_kind} source",
                xlabel="Added exact replicas",
                ylabel="Original honest receiver accuracy",
                ylim=(0, 1),
            )
            ax.grid(alpha=0.2)
    axes[-1, -1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Clone stress: fixed base N=10, base f=0.5, full visibility; seed means")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"accuracy_vs_clones.{ext}", dpi=170)
    plt.close(fig)
    primary = paired[(paired.treatment == "clone_cap_aip_gated") & (paired.control == "aip_gated")]
    focal = primary[(primary.clone_count == 15) & (primary.p_obs == 1.0)]
    positives, harms = int((primary.ci_low > 0).sum()), int((primary.ci_high < 0).sum())
    lines = [
        "# Extension 5: history-derived clone evidence caps",
        "",
        "**Research question:** Can grouping duplicate answer histories and capping each group's evidence reduce harmful copy amplification while retaining useful inversion?",
        "",
        f"Executed {metadata['worlds_completed']} worlds using three benchmarks, {len(metadata['seeds'])} seed(s), and {metadata['n_tasks_per_benchmark']} cached tasks per benchmark. No model inference or new checkpoints.",
        "",
        "## Design and controls",
        "",
        "The base ten-agent assignments and broadcasts stay fixed while 0, 5, or 15 exact replicas are added. A seeded original honest or Byzantine source is copied. Only the original honest receivers are scored; copied receivers never inflate sample size. Actual corruption fractions are recorded separately from base f.",
        "The static original AIP gate is fitted on history with an actual-N peer correction. Its fitted state is copied for the cap variant, isolating the pooling intervention within each world. Complete-link groups require at least 20 co-observed nonmissing history answers and at least 98% exact agreement for every within-group pair. No model or Byzantine label informs grouping. A group containing the receiver adds no extra peer self-evidence. Other peer-group mass is capped at its strongest present member before matched self parity (0.1).",
        "",
        "## Results, including harms",
        "",
        f"Across all {len(primary)} paired cap-versus-native cells, {positives} have a pointwise interval entirely above zero and {harms} entirely below zero. These exploratory counts have no multiplicity correction and are not proof of general improvement.",
        "",
        "| Benchmark | Base f | Copied source | Realized f | Capped gated − native gated | 95% task bootstrap interval |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in focal.itertuples(index=False):
        lines.append(
            f"| {row.benchmark} | {row.base_f:g} | {row.source_kind} | {row.f_realized:.3f} | {row.delta:+.3f} | [{row.ci_low:+.3f}, {row.ci_high:+.3f}] |"
        )
    for kind in ["honest", "byzantine"]:
        subset = primary[(primary.source_kind == kind) & (primary.clone_count > 0)]
        if len(subset):
            worst, best = subset.loc[subset.delta.idxmin()], subset.loc[subset.delta.idxmax()]
            lines += [
                "",
                f"For {kind}-source clones, cell mean changes range from {worst.delta:+.3f} to {best.delta:+.3f}. The most adverse cell is {worst.benchmark}, base f={worst.base_f:g}, clones={int(worst.clone_count)}, p_obs={worst.p_obs:g}; the most favorable cell is {best.benchmark}, base f={best.base_f:g}, clones={int(best.clone_count)}, p_obs={best.p_obs:g}.",
            ]
    lines += [
        "",
        "The cap has a precise limited guarantee: for fixed group membership and original gate statistics, a peer group's total pre-parity weight does not exceed its strongest member. Copies with identical decisions and current answers therefore cannot multiply that group's fixed-statistics evidence. Refitting AIP after adding copies can still change blind errors, selected coherence, agreement reference, and multiple-testing thresholds. Thus this experiment does not establish full-estimator clone invariance or arbitrary-Sybil security.",
        "",
        "## Source recovery and uncertainty",
        "",
        "group_recovery.csv scores two hidden-label targets: intentional copy-origin identity, and shared cached-model/coherent-attack source. Neither is used by the defense. A coherent bloc or two identical cached-model agents can be merged correctly as redundant broadcasts but count as a false merge under the narrower copy-origin target. Similar history is not proof of common provenance; honest near-correct workers can also be merged.",
        "Receiver accuracies are averaged within each task, then fixed seeds are averaged before paired task bootstrap. The history/validation/test split is 80/40/80 for a 200-item benchmark; validation is reserved and unused because the grouping parameters are fixed in advance. No test task changes the fitted groups. Intervals condition on these caches, seeds, and historical threshold calibration, which may overlap the current tasks.",
        "",
        "## Scope and reproducibility",
        "",
        "MMLU/BoolQ use exact labels. MATH-500 uses the repository's approximate math_match scorer and the v1 scorer-valid symbolic wrong-answer pool; exact-string sensitivity is also saved. Defense grouping/aggregation still use reported strings. Actual-N Bonferroni correction is matched between capped and native controls and is not held fixed across clone counts.",
        "See per_task.parquet, paired_comparisons.csv, clone_effects.csv, summary.csv, group_recovery.csv, group_pairs.parquet, groups.jsonl.gz, channel_diagnostics.parquet, traces.jsonl.gz, worlds.json, splits.json, run_manifest.json, resource_receipt.json, and accuracy_vs_clones.{png,pdf}.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


def run(args):
    started = time.monotonic()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    thresholds = InversionThresholds.load(ROOT / "configs/inversion_thresholds.yaml")
    registry = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    models = sorted(
        name for name, item in registry["models"].items() if item.get("arm") == "frozen"
    )
    seeds = [20260912] if args.profile == "smoke" else [20260912, 20260913, 20260914]
    f_values = [0.5] if args.profile == "smoke" else [0.3, 0.5]
    p_values = [1.0] if args.profile == "smoke" or args.visibility == "full" else [1.0, 0.5]
    cases = [(0, "none"), (5, "honest"), (15, "honest"), (5, "byzantine"), (15, "byzantine")]
    caches, splits = {}, {}
    for benchmark in ["mmlu", "boolq", "math500"]:
        cache = replay.load_verified_cache(benchmark, models)
        if args.profile == "smoke":
            cache = replay.pc.BenchmarkCache(
                benchmark,
                cache.task_ids[:60],
                cache.gold[:60],
                {key: value[:60] for key, value in cache.answers.items()},
            )
        caches[benchmark] = cache
        splits[benchmark] = replay.split_indices(cache.task_ids)
    (out / "splits.json").write_text(
        json.dumps(
            {
                b: {
                    part: [caches[b].task_ids[i] for i in indices]
                    for part, indices in split.items()
                }
                for b, split in splits.items()
            },
            indent=2,
        )
    )
    source_paths = [
        Path(__file__),
        ROOT / "src/aip/extensions/clone_aware.py",
        ROOT / "scripts/dgx_low_resource.py",
        ROOT / "scripts/phase_c_aggregate.py",
    ]
    source_paths += [
        ROOT / "src/aip" / path
        for path in [
            "aggregation/aip.py",
            "aggregation/base.py",
            "aggregation/baselines.py",
            "aggregation/correlation.py",
            "swarm/broadcast.py",
            "swarm/assignment.py",
            "swarm/topology.py",
            "tasks/math500.py",
        ]
    ]
    input_paths = [
        ROOT / "configs" / name
        for name in ["models.yaml", "inversion_thresholds.yaml", "task_lists.json"]
    ]
    input_paths += [ROOT / "data/cache" / b / f"{m}.parquet" for b in caches for m in models]
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    input_hashes = {str(path.relative_to(ROOT)): digest(path) for path in input_paths}
    rows, worlds, channels, recovery_rows, group_pairs = [], [], [], [], []
    with (
        gzip.open(out / "traces.jsonl.gz", "wt") as traces,
        gzip.open(out / "groups.jsonl.gz", "wt") as groups_file,
    ):
        number = 0
        for benchmark, base_f, seed in itertools.product(caches, f_values, seeds):
            cache, split = caches[benchmark], splits[benchmark]
            base_world = dict(
                seed=seed,
                benchmark=benchmark,
                composition="frozen_mix",
                attack="always_wrong",
                coherence_p=1.0,
                p_obs=1.0,
                f=base_f,
            )
            base_assignment, _, base_raw = replay.build_world(cache, models, base_world)
            for p_obs, (clone_count, source_kind) in itertools.product(p_values, cases):
                number += 1
                expanded, raw, observations = expand_world(
                    base_assignment,
                    base_raw,
                    clone_count,
                    source_kind,
                    seed,
                    benchmark,
                    base_f,
                    p_obs,
                )
                context = dict(
                    world_id=f"clone{number:04d}",
                    benchmark=benchmark,
                    base_f=base_f,
                    seed=seed,
                    p_obs=p_obs,
                    clone_count=clone_count,
                    source_kind=source_kind,
                    actual_n=expanded["actual_n"],
                    f_realized=expanded["f_realized"],
                )
                worlds.append(
                    dict(
                        context,
                        clone_source=expanded["source"],
                        copy_origin=expanded["origin"],
                        byzantine=expanded["byzantine"],
                        original_honest_receivers=list(base_assignment.honest),
                        base_models=base_assignment.models,
                        base_byzantine=sorted(base_assignment.byzantine),
                    )
                )
                scores = {
                    method: {t: [] for t in split["test"]}
                    for method in [
                        "self_only",
                        "majority",
                        "aip_gated",
                        "aip_trust_only",
                        "clone_cap_aip_gated",
                        "clone_cap_aip_trust_only",
                    ]
                }
                selected_traces = {
                    t: dict(
                        context,
                        task_id=cache.task_ids[t],
                        gold=cache.gold[t],
                        raw_broadcasts=[asdict(b) for b in raw[t]],
                        visibility={
                            r: [b.agent_id for b in obs.broadcasts]
                            for r, obs in observations[t].items()
                        },
                        predictions={},
                        cap_weights={},
                    )
                    for t in split["test"][:3]
                }
                source_family = {
                    agent: (
                        "coherent_attack"
                        if origin in base_assignment.byzantine
                        else f"cache:{base_assignment.models[origin]}"
                    )
                    for agent, origin in expanded["origin"].items()
                }
                for receiver in base_assignment.honest:
                    history = [(observations[t][receiver],) for t in split["history"]]
                    methods = {"self_only": None, "majority": MajorityVote()}
                    for mode in ["gated", "trust_only"]:
                        base = AIPAggregator(
                            benchmark,
                            mode,
                            thresholds,
                            ParityConfig(0.1),
                            replay.pc.LABEL_SPACES[benchmark],
                            n_peers_for_correction=expanded["actual_n"] - 1,
                        )
                        base.fit(history)
                        capped = CloneAwareAIP.from_fitted(
                            base, history, agreement_threshold=0.98, min_support=20
                        )
                        methods[base.name], methods[capped.name] = base, capped
                    grouping = methods["clone_cap_aip_gated"]
                    groups_file.write(
                        json.dumps(
                            dict(
                                context,
                                receiver=receiver,
                                groups=grouping.groups_[receiver],
                                training_task_ids=grouping.training_task_ids_[receiver],
                            )
                        )
                        + "\n"
                    )
                    for item in grouping.pair_support_[receiver]:
                        group_pairs.append(dict(context, receiver=receiver, **asdict(item)))
                    for item in recovery(
                        grouping.groups_[receiver], expanded["origin"], source_family
                    ):
                        recovery_rows.append(
                            dict(
                                context,
                                receiver=receiver,
                                n_groups=len(grouping.groups_[receiver]),
                                **item,
                            )
                        )
                    for name, method in methods.items():
                        if isinstance(method, AIPAggregator):
                            for peer, stats in method.diagnostics.channels[receiver].items():
                                channels.append(
                                    dict(
                                        context,
                                        method=name,
                                        receiver=receiver,
                                        peer=peer,
                                        peer_origin=expanded["origin"][peer],
                                        peer_is_byzantine=peer in expanded["byzantine"],
                                        **asdict(stats),
                                    )
                                )
                        for t in split["test"]:
                            obs = observations[t][receiver]
                            answer = (
                                obs.own.answer
                                if method is None
                                else method.aggregate(obs.broadcasts, receiver)
                            )
                            scores[name][t].append(
                                (
                                    replay.score_answer(benchmark, answer, cache.gold[t]),
                                    answer == cache.gold[t],
                                )
                            )
                            if t in selected_traces:
                                selected_traces[t]["predictions"].setdefault(name, {})[receiver] = (
                                    answer
                                )
                                if isinstance(method, CloneAwareAIP):
                                    selected_traces[t]["cap_weights"].setdefault(name, {})[
                                        receiver
                                    ] = method.weight_diagnostics(obs.broadcasts, receiver)
                    del methods
                for method, per_task in scores.items():
                    for t, values in per_task.items():
                        rows.append(
                            dict(
                                context,
                                method=method,
                                task_id=cache.task_ids[t],
                                split="test",
                                n_original_honest=len(base_assignment.honest),
                                accuracy=float(np.mean([value[0] for value in values])),
                                exact_match_accuracy=float(np.mean([value[1] for value in values])),
                            )
                        )
                for trace in selected_traces.values():
                    traces.write(json.dumps(trace) + "\n")
                print(
                    f"{number:03d} {benchmark} base_f={base_f} copies={clone_count} source={source_kind} obs={p_obs} seed={seed} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    grouping_frame = pd.DataFrame(recovery_rows)
    paired, effects = paired_tables(frame, args.resamples)
    frame.to_parquet(out / "per_task.parquet", index=False)
    pd.DataFrame(channels).to_parquet(out / "channel_diagnostics.parquet", index=False)
    pd.DataFrame(group_pairs).to_parquet(out / "group_pairs.parquet", index=False)
    grouping_frame.to_csv(out / "group_recovery.csv", index=False)
    paired.to_csv(out / "paired_comparisons.csv", index=False)
    effects.to_csv(out / "clone_effects.csv", index=False)
    axes = [
        "benchmark",
        "base_f",
        "clone_count",
        "source_kind",
        "p_obs",
        "f_realized",
        "seed",
        "method",
    ]
    frame.groupby(axes).accuracy.agg(["mean", "count"]).reset_index().to_csv(
        out / "summary.csv", index=False
    )
    (out / "worlds.json").write_text(json.dumps(worlds, indent=2))
    assert source_hashes == {path: digest(ROOT / path) for path in source_hashes}, (
        "Source changed during run"
    )
    metadata = dict(
        status="complete",
        profile=args.profile,
        extension="C5_clone_aware",
        worlds_completed=len(worlds),
        seeds=seeds,
        benchmarks=list(caches),
        n_tasks_per_benchmark=len(next(iter(caches.values())).task_ids),
        p_obs=p_values,
        base_f=f_values,
        clone_counts=[0, 5, 15],
        group_agreement_threshold=0.98,
        group_min_support=20,
        self_parity=0.1,
        peer_correction="actual_n - 1 for capped and native",
        resamples=args.resamples,
        inference_run=False,
        scoring="closed exact; MATH500 approximate math_match; exact-string sensitivity",
        attack_sampler="v1 scorer-valid base broadcasts; exact copies preserve original source answers",
        source_sha256=source_hashes,
        input_sha256=input_hashes,
        python=sys.version,
        platform=platform.platform(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        elapsed_seconds=time.monotonic() - started,
    )
    write_report(out, frame, paired, grouping_frame, metadata)
    (out / "run_manifest.json").write_text(json.dumps(metadata, indent=2))
    artifact_hashes = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "artifact_manifest.json").write_text(json.dumps(artifact_hashes, indent=2))
    print(
        f"COMPLETE {out} worlds={len(worlds)} elapsed={time.monotonic() - started:.1f}s", flush=True
    )


def bounded(args):
    out = args.out.resolve()
    if out.exists():
        raise SystemExit("Use a new output directory; existing results will not be overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Waiting for exclusive /tmp/aip-dgx-v2.lock", flush=True)
    with open("/tmp/aip-dgx-v2.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--profile",
            args.profile,
            "--out",
            str(out),
            "--visibility",
            args.visibility,
            "--resamples",
            str(args.resamples),
        ]
        cpus = [0, 1]
        if not set(cpus) <= os.sched_getaffinity(0):
            raise SystemExit("Required CPU cores 0 and 1 are unavailable")

        def setup():
            os.sched_setaffinity(0, cpus)
            os.nice(10)

        started = time.monotonic()
        peak, cpu, reason = 0, 0.0, None
        log_path = out.parent / f"{out.name}.log"
        with log_path.open("w") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, preexec_fn=setup
            )
            process = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    peak = max(peak, process.memory_info().rss)
                    times = process.cpu_times()
                    cpu = times.user + times.system
                except psutil.NoSuchProcess:
                    pass
                if peak > 2048 * 1024**2:
                    reason = "rss_limit"
                if time.monotonic() - started > args.timeout:
                    reason = "wall_timeout"
                if reason:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                time.sleep(0.2)
        out.mkdir(parents=True, exist_ok=True)
        complete = child.returncode == 0 and (out / "run_manifest.json").exists()
        receipt = dict(
            status="complete" if complete else "failed",
            exit_code=child.returncode,
            stop_reason=reason,
            elapsed_seconds=time.monotonic() - started,
            peak_sampled_rss_mib=peak / 1024**2,
            sampled_cpu_seconds=cpu,
            cpu_affinity=cpus,
            nice=10,
            computational_threads=1,
            inference_run=False,
            wall_limit_seconds=args.timeout,
            rss_limit_mib=2048,
            lock_file="/tmp/aip-dgx-v2.lock",
            command=command,
            log_file=log_path.name,
        )
        (out / "resource_receipt.json").write_text(json.dumps(receipt, indent=2))
        artifact_path = out / "artifact_manifest.json"
        if artifact_path.exists():
            artifacts = json.loads(artifact_path.read_text())
            artifacts["resource_receipt.json"] = digest(out / "resource_receipt.json")
            artifact_path.write_text(json.dumps(artifacts, indent=2))
        print(json.dumps(receipt, indent=2), flush=True)
        return 0 if complete else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    parser.add_argument("--out", type=Path, default=Path("results/extensions/clone/standard"))
    parser.add_argument("--visibility", choices=["full", "both"], default="both")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        run(args)
    else:
        raise SystemExit(bounded(args))
