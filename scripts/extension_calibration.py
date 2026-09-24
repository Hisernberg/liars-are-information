#!/usr/bin/env python3
"""C4: independent clean calibration and visibility/corruption transfer on cached swarms.

The default command acquires /tmp/aip-dgx-v2.lock and runs a bounded CPU child.
No models, network requests, or GPU inference are used.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import resource
import subprocess
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import yaml  # noqa: E402
import dgx_low_resource as replay  # noqa: E402
from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.extensions.calibration import (  # noqa: E402
    calibrate_partition, public_observation, split_task_ids, stable_seed,
    thresholds_with_ceiling,
)
from aip.swarm.broadcast import observe  # noqa: E402
from aip.swarm.topology import build_topology  # noqa: E402
from aip.tasks import roster  # noqa: E402
from aip.tasks.gsm8k import numeric_match  # noqa: E402

BENCHMARKS = ["arc", "mmlu", "boolq", "math500", "gsm8k", "medqa"]
METHODS = ["legacy_gated", "calibrated_pooled", "calibrated_partner", "trust_only", "self_only"]


@lru_cache(maxsize=50000)
def score(benchmark, answer, gold):
    if benchmark == "gsm8k":
        return bool(numeric_match(answer, gold))
    return replay.score_answer(benchmark, answer, gold)


def build_world(cache, models, world):
    """Reuse v1 identities/math sampler; apply the GSM8K numeric wrong criterion."""
    assignment, tasks, raw = replay.build_world(cache, models, world)
    if cache.benchmark != "gsm8k":
        return assignment, tasks, raw
    topology = build_topology("complete", 10)
    corrected, observations = [], []
    for t, (tid, gold) in enumerate(zip(cache.task_ids, cache.gold, strict=True)):
        rng = np.random.default_rng(replay.stable_seed(world["seed"], tid, "attack"))
        pool = sorted({b.answer for b in raw[t] if not b.is_byzantine and b.answer is not None
                       and not score("gsm8k", b.answer, gold)}) or [f"{gold}__adv"]
        shared = pool[int(rng.integers(len(pool)))]
        broadcasts = tuple(replace(b, answer=shared) if b.is_byzantine else b for b in raw[t])
        if any(score("gsm8k", b.answer, gold) for b in broadcasts if b.is_byzantine):
            raise ValueError("GSM8K symbolic attacker emitted a numerically correct answer")
        corrected.append(broadcasts)
        visibility_rng = np.random.default_rng(replay.stable_seed(world["seed"], tid, "visibility"))
        observations.append(observe(broadcasts, topology, world["p_obs"], visibility_rng))
    return assignment, observations, corrected


def paired_comparisons(frame, resamples):
    rows = []
    test = frame[frame.split == "test"]
    for (benchmark, fraction, visibility), group in test.groupby(["benchmark", "f", "p_obs"]):
        table = group.groupby(["task_id", "method"]).accuracy.mean().unstack()
        for method in ["calibrated_pooled", "calibrated_partner", "trust_only", "self_only"]:
            paired = table[[method, "legacy_gated"]].dropna()
            values = paired[method] - paired.legacy_gated
            point, low, high = replay.paired_interval(values.to_numpy(),
                stable_seed(benchmark, fraction, visibility, method, "test_ci"), resamples)
            rows.append(dict(benchmark=benchmark, f=fraction, p_obs=visibility, method=method,
                baseline="legacy_gated", delta=point, ci_low=low, ci_high=high,
                n_unique_test_tasks=len(paired), n_swarm_seeds=group.seed.nunique(),
                interval="pointwise task-cluster percentile bootstrap; seeds averaged"))
    return pd.DataFrame(rows)


def channel_error_rates(channels):
    frame = channels[channels.receiver != channels.peer].copy()
    frame["honest_channel"] = ~frame.peer_is_byzantine
    frame["false_inversion"] = (~frame.peer_is_byzantine) & (frame.decision == "invert")
    frame["adversarial_channel"] = frame.peer_is_byzantine
    frame["adversarial_trust"] = frame.peer_is_byzantine & (frame.decision == "trust")
    frame["adversarial_inversion"] = frame.peer_is_byzantine & (frame.decision == "invert")
    axes = ["benchmark", "f", "p_obs", "method"]
    summed = frame.groupby(axes)[["honest_channel", "false_inversion", "adversarial_channel",
        "adversarial_trust", "adversarial_inversion"]].sum().reset_index()
    for name, numerator, denominator in [
        ("honest_false_inversion_rate", "false_inversion", "honest_channel"),
        ("adversarial_false_trust_rate", "adversarial_trust", "adversarial_channel"),
        ("adversarial_inversion_rate", "adversarial_inversion", "adversarial_channel"),
    ]:
        summed[name] = summed[numerator] / summed[denominator].replace(0, np.nan)
    return summed


def write_report(out, frame, thresholds, comparisons, errors, manifest):
    import matplotlib.pyplot as plt
    data = frame[frame.split == "test"]
    fig, ax = plt.subplots(figsize=(10, 4))
    threshold_plot = thresholds.groupby("benchmark")[["legacy_ceiling", "pooled_ceiling", "partner_ceiling"]].mean()
    threshold_plot.plot.bar(ax=ax)
    ax.set(ylabel="Ceiling (mean across receivers and seeds)", ylim=(0, 1.05),
        title="Independent clean calibration: partner upper heuristic can saturate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"threshold_transfer.{extension}", dpi=160)
    plt.close(fig)
    benchmarks = list(data.benchmark.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, benchmark in zip(axes.flat, benchmarks, strict=False):
        for method in METHODS:
            subset = data[(data.benchmark == benchmark) & (data.method == method) & (data.p_obs == 1)]
            curve = subset.groupby("f").accuracy.mean()
            ax.plot(curve.index, curve.values, marker="o", label=method)
        ax.set(title=benchmark, xlabel="Byzantine fraction", ylabel="Test accuracy", ylim=(0, 1))
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(benchmarks):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("C4: disjoint calibration/history/test; full target visibility")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"accuracy_transfer.{extension}", dpi=160)
    plt.close(fig)
    accuracy = data.groupby(["benchmark", "f", "p_obs", "seed", "method"]).accuracy.mean().reset_index()
    averaged = accuracy.groupby(["benchmark", "method"]).accuracy.mean().unstack()
    saturation = float((thresholds.partner_ceiling >= .999999).mean())
    lines = ["# C4: independent clean calibration and transfer", "",
        "**Research question:** How do frozen legacy ceilings and current-population ceilings estimated on independent clean tasks change honest inversion and adversarial task accuracy?", "",
        f"Executed {manifest['worlds_completed']} cached worlds over {len(benchmarks)} benchmarks, {len(manifest['seeds'])} swarm seeds, Byzantine fractions {manifest['fractions']}, and visibility {manifest['visibility']}.",
        "Three matched gated variants use legacy, pooled upper-bootstrap, or conservative partner-selected upper-bootstrap ceilings. Trust-only and self-only provide mechanism controls. No new LLM inference was performed.", "",
        "## Independent design and transfer", "",
        "Task IDs were split once by hash into CAL/HISTORY/VALIDATION/TEST = 20/30/20/30%. Every clean receiver's calibration uses only CAL answers, at visibility 1.0, without labels. Complete task clusters are resampled to preserve all dependent peer responses. Thresholds then transfer unchanged to corrupted histories and both target visibility settings. Receiver channel fits use HISTORY only; VALIDATION and TEST are never fitted or used to select a threshold variant.",
        "Pooled upper bounds are the 97.5th bootstrap percentile of the ratio of all within-receiver joint-dissent coincidences. The partner heuristic takes the upper percentile of the maximum eligible pair rate (at least five joint dissents), bounded below by the pooled upper bound. This is a conservative heuristic, **not a simultaneous confidence bound or FWER guarantee**. No support triggers a declared ceiling of one. Binary tasks have structural ceiling one.", "",
        "## Measured accuracy", "",
        "These rows average each world first, then weight seeds, fractions and visibility equally. Per-cell paired intervals are in paired_comparisons.csv; benchmark averages below are descriptive.", "",
        "| Benchmark | Legacy | Pooled CAL | Partner CAL | Trust only | Self |",
        "|---|---:|---:|---:|---:|---:|"]
    for b, row in averaged.iterrows():
        lines.append(f"| {b} | {row.legacy_gated:.3f} | {row.calibrated_pooled:.3f} | {row.calibrated_partner:.3f} | {row.trust_only:.3f} | {row.self_only:.3f} |")
    lines += ["", "## Calibration behavior and negative outcomes", "",
        f"The partner ceiling reached one in {saturation:.1%} of receiver/seed calibrations; such a ceiling prevents the inherited q gate from rejecting that null. This is a concrete cost of conservative calibration under duplicated cached model responses and small calibration samples.", "",
        "| Variant | Test cells with interval above zero | Below zero | Overlapping zero |",
        "|---|---:|---:|---:|"]
    for method in ["calibrated_pooled", "calibrated_partner"]:
        subset = comparisons[comparisons.method == method]
        lines.append(f"| {method} | {int((subset.ci_low > 0).sum())} | {int((subset.ci_high < 0).sum())} | {int(((subset.ci_low <= 0) & (subset.ci_high >= 0)).sum())} |")
    clean = errors[(errors.f == 0) & (errors.p_obs == 1)]
    lines += ["", "Intervals above compare each variant against legacy AIP, averaging seeds within each task before task bootstrap. They are exploratory pointwise intervals with no multiplicity correction. An interval containing zero does not establish equivalence.", "",
        "## Clean-channel decision audit", "", "| Benchmark | Variant | False inversion / observed honest channels | Rate |", "|---|---|---:|---:|"]
    for _, row in clean.iterrows():
        lines.append(f"| {row.benchmark} | {row.method} | {int(row.false_inversion)}/{int(row.honest_channel)} | {row.honest_false_inversion_rate:.3f} |")
    lines += ["", "Channel decisions are fitted on HISTORY and frozen. Counts exclude receiver self channels; channels repeated across seeds/variants are dependent. Adversarial trust and inversion rates, including partial visibility, are saved in channel_error_rates.csv.", "",
        "## Limits", "",
        "The frozen legacy calibration may overlap any current split and used a historical population; only the new CAL estimates are task-disjoint here. Shared cached model outputs are reused by multiple agents, so ten agents are not ten independent models. Calibration has 40 tasks for the 200-task benchmarks and 100 for GSM8K. Upper bootstrap bounds on an aggregate statistic do not validate AIP's selected-partner binomial p-values. Replacing a ceiling also changes its inherited support floor and fallback calculations. Results concern coherent symbolic wrong-answer attacks over cached QA responses, not interactive or tool-using agents.",
        "MATH-500 uses the v1 semantic scorer/sampler. GSM8K uses the repository numeric_match scorer and rejects numerically correct candidates from the attack pool. Hidden Byzantine/model/attack metadata is removed before calibration and defender fitting; gold is available only for offline attack simulation and evaluation.", "",
        "## Reproducibility and resources", "",
        f"Worker host: {manifest['hostname']} ({manifest['machine']}); CPU affinity {manifest['cpu_affinity']}; one numerical thread; no GPU. Worker elapsed {manifest['elapsed_seconds']:.2f} seconds, max RSS {manifest['max_rss_mib']:.1f} MiB. The separate resource_receipt.json records enforced limits and monitored child status.",
        "Saved evidence: config.json, splits.json, calibration_thresholds.csv, calibration_inputs.jsonl.gz, per_task.parquet (including receiver predictions and correctness), channel_diagnostics.parquet, summary.csv, paired_comparisons.csv, channel_error_rates.csv, worlds.json, traces.jsonl.gz, figures, run_manifest.json, and resource_receipt.json.",
        "This extension tests calibration transfer; it does not claim algorithmic novelty or uniform improvement.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def run(profile, out, resamples, calibration_resamples):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    seeds = [20260912] if profile == "smoke" else [20260912, 20260913, 20260914]
    benchmarks = ["mmlu", "boolq", "math500", "gsm8k"] if profile == "smoke" else BENCHMARKS
    fractions = [0., .5] if profile == "smoke" else [0., .3, .5, .7]
    visibility = [1.] if profile == "smoke" else [1., .5]
    legacy = InversionThresholds.load(ROOT / "configs/inversion_thresholds.yaml")
    registry = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    models = sorted(name for name, entry in registry["models"].items() if entry.get("arm") == "frozen")
    config = dict(extension="C4", profile=profile, benchmarks=benchmarks, seeds=seeds,
        fractions=fractions, visibility=visibility, calibration_visibility=1., models=models,
        split_ratio=[.2, .3, .2, .3], split_seed=94027, resamples=resamples,
        calibration_resamples=calibration_resamples, min_partner_support=5,
        calibration_interval="task-bootstrap percentile .025/.975", methods=METHODS,
        calibration_selection="none; all predefined variants evaluated", attack="coherent_symbolic_wrong",
        scoring="closed exact, MATH500 math_match, GSM8K numeric_match", gpu_inference=False)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    caches, splits = {}, {}
    for benchmark in benchmarks:
        cache = replay.load_verified_cache(benchmark, models)
        if profile == "smoke":
            cache = replay.pc.BenchmarkCache(benchmark, cache.task_ids[:60], cache.gold[:60],
                {k: values[:60] for k, values in cache.answers.items()})
        caches[benchmark] = cache
        splits[benchmark] = split_task_ids(cache.task_ids)
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    calibrated, threshold_rows = {}, []
    with gzip.open(out / "calibration_inputs.jsonl.gz", "wt") as handle:
        for benchmark, seed in itertools.product(benchmarks, seeds):
            clean_world = dict(benchmark=benchmark, seed=seed, f=0., p_obs=1.,
                composition="frozen_mix", attack="always_wrong", coherence_p=1.)
            cache = caches[benchmark]
            assignment, clean, _ = build_world(cache, models, clean_world)
            for receiver in assignment.honest:
                observations = {cache.task_ids[t]: clean[t][receiver] for t in range(len(clean))}
                fitted = calibrate_partition(observations, splits[benchmark], receiver=receiver,
                    n_options=roster.n_options(benchmark), resamples=calibration_resamples,
                    seed=stable_seed(benchmark, seed, receiver, "calibration"))
                calibrated[(benchmark, seed, receiver)] = fitted
                threshold_rows.append(dict(benchmark=benchmark, seed=seed, calibration_p_obs=1.,
                    legacy_ceiling=legacy.ceiling_for(benchmark), **fitted.as_dict()))
                for task_id in splits[benchmark]["calibration"]:
                    handle.write(json.dumps(dict(benchmark=benchmark, seed=seed, receiver=receiver,
                        observation=asdict(public_observation(observations[task_id])))) + "\n")
            print(f"CAL {benchmark} seed={seed} tasks={len(splits[benchmark]['calibration'])} elapsed={time.monotonic()-started:.1f}s", flush=True)
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(out / "calibration_thresholds.csv", index=False)
    all_rows, channel_rows, world_rows = [], [], []
    with gzip.open(out / "traces.jsonl.gz", "wt") as trace:
        for number, (benchmark, seed, fraction, p_obs) in enumerate(itertools.product(
            benchmarks, seeds, fractions, visibility), 1):
            world = dict(benchmark=benchmark, seed=seed, f=fraction, p_obs=p_obs,
                composition="frozen_mix", attack="always_wrong", coherence_p=1.)
            cache, partitions = caches[benchmark], splits[benchmark]
            indices = {name: [cache.task_ids.index(tid) for tid in task_ids]
                       for name, task_ids in partitions.items()}
            assignment, tasks, raw = build_world(cache, models, world)
            defense = [tuple(public_observation(obs) for obs in row) for row in tasks]
            context = dict(benchmark=benchmark, seed=seed, f=fraction, p_obs=p_obs,
                world_id=f"c4w{number:04d}", calibration_p_obs=1.)
            world_rows.append(dict(context, assignment=asdict(assignment)))
            traced = {t: {} for t in indices["test"][:2]}
            for method in METHODS:
                predictions = {part: {t: [] for t in indices[part]} for part in ["validation", "test"]}
                for receiver in assignment.honest:
                    calibration = calibrated[(benchmark, seed, receiver)]
                    thresholds = legacy
                    if method == "calibrated_pooled":
                        thresholds = thresholds_with_ceiling(legacy, benchmark, calibration.pooled_ceiling)
                    elif method == "calibrated_partner":
                        thresholds = thresholds_with_ceiling(legacy, benchmark, calibration.partner_ceiling)
                    agg = None if method == "self_only" else AIPAggregator(benchmark,
                        "trust_only" if method == "trust_only" else "gated", thresholds,
                        ParityConfig(.1), replay.pc.LABEL_SPACES[benchmark])
                    if agg is not None:
                        agg.fit([(defense[t][receiver],) for t in indices["history"]])
                        for peer, stats in agg.diagnostics.channels[receiver].items():
                            channel_rows.append(dict(context, method=method, receiver=receiver, peer=peer,
                                peer_is_byzantine=peer in assignment.byzantine,
                                operative_ceiling=thresholds.ceiling_for(benchmark), **asdict(stats)))
                    for part, bucket in predictions.items():
                        for t in bucket:
                            observation = defense[t][receiver]
                            answer = observation.own.answer if agg is None else agg.aggregate(observation.broadcasts, receiver)
                            bucket[t].append((receiver, answer, bool(score(benchmark, answer, cache.gold[t]))))
                            if part == "test" and t in traced:
                                traced[t].setdefault(method, {})[receiver] = answer
                for part, bucket in predictions.items():
                    for t, values in bucket.items():
                        all_rows.append(dict(context, method=method, split=part, task_id=cache.task_ids[t],
                            accuracy=float(np.mean([v[2] for v in values])), n_honest=len(values),
                            receiver_ids=[v[0] for v in values], predictions=[v[1] for v in values],
                            correct=[v[2] for v in values]))
            for t, predictions in traced.items():
                trace.write(json.dumps(dict(context, task_id=cache.task_ids[t], gold=cache.gold[t],
                    raw_broadcasts=[asdict(b) for b in raw[t]], predictions=predictions,
                    public_observations=[asdict(defense[t][receiver]) for receiver in assignment.honest])) + "\n")
            print(f"WORLD {number:03d} {benchmark} seed={seed} f={fraction} obs={p_obs} elapsed={time.monotonic()-started:.1f}s", flush=True)
    frame, channels = pd.DataFrame(all_rows), pd.DataFrame(channel_rows)
    frame.to_parquet(out / "per_task.parquet", index=False)
    channels.to_parquet(out / "channel_diagnostics.parquet", index=False)
    (out / "worlds.json").write_text(json.dumps(world_rows, indent=2,
        default=lambda value: sorted(value) if isinstance(value, frozenset) else str(value)))
    axes = ["benchmark", "f", "p_obs", "seed", "method", "split"]
    frame.groupby(axes).accuracy.agg(["mean", "count"]).reset_index().to_csv(out / "summary.csv", index=False)
    comparisons = paired_comparisons(frame, resamples)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    errors = channel_error_rates(channels)
    errors.to_csv(out / "channel_error_rates.csv", index=False)
    source_files = [Path(__file__), ROOT / "src/aip/extensions/calibration.py",
        ROOT / "scripts/dgx_low_resource.py", ROOT / "src/aip/aggregation/aip.py",
        ROOT / "src/aip/swarm/broadcast.py", ROOT / "src/aip/tasks/gsm8k.py",
        ROOT / "src/aip/tasks/math500.py"]
    inputs = [ROOT / "configs/models.yaml", ROOT / "configs/inversion_thresholds.yaml"] + [
        ROOT / "data/cache" / benchmark / f"{model}.parquet" for benchmark in benchmarks for model in models]
    manifest = dict(config, status="complete", worlds_completed=len(world_rows),
        calibration_receivers=len(threshold_rows), task_method_records=len(frame),
        hostname=platform.node(), machine=platform.machine(), platform=platform.platform(),
        python=sys.version, cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0),
        numerical_threads=1, elapsed_seconds=time.monotonic()-started,
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        versions={name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "pyarrow", "sympy"]},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    write_report(out, frame, threshold_frame, comparisons, errors, manifest)
    manifest["elapsed_seconds_including_report"] = time.monotonic() - started
    manifest["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    manifest["output_sha256"] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir()) if p.is_file()}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"COMPLETE {out} worlds={len(world_rows)} elapsed={manifest['elapsed_seconds_including_report']:.1f}s", flush=True)


def bounded(args):
    out = args.out.resolve()
    if out.exists():
        raise ValueError("Output exists; choose a fresh run directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--profile", args.profile,
        "--out", str(out), "--resamples", str(args.resamples),
        "--calibration-resamples", str(args.calibration_resamples)]
    cpus = [0, 1]
    if not set(cpus).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Required CPUs 0,1 are unavailable")
    lock_path = Path("/tmp/aip-dgx-v2.lock")
    print(f"Waiting for shared experiment lock {lock_path}", flush=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.monotonic()
        peak, cpu_seconds, reason = 0, 0., None
        log_path = out.parent / f"{out.name}.log"
        def setup():
            os.sched_setaffinity(0, cpus)
            os.nice(10)
        with log_path.open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, preexec_fn=setup)
            process = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    peak = max(peak, process.memory_info().rss)
                    times = process.cpu_times()
                    cpu_seconds = times.user + times.system
                except psutil.NoSuchProcess:
                    pass
                if peak > args.rss_limit_mib * 1024**2:
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
                time.sleep(.2)
        manifest_path = out / "run_manifest.json"
        status = "complete" if child.returncode == 0 and manifest_path.exists() else "failed"
        out.mkdir(exist_ok=True)
        receipt = dict(status=status, exit_code=child.returncode, stop_reason=reason,
            elapsed_seconds=time.monotonic()-started, peak_sampled_rss_mib=peak / 1024**2,
            sampled_cpu_seconds=cpu_seconds, cpu_affinity=cpus, nice=10, numerical_threads=1,
            rss_limit_mib=args.rss_limit_mib, wall_limit_seconds=args.timeout,
            command=command, lock=str(lock_path), hostname=platform.node(),
            gpu_inference=False, log_file=log_path.name)
        (out / "resource_receipt.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2), flush=True)
        return 0 if status == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--calibration-resamples", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--rss-limit-mib", type=int, default=2048)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        run(args.profile, args.out.resolve(), args.resamples, args.calibration_resamples)
    else:
        raise SystemExit(bounded(args))
