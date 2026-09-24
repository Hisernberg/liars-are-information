#!/usr/bin/env python3
"""C6: bounded strictly causal streaming AIP evaluation, with same-task controls."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import resource
import shutil
import signal
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import dgx_low_resource as v1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

from aip.aggregation.aip import InversionThresholds
from aip.aggregation.baselines import MajorityVote
from aip.extensions.online import CausalAIPReceiver, public_observation
from aip.swarm.assignment import assign_swarm
from aip.swarm.broadcast import AdversaryConfig, build_broadcasts
from aip.types import Observation

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
WARMUP = 40
METHODS = ("self", "majority", "fixed40", "cumulative", "rolling32", "rolling64")
DYNAMIC = ("sleeper", "coherent_to_independent", "toggle")
CONTROLS = {"honest": "fixed_honest", "coherent": "fixed_coherent", "independent": "fixed_independent"}
SCHEDULES = DYNAMIC + tuple(CONTROLS.values())


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def json_line(handle, value):
    handle.write(json.dumps(clean(value), separators=(",", ":"), allow_nan=False) + "\n")


class CsvStream:
    def __init__(self, path):
        self.handle = Path(path).open("w", newline="")
        self.writer = None
        self.rows = 0

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, list(row))
            self.writer.writeheader()
        self.writer.writerow(clean(row))
        self.rows += 1

    def close(self):
        self.handle.close()


def regime_for(schedule: str, evaluation_step: int) -> str:
    if schedule in CONTROLS.values():
        return next(regime for regime, name in CONTROLS.items() if schedule == name)
    if schedule == "sleeper":
        return "honest" if evaluation_step < 40 else "coherent"
    if schedule == "coherent_to_independent":
        return "coherent" if evaluation_step < 40 else "independent"
    if schedule == "toggle":
        return "coherent" if evaluation_step < 40 or 80 <= evaluation_step < 120 else "independent"
    raise ValueError(f"Unknown schedule: {schedule}")


def base_identity(models, benchmark, seed, fraction):
    rng = np.random.default_rng(v1.stable_seed("C6", benchmark, seed, "identities"))
    base = assign_swarm(10, 0., list(rng.permutation(models)), rng)
    order = np.random.default_rng(v1.stable_seed("C6", benchmark, seed, "corruption")).permutation(10)
    latent = frozenset(int(i) for i in order[:round(10 * fraction)])
    attacked = replace(base, byzantine=latent, models=tuple(
        None if i in latent else model for i, model in enumerate(base.models)
    ))
    return base, attacked


def broadcasts_at(cache, index, base, attacked, regime, seed):
    tid, gold = cache.task_ids[index], cache.gold[index]
    answers = {m: cache.answers[(m, 0)][index] for m in set(base.models)}
    assignment = base if regime == "honest" else attacked
    adversary = AdversaryConfig(kind="gate_aware", coherence=0. if regime == "independent" else 1.)
    rng = np.random.default_rng(v1.stable_seed("C6", seed, cache.benchmark, tid, regime, "attack"))
    if cache.benchmark == "math500":
        return v1.math_broadcasts(tid, gold, assignment, answers, adversary, rng)
    return build_broadcasts(tid, gold, assignment, answers,
        v1.pc.LABEL_SPACES[cache.benchmark] or [], adversary, rng)


def new_policies(receiver, benchmark, thresholds):
    common = dict(receiver_id=receiver, benchmark=benchmark, warmup=WARMUP,
        refresh_interval=8, thresholds=thresholds, label_space=v1.pc.LABEL_SPACES[benchmark],
        parity_share=.1)
    return {
        "fixed40": CausalAIPReceiver(**common, retention="fixed"),
        "cumulative": CausalAIPReceiver(**common, retention="cumulative"),
        "rolling32": CausalAIPReceiver(**common, retention="rolling", history_size=32),
        "rolling64": CausalAIPReceiver(**common, retention="rolling", history_size=64),
    }


def block_interval(values, regimes, seed, resamples=200, block_size=16):
    """Pair complete seeds and task blocks; do not IID-bootstrap receiver events."""
    values = np.asarray(values, float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Complete seed × chronological-task matrix is required")
    regimes = np.asarray(regimes)
    bounds = np.r_[0, 1 + np.flatnonzero(regimes[1:] != regimes[:-1]), len(regimes)]
    segments = [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for draw in range(resamples):
        # Shared task-block choices across sampled seeds retain dependence from
        # replaying the same benchmark tasks. Blocks stay inside each regime.
        sampled = []
        for segment in segments:
            length = len(segment)
            starts = rng.integers(0, length, size=math.ceil(length / block_size))
            local = np.concatenate([(start + np.arange(min(block_size, length))) % length
                                    for start in starts])[:length]
            sampled.extend(segment[local])
        seeds = rng.integers(0, len(values), len(values))
        draws[draw] = values[seeds][:, sampled].mean()
    low, high = np.quantile(draws, [.025, .975])
    return float(values.mean()), float(low), float(high)


def analysis(out, resamples):
    frame = pd.read_csv(out / "per_step.csv")
    evaluation = frame[frame.evaluation_step >= 0].copy()
    group = ["benchmark", "f", "schedule", "method"]
    summary = evaluation.groupby(group).agg(
        accuracy=("accuracy", "mean"), regret_vs_self=("regret_vs_self", "mean"),
        mean_history_tasks=("fitted_history_tasks_mean", "mean"),
        measured_method_seconds=("method_compute_seconds", "sum"),
        receiver_refits=("receiver_refits", "sum"), records=("task_id", "size"),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    evaluation.assign(time_bin=lambda f: f.evaluation_step // 16).groupby(
        group + ["time_bin"]
    ).agg(accuracy=("accuracy", "mean"), regret_vs_self=("regret_vs_self", "mean"),
        history_tasks=("fitted_history_tasks_mean", "mean"),
        n_steps=("evaluation_step", "nunique")).reset_index().to_csv(out / "time_windows.csv", index=False)
    paired = []
    for key, cell in evaluation.groupby(["benchmark", "f", "schedule"]):
        pivot = cell.pivot(index=["seed", "evaluation_step"], columns="method", values="accuracy")
        regimes = [regime_for(key[2], int(t)) for t in sorted(cell.evaluation_step.unique())]
        for method, baseline in itertools.product(METHODS[2:], ("self", "fixed40")):
            if method == baseline:
                continue
            values = (pivot[method] - pivot[baseline]).unstack("evaluation_step").to_numpy()
            point, low, high = block_interval(values, regimes,
                v1.stable_seed("C6 blocks", key, method, baseline), resamples)
            paired.append(dict(zip(["benchmark", "f", "schedule"], key, strict=True)) | dict(
                method=method, baseline=baseline, accuracy_difference=point,
                ci_low=low, ci_high=high, seeds=len(values), steps=values.shape[1],
                block_size=16, resamples=resamples,
                interpretation="exploratory paired seed/regime-local block bootstrap; not IID"))
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)

    # Match each current-regime task to a complete fixed-regime run using the
    # identical task, identities, receivers, and attack random stream. The only
    # difference is historical exposure and consequently fitted channel state.
    dynamic = evaluation[evaluation.schedule.isin(DYNAMIC)].copy()
    dynamic["control_schedule"] = dynamic.regime.map(CONTROLS)
    controls = evaluation[evaluation.schedule.isin(CONTROLS.values())][[
        "benchmark", "f", "seed", "task_id", "method", "schedule", "accuracy"
    ]].rename(columns={"schedule": "control_schedule", "accuracy": "control_accuracy"})
    matched = dynamic.merge(controls,
        on=["benchmark", "f", "seed", "task_id", "method", "control_schedule"],
        validate="many_to_one")
    matched["accuracy_minus_same_task_control"] = matched.accuracy - matched.control_accuracy
    matched.to_csv(out / "same_task_controls.csv", index=False)
    matched.groupby(group).agg(
        accuracy=("accuracy", "mean"), same_task_control_accuracy=("control_accuracy", "mean"),
        history_effect=("accuracy_minus_same_task_control", "mean"),
    ).reset_index().to_csv(out / "control_summary.csv", index=False)

    recovery = []
    for key, cell in matched.groupby(["benchmark", "f", "schedule", "method"]):
        switches = [40, 80, 120] if key[2] == "toggle" else [40]
        for switch in switches:
            next_switch = min([s for s in switches if s > switch] + [int(cell.evaluation_step.max()) + 1])
            segment = cell[(cell.evaluation_step >= switch) & (cell.evaluation_step < min(next_switch, switch + 64))].copy()
            if segment.empty:
                continue
            segment["relative_step"] = segment.evaluation_step - switch
            curve = segment.groupby("relative_step").agg(
                accuracy=("accuracy", "mean"), regret=("regret_vs_self", "mean"),
                gap=("accuracy_minus_same_task_control", "mean"))
            bins = [curve.iloc[start:start + 16] for start in range(0, len(curve), 16)]
            initial_gap = float(bins[0].gap.mean())
            closure = 0 if initial_gap >= -.02 else next(
                (min(16 * (i + 1), len(curve)) for i, chunk in enumerate(bins[1:], 1)
                 if chunk.gap.mean() >= -.02), None)
            recovery.append(dict(zip(["benchmark", "f", "schedule", "method"], key, strict=True)) | dict(
                switch_evaluation_step=switch, new_regime=regime_for(key[2], switch),
                observed_post_switch_steps=len(curve), first16_accuracy=float(bins[0].accuracy.mean()),
                first16_regret_vs_self=float(bins[0].regret.mean()),
                first16_control_gap=initial_gap, last16_control_gap=float(curve.iloc[-16:].gap.mean()),
                control_gap_closure_lag=closure, right_censored=closure is None,
                definition="first later nonoverlapping <=16-step bin with gap>=-0.02; 0=no initial deficit; descriptive, not necessarily sustained"))
    pd.DataFrame(recovery).to_csv(out / "switch_recovery.csv", index=False)

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    for schedule in DYNAMIC:
        subset = evaluation[evaluation.schedule == schedule]
        if subset.empty:
            continue
        benchmarks, fractions = sorted(subset.benchmark.unique()), sorted(subset.f.unique())
        fig, axes = plt.subplots(len(fractions), len(benchmarks), figsize=(13, 3.3 * len(fractions)),
            squeeze=False, sharex=True, sharey=True, constrained_layout=True)
        for row, fraction in enumerate(fractions):
            for column, benchmark in enumerate(benchmarks):
                ax = axes[row, column]
                cell = subset[(subset.benchmark == benchmark) & (subset.f == fraction)]
                for method in METHODS[1:]:
                    curve = cell[cell.method == method].assign(bin=lambda f: f.evaluation_step // 8).groupby("bin").regret_vs_self.mean()
                    ax.plot(curve.index * 8 + 3.5, curve, label=method, linewidth=1.5)
                ax.axhline(0, color="black", linewidth=.7)
                for switch in ([40, 80, 120] if schedule == "toggle" else [40]):
                    ax.axvline(switch, color="gray", linestyle="--", linewidth=.8)
                ax.set(title=f"{benchmark}, f={fraction:g}", xlabel="Evaluation step after 40-task warmup")
                ax.grid(alpha=.15)
        axes[0, 0].set_ylabel("Regret = self accuracy − method accuracy")
        axes[-1, -1].legend(fontsize=7)
        fig.suptitle(f"Strictly past-only AIP: {schedule} · eight-task display bins")
        for extension in ("png", "pdf"):
            fig.savefig(out / f"regime_{schedule}.{extension}", dpi=160)
        plt.close(fig)
    return summary, pd.DataFrame(paired), pd.DataFrame(recovery)


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_usage = resource.getrusage(resource.RUSAGE_SELF)
    signal.alarm(600)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min([2 * 1024**3] + [v for v in (soft, hard) if v >= 0])
    resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    thresholds_path = ROOT / "configs/inversion_thresholds.yaml"
    source_paths = [Path(__file__), ROOT / "src/aip/extensions/online.py",
        ROOT / "tests/test_extension_online.py", ROOT / "src/aip/aggregation/aip.py",
        ROOT / "src/aip/aggregation/correlation.py", ROOT / "src/aip/swarm/broadcast.py",
        ROOT / "src/aip/tasks/math500.py", ROOT / "scripts/dgx_low_resource.py",
        ROOT / "scripts/phase_c_aggregate.py", thresholds_path, ROOT / "configs/models.yaml"]
    source_at_start = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    thresholds = InversionThresholds.load(thresholds_path)
    registry = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    models = sorted(m for m, config in registry["models"].items() if config.get("arm") == "frozen")
    benchmarks = ["mmlu"] if args.profile == "smoke" else ["mmlu", "boolq", "math500"]
    seeds = [20260912] if args.profile == "smoke" else [20260912, 20260913, 20260914]
    fractions = [.5] if args.profile == "smoke" else [.5, .7]
    schedules = ["sleeper", "fixed_honest", "fixed_coherent"] if args.profile == "smoke" else SCHEDULES
    n_evaluation = 64 if args.profile == "smoke" else 160
    caches, chronologies, inputs = {}, {}, {}
    for benchmark in benchmarks:
        cache = v1.load_verified_cache(benchmark, models)
        order = sorted(range(len(cache.task_ids)), key=lambda i: v1.stable_seed("C6 chronology", benchmark, cache.task_ids[i]))[:WARMUP + n_evaluation]
        if len(order) != WARMUP + n_evaluation or len({cache.task_ids[i] for i in order}) != len(order):
            raise ValueError("Insufficient unique chronological tasks")
        caches[benchmark] = cache, order
        ids = [cache.task_ids[i] for i in order]
        chronologies[benchmark] = dict(task_ids=ids, warmup_task_ids=ids[:WARMUP],
            evaluation_task_ids=ids[WARMUP:],
            chronology_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            ordering="predeclared deterministic hash order of unique static benchmark tasks; simulated time")
        for model in models:
            path = ROOT / "data/cache" / benchmark / f"{model}.parquet"
            inputs[str(path.relative_to(ROOT))] = digest(path)
    write_json(out / "chronology.json", chronologies)
    plan = dict(extension="C6", profile=args.profile, benchmarks=benchmarks, seeds=seeds,
        fractions=fractions, schedules=list(schedules), methods=list(METHODS), models=models,
        n_agents=10, warmup=WARMUP, evaluation_tasks=n_evaluation, refresh_interval=8,
        rolling_windows=[32, 64], switch_evaluation_steps=[40, 80, 120],
        alpha_source="unchanged legacy inversion_thresholds.yaml", parity_share=.1,
        block_bootstrap_size=16, block_bootstrap_resamples=args.resamples,
        drift_heuristic="none; predeclared retention comparison only",
        recovery_gap_tolerance=.02, recovery_display_horizon=64)
    write_json(out / "predeclared_design.json", plan)
    world_records, elapsed_by_world = [], []
    steps, channels = CsvStream(out / "per_step.csv"), CsvStream(out / "channel_states.csv")
    majority = MajorityVote()
    with gzip.open(out / "complete_traces.jsonl.gz", "wt") as trace, gzip.open(out / "fits.jsonl.gz", "wt") as fits:
        worlds = list(itertools.product(benchmarks, fractions, seeds, schedules))
        for world_index, (benchmark, fraction, seed, schedule) in enumerate(worlds):
            world_start = time.monotonic()
            context = dict(world_id=f"c6_{world_index:03d}", benchmark=benchmark, f=fraction, seed=seed, schedule=schedule)
            base, attacked = base_identity(models, benchmark, seed, fraction)
            honest = attacked.honest
            cache, order = caches[benchmark]
            policies = {receiver: new_policies(receiver, benchmark, thresholds) for receiver in honest}
            world_records.append(context | dict(
                original_models=list(base.models), latent_byzantine=sorted(attacked.byzantine),
                original_honest_receivers=list(honest),
                chronology_sha256=chronologies[benchmark]["chronology_sha256"]))
            for step, index in enumerate(order):
                evaluation_step = step - WARMUP
                regime = regime_for(schedule, evaluation_step)
                raw = broadcasts_at(cache, index, base, attacked, regime, seed)
                gold, task_id = cache.gold[index], cache.task_ids[index]
                observations = {receiver: public_observation(Observation(receiver, task_id, raw))
                                for receiver in honest}
                self_scores = [float(v1.score_answer(benchmark, observations[r].own.answer, gold)) for r in honest]
                self_accuracy = float(np.mean(self_scores))
                predictions, states = {}, {}
                for method in METHODS:
                    answers, supports, valid_supports, refreshed, elapsed, state_rows = [], [], [], 0, 0., []
                    for receiver in honest:
                        observation = observations[receiver]
                        stamp = time.monotonic()
                        if method == "self":
                            answer = observation.own.answer
                        elif method == "majority":
                            answer = majority.aggregate(observation.broadcasts, receiver)
                        else:
                            policy = policies[receiver][method]
                            prediction = policy.predict(observation)
                            answer = prediction.answer
                            supports.append(prediction.fitted_history_tasks)
                            valid_supports.append(prediction.fitted_valid_tasks)
                            state_rows.append(asdict(prediction))
                            refreshed += int(prediction.refreshed)
                            if prediction.refreshed:
                                snapshot = policy.fit_snapshot()
                                if task_id in snapshot["history_task_ids"]:
                                    raise AssertionError("Current task leaked into its fitted history")
                                metadata = context | dict(method=method, receiver=receiver, step=step,
                                    evaluation_step=evaluation_step, regime=regime,
                                    fit_generation=prediction.fit_generation,
                                    past_count=prediction.past_tasks_observed,
                                    fitted_history_tasks=prediction.fitted_history_tasks,
                                    fitted_valid_tasks=prediction.fitted_valid_tasks,
                                    fit_task_ids_sha256=prediction.fit_task_ids_sha256)
                                json_line(fits, metadata | {k: v for k, v in snapshot.items() if k != "channels"})
                                for peer, stats in snapshot["channels"].items():
                                    channels.write(metadata | dict(peer=peer,
                                        peer_originally_byzantine=peer in attacked.byzantine,
                                        peer_currently_adversarial=regime != "honest" and peer in attacked.byzantine) | stats)
                            policy.update(observation)
                        elapsed += time.monotonic() - stamp
                        answers.append(answer)
                    scores = [float(v1.score_answer(benchmark, answer, gold)) for answer in answers]
                    predictions[method] = answers
                    if state_rows:
                        states[method] = state_rows
                    steps.write(context | dict(step=step, evaluation_step=evaluation_step,
                        task_id=task_id, regime=regime, method=method,
                        original_honest_receivers=len(honest), accuracy=float(np.mean(scores)),
                        exact_string_accuracy=float(np.mean([a is not None and a == gold for a in answers])),
                        self_accuracy=self_accuracy, regret_vs_self=self_accuracy - float(np.mean(scores)),
                        fitted_history_tasks_mean=float(np.mean(supports)) if supports else 0.,
                        fitted_valid_tasks_mean=float(np.mean(valid_supports)) if valid_supports else 0.,
                        receiver_refits=refreshed, method_compute_seconds=elapsed))
                json_line(trace, context | dict(step=step, evaluation_step=evaluation_step,
                    task_id=task_id, regime=regime, evaluator_gold=gold,
                    receiver_order=list(honest), broadcasts=[asdict(b) for b in raw],
                    predictions=predictions, fitted_state_references=states))
            elapsed_by_world.append(context | dict(wall_seconds=time.monotonic() - world_start))
            print(json.dumps(dict(world=world_index + 1, total=len(worlds), benchmark=benchmark,
                f=fraction, schedule=schedule, elapsed=round(time.monotonic() - started, 2))), flush=True)
    steps.close()
    channels.close()
    write_json(out / "worlds.json", world_records)
    pd.DataFrame(elapsed_by_world).to_csv(out / "world_runtime.csv", index=False)
    summary, paired, recovery = analysis(out, args.resamples)
    source_at_end = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    if source_at_start != source_at_end:
        raise RuntimeError("Analysis source changed during the run; refusing a completed manifest")
    if any(digest(ROOT / name) != value for name, value in inputs.items()):
        raise RuntimeError("Input cache changed during the run; refusing a completed manifest")
    manifest = dict(plan, status="complete", started_at_utc=datetime.fromtimestamp(time.time() - (time.monotonic() - started), UTC).isoformat(),
        finished_at_utc=datetime.now(UTC).isoformat(), worlds_completed=len(world_records),
        per_step_records=steps.rows, channel_state_records=channels.rows,
        source_sha256=source_at_start, source_unchanged_during_run=True,
        inputs_unchanged_during_run=True,
        input_cache_sha256=inputs, chronology_sha256={b: d["chronology_sha256"] for b, d in chronologies.items()},
        no_current_or_future_fit=True, old_aip_window_used=False, labels_passed_to_defender=False,
        byzantine_or_model_flags_passed_to_defender=False, new_model_inference=False,
        comparison_scope="fixed original honest receivers; all methods and controls share tasks/identities",
        bootstrap_interpretation="exploratory hierarchical seed and paired circular task blocks of 16 within regime segments; shared task draws across seeds; only 3 seeds; no IID receiver or time-step claims",
        calibration_caveat="legacy threshold calibration task IDs overlap current benchmark caches; no independent recalibration claim",
        task_difficulty_caveat="static benchmark tasks in simulated chronological order; fixed-regime same-task controls isolate history differences, but recovery remains descriptive and order-dependent")
    write_json(out / "run_manifest.json", manifest)
    render_report(out, summary, paired, recovery, manifest)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    receipt = dict(status="complete", hostname=platform.node(), architecture=platform.machine(),
        elapsed_seconds=time.monotonic() - started, user_cpu_seconds=usage.ru_utime - started_usage.ru_utime,
        system_cpu_seconds=usage.ru_stime - started_usage.ru_stime, peak_rss_mib=usage.ru_maxrss / 1024,
        cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0),
        computational_threads=1, arrow_cpu_threads=pa.cpu_count(), gpu_compute_used=False,
        address_space_limit_bytes=cap, wall_timeout_seconds=600,
        serialized_lock="/tmp/aip-dgx-v2.lock", python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "matplotlib", "pyarrow"]})
    write_json(out / "resource_receipt.json", receipt)
    manifest["output_sha256"] = {p.name: digest(p) for p in sorted(out.iterdir())
        if p.is_file() and p.name != "run_manifest.json"}
    manifest["output_bytes_excluding_manifest"] = sum(p.stat().st_size for p in out.iterdir()
        if p.is_file() and p.name != "run_manifest.json")
    write_json(out / "run_manifest.json", manifest)
    signal.alarm(0)
    print(json.dumps(dict(complete=True, resources=receipt, worlds=len(world_records)), indent=2))


def render_report(out, summary, paired, recovery, manifest):
    sleeper = summary[summary.schedule == "sleeper"].pivot(index=["benchmark", "f"], columns="method", values="accuracy")
    lines = ["# C6 — Strictly causal online AIP", "",
        "Research question: can a receiver using only committed past broadcasts adapt to sleeping or regime-switching adversaries, and what accuracy, history, and refitting costs result?", "",
        f"Completed {manifest['worlds_completed']} worlds with {manifest['warmup']} warmup and {manifest['evaluation_tasks']} unique chronological evaluation tasks per world. This is existing-cache replay on the DGX CPU; no model was loaded or generated a new answer.", "",
        "## Sleeper comparison", "",
        "| Benchmark | f | Self | Majority | Fixed40 | Cumulative | Rolling32 | Rolling64 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (benchmark, fraction), row in sleeper.iterrows():
        lines.append(f"| {benchmark} | {fraction:g} | " + " | ".join(f"{row[m]:.3f}" for m in METHODS) + " |")
    dynamic = paired[(paired.schedule.isin(DYNAMIC)) & (paired.baseline == "fixed40")]
    positive = dynamic[dynamic.ci_low > 0]
    negative = dynamic[dynamic.ci_high < 0]
    lines += ["", "## What the experiment supports", "",
        f"Across {len(dynamic)} dynamic-schedule method-versus-fixed40 contrasts, {len(positive)} exploratory block intervals lie wholly above zero and {len(negative)} wholly below zero. These pointwise counts are descriptive, with no multiplicity correction. Inspect per-benchmark contrasts; they do not justify a universal online improvement claim.", "",
        "The switch plots show regret relative to self (positive means worse). `same_task_controls.csv` compares each dynamic task with the identical task and current-regime broadcast in a complete fixed-regime control. Thus a gap between those runs measures their different historical exposure, while changes in raw pre/post accuracy alone can reflect task difficulty.", "",
        "`switch_recovery.csv` records the first and final 16-task control gap within up to 64 post-switch tasks, and a predeclared descriptive gap-closure lag. A lag of zero means no initial deficit greater than two percentage points. Null means the gap did not close within the observed horizon. Closure is not guaranteed to persist, and short 40-task toggle phases truncate some recovery windows.", "",
        "## Protocol and causal boundary", "",
        "All seven main frozen models receive stable identities in ten agent slots before selecting a nested corrupted set. Original honest receivers are fixed throughout; latent sleepers are never added to the scoring population while asleep. Their model identities return unchanged during honest behavior. Attack schedules switch at evaluation step 40; toggle switches again at 80 and 120. The 40-task warmup follows each schedule's initial regime.", "",
        "Every receiver calls predict(current) and then update(current). Fits contain only previously committed task IDs. Fixed40 fits the first 40 once; cumulative refits all past observations every eight tasks; rolling32/64 refit the last 32/64 task positions every eight tasks. Missing-self rows supply no channel evidence, and missing peer observations retain alignment. No older AIP window/pooled-future path is used. Before a valid fit, the prediction is self. All AIP variants retain self-vote parity 0.1 and the original gate/threshold constants; no drift heuristic was added.", "",
        "The receiver API erases hidden Byzantine/model/attack fields, never accepts gold, and rejects update-before-predict, duplicate task IDs, and changed public content during commit. Gold belongs only to the symbolic attacker and offline scorer. MATH-500 reuses the v1 semantic-valid wrong pool and math_match scorer, with exact-string accuracy also saved. BoolQ independent wrong draws and coherent wrong draws coincide because only one wrong label exists, so that regime switch cannot reveal an evasion adaptation mechanism.", "",
        "## Statistical and scientific limits", "",
        "The predeclared chronological order is a deterministic hash permutation of static benchmark tasks, shared across seeds and methods. It is simulated time, not a naturally evolving deployment dataset. Each task's score averages the ORIGINAL honest receivers; those receivers and repeated same-model slots are not independent replicates. Only three swarm/attack seeds are used.", "",
        f"Intervals use {manifest['block_bootstrap_resamples']} paired hierarchical bootstrap draws: resample seeds and circular task blocks of 16 within regime segments, with identical task-block draws across seeds and methods. This preserves local temporal and task-sharing dependence rather than assuming IID steps. Sixteen is a predeclared approximation; longer dependence induced by cumulative fitting is not fully captured. All intervals are exploratory, conditional on this roster, cache realization, chronology, attack mechanism, and old calibration constants. The old threshold task IDs overlap evaluation caches, so this is not independent calibration validation.", "",
        "No formal regret bound, independent-generation diversity result, robustness guarantee, or causal real-world recovery time is claimed. Synthetic attackers know gold and use the fixed predeclared p=0 or p=1 schedule; they do not optimize over new online attack policies in this extension.", "",
        "## Artifacts and reproduction", "",
        "Run `../.venv/bin/python scripts/extension_online.py --profile standard --out results/extensions/online --resamples 200` from the project root. The script serializes with the shared flock, uses affinity [0,1], nice 10, one mathematical thread, a 2 GiB address-space cap, and a 600-second post-lock timeout. Actual runtime and peak RSS are in resource_receipt.json; per-world time and per-method inclusive compute/I/O time are saved.", "",
        "Evidence: per_step.csv; channel_states.csv keyed by world/method/receiver/fit generation; fits.jsonl.gz with exact past task IDs; complete_traces.jsonl.gz containing every broadcast and prediction with state references; chronology.json; worlds.json; summary.csv; paired_comparisons.csv; time_windows.csv; same_task_controls.csv; control_summary.csv; switch_recovery.csv; regime plots in PNG/PDF; source, input and output hashes in run_manifest.json. Traces contain benchmark answer data for auditing but no question text or credentials.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    parser.add_argument("--out", type=Path, default=ROOT / "results/extensions/online")
    parser.add_argument("--resamples", type=int, default=200)
    args = parser.parse_args()
    if not 20 <= args.resamples <= 200:
        parser.error("Use 20–200 block bootstrap resamples")
    if args.out.exists() and any(p.is_file() for p in args.out.iterdir()):
        raise FileExistsError("Output directory already contains files; choose a fresh output path")
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError("Less than 20 GiB free disk")
    print("Waiting for shared DGX experiment lock", flush=True)
    with Path("/tmp/aip-dgx-v2.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        print("Acquired shared DGX experiment lock; causal run starting", flush=True)
        signal.signal(signal.SIGALRM, alarm_timeout)
        try:
            run(args)
        except BaseException as error:
            signal.alarm(0)
            args.out.mkdir(parents=True, exist_ok=True)
            failure = dict(status="failed", error_type=type(error).__name__, error=str(error),
                finished_at_utc=datetime.now(UTC).isoformat(),
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                cpu_affinity=sorted(os.sched_getaffinity(0)), gpu_compute_used=False)
            write_json(args.out / "resource_receipt.json", failure)
            manifest_path = args.out / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            write_json(manifest_path, manifest | failure)
            raise


def alarm_timeout(_signal_number, _frame):
    raise TimeoutError("C6 exceeded the 600-second post-lock experiment limit")


if __name__ == "__main__":
    main()
