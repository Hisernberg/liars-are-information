#!/usr/bin/env python3
"""Independent reconciliation of C6 saved chronology, fits, traces and scores.

This reviewer never fits a channel or changes canonical result files. It uses
the repository task scorer for semantic correctness, but independently derives
window membership, refresh cadence, references, arithmetic and control joins.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import resource
import signal
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

POLICIES = ("fixed40", "cumulative", "rolling32", "rolling64")
METHODS = ("self", "majority", *POLICIES)
CONTROLS = {"honest": "fixed_honest", "coherent": "fixed_coherent", "independent": "fixed_independent"}
DYNAMIC = {"sleeper", "coherent_to_independent", "toggle"}
EMPTY_HASH = hashlib.sha256(b"[]").hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal_number(left, right, message):
    require(math.isfinite(float(left)) and math.isfinite(float(right))
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12),
        f"{message}: {left!r} != {right!r}")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def json_lines(path):
    with gzip.open(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"Blank JSON record: {path.name}:{line_number}")
            yield json.loads(line)


def csv_rows(path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def regime(schedule, evaluation_step):
    if schedule in CONTROLS.values():
        return next(name for name, fixed in CONTROLS.items() if fixed == schedule)
    if schedule == "sleeper":
        return "honest" if evaluation_step < 40 else "coherent"
    if schedule == "coherent_to_independent":
        return "coherent" if evaluation_step < 40 else "independent"
    if schedule == "toggle":
        return "coherent" if evaluation_step < 40 or 80 <= evaluation_step < 120 else "independent"
    raise AssertionError(f"Unknown declared schedule {schedule}")


def history_range(method, step):
    if method == "fixed40":
        return range(40)
    if method == "cumulative":
        return range(step)
    return range(max(0, step - int(method.removeprefix("rolling"))), step)


def bit_mask(indices):
    return sum(1 << i for i in indices)


def generation_at(method, step):
    if step < 40:
        return 0
    return 1 if method == "fixed40" else 1 + (step - 40) // 8


@lru_cache(maxsize=50000)
def score(benchmark, answer, gold):
    if benchmark == "math500":
        from aip.tasks.math500 import math_match
        return bool(math_match(answer, gold))
    require(benchmark in {"mmlu", "boolq"}, f"Unreviewed benchmark scorer {benchmark}")
    return answer is not None and answer == gold


def reconcile(run):
    manifest = json.loads((run / "run_manifest.json").read_text())
    require(manifest.get("status") == "complete" and manifest.get("output_sha256"),
        "Canonical run is not finalized with output hashes")
    plan = json.loads((run / "predeclared_design.json").read_text())
    require(plan["warmup"] == 40 and plan["refresh_interval"] == 8
        and plan["rolling_windows"] == [32, 64] and tuple(plan["methods"]) == METHODS,
        "Reviewer supports the declared fixed40/cumulative/rolling32/64 protocol only")
    for field, value in plan.items():
        require(manifest[field] == value, f"Manifest/plan mismatch: {field}")
    verified_hashes = {}
    for family, root in [("source_sha256", ROOT), ("input_cache_sha256", ROOT), ("output_sha256", run)]:
        for relative, expected in manifest[family].items():
            path = (root / relative).resolve()
            require(path.is_relative_to(root.resolve()), f"Hash path escapes declared root: {relative}")
            actual = digest(path)
            require(actual == expected, f"Declared {family} hash mismatch: {relative}")
            verified_hashes[f"{family}:{relative}"] = actual
    chronology = json.loads((run / "chronology.json").read_text())
    index = {}
    for benchmark, item in chronology.items():
        ids = item["task_ids"]
        require(len(ids) == 40 + plan["evaluation_tasks"] and len(ids) == len(set(ids)),
            f"Invalid chronological coverage: {benchmark}")
        require(ids[:40] == item["warmup_task_ids"] and ids[40:] == item["evaluation_task_ids"],
            f"Chronology partition mismatch: {benchmark}")
        require(json_hash(ids) == item["chronology_sha256"] == manifest["chronology_sha256"][benchmark],
            f"Chronology hash mismatch: {benchmark}")
        index[benchmark] = {task_id: i for i, task_id in enumerate(ids)}
    world_list = json.loads((run / "worlds.json").read_text())
    worlds = {item["world_id"]: item for item in world_list}
    require(len(worlds) == len(world_list) == manifest["worlds_completed"], "World IDs duplicate/missing")
    expected_worlds = set(itertools.product(plan["benchmarks"], plan["fractions"], plan["seeds"], plan["schedules"]))
    observed_worlds = {(w["benchmark"], w["f"], w["seed"], w["schedule"]) for w in world_list}
    require(observed_worlds == expected_worlds and len(observed_worlds) == len(world_list),
        "World factorial coverage is incomplete or duplicated")
    world_lookup = {key: world_id for world_id, w in worlds.items()
        for key in [(w["benchmark"], w["f"], w["seed"], w["schedule"]) ]}
    for world_id, world in worlds.items():
        require(len(world["original_models"]) == plan["n_agents"] == 10, f"Agent count: {world_id}")
        latent = set(world["latent_byzantine"])
        require(len(latent) == round(10 * world["f"]), f"Realized corruption count: {world_id}")
        require(world["original_honest_receivers"] == sorted(set(range(10)) - latent),
            f"Original honest receiver set: {world_id}")
        require(world["chronology_sha256"] == chronology[world["benchmark"]]["chronology_sha256"],
            f"World chronology reference: {world_id}")
    def context(record):
        world = worlds[record["world_id"]]
        for field in ["benchmark", "seed", "schedule"]:
            require(str(record[field]) == str(world[field]), f"Context mismatch: {field}")
        equal_number(record["f"], world["f"], "Context corruption fraction")
        step = int(record["step"])
        require(0 <= step < len(chronology[world["benchmark"]]["task_ids"]), "Step outside chronology")
        require(int(record["evaluation_step"]) == step - 40, "Evaluation-step offset mismatch")
        require(record["regime"] == regime(world["schedule"], step - 40), "Regime/chronology mismatch")
        if "task_id" in record:
            require(record["task_id"] == chronology[world["benchmark"]]["task_ids"][step],
                "Task ID at wrong chronological step")
        return world, step

    # Compact fit index: keep masks instead of duplicating thousands of ID lists.
    fits = {}
    fit_counts = Counter()
    for record in json_lines(run / "fits.jsonl.gz"):
        world, step = context(record)
        method, receiver = record["method"], record["receiver"]
        require(method in POLICIES and receiver in world["original_honest_receivers"], "Invalid fit receiver/method")
        require(step >= 40 and (step == 40 if method == "fixed40" else (step - 40) % 8 == 0),
            "Fit violates refresh cadence")
        require(record["receiver_id"] == receiver and record["past_count"] == step == record["fit_at_past_count"],
            "Fit past-count or receiver alias mismatch")
        ids = chronology[world["benchmark"]]["task_ids"]
        positions = list(history_range(method, step))
        expected = [ids[i] for i in positions]
        require(record["history_task_ids"] == expected, f"Incorrect {method} fit history at step {step}")
        require(all(index[world["benchmark"]][tid] < step for tid in record["history_task_ids"]),
            "Current/future task leaked into fit")
        require(record["fit_task_ids_sha256"] == json_hash(expected), "Fit task-ID hash mismatch")
        require(record["fitted_history_tasks"] == len(expected), "Fitted history count mismatch")
        valid_ids = record["valid_fit_task_ids"]
        valid_set = set(valid_ids)
        require(len(valid_ids) == len(valid_set) and valid_ids == [tid for tid in expected if tid in valid_set],
            "Valid-fit IDs duplicate, reorder, or escape history")
        require(record["fitted_valid_tasks"] == len(valid_ids), "Valid-fit task count mismatch")
        generation = generation_at(method, step)
        require(record["fit_generation"] == generation, "Fit generation mismatch")
        key = (record["world_id"], method, receiver, generation)
        require(key not in fits, "Duplicate fit key")
        fits[key] = dict(step=step, history_mask=bit_mask(positions),
            valid_mask=bit_mask(index[world["benchmark"]][tid] for tid in valid_ids),
            history_count=len(expected), valid_count=len(valid_ids), history_hash=json_hash(expected),
            last_task=expected[-1] if expected else None)
        fit_counts[method] += 1
    expected_fits = set()
    for world_id, world in worlds.items():
        for receiver, method in itertools.product(world["original_honest_receivers"], POLICIES):
            total = generation_at(method, len(chronology[world["benchmark"]]["task_ids"]) - 1)
            expected_fits.update((world_id, method, receiver, i) for i in range(1, total + 1))
    require(set(fits) == expected_fits, "Fit records do not cover every expected refresh exactly once")

    # The numerical tables are indexed without retaining verbose CSV dictionaries.
    steps = {}
    for record in csv_rows(run / "per_step.csv"):
        world, step = context(record)
        method = record["method"]
        require(method in METHODS, "Unexpected per-step method")
        require(int(record["original_honest_receivers"]) == len(world["original_honest_receivers"]),
            "Per-step receiver denominator mismatch")
        key = (record["world_id"], step, method)
        require(key not in steps, "Duplicate per-step score")
        steps[key] = tuple(float(record[name]) for name in ["accuracy", "exact_string_accuracy",
            "self_accuracy", "regret_vs_self", "fitted_history_tasks_mean", "fitted_valid_tasks_mean", "receiver_refits"])
    expected_step_count = sum(len(chronology[w["benchmark"]]["task_ids"]) * len(METHODS) for w in world_list)
    require(len(steps) == expected_step_count == manifest["per_step_records"], "Per-step row coverage/count mismatch")

    # Read genuine gold from the hashed cache rather than trusting trace labels.
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    gold = {}
    for benchmark in plan["benchmarks"]:
        filename = ROOT / "data/cache" / benchmark / f"{plan['models'][0]}.parquet"
        table = pq.read_table(filename, columns=["task_id", "gold_answer"]).to_pydict()
        require(len(table["task_id"]) == len(set(table["task_id"])), "Cache task-ID collision")
        gold[benchmark] = dict(zip(table["task_id"], map(str, table["gold_answer"]), strict=True))

    seen_traces, broadcasts_hash = set(), {}
    nonmissing = Counter()
    refreshed_refs, all_refs = Counter(), 0
    checked_predictions, checked_adversarial = 0, 0
    for record in json_lines(run / "complete_traces.jsonl.gz"):
        world, step = context(record)
        world_id, task_id, benchmark = record["world_id"], record["task_id"], world["benchmark"]
        key = (world_id, step)
        require(key not in seen_traces, "Duplicate chronological trace")
        seen_traces.add(key)
        receivers = world["original_honest_receivers"]
        require(record["receiver_order"] == receivers, "Trace receiver order changed")
        require(record["evaluator_gold"] == gold[benchmark][task_id], "Trace gold differs from hashed cache")
        raw = record["broadcasts"]
        require([b["agent_id"] for b in raw] == list(range(10)), "Raw broadcast identity coverage/order")
        require(all(b["task_id"] == task_id for b in raw), "Raw mixed task IDs")
        active = set() if record["regime"] == "honest" else set(world["latent_byzantine"])
        require({b["agent_id"] for b in raw if b["is_byzantine"]} == active, "Raw active Byzantine mask mismatch")
        for broadcast in raw:
            if broadcast["answer"] is not None:
                nonmissing[(world_id, broadcast["agent_id"])] |= 1 << step
            if broadcast["agent_id"] in active:
                require(broadcast["answer"] is not None and not score(benchmark, broadcast["answer"], record["evaluator_gold"]),
                    f"Symbolic adversary is correct under scorer: {world_id}/{step}/{broadcast['agent_id']}")
                checked_adversarial += 1
        broadcasts_hash[key] = json_hash([record["receiver_order"], record["evaluator_gold"], raw])
        require(set(record["predictions"]) == set(METHODS), "Trace prediction method coverage")
        require(set(record["fitted_state_references"]) == set(POLICIES), "Trace state-reference method coverage")
        self_answers = [raw[receiver]["answer"] for receiver in receivers]
        require(record["predictions"]["self"] == self_answers, "Self prediction does not equal current own report")
        self_accuracy = sum(score(benchmark, answer, record["evaluator_gold"]) for answer in self_answers) / len(receivers)
        for method in METHODS:
            answers = record["predictions"][method]
            require(len(answers) == len(receivers), "Prediction receiver denominator mismatch")
            accuracy = sum(score(benchmark, answer, record["evaluator_gold"]) for answer in answers) / len(answers)
            exact = sum(answer is not None and answer == record["evaluator_gold"] for answer in answers) / len(answers)
            values = steps[(world_id, step, method)]
            for computed, saved, name in zip([accuracy, exact, self_accuracy, self_accuracy - accuracy], values[:4],
                    ["accuracy", "exact accuracy", "self accuracy", "regret"], strict=True):
                equal_number(computed, saved, f"Trace/per_step {name}: {world_id}/{step}/{method}")
            checked_predictions += len(answers)
            if method not in POLICIES:
                require(values[4:] == (0., 0., 0.), "Non-fitting baseline reports fitted support")
                continue
            refs = record["fitted_state_references"][method]
            require([item["receiver_id"] for item in refs] == receivers, "State-reference receiver order mismatch")
            fitted_counts, valid_counts, refreshed = [], [], 0
            for receiver, answer, ref in zip(receivers, answers, refs, strict=True):
                require(ref["answer"] == answer and ref["task_id"] == task_id
                    and ref["past_tasks_observed"] == step, "Prediction/state identity or past-count mismatch")
                retained = min(step, 40) if method == "fixed40" else step if method == "cumulative" else min(step, int(method[7:]))
                require(ref["retained_history_tasks"] == retained, "Retained buffer count mismatch")
                generation = generation_at(method, step)
                require(ref["fit_generation"] == generation, "Trace references a stale/future generation")
                should_refresh = step >= 40 and (step == 40 if method == "fixed40" else (step - 40) % 8 == 0)
                require(ref["refreshed"] == should_refresh, "Trace refresh flag mismatch")
                if generation == 0:
                    require(ref["fitted_history_tasks"] == ref["fitted_valid_tasks"] == 0
                        and ref["fit_last_task_id"] is None and ref["fit_task_ids_sha256"] == EMPTY_HASH,
                        "Cold-start state references a fit")
                    require(answer == raw[receiver]["answer"], "Cold-start method did not return self")
                else:
                    fit_key = (world_id, method, receiver, generation)
                    fitted = fits[fit_key]
                    require(fitted["step"] <= step, "Trace uses a future fit")
                    require(ref["fitted_history_tasks"] == fitted["history_count"]
                        and ref["fitted_valid_tasks"] == fitted["valid_count"]
                        and ref["fit_last_task_id"] == fitted["last_task"]
                        and ref["fit_task_ids_sha256"] == fitted["history_hash"], "Fit-reference metadata mismatch")
                    if should_refresh:
                        require(fitted["step"] == step, "Refresh linked to another task")
                        expected_valid = fitted["history_mask"] & nonmissing[(world_id, receiver)]
                        require(fitted["valid_mask"] == expected_valid,
                            "Valid-fit task IDs disagree with prior nonmissing self answers")
                        refreshed_refs[fit_key] += 1
                    if fitted["valid_count"] == 0:
                        require(answer == raw[receiver]["answer"], "No-valid-history method did not return self")
                fitted_counts.append(ref["fitted_history_tasks"])
                valid_counts.append(ref["fitted_valid_tasks"])
                refreshed += int(should_refresh)
                all_refs += 1
            equal_number(sum(fitted_counts) / len(receivers), values[4], "Per-step mean fitted history")
            equal_number(sum(valid_counts) / len(receivers), values[5], "Per-step mean valid fitted history")
            equal_number(refreshed, values[6], "Per-step refit count")
    require(len(seen_traces) * len(METHODS) == len(steps), "Trace coverage does not match per-step rows")
    require(set(refreshed_refs) == set(fits) and all(count == 1 for count in refreshed_refs.values()),
        "Each saved fit must have exactly one refreshed trace reference")

    # Channel rows are checked against referenced fit support, not re-estimated.
    peer_coverage, channel_count = Counter(), 0
    for record in csv_rows(run / "channel_states.csv"):
        world, step = context(record)
        receiver, peer = int(record["receiver"]), int(record["peer"])
        key = (record["world_id"], record["method"], receiver, int(record["fit_generation"]))
        fitted = fits[key]
        require(fitted["step"] == step and record["fit_task_ids_sha256"] == fitted["history_hash"],
            "Channel row references the wrong fit")
        require(int(record["past_count"]) == step and int(record["fitted_history_tasks"]) == fitted["history_count"]
            and int(record["fitted_valid_tasks"]) == fitted["valid_count"], "Channel fit-count metadata mismatch")
        require(0 <= peer < 10 and not (peer_coverage[key] & (1 << peer)), "Channel peer duplicate/outside swarm")
        peer_coverage[key] |= 1 << peer
        observed = (fitted["valid_mask"] & nonmissing[(record["world_id"], peer)]).bit_count()
        require(int(record["n_observed"]) == observed, "Channel n_observed differs from nonmissing valid history")
        require(0 <= int(record["coincidences"]) <= int(record["joint_dissents"]) <= fitted["valid_count"],
            "Invalid channel coincidence/support bounds")
        require((record["peer_originally_byzantine"] == "True") == (peer in world["latent_byzantine"]),
            "Channel original-adversary label mismatch")
        require((record["peer_currently_adversarial"] == "True")
            == (record["regime"] != "honest" and peer in world["latent_byzantine"]),
            "Channel current-adversary label mismatch")
        channel_count += 1
    require(channel_count == manifest["channel_state_records"], "Channel row count mismatch")
    for key, fitted in fits.items():
        expected = bit_mask(peer for peer in range(10) if fitted["valid_mask"] & nonmissing[(key[0], peer)])
        require(peer_coverage[key] == expected, "Channel rows omit/add a historically observed peer")

    # Pair every dynamic current task with its fixed-regime broadcast control.
    broadcast_matches, expected_controls = 0, set()
    for world_id, world in worlds.items():
        if world["schedule"] not in DYNAMIC:
            continue
        for step in range(40, 40 + plan["evaluation_tasks"]):
            current_regime = regime(world["schedule"], step - 40)
            fixed_schedule = CONTROLS[current_regime]
            fixed_world = world_lookup[(world["benchmark"], world["f"], world["seed"], fixed_schedule)]
            require(broadcasts_hash[(world_id, step)] == broadcasts_hash[(fixed_world, step)],
                f"Dynamic/control current broadcasts differ: {world_id}/{step}")
            broadcast_matches += 1
            expected_controls.update((world_id, step, method) for method in METHODS)
    seen_controls = set()
    for record in csv_rows(run / "same_task_controls.csv"):
        world, step = context(record)
        key = (record["world_id"], step, record["method"])
        require(key in expected_controls and key not in seen_controls, "Control table duplicate/unexpected row")
        seen_controls.add(key)
        target = CONTROLS[record["regime"]]
        require(record["control_schedule"] == target, "Control schedule differs from current regime")
        fixed_world = world_lookup[(world["benchmark"], world["f"], world["seed"], target)]
        dynamic_accuracy = steps[key][0]
        fixed_accuracy = steps[(fixed_world, step, record["method"])][0]
        equal_number(record["accuracy"], dynamic_accuracy, "Dynamic control-table accuracy")
        equal_number(record["control_accuracy"], fixed_accuracy, "Fixed control-table accuracy")
        equal_number(record["accuracy_minus_same_task_control"], dynamic_accuracy - fixed_accuracy, "Same-task history effect")
    require(seen_controls == expected_controls, "Same-task control table has missing rows")
    # Protect against a run changing underneath this read-only audit.
    for relative, expected in manifest["output_sha256"].items():
        require(digest(run / relative) == expected, f"Canonical output changed during review: {relative}")
    return dict(status="passed", reviewer_role="independent research-audit agent",
        worlds=len(worlds), chronology_tasks={b: len(d["task_ids"]) for b, d in chronology.items()},
        fit_records=len(fits), fits_by_method=dict(fit_counts), trace_records=len(seen_traces),
        prediction_scores_reconciled=checked_predictions, trace_fit_references=all_refs,
        per_step_rows=len(steps), channel_rows=channel_count, adversarial_broadcasts_scored_wrong=checked_adversarial,
        identical_dynamic_control_broadcast_pairs=broadcast_matches, same_task_control_rows=len(seen_controls),
        declared_hashes_verified=len(verified_hashes), manifest_sha256=digest(run / "run_manifest.json"),
        checks=["factorial world/receiver/chronology coverage", "all declared source/input/output hashes",
            "strictly past exact first40/all-past/last32/last64 history membership",
            "eight-step refresh cadence and fit-generation coverage", "valid history IDs match nonmissing self observations",
            "history hashes and every refreshed/nonrefreshed state reference", "trace prediction accuracy/exact/regret against per_step",
            "gold labels match hashed cache and active attackers are scorer-wrong", "channel references and observed-history support",
            "same-task current broadcasts identical to fixed-regime controls", "same-task control-table arithmetic and coverage",
            "canonical output hashes unchanged during review"],
        limitations=["No channel coefficients were independently refitted", "No CI/bootstrap reconstruction was requested",
            "Semantic correctness uses the repository math_match scorer, so scorer limitations remain",
            "Artifact consistency is not a robustness, calibration, or independence guarantee"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/extensions/online")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    require(not args.out.exists(), "Review output already exists; choose a fresh receipt path")
    require({18, 19}.issubset(os.sched_getaffinity(0)), "Reviewer requires CPUs 18,19")
    os.sched_setaffinity(0, {18, 19})
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    if nice < 10:
        os.nice(10 - nice)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = min(2 * 1024**3, hard) if hard >= 0 else 2 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    signal.alarm(args.timeout)
    started = time.monotonic()
    run = args.run_dir.resolve()
    try:
        receipt = reconcile(run)
    except Exception as error:
        receipt = dict(status="failed", reviewer_role="independent research-audit agent",
            error_type=type(error).__name__, error=str(error))
    signal.alarm(0)
    receipt.update(run_dir=str(run), review_script_sha256=digest(Path(__file__)),
        elapsed_seconds=time.monotonic()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0),
        numerical_threads=1, address_space_limit_bytes=limit, hostname=platform.node(), gpu_inference=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
