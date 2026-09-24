#!/usr/bin/env python3
"""Independent reconciliation of the canonical C4 result, by the integration agent."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
os.nice(10)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from aip.tasks import roster  # noqa: E402
from extension_calibration import score  # noqa: E402


def main():
    started = time.monotonic()
    directory = ROOT / "results/extensions/calibration/standard"
    manifest = json.loads((directory / "run_manifest.json").read_text())
    checked_hashes = 0
    for name in ["source_sha256", "input_sha256", "output_sha256"]:
        base = directory if name == "output_sha256" else ROOT
        for path, expected in manifest[name].items():
            assert hashlib.sha256((base / path).read_bytes()).hexdigest() == expected, path
            checked_hashes += 1
    splits = json.loads((directory / "splits.json").read_text())
    gold, all_tasks = {}, {}
    for bm in manifest["benchmarks"]:
        frame = pd.read_parquet(ROOT / "data/cache" / bm / f"{manifest['models'][0]}.parquet")
        frame = frame[frame.sample_index == 0]
        gold[bm] = dict(zip(frame.task_id, frame.gold_answer, strict=True))
        all_tasks[bm] = set(frame.task_id)
        ids = [task for values in splits[bm].values() for task in values]
        assert len(set(ids)) == len(ids) and set(ids) == all_tasks[bm]
    worlds = {w["world_id"]: w for w in json.loads((directory / "worlds.json").read_text())}
    frame = pd.read_parquet(directory / "per_task.parquet")
    assert len(frame) == manifest["task_method_records"] == 90000
    assert len(worlds) == manifest["worlds_completed"] == 144
    assert not frame.duplicated(["world_id", "method", "task_id"]).any()
    scored = 0
    for row in frame.itertuples():
        assignment = worlds[row.world_id]["assignment"]
        receivers = sorted(set(range(10)) - set(assignment["byzantine"]))
        assert list(row.receiver_ids) == receivers and row.n_honest == len(receivers)
        assert row.task_id in splits[row.benchmark][row.split]
        correct = [score(row.benchmark, answer, gold[row.benchmark][row.task_id]) for answer in row.predictions]
        assert correct == list(row.correct)
        assert np.isclose(np.mean(correct), row.accuracy, atol=1e-12)
        scored += len(correct)
    axes = ["benchmark", "f", "p_obs", "seed", "method", "split"]
    rebuilt = frame.groupby(axes).accuracy.agg(["mean", "count"]).reset_index()
    published = pd.read_csv(directory / "summary.csv")
    pd.testing.assert_frame_equal(rebuilt, published, check_dtype=False, atol=1e-12, rtol=1e-12)
    comparisons = pd.read_csv(directory / "paired_comparisons.csv")
    for row in comparisons.itertuples():
        sub = frame[(frame.split == "test") & (frame.benchmark == row.benchmark)
                    & (frame.f == row.f) & (frame.p_obs == row.p_obs)]
        table = sub.groupby(["task_id", "method"]).accuracy.mean().unstack()
        values = (table[row.method] - table.legacy_gated).to_numpy()
        # Reconstruct seed and percentile draws here, not via the producing CI helper.
        parts = (row.benchmark, row.f, row.p_obs, row.method, "test_ci")
        seed = int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        means = rng.choice(values, (manifest["resamples"], len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(means, [.025, .975])
        assert np.allclose([values.mean(), low, high], [row.delta, row.ci_low, row.ci_high], atol=1e-12)
        assert row.n_unique_test_tasks == len(values) and row.n_swarm_seeds == 3
    channels = pd.read_parquet(directory / "channel_diagnostics.parquet")
    stats = {(r.world_id, r.method, r.receiver, r.peer): r for r in channels.itertuples()}
    trace_count, trace_predictions, wrong = 0, 0, 0
    with gzip.open(directory / "traces.jsonl.gz", "rt") as stream:
        for line in stream:
            trace = json.loads(line)
            assert trace["gold"] == gold[trace["benchmark"]][trace["task_id"]]
            assert trace["task_id"] in splits[trace["benchmark"]]["test"]
            for b in trace["raw_broadcasts"]:
                if b["is_byzantine"]:
                    assert not score(trace["benchmark"], b["answer"], trace["gold"])
                    wrong += 1
            for obs in trace["public_observations"]:
                receiver, reports = obs["self_id"], obs["broadcasts"]
                assert all(not b["is_byzantine"] and b["source_model"] is None and b["attack"] is None for b in reports)
                own = next(b["answer"] for b in reports if b["agent_id"] == receiver)
                for method, predictions in trace["predictions"].items():
                    if method == "self_only":
                        answer = own
                    else:
                        labels = roster.label_space(trace["benchmark"]) or sorted({b["answer"] for b in reports if b["answer"] is not None})
                        weights = {b["agent_id"]: stats[(trace["world_id"], method, receiver, b["agent_id"])].weight for b in reports}
                        total = sum(abs(w) for peer, w in weights.items() if peer != receiver)
                        if total > 0:
                            weights[receiver] = total / 9
                        scores = dict.fromkeys(labels, 0.)
                        for b in reports:
                            if b["answer"] not in scores:
                                continue
                            peer = b["agent_id"]
                            decision = "trust" if peer == receiver else stats[(trace["world_id"], method, receiver, peer)].decision
                            weight = weights[peer]
                            if decision == "trust":
                                scores[b["answer"]] += weight
                            elif decision == "invert" and len(labels) > 1:
                                scores[b["answer"]] -= weight
                                for label in labels:
                                    if label != b["answer"]:
                                        scores[label] += weight / (len(labels) - 1)
                        answer = min(k for k, v in scores.items() if np.isclose(v, max(scores.values()))) if labels else None
                    assert answer == predictions[str(receiver)]
                    trace_predictions += 1
            trace_count += 1
    thresholds = pd.read_csv(directory / "calibration_thresholds.csv")
    assert len(thresholds) == 180 and (thresholds.partner_ceiling == 1).all()
    cal_inputs = 0
    with gzip.open(directory / "calibration_inputs.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            obs = record["observation"]
            assert obs["task_id"] in splits[record["benchmark"]]["calibration"]
            assert obs["self_id"] == record["receiver"]
            assert all(not b["is_byzantine"] and b["source_model"] is None and b["attack"] is None for b in obs["broadcasts"])
            cal_inputs += 1
    result = dict(status="passed", checked_hashes=checked_hashes, worlds=144, task_method_rows=len(frame),
                  receiver_scores_recomputed=scored, summary_rows=len(published), paired_intervals_recomputed=len(comparisons),
                  traces=trace_count, trace_predictions_independent_weight_replay=trace_predictions,
                  scorer_valid_byzantine_trace_broadcasts=wrong, clean_calibration_observations=cal_inputs,
                  receiver_calibrations=len(thresholds), all_partner_ceilings_saturated=True,
                  elapsed_seconds=time.monotonic()-started,
                  limitation="Reuses the declared benchmark scorer; does not rederive all fitted channel states or externally adjudicate answers")
    out = ROOT / "results/extensions/calibration/independent_review.json"
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
