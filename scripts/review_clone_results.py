#!/usr/bin/env python3
"""Independently reconcile C5 task statistics, clone controls and weight traces."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter
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
from dgx_low_resource import score_answer  # noqa: E402


def main():
    started = time.monotonic()
    out = ROOT / "results/extensions/clone/standard"
    manifest = json.loads((out / "run_manifest.json").read_text())
    hashes = 0
    for key in ["source_sha256", "input_sha256"]:
        for name, expected in manifest[key].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
            hashes += 1
    for name, expected in json.loads((out / "artifact_manifest.json").read_text()).items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected, name
        hashes += 1
    frame = pd.read_parquet(out / "per_task.parquet")
    worlds = {w["world_id"]: w for w in json.loads((out / "worlds.json").read_text())}
    assert len(worlds) == manifest["worlds_completed"] == 180
    assert not frame.duplicated(["world_id", "method", "task_id"]).any()
    for world in worlds.values():
        assert world["actual_n"] == 10 + world["clone_count"]
        assert np.isclose(world["f_realized"], len(world["byzantine"])/world["actual_n"])
        assert world["original_honest_receivers"] == sorted(set(range(10))-set(world["base_byzantine"]))
        cell = frame[frame.world_id == world["world_id"]]
        assert cell.n_original_honest.eq(len(world["original_honest_receivers"])).all()
        assert cell.method.nunique() == 6 and cell.task_id.nunique() == 80 and len(cell) == 480
    axes = ["benchmark", "base_f", "clone_count", "source_kind", "p_obs", "f_realized", "seed", "method"]
    rebuilt = frame.groupby(axes).accuracy.agg(["mean", "count"]).reset_index()
    pd.testing.assert_frame_equal(rebuilt, pd.read_csv(out / "summary.csv"), check_dtype=False, atol=1e-12, rtol=1e-12)
    comparisons = pd.read_csv(out / "paired_comparisons.csv")
    for row in comparisons.itertuples():
        cell = frame[(frame.benchmark == row.benchmark) & (frame.base_f == row.base_f)
                     & (frame.clone_count == row.clone_count) & (frame.source_kind == row.source_kind)
                     & (frame.p_obs == row.p_obs)]
        paired = cell.groupby(["task_id", "method"]).accuracy.mean().unstack()
        differences = (paired[row.treatment]-paired[row.control]).to_numpy()
        rng = np.random.default_rng(int(row.bootstrap_seed))
        draws = rng.choice(differences, (manifest["resamples"], len(differences)), replace=True).mean(axis=1)
        low, high = np.quantile(draws, [.025, .975])
        assert np.allclose([differences.mean(), low, high], [row.delta, row.ci_low, row.ci_high], atol=1e-12)
    channels = pd.read_parquet(out / "channel_diagnostics.parquet")
    stats = {(r.world_id,r.method,r.receiver,r.peer):r for r in channels.itertuples()}
    rows = {(r.world_id,r.method,r.task_id):r for r in frame.itertuples()}
    base_inputs, fixed_sources = {}, {}
    traces, predictions, caps, wrong = 0, 0, 0, 0
    with gzip.open(out / "traces.jsonl.gz", "rt") as stream:
        for line in stream:
            trace = json.loads(line)
            world = worlds[trace["world_id"]]
            raw = {b["agent_id"]:b for b in trace["raw_broadcasts"]}
            key = (trace["benchmark"],trace["base_f"],trace["seed"],trace["p_obs"],trace["task_id"])
            base = ([raw[i] for i in range(10)], {r:[i for i in seen if i<10] for r,seen in trace["visibility"].items()})
            assert base_inputs.setdefault(key,base) == base
            if world["clone_source"] is not None:
                source_key = key[:-1] + (world["source_kind"],)
                assert fixed_sources.setdefault(source_key,world["clone_source"]) == world["clone_source"]
                for i in range(10,world["actual_n"]):
                    assert raw[i]["answer"] == raw[world["clone_source"]]["answer"]
            for b in raw.values():
                if b["is_byzantine"]:
                    assert not score_answer(trace["benchmark"],b["answer"],trace["gold"])
                    wrong += 1
            for method, answers in trace["predictions"].items():
                assert sorted(map(int,answers)) == world["original_honest_receivers"]
                correctness = []
                for receiver_text, expected in answers.items():
                    receiver = int(receiver_text)
                    reports = [raw[i] for i in trace["visibility"][receiver_text]]
                    labels = roster.label_space(trace["benchmark"]) or sorted({b["answer"] for b in reports if b["answer"] is not None})
                    if method == "self_only":
                        answer = raw[receiver]["answer"]
                    elif method == "majority":
                        counts = Counter(b["answer"] for b in reports if b["answer"] is not None)
                        answer = min(k for k,v in counts.items() if v == max(counts.values())) if counts else None
                    else:
                        native = method.removeprefix("clone_cap_")
                        weights = {b["agent_id"]:stats[(trace["world_id"],native,receiver,b["agent_id"])].weight for b in reports}
                        if method.startswith("clone_cap_"):
                            diagnostic = trace["cap_weights"][method][receiver_text]
                            assert all(np.isclose(weights[int(i)],v) for i,v in diagnostic["before_cap"].items())
                            for group in diagnostic["groups"]:
                                peers = [p for p in group if p in weights and p != receiver]
                                if receiver in group:
                                    for p in peers: weights[p] = 0.
                                else:
                                    total = sum(weights[p] for p in peers)
                                    highest = max((weights[p] for p in peers), default=0.)
                                    if total:
                                        for p in peers: weights[p] *= highest/total
                                    assert sum(weights[p] for p in peers) <= highest + 1e-12
                                caps += 1
                            assert all(np.isclose(weights[int(i)],v) for i,v in diagnostic["after_cap"].items())
                        total = sum(abs(w) for p,w in weights.items() if p != receiver)
                        if total: weights[receiver] = total/9
                        scores = dict.fromkeys(labels,0.)
                        for b in reports:
                            if b["answer"] not in scores: continue
                            peer = b["agent_id"]
                            action = "trust" if peer == receiver else stats[(trace["world_id"],native,receiver,peer)].decision
                            if action == "trust": scores[b["answer"]] += weights[peer]
                            elif action == "invert" and len(labels)>1:
                                scores[b["answer"]] -= weights[peer]
                                for label in labels:
                                    if label != b["answer"]: scores[label] += weights[peer]/(len(labels)-1)
                        answer = min(k for k,v in scores.items() if np.isclose(v,max(scores.values()))) if scores else None
                    assert answer == expected
                    correctness.append(score_answer(trace["benchmark"],answer,trace["gold"]))
                    predictions += 1
                row = rows[(trace["world_id"],method,trace["task_id"])]
                assert np.isclose(np.mean(correctness),row.accuracy)
            traces += 1
    result = dict(status="passed", checked_hashes=hashes, worlds=len(worlds), task_method_rows=len(frame),
                  summary_rows=len(rebuilt), paired_intervals_recomputed=len(comparisons),
                  traces=traces, independently_replayed_predictions=predictions, peer_group_caps_checked=caps,
                  scorer_valid_byzantine_trace_broadcasts=wrong, original_inputs_and_visibility_preserved=True,
                  clone_source_fixed_and_original_receivers_only=True, elapsed_seconds=time.monotonic()-started,
                  limitation="Uses saved gate statistics and the declared scorer; does not independently refit every channel/group or adjudicate truth")
    (out.parent/"result_reconciliation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
