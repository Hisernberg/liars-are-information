#!/usr/bin/env python3
"""Reproduce paired C5 additional-copy effects after subtracting zero-copy gain."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def analyze(frame):
    rows = []
    axes = ["benchmark", "base_f", "clone_count", "source_kind", "p_obs"]
    for key, cell in frame[frame.clone_count > 0].groupby(axes):
        benchmark, fraction, copies, source, visibility = key
        base = frame[(frame.benchmark == benchmark) & (frame.base_f == fraction)
                     & (frame.p_obs == visibility) & (frame.clone_count == 0)]
        base_table = base.groupby(["task_id", "method"]).accuracy.mean().unstack()
        clone_table = cell.groupby(["task_id", "method"]).accuracy.mean().unstack()
        assert base_table.index.equals(clone_table.index)
        native_base, cap_base = base_table.aip_gated, base_table.clone_cap_aip_gated
        native_new, cap_new = clone_table.aip_gated, clone_table.clone_cap_aip_gated
        difference = (cap_new-native_new)-(cap_base-native_base)
        seed = int.from_bytes(hashlib.sha256(json.dumps(
            ("C5 clone-amplification DID", *key), sort_keys=True).encode()).digest()[:8], "big")
        draws = np.random.default_rng(seed).choice(difference.to_numpy(), (1000,len(difference)), replace=True).mean(axis=1)
        low, high = np.quantile(draws,[.025,.975])
        rows.append(dict(zip(axes,key,strict=True)) | dict(
            base_native_accuracy=native_base.mean(), base_cap_accuracy=cap_base.mean(),
            cloned_native_accuracy=native_new.mean(), cloned_cap_accuracy=cap_new.mean(),
            native_clone_effect=(native_new-native_base).mean(), cap_clone_effect=(cap_new-cap_base).mean(),
            difference_in_differences=difference.mean(), ci_low=low, ci_high=high,
            n_tasks=len(difference), n_seeds=cell.seed.nunique(), resamples=1000, bootstrap_seed=seed))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"results/extensions/clone/standard")
    parser.add_argument("--out",type=Path,help="Fresh CSV output; omitted for read-only canonical verification")
    args=parser.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    os.nice(10)
    actual=analyze(pd.read_parquet(args.run/"per_task.parquet"))
    if args.out:
        if args.out.exists(): raise FileExistsError("Use a fresh output path")
        args.out.parent.mkdir(parents=True,exist_ok=True)
        actual.to_csv(args.out,index=False)
    else:
        expected=pd.read_csv(args.run.parent/"clone_amplification_contrasts.csv")
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,atol=1e-12,rtol=1e-12)
    print(json.dumps(dict(status="passed",cells=len(actual),reconstructed_intervals=len(actual),
                         mode="export" if args.out else "read-only verification")))


if __name__ == "__main__":
    main()
