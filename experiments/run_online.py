#!/usr/bin/env python3
"""E6: causal online evaluation under sleeper and regime-switching attackers."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from lai.online import SCHEDULES, OnlineWorld, run_online  # noqa: E402


def _one(world: OnlineWorld) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    return run_online(world)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "online")
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    worlds = [
        OnlineWorld(b, f, s, seed)
        for b, f, s, seed in itertools.product(
            ("mmlu", "medqa", "boolq", "math500"), (0.3, 0.5, 0.7), tuple(SCHEDULES), range(args.seeds)
        )
    ]
    import multiprocessing as mp

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frames = []
    with mp.get_context("fork").Pool(args.processes) as pool:
        for i, frame in enumerate(pool.imap_unordered(_one, worlds), 1):
            frames.append(frame)
            print(f"[{i}/{len(worlds)}] {time.monotonic() - started:.0f}s", flush=True)
    per_step = pd.concat(frames, ignore_index=True)
    per_step.to_parquet(args.out / "per_step.parquet", index=False)
    (args.out / "run_manifest.json").write_text(json.dumps(
        {"worlds": len(worlds), "rows": len(per_step), "elapsed_seconds": round(time.monotonic() - started, 1),
         "schedules": list(SCHEDULES), "gpu_inference": False}, indent=2))


if __name__ == "__main__":
    main()
