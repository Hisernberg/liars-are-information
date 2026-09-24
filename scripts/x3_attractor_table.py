#!/usr/bin/env python3
"""The attractor-cell table MX's D-2 decision rests on, recomputed from the cache.

The attractor cell is the state L1 (the attractor-rider liar) needs: two honest
models both wrong AND agreeing on the same wrong answer. Its frequency decides
which benchmark can test MX-H1 at all, so it has to be recomputed after X3 --
the manufactured rows were near-random disagreement, which suppressed exactly
this quantity.

Pooled over every cross-model pair of the cached roster, per benchmark.

    python scripts/x3_attractor_table.py
    python scripts/x3_attractor_table.py --ref pre-x3-cache   # from a git tag
"""

from __future__ import annotations

import argparse
import subprocess
from io import BytesIO
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = "data/cache"


def cell(benchmark: str, model: str, ref: str | None) -> pd.DataFrame | None:
    rel = f"{CACHE}/{benchmark}/{model}.parquet"
    if ref:
        proc = subprocess.run(["git", "show", f"{ref}:{rel}"],
                              cwd=ROOT, capture_output=True)
        if proc.returncode:
            return None
        return pd.read_parquet(BytesIO(proc.stdout))
    path = ROOT / rel
    return pd.read_parquet(path) if path.exists() else None


def models_in(benchmark: str, ref: str | None) -> list[str]:
    if ref:
        out = subprocess.run(
            ["git", "ls-tree", "--name-only", f"{ref}:{CACHE}/{benchmark}"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    else:
        out = [p.name for p in (ROOT / CACHE / benchmark).iterdir()]
    return sorted(n[: -len(".parquet")] for n in out if n.endswith(".parquet"))


def table(ref: str | None) -> pd.DataFrame:
    benchmarks = sorted(p.name for p in (ROOT / CACHE).iterdir() if p.is_dir())
    rows = []
    for bench in benchmarks:
        names = models_in(bench, ref)
        frames = {}
        for m in names:
            f = cell(bench, m, ref)
            if f is None:
                continue
            frames[m] = f.set_index("task_id")[["extracted_answer", "is_correct"]]
        both_wrong = attractor = total = 0
        for a, b in combinations(sorted(frames), 2):
            fa, fb = frames[a], frames[b]
            shared = fa.index.intersection(fb.index)
            wa = ~fa.loc[shared, "is_correct"].astype("boolean").fillna(False)
            wb = ~fb.loc[shared, "is_correct"].astype("boolean").fillna(False)
            wrong = wa & wb
            # An unparsed answer is not agreement: two models that both failed to
            # commit have not coordinated on anything.
            aa = fa.loc[shared, "extracted_answer"]
            bb = fb.loc[shared, "extracted_answer"]
            agree = aa.notna() & bb.notna() & (aa.astype(str) == bb.astype(str))
            total += len(shared)
            both_wrong += int(wrong.sum())
            attractor += int((wrong & agree).sum())
        rows.append({"benchmark": bench, "n_pairs": len(frames) * (len(frames) - 1) // 2,
                     "n_items": total,
                     "attractor_rate": round(attractor / total, 4) if total else float("nan"),
                     "p_both_wrong": round(both_wrong / total, 4) if total else float("nan")})
    return pd.DataFrame(rows).sort_values("attractor_rate", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default=None, help="git ref to read the cache from")
    args = ap.parse_args()
    print(table(args.ref).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
