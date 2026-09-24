#!/usr/bin/env python3
"""Before/after audit of a repaired cache: manufactured rows, not pinned rows.

The first repaired cell showed why the distinction matters. phi4 on GSM8K went
from 73 cap-pinned rows to 60 -- which reads like a failure -- but 41 of those
60 had emitted `#### <answer>` and then carried on rambling past it. They are
answered rows that merely did not stop. Manufactured rows in that cell fell from
31 to 19, and cell accuracy rose 0.918 -> 0.930.

So "still cap-pinned" is a generation-hygiene statistic, and "still
manufactured" is the defect. Both are reported; only the second is the thing
X3 exists to remove.

    python scripts/x3_reaudit.py --before pre-x3-cache
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: Each benchmark asks for its answer in its own format, so "did the model
#: actually commit" is a different test per benchmark.
COMMITMENT = {"math500": lambda t: "\\boxed" in t, "gsm8k": lambda t: "####" in t}
MC_MARKER = re.compile(r"(?im)^\s*answer\s*:")


def committed(benchmark: str, text: str) -> bool:
    marker = COMMITMENT.get(benchmark)
    return marker(text) if marker else MC_MARKER.search(text) is not None


def annotate(frame: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["finish_reason"] = frame["extra_json"].map(
        lambda s: json.loads(s).get("finish_reason") if s else None)
    frame["committed"] = [committed(benchmark, t) for t in frame.raw_completion]
    frame["manufactured"] = (
        (frame.finish_reason == "length") & ~frame.committed & frame.extracted_answer.notna())
    return frame


def read(ref: str | None, rel: str) -> pd.DataFrame | None:
    if ref:
        proc = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True)
        return None if proc.returncode else pd.read_parquet(BytesIO(proc.stdout))
    path = ROOT / rel
    return pd.read_parquet(path) if path.exists() else None


def summarise(root: str, ref: str | None) -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / root).glob("*/*.parquet")):
        bench, model = path.parent.name, path.stem
        rel = f"{root}/{bench}/{model}.parquet"
        frame = read(ref, rel)
        if frame is None:
            continue
        frame = annotate(frame, bench)
        rows.append({"benchmark": bench, "model": model, "n": len(frame),
                     "pinned": int((frame.finish_reason == "length").sum()),
                     "manufactured": int(frame.manufactured.sum()),
                     "accuracy": round(float(frame.is_correct.mean()), 4)})
    return pd.DataFrame(rows).set_index(["benchmark", "model"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/cache")
    ap.add_argument("--before", default="pre-x3-cache", help="git ref for the pre-repair cache")
    ap.add_argument("--out", type=Path, default=Path("results/x3_reaudit.md"))
    args = ap.parse_args()

    before = summarise(args.root, args.before)
    after = summarise(args.root, None)
    joined = before.join(after, lsuffix="_before", rsuffix="_after")
    joined["manuf_delta"] = joined.manufactured_after - joined.manufactured_before
    joined["acc_delta"] = (joined.accuracy_after - joined.accuracy_before).round(4)

    totals = {
        "rows": int(joined.n_after.sum()),
        "pinned before": int(joined.pinned_before.sum()),
        "pinned after": int(joined.pinned_after.sum()),
        "manufactured before": int(joined.manufactured_before.sum()),
        "manufactured after": int(joined.manufactured_after.sum()),
    }
    cols = ["n_after", "pinned_before", "pinned_after", "manufactured_before",
            "manufactured_after", "manuf_delta", "accuracy_before", "accuracy_after",
            "acc_delta"]
    lines = [f"# X3 re-audit: `{args.root}`", "",
             f"`{args.before}` -> working tree.", "",
             "| | count |", "|---|---|",
             *[f"| {k} | {v} |" for k, v in totals.items()], "",
             "Cap-pinned is generation hygiene; manufactured is the defect. A row that",
             "emitted its answer marker and then kept rambling past it is pinned but",
             "not manufactured.", "",
             joined[cols].sort_values("manuf_delta").to_markdown()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Machine-readable twin. paper_numbers.py reads this rather than the git
    # tag: the released artifact has to be able to re-derive the hygiene macros
    # from files in the release, not from a ref in this repository's history.
    joined.reset_index().to_parquet(args.out.with_suffix(".parquet"), index=False)
    for k, v in totals.items():
        print(f"{k:22s} {v}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
