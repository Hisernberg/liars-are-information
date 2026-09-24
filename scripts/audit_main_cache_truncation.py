"""Audit the main-study generation cache for truncation-manufactured answers.

Offline. Reads only `data/cache/**.parquet`; writes a report; changes no macro,
no ledger entry and no published number. The classification is:

  finish_reason == "length"  AND  no benchmark-specific commitment marker
  AND  an answer was nevertheless extracted   ->  MANUFACTURED

`finish_reason` is read from each row's `extra_json`, not inferred from a token
count against the registry budget. An earlier pass of this audit used the
budget-cap heuristic and a single `^Answer:` marker for every benchmark; that
was wrong for math500 (\\boxed) and gsm8k (####), and it is superseded here.
"""
from __future__ import annotations

import glob
import json
import re

import pandas as pd

#: Each benchmark asks for its answer in its own format, so "did the model
#: actually commit to an answer" is a different test per benchmark.
COMMITMENT = {
    "math500": lambda t: "\\boxed" in t,
    "gsm8k": lambda t: "####" in t,
}
MC_MARKER = re.compile(r"(?im)^\s*answer\s*:")


def committed(benchmark: str, text: str) -> bool:
    marker = COMMITMENT.get(benchmark)
    return marker(text) if marker else MC_MARKER.search(text) is not None


def load() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob("data/cache/*/*.parquet")):
        frame = pd.read_parquet(path)
        frame["finish_reason"] = frame["extra_json"].map(
            lambda s: json.loads(s).get("finish_reason")
        )
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["committed"] = [committed(b, t) for b, t in zip(df.benchmark, df.raw_completion)]
    df["manufactured"] = (
        (df.finish_reason == "length") & ~df.committed & df.extracted_answer.notna()
    )
    return df


def cells(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["benchmark", "model"]).agg(
        n=("manufactured", "size"),
        truncated=("finish_reason", lambda s: int((s == "length").sum())),
        manufactured=("manufactured", "sum"),
        accuracy=("is_correct", "mean"),
    )
    out["trunc_rate"] = (out.truncated / out.n).round(3)
    out["manuf_rate"] = (out.manufactured / out.n).round(3)
    correct = df[df.manufactured].groupby(["benchmark", "model"]).is_correct.sum()
    out["manuf_correct"] = correct.reindex(out.index).fillna(0).astype(int)
    return out


if __name__ == "__main__":
    df = load()
    print(cells(df).to_string())
