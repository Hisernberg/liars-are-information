#!/usr/bin/env python3
"""Phase 1 gate report: gates a-f over the whole model x benchmark grid.

Reads the cache and the per-model run reports and applies the pre-committed
decision rules mechanically, so the rules live in code rather than in whoever is
reading the table:

* **R1 CEILING** -- a cell whose honest accuracy at f=0 is >= 0.93 is flagged
  ``HEADROOM-LOW``. It stays in the cache and in appendix tables, and is excluded
  from headline figures. Headline benchmarks are math500, medqa, mmlu and arc;
  GSM8K and BoolQ carry headline claims only for the weak and mid tiers.
* **R2 AUC POWER** -- confidence AUC is interpreted only where the cell has at
  least 30 wrong answers. Below that it is reported as ``underpowered`` with the
  count, and no direction-of-effect language is attached to it. A cell can be
  underpowered *because* it is saturated, which is exactly when an AUC below 0.5
  would otherwise read as "miscalibrated" rather than "we only saw 15 errors".

Gate f is the reason both rules exist: swarm headroom (1 - accuracy), not
accuracy, decides whether a cell can carry a headline figure. At 0.02 headroom a
ten-agent swarm is almost always aggregating unanimous correctness, so every
method scores alike and the comparison is vacuous however good the accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.harness.rules import load_rules  # noqa: E402
from aip.models.registry import load_registry  # noqa: E402
from aip.tasks import roster  # noqa: E402

RULES = load_rules()
CEILING = RULES.ceiling
MIN_WRONG_FOR_AUC = RULES.min_wrong_for_auc


def cell_rows(cache: Path, registry: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark in roster.benchmarks():
        for pq in sorted((cache / benchmark).glob("*.parquet")):
            model = pq.stem
            frame = pd.read_parquet(pq)
            n = len(frame)
            correct = int(frame["is_correct"].fillna(False).sum())
            n_wrong = n - correct
            acc = correct / n if n else 0.0
            extracted = int(frame["extracted_answer"].notna().sum())
            srep = int(frame["self_reported_confidence"].notna().sum())
            try:
                tier = registry.get(model).tier or "?"
            except KeyError:
                tier = "?"
            headline = RULES.headline_eligible(benchmark, acc)
            rows.append(
                {
                    "model": model,
                    "tier": tier,
                    "benchmark": benchmark,
                    "n": n,
                    "accuracy": round(acc, 4),
                    "swarm_headroom": round(1.0 - acc, 4),
                    "n_wrong": n_wrong,
                    "extraction_rate": round(extracted / n, 4) if n else 0.0,
                    "self_report_rate": round(srep / n, 4) if n else 0.0,
                    "nonreproducing": int(frame.get("nonreproducing", pd.Series(dtype=bool)).sum()),
                    "resample_kind": (
                        frame["resample_kind"].iloc[0] if "resample_kind" in frame else "?"
                    ),
                    "headroom_flag": "HEADROOM-LOW" if acc >= CEILING else "ok",
                    "auc_power": (
                        "ok" if n_wrong >= MIN_WRONG_FOR_AUC else f"underpowered (n_wrong={n_wrong})"
                    ),
                    "headline_eligible": bool(headline),
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--registry", type=Path, default=Path("configs/models.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/phase_1_gate_report.md"))
    args = ap.parse_args()

    registry = load_registry(args.registry)
    rows = cell_rows(args.cache, registry)
    if not rows:
        print("no cached cells found", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)

    lines: list[str] = []
    add = lines.append
    add("# Phase 1 gate report\n")
    add(f"Grid: {df['model'].nunique()} models x {df['benchmark'].nunique()} benchmarks "
        f"= {len(df)} cells, {int(df['n'].sum())} cached predictions.\n")
    add(f"Rules applied mechanically: **R1** ceiling at accuracy >= {CEILING}; "
        f"**R2** AUC interpreted only at n_wrong >= {MIN_WRONG_FOR_AUC}.\n")

    add("\n## Gates a-f, per cell\n")
    add("| model | tier | benchmark | n | acc | **head** | extract | srep | resample | flag | AUC power |")
    add("|---|---|---|---:|---:|---:|---:|---:|---|---|---|")
    for r in df.sort_values(["benchmark", "model"]).to_dict("records"):
        add(
            f"| {r['model']} | {r['tier']} | {r['benchmark']} | {r['n']} "
            f"| {r['accuracy']:.2f} | **{r['swarm_headroom']:.2f}** "
            f"| {r['extraction_rate']:.0%} | {r['self_report_rate']:.0%} "
            f"| {r['resample_kind']} | {r['headroom_flag']} | {r['auc_power']} |"
        )

    add("\n## Gate f: headroom summary (R1)\n")
    low = df[df["headroom_flag"] == "HEADROOM-LOW"]
    add(f"**{len(low)} of {len(df)} cells are HEADROOM-LOW** (accuracy >= {CEILING}). "
        "Kept in cache and appendix tables; excluded from headline figures.\n")
    if len(low):
        add("| benchmark | HEADROOM-LOW cells | models |")
        add("|---|---:|---|")
        for b, g in low.groupby("benchmark"):
            add(f"| {b} | {len(g)} | {', '.join(sorted(g['model']))} |")
    add(f"\n**Headline-eligible cells: {int(df['headline_eligible'].sum())} of {len(df)}.**\n")
    add("| benchmark | headline-eligible | median headroom |")
    add("|---|---:|---:|")
    for b, g in df.groupby("benchmark"):
        add(f"| {b} | {int(g['headline_eligible'].sum())} | {g['swarm_headroom'].median():.2f} |")

    add("\n## Gate b power (R2)\n")
    weak = df[df["n_wrong"] < MIN_WRONG_FOR_AUC]
    add(f"**{len(weak)} cells are underpowered for AUC** (n_wrong < {MIN_WRONG_FOR_AUC}). "
        "No direction-of-effect language may be attached to their confidence AUC.\n")
    if len(weak):
        add("| cell | n_wrong |")
        add("|---|---:|")
        for r in weak.sort_values("n_wrong").to_dict("records"):
            add(f"| {r['model']} x {r['benchmark']} | {r['n_wrong']} |")

    add("\n## Gate a: extraction\n")
    bad = df[df["extraction_rate"] < 0.95]
    add("All cells at 100% extraction.\n" if bad.empty else
        "\n".join(f"- {r['model']} x {r['benchmark']}: {r['extraction_rate']:.0%}"
                  for r in bad.to_dict("records")))

    add("\n## Gate c: self-report parse rate (R3 -- finding, not a defect)\n")
    add("| cell | self-report rate |")
    add("|---|---:|")
    for r in df[df["self_report_rate"] < 0.99].sort_values("self_report_rate").to_dict("records"):
        add(f"| {r['model']} x {r['benchmark']} | {r['self_report_rate']:.0%} |")

    add("\n## Gate d: reproducibility\n")
    nr = df[df["nonreproducing"] > 0]
    add(f"{int(df['nonreproducing'].sum())} rows flagged `nonreproducing` across "
        f"{len(nr)} cells (recorded in configs/experiments/phase_a_cache.yaml).\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_parquet(args.out.with_suffix(".parquet"), index=False)
    print(f"wrote {args.out} and {args.out.with_suffix('.parquet')}  ({len(df)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
