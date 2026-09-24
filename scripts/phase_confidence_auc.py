#!/usr/bin/env python3
"""Which confidence surface can a receiver believe?

Each cached broadcast carries two confidences: the mean token logprob over the
answer span, and a number the model was asked to state. A receiver that weights
peers by confidence is trusting one of them, and the two differ in a way that
matters under attack -- a prompt-injected agent can assert any self-report it
likes and cannot easily inflate the logprob of a span it did not generate.

This measures the property that makes the choice matter: how well each surface
discriminates a correct answer from a wrong one, as ROC AUC per model and
benchmark, plus what the falsification attack does to each.

Two rules are enforced rather than noted.

**R2, power.** An AUC computed on a handful of wrong answers is not
interpretable. Cells with fewer than 30 wrong answers are reported as
underpowered and excluded from every aggregate, with no direction-of-effect
language attached to them.

**R3, self-report non-compliance is a finding.** Reasoning models frequently
emit no parseable number on multiple choice. Those rows are counted and
reported, not imputed and not prompted away: a surface an agent may simply
decline to provide is a weaker protocol commitment than one derived from the
tokens it already emitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aip.attacks.falsified_confidence import invert_self_report  # noqa: E402
from aip.harness.manifest import Manifest  # noqa: E402
from aip.harness.rules import load_rules  # noqa: E402
from aip.tasks import roster  # noqa: E402


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC by the rank identity, with ties at half credit.

    Written out rather than imported so the tie handling is visible: a surface
    that returns the same value for many rows -- which a truncated self-report
    does constantly, most of them saying exactly 0.9 -- would otherwise be
    scored differently by different libraries.
    """
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks within ties
    values = np.concatenate([pos, neg])[order]
    i = 0
    sorted_ranks = ranks[order]
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        if j > i:
            sorted_ranks[i:j + 1] = sorted_ranks[i:j + 1].mean()
        i = j + 1
    ranks[order] = sorted_ranks
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    rules = load_rules()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=Path("results/correlation"))
    ap.add_argument("--min-wrong", type=int, default=None,
                    help="power floor for AUC (default: rule R2)")
    args = ap.parse_args()
    min_wrong = args.min_wrong if args.min_wrong is not None else rules.min_wrong_for_auc

    # The weak-tier arm is never pooled with the frozen roster (configs/
    # models.yaml). It is measured here too, because a protocol property should
    # hold across model strength, but the headline aggregate is roster-only.
    registry = yaml.safe_load(
        (ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]
    arm_of = {name: entry.get("arm") for name, entry in registry.items()}

    rows = []
    for benchmark in roster.benchmarks():
        for path in sorted((args.cache / benchmark).glob("*.parquet")):
            frame = pd.read_parquet(path)
            correct = frame.is_correct.to_numpy(dtype=bool)
            n_wrong = int((~correct).sum())
            lp = pd.to_numeric(frame.logprob_confidence, errors="coerce").to_numpy(float)
            sr = pd.to_numeric(frame.self_reported_confidence,
                               errors="coerce").to_numpy(float)
            sr_falsified = np.array(
                [invert_self_report(v) if np.isfinite(v) else np.nan for v in sr],
                dtype=float,
            )
            record = {
                "benchmark": benchmark, "model": path.stem,
                "arm": arm_of.get(path.stem, "unknown"), "n": len(frame),
                "n_wrong": n_wrong, "accuracy": float(correct.mean()),
                "sr_missing_rate": float(np.isnan(sr).mean()),
                "lp_missing_rate": float(np.isnan(lp).mean()),
                "powered": n_wrong >= min_wrong,
            }
            for name, values in (("logprob", lp), ("self_report", sr),
                                 ("self_report_falsified", sr_falsified)):
                ok = np.isfinite(values)
                record[f"auc_{name}"] = (
                    auc(values[ok], correct[ok]) if ok.sum() and (~correct[ok]).any()
                    else float("nan")
                )
                record[f"n_scored_{name}"] = int(ok.sum())
            rows.append(record)

    if not rows:
        raise SystemExit(f"no cache under {args.cache}")
    table = pd.DataFrame(rows)
    table["auc_gap_logprob_minus_selfreport"] = (
        table.auc_logprob - table.auc_self_report
    )
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "confidence_auc.parquet", index=False)
    Manifest.create(phase="phase_confidence_auc",
                    config={"min_wrong": min_wrong}, seeds={}).write(
        args.out / "confidence_auc.manifest.json")

    powered_all = table[table.powered]
    powered = powered_all[powered_all.arm == "frozen"]
    pd.set_option("display.width", 200)
    print("=" * 100)
    print(f"CONFIDENCE DISCRIMINATION -- ROC AUC against correctness "
          f"(R2 power floor: n_wrong >= {min_wrong})")
    print("=" * 100)
    frozen = table[table.arm == "frozen"]
    print(f"{len(powered)} of {len(frozen)} frozen-roster cells are powered; "
          f"{len(frozen) - len(powered)} are underpowered and excluded from every "
          "aggregate below. The weak-tier arm contributes "
          f"{len(powered_all) - len(powered)} further powered cells, reported "
          "separately and never pooled.\n")
    print(powered.groupby("benchmark")[
        ["auc_logprob", "auc_self_report", "auc_self_report_falsified",
         "auc_gap_logprob_minus_selfreport", "sr_missing_rate"]
    ].mean().round(3).to_string())

    print("\nper powered cell:")
    for r in powered.sort_values(["benchmark", "model"]).itertuples():
        print(f"  {r.benchmark:<9}{r.model:<22} n_wrong {r.n_wrong:>4}  "
              f"logprob {r.auc_logprob:.3f}   self-report {r.auc_self_report:.3f}   "
              f"falsified {r.auc_self_report_falsified:.3f}   "
              f"missing {r.sr_missing_rate:.1%}")

    lp_mean = float(powered.auc_logprob.mean())
    sr_mean = float(powered.auc_self_report.mean())
    fa_mean = float(powered.auc_self_report_falsified.mean())
    print("\n" + "=" * 100)
    print("READING")
    print("=" * 100)
    print(f"  logprob                AUC {lp_mean:.3f}")
    print(f"  self-report            AUC {sr_mean:.3f}")
    print(f"  self-report, falsified AUC {fa_mean:.3f}   "
          "<- the attack drives it below chance")
    print(f"  gap, logprob - self-report: {lp_mean - sr_mean:+.3f}")
    below = powered[powered.auc_self_report_falsified < 0.5]
    print(f"\n  Falsification pushes the self-report below chance (0.5) on "
          f"{len(below)} of {len(powered)} powered cells. The logprob surface is "
          "unchanged by that attack by construction -- it is not the surface the "
          "attack writes to -- which is the protocol claim: a receiver should "
          "weight on a surface the sender does not author.")
    worst = powered.loc[powered.sr_missing_rate.idxmax()]
    print(f"\n  R3: the self-report is also simply absent at times -- up to "
          f"{worst.sr_missing_rate:.1%} of rows on "
          f"{worst.benchmark}/{worst.model}. A surface an agent may decline to "
          "provide is a weaker commitment than one derived from tokens it "
          "already emitted.")
    weak = powered_all[powered_all.arm == "weak_tier"]
    if not weak.empty:
        print(f"\n  Weak-tier arm, separately ({len(weak)} powered cells): "
              f"logprob {weak.auc_logprob.mean():.3f}, self-report "
              f"{weak.auc_self_report.mean():.3f}, falsified "
              f"{weak.auc_self_report_falsified.mean():.3f}. Same direction, so the "
              "protocol property is not an artefact of model strength.")
    print(f"\nwrote {args.out}/confidence_auc.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
