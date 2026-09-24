#!/usr/bin/env python3
"""Verify MX Phase 1 against the four gates in `docs/mx_prereg.md` §8.

Separate from the run script on purpose. A run that reports its own success is
the failure mode the mitigation probes in the main study were written to catch,
so the checks live in a script that reads the artifacts back from disk and does
not share a code path with whatever produced them.

Gates:
  1. per-stage outputs logged
  2. gate decisions logged on every AIP stage-row
  3. liar firings logged -- zero here, by construction, but the field must exist
  4. determinism: the same seed reproduces identical generations
  5. truncation audit: per model x stage, how much was truncated and how much
     was discarded for being truncated without committing to an answer
  6. external consistency: each model's stage-1 accuracy against its own cached
     main-study MedQA accuracy, PER MODEL -- the only check that compares this
     harness against independently produced historical data, and the one that
     catches task-list mixups, prompt drift, extraction drift and wrong weights,
     none of which any internal gate can see. A pooled average would hide a
     swapped slot, so the comparison is never pooled.

**Parse rate is retired as a health metric.** It counts extractor outputs, not
faithful answers. Phase 1 run 3 had a 100% parse rate while 24% of generations
carried answers the model never gave, manufactured by the extractor's fallback
chain from completions truncated mid-reasoning. Parse rate is still reported --
it is diagnostic -- but it no longer certifies anything on its own.

Also evaluates the §5 negative-control invariant early, at N=20: with no liar
firing, P2 must equal P1 exactly and P3 must equal P4 exactly, and no honest
channel may be inverted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aip.pipeline.stages import STAGE_ORDER  # noqa: E402

PASS, FAIL = "PASS", "FAIL"


def check(name: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    return (name, ok, detail)


def external_consistency(cache: pd.DataFrame, floor: float = 0.100,
                         bench_dir: Path = Path("data/cache/medqa")
                         ) -> list[tuple[str, bool, str]]:
    """Gate 6. Stage-1 accuracy per model against the cached main-study run.

    KNOWN CONFOUND, stated rather than buried: the main-study cache is greedy
    (registry temperature 0.0) and MX stage-1 replicas are sampled at T=0.7, so
    a difference within a few points is expected from temperature alone. The
    floor tolerance absorbs that. What the gate is actually hunting is a model
    landing nowhere near its own known accuracy, which is a wiring fault -- wrong
    weights, wrong task list, drifted prompt or drifted extractor -- until proven
    otherwise.
    """
    out: list[tuple[str, bool, str]] = []
    s1 = cache[(cache.stage == "solver") & cache.answer.notna()]
    if not len(s1):
        return [("G6 external consistency", False, "no stage-1 generations")]
    for model in sorted(s1.model.unique()):
        ref_path = bench_dir / f"{model}.parquet"
        if not ref_path.exists():
            out.append(("G6 " + model, False, f"no main-study cache at {ref_path}"))
            continue
        ref = pd.read_parquet(ref_path)
        sub = s1[s1.model == model]
        # SAME TASK IDS, always. The first version of this gate compared MX's
        # 20-task subsample against the cached 200-task population and flagged
        # two models as wiring faults. They were not: at n=20 the binomial
        # standard error on accuracy is sqrt(0.25/20) ~ 0.11, so a 1-sigma
        # sampling deviation exceeded the 0.100 floor by construction. Restricted
        # to the same ids, ministral3_14b reproduced its cached accuracy exactly.
        # A gate that fires on its own sampling noise trains you to ignore it.
        mx_ids = set(sub.task_id)
        ref_same = ref[ref.task_id.isin(mx_ids)]
        ref_acc = float(ref_same.is_correct.mean()) if len(ref_same) else float("nan")
        gold = dict(zip(ref.task_id, ref.gold_answer.astype(str), strict=False))
        hit = [a == gold.get(t) for a, t in zip(sub.answer, sub.task_id, strict=False)
               if t in gold]
        mx_acc = float(sum(hit) / len(hit)) if hit else float("nan")
        delta = mx_acc - ref_acc
        ok = abs(delta) <= floor
        out.append((f"G6 {model} stage-1 vs main-study MedQA", ok,
                    f"MX {mx_acc:.3f} (n={len(hit)}) vs cached-same-ids {ref_acc:.3f} "
                    f"(n={len(ref_same)} tasks), delta {delta:+.3f}, floor {floor:.3f}"
                    + ("" if ok else "  <-- WIRING FAULT until proven otherwise")))
    return out


def raw_sample(cache: pd.DataFrame, per_cell: int = 5,
               seed: int = 20250825) -> str:
    """Standing ritual: raw completions, verbatim and untruncated.

    The truncation defect was found by reading one raw completion. That channel
    stays open permanently -- no summary statistic replaces a human looking at
    what the model actually emitted.
    """
    rng = np.random.default_rng(seed)
    blocks = [f"RAW SAMPLE -- {per_cell} per model x stage, seed {seed}"]
    for (model, st), g in cache.groupby(["model", "stage"]):
        take = g.iloc[rng.choice(len(g), size=min(per_cell, len(g)), replace=False)]
        blocks.append(f"\n{'=' * 78}\n{model} / {st}\n{'=' * 78}")
        for r in take.itertuples():
            blocks.append(
                f"\n-- key {r.key} | finish={r.finish_reason} "
                f"| tokens={r.n_tokens} | truncated={r.truncated} "
                f"| committed={r.committed_answer}\n"
                f"-- EXTRACTED: {r.answer!r}"
                + (f"  [DISCARDED: {r.answer_discarded}]"
                   if r.answer_discarded else "")
                + f"\n{r.raw}\n")
    return "\n".join(blocks)


def verify(frame: pd.DataFrame, cache: pd.DataFrame,
           det: pd.DataFrame | None = None) -> list[tuple[str, bool, str]]:
    out = []

    # -- gate 1: per-stage outputs ------------------------------------------
    stages_seen = set(frame.stage.unique())
    ok = stages_seen == set(STAGE_ORDER)
    out.append(check("G1 per-stage outputs logged", ok,
                     f"stages present: {sorted(stages_seen)}"))

    per_arm = frame.groupby(["arm", "stage"]).size().unstack(fill_value=0)
    complete = bool((per_arm.to_numpy() > 0).all())
    out.append(check("G1b every arm reached every stage", complete,
                     f"min rows in any (arm, stage) cell: {int(per_arm.to_numpy().min())}"))

    # -- gate 2: gate decisions ---------------------------------------------
    aip = frame[frame.rule == "aip"]
    logged = int((aip.decisions.map(len) > 0).sum())
    ok = len(aip) > 0 and logged == len(aip)
    out.append(check("G2 gate decisions logged on every AIP row", ok,
                     f"{logged}/{len(aip)} AIP stage-rows carry decisions"))

    if len(aip):
        vocab = {d for row in aip.decisions for d in row}
        ok = vocab <= {"invert", "trust", "discard"}
        out.append(check("G2b decision vocabulary is the gate's own", ok,
                         f"observed: {sorted(vocab)}"))

    # -- gate 3: liar telemetry ---------------------------------------------
    has_fields = {"liar_slot", "liar_variant", "liar_fired"} <= set(frame.columns)
    fired = int(frame.liar_fired.sum()) if has_fields else -1
    out.append(check("G3 liar telemetry present and reads zero (no-fault run)",
                     has_fields and fired == 0,
                     f"fields present: {has_fields}, firings: {fired}"))

    # -- negative control (pre-reg §5) --------------------------------------
    for a, b in (("P1", "P2"), ("P4", "P3")):
        left = frame[frame.arm == a].set_index(["stage", "task_id"]).output
        right = frame[frame.arm == b].set_index(["stage", "task_id"]).output
        common = left.index.intersection(right.index)
        diff = [i for i in common if left[i] != right[i]]
        out.append(check(f"NC {b} outputs equal {a} exactly (no liar)",
                        len(diff) == 0,
                        f"{len(diff)}/{len(common)} differ"
                        + (f"; first: {diff[:3]}" if diff else "")))

    inv = int(frame.honest_inverts.sum())
    out.append(check("NC zero INVERT on honest channels", inv == 0,
                     f"false inversions: {inv}"))

    # -- extraction ----------------------------------------------------------
    miss = int(frame.output.isna().sum())
    out.append(check("extraction produced an answer on every stage-row", miss == 0,
                     f"{miss}/{len(frame)} rows have no output"))

    if cache is not None and len(cache):
        nulls = int(cache.answer.isna().sum())
        out.append(check("every cached generation extracted an answer", nulls == 0,
                         f"{nulls}/{len(cache)} generations unparsed"))

    # -- gate 5: truncation audit -------------------------------------------
    if cache is not None and len(cache) and "truncated" in cache.columns:
        audit = (cache.assign(
                     discarded=cache.answer_discarded.notna(),
                     parsed=cache.answer.notna())
                 .groupby(["model", "stage"])
                 .agg(generations=("key", "size"),
                      truncated=("truncated", "sum"),
                      discarded=("discarded", "sum"),
                      parsed=("parsed", "sum")))
        audit["trunc_rate"] = audit.truncated / audit.generations
        audit["discard_rate"] = audit.discarded / audit.generations
        audit["parse_rate"] = audit.parsed / audit.generations
        out.append(("TRUNCATION AUDIT (reported, see table below)", True,
                    f"{int(audit.truncated.sum())} truncated, "
                    f"{int(audit.discarded.sum())} discarded, of "
                    f"{int(audit.generations.sum())} generations"))
        hot = audit[audit.discard_rate > 0.01]
        out.append(check("G5 no model x stage cell discards > 1%",
                         len(hot) == 0,
                         "all cells <= 1%" if len(hot) == 0
                         else "OVER 1%: " + ", ".join(
                             f"{m}/{st} {r.discard_rate:.1%}"
                             for (m, st), r in hot.iterrows())))
        out.append(("__AUDIT_TABLE__", True, audit.round(4).to_string()))
    else:
        out.append(check("G5 truncation audit", False,
                         "cache lacks truncation columns -- regenerate with the "
                         "current record_generation"))

    # -- gate 4: determinism -------------------------------------------------
    if det is None or not len(det):
        out.append(check("G4 determinism: same seed reproduces outputs", False,
                         "NO determinism sample -- run with --determinism-sample > 0"))
    else:
        ans = int((~det.answer_match).sum())
        txt = int((~det.text_match).sum())
        out.append(check("G4 determinism: extracted answers identical", ans == 0,
                         f"{ans}/{len(det)} differ"))
        out.append(check("G4b determinism: raw completions byte-identical", txt == 0,
                         f"{txt}/{len(det)} differ"
                         + ("" if txt == 0 else " -- answers may still match; "
                            "a text-only difference still breaks the cache-once "
                            "premise and must be explained")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", type=Path,
                    default=Path("results/mx/phase1_stages.parquet"))
    ap.add_argument("--cache", type=Path,
                    default=Path("data/mx_cache/phase1.parquet"))
    ap.add_argument("--raw-sample", type=int, default=5,
                    help="raw completions to print per model x stage (0 disables)")
    ap.add_argument("--raw-seed", type=int, default=20250825)
    ap.add_argument("--determinism", type=Path,
                    default=Path("results/mx/phase1_determinism.parquet"))
    args = ap.parse_args()
    if not args.stages.exists():
        raise SystemExit(f"missing {args.stages}; run scripts/mx_phase1.py first")
    frame = pd.read_parquet(args.stages)
    cache = pd.read_parquet(args.cache) if args.cache.exists() else None
    det = pd.read_parquet(args.determinism) if args.determinism.exists() else None

    results = verify(frame, cache, det)
    if cache is not None and len(cache):
        results.extend(external_consistency(cache))
    print("=" * 88)
    print("MX PHASE 1 VERIFICATION")
    print("=" * 88)
    worst = 0
    for name, ok, detail in results:
        if name == "__AUDIT_TABLE__":
            print("\n  per model x stage:")
            for line in detail.splitlines():
                print(f"    {line}")
            print()
            continue
        print(f"  [{PASS if ok else FAIL}] {name:<52} {detail}")
        worst |= 0 if ok else 1
    print()
    if cache is not None and len(cache) and args.raw_sample:
        print(raw_sample(cache, per_cell=args.raw_sample, seed=args.raw_seed))
    print("PHASE 1 " + ("PASSES -- Phase 2 may proceed" if not worst
                        else "FAILS -- stop, trace, fix before Phase 2"))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
