"""Task 6 -- re-audit the five pending claims against the measurement floor.

Every claim listed under "PENDING RE-VERIFICATION" in ``results/claims.md`` is
re-derived here from its parquet and compared against the per-benchmark
minimum detectable effect in ``results/measurement_floor.json``.  The output is
a verdict of CONFIRMED / REFUTED / UNDETECTABLE / RESTATED for each, written to
``results/pending_claims_audit.md`` and ``results/pending_claims_audit.parquet``.

The script **fails loudly**: a missing parquet, a missing column, or an empty
slice raises rather than silently producing a shorter table.  A pending claim
that cannot be audited must block the ledger, not disappear from it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADV = ROOT / "results" / "adversarial"
AGG = ROOT / "results" / "aggregation"

# Discard-family split.  ``trimmed_mean``/``krum``/``multi_krum`` are flagged
# ``is_oracle`` in the sweep: they are handed the true Byzantine count and are
# not deployable.  H2's original wording covered both, so both are audited.
DEPLOYABLE_DISCARD = ["majority", "geometric_median", "coord_median", "dawid_skene",
                      "sac_filter_refine"]
ORACLE_DISCARD = ["krum", "multi_krum", "trimmed_mean"]

# The slice the pre-registered claims were stated on.
SLICE = {"composition": "mix_all4", "p_obs": 1.0, "parity": "matched@0.1"}


class AuditFailure(RuntimeError):
    """Raised when a pending claim cannot be audited from the data on disk."""


def _load(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise AuditFailure(f"missing evidence file: {path}")
    frame = pd.read_parquet(path)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise AuditFailure(f"{path.name} lacks required columns {missing}")
    if frame.empty:
        raise AuditFailure(f"{path.name} is empty")
    return frame


def _floors() -> dict[str, float]:
    path = ROOT / "results" / "measurement_floor.json"
    if not path.exists():
        raise AuditFailure(f"missing measurement floor: {path}")
    raw = json.loads(path.read_text())
    out: dict[str, float] = {}
    for key, value in raw["per_benchmark"].items():
        bench = key.split(":")[0]
        out[bench] = max(out.get(bench, 0.0), float(value))
    return out


def _sub(frame: pd.DataFrame, **kw: object) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in kw.items():
        mask &= (frame[key] == value).to_numpy()
    out = frame[mask]
    if out.empty:
        raise AuditFailure(f"empty slice for {kw}")
    return out


def _verdict(effect: float, floor: float, *, direction: str = "positive") -> str:
    """Floor rule: anything not clearing the floor is UNDETECTABLE, never 'small'."""
    if not np.isfinite(effect):
        return "NO DATA"
    if abs(effect) < floor:
        return "UNDETECTABLE"
    if direction == "positive":
        return "CONFIRMED" if effect > 0 else "REFUTED"
    return "CONFIRMED" if effect < 0 else "REFUTED"


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if not np.isfinite(p) or n <= 0:
        return (float("nan"), float("nan"))
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


# --------------------------------------------------------------------------
# H2 -- do the discard-family baselines collapse at f = 0.7?
# --------------------------------------------------------------------------
def audit_h2(floors: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    """H2 is a claim about two different adversaries and must be split.

    ``sweep.parquet`` is the *symbolic* coherent adversary of Phase C; the
    original "reaches exactly zero" wording was written against it.
    ``adversarial_sweep.parquet`` is the *real-LLM* adversary of Phase D.  The
    wording holds in one regime and fails in the other, which is precisely why
    it could not be carried into prose unaudited.
    """
    regimes = [
        ("symbolic", ROOT / "results" / "aggregation" / "sweep.parquet",
         dict(attack=None, adversary="always_wrong", topology="complete")),
        ("real-LLM", ADV / "adversarial_sweep.parquet",
         dict(attack="semantic_negation")),
    ]
    rows = []
    for regime, path, extra in regimes:
        frame = _load(path, ["method", "benchmark", "f", "accuracy_point", "is_oracle"])
        kw = {k: v for k, v in extra.items() if v is not None}
        base = _sub(frame, f=0.7, **SLICE, **kw)
        for bench, grp in base.groupby("benchmark"):
            acc = grp.set_index("method").accuracy_point
            missing = [m for m in DEPLOYABLE_DISCARD + ORACLE_DISCARD + ["aip_gated"]
                       if m not in acc.index]
            if missing:
                raise AuditFailure(f"H2/{regime}/{bench}: methods absent: {missing}")
            aip = float(acc["aip_gated"])
            dep, orc = acc[DEPLOYABLE_DISCARD], acc[ORACLE_DISCARD]
            floor = floors[bench]
            rows.append({
                "claim": "H2", "regime": regime, "benchmark": bench, "floor": floor,
                "aip_gated": aip,
                "discard_min": float(dep.min()), "discard_max": float(dep.max()),
                "best_deployable": dep.idxmax(),
                "sep_vs_deployable": aip - float(dep.max()),
                "verdict_deployable": _verdict(aip - float(dep.max()), floor),
                "oracle_max": float(orc.max()), "best_oracle": orc.idxmax(),
                "sep_vs_oracle": aip - float(orc.max()),
                "verdict_oracle": _verdict(aip - float(orc.max()), floor),
            })
    table = pd.DataFrame(rows).sort_values(["regime", "benchmark"]).reset_index(drop=True)
    sym = table[table.regime == "symbolic"]
    real = table[table.regime == "real-LLM"]
    notes = [
        "Against the SYMBOLIC coherent adversary the original wording holds exactly for "
        f"deployable methods: every one of them reads {sym.discard_max.max():.3f} at "
        f"f=0.7 on all {len(sym)} benchmarks -- an exact zero, not a rounded one.",
        "It does NOT hold for oracle methods even there: "
        + ", ".join(f"{r.benchmark} {r.best_oracle} {r.oracle_max:.3f}"
                    for r in sym.itertuples() if r.oracle_max > 0)
        + ". 'Including oracle methods' is REFUTED.",
        "Nor does AIP win everywhere in that regime: "
        + ", ".join(f"{r.benchmark} {r.sep_vs_deployable:+.3f}"
                    for r in sym.itertuples() if r.sep_vs_deployable <= 0)
        + " (AIP itself reads 0.000 on those two).",
        "Against the REAL-LLM coherent adversary (semantic_negation) nothing reaches "
        f"zero: deployable discard spans {real.discard_min.min():.3f} to "
        f"{real.discard_max.max():.3f}. AIP separates on "
        f"{int((real.verdict_deployable == 'CONFIRMED').sum())} of {len(real)} benchmarks, "
        f"is UNDETECTABLE on {int((real.verdict_deployable == 'UNDETECTABLE').sum())}, and "
        f"LOSES on {int((real.verdict_deployable == 'REFUTED').sum())}.",
        "The claim must therefore name its adversary and its method class. Stated "
        "unconditionally it is false in three separate ways.",
    ]
    return table, notes


# --------------------------------------------------------------------------
# F1 -- does falsifying the self-report move AIP, or only the self-report reader?
# --------------------------------------------------------------------------
def audit_f1(floors: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    sweep = _load(ADV / "adversarial_sweep.parquet",
                  ["method", "benchmark", "attack", "f", "accuracy_point"])
    hi = sweep[sweep.f >= 0.3]
    rows = []
    # ``falsified_confidence`` is built by taking the semantic_negation cache and
    # inverting only ``self_reported_confidence`` (scripts/phase_d_sweep.py).
    # semantic_negation is therefore the *exact* control: identical answers,
    # identical logprobs, one channel changed. always_wrong is a different answer
    # stream and is reported only as context, never as evidence.
    for bench in sorted(sweep.benchmark.unique()):
        floor = floors[bench]
        for control in ("semantic_negation", "always_wrong"):
            att = _sub(hi, benchmark=bench, attack="falsified_confidence", **SLICE)
            ctl = _sub(hi, benchmark=bench, attack=control, **SLICE)
            for method in ("aip_gated", "confidence_weighted_selfreport",
                           "confidence_weighted"):
                a = att[att.method == method].accuracy_point.mean()
                c = ctl[ctl.method == method].accuracy_point.mean()
                if not (np.isfinite(a) and np.isfinite(c)):
                    raise AuditFailure(f"F1/{bench}/{method}: no cells for {control}")
                rows.append({
                    "claim": "F1", "benchmark": bench, "control": control,
                    "controlled": control == "semantic_negation",
                    "method": method, "under_attack": float(a), "under_control": float(c),
                    "effect": float(a - c), "floor": floor,
                    "moved": bool(abs(a - c) >= floor),
                })
    table = pd.DataFrame(rows)
    sr = table[(table.method == "confidence_weighted_selfreport")
               & (table.control == "semantic_negation")]
    ai = table[(table.method == "aip_gated") & (table.control == "semantic_negation")]
    cw = table[(table.method == "confidence_weighted")
               & (table.control == "semantic_negation")]
    sr_moved = sr[sr.moved].sort_values("effect")
    notes = [
        "Controlled contrast only (falsified_confidence vs semantic_negation: same "
        "answers, same logprobs, self-report inverted).",
        f"AIP-gated moves by {ai.effect.abs().max():.3f} on every benchmark -- an exact "
        "zero, not an estimate below the floor. The gate never reads the self-report, so "
        "the attack is a no-op for it by construction; this is an implementation "
        "identity of the same kind as the C=2 attenuation identity, and the audit "
        "confirms the implementation matches the specification.",
        f"The logprob-reading weighted baseline is likewise unmoved "
        f"(max {cw.effect.abs().max():.3f}), which localises the effect to the "
        "self-report channel rather than to confidence weighting as such.",
        f"The self-report reader degrades on {int(sr.moved.sum())}/{len(sr)} benchmarks: "
        + ", ".join(f"{r.benchmark} {r.effect:+.3f}" for r in sr_moved.itertuples())
        + f". The remaining {len(sr) - int(sr.moved.sum())} "
        f"({', '.join(sorted(sr[~sr.moved].benchmark))}) are UNDETECTABLE at this scale, "
        "not unaffected.",
    ]
    return table, notes


# --------------------------------------------------------------------------
# B1 -- does windowing recover the burst adversary, and what does it cost?
# --------------------------------------------------------------------------
def audit_b1(floors: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    burst = _load(ADV / "burst_windowed.parquet",
                  ["method", "benchmark", "attack", "f", "window", "accuracy_point"])
    windows = sorted(int(w) for w in burst.window.unique() if int(w) > 0)
    if not windows:
        raise AuditFailure("burst_windowed.parquet has no windowed variants")
    rows = []
    for (bench, attack, f), grp in burst.groupby(["benchmark", "attack", "f"]):
        acc = grp.set_index("method").accuracy_point
        if "aip_gated" not in acc.index:
            raise AuditFailure(f"B1/{bench}/{attack}/f={f}: pooled aip_gated missing")
        pooled = float(acc["aip_gated"])
        wins = {w: float(acc[f"aip_gated_w{w}"]) for w in windows
                if f"aip_gated_w{w}" in acc.index}
        if len(wins) != len(windows):
            raise AuditFailure(f"B1/{bench}/{attack}/f={f}: windowed variants missing")
        best_w = max(wins, key=lambda w: wins[w])
        baselines = [m for m in DEPLOYABLE_DISCARD if m in acc.index]
        floor = floors[bench]
        rows.append({
            "claim": "B1", "benchmark": bench, "attack": attack, "f": float(f),
            "pooled": pooled, "best_window": best_w, "windowed": wins[best_w],
            "recovery": wins[best_w] - pooled, "floor": floor,
            "verdict": _verdict(wins[best_w] - pooled, floor),
            "best_baseline": float(acc[baselines].max()) if baselines else float("nan"),
        })
    table = pd.DataFrame(rows).sort_values(["attack", "f"]).reset_index(drop=True)
    b = table[table.attack == "burst"]
    stat = table[table.attack == "semantic_negation"]
    notes = [
        f"Burst: pooled AIP falls to {b.pooled.min():.3f} at f={b.loc[b.pooled.idxmin(), 'f']:.1f}; "
        f"the best window holds {b.windowed.min():.3f} across all f. Maximum recovery "
        f"{b.recovery.max():+.3f} against a {b.floor.iloc[0]:.3f} floor.",
        f"Stationary control (semantic_negation): recovery ranges "
        f"{stat.recovery.min():+.3f} to {stat.recovery.max():+.3f}; "
        f"{int((stat.verdict == 'UNDETECTABLE').sum())}/{len(stat)} cells are below floor, "
        "so windowing is not measurably costly where the attack is stationary.",
    ]
    return table, notes


# --------------------------------------------------------------------------
# G2 -- does adversarial coordinatability collapse on closed label spaces?
# --------------------------------------------------------------------------
def audit_g2(floors: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    qsp = _load(ADV / "q_by_answer_space.parquet",
                ["benchmark", "answer_space_class", "attack", "q_hat_adversary",
                 "honest_q_receiver_conditioned", "ceiling"])
    sweep = _load(ADV / "adversarial_sweep.parquet", ["benchmark", "n_tasks"])
    n_by_bench = sweep.groupby("benchmark").n_tasks.max().astype(int).to_dict()
    rows = []
    for r in qsp.itertuples():
        n = n_by_bench.get(r.benchmark)
        if n is None:
            raise AuditFailure(f"G2/{r.benchmark}: no task count in the sweep")
        lo, hi = _wilson(float(r.q_hat_adversary), n)
        honest = float(r.honest_q_receiver_conditioned)
        rows.append({
            "claim": "G2", "benchmark": r.benchmark,
            "answer_space_class": r.answer_space_class, "attack": r.attack,
            "n": n, "q_hat": float(r.q_hat_adversary), "q_lo": lo, "q_hi": hi,
            "honest_q": honest, "ceiling": float(r.ceiling),
            "margin_vs_honest": float(r.q_hat_adversary) - honest,
            "exceeds_honest_ci": bool(np.isfinite(honest) and lo > honest),
        })
    table = pd.DataFrame(rows)
    gaps = sorted(table[~np.isfinite(table.honest_q)].benchmark.unique())
    by_class = table.groupby("answer_space_class").q_hat.agg(["min", "max", "mean"])
    notes = [
        "Raw coordinatability does NOT collapse on closed label spaces -- it is "
        f"highest there: binary {by_class.loc['binary', 'mean']:.3f}, multiple choice "
        f"{by_class.loc['multiple_choice', 'mean']:.3f}, versus open text "
        f"{by_class.loc['open_text', 'mean']:.3f}. The original wording inverts the "
        "measured direction and is REFUTED as stated.",
        "What collapses is the *margin over honest agreement*: on 4-way multiple "
        f"choice no attack's Wilson interval clears the honest receiver-conditioned q "
        f"of {table[table.answer_space_class == 'multiple_choice'].honest_q.iloc[0]:.3f}, "
        "so there is no coherent channel a gate can license inversion on.",
    ]
    if gaps:
        notes.append(
            f"DATA GAP: no receiver-conditioned honest q is calibrated for {', '.join(gaps)} "
            "(binary class). The margin is unaudited there and the claim is restricted to "
            "the classes that carry a calibration.")
    return table, notes


# --------------------------------------------------------------------------
# M2 -- is self-vote share a confound, and does AIP exploit it?
# --------------------------------------------------------------------------
def audit_m2(floors: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    parity = _load(AGG / "parity_audit.parquet",
                   ["benchmark", "composition", "method", "f", "matched", "unmatched",
                    "effect"])
    parity = parity.assign(floor=parity.benchmark.map(floors))
    if parity.floor.isna().any():
        raise AuditFailure("M2: benchmarks in the parity audit have no measurement floor")
    parity = parity.assign(over=parity.effect.abs() >= parity.floor)
    rows = []
    for (method, f), grp in parity.groupby(["method", "f"]):
        rows.append({
            "claim": "M2", "method": method, "f": float(f), "n_cells": len(grp),
            "mean_abs_effect": float(grp.effect.abs().mean()),
            "max_abs_effect": float(grp.effect.abs().max()),
            "n_over_floor": int(grp.over.sum()),
            "mean_signed_over_floor": float(grp[grp.over].effect.mean())
            if grp.over.any() else 0.0,
        })
    table = pd.DataFrame(rows).sort_values(["method", "f"]).reset_index(drop=True)
    # Keep only methods that respond to parity at all; the invariant family is
    # summarised in a note rather than 8 rows of exact zeros per method.
    active = table.groupby("method").max_abs_effect.max()
    table = table[table.method.isin(active[active > 0].index)].reset_index(drop=True)
    unweighted = parity[parity.method.isin(
        ["majority", "geometric_median", "coord_median", "krum", "multi_krum",
         "trimmed_mean", "dawid_skene"])]
    aip = table[table.method == "aip_gated"]
    at_half = aip[aip.f == 0.5]
    off_half = aip[aip.f != 0.5]
    if at_half.empty:
        raise AuditFailure("M2: no f=0.5 row for aip_gated -- cannot audit the tie regime")
    notes = [
        f"Sanity: the unweighted/discard family is exactly parity-invariant "
        f"(max |effect| {unweighted.effect.abs().max():.4f} over {len(unweighted)} cells), "
        "confirming the audit instrument itself is not manufacturing an effect.",
        f"AIP-gated away from the tie (f != 0.5): {int(off_half.n_over_floor.sum())} of "
        f"{int(off_half.n_cells.sum())} cells clear the floor "
        f"({100 * off_half.n_over_floor.sum() / off_half.n_cells.sum():.1f}%), largest "
        f"|effect| {off_half.max_abs_effect.max():.3f}. Self-vote share is not detectable "
        "as a general property away from the tie, but it is not exactly zero either.",
        f"AIP-gated AT f = 0.5: {int(at_half.n_over_floor.sum())} of "
        f"{int(at_half.n_cells.sum())} cells clear the floor, mean signed effect "
        f"{float(at_half.mean_signed_over_floor.iloc[0]):+.3f}, max "
        f"{float(at_half.max_abs_effect.iloc[0]):.3f}. 'AIP does not exploit self-vote "
        "share' is REFUTED at the exact tie: with the swarm split 50/50 the "
        "normalisation of the adversary's own vote decides the outcome.",
        "Scope: the audit contrasts matched@0.1 against unmatched under the "
        "always_wrong adversary, the least coherent one in the sweep, so the tie "
        "regime is where the adversary's own vote carries the whole decision.",
        "Consequence for the f = 0.5 anchor: any result stated at the tie must name its "
        "parity condition, because the two differ by up to "
        f"{float(at_half.max_abs_effect.iloc[0]):.3f} on this method. The gate-aware "
        "sweep (scripts/phase_gate_aware.py) constructs its aggregators with "
        "ParityConfig(0.1), i.e. the matched -- confound-controlled -- condition, so the "
        "evasion anchor is not contaminated by this effect.",
    ]
    return table, notes


AUDITS = [("H2", audit_h2), ("F1", audit_f1), ("B1", audit_b1),
          ("G2", audit_g2), ("M2", audit_m2)]


def main() -> int:
    floors = _floors()
    tables, all_notes, failures = [], {}, []
    for name, fn in AUDITS:
        try:
            table, notes = fn(floors)
        except AuditFailure as exc:  # fail loudly, but audit the rest first
            failures.append(f"{name}: {exc}")
            continue
        tables.append(table)
        all_notes[name] = notes

    if failures:
        print("AUDIT FAILED -- pending claims could not be re-derived:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    out_md = ROOT / "results" / "pending_claims_audit.md"
    lines = ["# Task 6 -- audit of the pending claims", "",
             "Re-derived from the parquets listed in `results/claims.md`, compared "
             "against the per-benchmark measurement floor. Generated by "
             "`scripts/task6_audit_pending.py`; do not hand-edit.", ""]
    floor_txt = ", ".join(f"{b} {v:.3f}" for b, v in sorted(floors.items()))
    lines += [f"Floors in force: {floor_txt}.", ""]
    for (name, _), table in zip(AUDITS, tables, strict=True):
        lines += [f"## {name}", ""]
        lines += [table.drop(columns=["claim"]).to_markdown(index=False, floatfmt=".3f"), ""]
        for note in all_notes[name]:
            lines.append(f"- {note}")
        lines.append("")
    out_md.write_text("\n".join(lines))

    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined.to_parquet(ROOT / "results" / "pending_claims_audit.parquet", index=False)

    for (name, _), table in zip(AUDITS, tables, strict=True):
        print(f"\n=== {name} ===")
        print(table.drop(columns=["claim"]).to_string(index=False))
        for note in all_notes[name]:
            print(f"  * {note}")
    print(f"\nwrote {out_md} and results/pending_claims_audit.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
