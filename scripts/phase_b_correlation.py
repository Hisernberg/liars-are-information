#!/usr/bin/env python3
"""Phase B: pairwise error correlation and blind-estimator validation.

**Cache-only.** This script never loads a model. It reads the Phase A parquet
caches read-only and writes analysis artefacts; if it ever needs new inference,
that is a bug to report rather than an action to take.

It answers the headline question -- whether a homogeneous swarm is epistemically
one agent -- and validates the attenuation-law estimator that lets an agent
measure pair quality from broadcasts alone, with no labels.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.aggregation.correlation import (  # noqa: E402
    answer_agreement,
    blind_marginal_error_rates,
    blind_product_estimate,
    co_error_rate,
    correctness_agreement,
    effective_swarm_size,
    error_vector,
    mixed_swarm_rho_bar,
    phi_coefficient,
    true_product,
)
from aip.analysis.bootstrap import bootstrap_mean, bootstrap_statistic  # noqa: E402
from aip.analysis.figures import correlation_heatmap  # noqa: E402
from aip.analysis.provenance import stamp  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest  # noqa: E402
from aip.tasks import roster  # noqa: E402

log = configure_logging(level="INFO")

BENCHMARKS = roster.benchmarks()  # from configs/task_lists.json
N_SWARM = 10


def load_slot(benchmark: str, model: str, root: Path) -> pd.DataFrame:
    """Read one cache cell, indexed by task_id. Read-only, no transforms."""
    path = root / benchmark / f"{model}.parquet"
    frame = pd.read_parquet(path)
    return frame.set_index("task_id").sort_index()


def _answers(frame: pd.DataFrame) -> np.ndarray:
    """Extracted answers as a plain object array, with None for missing."""
    values = [None if pd.isna(v) else str(v) for v in frame["extracted_answer"].tolist()]
    return np.array(values, dtype=object)


def aligned(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = a.index.intersection(b.index)
    return a.loc[shared], b.loc[shared]


def pair_row(
    benchmark: str,
    name_i: str,
    name_j: str,
    slot_i: int,
    slot_j: int,
    kind: str,
    fi: pd.DataFrame,
    fj: pd.DataFrame,
    flagged: set[str],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Every Phase B quantity for one pair, with bootstrap CIs over tasks."""
    fi, fj = aligned(fi, fj)
    correct_i = fi["is_correct"].to_numpy(dtype=int)
    correct_j = fj["is_correct"].to_numpy(dtype=int)
    err_i, err_j = error_vector(correct_i), error_vector(correct_j)
    # pandas' nullable string dtype yields pd.NA, which poisons comparisons;
    # convert to plain Python objects with None for "no answer extracted".
    ans_i = _answers(fi)
    ans_j = _answers(fj)
    n = len(fi)

    def _phi(idx: np.ndarray) -> float:
        return phi_coefficient(err_i[idx], err_j[idx])

    def _co(idx: np.ndarray) -> float:
        return co_error_rate(err_i[idx], err_j[idx])

    def _ans(idx: np.ndarray) -> float:
        return answer_agreement(ans_i[idx], ans_j[idx])

    def _corr_agree(idx: np.ndarray) -> float:
        return correctness_agreement(correct_i[idx], correct_j[idx])

    def _blind(idx: np.ndarray) -> float:
        # ANSWER agreement, not correctness agreement. A blind estimator may only
        # use what an agent can actually observe on the wire; correctness
        # agreement requires knowing who was right, which is the label the
        # estimator is supposed to do without. The two coincide only for a
        # genuinely binary answer space, which is why the error hid on GSM8K.
        return blind_product_estimate(answer_agreement(ans_i[idx], ans_j[idx]))

    def _truth(idx: np.ndarray) -> float:
        return true_product(float(np.mean(err_i[idx])), float(np.mean(err_j[idx])))

    def _abs_err(idx: np.ndarray) -> float:
        return abs(_blind(idx) - _truth(idx))

    ci_phi = bootstrap_statistic(_phi, n, n_resamples, seed=seed)
    ci_co = bootstrap_statistic(_co, n, n_resamples, seed=seed + 1)
    ci_ans = bootstrap_statistic(_ans, n, n_resamples, seed=seed + 2)
    ci_ca = bootstrap_statistic(_corr_agree, n, n_resamples, seed=seed + 3)
    ci_blind = bootstrap_statistic(_blind, n, n_resamples, seed=seed + 4)
    ci_truth = bootstrap_statistic(_truth, n, n_resamples, seed=seed + 5)
    ci_abs = bootstrap_statistic(_abs_err, n, n_resamples, seed=seed + 6)

    row: dict[str, Any] = {
        "benchmark": benchmark,
        "model_i": name_i,
        "model_j": name_j,
        "slot_i": slot_i,
        "slot_j": slot_j,
        "kind": kind,
        "n_tasks": n,
        "error_rate_i": float(np.mean(err_i)),
        "error_rate_j": float(np.mean(err_j)),
        "accuracy_gap": abs(float(np.mean(correct_i)) - float(np.mean(correct_j))),
        # Known non-reproducing ids stay in the analysis, flagged not dropped.
        "n_flagged_nonreproducing": int(sum(1 for t in fi.index if t in flagged)),
    }
    row.update(ci_phi.as_dict("phi_"))
    row.update(ci_co.as_dict("co_error_"))
    row.update(ci_ans.as_dict("answer_agreement_"))
    row.update(ci_ca.as_dict("correctness_agreement_"))
    row.update(ci_blind.as_dict("blind_product_"))
    row.update(ci_truth.as_dict("true_product_"))
    row.update(ci_abs.as_dict("blind_abs_error_"))
    return row


def build_rows(
    cache: Path, cache_t07: Path, models: list[str], flagged: set[str], n_resamples: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # 1. within-model: greedy (slot 0) vs temperature 0.7 (slot 1), GSM8K only
    for model in models:
        fi = load_slot("gsm8k", model, cache)
        fj = load_slot("gsm8k", model, cache_t07)
        rows.append(
            pair_row(
                "gsm8k", model, model, 0, 1, "within_model", fi, fj, flagged, n_resamples, seed
            )
        )

    # 2. cross-model: greedy caches, every unordered pair, every benchmark
    for benchmark in BENCHMARKS:
        frames = {m: load_slot(benchmark, m, cache) for m in models}
        for a, b in itertools.combinations(models, 2):
            rows.append(
                pair_row(
                    benchmark,
                    a,
                    b,
                    0,
                    0,
                    "cross_model",
                    frames[a],
                    frames[b],
                    flagged,
                    n_resamples,
                    seed,
                )
            )
    return pd.DataFrame(rows)


def blind_marginals(cache: Path, models: list[str]) -> pd.DataFrame:
    """Blind marginal error rate per model per benchmark, versus the true rate."""
    out = []
    for benchmark in BENCHMARKS:
        frames = {m: load_slot(benchmark, m, cache) for m in models}
        shared = None
        for f in frames.values():
            shared = f.index if shared is None else shared.intersection(f.index)
        correct = {m: frames[m].loc[shared, "is_correct"].to_numpy(dtype=int) for m in models}
        answers = {m: _answers(frames[m].loc[shared]) for m in models}
        # Answer agreement only: the marginal recovery is a blind estimator too.
        agreement = {
            (a, b): answer_agreement(answers[a], answers[b])
            for a, b in itertools.combinations(models, 2)
        }
        blind = blind_marginal_error_rates(agreement, models)
        for m in models:
            true_e = float(1.0 - np.mean(correct[m]))
            out.append(
                {
                    "benchmark": benchmark,
                    "model": m,
                    "true_error_rate": true_e,
                    "blind_error_rate": blind[m],
                    "abs_error": abs(blind[m] - true_e) if np.isfinite(blind[m]) else np.nan,
                    "n_tasks": len(shared),
                }
            )
    return pd.DataFrame(out)


def neff_table(pairs: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """N_eff for homogeneous swarms and for even two-model mixes."""
    within = {r.model_i: r.phi_point for r in pairs[pairs.kind == "within_model"].itertuples()}
    rows = []
    for m in models:
        rho = within.get(m, np.nan)
        rows.append(
            {
                "composition": f"homogeneous: {m}",
                "kind": "homogeneous",
                "benchmark": "gsm8k",
                "rho_bar": rho,
                "n_eff": effective_swarm_size(N_SWARM, rho),
                "basis": "within-model rho (greedy vs T=0.7)",
            }
        )
    for benchmark in BENCHMARKS:
        cross = pairs[(pairs.kind == "cross_model") & (pairs.benchmark == benchmark)]
        for r in cross.itertuples():
            if benchmark == "gsm8k":
                rho_bar = mixed_swarm_rho_bar(
                    N_SWARM,
                    within.get(r.model_i, np.nan),
                    within.get(r.model_j, np.nan),
                    r.phi_point,
                )
                basis = "5+5 mix: 20 same-model pairs + 25 cross-model pairs"
            else:
                rho_bar = r.phi_point
                basis = "cross-model rho only (within-model unmeasured; upper bound on N_eff)"
            rows.append(
                {
                    "composition": f"mixed: {r.model_i} + {r.model_j}",
                    "kind": "heterogeneous",
                    "benchmark": benchmark,
                    "rho_bar": rho_bar,
                    "n_eff": effective_swarm_size(N_SWARM, rho_bar),
                    "basis": basis,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--cache-t07", type=Path, default=Path("data/cache_t07"))
    parser.add_argument("--out", type=Path, default=Path("results/correlation"))
    parser.add_argument("--figure", type=Path, default=Path("paper/figures/correlation_heatmap.pdf"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20250825)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/phase_a_cache.yaml")
    )
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    flagged = set(cfg.get("reproducibility", {}).get("known_nonreproducing", []))
    # The frozen roster, not whatever happens to be in the cache directory.
    # The weak-tier arm was cached alongside it and has no T=0.7 resample, so
    # globbing the directory made this phase crash on a missing file -- and the
    # weak arm is reported separately and never pooled with the roster anyway.
    registry = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    frozen = sorted(
        name for name, entry in registry["models"].items()
        if entry.get("arm") == "frozen" and entry.get("enabled", True)
    )
    present = {p.stem for p in (args.cache / "gsm8k").glob("*.parquet")}
    missing = [m for m in frozen if m not in present]
    if missing:
        raise SystemExit(f"frozen-roster models missing from {args.cache}: {missing}")
    models = frozen
    log.info("phase_b.start", models=models, resamples=args.resamples, flagged=len(flagged))

    pairs = build_rows(args.cache, args.cache_t07, models, flagged, args.resamples, args.seed)
    marginals = blind_marginals(args.cache, models)
    neff = neff_table(pairs, models)

    args.out.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.out / "pairwise_estimates.parquet", index=False)
    marginals.to_parquet(args.out / "blind_marginals.parquet", index=False)
    neff.to_parquet(args.out / "effective_swarm_size.parquet", index=False)

    # -- heatmap: phi matrices, diagonal = within-model where measured --------
    panels = []
    for benchmark in BENCHMARKS:
        mat = np.full((len(models), len(models)), np.nan)
        for i, a in enumerate(models):
            for j, b in enumerate(models):
                if i == j:
                    w = pairs[(pairs.kind == "within_model") & (pairs.model_i == a)]
                    if benchmark == "gsm8k" and len(w):
                        mat[i, j] = float(w.iloc[0]["phi_point"])
                    continue
                sel = pairs[
                    (pairs.kind == "cross_model")
                    & (pairs.benchmark == benchmark)
                    & (
                        ((pairs.model_i == a) & (pairs.model_j == b))
                        | ((pairs.model_i == b) & (pairs.model_j == a))
                    )
                ]
                if len(sel):
                    mat[i, j] = float(sel.iloc[0]["phi_point"])
        panels.append((benchmark, models, mat))
    correlation_heatmap(panels, args.figure)
    stamp(args.figure, script="scripts/phase_b_correlation.py",
          sources=[args.out / "pairwise_estimates.parquet"])

    manifest = Manifest.create(
        phase="phase_b",
        config={
            "cache": str(args.cache),
            "cache_t07": str(args.cache_t07),
            "resamples": args.resamples,
            "models": models,
        },
        seeds={"bootstrap": args.seed},
        notes={
            "inference_run": False,
            "caches_read_only": True,
            "nan_policy": [
                "phi4_mini_reasoning x mmlu: 30/100 self-reported confidences are NaN "
                "(the model reasons about the question and never commits to an integer). "
                "Phase B does not use self-reports, so no rows are affected. Missing-data "
                "handling is a Phase C decision.",
                "llama32_3b x mmlu: 1/100 answer extraction failed (generation hit the token "
                "cap mid-reevaluation). It is scored incorrect, kept in the analysis, and "
                "counts as disagreement with every other agent.",
                "phi is NaN when either error vector is constant; such pairs are reported as "
                "undefined rather than as zero correlation.",
            ],
            "known_nonreproducing_task_ids": sorted(flagged),
            "known_nonreproducing_policy": "retained in the analysis and flagged per pair, never dropped",
        },
    )
    manifest.finish(status="ok", n_pairs=len(pairs)).write(
        Path("results/manifests") / f"{manifest.run_id}.json"
    )

    print_report(pairs, marginals, neff, models, args)
    return 0


def _fmt_ci(row: pd.Series, prefix: str, decimals: int = 3) -> str:
    p, lo, hi = row[f"{prefix}point"], row[f"{prefix}ci_low"], row[f"{prefix}ci_high"]
    if not np.isfinite(p):
        return "undefined"
    return f"{p:+.{decimals}f} [{lo:+.{decimals}f}, {hi:+.{decimals}f}]"


def print_report(
    pairs: pd.DataFrame, marginals: pd.DataFrame, neff: pd.DataFrame, models: list[str], args
) -> None:
    bar = "=" * 104
    print(f"\n{bar}\nPHASE B — ERROR CORRELATION\n{bar}")
    print(f"bootstrap: {args.resamples} resamples over tasks, seed {args.seed}; 95% percentile CIs")

    print("\n1. WITHIN-MODEL  (GSM8K: greedy sample_index=0 vs T=0.7 sample_index=1)")
    print(f"{'model':<22}{'n':>4}  {'phi (error corr)':<26}{'co-error':<24}{'answer agreement'}")
    w = pairs[pairs.kind == "within_model"]
    for r in w.itertuples():
        s = pairs.loc[r.Index]
        print(
            f"{r.model_i:<22}{r.n_tasks:>4}  {_fmt_ci(s, 'phi_'):<26}"
            f"{_fmt_ci(s, 'co_error_'):<24}{_fmt_ci(s, 'answer_agreement_')}"
        )

    print("\n2. CROSS-MODEL  (greedy caches)")
    for benchmark in BENCHMARKS:
        sub = pairs[(pairs.kind == "cross_model") & (pairs.benchmark == benchmark)]
        print(f"\n  {benchmark}")
        print(f"  {'pair':<42}{'phi':<26}{'co-error':<24}{'answer agreement'}")
        for r in sub.itertuples():
            s = pairs.loc[r.Index]
            label = f"{r.model_i} + {r.model_j}"
            print(
                f"  {label:<42}{_fmt_ci(s, 'phi_'):<26}"
                f"{_fmt_ci(s, 'co_error_'):<24}{_fmt_ci(s, 'answer_agreement_')}"
            )

    print("\n3. BLIND ESTIMATOR VALIDATION  (attenuation law, no gold labels)")
    print(
        f"  {'benchmark':<12}{'pairs':>6}{'MAE  [95% CI]':>34}{'mean blind':>13}{'mean truth':>12}"
    )
    cross = pairs[pairs.kind == "cross_model"]
    for benchmark in BENCHMARKS:
        sub = cross[cross.benchmark == benchmark]
        ci = bootstrap_mean(sub["blind_abs_error_point"].to_numpy(), args.resamples, seed=args.seed)
        print(
            f"  {benchmark:<12}{len(sub):>6}"
            f"{f'{ci.point:.4f}  [{ci.low:.4f}, {ci.high:.4f}]':>34}"
            f"{sub['blind_product_point'].mean():>13.4f}{sub['true_product_point'].mean():>12.4f}"
        )
    ci_all = bootstrap_mean(
        cross["blind_abs_error_point"].to_numpy(), args.resamples, seed=args.seed
    )
    print(
        f"  {'OVERALL':<12}{len(cross):>6}"
        f"{f'{ci_all.point:.4f}  [{ci_all.low:.4f}, {ci_all.high:.4f}]':>34}"
        f"{cross['blind_product_point'].mean():>13.4f}{cross['true_product_point'].mean():>12.4f}"
    )

    print("\n   induced blind marginal error rate vs true error rate")
    print(f"   {'benchmark':<10}{'model':<22}{'true e':>9}{'blind e':>10}{'abs err':>10}")
    for r in marginals.itertuples():
        blind = "undefined" if not np.isfinite(r.blind_error_rate) else f"{r.blind_error_rate:.3f}"
        abserr = "--" if not np.isfinite(r.abs_error) else f"{r.abs_error:.3f}"
        print(f"   {r.benchmark:<10}{r.model:<22}{r.true_error_rate:>9.3f}{blind:>10}{abserr:>10}")
    mae_marg = marginals["abs_error"].mean()
    print(f"   mean absolute error of blind marginal recovery: {mae_marg:.4f}")

    print(f"\n4. EFFECTIVE SWARM SIZE  (N={N_SWARM})")
    print(f"  {'composition':<46}{'benchmark':<10}{'rho_bar':>9}{'N_eff':>8}")
    for r in neff.itertuples():
        rho = "undefined" if not np.isfinite(r.rho_bar) else f"{r.rho_bar:+.3f}"
        ne = "--" if not np.isfinite(r.n_eff) else f"{r.n_eff:.2f}"
        print(f"  {r.composition:<46}{r.benchmark:<10}{rho:>9}{ne:>8}")

    print(f"\nwrote {args.out}/*.parquet and {args.figure}")


if __name__ == "__main__":
    raise SystemExit(main())
