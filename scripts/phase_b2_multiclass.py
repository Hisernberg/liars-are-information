#!/usr/bin/env python3
"""Phase B2: validate the multiclass attenuation law against the binary baseline.

Cache-only; no model is loaded. See ``docs/attenuation_law.md`` for the
derivation this script tests.

Three questions:

1. Does estimating ``q_ij`` blindly, instead of assuming ``q = 1``, reduce the
   error in the recovered pairwise quality product on MATH-500 and MMLU without
   regressing on GSM8K?
2. How well are the marginals ``e_i`` and the coincidences ``q_ij`` themselves
   recovered, with no labels?
3. What *is* honest ``q_ij``? Theory predicts ~0 on open-ended answers and 1/3 on
   4-way MMLU. Anything materially above chance means honest models share
   wrong-answer attractors, which is what forces inversion to carry a threshold
   rather than trusting coherence outright.
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
    blind_estimate_mc,
    blind_marginal_error_rates,
    blind_product_estimate,
    q_empirical,
    true_product,
)
from aip.analysis.bootstrap import bootstrap_statistic  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest  # noqa: E402
from aip.tasks import roster  # noqa: E402

log = configure_logging(level="INFO")

BENCHMARKS = roster.benchmarks()  # from configs/task_lists.json
#: Closed answer spaces get their full option set, so chance-level q is
#: meaningful. Derived from each benchmark's answer-space class rather than
#: listed: mmlu is no longer "the multiple-choice one" now that medqa and arc are
#: also C=4 and boolq is C=2.
LABEL_SPACES: dict[str, list[Any] | None] = {b: roster.label_space(b) for b in BENCHMARKS}


def load_benchmark(benchmark: str, models: list[str], root: Path):
    frames = {}
    shared = None
    for m in models:
        f = pd.read_parquet(root / benchmark / f"{m}.parquet").set_index("task_id").sort_index()
        frames[m] = f
        shared = f.index if shared is None else shared.intersection(f.index)
    answers = {
        m: np.array(
            [None if pd.isna(v) else str(v) for v in frames[m].loc[shared, "extracted_answer"]],
            dtype=object,
        )
        for m in models
    }
    correct = {m: frames[m].loc[shared, "is_correct"].to_numpy(dtype=int) for m in models}
    rows = [{m: answers[m][t] for m in models} for t in range(len(shared))]
    return rows, answers, correct, list(shared)


def estimates_on(
    idx: np.ndarray,
    rows: list[dict[str, Any]],
    answers: dict[str, np.ndarray],
    correct: dict[str, np.ndarray],
    models: list[str],
    label_space: list[Any] | None,
) -> dict[str, float]:
    """Blind and true quantities on one (possibly resampled) task set."""
    sub_rows = [rows[i] for i in idx]
    est = blind_estimate_mc(sub_rows, agents=models, label_space=label_space)

    # binary-law baseline, fed the only blind observable: answer agreement
    binary_products, mc_products, truths = [], [], []
    q_blind, q_true = [], []
    for a, b in itertools.combinations(models, 2):
        ai, aj = answers[a][idx], answers[b][idx]
        ci, cj = correct[a][idx], correct[b][idx]
        truth = true_product(float(1 - ci.mean()), float(1 - cj.mean()))
        truths.append(truth)
        binary_products.append(blind_product_estimate(answer_agreement(ai, aj)))
        mc_products.append(est.products[(a, b)])
        qt = q_empirical(ai, aj, ci, cj)
        if np.isfinite(qt) and np.isfinite(est.q[(a, b)]):
            q_true.append(qt)
            q_blind.append(est.q[(a, b)])

    e_true = np.array([float(1 - correct[m][idx].mean()) for m in models])
    e_mc = np.array([est.error_rates[m] for m in models])
    agreement = {
        (a, b): answer_agreement(answers[a][idx], answers[b][idx])
        for a, b in itertools.combinations(models, 2)
    }
    e_binary_map = blind_marginal_error_rates(agreement, models)
    e_binary = np.array([e_binary_map[m] for m in models])

    def mae(x, y) -> float:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.mean(np.abs(x[ok] - y[ok]))) if ok.any() else float("nan")

    # Pooled q: one ratio over every jointly-wrong event on the benchmark, which
    # has far more power than any single pair's handful of events.
    pooled_num = 0.0
    pooled_den = 0.0
    for a, b in itertools.combinations(models, 2):
        ai, aj = answers[a][idx], answers[b][idx]
        ci, cj = correct[a][idx], correct[b][idx]
        both_wrong = (ci == 0) & (cj == 0)
        if both_wrong.any():
            same = np.array(
                [
                    (x is not None) and (y is not None) and (x == y)
                    for x, y in zip(ai, aj, strict=True)
                ]
            )
            pooled_num += float(np.sum(same & both_wrong))
            pooled_den += float(np.sum(both_wrong))

    mae_binary = mae(binary_products, truths)
    mae_mc = mae(mc_products, truths)
    return {
        "mae_product_binary": mae_binary,
        "mae_product_mc": mae_mc,
        # Paired difference: the powerful comparison, since both estimators see
        # the identical resampled tasks. Negative means the multiclass law wins.
        "mae_product_delta": mae_mc - mae_binary,
        "q_pooled": pooled_num / pooled_den if pooled_den > 0 else float("nan"),
        "n_both_wrong_pooled": pooled_den,
        "mae_error_binary": mae(e_binary, e_true),
        "mae_error_mc": mae(e_mc, e_true),
        "mae_q": mae(q_blind, q_true) if q_true else float("nan"),
        "mean_q_true": float(np.mean(q_true)) if q_true else float("nan"),
        "mean_q_blind": float(np.mean(q_blind)) if q_blind else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--out", type=Path, default=Path("results/correlation"))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20250825)
    args = parser.parse_args()

    models = sorted(p.stem for p in (args.cache / "gsm8k").glob("*.parquet"))
    log.info("phase_b2.start", models=models, resamples=args.resamples)

    validation_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []

    for benchmark in BENCHMARKS:
        rows, answers, correct, task_ids = load_benchmark(benchmark, models, args.cache)
        label_space = LABEL_SPACES[benchmark]
        n = len(rows)

        def stat(key: str, rows=rows, answers=answers, correct=correct, label_space=label_space):
            # Bind the loop variables explicitly: the closure outlives the
            # iteration that created it only in intent, but binding makes that
            # independent of how bootstrap_statistic chooses to call it.
            def f(idx: np.ndarray) -> float:
                return estimates_on(idx, rows, answers, correct, models, label_space)[key]

            return f

        point = estimates_on(np.arange(n), rows, answers, correct, models, label_space)
        record: dict[str, Any] = {"benchmark": benchmark, "n_tasks": n, "models": len(models)}
        for key in (
            "mae_product_binary",
            "mae_product_mc",
            "mae_product_delta",
            "mae_error_binary",
            "mae_error_mc",
            "mae_q",
            "q_pooled",
        ):
            ci = bootstrap_statistic(stat(key), n, args.resamples, seed=args.seed)
            record.update(ci.as_dict(f"{key}_"))
        record["mean_q_true"] = point["mean_q_true"]
        record["mean_q_blind"] = point["mean_q_blind"]
        record["n_both_wrong_pooled"] = point["n_both_wrong_pooled"]
        record["chance_q"] = roster.chance_coherence(benchmark)
        validation_rows.append(record)
        log.info(
            "phase_b2.benchmark_done",
            benchmark=benchmark,
            **{k: round(v, 4) for k, v in point.items() if np.isfinite(v)},
        )

        # -- interpretation pass: ground-truth q per pair, with CIs ----------
        est_full = blind_estimate_mc(rows, agents=models, label_space=label_space)
        for a, b in itertools.combinations(models, 2):
            ai, aj, ci_, cj_ = answers[a], answers[b], correct[a], correct[b]

            def q_stat(idx: np.ndarray, ai=ai, aj=aj, ci_=ci_, cj_=cj_) -> float:
                return q_empirical(ai[idx], aj[idx], ci_[idx], cj_[idx])

            ci = bootstrap_statistic(q_stat, n, args.resamples, seed=args.seed)
            both_wrong = int(np.sum((ci_ == 0) & (cj_ == 0)))
            chance = roster.chance_coherence(benchmark)
            q_rows.append(
                {
                    "benchmark": benchmark,
                    "model_i": a,
                    "model_j": b,
                    "n_both_wrong": both_wrong,
                    "q_true": ci.point,
                    "q_true_ci_low": ci.low,
                    "q_true_ci_high": ci.high,
                    "q_blind": est_full.q[(a, b)],
                    "chance_q": chance,
                    "above_chance": bool(np.isfinite(ci.low) and ci.low > chance),
                }
            )

    validation = pd.DataFrame(validation_rows)
    q_table = pd.DataFrame(q_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(args.out / "multiclass_validation.parquet", index=False)
    q_table.to_parquet(args.out / "q_pairwise.parquet", index=False)

    manifest = Manifest.create(
        phase="phase_b2",
        config={"cache": str(args.cache), "resamples": args.resamples, "models": models},
        seeds={"bootstrap": args.seed},
        notes={
            "inference_run": False,
            "label_spaces": {
                k: (v if v else "open-ended, per-task candidates") for k, v in LABEL_SPACES.items()
            },
            "correction": (
                "Phase B fed the binary law correctness_agreement, which requires gold "
                "labels and is therefore not a blind observable. The baseline here uses "
                "answer_agreement, which is what an agent can actually see."
            ),
        },
    )
    manifest.finish(status="ok").write(Path("results/manifests") / f"{manifest.run_id}.json")

    print_report(validation, q_table, models, args)
    return 0


def print_report(validation: pd.DataFrame, q_table: pd.DataFrame, models, args) -> None:
    bar = "=" * 100
    print(f"\n{bar}\nPHASE B2 — MULTICLASS ATTENUATION LAW\n{bar}")
    print(f"bootstrap: {args.resamples} resamples over tasks, seed {args.seed}\n")

    print("1. PAIRWISE PRODUCT  (1-2e_i)(1-2e_j)/2 — MAE vs ground truth")
    print(
        f"{'benchmark':<10}{'binary law [95% CI]':>32}{'multiclass law [95% CI]':>34}{'verdict':>14}"
    )
    for r in validation.itertuples():
        b = f"{r.mae_product_binary_point:.4f} [{r.mae_product_binary_ci_low:.4f}, {r.mae_product_binary_ci_high:.4f}]"
        m = f"{r.mae_product_mc_point:.4f} [{r.mae_product_mc_ci_low:.4f}, {r.mae_product_mc_ci_high:.4f}]"
        # The paired CI is what decides, not the overlap of the two marginal CIs.
        if r.mae_product_delta_ci_high < 0:
            verdict = "IMPROVED"
        elif r.mae_product_delta_ci_low > 0:
            verdict = "REGRESSED"
        else:
            verdict = "no diff"
        print(f"{r.benchmark:<10}{b:>32}{m:>34}{verdict:>14}")
    print("\n   paired difference (multiclass - binary); negative favours multiclass")
    print(f"   {'benchmark':<10}{'delta MAE [95% CI]':>34}")
    for r in validation.itertuples():
        d = (
            f"{r.mae_product_delta_point:+.4f} "
            f"[{r.mae_product_delta_ci_low:+.4f}, {r.mae_product_delta_ci_high:+.4f}]"
        )
        print(f"   {r.benchmark:<10}{d:>34}")

    print("\n2. MARGINAL ERROR RATES e_i — MAE vs ground truth")
    print(
        f"{'benchmark':<10}{'binary law [95% CI]':>32}{'multiclass law [95% CI]':>34}{'verdict':>14}"
    )
    for r in validation.itertuples():
        b = (
            "undefined"
            if not np.isfinite(r.mae_error_binary_point)
            else f"{r.mae_error_binary_point:.4f} [{r.mae_error_binary_ci_low:.4f}, {r.mae_error_binary_ci_high:.4f}]"
        )
        m = f"{r.mae_error_mc_point:.4f} [{r.mae_error_mc_ci_low:.4f}, {r.mae_error_mc_ci_high:.4f}]"
        better = (not np.isfinite(r.mae_error_binary_point)) or (
            r.mae_error_mc_point < r.mae_error_binary_point
        )
        print(f"{r.benchmark:<10}{b:>32}{m:>34}{'IMPROVED' if better else 'regressed':>14}")

    print("\n3. COINCIDENCE q_ij — blind estimate vs ground truth")
    print(f"{'benchmark':<10}{'MAE(q) [95% CI]':>32}{'mean q true':>13}{'mean q blind':>14}")
    for r in validation.itertuples():
        m = f"{r.mae_q_point:.4f} [{r.mae_q_ci_low:.4f}, {r.mae_q_ci_high:.4f}]"
        print(f"{r.benchmark:<10}{m:>32}{r.mean_q_true:>13.4f}{r.mean_q_blind:>14.4f}")

    print("\n4. POOLED HONEST q  (all jointly-wrong events on the benchmark)")
    print(f"{'benchmark':<10}{'events':>8}{'q pooled [95% CI]':>32}{'chance':>8}  verdict")
    for r in validation.itertuples():
        q = f"{r.q_pooled_point:.3f} [{r.q_pooled_ci_low:.3f}, {r.q_pooled_ci_high:.3f}]"
        above = np.isfinite(r.q_pooled_ci_low) and r.q_pooled_ci_low > r.chance_q
        verdict = "ABOVE CHANCE" if above else "consistent with chance"
        print(
            f"{r.benchmark:<10}{int(r.n_both_wrong_pooled):>8}{q:>32}{r.chance_q:>8.3f}  {verdict}"
        )

    print("\n5. HONEST q_ij PER PAIR  (ground truth; chance is 1/3 on MMLU, ~0 open-ended)")
    print(
        f"{'benchmark':<9}{'pair':<40}{'n both wrong':>13}{'q [95% CI]':>28}{'chance':>8}  above chance?"
    )
    for r in q_table.itertuples():
        q = f"{r.q_true:.3f} [{r.q_true_ci_low:.3f}, {r.q_true_ci_high:.3f}]"
        print(
            f"{r.benchmark:<9}{r.model_i + ' + ' + r.model_j:<40}{r.n_both_wrong:>13}"
            f"{q:>28}{r.chance_q:>8.3f}  {'YES' if r.above_chance else 'no'}"
        )
    print(f"\nwrote {args.out}/multiclass_validation.parquet and {args.out}/q_pairwise.parquet")


if __name__ == "__main__":
    raise SystemExit(main())
