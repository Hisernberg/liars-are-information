#!/usr/bin/env python3
"""9b: which model subsets make a diverse swarm, and what predicts it.

For every 3-model and 5-model subset of the frozen roster, build a 10-agent
swarm split as evenly as possible across the subset, average rho over the real
pair census, and report the effective swarm size. Then ask which cheap,
observable property of a subset predicts its N_eff, because "measure the
correlation matrix first" is not deployment guidance.

Three things about the basis, all of which limit the answer.

1. **Within-model rho exists only on GSM8K.** It needs the same model sampled
   twice, which only GSM8K has (greedy plus a T=0.7 resample). GSM8K is
   therefore the only benchmark where N_eff is *measured*; everywhere else the
   same-model pairs are dropped from the average, which understates rho and
   yields an **upper bound**. Both are reported, never mixed.

2. **Family diversity cannot be the recipe in this roster.** All seven models
   come from seven different labs, so every k-subset has exactly k families and
   the predictor is constant by construction. This is reported as a limitation
   of the roster, not as evidence that lineage does not matter.

3. **Accuracy spread is confounded with mean accuracy.** A subset containing
   Llama-3.2 3B has both a wide spread and a lower mean. The candidate
   predictors are therefore reported with their correlations side by side, and
   a partial correlation controlling for the mean.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aip.aggregation.correlation import (  # noqa: E402
    allocate_agents,
    effective_swarm_size,
    subset_swarm_rho_bar,
)
from aip.harness.manifest import Manifest  # noqa: E402
from aip.harness.rules import load_rules  # noqa: E402
from aip.tasks import roster  # noqa: E402

N_SWARM = 10
SUBSET_SIZES = (3, 5)


def frozen_roster(path: Path) -> dict[str, dict]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))["models"]
    return {
        name: entry for name, entry in registry.items()
        if entry.get("arm") == "frozen" and entry.get("enabled", True)
    }


def model_accuracy(cache: Path, benchmarks: list[str], models: list[str]) -> pd.DataFrame:
    rows = []
    for benchmark in benchmarks:
        for model in models:
            path = cache / benchmark / f"{model}.parquet"
            if not path.exists():
                raise SystemExit(f"missing cache cell: {path}")
            frame = pd.read_parquet(path)
            rows.append({"benchmark": benchmark, "model": model,
                         "accuracy": float(frame.is_correct.mean())})
    return pd.DataFrame(rows)


def subset_accuracy(
    answers: dict[str, pd.Series], gold: pd.Series, subset: tuple[str, ...],
    counts: list[int],
) -> float:
    """Plurality accuracy of a swarm built from ``subset`` with ``counts`` agents.

    Agents running the same model share a cached answer, so a 4/3/3 swarm is a
    plurality vote with weights 4/3/3. That is not a modelling shortcut -- it is
    what a swarm of duplicated models actually is, and it is the reason N_eff
    matters at all. Ties are resolved toward the model with more agents, then
    lexically, so the tie-break cannot depend on dict ordering.
    """
    votes: dict[str, float] = {}
    per_task = np.zeros(len(gold), dtype=float)
    for t in range(len(gold)):
        votes.clear()
        for model, weight in zip(subset, counts, strict=True):
            answer = answers[model].iloc[t]
            if pd.isna(answer):
                continue
            votes[str(answer)] = votes.get(str(answer), 0.0) + weight
        if not votes:
            per_task[t] = 0.0
            continue
        best = max(sorted(votes), key=lambda a: votes[a])
        target = gold.iloc[t]
        per_task[t] = float(not pd.isna(target) and best == str(target))
    return float(per_task.mean())


def cross_rho(pairs: pd.DataFrame, benchmark: str) -> dict[frozenset[str], float]:
    sub = pairs[(pairs.kind == "cross_model") & (pairs.benchmark == benchmark)]
    return {frozenset((r.model_i, r.model_j)): float(r.phi_point) for r in sub.itertuples()}


def within_rho(pairs: pd.DataFrame) -> dict[str, float]:
    sub = pairs[pairs.kind == "within_model"]
    return {r.model_i: float(r.phi_point) for r in sub.itertuples()}


def count_assignments(n_agents: int, n_models: int) -> list[tuple[int, ...]]:
    """Every distinct way to spread the surplus agents over the models.

    Ten agents over three models is 4/3/3, and *which* model gets the fourth
    agent decides every three-way disagreement -- the heavier model wins by
    weight alone. Allocating the surplus to the first model would make the
    result depend on alphabetical order, so all distinct assignments are
    enumerated and averaged over. For k=5 the split is 2/2/2/2/2 and there is
    only one assignment, which is why that column needs no such correction.
    """
    base = allocate_agents(n_agents, n_models)
    return sorted(set(itertools.permutations(base)))


def evaluate(
    subset: tuple[str, ...],
    benchmark: str,
    cross: dict[frozenset[str], float],
    within: dict[str, float],
    accuracy: dict[str, float],
    registry: dict[str, dict],
    answers: dict[str, pd.Series],
    gold: pd.Series,
    *,
    measured_within: bool,
) -> dict | None:
    assignments = count_assignments(N_SWARM, len(subset))
    idx_cross = {}
    for i, j in itertools.combinations(range(len(subset)), 2):
        value = cross.get(frozenset((subset[i], subset[j])))
        if value is None:
            return None
        idx_cross[(i, j)] = value
    rho_within = [within.get(m, float("nan")) for m in subset]
    rhos, accs_sim = [], []
    for counts in assignments:
        value = subset_swarm_rho_bar(
            list(counts), rho_within, idx_cross, include_within=measured_within
        )
        if not np.isfinite(value):
            return None
        rhos.append(value)
        accs_sim.append(subset_accuracy(answers, gold, subset, list(counts)))
    rho_bar = float(np.mean(rhos))
    swarm_acc = float(np.mean(accs_sim))
    swarm_acc_spread = float(max(accs_sim) - min(accs_sim))
    accs = [accuracy[m] for m in subset]
    params = [float(registry[m]["params_b"]) for m in subset]
    tiers = {registry[m].get("tier") for m in subset}
    return {
        "benchmark": benchmark,
        "k": len(subset),
        "composition": " + ".join(subset),
        "basis": ("measured (within-model rho available)" if measured_within
                  else "cross-model pairs only -- UPPER BOUND on N_eff"),
        "rho_bar": rho_bar,
        "n_eff": effective_swarm_size(N_SWARM, rho_bar),
        "swarm_accuracy": swarm_acc,
        "n_agent_assignments": len(assignments),
        "swarm_accuracy_assignment_spread": swarm_acc_spread,
        "best_single_model": float(max(accs)),
        "swarm_gain_over_best": swarm_acc - float(max(accs)),
        "acc_mean": float(np.mean(accs)),
        "acc_spread": float(max(accs) - min(accs)),
        "acc_min": float(min(accs)),
        "param_ratio": float(max(params) / min(params)),
        "n_tiers": len(tiers),
        "n_families": len(subset),  # one lab per model in this roster
        "has_weak": bool(any(registry[m].get("tier") == "weak_fast" for m in subset)),
    }


def predictors(table: pd.DataFrame, target: str = "n_eff") -> pd.DataFrame:
    """Spearman correlation of each candidate predictor with ``target``."""
    from scipy.stats import spearmanr

    candidates = ["acc_spread", "acc_mean", "acc_min", "param_ratio", "n_tiers",
                  "n_families", "has_weak", "n_eff"]
    rows = []
    for (benchmark, k), group in table.groupby(["benchmark", "k"]):
        for name in candidates:
            if name == target:
                continue
            values = group[name].astype(float)
            if values.nunique() < 2:
                rows.append({"benchmark": benchmark, "k": k, "target": target,
                             "predictor": name, "spearman": float("nan"),
                             "p_value": float("nan"),
                             "note": "constant across subsets -- cannot discriminate"})
                continue
            rho, p = spearmanr(values, group[target])
            # Partial: residualise both on mean accuracy, which is confounded
            # with every spread-like predictor.
            note = ""
            if name != "acc_mean" and group.acc_mean.nunique() > 2:
                a = np.polyfit(group.acc_mean, values, 1)
                b = np.polyfit(group.acc_mean, group[target], 1)
                r_x = values - np.polyval(a, group.acc_mean)
                r_y = group[target] - np.polyval(b, group.acc_mean)
                if r_x.nunique() > 1:
                    pr, _ = spearmanr(r_x, r_y)
                    note = f"partial (controlling mean accuracy) {pr:+.2f}"
            rows.append({"benchmark": benchmark, "k": k, "target": target,
                         "predictor": name, "spearman": float(rho),
                         "p_value": float(p), "note": note})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--pairs", type=Path,
                    default=Path("results/correlation/pairwise_estimates.parquet"))
    ap.add_argument("--models", type=Path, default=Path("configs/models.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/correlation"))
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if not args.pairs.exists():
        raise SystemExit(f"missing {args.pairs}; run scripts/phase_b_correlation.py first")
    registry = frozen_roster(args.models)
    models = sorted(registry)
    if len(models) < max(SUBSET_SIZES):
        raise SystemExit(f"frozen roster has {len(models)} models; need "
                         f"{max(SUBSET_SIZES)}")
    benchmarks = roster.benchmarks()
    rules_headline = list(load_rules().headline_benchmarks)
    pairs = pd.read_parquet(args.pairs)
    within = within_rho(pairs)
    acc = model_accuracy(args.cache, benchmarks, models)

    rows = []
    for benchmark in benchmarks:
        cross = cross_rho(pairs, benchmark)
        if not cross:
            continue
        accuracy = dict(acc[acc.benchmark == benchmark][["model", "accuracy"]].values)
        answers, gold = {}, None
        for model in models:
            frame = pd.read_parquet(args.cache / benchmark / f"{model}.parquet")
            frame = frame.sort_values("task_id").reset_index(drop=True)
            answers[model] = frame.extracted_answer
            if gold is None:
                gold = frame.gold_answer
            elif not gold.equals(frame.gold_answer):
                raise SystemExit(
                    f"{benchmark}: cache cells disagree on gold answers; the "
                    "subsets would not be scored on the same questions"
                )
        # Within-model rho is a property of the benchmark, not of the model, so
        # a subset is only "measured" where that benchmark has a resample.
        measured = all(m in within for m in models) and benchmark == "gsm8k"
        for k in SUBSET_SIZES:
            for subset in itertools.combinations(models, k):
                row = evaluate(subset, benchmark, cross, within, accuracy,
                               registry, answers, gold, measured_within=measured)
                if row is not None:
                    rows.append(row)

    if not rows:
        raise SystemExit("no compositions evaluated")
    table = pd.DataFrame(rows)
    preds = pd.concat(
        [predictors(table, "n_eff"), predictors(table, "swarm_accuracy")],
        ignore_index=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "composition_neff.parquet", index=False)
    preds.to_parquet(args.out / "composition_predictors.parquet", index=False)
    Manifest.create(
        phase="phase_composition",
        config={"subset_sizes": list(SUBSET_SIZES), "n_swarm": N_SWARM,
                "benchmarks": benchmarks},
        seeds={},
    ).write(args.out / "composition_neff.manifest.json")

    pd.set_option("display.width", 220)
    n_sub = {k: len(list(itertools.combinations(models, k))) for k in SUBSET_SIZES}
    print("=" * 104)
    print(f"9b COMPOSITION GUIDANCE -- {n_sub} subsets of {len(models)} models, "
          f"{N_SWARM}-agent swarms")
    print("=" * 104)

    # Does composition move the outcome at all? This has to be asked before any
    # ranking is read, because a ranking of differences below the measurement
    # floor is a ranking of noise.
    floors = {
        key.split(":")[0]: value
        for key, value in json.loads(
            (ROOT / "results" / "measurement_floor.json").read_text()
        )["per_benchmark"].items()
    }
    print("\nDOES COMPOSITION MATTER? swarm-accuracy range across subsets vs the floor")
    verdicts = []
    for (benchmark, k), group in table.groupby(["benchmark", "k"]):
        span = float(group.swarm_accuracy.max() - group.swarm_accuracy.min())
        floor = floors[benchmark]
        verdicts.append({"benchmark": benchmark, "k": k, "n_subsets": len(group),
                         "range": span, "floor": floor,
                         "detectable": span >= floor,
                         "best_gain_over_single": float(
                             group.swarm_gain_over_best.max())})
        print(f"  {benchmark:<9} k={k}  range {span:.3f}  floor {floor:.3f}  "
              f"{'DETECTABLE' if span >= floor else 'undetectable':<13}"
              f"best gain over best single model "
              f"{group.swarm_gain_over_best.max():+.3f}")
    pd.DataFrame(verdicts).to_parquet(args.out / "composition_floor.parquet", index=False)

    headline = [b for b in benchmarks if b in set(rules_headline)]
    for (benchmark, k), group in table.groupby(["benchmark", "k"]):
        if benchmark not in {*headline, "gsm8k"}:
            continue
        print(f"\n--- {benchmark}, k={k} :: {group.basis.iloc[0]} ---")
        for metric in ("n_eff", "swarm_accuracy"):
            ranked = group.sort_values(metric, ascending=False)
            print(f"  ranked by {metric}")
            for label, part in (("TOP   ", ranked.head(args.top)),
                                ("BOTTOM", ranked.tail(args.top))):
                for r in part.itertuples():
                    print(f"    {label} N_eff {r.n_eff:5.2f}  swarm acc "
                          f"{r.swarm_accuracy:.3f}  best single "
                          f"{r.best_single_model:.3f}  gain "
                          f"{r.swarm_gain_over_best:+.3f}  "
                          f"acc mean {r.acc_mean:.3f} spread {r.acc_spread:.3f}  "
                          f"{r.composition}")

    print("\n" + "=" * 104)
    print("WHAT PREDICTS N_eff (Spearman against N_eff, within benchmark and k)")
    print("=" * 104)
    for target in ("n_eff", "swarm_accuracy"):
        print(f"\n### target: {target}")
        for (benchmark, k), group in preds[preds.target == target].groupby(
                ["benchmark", "k"]):
            if benchmark not in {*headline, "gsm8k"}:
                continue
            print(f"  {benchmark}, k={k}")
            for r in group.sort_values("spearman",
                                       key=lambda s: -s.abs()).itertuples():
                if not np.isfinite(r.spearman):
                    print(f"    {r.predictor:<14}      --      {r.note}")
                    continue
                print(f"    {r.predictor:<14} {r.spearman:+.3f}  "
                      f"p={r.p_value:.3g}  {r.note}")

    bound = table[table.basis.str.contains("UPPER BOUND")]
    if not bound.empty:
        print(f"\n{len(bound)} rows on {bound.benchmark.nunique()} benchmarks are upper "
              "bounds (no within-model resample); they are in the parquet, labelled, "
              "and are not ranked beside the measured ones.")
    print(f"\nwrote {args.out}/composition_neff.parquet, "
          f"{args.out}/composition_predictors.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
