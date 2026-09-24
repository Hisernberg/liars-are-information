#!/usr/bin/env python3
"""A threshold-evading adversary: what happens when the attacker reads the defence?

Every result in this study so far assumes the adversary does not know AIP exists.
That is the assumption a reviewer removes first, and it is the one that matters,
because AIP's threshold is not a secret: the calibrated honest-q ceiling is
published in configs/inversion_thresholds.yaml. An adversary who reads it can aim
just underneath -- coordinated enough to move the vote, too incoherent to be
inverted.

The knob is the probability `p` that each Byzantine agent follows the bloc's
shared lie instead of independently sampling the same wrong-answer pool.
For a uniform candidate pool of size K, expected Byzantine-pair agreement is

    q_expected = p^2 + (1-p^2) / K.

Closed spaces have K=C-1. Open spaces use the distinct wrong answers produced by
the assigned honest agents on that task, or a singleton synthetic fallback.
Mixed shared/independent draws can collide, as can two independent draws.

The sweep varies p and measures accuracy. It saves theoretical and empirical
Byzantine-pair agreement separately; neither is AIP's receiver-conditioned,
partner-selected gate statistic. A comparison against the receiver ceiling is
descriptive and does not establish whether the default binomial gate fires.
Original result files used p^2+(1-p)^2/(C-1) or p^2 for their q_realised axis.
Those historical axes are mislabelled; their numerical attack probabilities and
accuracy measurements are preserved and require reinterpretation.

Reported as measured. If the evasion band is wide everywhere, that is a serious
limitation of the method and belongs in the paper in those words.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import phase_c_aggregate as pc  # noqa: E402

from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.aggregation.baselines import MajorityVote, SACFilterRefine  # noqa: E402
from aip.analysis.bootstrap import bootstrap_statistic  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest, stable_seed  # noqa: E402
from aip.harness.rules import load_rules  # noqa: E402
from aip.swarm.broadcast import AdversaryConfig  # noqa: E402
from aip.tasks import roster  # noqa: E402

log = configure_logging(level="WARNING")

COHERENCE_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
F_GRID = [0.3, 0.5, 0.7]


def realised_q(
    p: float,
    n_options: int | None,
    *,
    n_wrong_candidates: int | None = None,
) -> float:
    """Expected Byzantine-pair agreement for the actual uniform-pool sampler.

    The name is retained for callers, but this is a population expectation,
    not a measured gate statistic. Open spaces must supply the task's positive
    candidate count (one for the fallback); answer-space openness alone does
    not determine collision probability.
    """
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be finite and in [0, 1]")
    if n_options is not None:
        if n_wrong_candidates is not None:
            raise ValueError("supply n_options or n_wrong_candidates, not both")
        if isinstance(n_options, bool) or not isinstance(n_options, (int, np.integer)):
            raise ValueError("n_options must be an integer >= 2")
        if n_options < 2:
            raise ValueError("n_options must be an integer >= 2")
        candidates = n_options - 1
    else:
        if (
            isinstance(n_wrong_candidates, bool)
            or not isinstance(n_wrong_candidates, (int, np.integer))
            or n_wrong_candidates < 1
        ):
            raise ValueError("open spaces require a positive integer n_wrong_candidates")
        candidates = n_wrong_candidates
    return float(p**2 + (1.0 - p**2) / candidates)


def pairwise_q_metadata(p: float, n_options: int | None, tasks, gold: list[str]) -> dict:
    """Describe attack-pair collisions from complete-broadcast task observations.

    Pool sizes use the same assigned honest answers as the sampler. Measured
    collisions count only distinct Byzantine pairs, independently of the
    receiver's gate conditioning. No new random draw changes the experiment.
    """
    expected, pool_sizes = [], []
    coincidences, pairs = 0, 0
    for observations, answer in zip(tasks, gold, strict=True):
        broadcasts = observations[0].broadcasts
        if len(broadcasts) != len(observations):
            raise ValueError("pairwise q metadata requires complete broadcast observations")
        pool_size = (
            n_options - 1
            if n_options is not None
            else max(1, len({
                b.answer for b in broadcasts
                if not b.is_byzantine and b.answer is not None and b.answer != answer
            }))
        )
        pool_sizes.append(pool_size)
        expected.append(realised_q(p, None, n_wrong_candidates=pool_size))
        malicious = [b.answer for b in broadcasts if b.is_byzantine]
        for i, left in enumerate(malicious):
            for right in malicious[i + 1:]:
                pairs += 1
                coincidences += int(left == right)
    q_expected = float(np.mean(expected)) if expected else float("nan")
    return {
        # Compatibility alias; its semantics are explicit rather than implying
        # a measured receiver-level statistic.
        "q_realised": q_expected,
        "q_realised_kind": "legacy_column_alias_of_q_expected_pairwise",
        "q_expected_pairwise": q_expected,
        "q_empirical_pairwise": coincidences / pairs if pairs else float("nan"),
        "q_pair_count": pairs,
        "q_coincidence_count": coincidences,
        "q_candidate_pool_min": min(pool_sizes) if pool_sizes else None,
        "q_candidate_pool_max": max(pool_sizes) if pool_sizes else None,
        "q_formula_version": "shared_uniform_wrong_v2",
        "q_scope": "unconditioned_byzantine_pairs_not_receiver_gate",
    }


def main() -> int:
    rules = load_rules()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="*", default=list(rules.headline_benchmarks))
    ap.add_argument("--resamples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20250825)
    ap.add_argument("--out", type=Path, default=Path("results/adversarial"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        globals()["COHERENCE_GRID"] = [0.4, 1.0]
        globals()["F_GRID"] = [0.5]
        args.resamples, args.benchmarks = 20, args.benchmarks[:1]

    thresholds = InversionThresholds.load()
    parity = ParityConfig(0.1)
    manifest = Manifest.create(
        phase="phase_gate_aware",
        config={"benchmarks": args.benchmarks, "coherence_grid": COHERENCE_GRID,
                "f_grid": F_GRID, "q_formula_version": "shared_uniform_wrong_v2",
                "q_scope": "unconditioned_byzantine_pairs_not_receiver_gate",
                "historical_q_axis": "mislabelled_in_original_result_files"},
        seeds={"analysis": args.seed},
    )

    rows: list[dict] = []
    for benchmark in args.benchmarks:
        models = sorted(q.stem for q in (Path("data/cache") / benchmark).glob("*.parquet"))
        if not models:
            continue
        cache = pc.load_cache(benchmark, models, Path("data/cache"), Path("data/cache_t07"))
        labels = pc.LABEL_SPACES[benchmark]
        ceiling = thresholds.ceiling_for(benchmark)
        n_opt = roster.n_options(benchmark)

        for p_coh in COHERENCE_GRID:
            for f in F_GRID:
                assignment, tasks = pc.build_task_observations(
                    cache, list(models), f, 1.0, "complete", None,
                    AdversaryConfig(kind="gate_aware", error_rate=1.0, coherence=p_coh),
                    seed=args.seed + stable_seed((benchmark, "gate_aware", p_coh, f)),
                )
                honest = [a for a in range(pc.N_AGENTS) if a not in assignment.byzantine]
                if not honest:
                    continue
                q_metadata = pairwise_q_metadata(p_coh, n_opt, tasks, cache.gold)
                for agg in (
                    AIPAggregator(benchmark, "gated", thresholds, parity, labels),
                    AIPAggregator(benchmark, "trust_only", thresholds, parity, labels),
                    # M1/M2/M3: the mitigations under pre-registered test.
                    AIPAggregator(benchmark, "gated", thresholds, parity, labels,
                                  soft_gate=6.0),
                    AIPAggregator(benchmark, "gated", thresholds, parity, labels,
                                  randomized_threshold=0.3, threshold_seed=args.seed),
                    AIPAggregator(benchmark, "gated", thresholds, parity, labels,
                                  soft_gate=6.0, randomized_threshold=0.3,
                                  threshold_seed=args.seed),
                    SACFilterRefine(parity=parity),
                    MajorityVote(),
                ):
                    if agg.needs_fit:
                        agg.fit(tasks)
                    per_task = np.array(
                        [
                            np.mean([
                                agg.aggregate(tasks[t][a].broadcasts, a) == cache.gold[t]
                                for a in honest
                            ])
                            for t in range(len(tasks))
                        ],
                        dtype=float,
                    )
                    ci = bootstrap_statistic(
                        lambda i, v=per_task: float(np.mean(v[i])),
                        per_task.size, args.resamples, seed=args.seed,
                    )
                    rows.append(
                        {
                            "benchmark": benchmark, "coherence_p": p_coh,
                            **q_metadata, "ceiling": ceiling,
                            "below_ceiling": bool(q_metadata["q_expected_pairwise"] < ceiling),
                            "below_ceiling_scope": "descriptive_not_gate_decision",
                            "f": f, "method": agg.name,
                            **ci.as_dict("accuracy_"),
                        }
                    )
            log.warning("gate_aware.cell", benchmark=benchmark, p=p_coh)

    if not rows:
        print("no rows produced", file=sys.stderr)
        return 1

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / f"gate_aware{args.tag}.parquet", index=False)
    manifest.finish(status="ok", n_rows=len(frame)).write(
        Path("results/manifests") / f"{manifest.run_id}.json"
    )

    print(f"\nwrote {args.out}/gate_aware.parquet ({len(frame)} rows)\n")
    gated = frame[frame.method == "aip_gated"]
    trust = frame[frame.method == "aip_trust_only"]
    merged = gated.merge(
        trust, on=["benchmark", "coherence_p", "f"], suffixes=("_gated", "_trust")
    )
    merged["inversion_gain"] = merged.accuracy_point_gated - merged.accuracy_point_trust
    print("INVERSION GAIN by attack probability (pairwise q is not the gate statistic):")
    for b in merged.benchmark.unique():
        d = merged[merged.benchmark == b]
        print(f"\n  {b}  (ceiling {d.ceiling_gated.iloc[0]:.3f})")
        piv = d.pivot_table(index="coherence_p", columns="f", values="inversion_gain")
        print(piv.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
