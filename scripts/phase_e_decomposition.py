#!/usr/bin/env python3
"""Channel-decision decomposition: where does AIP's MMLU loss come from?

For every benchmark x attack, tabulate what an honest receiver decided about each
channel -- inverted, trusted, or discarded -- split by whether the channel was
honest or adversarial. The question is whether AIP loses on MMLU because it
inverts honest agents (a safety failure) or because it fails to invert
adversarial ones (a coordinatability failure). Those call for opposite fixes.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import phase_c_aggregate as pc  # noqa: E402
import phase_d_sweep as pds  # noqa: E402

from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.harness.manifest import Manifest, stable_seed  # noqa: E402

ATTACKS = ["semantic_negation", "always_wrong", "semantic_hallucination", "rushing"]
F_LEVELS = [0.3, 0.5, 0.7]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=20250825)
    p.add_argument("--out", type=Path, default=Path("results/adversarial"))
    args = p.parse_args()

    thresholds = InversionThresholds.load()
    cfg = yaml.safe_load(Path("configs/inversion_thresholds.yaml").read_text())
    models = sorted(q.stem for q in (Path("data/cache") / "gsm8k").glob("*.parquet"))
    rows, qrows = [], []

    for benchmark in pc.BENCHMARKS:
        cache = pc.load_cache(benchmark, models, Path("data/cache"), Path("data/cache_t07"))
        cls = cfg["benchmark_classes"][benchmark]
        ceiling = float(cfg["classes"][cls]["ceiling"])
        honest_q = float(cfg["classes"][cls].get("receiver_conditioned_q", np.nan))
        for attack in ATTACKS:
            adv = pds.load_adversarial(attack, benchmark, cache.task_ids)
            if not adv:
                continue
            # Cross-model coherence of the adversarial channel itself.
            ms = list(adv)
            if len(ms) >= 2:
                a, b = adv[ms[0]], adv[ms[1]]
                q_hat = float(
                    np.mean(
                        [
                            x[0] is not None and y[0] is not None and x[0] == y[0]
                            for x, y in zip(a, b, strict=True)
                        ]
                    )
                )
            else:
                q_hat = np.nan
            qrows.append(
                {
                    "benchmark": benchmark,
                    "answer_space_class": cls,
                    "attack": attack,
                    "q_hat_adversary": q_hat,
                    "honest_q_receiver_conditioned": honest_q,
                    "ceiling": ceiling,
                    "exceeds_honest": bool(q_hat > honest_q),
                }
            )

            for f in F_LEVELS:
                assignment, tasks = pds.build_tasks(
                    cache,
                    list(models),
                    attack,
                    f,
                    1.0,
                    adv,
                    seed=args.seed + stable_seed((benchmark, attack, f)),
                )
                agg = AIPAggregator(
                    benchmark, "gated", thresholds, ParityConfig(0.1), pc.LABEL_SPACES[benchmark]
                )
                agg.fit(tasks)
                byz = set(assignment.byzantine)
                counts = {"honest": Counter(), "adversary": Counter()}
                for receiver, peers in agg.diagnostics.channels.items():
                    if receiver in byz:
                        continue
                    for peer, s in peers.items():
                        if peer == receiver:
                            continue
                        counts["adversary" if peer in byz else "honest"][s.decision] += 1
                for kind, c in counts.items():
                    total = sum(c.values())
                    if not total:
                        continue
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "attack": attack,
                            "f": f,
                            "channel": kind,
                            "n_channels": total,
                            "pct_inverted": 100 * c["invert"] / total,
                            "pct_trusted": 100 * c["trust"] / total,
                            "pct_discarded": 100 * c["discard"] / total,
                        }
                    )

    decisions = pd.DataFrame(rows)
    qtable = pd.DataFrame(qrows)
    args.out.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(args.out / "channel_decisions.parquet", index=False)
    qtable.to_parquet(args.out / "q_by_answer_space.parquet", index=False)

    bar = "=" * 100
    print(f"\n{bar}\nCOORDINATABILITY: adversary coherence vs the honest floor\n{bar}")
    print(
        f"{'benchmark':<9}{'class':<17}{'attack':<24}{'q_hat adv':>10}{'honest q':>10}"
        f"{'ceiling':>9}  above honest?"
    )
    for r in qtable.itertuples():
        print(
            f"{r.benchmark:<9}{r.answer_space_class:<17}{r.attack:<24}"
            f"{r.q_hat_adversary:>10.3f}{r.honest_q_receiver_conditioned:>10.3f}"
            f"{r.ceiling:>9.3f}  {'YES' if r.exceeds_honest else 'no'}"
        )

    print(f"\n{bar}\nCHANNEL DECISIONS (honest receivers only)\n{bar}")
    print(
        f"{'benchmark':<9}{'attack':<24}{'f':>5}{'channel':<11}"
        f"{'inverted':>10}{'trusted':>9}{'discarded':>11}"
    )
    for r in decisions.itertuples():
        print(
            f"{r.benchmark:<9}{r.attack:<24}{r.f:>5.1f}{r.channel:<11}"
            f"{r.pct_inverted:>9.0f}%{r.pct_trusted:>8.0f}%{r.pct_discarded:>10.0f}%"
        )

    print(f"\n{bar}\nDIAGNOSIS\n{bar}")
    for benchmark in pc.BENCHMARKS:
        sub = decisions[
            (decisions.benchmark == benchmark) & (decisions.attack == "semantic_negation")
        ]
        hon = sub[sub.channel == "honest"]["pct_inverted"].mean()
        adv = sub[sub.channel == "adversary"]["pct_inverted"].mean()
        q = qtable[(qtable.benchmark == benchmark) & (qtable.attack == "semantic_negation")][
            "q_hat_adversary"
        ]
        hq = qtable[(qtable.benchmark == benchmark) & (qtable.attack == "semantic_negation")][
            "honest_q_receiver_conditioned"
        ]
        print(
            f"  {benchmark}: negation q_hat={float(q.iloc[0]):.3f} vs honest {float(hq.iloc[0]):.3f}; "
            f"honest channels inverted {hon:.0f}%, adversary channels inverted {adv:.0f}%"
        )
    m = Manifest.create(
        phase="phase_e_decomposition",
        config={"attacks": ATTACKS},
        seeds={"sweep": args.seed},
        notes={"inference_run": False},
    )
    m.finish(status="ok").write(Path("results/manifests") / f"{m.run_id}.json")
    print(f"\nwrote {args.out}/channel_decisions.parquet, {args.out}/q_by_answer_space.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
