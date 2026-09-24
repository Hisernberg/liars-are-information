#!/usr/bin/env python3
"""E8 evaluation: the live six-model swarm (confirmation set, generated after the freeze).

Outputs in ``results/live/``:

* ``role_stats.csv`` -- per model x benchmark x role: accuracy, and for liar roles
  the *lie rate* and the truth-dependence a - 1/K that RACE exploits;
* ``contagion.csv`` -- how honest agents change after seeing a panel in which half
  the votes were lies (switch rate, switch-to-a-lie rate, accuracy drop);
* ``per_task.parquet`` -- aggregation results with the same harness and methods as
  the replay studies, for independent (``live``) and post-debate
  (``live_debate``) honest answers;
* ``live_summary.json`` -- headline numbers.
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lai.data import LIVE_MODELS  # noqa: E402
from lai.sim import CORE_METHODS, World, run_worlds  # noqa: E402
from lai.stats import compare, summary  # noqa: E402

LIVE = ROOT / "data" / "live_cache"
OUT = ROOT / "results" / "live"


def role_stats() -> pd.DataFrame:
    rows = []
    frames = [pd.read_parquet(p) for p in (LIVE / "raw").glob("stage*/*/*.parquet")]
    raw = pd.concat(frames, ignore_index=True)
    for (m, b, role), g in raw.groupby(["model", "benchmark", "role"]):
        k = 2 if b == "boolq" else 4
        acc = g.is_correct.mean()
        rows.append(dict(model=m, benchmark=b, role=role, n=len(g), accuracy=acc,
                         lie_rate=1 - acc if role in ("solo", "rushing") else np.nan,
                         truth_dependence=acc - 1 / k))
    return pd.DataFrame(rows)


def contagion() -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(p) for p in (LIVE / "raw").glob("stage*/*/*.parquet")], ignore_index=True)
    piv = raw.pivot_table(index=["benchmark", "task_id", "model"], columns="role", values="extracted_answer", aggfunc="first")
    gold = raw.drop_duplicates(["benchmark", "task_id"]).set_index(["benchmark", "task_id"]).gold_answer
    rows = []
    for (b, m), g in piv.groupby(level=["benchmark", "model"]):
        g = g.dropna(subset=["honest", "debate"])
        gg = gold.loc[[(b, t) for t in g.index.get_level_values("task_id")]].to_numpy()
        solo_panel = raw[(raw.benchmark == b) & (raw.role == "solo")].pivot(index="task_id", columns="model",
                                                                           values="extracted_answer")
        top_lie = solo_panel.mode(axis=1)[0]
        tids = g.index.get_level_values("task_id")
        switched = g.honest.to_numpy() != g.debate.to_numpy()
        to_lie = switched & (g.debate.to_numpy() == top_lie.loc[tids].to_numpy()) & (g.debate.to_numpy() != gg)
        rows.append(dict(benchmark=b, model=m, n=len(g), acc_independent=np.mean(g.honest.to_numpy() == gg),
                         acc_after_debate=np.mean(g.debate.to_numpy() == gg), switch_rate=switched.mean(),
                         switch_to_liar_plurality=to_lie.mean(),
                         right_to_wrong=np.mean((g.honest.to_numpy() == gg) & (g.debate.to_numpy() != gg)),
                         wrong_to_right=np.mean((g.honest.to_numpy() != gg) & (g.debate.to_numpy() == gg))))
    return pd.DataFrame(rows)


def worlds() -> list[World]:
    out = []
    for b, f, a, s in itertools.product(("mmlu", "boolq"), (0.1, 0.3, 0.5, 0.7),
                                        ("llm:solo", "llm:rushing", "llm:collude", "coherent"), range(5)):
        out.append(World(b, f, a, 1.0, 1.0, s, composition="live", study="live", source="live"))
    for b, a, s in itertools.product(("mmlu", "boolq"), ("llm:solo", "llm:collude"), range(5)):
        out.append(World(b, 0.5, a, 1.0, 1.0, s, composition="live", study="live", source="live_debate"))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rs = role_stats()
    rs.to_csv(OUT / "role_stats.csv", index=False)
    ct = contagion()
    ct.to_csv(OUT / "contagion.csv", index=False)
    frame, _, _ = run_worlds(worlds(), CORE_METHODS, OUT, processes=4)
    s = summary(frame, ["source", "benchmark", "attack", "f"])
    s.to_csv(OUT / "summary.csv")
    cmp = compare(frame[frame.source == "live"], "race", ["self", "majority", "aip_gated", "ds_onecoin"],
                  ["benchmark", "attack", "f"])
    cmp.to_csv(OUT / "paired_comparisons.csv", index=False)
    liar = rs[rs.role.isin(["solo", "rushing"])]
    summary_json = {
        "models": list(LIVE_MODELS),
        "honest_accuracy": rs[rs.role == "honest"].groupby("benchmark").accuracy.mean().round(4).to_dict(),
        "liar_accuracy_by_role": liar.groupby(["benchmark", "role"]).accuracy.mean().round(4).unstack().to_dict(),
        "share_of_liars_below_chance": float((liar.truth_dependence < 0).mean()),
        "contagion_mean": ct[["acc_independent", "acc_after_debate", "switch_rate", "switch_to_liar_plurality"]].mean().round(4).to_dict(),
        "race_vs_baselines_verdicts": cmp.groupby("baseline").verdict.value_counts().unstack(fill_value=0).to_dict(),
    }
    (OUT / "live_summary.json").write_text(json.dumps(summary_json, indent=2, default=str))
    print(json.dumps(summary_json, indent=2, default=str))


if __name__ == "__main__":
    main()
