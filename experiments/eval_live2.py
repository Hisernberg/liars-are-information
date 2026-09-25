#!/usr/bin/env python3
"""E10 evaluation: the pre-registered fresh live run (docs/PREREGISTRATION_E10.md).

Reads ``data/live_cache_v2`` (ARC items 0-119, BoolQ items 120-199; six live models;
honest, solo, rushing, debate and RACE-informed debate roles) and writes ``results/live2/``:

* ``role_stats.csv``, ``answer_bias.csv`` -- as in E8;
* ``contagion.csv`` -- per model x benchmark x condition (plain / informed debate):
  accuracy before and after, switch rates;
* ``per_task.parquet`` -- aggregation with the E8 harness (sources ``live2``,
  ``live2_debate``, ``live2_informed``), core methods plus classical crowd estimators;
* ``receivers.csv`` -- RACE v3.1 / v3.0 / receiver alone per honest receiver model;
* ``hypotheses.csv`` and ``hypotheses.json`` -- H1-H4 with the pre-registered decision rule:
  *supported* if the stated ordering holds in the mean and no cell shows a significant
  reversal (Holm-corrected Wilcoxon and paired bootstrap interval agree), *contradicted*
  if the ordering is reversed in the mean, otherwise *inconclusive*;
* ``h3_individual.csv`` -- informed vs plain debate, paired over (model, question);
* ``run_manifest.json``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eval_live  # noqa: E402
from lai.sim import CORE_METHODS, World, run_worlds  # noqa: E402
from lai.stats import compare, holm, paired_bootstrap, signed_rank_p, summary  # noqa: E402

LIVE2 = ROOT / "data" / "live_cache_v2"
OUT = ROOT / "results" / "live2"
BENCHES = ("arc", "boolq")
LLM_ATTACKS = ("llm:solo", "llm:rushing", "llm:collude")
METHODS = (*CORE_METHODS, "iwmv", "mace", "glad", "kos")


def raw_answers() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in (LIVE2 / "raw").glob("stage*/*/*.parquet")], ignore_index=True)


def contagion(raw: pd.DataFrame) -> pd.DataFrame:
    """Honest agents before (independent) and after seeing the half-liar panel, plain and informed."""
    piv = raw.pivot_table(index=["benchmark", "task_id", "model"], columns="role", values="extracted_answer",
                          aggfunc="first")
    gold = raw.drop_duplicates(["benchmark", "task_id"]).set_index(["benchmark", "task_id"]).gold_answer
    rows = []
    for (b, m), g in piv.groupby(level=["benchmark", "model"]):
        for cond in ("debate", "informed"):
            if cond not in g:
                continue
            h = g.dropna(subset=["honest", cond])
            gg = gold.loc[[(b, t) for t in h.index.get_level_values("task_id")]].to_numpy()
            before, after = h.honest.to_numpy(), h[cond].to_numpy()
            rows.append(dict(benchmark=b, model=m, condition=cond, n=len(h), acc_independent=np.mean(before == gg),
                             acc_after=np.mean(after == gg), switch_rate=np.mean(before != after),
                             right_to_wrong=np.mean((before == gg) & (after != gg)),
                             wrong_to_right=np.mean((before != gg) & (after == gg))))
    return pd.DataFrame(rows)


def h3_individual(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H3: individual post-debate accuracy, informed minus plain, paired over (model, question)."""
    piv = raw.pivot_table(index=["benchmark", "task_id", "model"], columns="role", values="is_correct",
                          aggfunc="first").dropna(subset=["debate", "informed"])
    # the first 8 questions in presentation order carried no reliability annotation
    warm = raw[raw.role == "informed"].groupby("benchmark", sort=False).task_id.apply(lambda s: set(pd.unique(s)[:8]))
    rows = []
    for scope, part in [("pooled", piv), *((b, piv.xs(b, level="benchmark", drop_level=False)) for b in BENCHES)]:
        for sub, frame in (("all", part), ("after_warmup", part[[t not in warm.get(b, set())
                                                                  for b, t, _ in part.index]])):
            diff = (frame.informed.astype(float) - frame.debate.astype(float)).to_numpy()
            seed = int.from_bytes(hashlib.sha256(f"h3|{scope}|{sub}".encode()).digest()[:4], "big")
            mean, lo, hi, _ = paired_bootstrap(diff, 2000, seed)
            rows.append(dict(scope=scope, subset=sub, n_pairs=len(diff), acc_plain=frame.debate.mean(),
                             acc_informed=frame.informed.mean(), delta=mean, ci_low=lo, ci_high=hi,
                             p=signed_rank_p(diff)))
    out = pd.DataFrame(rows)
    per_model = piv.groupby(["benchmark", "model"])[["honest", "debate", "informed"]].mean().reset_index()
    return out, per_model


def receivers() -> pd.DataFrame:
    from aip.aggregation.baselines import MajorityVote

    from lai.data import score
    from lai.race import RACEAggregator
    from lai.sim import build_world

    rows = []
    for b, atk, f, seed in itertools.product(BENCHES, LLM_ATTACKS, (0.3, 0.5, 0.7), range(5)):
        bw = build_world(World(b, f, atk, 1.0, 1.0, seed, composition="live", study="live2", source="live2"))
        for r in bw.honest:
            fits = {}
            for key, model in (("race", "auto"), ("race_onecoin", "onecoin")):
                agg = RACEAggregator(bw.data.label_space, model=model)
                agg.fit([(bw.defense[t][r],) for t in bw.splits["history"]])
                fits[key] = agg
            for t in bw.splits["test"]:
                o, g = bw.defense[t][r], bw.data.gold[t]
                rows.append(dict(benchmark=b, model=bw.models[r], self=score(b, o.own.answer, g),
                                 race=score(b, fits["race"].aggregate(o.broadcasts, r), g),
                                 race_onecoin=score(b, fits["race_onecoin"].aggregate(o.broadcasts, r), g),
                                 majority=score(b, MajorityVote().aggregate(o.broadcasts, r), g)))
    d = pd.DataFrame(rows)
    return d.groupby(["benchmark", "model"])[["self", "majority", "race", "race_onecoin"]].mean().reset_index()


def worlds() -> list[World]:
    out = []
    for b, f, a, s in itertools.product(BENCHES, (0.1, 0.3, 0.5, 0.7), (*LLM_ATTACKS, "coherent"), range(5)):
        out.append(World(b, f, a, 1.0, 1.0, s, composition="live", study="live2", source="live2"))
    for src, b, a, s in itertools.product(("live2_debate", "live2_informed"), BENCHES, ("llm:solo", "llm:collude"),
                                          range(5)):
        out.append(World(b, 0.5, a, 1.0, 1.0, s, composition="live", study="live2", source=src))
    return out


def verdict(mean_delta: float, cells: pd.DataFrame, strict: bool) -> str:
    """Pre-registered rule. ``strict``: the ordering is '>' (else '>=', where a zero mean counts as holding)."""
    reversed_sig = bool((cells.verdict == "loss").any()) if len(cells) else False
    holds = mean_delta > 0 if strict else mean_delta >= -1e-9
    if holds and not reversed_sig:
        return "supported"
    if mean_delta < 0 and (strict or mean_delta < -1e-9):
        return "contradicted"
    return "inconclusive"


def hypotheses(frame: pd.DataFrame, h3: pd.DataFrame) -> list[dict]:
    ind = frame[(frame.source == "live2") & (frame.split == "test")]
    out = []

    def part(name, bench, fs, target, base, strict, text):
        sub = frame[(frame.source == "live2") & (frame.benchmark == bench) & frame.attack.isin(LLM_ATTACKS)
                    & frame.f.isin(fs)]
        cells = compare(sub, target, [base], ["attack", "f"])
        t = ind[(ind.benchmark == bench) & ind.attack.isin(LLM_ATTACKS) & ind.f.isin(fs)]
        cell_means = t.groupby(["attack", "f", "seed", "method"]).accuracy.mean().unstack()
        a, b = cell_means[target].mean(), cell_means[base].mean()
        out.append(dict(hypothesis=name, statement=text, target=target, baseline=base, target_acc=100 * a,
                        baseline_acc=100 * b, delta=100 * (a - b), cells=len(cells),
                        wins=int((cells.verdict == "win").sum()), ties=int((cells.verdict == "tie").sum()),
                        losses=int((cells.verdict == "loss").sum()), verdict=verdict(a - b, cells, strict)))

    part("H1a", "boolq", (0.1, 0.3, 0.5, 0.7), "race", "race_onecoin", True, "BoolQ: RACE v3.1 > RACE v3.0")
    part("H1b", "boolq", (0.1, 0.3, 0.5, 0.7), "race", "self", False, "BoolQ: RACE v3.1 >= receiver alone")
    part("H2a", "arc", (0.5, 0.7), "race", "majority", False, "ARC, f>=0.5: RACE >= majority vote")
    part("H2b", "arc", (0.5, 0.7), "race", "self", False, "ARC, f>=0.5: RACE >= receiver alone")
    # H3: pooled paired mean; cells are the two benchmarks, Holm over them
    pooled = h3[(h3.scope == "pooled") & (h3.subset == "all")].iloc[0]
    per_b = h3[(h3.scope != "pooled") & (h3.subset == "all")].copy()
    per_b["p_holm"] = holm(per_b.p).to_numpy()
    per_b["verdict"] = np.where((per_b.p_holm < 0.05) & (per_b.ci_high < 0), "loss",
                                np.where((per_b.p_holm < 0.05) & (per_b.ci_low > 0), "win", "tie"))
    out.append(dict(hypothesis="H3", statement="Individual accuracy: informed debate > plain debate",
                    target="informed", baseline="debate", target_acc=100 * pooled.acc_informed,
                    baseline_acc=100 * pooled.acc_plain, delta=100 * pooled.delta, cells=len(per_b),
                    wins=int((per_b.verdict == "win").sum()), ties=int((per_b.verdict == "tie").sum()),
                    losses=int((per_b.verdict == "loss").sum()), p_pooled=pooled.p,
                    verdict=verdict(pooled.delta, per_b, True)))
    return out


def h4(frame: pd.DataFrame) -> pd.DataFrame:
    """H4 (exploratory): pooled accuracy on independent, plain-debate and informed-debate answers."""
    t = frame[(frame.split == "test") & (frame.f == 0.5) & frame.attack.isin(["llm:solo", "llm:collude"])]
    m = t.groupby(["benchmark", "source", "method"]).accuracy.mean().unstack("source") * 100
    return m.loc[(slice(None), ["self", "majority", "race", "race_onecoin", "aip_gated", "oracle_channel"]), :]


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    eval_live.LIVE = LIVE2
    raw = raw_answers()
    eval_live.role_stats().to_csv(OUT / "role_stats.csv", index=False)
    eval_live.answer_bias().to_csv(OUT / "answer_bias.csv", index=False)
    contagion(raw).to_csv(OUT / "contagion.csv", index=False)
    h3, per_model = h3_individual(raw)
    h3.to_csv(OUT / "h3_individual.csv", index=False)
    per_model.to_csv(OUT / "h3_per_model.csv", index=False)
    receivers().to_csv(OUT / "receivers.csv", index=False)
    ws = worlds()
    frame, _, _ = run_worlds(ws, METHODS, OUT, processes=4)
    summary(frame, ["source", "benchmark", "attack", "f"]).to_csv(OUT / "summary.csv")
    cmp = compare(frame[frame.source == "live2"], "race", ["self", "majority", "aip_gated", "ds_onecoin", "race_onecoin"],
                  ["benchmark", "attack", "f"])
    cmp.to_csv(OUT / "paired_comparisons.csv", index=False)
    hyp = pd.DataFrame(hypotheses(frame, h3))
    hyp.to_csv(OUT / "hypotheses.csv", index=False)
    h4(frame).to_csv(OUT / "h4_pooled.csv")
    (OUT / "hypotheses.json").write_text(json.dumps(hyp.to_dict(orient="records"), indent=2, default=float))
    print(hyp.to_string())
    print(h3.to_string())
    inputs = sorted(LIVE2.glob("cache*/**/*.parquet"))
    manifest = {
        "study": "live2", "worlds": len(ws), "methods": list(METHODS), "rows": len(frame),
        "elapsed_seconds": round(time.monotonic() - started, 2), "processes": 4,
        "python": sys.version.split()[0], "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "gpu_inference": False,
        "inputs_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest(),
        "code_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sorted((ROOT / "src/lai").glob("*.py")))).hexdigest(),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
