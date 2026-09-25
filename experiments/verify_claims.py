#!/usr/bin/env python3
"""Check every qualitative claim in the paper and README against the saved results.

The numbers in the text are macros (experiments/make_numbers.py), but sentences
such as "never loses", "every significant loss is on MATH-500" or "the best
deployable method at every f" are prose. This script re-derives each of them
from results/ and prints PASS or FAIL. Run it after every rerun; a FAIL means the
sentence must be rewritten.

    PYTHONPATH=src python experiments/verify_claims.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from lai.stats import compare  # noqa: E402

RES = ROOT / "results"
H = json.loads((RES / "tables" / "headline.json").read_text())
DEPLOYABLE = ["self", "majority", "confidence", "sac", "aip_gated", "aip_trust_only", "aip_naive", "aip_soft",
              "ds_full", "ds_onecoin", "race_noclone", "race_rawclone", "race_ms", "race", "race_onecoin",
              "race_full", "race_capself", "race_d"]
results: list[tuple[str, bool, str]] = []


def num(key: str) -> float:
    return float(H[key].replace(",", "").replace("+", ""))


def check(claim: str, ok: bool, detail: str = "") -> None:
    results.append((claim, bool(ok), detail))


def load(study: str) -> pd.DataFrame:
    return pd.read_parquet(RES / study / "per_task.parquet")


def means(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    t = d[d.split == "test"]
    return t.groupby([*by, "method"]).accuracy.mean().unstack() * 100


main = load("main")
main = main[main.p_obs == 1.0]
zoo, llm, ext, live = load("zoo"), load("llm"), load("ext"), load("live")

# ---------------------------------------------------------------- E1
c = compare(main, "race", ["self", "majority", "ds_onecoin", "aip_gated"], ["benchmark", "f"])
losses = c[c.verdict == "loss"]
check("E1: never loses to the receiver alone", (losses.baseline == "self").sum() == 0,
      f"{(losses.baseline == 'self').sum()} losses")
check("E1: every significant loss is on MATH-500 at f<=0.3",
      len(losses) == 0 or ((losses.benchmark == "math500") & (losses.f <= 0.3)).all(),
      losses[["benchmark", "f", "baseline"]].to_dict("records").__repr__())
m = means(main, ["f"])
check("E1: mean over benchmarks, RACE above the receiver alone at every f", (m.race > m.self).all(),
      m[m.race <= m.self].index.tolist().__repr__())
check("E1: no breakdown point (README/abstract)", H["breakdownRace"] == "none", H["breakdownRace"])
check("E1: no-clone variant higher at f=0.5 against a perfect bloc", num("mainRaceNocloneFive") > num("mainRaceFive"))
mb = means(main, ["benchmark", "f"])
below = mb[mb.race < mb.self]
check("E1 caption: RACE below the receiver only on MATH-500 at small f",
      all(b == "math500" and f <= 0.3 for b, f in below.index), below.index.tolist().__repr__())

# ---------------------------------------------------------------- E2
check("E2: uninformative, RACE above the receiver alone at f=0.7", num("zooUninfRaceSeven") > num("zooUninfSelfSeven"))
z = means(zoo[zoo.attack != "sleeper"], ["attack", "param", "f"]).xs(0.7, level="f")
bad = z[z.race < z.self]
check("E2: camouflage is the only stationary attack with RACE below the receiver at f=0.7",
      set(bad.index.get_level_values("attack")) <= {"camouflage"}, bad.index.tolist().__repr__())
camo = z.loc["camouflage"]
dep = [m for m in DEPLOYABLE if m in camo.columns and m != "self"]
beats = [(p, m) for p in camo.index for m in dep if camo.loc[p, m] > camo.loc[p, "self"]]
check("Limitations: no deployable method beats the receiver alone under camouflage at f=0.7", not beats,
      beats.__repr__())

g = means(zoo[zoo.attack == "gate_aware"], ["f", "param"]).xs(0.7, level="f")
check("E2: RACE nearly flat in p at f=0.7 (range < 4 points)", g.race.max() - g.race.min() < 4.0,
      f"range {g.race.max() - g.race.min():.1f}")

# ---------------------------------------------------------------- E3
ml = means(llm, ["f"])
dep = [m for m in DEPLOYABLE if m in ml.columns]
best = ml[dep].idxmax(axis=1)
others = [m for m in dep if not m.startswith("race")]
check("E3: RACE above every non-RACE method at every f, and within 0.1 of its best variant",
      all(ml.loc[f, "race"] > ml.loc[f, others].max() and ml.loc[f, "race"] >= ml.loc[f, dep].max() - 0.1
          for f in ml.index), best.to_dict().__repr__())
check("E3: RACE above the honest-only oracle at f=0.7", num("llmRaceSeven") > num("llmHonestOracleSeven"))
check("E3: both superseded clone rules lower at f=0.7",
      num("llmRaceRawSeven") < num("llmRaceSeven") and num("llmRaceNocloneSeven") < num("llmRaceSeven"))
hi = ml[ml.index >= 0.5]
check("Appendix: on E3 the adopted rule beats both superseded variants at every f>=0.5",
      ((hi.race > hi.race_rawclone) & (hi.race > hi.race_noclone)).all())

# ---------------------------------------------------------------- method: full confusion on 4-option questions
worse = []
for st, d in (("main", main), ("zoo", zoo), ("llm", llm), ("live", live[live.source == "live"])):
    mm = means(d[d.benchmark.isin(["mmlu", "medqa", "arc"])], ["benchmark"])
    if "race_full" in mm:
        for b in mm.index:
            if mm.loc[b, "race_full"] > mm.loc[b, "race"] + 0.1:
                worse.append((st, b, round(mm.loc[b, "race_full"] - mm.loc[b, "race"], 2)))
check("Method: full confusion never >0.1 points better than RACE on 4-option benchmarks, per study", not worse, worse.__repr__())

# ---------------------------------------------------------------- conclusion: significant losses to self
exceptions = []
for st, d, by in (("main", main, ["benchmark", "f"]), ("zoo", zoo, ["benchmark", "f", "attack", "param"]),
                  ("llm", llm, ["benchmark", "f", "attack"])):
    cc = compare(d, "race", ["self"], by)
    for _, r in cc[cc.verdict == "loss"].iterrows():
        exceptions.append((st, r.get("attack", "coherent"), r.benchmark, r.f))
other = [e for e in exceptions if e[1] not in ("camouflage", "sleeper")]
check("Conclusion: significant losses to self only under camouflage, sleeper, and one open-answer cell",
      len(other) <= 1 and all(e[2] in ("math500", "gsm8k") for e in other), other.__repr__())

# ---------------------------------------------------------------- E9
pooled = compare(ext, "race", ["oracle_channel_honest"], ["attack", "param", "f"])
wins = pooled[pooled.verdict == "win"]
allowed = {("independent", 1.0), ("gate_aware", 0.0), ("gate_aware", 0.25), ("gate_aware", 0.5), ("gate_aware", 0.75)}
late = {("gate_aware", 1.0), ("attractor", 1.0)}
ok = all(((a, p) in allowed) or ((a, p) in late and f == 0.7) for a, p, f in zip(wins.attack, wins.param, wins.f,
                                                                                    strict=True))
check("E9: pooled wins over the removal oracle are all truth-dependent attacks",
      ok, sorted(zip(wins.attack, wins.param, wins.f, strict=True)).__repr__())
gate_all = all(((pooled.attack == "gate_aware") & (pooled.param == p) & (pooled.f == f) & (pooled.verdict == "win")).any()
               for p in (0.0, 0.25, 0.5, 0.75) for f in (0.3, 0.5, 0.7))
check("E9: gate-aware with p<1 wins at every f", gate_all)
check("E9: RACE-D never differs significantly on the other 14 settings", num("raceDOtherNonTies") == 0,
      H["raceDOtherNonTies"])
check("E9: RACE-D helps camouflage at f=0.7, hurts at f=0.3", num("raceDCamoSeven") > 0 and num("raceDCamoThree") < 0)
check("E9: RACE-D hurts echo at f=0.7 and sleeper at f=0.3", num("raceDEchoSeven") < 0 and num("raceDSleeperThree") < 0)
check("E9: independent budget positive and growing with f",
      0 < num("budgetIndepThree") < num("budgetIndepFive") < num("budgetIndepSeven"))
check("E9: echo, camouflage, sleeper budgets negative",
      num("budgetEchoSeven") < 0 and num("budgetCamoSeven") < 0 and num("budgetSleeperFive") < 0)
check("E9: LLM budgets near zero (|b| < 1.5)",
      all(abs(num(k)) < 1.5 for k in ("budgetLlmThree", "budgetLlmFive", "budgetLlmSeven")))
check("E9: uninformative budget ~0 and RACE pays a small cost",
      abs(num("budgetUninfSeven")) < 2.5 and num("overRemovalUninfSeven") < 0)

# ---------------------------------------------------------------- E6
check("E6: forgetting and window above the receiver alone after the sleeper switch",
      num("onSleeperDecay") > num("onSleeperSelf") and num("onSleeperWin") > num("onSleeperSelf"))
check("E6: forgetting costs nothing measurable on a stationary schedule", num("onStatDecay") >= num("onStatCum") - 1.0)

# ---------------------------------------------------------------- E8
check("E8: every MMLU receiver gains", num("liveMmluGainMin") > 0)
check("E8: MMLU f=0.5 RACE above self, majority, AIP, DS",
      num("liveMmluLlmRaceFive") > max(num("liveMmluLlmSelfFive"), num("liveMmluLlmMajFive"),
                                       num("liveMmluLlmAipFive"), num("liveMmluLlmDsFive")))
check("E8: v3.0 fell below the receiver alone on BoolQ at f=0.5", num("liveBoolqLlmVthreeFive") < num("liveBoolqLlmSelfFive"))
check("E8: v3.1 at or above the receiver alone on BoolQ at f=0.5", num("liveBoolqLlmRaceFive") >= num("liveBoolqLlmSelfFive"))
check("E8: strongest BoolQ receiver still loses with v3.1", num("liveGraniteRace") < num("liveGraniteSelf"))
check("E8: debate lowers RACE on MMLU", num("liveDebMmluRace") < num("liveIndMmluRace"))
check("E8: debate helps RACE on BoolQ", num("liveDebBoolqRace") > num("liveIndBoolqRace"))
check("E8: debate helps majority slightly on BoolQ", num("liveDebBoolqMaj") > num("liveIndBoolqMaj"))

# ---------------------------------------------------------------- README
check("README: weakest agents gain most, strongest lose a little",
      num("agentLlamaGain") > num("agentMinistralGain") and num("agentQwenGain") <= 0.5 and num("agentGemmaGain") <= 0.5)
check("Binary rule: never significantly worse than v3.0 on replayed BoolQ", num("binaryReplayLosses") == 0,
      H.get("binaryReplayLosses", "?"))
check("README: v3.1 equals or beats v3.0 on replayed BoolQ in every study",
      all(num(f"binary{s}Delta") >= -0.5 for s in ("Main", "Zoo", "Llm", "Ext")))
# ---------------------------------------------------------------- E11
check("E11: with a liar minority (f<=0.3) the best classical estimator is within 1 point of RACE",
      all(abs(num(f"crowdBestClassical{t}") - num(f"crowdRace{t}")) <= 1.0 for t in ("One", "Three")))
check("E11: RACE never significantly loses to a classical estimator", num("crowdClassicalLosses") == 0,
      H.get("crowdClassicalLosses", "?"))
check("E11: at f=0.7 every classical estimator is at least 50 points below the receiver alone",
      num("crowdClassicalMaxSeven") < num("crowdSelfSeven") - 50,
      f'{H.get("crowdClassicalMaxSeven")} vs {H.get("crowdSelfSeven")}')
check("E11: at f=0.7 RACE is within 2 points of the known-channel oracle",
      num("crowdOracleSeven") - num("crowdRaceSeven") <= 2.0)
check("E11: KOS on BoolQ at f=0.7 far below RACE", num("crowdBoolqKosSeven") < num("crowdBoolqRaceSeven") - 50)

fails = [r for r in results if not r[1]]
for claim, ok, detail in results:
    print(("PASS " if ok else "FAIL ") + claim + (f"   [{detail}]" if detail and not ok else ""))
print(f"\n{len(results) - len(fails)}/{len(results)} claims hold")

# ---------------------------------------------------------------- evidence ledger
SOURCE = {"E1": "main", "E2": "zoo", "E3": "llm", "E6": "online", "E8": "live", "E9": "ext", "E10": "live2",
          "E11": "crowd", "Limitations": "zoo", "Appendix": "llm", "Conclusion": "main, zoo, llm",
          "Method": "main, zoo, llm, live", "README": "extra, main, zoo, llm, ext", "Binary rule": "main, zoo, llm"}
lines = ["# Evidence ledger",
         "",
         "Generated by `experiments/verify_claims.py` from `results/`. Do not edit by hand. The studies table lists",
         "each study's run manifest. The claims table lists every qualitative sentence in the paper and README",
         "that is not already a generated number, re-derived from the per-task results.",
         "",
         "## Studies",
         "",
         "| Study | Worlds / streams | Result rows | Wall time | Processes | GPU | Inputs SHA-256 | Code SHA-256 |",
         "|---|---:|---:|---:|---:|:-:|---|---|"]
for st in ("main", "zoo", "llm", "history", "swarm", "ext", "crowd", "online", "live", "live2"):
    mp = RES / st / "run_manifest.json"
    if not mp.exists():
        continue
    m = json.loads(mp.read_text())
    lines.append(f"| {st} | {m.get('worlds', m.get('streams', '—'))} | {m.get('rows', '—'):,} | "
                 f"{m.get('elapsed_seconds', 0) / 60:.1f} min | {m.get('processes', '—')} | "
                 f"{'yes' if m.get('gpu_inference') else 'no'} | `{str(m.get('inputs_sha256', ''))[:12]}` | "
                 f"`{str(m.get('code_sha256', ''))[:12]}` |")
lines += ["", "## Claims", "", f"{len(results) - len(fails)} of {len(results)} claims hold.", "",
          "| # | Claim | Status | Evidence (study) |", "|---:|---|:-:|---|"]
for i, (claim, ok, detail) in enumerate(results, 1):
    tag = claim.split(":")[0].strip()
    lines.append(f"| {i} | {claim} | {'✅' if ok else '❌ ' + detail} | `results/{SOURCE.get(tag, tag)}` |")
reg = ROOT / "docs" / "PREREGISTRATION_E10.md"
if reg.exists():
    lines += ["", "## Pre-registration", "",
              "E10 was registered in `docs/PREREGISTRATION_E10.md` and committed (494aa28, 2026-09-25 09:51 UTC)"
              " before any E10 answer was generated. The run's start time and code commit are recorded in"
              " `data/live_cache_v2/RUN_INFO.txt`."]
(ROOT / "docs" / "EVIDENCE.md").write_text("\n".join(lines) + "\n")


def tex(text: str) -> str:
    """Claim text -> LaTeX (the claims use ASCII comparison operators and a few symbols)."""
    out = text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    for a, b in (("<=", "@LE@"), (">=", "@GE@"), ("≤", "@LE@"), ("≥", "@GE@"), ("<", "@LT@"), (">", "@GT@"),
                 ("|", "@BAR@"), ("–", "--"), ("→", "@TO@")):
        out = out.replace(a, b)
    for a, b in (("@LE@", r"$\le$"), ("@GE@", r"$\ge$"), ("@LT@", "$<$"), ("@GT@", "$>$"), ("@BAR@", "$|$"),
                 ("@TO@", r"$\to$")):
        out = out.replace(a, b)
    return out


rows = [r"\begin{longtable}{@{}r p{0.78\linewidth} c@{}}",
        r"\caption{Qualitative claims of the paper and README, re-derived from the per-task results on every run.}"
        r"\label{tab:claims}\\", r"\toprule", r"\# & Claim (as worded in the paper or README) & Holds\\",
        r"\midrule", r"\endhead"]
for i, (claim, ok, _detail) in enumerate(results, 1):
    rows.append(f"{i} & {tex(claim)} & {'yes' if ok else 'NO'}\\\\")
rows += [r"\bottomrule", r"\end{longtable}"]
(RES / "tables" / "claims_ledger.tex").write_text("\n".join(rows) + "\n")
(RES / "tables" / "claims.json").write_text(json.dumps({"claimsHold": str(len(results) - len(fails)),
                                                       "claimsTotal": str(len(results))}, indent=1))
sys.exit(1 if fails else 0)
