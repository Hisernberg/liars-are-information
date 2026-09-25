#!/usr/bin/env python3
"""Generate ``paper/numbers.tex`` and ``results/tables/headline.json``.

Every number quoted in the manuscript and README is produced here from the
saved study outputs. A missing study yields ``??`` so a stale draft is visible.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lai.stats import compare, summary  # noqa: E402

RES = ROOT / "results"
NUM: dict[str, str] = {}


DIGITS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"], strict=True))


def put(name: str, value, fmt: str = "{:.1f}") -> None:
    # LaTeX control words cannot contain digits: histRace80 -> histRaceEightZero.
    name = "".join(DIGITS.get(ch, ch) for ch in name)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        NUM[name] = "??"
    elif isinstance(value, str):
        NUM[name] = value
    else:
        NUM[name] = fmt.format(value)


def load(study: str) -> pd.DataFrame | None:
    p = RES / study / "per_task.parquet"
    return pd.read_parquet(p) if p.exists() else None


def pct(x: float) -> float:
    return 100.0 * float(x)


def main_numbers(main: pd.DataFrame) -> None:
    s = summary(main[main.p_obs == 1.0], ["benchmark", "f"])
    for f, tag in ((0.5, "Five"), (0.7, "Seven"), (0.9, "Nine")):
        mean = s.xs(f, level="f").mean()
        for m, mt in (("race", "Race"), ("aip_gated", "Aip"), ("majority", "Maj"), ("ds_onecoin", "Ds"),
                      ("self", "Self"), ("oracle_channel", "Oracle"), ("race_noclone", "RaceNoclone")):
            put(f"main{mt}{tag}", pct(mean[m]))
    for b, bt in (("boolq", "Boolq"), ("math500", "Math"), ("mmlu", "Mmlu"), ("medqa", "Medqa")):
        for f, tag in ((0.5, "Five"), (0.7, "Seven"), (0.1, "One")):
            row = s.loc[(b, f)]
            for m, mt in (("race", "Race"), ("aip_gated", "Aip"), ("self", "Self"), ("majority", "Maj")):
                put(f"main{bt}{mt}{tag}", pct(row[m]))
    # Breakdown: smallest f at which each method's mean accuracy falls below half the receiver's.
    mean_f = s.groupby(level="f").mean()
    for m, mt in (("majority", "Maj"), ("aip_gated", "Aip"), ("ds_onecoin", "Ds"), ("race", "Race")):
        bad = mean_f.index[mean_f[m] < 0.5 * mean_f["self"]]
        put(f"breakdown{mt}", f"{bad.min():g}" if len(bad) else "none")
    # Honest-only cost at f = 0.
    zero = s.xs(0.0, level="f").mean()
    put("mainRaceZero", pct(zero["race"]))
    put("mainAipZero", pct(zero["aip_gated"]))
    put("mainMajZero", pct(zero["majority"]))
    put("mainSelfZero", pct(zero["self"]))
    c = compare(main[main.p_obs == 1.0], "race", ["self", "majority", "ds_onecoin", "aip_gated"], ["benchmark", "f"])
    for b, bt in (("aip_gated", "Aip"), ("majority", "Maj"), ("self", "Self"), ("ds_onecoin", "Ds")):
        v = c[c.baseline == b].verdict.value_counts()
        put(f"mainWins{bt}", int(v.get("win", 0)), "{:d}")
        put(f"mainLosses{bt}", int(v.get("loss", 0)), "{:d}")
        put(f"mainCells{bt}", int(len(c[c.baseline == b])), "{:d}")


def zoo_numbers(zoo: pd.DataFrame) -> None:
    s = summary(zoo, ["f", "attack", "param"])
    ga = s.xs(0.7, level="f").xs("gate_aware", level="attack")
    low = ga.loc[[p for p in ga.index if p <= 0.5]].mean()
    put("zooGateRaceSeven", pct(low["race"]))
    put("zooGateAipSeven", pct(low["aip_gated"]))
    put("zooGateSoftSeven", pct(low["aip_soft"]))
    put("zooGateMajSeven", pct(low["majority"]))
    for f, tag in ((0.5, "Five"), (0.7, "Seven")):
        ind = s.loc[(f, "independent", 1.0)]
        put(f"zooIndepRace{tag}", pct(ind["race"]))
        put(f"zooIndepAip{tag}", pct(ind["aip_gated"]))
        camo = s.loc[(f, "camouflage")].mean()
        put(f"zooCamoRace{tag}", pct(camo["race"]))
        put(f"zooCamoCap{tag}", pct(camo["race_capself"]))
        put(f"zooCamoSelf{tag}", pct(camo["self"]))
        put(f"zooCamoAip{tag}", pct(camo["aip_gated"]))
        sl = s.loc[(f, "sleeper", 1.0)]
        put(f"zooSleeperRace{tag}", pct(sl["race"]))
        put(f"zooSleeperCap{tag}", pct(sl["race_capself"]))
        put(f"zooSleeperSelf{tag}", pct(sl["self"]))
        un = s.loc[(f, "uninformative", 1.0)]
        put(f"zooUninfRace{tag}", pct(un["race"]))
        put(f"zooUninfSelf{tag}", pct(un["self"]))
        put(f"zooUninfOracle{tag}", pct(un["oracle_honest_majority"]))
    camo7 = s.loc[(0.7, "camouflage")].mean()
    put("zooCamoRaceGap", pct(camo7["self"] - camo7["race"]))
    br = RES / "tables" / "best_response_accuracy_stationary.csv"
    if br.exists():
        t = pd.read_csv(br, index_col=0)
        for f, tag in ((0.5, "Five"), (0.7, "Seven")):
            row = t.loc[f]
            put(f"brRace{tag}", row["RACE (ours)"])
            put(f"brAip{tag}", row["AIP gated"])
            put(f"brMaj{tag}", row["Majority vote"])
            put(f"brSelf{tag}", row["Receiver alone"])
            put(f"brDs{tag}", row["Dawid–Skene (label-free)"])


def llm_numbers(llm: pd.DataFrame) -> None:
    s = summary(llm, ["f"])
    for f, tag in ((0.5, "Five"), (0.7, "Seven"), (0.3, "Three")):
        for m, mt in (("race", "Race"), ("aip_gated", "Aip"), ("majority", "Maj"), ("ds_onecoin", "Ds"),
                      ("self", "Self"), ("oracle_channel", "Oracle"), ("oracle_honest_majority", "HonestOracle"),
                      ("race_rawclone", "RaceRaw"), ("race_noclone", "RaceNoclone"), ("aip_soft", "Soft")):
            put(f"llm{mt}{tag}", pct(s.loc[f, m]))
    c = compare(llm, "race", ["self", "majority", "aip_gated", "aip_soft", "ds_onecoin"], ["benchmark", "f", "attack"])
    put("llmCells", int(len(c)), "{:d}")
    put("llmLosses", int((c.verdict == "loss").sum()), "{:d}")
    put("llmWins", int((c.verdict == "win").sum()), "{:d}")
    for b, bt in (("aip_gated", "Aip"), ("majority", "Maj"), ("self", "Self")):
        put(f"llmWins{bt}", int(((c.baseline == b) & (c.verdict == "win")).sum()), "{:d}")
    put("llmCellsPerBaseline", int((c.baseline == "self").sum()), "{:d}")
    if "race_onecoin" in set(llm.method):
        for f, tag in ((0.5, "Five"), (0.7, "Seven"), (0.3, "Three")):
            put(f"llmVthree{tag}", pct(s.loc[f, "race_onecoin"]))
        c0 = compare(llm, "race_onecoin", ["self", "majority", "aip_gated", "aip_soft", "ds_onecoin"],
                     ["benchmark", "f", "attack"])
        put("llmWinsVthree", int((c0.verdict == "win").sum()), "{:d}")
        put("llmLossesVthree", int((c0.verdict == "loss").sum()), "{:d}")


def binary_numbers() -> None:
    """RACE v3.1 (class-conditional channels on binary questions) vs v3.0 (one-coin) on BoolQ."""
    for st, tag in (("main", "Main"), ("zoo", "Zoo"), ("llm", "Llm"), ("ext", "Ext"), ("live", "Live")):
        d = load(st)
        if d is None or "race_onecoin" not in set(d.method):
            continue
        d = d[(d.split == "test") & (d.benchmark == "boolq")]
        if st == "live":
            d = d[(d.source == "live") & d.attack.str.startswith("llm:")]
        m = d.groupby("method").accuracy.mean()
        put(f"binary{tag}Race", pct(m["race"]))
        put(f"binary{tag}Vthree", pct(m["race_onecoin"]))
        put(f"binary{tag}Self", pct(m["self"]))
        put(f"binary{tag}Delta", pct(m["race"] - m["race_onecoin"]), "{:+.1f}")
    cells = nonties = wins = losses = 0
    for st, by in (("main", ["f", "p_obs"]), ("zoo", ["f", "attack", "param"]), ("llm", ["f", "attack"])):
        d = load(st)
        if d is None or "race_onecoin" not in set(d.method):
            continue
        c = compare(d[d.benchmark == "boolq"], "race", ["race_onecoin"], by)
        cells += len(c)
        nonties += int((c.verdict != "tie").sum())
        wins += int((c.verdict == "win").sum())
        losses += int((c.verdict == "loss").sum())
    put("binaryReplayCells", cells, "{:d}")
    put("binaryReplayNonTies", nonties, "{:d}")
    put("binaryReplayWins", wins, "{:d}")
    put("binaryReplayLosses", losses, "{:d}")


def agent_numbers() -> None:
    """Per honest model: accuracy alone vs pooled (E7 receiver-gain worlds, f >= 0.5)."""
    p = RES / "extra" / "receiver_gain.parquet"
    if not p.exists():
        return
    rg = pd.read_parquet(p)
    rg = rg[rg.f >= 0.5]
    t = rg.groupby("model")[["self_acc", "majority_acc", "aip_acc", "race_acc"]].mean()
    names = {"llama32_3b": "Llama", "phi4_mini_reasoning": "Phi", "granite42_30b": "Granite",
             "olmo3_32b_think": "Olmo", "ministral3_14b": "Ministral", "gemma4_31b": "Gemma", "qwen38_27b": "Qwen"}
    for m, nt in names.items():
        if m in t.index:
            row = t.loc[m]
            put(f"agent{nt}Self", pct(row.self_acc))
            put(f"agent{nt}Maj", pct(row.majority_acc))
            put(f"agent{nt}Aip", pct(row.aip_acc))
            put(f"agent{nt}Race", pct(row.race_acc))
            put(f"agent{nt}Gain", pct(row.race_acc - row.self_acc), "{:+.1f}")
    put("agentSpreadSelf", pct(t.self_acc.max() - t.self_acc.min()))
    put("agentSpreadRace", pct(t.race_acc.max() - t.race_acc.min()))


def history_numbers(h: pd.DataFrame) -> None:
    s = summary(h, ["attack", "f", "history_n"])
    for n in (5, 10, 20, 80):
        row = s.xs(n, level="history_n").xs(0.5, level="f").mean()
        put(f"histRace{n}", pct(row["race"]))
        put(f"histAip{n}", pct(row["aip_gated"]))


def extra_numbers() -> None:
    p = RES / "extra" / "channel_estimates.parquet"
    if not p.exists():
        return
    ch = pd.read_parquet(p)
    put("chanCorr", float(np.corrcoef(ch.a_true, ch.a_hat)[0, 1]), "{:.3f}")
    put("chanMae", float(np.mean(np.abs(ch.a_true - ch.a_hat))), "{:.3f}")
    byz, hon = ch[ch.byzantine], ch[~ch.byzantine]
    put("chanByzInvRace", pct((byz.race_decision == "invert").mean()))
    put("chanByzInvAip", pct((byz.aip_decision == "invert").mean()))
    put("chanByzTrustRace", pct((byz.race_decision == "trust").mean()))
    put("chanByzTrustAip", pct((byz.aip_decision == "trust").mean()))
    put("chanHonInvRace", pct((hon.race_decision == "invert").mean()))
    put("chanHonInvAip", pct((hon.aip_decision == "invert").mean()))
    put("chanHonTrustRace", pct((hon.race_decision == "trust").mean()))
    put("chanHonTrustAip", pct((hon.aip_decision == "trust").mean()))
    put("chanN", len(ch), "{:,}")
    p = RES / "extra" / "receiver_gain.parquet"
    if p.exists():
        rg = pd.read_parquet(p)
        rg = rg[rg.f > 0]
        weak = rg[rg.receiver_hist_acc < 0.6]
        strong = rg[rg.receiver_hist_acc >= 0.8]
        put("gainWeak", pct((weak.race_acc - weak.self_acc).mean()))
        put("gainStrong", pct((strong.race_acc - strong.self_acc).mean()))
        put("worseWeak", pct((weak.race_acc < weak.self_acc - 0.05).mean()))
        put("worseStrong", pct((strong.race_acc < strong.self_acc - 0.05).mean()))
        above = rg[rg.receiver_hist_acc > 0.5]
        below = rg[rg.receiver_hist_acc <= 0.5]
        for col, mt in (("race_acc", "Race"), ("aip_acc", "Aip"), ("majority_acc", "Maj")):
            put(f"riskAbove{mt}", pct((above[col] < above.self_acc - 0.05).mean()))
            put(f"riskBelow{mt}", pct((below[col] < below.self_acc - 0.05).mean()))


def ext_numbers(ext: pd.DataFrame) -> None:
    """E9: information budget (known-channel vs liar-removal oracle) and RACE-D."""
    test = ext[ext.split == "test"]
    cell = test.groupby(["attack", "param", "f", "benchmark", "seed", "method"]).accuracy.mean().unstack("method")
    m = cell.groupby(["attack", "param", "f"]).mean() * 100
    llm = m[m.index.get_level_values("attack").str.startswith("llm:")].groupby(level="f").mean()
    rows = {"Indep": ("independent", 1.0), "GateZero": ("gate_aware", 0.0), "Coh": ("gate_aware", 1.0),
            "Attr": ("attractor", 1.0), "Uninf": ("uninformative", 1.0), "Camo": ("camouflage", 0.9),
            "Echo": ("echo", 1.0), "Sleeper": ("sleeper", 1.0), "PartialHalf": ("partial", 0.5)}
    for f, tag in ((0.3, "Three"), (0.5, "Five"), (0.7, "Seven")):
        put(f"removal{tag}", m.xs(f, level="f").oracle_channel_honest.mean())
        for rt, (atk, prm) in rows.items():
            r = m.loc[(atk, prm, f)]
            put(f"budget{rt}{tag}", r.oracle_channel - r.oracle_channel_honest, "{:+.1f}")
            put(f"overRemoval{rt}{tag}", r.race - r.oracle_channel_honest, "{:+.1f}")
            put(f"raceD{rt}{tag}", r.race_d - r.race, "{:+.1f}")
            put(f"extRace{rt}{tag}", r.race)
            put(f"extSelf{rt}{tag}", r.self)
            put(f"extMaj{rt}{tag}", r.majority)
            put(f"extAip{rt}{tag}", r.aip_gated)
            put(f"extRemoval{rt}{tag}", r.oracle_channel_honest)
        r = llm.loc[f]
        put(f"budgetLlm{tag}", r.oracle_channel - r.oracle_channel_honest, "{:+.1f}")
        put(f"overRemovalLlm{tag}", r.race - r.oracle_channel_honest, "{:+.1f}")
        put(f"extLlmRace{tag}", r.race)
        put(f"extLlmSelf{tag}", r.self)
        put(f"extLlmMaj{tag}", r.majority)
        put(f"recoveryLlm{tag}", 100 * (r.race - r.self) / (r.oracle_channel_honest - r.self), "{:.0f}")
    # per-benchmark cells, Holm over all cells of the study
    c = compare(ext, "race", ["oracle_channel_honest"], ["benchmark", "f", "attack", "param"])
    llm_c = c[c.attack.str.startswith("llm:")]
    put("removalLlmCells", len(llm_c), "{:d}")
    put("removalLlmTies", int((llm_c.verdict == "tie").sum()), "{:d}")
    put("removalLlmLosses", int((llm_c.verdict == "loss").sum()), "{:d}")
    # pooled over benchmarks, per (attack, f)
    pooled = compare(ext, "race", ["oracle_channel_honest"], ["attack", "param", "f"])
    wins = pooled[pooled.verdict == "win"]
    put("pooledRemovalCells", len(pooled), "{:d}")
    put("pooledRemovalWins", len(wins), "{:d}")
    put("pooledRemovalWinAttacks", len(set(zip(wins.attack, wins.param, strict=True))), "{:d}")
    stationary = pooled[~pooled.attack.isin(["sleeper"])]
    put("pooledRemovalLosses", int((stationary.verdict == "loss").sum()), "{:d}")
    d = compare(ext, "race_d", ["race"], ["benchmark", "f", "attack", "param"])
    put("raceDCells", len(d), "{:d}")
    put("raceDWins", int((d.verdict == "win").sum()), "{:d}")
    put("raceDLosses", int((d.verdict == "loss").sum()), "{:d}")
    camo = d[d.attack == "camouflage"]
    put("raceDCamoWins", int((camo.verdict == "win").sum()), "{:d}")
    put("raceDCamoLosses", int((camo.verdict == "loss").sum()), "{:d}")
    put("raceDCamoCells", len(camo), "{:d}")
    other = d[~d.attack.isin(["camouflage", "echo", "sleeper"])]
    put("raceDOtherNonTies", int((other.verdict != "tie").sum()), "{:d}")
    put("raceDOtherCells", len(other), "{:d}")
    put("nWorldsExt", json.loads((RES / "ext" / "run_manifest.json").read_text())["worlds"], "{:,}")


def swarm_numbers(sw: pd.DataFrame) -> None:
    s = summary(sw[sw.n_agents == 10], ["composition", "f"])
    for comp, ct in (("hom_llama32_3b", "HomLlama"), ("hom_qwen38_27b", "HomQwen"), ("frozen+weak", "Weak")):
        if (comp, 0.5) in s.index:
            row = s.loc[(comp, 0.5)]
            put(f"swarm{ct}Race", pct(row["race"]))
            put(f"swarm{ct}Noclone", pct(row["race_noclone"]))
            put(f"swarm{ct}Aip", pct(row["aip_gated"]))
            put(f"swarm{ct}Self", pct(row["self"]))
    size = summary(sw[sw.composition == "frozen"], ["attack", "n_agents"])
    for atk, at in (("coherent", "Coh"), ("gate_aware", "Gate")):
        for n in (5, 40):
            if (atk, n) in size.index:
                put(f"size{at}Race{n}", pct(size.loc[(atk, n), "race"]))
                put(f"size{at}Aip{n}", pct(size.loc[(atk, n), "aip_gated"]))
    weak = summary(sw[(sw.n_agents == 10) & (sw.composition == "weak")], ["f"])
    if 0.5 in weak.index:
        put("swarmWeakRosterRace", pct(weak.loc[0.5, "race"]))
        put("swarmWeakRosterNoclone", pct(weak.loc[0.5, "race_noclone"]))
        put("swarmWeakRosterOracle", pct(weak.loc[0.5, "oracle_channel"]))


def online_numbers() -> None:
    p = RES / "online" / "per_step.parquet"
    if not p.exists():
        return
    on = pd.read_parquet(p)
    late = on[(on.schedule == "sleeper") & (on.pos >= 140)].groupby("method").correct.mean()
    for m, mt in (("race_fixed", "Fixed"), ("race_cumulative", "Cum"), ("race_decay0.97", "Decay"),
                  ("race_window64", "Win"), ("aip_gated_window64", "AipWin"), ("aip_gated_fixed", "AipFixed"),
                  ("self", "Self"), ("majority", "Maj")):
        if m in late:
            put(f"onSleeper{mt}", pct(late[m]))
    late7 = on[(on.schedule == "sleeper") & (on.pos >= 140) & (on.f == 0.7)].groupby("method").correct.mean()
    for m, mt in (("race_fixed", "Fixed"), ("race_decay0.97", "Decay"), ("race_window64", "Win"),
                  ("aip_gated_window64", "AipWin"), ("self", "Self"), ("majority", "Maj")):
        if m in late7:
            put(f"onSleeperSeven{mt}", pct(late7[m]))
    for sch, st in (("coherent_to_independent", "CohInd"), ("independent_to_coherent", "IndCoh"), ("toggle40", "Toggle")):
        sub = on[(on.schedule == sch) & (on.pos >= 120)].groupby("method").correct.mean()
        for m, mt in (("race_decay0.97", "Decay"), ("race_cumulative", "Cum"), ("aip_gated_window64", "AipWin"),
                      ("majority", "Maj"), ("self", "Self")):
            if m in sub:
                put(f"on{st}{mt}", pct(sub[m]))
    stat = on[on.schedule == "stationary_coherent"].groupby("method").correct.mean()
    for m, mt in (("race_fixed", "Fixed"), ("race_decay0.97", "Decay"), ("race_cumulative", "Cum"),
                  ("race_window64", "Win")):
        if m in stat:
            put(f"onStat{mt}", pct(stat[m]))


def live_numbers() -> None:
    p = RES / "live" / "role_stats.csv"
    if not p.exists():
        return
    rs = pd.read_csv(p)
    bench = (("mmlu", "Mmlu"), ("boolq", "Boolq"))
    for b, bt in bench:
        g = rs[rs.benchmark == b]
        for role, rt in (("honest", "Honest"), ("solo", "Solo"), ("rushing", "Rushing"), ("debate", "Debate")):
            put(f"live{bt}{rt}", pct(g[g.role == role].accuracy.mean()))
        h = g[g.role == "honest"].set_index("model").accuracy
        put(f"live{bt}HonestMin", pct(h.min()))
        put(f"live{bt}HonestMax", pct(h.max()))
    liars = rs[rs.role.isin(["solo", "rushing"])]
    put("liveBelowChance", pct((liars.truth_dependence < 0).mean()), "{:.0f}")
    put("liveMmluBelowChance", pct((liars[liars.benchmark == "mmlu"].truth_dependence < 0).mean()), "{:.0f}")
    put("liveBoolqBelowChance", pct((liars[liars.benchmark == "boolq"].truth_dependence < 0).mean()), "{:.0f}")
    ab = pd.read_csv(RES / "live" / "answer_bias.csv")
    bq = ab[(ab.benchmark == "boolq") & (ab.role == "honest")].set_index("model")
    put("liveBoolqGoldYes", pct(bq.gold_share_A.iloc[0]))
    put("liveSmolYes", pct(bq.loc["smollm2_1p7b", "share_A"]))
    put("liveSmolAccYes", pct(bq.loc["smollm2_1p7b", "acc_given_A"]))
    put("liveSmolAccNo", pct(bq.loc["smollm2_1p7b", "acc_given_B"]))
    lie = ab[(ab.benchmark == "boolq") & (ab.role == "solo")].set_index("model")
    put("liveGraniteLieYes", pct(lie.loc["granite33_2b", "share_A"]))
    ct = pd.read_csv(RES / "live" / "contagion.csv")
    put("liveSwitch", pct(ct.switch_rate.mean()))
    put("liveSwitchToLie", pct(ct.switch_to_liar_plurality.mean()))
    put("liveRightToWrong", pct(ct.right_to_wrong.mean()))
    put("liveWrongToRight", pct(ct.wrong_to_right.mean()))
    put("liveAccIndependent", pct(ct.acc_independent.mean()))
    put("liveAccDebate", pct(ct.acc_after_debate.mean()))
    live = load("live")
    if live is None:
        return
    test = live[live.split == "test"]
    ind = test[test.source == "live"].assign(fam=lambda d: np.where(d.attack == "coherent", "Coh", "Llm"))
    methods = (("race", "Race"), ("race_onecoin", "Vthree"), ("race_full", "Full"), ("majority", "Maj"), ("self", "Self"),
               ("aip_gated", "Aip"), ("ds_onecoin", "Ds"), ("oracle_channel", "Oracle"))
    fs = ((0.1, "One"), (0.3, "Three"), (0.5, "Five"), (0.7, "Seven"))
    t = ind.groupby(["benchmark", "fam", "f", "method"]).accuracy.mean()
    for b, bt in bench:
        for fam in ("Coh", "Llm"):
            for f, ft in fs:
                for m, mt in methods:
                    if (b, fam, f, m) in t.index:
                        put(f"live{bt}{fam}{mt}{ft}", pct(t[(b, fam, f, m)]))
    # debate vs independent on the matched attacks (f = 0.5, solo + collude)
    matched = test[(test.f == 0.5) & test.attack.isin(["llm:solo", "llm:collude"])]
    md = matched.groupby(["source", "benchmark", "method"]).accuracy.mean()
    for b, bt in bench:
        for m, mt in methods:
            for src, st in (("live", "Ind"), ("live_debate", "Deb")):
                if (src, b, m) in md.index:
                    put(f"live{st}{bt}{mt}", pct(md[(src, b, m)]))
    c = pd.read_csv(RES / "live" / "paired_comparisons.csv")
    put("liveCellsPerBaseline", int(c.groupby("baseline").size().iloc[0]), "{:d}")
    for base, bt in (("self", "Self"), ("majority", "Maj"), ("aip_gated", "Aip"), ("ds_onecoin", "Ds")):
        cb = c[c.baseline == base]
        put(f"liveWins{bt}", int((cb.verdict == "win").sum()), "{:d}")
        put(f"liveLosses{bt}", int((cb.verdict == "loss").sum()), "{:d}")
    put("liveCells", len(c), "{:d}")
    put("liveWins", int((c.verdict == "win").sum()), "{:d}")
    put("liveLosses", int((c.verdict == "loss").sum()), "{:d}")
    rec = pd.read_csv(RES / "live" / "receivers.csv").set_index(["benchmark", "model"])
    for b, bt in bench:
        r = rec.loc[b]
        gain = r.race - r.self
        put(f"live{bt}GainMin", pct(gain.min()))
        put(f"live{bt}GainMax", pct(gain.max()))
    put("liveGraniteSelf", pct(rec.loc[("boolq", "granite33_2b"), "self"]))
    put("liveGraniteRace", pct(rec.loc[("boolq", "granite33_2b"), "race"]))
    put("liveGraniteFull", pct(rec.loc[("boolq", "granite33_2b"), "race_full"]))
    if "race_onecoin" in rec.columns:
        put("liveGraniteVthree", pct(rec.loc[("boolq", "granite33_2b"), "race_onecoin"]))
        for b, bt in bench:
            gain0 = rec.loc[b].race_onecoin - rec.loc[b].self
            put(f"live{bt}VthreeGainMin", pct(gain0.min()))
            put(f"live{bt}VthreeGainMax", pct(gain0.max()))
    put("nLiveTestTasks", int(test[test.source == "live"].groupby("world_id").task.nunique().median()), "{:d}")


def world_counts() -> None:
    put("nLiveTasks", "240")
    total = 0
    for st in ("main", "zoo", "llm", "history", "swarm", "ext"):
        p = RES / st / "run_manifest.json"
        if p.exists():
            total += json.loads(p.read_text())["worlds"]
    put("nWorldsReplay", f"{total:,}")


def main() -> None:
    world_counts()
    for name, fn in (("main", main_numbers), ("zoo", zoo_numbers), ("llm", llm_numbers), ("history", history_numbers),
                     ("swarm", swarm_numbers), ("ext", ext_numbers)):
        frame = load(name)
        if frame is not None:
            fn(frame)
    extra_numbers()
    binary_numbers()
    agent_numbers()
    online_numbers()
    live_numbers()
    # Every result macro the manuscript uses must exist: undefined ones become a
    # visible "??" (and a warning), never a silent LaTeX error or a stale value.
    import re

    prefixes = ("main", "llm", "zoo", "br", "chan", "hist", "risk", "live", "on", "swarm", "size", "breakdown",
                "gain", "worse", "nWorlds", "nLive", "budget", "overRemoval", "raceD", "removal", "recovery",
                "pooledRemoval", "extLlm", "binary", "agent", "extRace", "extRemoval", "extSelf", "extMaj", "extAip")
    used = set()
    for tex in (ROOT / "paper").rglob("*.tex"):
        if tex.name == "numbers.tex":
            continue
        used |= set(re.findall(r"\\([A-Za-z]+)", tex.read_text()))
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "paper" / "main.tex").read_text()))
    missing = sorted(u for u in used if u.startswith(prefixes) and u not in NUM and u not in defined)
    for u in missing:
        NUM[u] = "??"
    if missing:
        print("WARNING undefined result macros:", ", ".join(missing))
    lines = ["% GENERATED by experiments/make_numbers.py -- do not edit by hand."]
    for k, v in sorted(NUM.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    (RES / "tables" / "headline.json").write_text(json.dumps(NUM, indent=1, sort_keys=True))
    print(f"{len(NUM)} numbers written")


if __name__ == "__main__":
    main()
