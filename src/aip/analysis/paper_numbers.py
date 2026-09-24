"""Every number the paper reports, computed once from the result parquets.

The paper may not contain a hand-typed number. Each quantity here becomes a
LaTeX macro in ``paper/numbers.tex`` via ``scripts/make_numbers.py``, and
``tests/test_paper.py`` fails the build if a bare decimal appears in any ``.tex``
file outside that generated one.

Values are computed from the same parquets, by the same expressions, that
produced ``results/claims.md``, so the ledger and the manuscript cannot disagree.
Each record carries the claim id it supports, which is what lets
``paper/claims_map.md`` be checked mechanically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aip.harness.rules import load_rules
from aip.tasks import roster

R = Path("results")
AGG = R / "aggregation"
ADV = R / "adversarial"
COR = R / "correlation"

DISCARD_METHODS = [
    "majority", "geometric_median", "sac_filter_refine", "coord_median",
    "trimmed_mean", "krum", "multi_krum", "dawid_skene",
]

#: The subset a real deployment can run. Krum, Multi-Krum and trimmed mean are
#: handed the true Byzantine count (``is_oracle`` in the sweep) and beat AIP on
#: two benchmarks, so the split has to be explicit wherever a "baselines
#: collapse" number is reported.
DEPLOYABLE_DISCARD = [
    "majority", "geometric_median", "coord_median", "dawid_skene",
    "sac_filter_refine",
]
ORACLE_DISCARD = ["krum", "multi_krum", "trimmed_mean"]

#: Ledger id for the real-LLM half of the split H2 claim (three primes).
H2_REAL = "H2" + chr(39) * 3

#: The evasion story is anchored here and nowhere else; see the block in
#: :func:`collect` for why f = 0.7 is not an admissible anchor.
EVASION_ANCHOR_F = 0.5

#: The coherence interval the residual attack occupies. Defined once here
#: because the figure and the appendix's resolution caveat must quote the
#: same window; scripts/fig_evasion_band.py imports it.
EVASION_BAND = (0.15, 0.40)


@dataclass(frozen=True)
class Num:
    """One reported quantity."""

    macro: str
    value: float
    fmt: str = "{:.3f}"
    claim: str = ""
    note: str = ""

    def rendered(self) -> str:
        if not np.isfinite(self.value):
            return r"\textsc{n/a}"
        return self.fmt.format(self.value)


def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _sym_ref(sweep: pd.DataFrame, benchmark: str, method: str) -> pd.DataFrame:
    return sweep[
        (sweep.p_obs == 1.0) & (sweep.topology == "complete")
        & (sweep.adversary == "always_wrong") & (sweep.parity == "matched@0.1")
        & (sweep.composition == "mix_all4") & (sweep.benchmark == benchmark)
        & (sweep.method == method)
    ]


def _adv_ref(adv: pd.DataFrame, benchmark: str, attack: str, method: str,
             fmin: float = 0.5) -> float:
    s = adv[
        (adv.p_obs == 1.0) & (adv.parity == "matched@0.1") & (adv.composition == "mix_all4")
        & (adv.benchmark == benchmark) & (adv.attack == attack) & (adv.method == method)
        & (adv.f >= fmin)
    ]
    return float(s["accuracy_point"].mean()) if len(s) else float("nan")


def _best_discard(adv: pd.DataFrame, benchmark: str, attack: str, fmin: float = 0.5) -> float:
    s = adv[
        (adv.p_obs == 1.0) & (adv.parity == "matched@0.1") & (adv.composition == "mix_all4")
        & (adv.benchmark == benchmark) & (adv.attack == attack)
        & (adv.method.isin(DISCARD_METHODS)) & (adv.f >= fmin)
    ]
    return float(s.groupby("f")["accuracy_point"].max().mean()) if len(s) else float("nan")


#: benchmark -> LaTeX macro fragment. Macro names cannot contain digits, so the
#: mapping is explicit rather than derived from the benchmark name.
#:
#: VERIFIED AGAINST THE ROSTER AT IMPORT. This is the ninth place in this project
#: where a three-benchmark assumption was hardcoded, and the eighth to fail on a
#: benchmark added later -- here as `KeyError: 'medqa'`, four hours into a batch
#: run, after every expensive phase had already succeeded. Asserting the map is
#: complete turns that into an import-time failure with a message that names the
#: missing benchmark.
BENCHMARK_TAGS: dict[str, str] = {
    "gsm8k": "Gsmk",
    "math500": "Mathfive",
    "mmlu": "Mmlu",
    "medqa": "Medqa",
    "boolq": "Boolq",
    "arc": "Arc",
}


def benchmark_tag(benchmark: str) -> str:
    """LaTeX macro fragment for a benchmark, or a clear error naming the gap."""
    try:
        return BENCHMARK_TAGS[benchmark]
    except KeyError:
        raise KeyError(
            f"no LaTeX macro tag for benchmark {benchmark!r}. Add it to "
            f"BENCHMARK_TAGS in aip.analysis.paper_numbers -- macro names cannot "
            f"contain digits, so it cannot be derived from the name."
        ) from None


def collect() -> list[Num]:  # noqa: C901 - a flat registry reads better than nesting
    out: list[Num] = []
    add = out.append

    sweep = _load(AGG / "sweep.parquet")
    adv = _load(ADV / "adversarial_sweep.parquet")
    chan = _load(ADV / "channel_stats.parquet")
    dec = _load(ADV / "channel_decisions.parquet")
    qsp = _load(ADV / "q_by_answer_space.parquet")
    burst = _load(ADV / "burst_windowed.parquet")
    bandit = _load(ADV / "bandit_log.parquet")
    pairs = _load(COR / "pairwise_estimates.parquet")
    neff = _load(COR / "effective_swarm_size.parquet")
    qpair = _load(COR / "q_pairwise.parquet")
    mc = _load(COR / "multiclass_validation.parquet")
    regret = _load(AGG / "minimax_regret.parquet")
    parity = _load(AGG / "parity_audit.parquet")
    delta = _load(AGG / "gate_version_delta.parquet")
    landscape = _load(AGG / "accuracy_landscape.parquet")
    gate = _load(ADV / "gate_aware_mitigations.parquet")
    sleeper = _load(ADV / "sleeper_e1.parquet")
    sybil = _load(ADV / "sybil_e2.parquet")
    nscale = _load(AGG / "n_scaling.parquet")
    nweak = _load(AGG / "n_scaling_weak.parquet")
    cost = _load(AGG / "cost_of_defence.parquet")
    infer = _load(AGG / "inference_cost.parquet")
    comp = _load(COR / "composition_neff.parquet")
    cpred = _load(COR / "composition_predictors.parquet")
    cauc = _load(COR / "confidence_auc.parquet")

    # -- H1/H2: the headline invariance claim ---------------------------
    if sweep is not None:
        for bench, tag in (("gsm8k", "Gsmk"), ("mmlu", "Mmlu"), ("math500", "Mathfive")):
            s = _sym_ref(sweep, bench, "aip_gated")
            if s.empty:
                continue
            lo, hi = float(s.accuracy_point.min()), float(s.accuracy_point.max())
            claim = {"gsm8k": "H1-gsm8k", "mmlu": "H1-mmlu", "math500": "H1-math500"}[bench]
            add(Num(f"num{tag}AipMin", lo, claim=claim))
            add(Num(f"num{tag}AipMax", hi, claim=claim))
            add(Num(f"num{tag}AipSpread", hi - lo, claim=claim))
            f7 = s[s.f == 0.7]["accuracy_point"]
            if len(f7):
                add(Num(f"num{tag}AipFseven", float(f7.iloc[0]), claim=claim))
            f0 = s[s.f == 0.0]["accuracy_point"]
            if len(f0):
                add(Num(f"num{tag}AipFzero", float(f0.iloc[0]), claim=claim))
        coll = sweep[
            (sweep.p_obs == 1.0) & (sweep.topology == "complete")
            & (sweep.adversary == "always_wrong") & (sweep.parity == "matched@0.1")
            & (sweep.composition == "mix_all4") & (sweep.f == 0.7)
            & (sweep.method.isin(["majority", "geometric_median", "trimmed_mean",
                                  "coord_median"]))
        ]
        if len(coll):
            add(Num("numDiscardCollapseFseven", float(coll.accuracy_point.max()),
                    claim="H2'"))
        # Task 6 split: the "reaches exactly zero" wording is true only for the
        # symbolic adversary AND only for deployable methods. Both exceptions
        # get their own macro so the prose cannot quietly drop them.
        for bench, tag in (("boolq", "Boolq"), ("medqa", "Medqa")):
            orc = sweep[
                (sweep.p_obs == 1.0) & (sweep.topology == "complete")
                & (sweep.adversary == "always_wrong") & (sweep.parity == "matched@0.1")
                & (sweep.composition == "mix_all4") & (sweep.f == 0.7)
                & (sweep.benchmark == bench) & (sweep.method.isin(ORACLE_DISCARD))
            ]
            if len(orc):
                add(Num(f"numDiscardOracle{tag}Fseven", float(orc.accuracy_point.max()),
                        claim="H2''"))

    # -- H2 (real-LLM regime): nothing reaches zero ----------------------
    if adv is not None:
        real = adv[
            (adv.p_obs == 1.0) & (adv.parity == "matched@0.1")
            & (adv.composition == "mix_all4") & (adv.attack == "semantic_negation")
            & (adv.f == 0.7)
        ]
        span = real[real.method.isin(DEPLOYABLE_DISCARD)]["accuracy_point"]
        if len(span):
            add(Num("numRealDiscardMinFseven", float(span.min()), claim=H2_REAL))
            add(Num("numRealDiscardMaxFseven", float(span.max()), claim=H2_REAL))
        for bench, tag in BENCHMARK_TAGS.items():
            g = real[real.benchmark == bench]
            aipg = g[g.method == "aip_gated"]["accuracy_point"]
            dep = g[g.method.isin(DEPLOYABLE_DISCARD)]["accuracy_point"]
            if len(aipg) and len(dep):
                add(Num(f"numRealSep{tag}Fseven", float(aipg.iloc[0]) - float(dep.max()),
                        fmt="{:+.3f}", claim=H2_REAL))

    # -- G2': the direction the original wording had backwards ------------
    if qsp is not None:
        for cls, tag in (("binary", "Binary"), ("multiple_choice", "Mc"),
                         ("open_text", "OpenText"), ("open_numeric", "OpenNumeric")):
            g = qsp[qsp.answer_space_class == cls]["q_hat_adversary"]
            if len(g):
                add(Num(f"numQhat{tag}Mean", float(g.mean()), claim="G2'"))
        # p_obs constraint (limitations)
        p = sweep[
            (sweep.topology == "complete") & (sweep.adversary == "always_wrong")
            & (sweep.parity == "matched@0.1") & (sweep.composition == "mix_all4")
            & (sweep.method == "aip_gated") & (sweep.benchmark == "gsm8k") & (sweep.f == 0.6)
        ]
        for level, tag in ((1.0, "Full"), (0.5, "Half")):
            v = p[p.p_obs == level]["accuracy_point"]
            if len(v):
                add(Num(f"numPobs{tag}GsmkFsix", float(v.iloc[0]), note="p_obs limitation"))

    # -- I1/I2: invertibility spectrum ----------------------------------
    if chan is not None and adv is not None:
        for attack, tag, claim in (("semantic_negation", "Negation", "I1"),
                                   ("semantic_hallucination", "Hallucination", "I2")):
            c = chan[(chan.benchmark == "gsm8k") & (chan.attack == attack)]
            if len(c):
                add(Num(f"numGsmk{tag}Q", float(c.coherence.iloc[0]), claim=claim))
            gain = (_adv_ref(adv, "gsm8k", attack, "aip_gated")
                    - _best_discard(adv, "gsm8k", attack))
            add(Num(f"numGsmk{tag}Gain", gain, fmt="{:+.3f}", claim=claim))

    # -- G1/G2: the gate ------------------------------------------------
    if dec is not None:
        add(Num("numHonestInvertedMax",
                float(dec[dec.channel == "honest"]["pct_inverted"].max()),
                fmt="{:.1f}", claim="G1"))
        add(Num("numAdversaryInvertedMean",
                float(dec[(dec.channel == "adversary") & (dec.f >= 0.5)]["pct_inverted"].mean()),
                fmt="{:.1f}", claim="G1"))
    if qsp is not None:
        for bench, tag in (("mmlu", "Mmlu"), ("gsm8k", "Gsmk")):
            s = qsp[(qsp.benchmark == bench) & (qsp.attack == "semantic_negation")]
            if len(s):
                add(Num(f"num{tag}NegQhat", float(s.q_hat_adversary.iloc[0]), claim="G2"))
                add(Num(f"num{tag}HonestFloor",
                        float(s.honest_q_receiver_conditioned.iloc[0]), claim="G2"))

    # -- C1..C4: correlation and theory ---------------------------------
    if pairs is not None:
        w = float(pairs[pairs.kind == "within_model"]["phi_point"].mean())
        c = float(pairs[(pairs.kind == "cross_model")
                        & (pairs.benchmark == "gsm8k")]["phi_point"].mean())
        add(Num("numWithinModelPhi", w, claim="C1"))
        add(Num("numCrossModelPhi", c, claim="C1"))
        add(Num("numPhiRatio", w / c, fmt="{:.2f}", claim="C1"))
    if neff is not None:
        # BASIS RESTRICTION. Only GSM8K carries a measured within-model rho; on
        # every other benchmark the heterogeneous rows are computed from
        # cross-model rho alone and are labelled an *upper bound* on N_eff. The
        # unrestricted maximum was 6.86, an upper bound from ARC -- a demoted
        # benchmark, on a different estimator. Comparing it with a homogeneous
        # number computed the other way is not a comparison.
        measured = neff[~neff.basis.str.contains("upper bound", na=False)]
        if measured.empty:
            raise ValueError("effective_swarm_size.parquet has no measured-basis rows")
        benches = sorted(measured.benchmark.unique())
        hom = measured[measured.kind == "homogeneous"]["n_eff"]
        het = measured[measured.kind == "heterogeneous"]["n_eff"]
        note = f"measured within-model basis only ({', '.join(benches)})"
        add(Num("numNeffHomMin", float(hom.min()), fmt="{:.2f}", claim="C2", note=note))
        add(Num("numNeffHomMax", float(hom.max()), fmt="{:.2f}", claim="C2", note=note))
        add(Num("numNeffHomMean", float(hom.mean()), fmt="{:.2f}", claim="C2", note=note))
        add(Num("numNeffHetMax", float(het.max()), fmt="{:.2f}", claim="C2", note=note))
        bound = neff[neff.basis.str.contains("upper bound", na=False)]["n_eff"]
        if bound.notna().any():
            add(Num("numNeffUpperBoundMax", float(bound.max()), fmt="{:.2f}", claim="C2",
                    note="cross-model basis: an UPPER BOUND, not comparable with the "
                         "measured numbers above"))
    if qpair is not None:
        add(Num("numHonestQMmlu",
                float(qpair[qpair.benchmark == "mmlu"]["q_true"].mean()), claim="C3"))
        add(Num("numHonestQMathfive",
                float(qpair[qpair.benchmark == "math500"]["q_true"].mean()), claim="C3"))
        add(Num("numMmluChanceQ", 1.0 / 3.0, claim="C3"))
    if mc is not None:
        for r in mc.itertuples():
            tag = benchmark_tag(r.benchmark)
            add(Num(f"numMae{tag}Binary", float(r.mae_product_binary_point), claim="C4"))
            add(Num(f"numMae{tag}Mc", float(r.mae_product_mc_point), claim="C4"))

    # -- F1: falsified confidence ---------------------------------------
    if adv is not None:
        for bench, tag in (("gsm8k", "Gsmk"), ("mmlu", "Mmlu")):
            base = _adv_ref(adv, bench, "semantic_negation", "aip_gated", 0.3)
            att = _adv_ref(adv, bench, "falsified_confidence", "aip_gated", 0.3)
            sb = _adv_ref(adv, bench, "semantic_negation", "confidence_weighted_selfreport", 0.3)
            sa = _adv_ref(adv, bench, "falsified_confidence",
                          "confidence_weighted_selfreport", 0.3)
            add(Num(f"numFalsified{tag}Aip", att - base, fmt="{:+.3f}", claim="F1"))
            add(Num(f"numFalsified{tag}Selfreport", sa - sb, fmt="{:+.3f}", claim="F1"))

    # -- B1: burst and the windowed variant -----------------------------
    if burst is not None:
        b = burst[burst.attack == "burst"]
        static = float(b[b.method == "aip_gated"]["accuracy_point"].mean())
        add(Num("numBurstStatic", static, claim="B1"))
        # LaTeX macro names cannot contain digits, so window sizes are spelled.
        spell = {5: "Five", 10: "Ten", 20: "Twenty"}
        for w in sorted(int(x) for x in burst.window.unique() if x):
            v = float(b[b.method == f"aip_gated_w{w}"]["accuracy_point"].mean())
            word = spell.get(w, str(w))
            add(Num(f"numBurstWindow{word}", v, claim="B1"))
            add(Num(f"numBurstRecovered{word}", v - static, fmt="{:+.3f}", claim="B1"))

    # -- D1/D2: deterrence ----------------------------------------------
    if bandit is not None:
        for defence, tag in (("aip_gated", "Aip"), ("sac_filter_refine", "Sac")):
            g = bandit[bandit.defence == defence]
            if g.empty:
                continue
            last = g.sort_values("epoch").iloc[-1]
            add(Num(f"numBandit{tag}ValueCoherent",
                    float(last.value_coherent_lie), claim="D1"))
            add(Num(f"numBandit{tag}ValueNoise", float(last.value_noise), claim="D1"))
            add(Num(f"numBandit{tag}Accuracy",
                    float(g.swarm_accuracy.mean()), claim="D1"))

    # -- M1/M2: methodology ---------------------------------------------
    if parity is not None:
        s = parity.assign(a=parity.effect.abs()).groupby("method")["a"].mean()
        unweighted = ["majority", "geometric_median", "coord_median", "trimmed_mean",
                      "krum", "multi_krum", "dawid_skene"]
        present = [m for m in unweighted if m in s.index]
        add(Num("numParityUnweightedMax", float(s[present].max()) if present else np.nan,
                fmt="{:.4f}", claim="M2a"))
        add(Num("numParityAipGated", float(s.get("aip_gated", np.nan)),
                fmt="{:.4f}", claim="M2b",
                note="mean |effect| over all f; inert away from the tie, large at f=0.5"))
        # Task 6: the mean above hides the whole effect. The tie regime gets its
        # own macros so no sentence can report the average and stop there.
        tie = parity[(parity.method == "aip_gated") & (parity.f == 0.5)]
        off = parity[(parity.method == "aip_gated") & (parity.f != 0.5)]
        if len(tie) and len(off):
            add(Num("numParityAipTieMax", float(tie.effect.abs().max()), claim="M2b"))
            add(Num("numParityAipTieMean", float(tie.effect.abs().mean()), claim="M2b"))
            add(Num("numParityAipOffTieMax", float(off.effect.abs().max()), claim="M2b"))
    if regret is not None:
        for bench, tag in (("gsm8k", "Gsmk"), ("mmlu", "Mmlu"), ("math500", "Mathfive")):
            g = regret[regret.benchmark == bench]
            if g.empty:
                continue
            aipg = g[g.method == "aip_gated"]["minimax_regret"]
            if len(aipg):
                add(Num(f"numRegret{tag}Aip", float(aipg.iloc[0]), claim="M1"))
            others = g[g.method != "aip_gated"].sort_values("minimax_regret")
            if len(others):
                add(Num(f"numRegret{tag}Next", float(others.iloc[0]["minimax_regret"]),
                        claim="M1"))

    # -- reproducibility / limitations ----------------------------------
    if delta is not None:
        non_aip = delta[~delta.method.str.startswith("aip")]
        add(Num("numSeedNoiseFloor", float(non_aip.delta.abs().quantile(0.95)),
                note="Phase E consistency pass; 95th pct measurement floor"))
        add(Num("numSeedNoiseMedian", float(non_aip.delta.abs().median()),
                fmt="{:.4f}", note="Phase E consistency pass; median seed effect"))
        add(Num("numSeedNoiseChanged", float((non_aip.delta.abs() > 1e-9).sum()),
                fmt="{:.0f}", note="non-AIP cells differing between seeding regimes"))
        add(Num("numSeedNoiseTotal", float(len(non_aip)), fmt="{:.0f}",
                note="non-AIP cells compared"))
        ref = delta[(delta.p_obs == 1.0) & (delta.topology == "complete")
                    & (delta.adversary == "always_wrong") & (delta.parity == "matched@0.1")
                    & (delta.composition == "mix_all4") & (delta.method == "aip_gated")]
        for bench, tag in (("gsm8k", "Gsmk"), ("mmlu", "Mmlu"), ("math500", "Mathfive")):
            s = ref[ref.benchmark == bench]
            if len(s):
                add(Num(f"numGateDelta{tag}", float(s.delta.abs().max()),
                        note="gate v1 vs v2 max |delta|"))
    if landscape is not None:
        add(Num("numModelAccMin", float(landscape.accuracy.min()), note="accuracy landscape"))
        add(Num("numModelAccMax", float(landscape.accuracy.max()), note="accuracy landscape"))

    # -- G1/D1/D2/D3: the evasion story, anchored at f = 0.5 --------------
    #
    # ANCHOR. Every number below is read off the f = 0.5 rows and nowhere else.
    # At f = 0.7 the Byzantine bloc is a numerical majority, so a collapse
    # cannot be attributed to the gate -- the adversary wins by counting. The
    # 0.92 attacker gain this project previously reported was the f = 0.7 MMLU
    # cell and is void; the anchored value is MedQA's 0.796.
    if gate is not None:
        anchor = gate[gate.f == EVASION_ANCHOR_F]
        if anchor.empty:
            raise ValueError("gate_aware_mitigations.parquet has no f=0.5 rows")
        add(Num("numEvasionAnchorF", EVASION_ANCHOR_F, fmt="{:.1f}", claim="G1",
                note="every evasion number is read off this f and no other"))
        hard = anchor[anchor.method == "aip_gated"]
        for bench, tag in BENCHMARK_TAGS.items():
            g = hard[hard.benchmark == bench]
            if g.empty:
                continue
            ceiling = float(g.ceiling.iloc[0])
            coherent = g[g.q_realised == 1.0]["accuracy_point"]
            under = g[g.q_realised < ceiling].sort_values("accuracy_point")
            if under.empty or coherent.empty:
                continue
            best = under.iloc[0]
            add(Num(f"numEvasionQ{tag}", float(best.q_realised), claim="D3"))
            add(Num(f"numEvasionAcc{tag}", float(best.accuracy_point), claim="G1"))
            add(Num(f"numEvasionCoherent{tag}", float(coherent.iloc[0]), claim="G1"))
            add(Num(f"numEvasionGain{tag}",
                    float(coherent.iloc[0]) - float(best.accuracy_point),
                    fmt="{:+.3f}", claim="G1"))
        # Resolution of the grid in the interval the attack occupies. This is a
        # limitation, so it is a generated number rather than a sentence.
        # Scoped to the closed-label (C = 4) benchmarks. Pooling MATH-500's
        # open-text grid (q = p^2) with the multiple-choice grid
        # (q = p^2 + (1-p)^2/(C-1)) doubles the apparent resolution and hides
        # the fact that the two families cannot reach the same q values.
        lo, hi = EVASION_BAND
        closed = sorted({b for b in anchor.benchmark.unique() if roster.n_options(b)})
        closed_q = anchor[anchor.benchmark.isin(closed)]["q_realised"]
        band_q = sorted({round(float(q), 6) for q in closed_q if lo <= float(q) <= hi})
        add(Num("numEvasionQDistinct", float(len(band_q)), fmt="{:.0f}", claim="D3",
                note="distinct realised q in [0.15, 0.40] on C=4 -- the anchor's "
                     "resolution"))
        add(Num("numEvasionBandLo", lo, fmt="{:.2f}", claim="D3"))
        add(Num("numEvasionBandHi", hi, fmt="{:.2f}", claim="D3"))
        add(Num("numEvasionQFloorMc", float(min(closed_q)) if len(closed_q) else np.nan,
                claim="D3", note="lowest q this parameterisation can reach on C=4"))
        if closed:
            n_opt = roster.n_options(closed[0])
            add(Num("numEvasionChance", 1.0 / (n_opt - 1), claim="D3",
                    note="1/(C-1), the coherence of independent wrong answers"))
        # Mitigations, at the anchor and restricted to the evasion band.
        variants = (("aip_gated", "Hard"), ("aip_gated_soft6", "Soft"),
                    ("aip_gated_rand0.3", "Rand"), ("aip_gated_soft6_rand0.3", "Both"))
        for method, tag in variants:
            whole = anchor[anchor.method == method]["accuracy_point"]
            band = anchor[(anchor.method == method) & anchor.below_ceiling]["accuracy_point"]
            if len(whole):
                add(Num(f"numMitig{tag}Mean", float(whole.mean()), claim="D1"))
            if len(band):
                add(Num(f"numMitig{tag}BandMean", float(band.mean()), claim="D1"))
                add(Num(f"numMitig{tag}BandWorst", float(band.min()), claim="D1"))
        # Deltas against the hard gate. D2's verdict is a difference, and the
        # manuscript must not hand-type it.
        hard_all = anchor[anchor.method == "aip_gated"]["accuracy_point"]
        hard_band = anchor[(anchor.method == "aip_gated") & anchor.below_ceiling][
            "accuracy_point"]
        for method, tag in variants[1:]:
            w = anchor[anchor.method == method]["accuracy_point"]
            b = anchor[(anchor.method == method) & anchor.below_ceiling]["accuracy_point"]
            if len(w) and len(hard_all):
                add(Num(f"numMitig{tag}Delta", float(w.mean() - hard_all.mean()),
                        fmt="{:+.3f}", claim="D2"))
            if len(b) and len(hard_band):
                add(Num(f"numMitig{tag}BandDelta", float(b.mean() - hard_band.mean()),
                        fmt="{:+.3f}", claim="D2"))

    # -- attack-class table: measured coherence per class ------------------
    if chan is not None:
        tags = {"always_wrong": "AlwaysWrong", "semantic_negation": "Negation",
                "semantic_hallucination": "Hallucination", "rushing": "Rushing",
                "falsified_confidence": "Falsified", "burst": "Burst"}
        ref = chan[chan.benchmark == "gsm8k"]
        for attack, tag in tags.items():
            g = ref[ref.attack == attack]
            if g.empty:
                continue
            add(Num(f"numAttackQ{tag}", float(g.coherence.iloc[0]), claim="I1",
                    note="measured channel coherence on GSM8K"))
            add(Num(f"numAttackErr{tag}", float(g.error_rate.iloc[0]), claim="I1",
                    note="measured channel error rate on GSM8K"))

    # -- H1 under the real-LLM coherent adversary -------------------------
    # The symbolic-sweep spreads above are the claim's original evidence. Under
    # a real prompt-injected coherent adversary the same quantity is a fall, not
    # a spread, and on four of six benchmarks it clears the floor by a wide
    # margin. Both are emitted; the paper reports both.
    if adv is not None:
        ref = adv[
            (adv.method == "aip_gated") & (adv.composition == "mix_all4")
            & (adv.p_obs == 1.0) & (adv.parity == "matched@0.1")
            & (adv.attack == "semantic_negation")
        ]
        for bench, tag in BENCHMARK_TAGS.items():
            g = ref[ref.benchmark == bench].set_index("f")["accuracy_point"]
            if 0.0 in g.index and 0.7 in g.index:
                add(Num(f"numHoneFall{tag}", float(g[0.7] - g[0.0]), fmt="{:+.3f}",
                        claim="H1", note="aip_gated at f=0.7 minus f=0, real-LLM "
                                         "coherent adversary"))

    # -- I1/I2/I3/N1: the inversion ablation and how it scales ------------
    def _gain(frame: pd.DataFrame, f: float, benches: list[str]) -> pd.Series:
        s = frame[(frame.f == f) & frame.benchmark.isin(benches)]
        gated = s[s.method == "aip_gated"].groupby("n_agents")["accuracy_point"].mean()
        trust = s[s.method == "aip_trust_only"].groupby("n_agents")["accuracy_point"].mean()
        return (gated - trust).sort_index()

    if nscale is not None:
        strong_benches = sorted(nscale.benchmark.unique())
        g7 = _gain(nscale, 0.7, strong_benches)
        g0 = _gain(nscale, 0.0, strong_benches)
        add(Num("numInversionGainFseven", float(g7.mean()), claim="I1"))
        add(Num("numInversionCostFzero", float(g0.mean()), fmt="{:.3f}", claim="I2"))
        spell = {10: "Ten", 20: "Twenty", 50: "Fifty"}
        for n, value in g7.items():
            add(Num(f"numInversionGainN{spell.get(int(n), int(n))}", float(value),
                    claim="N1"))
        add(Num("numInversionGainNRange", float(g7.max() - g7.min()), claim="N1",
                note="range across N = 10/20/50; compare with the floor before "
                     "calling it flat OR growing"))
        # I3: the weak arm, on the SAME benchmarks. Pooling the weak arm's six
        # benchmarks against the strong roster's three compares different task
        # sets and made a 0.165 difference look like 0.089.
        if nweak is not None:
            w7 = _gain(nweak, 0.7, strong_benches)
            add(Num("numInversionGainWeak", float(w7.mean()), claim="I3",
                    note=f"matched benchmark set: {', '.join(strong_benches)}"))
            add(Num("numInversionGainCompetenceDiff",
                    float(g7.mean() - w7.mean()), fmt="{:+.3f}", claim="I3",
                    note="strong roster minus weak arm, matched benchmarks"))
            all_w = _gain(nweak, 0.7, sorted(nweak.benchmark.unique()))
            add(Num("numInversionGainWeakAllBenches", float(all_w.mean()), claim="I3",
                    note="weak arm over ALL six of its benchmarks -- NOT comparable "
                         "with the strong roster's three"))

    # -- E2: the Sybil bloc ----------------------------------------------
    if sybil is not None:
        big = sybil[sybil.sybil_size == sybil.sybil_size.max()]
        for bloc, tag in (("coherent", "Coherent"), ("disagreeing", "Disagreeing")):
            g = big[big.bloc == bloc]
            if g.empty:
                continue
            # Average over benchmarks first, then take the best method: the
            # comparison is between defenders, not between lucky cells.
            by_method = g.groupby("method")["accuracy_point"].mean()
            if "aip_gated_w20" in by_method.index:
                add(Num(f"numSybil{tag}Aip", float(by_method["aip_gated_w20"]),
                        claim="E2a"))
            others = by_method[~by_method.index.str.startswith("aip")]
            if len(others):
                add(Num(f"numSybil{tag}Baselines", float(others.max()), claim="E2a",
                        note=f"best non-AIP defender ({others.idxmax()}) at the "
                             "largest bloc, averaged over benchmarks"))
        add(Num("numSybilSizeMax", float(sybil.sybil_size.max()), fmt="{:.0f}",
                claim="E2a"))

    # -- E3: cost of defence (Task 9a) ------------------------------------
    if cost is not None:
        spell = {"aip_gated": "AipGated", "aip_trust_only": "AipTrustOnly",
                 "krum": "Krum", "multi_krum": "MultiKrum",
                 "trimmed_mean": "TrimmedMean", "majority": "Majority",
                 "geometric_median": "GeometricMedian", "coord_median": "CoordMedian"}
        for method, tag in spell.items():
            g = cost[cost.method == method]["us_per_decision_mean"]
            if len(g):
                add(Num(f"numCost{tag}", float(g.mean()), fmt="{:.1f}", claim="E3",
                        note="microseconds per receiver-decision, mean over "
                             "headline benchmarks"))
        gated = cost[cost.method == "aip_gated"]["us_per_decision_mean"].mean()
        trust = cost[cost.method == "aip_trust_only"]["us_per_decision_mean"].mean()
        if np.isfinite(gated) and np.isfinite(trust):
            add(Num("numCostGateOverhead", float(gated - trust), fmt="{:+.1f}",
                    claim="E3", note="what the inversion machinery adds over the "
                                     "trust-only ablation"))
            add(Num("numCostGateOverheadPct", 100.0 * float(gated - trust) / float(trust),
                    fmt="{:.1f}", claim="E3"))
        add(Num("numCostSlowest", float(cost.us_per_decision_mean.max()), fmt="{:.0f}",
                claim="E3", note="slowest aggregation rule measured"))
    if infer is not None:
        add(Num("numCompletionTokensMean", float(infer.completion_tokens_mean.mean()),
                fmt="{:.0f}", claim="E3", note="mean completion length, from the cache"))
        if cost is not None:
            slowest = float(cost.us_per_decision_mean.max())
            tokens = float(infer.completion_tokens_mean.mean())
            add(Num("numCostInferenceRatio", tokens / 100.0 * 1e6 / slowest,
                    fmt="{:.0f}", claim="E3",
                    note="generation is this many times the slowest aggregation "
                         "cost, at a generous 100 tokens/s"))

    # -- composition guidance (Task 9b) -----------------------------------
    if comp is not None:
        gs3 = comp[(comp.benchmark == "gsm8k") & (comp.k == 3)]
        if not gs3.empty:
            add(Num("numCompNeffBest", float(gs3.n_eff.max()), fmt="{:.2f}", claim="S1"))
            add(Num("numCompNeffWorst", float(gs3.n_eff.min()), fmt="{:.2f}", claim="S1"))
        for bench, tag in BENCHMARK_TAGS.items():
            g = comp[(comp.benchmark == bench) & (comp.k == 3)]
            if g.empty:
                continue
            add(Num(f"numCompRange{tag}", float(g.swarm_accuracy.max()
                                                - g.swarm_accuracy.min()),
                    claim="S1", note="swarm-accuracy range across 3-model subsets"))
            add(Num(f"numCompBestGain{tag}", float(g.swarm_gain_over_best.max()),
                    fmt="{:+.3f}", claim="S1",
                    note="best subset's gain over its own best single model"))
        # Per benchmark, not pooled: ARC is missing some cross-model pairs, so
        # dividing the pooled row count by the benchmark count gave 32 where the
        # combinatorial answer is 35.
        for k, tag in ((3, "Three"), (5, "Five")):
            counts = comp[comp.k == k].groupby("benchmark").size()
            if len(counts):
                add(Num(f"numCompSubsets{tag}", float(counts.max()), fmt="{:.0f}",
                        claim="S1", note=f"{k}-model subsets of the frozen roster"))
    if cpred is not None:
        def _pred(target: str, name: str, k: int) -> float:
            g = cpred[(cpred.target == target) & (cpred.predictor == name)
                      & (cpred.k == k) & (cpred.benchmark == "gsm8k")]
            return float(g.spearman.iloc[0]) if len(g) else float("nan")
        add(Num("numCompNeffVsAccMean", _pred("n_eff", "acc_mean", 3), fmt="{:+.3f}",
                claim="S1", note="Spearman: N_eff against mean roster accuracy"))
        add(Num("numCompAccVsNeff", _pred("swarm_accuracy", "n_eff", 3), fmt="{:+.3f}",
                claim="S1", note="Spearman: swarm accuracy against N_eff"))
        add(Num("numCompAccVsAccMean", _pred("swarm_accuracy", "acc_mean", 3),
                fmt="{:+.3f}", claim="S1"))

    # -- the confidence protocol ------------------------------------------
    if cauc is not None:
        powered = cauc[(cauc.powered) & (cauc.arm == "frozen")]
        if not powered.empty:
            add(Num("numAucLogprob", float(powered.auc_logprob.mean()), claim="P1"))
            add(Num("numAucSelfReport", float(powered.auc_self_report.mean()), claim="P1"))
            add(Num("numAucSelfReportFalsified",
                    float(powered.auc_self_report_falsified.mean()), claim="P1",
                    note="the falsification attack drives this below chance"))
            add(Num("numAucGap",
                    float((powered.auc_logprob - powered.auc_self_report).mean()),
                    fmt="{:+.3f}", claim="P1"))
            add(Num("numAucCellsPowered", float(len(powered)), fmt="{:.0f}", claim="P1"))
            add(Num("numAucCellsTotal", float((cauc.arm == "frozen").sum()),
                    fmt="{:.0f}", claim="P1",
                    note="frozen-roster cells; the rest are underpowered under R2"))
            add(Num("numAucBelowChanceCells",
                    float((powered.auc_self_report_falsified < 0.5).sum()),
                    fmt="{:.0f}", claim="P1"))
            add(Num("numSelfReportMissingMax", float(powered.sr_missing_rate.max()),
                    claim="P1", note="R3: the self-report is sometimes simply absent"))

    # -- the measurement floor itself ------------------------------------
    # Every "undetectable at this scale" sentence in the paper compares against
    # one of these, so they are macros rather than a number in the prose.
    floor_path = R / "measurement_floor.json"
    if floor_path.exists():
        raw = json.loads(floor_path.read_text())
        per: dict[str, float] = {}
        for key, value in raw["per_benchmark"].items():
            bench = key.split(":")[0]
            per[bench] = max(per.get(bench, 0.0), float(value))
        add(Num("numFloorWorst", float(raw["worst_case_floor"]),
                note="minimum detectable effect, worst benchmark"))
        for bench, tag in BENCHMARK_TAGS.items():
            if bench in per:
                add(Num(f"numFloor{tag}", per[bench], note="minimum detectable effect"))

    # -- E1: the sleeper, cited by the appendix's escape-class note -------
    if sleeper is not None:
        e1 = sleeper[sleeper.experiment == "E1_sleeper"]
        if not e1.empty:
            by_method = e1.groupby("method")["drop"].mean()
            for method, tag in (("reputation_decay", "Reputation"), ("aip_gated", "Aip"),
                                ("aip_gated_w20", "AipWindowed"), ("majority", "Majority")):
                if method in by_method.index:
                    add(Num(f"numSleeperDrop{tag}", float(by_method[method]), claim="E1"))
            if {"reputation_decay", "aip_gated"} <= set(by_method.index):
                add(Num("numSleeperDropSpread",
                        float(by_method["aip_gated"] - by_method["reputation_decay"]),
                        claim="E1",
                        note="AIP degrades MORE than reputation decay; E1 as designed "
                             "is refuted, and the spread is marginal against the floor"))

    # -- experimental constants, READ FROM CONFIG not typed --------------
    #
    # These four were the last L4-era numbers left in the manuscript: 4 models,
    # 100 tasks per benchmark, "three benchmarks", "3--9B". Every one of them is
    # false in this regime (RULE 1), and none of them was caught by the
    # no-bare-decimal test because they lived in a macro. They are now derived.
    # Swarm size is read back off the sweep that produced the results rather
    # than from the runner constant, so the paper cannot state a size the data
    # was not generated at.
    if sweep is not None and "n_agents" in sweep.columns:
        sizes = sorted(int(x) for x in sweep.n_agents.unique())
        if len(sizes) != 1:
            raise ValueError(f"sweep mixes swarm sizes {sizes}; numAgents is ambiguous")
        add(Num("numAgents", float(sizes[0]), fmt="{:.0f}", note="read from sweep.parquet"))
    counts = json.loads(Path("configs/task_lists.json").read_text())["n_per_benchmark"]
    add(Num("numTasksMin", float(min(counts.values())), fmt="{:.0f}",
            note="frozen task lists"))
    add(Num("numTasksMax", float(max(counts.values())), fmt="{:.0f}",
            note="frozen task lists"))
    add(Num("numBenchmarks", float(len(counts)), fmt="{:.0f}", note="frozen task lists"))
    add(Num("numHeadlineBenchmarks", float(len(load_rules().headline_benchmarks)),
            fmt="{:.0f}", note="configs/decision_rules.yaml"))
    registry = yaml.safe_load(Path("configs/models.yaml").read_text())["models"]
    arms: dict[str, list[str]] = {}
    for name, cfg in registry.items():
        arm = cfg.get("arm")
        if arm is None:
            raise ValueError(f"configs/models.yaml: {name} has no `arm` tag")
        if cfg.get("enabled", True):
            arms.setdefault(arm, []).append(name)
    add(Num("numModels", float(len(arms.get("frozen", []))), fmt="{:.0f}",
            note="frozen roster (configs/models.yaml arm: frozen)"))
    add(Num("numModelsWeakArm", float(len(arms.get("weak_tier", []))), fmt="{:.0f}",
            note="weak-tier arm, reported separately and never pooled"))
    add(Num("numModelsAll", float(sum(len(v) for v in arms.values())), fmt="{:.0f}",
            note="every enabled model that generated cache"))
    sizes = [float(registry[m]["params_b"]) for m in arms.get("frozen", [])
             if "params_b" in registry[m]]
    if sizes:
        add(Num("numParamsMinB", min(sizes), fmt="{:.0f}",
                note="smallest model in the frozen roster, billions of parameters"))
        add(Num("numParamsMaxB", max(sizes), fmt="{:.0f}",
                note="largest model in the frozen roster, billions of parameters"))
    add(Num("numAttackClasses", 7.0, fmt="{:.0f}", note="attack registry"))
    add(Num("numBootstrapResamples", 500.0, fmt="{:.0f}", note="sweep config"))
    # Observability grid. These are configuration constants rather than
    # measurements, but they are still emitted as macros: the no-hand-typed-number
    # rule admits no exceptions, because an exception is where a wrong number hides.
    add(Num("numPobsQuarter", 0.25, fmt="{:.2f}", note="p_obs grid level"))
    add(Num("numPobsTenth", 0.10, fmt="{:.2f}", note="p_obs grid level"))
    return out


def as_dict(nums: list[Num]) -> dict[str, Any]:
    return {n.macro: n for n in nums}
