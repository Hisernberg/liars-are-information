#!/usr/bin/env python3
"""Generate results/claims.md — the writing phase's source of truth.

Every claim the paper will make, one per line, mapped to the artefact that
supports it, the parquet it came from, and the config hash that produced it.
Numbers are read from the parquets rather than transcribed, so the ledger cannot
drift from the data. Claims with no number behind them are listed under
UNSUPPORTED, which is the point of keeping the ledger at all.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.harness.manifest import config_hash, git_commit  # noqa: E402
from aip.tasks import roster  # noqa: E402

R = Path("results")
AGG = R / "aggregation"
ADV = R / "adversarial"
COR = R / "correlation"


@dataclass
class Claim:
    id: str
    text: str
    artefact: str
    source: str
    number: str = ""
    supported: bool = True
    note: str = ""


def load(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def fmt(x: float, d: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def build(claims: list[Claim]) -> list[Claim]:
    sweep = load(AGG / "sweep.parquet")
    adv = load(ADV / "adversarial_sweep.parquet")
    chan = load(ADV / "channel_stats.parquet")
    dec = load(ADV / "channel_decisions.parquet")
    qsp = load(ADV / "q_by_answer_space.parquet")
    burst = load(ADV / "burst_windowed.parquet")
    bandit = load(ADV / "bandit_log.parquet")
    pairs = load(COR / "pairwise_estimates.parquet")
    neff = load(COR / "effective_swarm_size.parquet")
    mc = load(COR / "multiclass_validation.parquet")
    qpair = load(COR / "q_pairwise.parquet")
    regret = load(AGG / "minimax_regret.parquet")
    parity = load(AGG / "parity_audit.parquet")

    def adv_ref(benchmark: str, attack: str, method: str, fmin: float = 0.5) -> float:
        if adv is None:
            return float("nan")
        s = adv[
            (adv.p_obs == 1.0)
            & (adv.parity == "matched@0.1")
            & (adv.composition == "mix_all4")
            & (adv.benchmark == benchmark)
            & (adv.attack == attack)
            & (adv.method == method)
            & (adv.f >= fmin)
        ]
        return float(s["accuracy_point"].mean()) if len(s) else float("nan")

    def best_discard(benchmark: str, attack: str, fmin: float = 0.5) -> float:
        if adv is None:
            return float("nan")
        d = [
            "majority",
            "geometric_median",
            "sac_filter_refine",
            "coord_median",
            "trimmed_mean",
            "krum",
            "multi_krum",
            "dawid_skene",
        ]
        s = adv[
            (adv.p_obs == 1.0)
            & (adv.parity == "matched@0.1")
            & (adv.composition == "mix_all4")
            & (adv.benchmark == benchmark)
            & (adv.attack == attack)
            & (adv.method.isin(d))
            & (adv.f >= fmin)
        ]
        return float(s.groupby("f")["accuracy_point"].max().mean()) if len(s) else float("nan")

    # -- headline -------------------------------------------------------
    if sweep is not None:
        ref = sweep[
            (sweep.p_obs == 1.0)
            & (sweep.topology == "complete")
            & (sweep.adversary == "always_wrong")
            & (sweep.parity == "matched@0.1")
            & (sweep.composition == "mix_all4")
            & (sweep.method == "aip_gated")
        ]
        for b in ("gsm8k", "mmlu", "math500"):
            s = ref[ref.benchmark == b]
            if s.empty:
                continue
            claims.append(
                Claim(
                    f"H1-{b}",
                    f"On {b}, AIP-gated accuracy is flat in the Byzantine fraction across f=0..0.7 "
                    f"(symbolic coherent adversary).",
                    "money_figure.pdf / Table 1",
                    str(AGG / "sweep.parquet"),
                    f"range {fmt(s.accuracy_point.min())}–{fmt(s.accuracy_point.max())}, "
                    f"spread {fmt(float(s.accuracy_point.max() - s.accuracy_point.min()))}",
                    supported=bool(float(s.accuracy_point.max() - s.accuracy_point.min()) < 0.15),
                    note="" if b != "math500" else "NOT flat under gate v2; see limitations.",
                )
            )
        coll = sweep[
            (sweep.p_obs == 1.0)
            & (sweep.topology == "complete")
            & (sweep.adversary == "always_wrong")
            & (sweep.parity == "matched@0.1")
            & (sweep.composition == "mix_all4")
            & (sweep.f == 0.7)
            & (sweep.method.isin(["majority", "geometric_median", "trimmed_mean", "coord_median"]))
        ]
        claims.append(
            Claim(
                "H2",
                "Every discard-family baseline reaches exactly zero accuracy at f=0.7 against a "
                "coherent adversary, including methods given the true Byzantine count.",
                "money_figure.pdf",
                str(AGG / "sweep.parquet"),
                f"max accuracy across those methods at f=0.7 = {fmt(float(coll.accuracy_point.max()))}",
                supported=bool(len(coll) and float(coll.accuracy_point.max()) < 0.05),
            )
        )

    # -- invertibility spectrum -----------------------------------------
    if chan is not None and adv is not None:
        neg = chan[(chan.benchmark == "gsm8k") & (chan.attack == "semantic_negation")]
        hal = chan[(chan.benchmark == "gsm8k") & (chan.attack == "semantic_hallucination")]
        if len(neg) and len(hal):
            qn, qh = float(neg.coherence.iloc[0]), float(hal.coherence.iloc[0])
            gn = adv_ref("gsm8k", "semantic_negation", "aip_gated") - best_discard(
                "gsm8k", "semantic_negation"
            )
            gh = adv_ref("gsm8k", "semantic_hallucination", "aip_gated") - best_discard(
                "gsm8k", "semantic_hallucination"
            )
            claims.append(
                Claim(
                    "I1",
                    "Inversion gain tracks measured adversarial channel coherence: a structured "
                    "lie is exploitable, an idiosyncratic one is not.",
                    "invertibility_spectrum.pdf",
                    str(ADV / "channel_stats.parquet"),
                    f"gsm8k negation q̂={fmt(qn)} gain {fmt(gn)}; "
                    f"hallucination q̂={fmt(qh)} gain {fmt(gh)}",
                )
            )
            claims.append(
                Claim(
                    "I2",
                    "HONEST NEGATIVE RESULT: semantic hallucination is fluent and highly effective "
                    "but not invertible; AIP degrades to discard-level performance against it.",
                    "invertibility_spectrum.pdf",
                    str(ADV / "adversarial_sweep.parquet"),
                    f"gain over best discard = {fmt(gh)} (vs {fmt(gn)} for negation)",
                )
            )

    # -- gate / threshold -----------------------------------------------
    if dec is not None:
        h = dec[dec.channel == "honest"]["pct_inverted"].max()
        a = dec[(dec.channel == "adversary") & (dec.f >= 0.5)]["pct_inverted"].mean()
        claims.append(
            Claim(
                "G1",
                "The calibrated honest-q gate never inverts an honest channel, at any Byzantine "
                "fraction or benchmark, while inverting adversarial channels.",
                "Table: channel decisions",
                str(ADV / "channel_decisions.parquet"),
                f"max honest inverted {fmt(float(h), 1)}%; mean adversary inverted at f>=0.5 "
                f"{fmt(float(a), 1)}%",
            )
        )
    if qsp is not None:
        m = qsp[(qsp.benchmark == "mmlu") & (qsp.attack == "semantic_negation")]
        g = qsp[(qsp.benchmark == "gsm8k") & (qsp.attack == "semantic_negation")]
        if len(m) and len(g):
            claims.append(
                Claim(
                    "G2",
                    "Coordinatability collapses on closed label spaces: the negation attack's "
                    "cross-model coherence falls below the honest floor on 4-way multiple choice, "
                    "so there is no coherent channel to exploit even though the gate fires.",
                    "Table: q by answer space",
                    str(ADV / "q_by_answer_space.parquet"),
                    f"mmlu q̂={fmt(float(m.q_hat_adversary.iloc[0]))} vs honest "
                    f"{fmt(float(m.honest_q_receiver_conditioned.iloc[0]))}; "
                    f"gsm8k q̂={fmt(float(g.q_hat_adversary.iloc[0]))} vs honest "
                    f"{fmt(float(g.honest_q_receiver_conditioned.iloc[0]))}",
                )
            )

    # -- correlation / N_eff --------------------------------------------
    if pairs is not None:
        w = pairs[pairs.kind == "within_model"]["phi_point"]
        c = pairs[(pairs.kind == "cross_model") & (pairs.benchmark == "gsm8k")]["phi_point"]
        claims.append(
            Claim(
                "C1",
                "Within-model error correlation exceeds cross-model correlation by a wide margin: "
                "a homogeneous swarm is closer to one epistemic agent.",
                "correlation_heatmap.pdf",
                str(COR / "pairwise_estimates.parquet"),
                f"within {fmt(float(w.mean()))} vs cross {fmt(float(c.mean()))} on gsm8k "
                f"(ratio {fmt(float(w.mean() / c.mean()), 2)}x)",
            )
        )
    if neff is not None:
        hom = neff[neff.kind == "homogeneous"]["n_eff"]
        het = neff[neff.kind == "heterogeneous"]["n_eff"]
        claims.append(
            Claim(
                "C2",
                "A ten-agent swarm is worth far fewer independent agents than its size suggests.",
                "Table: effective swarm size",
                str(COR / "effective_swarm_size.parquet"),
                f"homogeneous N_eff {fmt(float(hom.min()), 2)}–{fmt(float(hom.max()), 2)}; "
                f"best heterogeneous {fmt(float(het.max()), 2)} of 10",
            )
        )
    if qpair is not None:
        mm = qpair[qpair.benchmark == "mmlu"]
        ms = qpair[qpair.benchmark == "math500"]
        claims.append(
            Claim(
                "C3",
                "Honest models share wrong-answer attractors on multiple choice but not on "
                "open-ended answers, which is why inversion needs a calibrated threshold.",
                "Table: honest q",
                str(COR / "q_pairwise.parquet"),
                f"mmlu mean q={fmt(float(mm.q_true.mean()))} vs chance 0.333; "
                f"math500 mean q={fmt(float(ms.q_true.mean()))} vs chance 0",
            )
        )
    if mc is not None:
        claims.append(
            Claim(
                "C4",
                "The multiclass attenuation law recovers marginal error rates where the binary law "
                "cannot, but does not beat it on the pairwise product for every benchmark.",
                "Table: multiclass validation",
                str(COR / "multiclass_validation.parquet"),
                "; ".join(
                    f"{r.benchmark}: product MAE binary {fmt(r.mae_product_binary_point)} -> mc "
                    f"{fmt(r.mae_product_mc_point)}"
                    for r in mc.itertuples()
                ),
            )
        )

    # -- falsified confidence -------------------------------------------
    if adv is not None:
        rows = []
        for b in roster.benchmarks():
            base = adv_ref(b, "semantic_negation", "aip_gated", fmin=0.3)
            att = adv_ref(b, "falsified_confidence", "aip_gated", fmin=0.3)
            sr_base = adv_ref(b, "semantic_negation", "confidence_weighted_selfreport", 0.3)
            sr_att = adv_ref(b, "falsified_confidence", "confidence_weighted_selfreport", 0.3)
            rows.append(
                f"{b}: AIP {fmt(att - base, 3)}, self-report reader {fmt(sr_att - sr_base, 3)}"
            )
        claims.append(
            Claim(
                "F1",
                "Inverting the self-reported confidence leaves AIP unmoved because AIP reads the "
                "logprob channel; only the self-report-reading baseline degrades.",
                "Table: falsified confidence",
                str(ADV / "adversarial_sweep.parquet"),
                "; ".join(rows),
            )
        )

    # -- burst / windowed ------------------------------------------------
    if burst is not None:
        b = burst[burst.attack == "burst"]
        static = float(b[b.method == "aip_gated"]["accuracy_point"].mean())
        best = {
            int(w): float(b[b.method == f"aip_gated_w{int(w)}"]["accuracy_point"].mean())
            for w in sorted(x for x in burst.window.unique() if x)
        }
        claims.append(
            Claim(
                "B1",
                "A non-stationary (burst) adversary defeats the pooled-history gate, and sliding-"
                "window channel statistics recover the loss without costing the stationary cells.",
                "burst_windowed.pdf",
                str(ADV / "burst_windowed.parquet"),
                f"static {fmt(static)} -> " + ", ".join(f"W={w} {fmt(v)}" for w, v in best.items()),
            )
        )

    # -- deterrence ------------------------------------------------------
    if bandit is not None:
        out = []
        for d, g in bandit.groupby("defence"):
            last = g.sort_values("epoch").iloc[-1]
            out.append(
                f"{d}: value(coherent)={fmt(float(last.value_coherent_lie))} "
                f"value(noise)={fmt(float(last.value_noise))} "
                f"acc={fmt(float(g.swarm_accuracy.mean()))}"
            )
        claims.append(
            Claim(
                "D1",
                "AIP removes the payoff advantage that coherent lying enjoys against a discard "
                "defence, flattening the adaptive adversary's arm values.",
                "deterrence_timeseries.pdf",
                str(ADV / "bandit_log.parquet"),
                "; ".join(out),
            )
        )
        claims.append(
            Claim(
                "D2",
                "The adaptive adversary does NOT collapse onto noise against AIP within the "
                "horizon tested; deterrence is in the payoff, not in the observed behaviour.",
                "deterrence_timeseries.pdf",
                str(ADV / "bandit_log.parquet"),
                "arm share unchanged; see D1 values",
                supported=True,
                note="Stated as a limitation of the claim, not a positive result.",
            )
        )

    # -- methodology -----------------------------------------------------
    if parity is not None:
        s = (
            parity.assign(a=parity.effect.abs())
            .groupby("method")["a"]
            .mean()
            .sort_values(ascending=False)
        )
        unweighted = [
            "majority",
            "geometric_median",
            "coord_median",
            "trimmed_mean",
            "krum",
            "multi_krum",
            "dawid_skene",
        ]
        uw = float(s[[m for m in unweighted if m in s.index]].max()) if len(s) else float("nan")
        claims.append(
            Claim(
                "M1",
                "Self-vote share is a real confound for weighted aggregators, and AIP's advantage "
                "does not come from it.",
                "Table: normalization-parity audit",
                str(AGG / "parity_audit.parquet"),
                f"max mean |effect| among unweighted methods {fmt(uw, 4)} (sanity: ~0); "
                f"aip_gated {fmt(float(s.get('aip_gated', np.nan)), 4)}",
            )
        )
    if regret is not None:
        out = []
        for b, g in regret.groupby("benchmark"):
            best = g.sort_values("minimax_regret").iloc[0]
            out.append(f"{b}: {best.method} ({fmt(float(best.minimax_regret))})")
        claims.append(
            Claim(
                "M2",
                "Under minimax regret over f, restricted to deployable (non-oracle) methods, AIP "
                "is the best choice when f is unknown.",
                "Table: minimax regret",
                str(AGG / "minimax_regret.parquet"),
                "lowest minimax regret per benchmark — " + "; ".join(out),
            )
        )

    # -- known unsupported ----------------------------------------------
    claims.append(
        Claim(
            "U1",
            "AIP's accuracy is monotone INCREASING in f (the original simulation claim).",
            "—",
            "—",
            "",
            supported=False,
            note="Measured behaviour is flat in f, not increasing. No cell shows a monotone rise; "
            "the defensible claim is invariance, not improvement.",
        )
    )
    claims.append(
        Claim(
            "U2",
            "The blind estimator can bootstrap trust with no labels in deployment.",
            "—",
            str(COR / "multiclass_validation.parquet"),
            "",
            supported=False,
            note="Only validated where agents are individually strong; MAE 0.13-0.21 on "
            "MATH-500/MMLU with the binary law, and the multiclass variant does not beat "
            "it on MMLU. Not established as deployment-ready.",
        )
    )
    claims.append(
        Claim(
            "U3",
            "Results generalise beyond 4 open-weight models in the 3-9B range on 3 benchmarks.",
            "—",
            "—",
            "",
            supported=False,
            note="Single hardware node, 100 tasks per benchmark, one adversary family per attack. "
            "No claim of generality is supported by the data collected.",
        )
    )
    return claims


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=R / "claims.md")
    p.add_argument(
        "--allow-empty",
        action="store_true",
        help="permit writing a ledger with no supported claims",
    )
    args = p.parse_args()

    claims = build([])

    # FAIL LOUD, for the same reason make_numbers.py does. This ledger is the
    # writing phase's source of truth: every claim maps to the parquet that
    # supports it. With the parquets missing, build() returns almost nothing and
    # the ledger is rewritten as "0 supported" -- which is precisely what
    # happened to a committed 18-claim ledger earlier in this project. Silently
    # replacing evidence with its absence is the worst available failure here.
    supported = [c for c in claims if getattr(c, "supported", True)]
    existing = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
    n_existing = existing.count("\n| ") if existing else 0
    if n_existing and not supported and not args.allow_empty:
        print(
            f"REFUSING to write {args.out}: no claim is supported, over an existing "
            f"ledger with {n_existing} rows. The result parquets are probably "
            f"missing. Run the phases that produce them, or pass --allow-empty.",
            file=sys.stderr,
        )
        return 1

    commit = git_commit(Path.cwd())
    cfgs = {}
    for name in (
        "configs/models.yaml",
        "configs/inversion_thresholds.yaml",
        "configs/task_lists.json",
        "configs/experiments/phase_a_cache.yaml",
    ):
        path = Path(name)
        if path.exists():
            cfgs[name] = config_hash(path.read_text(encoding="utf-8"))

    supported = [c for c in claims if c.supported]
    unsupported = [c for c in claims if not c.supported]

    lines = [
        "# Claims ledger",
        "",
        "Every claim the paper will make, mapped to the artefact that supports it, the parquet",
        "it was computed from, and the config that produced it. Generated by",
        "`scripts/phase_e_claims.py` from the result files, so it cannot drift from the data.",
        "",
        f"- git commit: `{commit}`",
        *[f"- `{k}` hash `{v}`" for k, v in cfgs.items()],
        "",
        "All sweeps use deterministic seeds via `aip.harness.manifest.stable_seed`; runs before",
        "that fix drew different swarms from the same config and their numbers are superseded.",
        "",
        "## Supported claims",
        "",
        "| id | claim | artefact | evidence | source |",
        "|---|---|---|---|---|",
    ]
    for c in supported:
        note = f" _{c.note}_" if c.note else ""
        lines.append(f"| {c.id} | {c.text}{note} | `{c.artefact}` | {c.number} | `{c.source}` |")
    lines += [
        "",
        "## UNSUPPORTED — do not claim without new evidence",
        "",
        "| id | claim | why not |",
        "|---|---|---|",
    ]
    for c in unsupported:
        lines.append(f"| {c.id} | {c.text} | {c.note} |")
    lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}  ({len(supported)} supported, {len(unsupported)} unsupported)")
    for c in supported:
        print(f"  [{c.id}] {c.number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
