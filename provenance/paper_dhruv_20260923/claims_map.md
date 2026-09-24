# Section → claim map (DSN 2027 scaffold)

Which section carries which entry from `results/claims.md`. Checked mechanically
by `tests/test_paper.py`: every ledger id must appear here.

**Working title:** *Exploiting Coherent Liars in LLM Swarms: A Real Mechanism,
a Partial Defence, and Its Fundamental Limit*

## Contributions

1. **Inversion is real, large, and free at f=0** — and is a property of the
   adversarial channel rather than of honest-agent competence. `I1 I2 I3`
2. **Measurement laws for LLM swarms** — N_eff ≈ 1.5 homogeneous, the
   attenuation law exact at C=2, seven-family cross-correlation. `C1 C2 C3 C4`
3. **A partial defence with a characterised fundamental limit** — soft gating
   closes the exploitable mid-band (2.1× floor), randomisation hurts, and a
   residual attack at or below chance coherence is invisible to any per-question
   coherence statistic (Proposition 1). `G1 D1 D2 D3`
4. **What escapes the limit** — external-information and longitudinal defences,
   stated as untested hypotheses, pointing at the field's open identity problem.
   `U3` and the E1 comparative note.

## Section map (as written)

| § | file | carries |
|---|---|---|
| 1 | `introduction.tex` | `I1 I2 G1 D3` — the arc: a real mechanism, then its limit; the measurement-floor rule stated up front |
| 2 | `related.tex` | `H2' H2'' H2''' M1 L1` — discard family, and AgentChain's f≥N/2 admission mapped onto our `U3` escape class (a) |
| 3 | `system.tex` | `M2a M2b H2'' I1 G2' G1 B1 F1` — swarm, threat model, the three-way gate, windowing, unfakeable confidence, attack-class table |
| 4 | `theory.tex` | `C3 C4 G2'` — attenuation law, exactness at C=2, identifiability, receiver conditioning, coordinatability *with its direction corrected* |
| 5 | `design_laws.tex` | `C1 C2 C3 C4 G2' S1` — three design laws: correlated failure is the default; decorrelation does not buy accuracy (**S1 refuted**, 9b table); coordinatability belongs to the answer space |
| 6 | `evaluation.tex` | `H1 H2' H2'' H2''' I1 I2 I3 N1 M1 M2a M2b E3` — symbolic adversary; **N1 caption reads "undetectable at this scale"**; E3 cost-of-defence table (9a) |
| 7 | `protocol.tex` | `P1 F1` — **new section**: which confidence surface a receiver can believe. Logprob AUC 0.724 vs self-report 0.661, falsified to 0.339 (below chance) |
| 8 | `real_adversaries.tex` | `B1 E1 E2a E2b` — real prompt-injected adversaries. Falsified confidence moved to §7; **deterrence removed entirely** (see below) |
| 9 | `gate_aware.tex` | `G1 D1 D2 L1 R1` — the paper's turn, anchored at f = 0.5 |
| 10 | `limits_gating.tex` | `D3 U3 E1` — Proposition 1 and the two escape classes |
| 11 | `prior_art_revisited.tex` | `H2' H2'' H2''' E1 F1 M1` — where prior art's limits are real, and where ours are worse |
| 12 | `limitations.tex` | `H1 L1 I3 S1 M2b E2b E1 G2' D2 N1 U1 U2 U3 X1 R1 B2` — **every refutation lives here, at the same length as the confirmations** |
| 13 | `reproducibility.tex` | `R1` — determinism, one hardware regime, the self-checking build, and the conclusions |
| A | `appendix_indistinguishability.tex` | `D3 M2b E1` — proposition, proof sketch, anchored empirical section |

`mechanism.tex` was folded into §3 and `conclusion.tex` into §12; both files are
removed. The manuscript is twelve numbered sections plus one appendix.

## Figures

| figure | carries | requirement |
|---|---|---|
| money figure | `I1 H1 N1` | **annotation must state the N-dependence floor**, not "invariant" |
| inversion gain vs f | `I1 I2` | ablation curve, floor band shaded |
| evasion band (`evasion_band.pdf`, `scripts/fig_evasion_band.py`) | `G1 D1 D2 D3 M2b` | **DONE.** Anchored at f = 0.5 only. Honest-coherence rug overlaid with underpowered pairs (R2) drawn separately, chance line 1/(C−1) marked, ceiling marked, M1–M3 on the same axes. **Markers only: the grid resolves [0.15, 0.40] with three points on C=4, so nothing is interpolated.** |
| N-scaling | `N1` | NOT BUILT. §6 reports the three point estimates and the range against the floor in text, with the undetectability stated. |
| correlation heatmaps | `C1` | 7×7 per benchmark |
| Sybil accuracy vs s (`sybil_accuracy.pdf`, `scripts/fig_sybil.py`) | `E2a E2b` | **DONE.** Coherent and disagreeing panels side by side at a shared scale — the comparison is the test of E2b, so they are not averaged. |
| sleeper timeline | `E1` | NOT BUILT. The sleeper result is reported in §7 and §11 as text; no figure is cited, so nothing dangles. |

## Refutations that must appear in the paper

`H1` flat-in-f · `E1` sleeper · `E2b` Sybil mechanism · `L1` low-ceiling defence ·
`N1` N-dependence (as undetectable, not refuted) · `D2` randomisation harmful ·
`R1` instrumentation disclosure.

## Task 6 audit outcomes — pending section closed

`results/claims.md` no longer carries a pending section. The five ids below were
re-derived against the measurement floor by `scripts/task6_audit_pending.py`
(tables in `results/pending_claims_audit.md`) and each resolved to a verdict.

| id | section | outcome |
|---|---|---|
| `H2` | §2, §6 | **split by adversary regime.** Superseded by `H2'` (symbolic, deployable methods: exactly 0.000), `H2''` (oracle methods: refuted in both regimes), `H2'''` (real-LLM: nothing reaches zero; AIP separates on 4/6, loses on BoolQ). The bare id `H2` is retired and must not be cited. |
| `F1` | §7 | **CONFIRMED.** AIP moves by exactly 0.000 — an implementation identity, since the gate never reads the self-report — while the self-report reader loses 0.104–0.234 on 4 of 6 benchmarks. |
| `B1` | §7, §11 | **CONFIRMED.** Pooled 0.470 → windowed 0.965 at f=0.7 (7.3× floor), with UNDETECTABLE cost on the stationary control. |
| `G2` | §4, §11 | **REFUTED as worded**, restated as `G2'`: the margin over honest agreement collapses, not coordinatability, which is in fact highest on the most closed space. Cite `G2'`. |
| `M2` | §6 | **split.** `M2a` (parity is a real confound; the discard family is exactly invariant) is confirmed; `M2b` (AIP does not exploit it) is **REFUTED at f = 0.5** and undetectable elsewhere. The bare id `M2` is retired. |
| `X1` | §11 | **DATA GAP**, declared: no calibrated honest q for the binary answer-space class, so BoolQ carries no margin-over-honest claim. |
| `X2` | §5, §12 | **CORRECTION**, 2026-09-20: the honest-coincidence constant was reported as 0.806, an R2-underpowered MMLU all-pairs mean that reached the abstract. Replaced by 0.560, pooled over the 9 powered 4-way MC pairs. Root cause was a missing power filter in `paper_numbers.py`, now applied at source. Not cited as a claim in the text — it is the provenance record for a number the text uses. |

`H2'` `H2''` `H2'''` `G2'` `M2a` `M2b` `X1` `S1` `P1` `E3` `B2`

## Withdrawn permanently

`B2` deterrence / payoff flattening. **Do not re-add.** The conditional was
"G2 passes Task 6 re-verification"; G2 failed. Confirmed permanent by the author
on 2026-09-11, also on the independent ground that the bandit data shows payoff
and not conduct — no arm switch occurred within the horizon tested. The data
stays in the release; the claim is out of the paper for good.

## Additional refutations that must appear in the paper

`H2''` oracle methods beat AIP on MedQA and BoolQ · `H2'''` AIP loses to majority
vote on BoolQ under the real-LLM adversary · `G2'` the original direction was
inverted · `M2b` the f = 0.5 tie is decided by self-vote-share normalisation,
not by the mechanism · `X1` BoolQ's honest ceiling is a placeholder.
