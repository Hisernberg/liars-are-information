# Claims ledger v2 — H100 regeneration

Every claim, its status, its evidence, and its position relative to the
measurement floor established in `results/measurement_floor_audit.md`
(0.068 on GSM8K, 0.100 on the 200-task benchmarks).

**Rule in force:** an effect whose magnitude does not clear its cell's floor is
**UNDETECTABLE AT THIS SCALE**. It is never reported as "small", "no effect", or
"invariant". All L4-era claims are void (RULE 1) and are not carried forward.

## CONFIRMED

| id | claim | evidence | vs floor |
|---|---|---|---|
| **I1** | Inversion is a large real mechanism: gain 0.65 at f=0.7, isolated by ablating `aip_trust_only` against `aip_gated` | `results/aggregation/n_scaling.parquet` | **9.6×** floor |
| **I2** | Inversion has no detectable cost at f=0 (measured 0.001) | same | 0.001 ≪ 0.068 — *no detectable cost*, NOT "zero cost" |

| **M1** | AIP-gated has the lowest minimax regret under unknown f on 4 of 6 benchmarks (gsm8k 0.004, arc 0.006, mmlu 0.038, medqa 0.068 vs 0.84–0.98 for all non-AIP) | `minimax_regret.parquet` | 100–200× separation |
| **C1** | Within-model error correlation exceeds cross-model (gsm8k 0.652 vs 0.444) | `pairwise_estimates.parquet` | — |
| **C2** | A ten-agent homogeneous swarm is worth **N_eff ≈ 1.50** independent agents (range 1.07–1.91); best heterogeneous **2.46** | `effective_swarm_size.parquet` | Restricted to the **measured** within-model basis, which only GSM8K carries. Rows on other benchmarks use cross-model ρ alone and are labelled an upper bound; their maximum, 6.86 (ARC), is not comparable and is never quoted beside these |
| **C3** | The attenuation law is **exact at C=2**: BoolQ measured q = 1.000, MAE 0.000 | `multiclass_validation.parquet` | analytic identity, not an estimate |
| **C4** | The multiclass law beats the binary law on every benchmark (math500 MAE 0.124 → 0.013) | same | — |
| **G1** | A gate-aware adversary defeats AIP: accuracy 0.082 at the optimum (medqa) against 0.878 for a fully coherent adversary, **anchored at f = 0.5** | `gate_aware_mitigations.parquet` | attacker gain **0.796, 8.0×** floor on medqa; **0.529, 5.3×** on mmlu. The previously reported 0.92 was the **f = 0.7** cell, where a Byzantine majority drives every coherence level to zero; it is not an admissible anchor and is withdrawn |
| **P1** | **Protocol finding.** The logprob confidence surface is both a better correctness discriminator and unfakeable; the self-report is falsifiable to **below chance** | `correlation/confidence_auc.parquet` | logprob AUC **0.724** vs self-report **0.661** (gap +0.063) over the 22 of 42 frozen-roster cells that clear R2's n_wrong ≥ 30. Under `falsified_confidence` the self-report falls to **0.339** — anti-informative — on **21 of 22** powered cells, while the logprob surface is untouched by construction. R3: the self-report is also simply absent on up to **34.0%** of rows (mmlu/phi4_mini_reasoning). The weak-tier arm shows the same ordering (0.716 / 0.573 / 0.427), so it is not an artefact of model strength |
| **E3** | **Cost of defence.** Threshold-gated inversion is computationally almost free, and the gate is the cheap part | `aggregation/cost_of_defence.parquet` | µs per receiver-decision at f=0.5, mean over headline benchmarks: majority **3.2**, krum 20.9, trimmed mean 68.5, coord median 75.6, AIP trust-only **77.8**, AIP-gated **79.6**, multi-Krum 92.9, geometric median **671.6**. The gate adds **+1.7 µs (2.2%)** over its own inversion-free ablation. One completion averages 493 tokens, ≈**5073×** the slowest rule's cost at 100 tok/s. Measured by timing the aggregators over the cache, not apportioned from block timestamps, which cannot separate methods |
| **F1** | Falsifying the self-report degrades the self-report-reading baseline and leaves AIP exactly where it was. Controlled contrast: same answers, same logprobs, only `self_reported_confidence` inverted | `adversarial_sweep.parquet`, `results/pending_claims_audit.md` | reader −0.234 arc, −0.159 mmlu, −0.133 math500, −0.104 medqa (4/6 above floor); AIP **exactly 0.000** on all six — an implementation identity, not an estimate below the floor. boolq/gsm8k reader effects are UNDETECTABLE (−0.053, −0.043), not absent |
| **B1** | A non-stationary (burst) adversary defeats the pooled-history gate, and sliding-window channel statistics recover it at no measurable cost on stationary attacks | `burst_windowed.parquet` | pooled 0.470 at f=0.7 → W=20 **0.965**, recovery **+0.495 = 7.3×** the 0.068 gsm8k floor; the stationary control (semantic_negation) moves at most +0.003, i.e. **UNDETECTABLE cost** rather than zero cost. Windowing also recovers the noise attack (+0.129 to +0.214) |

## REFUTED

| id | claim | status | magnitude |
|---|---|---|---|
| **H1** | AIP-gated accuracy is flat in f | **REFUTED on 4 of 6** | Under the real-LLM coherent adversary (mix_all4, p_obs=1, matched@0.1, semantic_negation), accuracy at f=0.7 minus f=0: boolq **−0.615**, medqa **−0.376**, mmlu **−0.321**, arc **−0.316** — all ≥3× floor. gsm8k −0.012 and math500 −0.081 are **UNDETECTABLE**, i.e. flat is not refuted there but neither is it demonstrated. Under the *symbolic* adversary the spreads are 0.008 gsm8k and 0.043 mmlu (both below floor) but 0.477 on math500, so the original wording is regime-dependent as well as benchmark-dependent |
| **E1** | A sleeper defeats reputation decay while AIP-windowed reclassifies it | **REFUTED as designed** | reputation dropped **least** (0.040), AIP **most** (0.150); diff 0.110 vs 0.100 floor — **marginal**, see note |
| **E2b** | AIP's Sybil robustness comes from coherence amplification | **MECHANISM REFUTED** | coherent −0.066 vs disagreeing −0.077; diff 0.011 ≪ 0.100 |
| **I3** | Inversion gain is a property of the adversarial channel, **not** of honest-agent competence | **REFUTED on a matched comparison** | The 0.089 difference that supported this compared the weak arm's **six** benchmarks (0.547) against the strong roster's **three** (0.645) — different task sets. On the common set the weak arm reads **0.470**, a difference of **0.175 = 1.8× floor**, and it is below the strong roster on all three individually (math500 0.000 vs 0.175, medqa 0.663 vs 0.855, mmlu 0.772 vs 0.900). Inversion gain **is** smaller on a weaker swarm — and in the opposite direction to the pre-registered expectation that more honest error would give inversion more to correct |
| **L1** | A low honest-q ceiling is a structural defence | **REFUTED** | math500 (ceiling 0.198) gate is near-inert: inversion gain 0.000 across the f=0.5 row |
| **S1** | Maximising effective swarm size is the composition recipe ("diversity beats scale") | **REFUTED. Both halves.** | Over all 35 three-model and 21 five-model subsets of the frozen roster: (a) N_eff is close to a restatement of roster weakness — Spearman **−0.958** against mean roster accuracy, and every diversity predictor loses its association once mean accuracy is partialled out; (b) N_eff is **negatively** associated with swarm accuracy, **−0.702** at k=3, against **+0.769** for mean accuracy. What survives partialling is accuracy *spread* (partial ≈ +0.7). **And no composition beats its own best single member by a detectable margin on any benchmark** (largest +0.015); on math500 every subset is worse by ≥0.097. Family diversity is unmeasurable here — all 7 models are 7 lineages, so the predictor is constant |
| **G2** | Coordinatability collapses on closed label spaces | **REFUTED AS WORDED — the measured direction is the opposite** | mean adversarial q̂ is *highest* on the most closed space: binary 0.557, 4-way multiple choice 0.283, open text 0.069. Restated below as G2' |

## HELD, WITH THE MECHANISM SEPARATED

| id | claim | status |
|---|---|---|
| **E2a** | AIP-windowed survives a plurality-sized Sybil bloc | **ROBUSTNESS HELD** — 0.672 at s=8 where every bloc defence collapses to 0.265; diff 0.407, **4.1×** floor |

## UNDETECTABLE AT THIS SCALE

| id | claim | why |
|---|---|---|
| **N1** | Whether inversion gain depends on swarm size | Measured 0.643 / 0.638 / 0.655 at N = 10 / 20 / 50; range **0.017** against a 0.068 floor. **Neither "grows with N" nor "invariant in N" is supported.** The experiment was underpowered to separate them. Every figure caption and the money-figure annotation must say so. |

## MITIGATIONS — corrected

| id | claim | status |
|---|---|---|
| **D1** | **M1 soft gate closes the exploitable mid-band** | **PARTIAL SUCCESS, re-anchored at f = 0.5.** Within the evasion band (below-ceiling cells) mean accuracy 0.463 → **0.623** (+0.160, **1.6×** floor) and worst case 0.082 → **0.200**. Over the *whole* attack space at the anchor the gain is 0.457 → 0.557 = **+0.100, exactly at the floor** — so the defensible statement is the band-restricted one. Still fails pre-registered C1. |
| **D2** | **M2 randomized threshold helps** | **REFUTED, and the direction is UNDETECTABLE.** At the anchor M2 reads 0.444 against the hard gate's 0.457 over the whole space (−0.013) and 0.474 against 0.463 within the band (+0.011). Both are far below the 0.100 floor: randomising an uninformative statistic buys **no measurable discrimination**. The earlier wording "actively harmful" over-read a −0.059 difference that was itself below floor and is withdrawn. M3 (band 0.638) is not separable from M1 (0.623) either. |
| **D3** | The residual attack is invisible to any per-question coherence statistic | **PROPOSED — Proposition 1**, `paper/sections/appendix_indistinguishability.tex`. Empirical anchor q = 0.280 < 1/3 chance, at f = 0.5, in the matched-parity condition. Figure `paper/figures/evasion_band.pdf` plots the measured grid as points with no interpolation. **Anchor resolution is limited**: only 3 distinct q values in [0.15, 0.40] on C=4, q < 0.253 unreachable by this parameterisation (the interior minimum of q(p) = p² + (1−p)²/(C−1)), and f=0.7 is confounded by Byzantine numerical dominance. |

## INSTRUMENTATION DISCLOSURE

| id | note |
|---|---|
| **R1** | Two mitigation variants were initially inert and produced "identical to baseline" results that were mistaken for findings. M1 fell through to the hard gate for any channel above `max_blind_error_rate` (i.e. every adversarial channel); M2's jitter reached only the fixed-ceiling ablation, not the default binomial rule. Both were caught by the Task 3b probe tests (`tests/test_mitigation_variants.py`), which now assert variant activity *before* any verdict is read. Two published verdicts were retracted. This belongs in the reproducibility section: an artifact that can detect its own instrumentation failures is stronger evidence than one that cannot. |

## UNSUPPORTED — do not claim

| id | claim | why not |
|---|---|---|
| **U1** | AIP accuracy is monotone increasing in f | Only `aip_naive` shows this (math500 0.151 → 0.489), and it is catastrophic at low f. Not a property of the gated method. |
| **U2** | Results generalise beyond this population | 9 models (7 frozen roster + 2 weak-tier arm, never pooled), 6 benchmarks, one hardware regime, one adversary parameterisation. |
| **U3** | Longitudinal or external-information defences survive the residual attack | Stated as hypotheses in Proposition 1's scope note. **Neither has been evaluated against the gate-aware adversary.** |

## RESTATED AFTER AUDIT (Task 6)

Re-derived by `scripts/task6_audit_pending.py`; full tables in
`results/pending_claims_audit.md`. Each of these carried real data but a wording
that the data does not support.

| id | restated claim | evidence | vs floor |
|---|---|---|---|
| **H2'** | Against the **symbolic** coherent adversary at f=0.7, every *deployable* discard method reads **exactly 0.000** on all six benchmarks. The original wording is correct in this regime and this method class only | `aggregation/sweep.parquet` (mix_all4, p_obs=1.0, complete, always_wrong, matched@0.1) | an exact zero, not an estimate. AIP-gated holds 0.978 arc / 0.976 gsm8k / 0.902 mmlu / 0.842 medqa, but **also reads 0.000 on boolq and math500** — the separation there is UNDETECTABLE because both sides are at the floor of the scale |
| **H2''** | "…including oracle methods" is **REFUTED in both regimes** | same, plus `adversarial_sweep.parquet` | symbolic: Krum holds **0.815** on boolq and **0.745** on medqa. Real-LLM: multi-Krum **0.760** medqa and **0.610** boolq beat AIP by 0.267 and 0.305. Oracle knowledge of the Byzantine count buys real robustness; the deployability restriction is doing the work in this comparison and must be stated |
| **H2'''** | Against the **real-LLM** coherent adversary (semantic_negation) at f=0.7 **nothing reaches zero**: deployable discard spans 0.002 (sac, gsm8k) to 0.595 (majority, boolq) | `adversarial_sweep.parquet` (mix_all4, p_obs=1.0, matched@0.1) | AIP separates from the best deployable discard on **4 of 6**: arc **+0.268**, gsm8k **+0.857**, math500 **+0.372**, mmlu **+0.188**. medqa **+0.008 UNDETECTABLE**. boolq **−0.290 — AIP is measurably worse than plain majority vote** |
| **G2'** | What collapses on closed label spaces is not coordinatability but the adversary's **margin over honest agreement**. On 4-way multiple choice no attack's Wilson interval clears the honest receiver-conditioned q of 0.535, so there is no channel a coherence gate can license inversion on | `q_by_answer_space.parquet` + Wilson intervals at n=200/500 | mmlu −0.215, medqa −0.340, arc −0.260 below honest q; only gsm8k negation clears it (q̂ 0.674, CI [0.632, 0.714] vs honest 0.597) |
| **M2a** | Self-vote-share parity **is** a real confound, and the audit instrument is clean | `parity_audit.parquet`, 6048 rows | the entire unweighted/discard family is **exactly** parity-invariant across 3024 cells (max \|effect\| 0.0000); every weighted method responds |
| **M2b** | "AIP does not exploit self-vote share" is **REFUTED at f = 0.5** and UNDETECTABLE elsewhere | same | at the exact tie, **25 of 54** cells clear floor, mean signed effect **+0.443**, max **0.840**. Away from the tie, 4 of 378 cells (1.1%), max 0.135. At a 50/50 split the normalisation of the adversary's own vote, not the mechanism, decides the outcome |

**Consequence carried into the f = 0.5 anchor.** Any number stated at the tie must
name its parity condition. The gate-aware sweep constructs its aggregators with
`ParityConfig(0.1)` — the matched, confound-controlled condition — so the evasion
anchor is not contaminated by M2b. This is stated in the paper, not assumed.

## DATA GAPS — declared, not silently absorbed

| id | gap | consequence |
|---|---|---|
| **X1** | `configs/inversion_thresholds.yaml` carries no `receiver_conditioned_q` for the **binary** answer-space class, so BoolQ's honest coordinatability is uncalibrated (`honest_q = NaN`) | G2' is stated for the open and multiple-choice classes only. No margin-over-honest claim is made for BoolQ, and its ceiling of 1.000 is a placeholder, not a measurement |

## CORRECTIONS — issued values that were wrong, and what replaced them

A correction is a ledger entry. This section is append-only: an entry is never
edited away once the wrong value has been in a compiled artifact.

| id | date | what was wrong | replacement | root cause |
|---|---|---|---|---|
| **X2** | 2026-09-20 | **Honest-coincidence constant reported as 0.806.** Live in `paper/numbers.tex` as `\numHonestQMmlu` and used in four manuscript sites — `abstract.tex`, `introduction.tex`, `design_laws.tex`, `reproducibility.tex` — including the abstract. 0.806 is the MMLU **all-pairs** mean of `q_true`, of which **1 of 21 pairs** clears R2's `n_both_wrong ≥ 30`; the powered MMLU value is 0.594. | **0.560**, pooled over the **9** powered 4-way multiple-choice pairs that exist across MMLU, MedQA and ARC (**337** jointly-wrong items). Emitted as `\numHonestQMc`, with `\numHonestQMcPairs` and `\numHonestQMcItems` alongside so the power is visible in the macro set. `\numHonestQMmlu` is removed, not redefined. | `paper_numbers.py` computed `q_true.mean()` with **no power filter**. R2 was enforced in the AUC analysis and in the figures but never in the numbers generator, so an underpowered estimate reached the abstract. Filter now applied at source; **four of six benchmarks (ARC, BoolQ, GSM8K) have zero powered pairs and can support no honest-coincidence estimate at all.** `numHonestQMathfive` carried the same missing filter but is unaffected in value — MATH-500 is 21/21 powered. |

**Audit performed at the same time.** Every other generator macro derived from a
pairwise statistic was checked. `numWithinModelPhi`, `numCrossModelPhi` and
`numPhiRatio` come from `phi_point`, which is computed over all 200–500 tasks and
is not conditioned on both agents being wrong, so no power filter applies.
`numNeff*` were already basis-restricted (see C2). `numMmluHonestFloor` and
`numGsmkHonestFloor` are the gate's **configured** constants from
`inversion_thresholds.yaml`, not measurements; §Law 3 of the manuscript now says
so explicitly and notes that the measured pooled value 0.560 exceeds the
configured 0.535, so the G2′ ordering is unchanged under either.

## WITHDRAWN PERMANENTLY

| id | claim | why it is not in the paper |
|---|---|---|
| **B2** | Deterrence / payoff flattening: AIP removes the payoff advantage that coherent lying enjoys against a discard defence | **WITHDRAWN PERMANENTLY — do not re-add.** The condition set before the audit was that this enters the paper only if G2 passed Task 6 re-verification. G2 failed: refuted as worded, restated as G2′. The author confirmed the withdrawal as permanent on 2026-09-11, on two grounds — the conditional was "G2 passes", and the bandit data shows **payoff, not conduct**: the adversary never switched arms within the horizon tested, so even on its own terms the claim was about arm values rather than observed behaviour. The data stays in the release (`adversarial/bandit_log.parquet`) and is cited nowhere in the manuscript. This row is closed and is not subject to re-litigation. |

---

## MEASUREMENT FLOOR — clean floor adopted 2026-09-20 (X3)

**Adopted:** GSM8K **0.066**, elsewhere **0.099**. Previously 0.068 / 0.100.

`resample_noise` now excludes T=0.7 rows that are still cap-pinned after X3 —
**97 rows**, all on GSM8K, the only benchmark carrying a second pass:
phi4_mini_reasoning 44, olmo3_32b_think 24, granite42_30b 13, qwen38_27b 11,
gemma4_31b 4, ministral3_14b 1. The count is pinned by
`tests/test_artifact.py::TestT07TruncationExclusionIsPinned` so it can neither
shrink nor grow silently.

**The mechanism, which is why keeping them was not the safe choice.** A
cap-pinned row is scored wrong in the greedy pass *and* in the T=0.7 pass.
Manufactured-wrong times manufactured-wrong is a spurious *agreement*, and
agreement is exactly what `sigma_resample` measures the absence of. So the
contaminated rows suppressed the measured flip rate and pulled the floor down —
they made the study look more sensitive than its data supports.

**The bias is signed by resample class**, which is why it was not visible in the
aggregate:

- `answer_level` — kept 0.0645, clean 0.0628. Keeping it was CONSERVATIVE.
- `path_level` — kept 0.0642, clean **0.0664**. Keeping it was **ANTI-conservative**.

(Written as a list, not a table: the ledger's first table column is the claim-id
namespace, and `tests/test_paper.py` reads it as one.)

`path_level` is where truncation concentrates — both of its models are reasoning
models — so the class that mattered was biased the unsafe way, and the aggregate
direction hid it.

**How it was caught.** A sensitivity check run before adopting anything, not a
gate. The proposal on the table was to keep the contaminated rows and document
them as a conservative bound; the check showed that bound was false for
`path_level` and the decision was re-opened. Recorded here because the check
earned its place in the recompute chain.

**Verdict re-check: no verdicts flip.** Both floors moved *down*, so a verdict
could only move UNDETECTABLE → DETECTABLE. Every floor-compared claim was
re-tested at exact precision. The one apparent flip, **D1/M1**'s whole-space
gain, was an artefact of comparing the *rounded* published effect (+0.100)
against the old floor: the exact value is **+0.100667**, which already cleared
0.1003. Its published description as "exactly at the floor" is a rounding
artefact — it is marginally above, at 1.01× floor, and was so before this change.

**Deferred to the pre-camera-ready checklist.** Re-downloading the four evicted
checkpoints (~225 GB) would repair the 52 rows they own and make the floor clean
by construction rather than by exclusion. Optional; the exclusion is sound
without it.
