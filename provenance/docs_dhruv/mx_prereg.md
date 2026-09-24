# MX pre-registration — AIP as a trust layer for multi-stage agentic pipelines

**Status: DRAFT, awaiting go-ahead. No GPU runs have been made against this plan.**
Written 2026-09-20. Format follows the R1–R6 decision rules in
`configs/decision_rules.yaml`. Every prior-work number carries its `claims.md` ID.

---

## 0. Deviations from the work-package spec

Three, all proposed rather than taken. Two are forced by measurement.

### D-1. GSM8K dropped — accepted, justified by R1

The spec named GSM8K. Cached-roster GSM8K accuracy is 0.802–0.972, so R1 marks
those cells HEADROOM-LOW. With k=3 and one liar, the liar is inert whenever both
honest replicas are right; at 0.95 accuracy that is ~90% of problems. The
containment differential would sit under the 0.068 GSM8K floor. **Dropped.**

### D-2. MATH-500 cannot test MX-H1 either — this reverses my own earlier advice

I recommended MATH-500 on headroom grounds and was right about headroom and wrong
about the mechanism. L1 (attractor-rider) needs the **both-honest-wrong-and-
agreeing** cell to exist. Measured over all 36 cross-model pairs of the cached
roster, that cell's frequency is:

| benchmark | attractor-cell rate | P(both wrong) | note |
|---|---|---|---|
| **MedQA** | **0.0786** | 0.150 | highest; C=4 closed space |
| MMLU | 0.0737 | 0.104 | C=4 |
| BoolQ | 0.0657 | 0.066 | C=2, degenerate — agreement is forced |
| ARC | 0.0179 | 0.024 | saturated |
| GSM8K | 0.0107 | 0.030 | saturated |
| **MATH-500** | **0.0050** | 0.334 | open text: plenty of error, almost no *shared* error |

MATH-500 has the most honest error and the least *coordinated* honest error — the
open answer space means two wrong models pick different wrong answers. At 0.0050,
a 200-problem × 5-seed run yields ≈5 exploitable cells. **MX-H1 and MX-H4 would be
untestable on MATH-500.** This is the same G2′ effect the paper already reports,
arriving as an experimental-design constraint.

**Proposed: MedQA as the primary MX benchmark.** It maximises the attractor cell
(0.0786, ~16× MATH-500) and retains headroom (cached-roster accuracy 0.450–0.925,
no cell at the 0.93 R1 ceiling except gemma4_31b at 0.925, which is not in the
proposed roster).

### D-3. MATH-500 retained as a contrast arm, not the primary

Running the same pipeline on MATH-500 at reduced scale tests the **answer-space
dichotomy inside a pipeline**: where the honest attractor exists, L1 bites and AIP
can invert it; where it does not, L1 is inert and there is nothing to contain.
That contrast is a stronger result than either benchmark alone and costs ~25%
more GPU time. Included as MX-H5, dropped first if budget binds.

### Task-list size

The spec says "all 500 problems". The frozen MedQA list is **N=200**
(`configs/task_lists.json`), as is MATH-500's — the name is the dataset's, not our
sample's. Extending to 500 means new task IDs and a new `task_list_hash`,
breaking comparability with every cached MedQA result. **Proposed: keep N=200 and
buy power from seeds (5 seeds × 200 = 1000 problem-instances).** Power is
calculated for that in §6.

---

## 0b. Amendments after Phase 1

### Amendment A2 — phi4_mini_reasoning generation budget, 2026-09-20 — **SUPERSEDED by X3, same day**

> **Superseded.** The main-study cache audit that A2's evidence prompted found
> the same defect across the whole roster, not just phi4: 1,036 of 13,500
> generations hit the cap and 944 carried a manufactured answer. That became
> correction campaign X3, which amends `configs/models.yaml` itself to 3072 for
> every entry and regenerates the affected rows. The MX-local override is
> therefore a duplicate of what the registry now says, and `MX_MAX_TOKENS` has
> been removed; MX reads its budgets from the registry like everything else.
> A2's reasoning stands and its number is unchanged — it is absorbed, not
> reversed. The gate-6 consequence below also changes: after X3 the reference
> cache is itself repaired, so MX and the cache are generated at the same budget
> and **exact reproduction is again the expectation for phi4**. See item 7 of
> the X3 scope: the combined paste carries a gate-6 column against the corrected
> cache.

### Amendment A2 — phi4_mini_reasoning generation budget, 2026-09-20

**Change.** `phi4_mini_reasoning` generates at **3072** tokens in MX, via an
MX-local override (`MX_MAX_TOKENS` in `scripts/mx_phase1.py`). **`configs/models.yaml`
is not edited** and remains byte-identical: it is the frozen roster the main
study's manifests reference, and changing it would alter the provenance of
already-published results.

**Evidence.** At its 2048 registry budget in the Phase 1 probe, phi4 was
truncated and discarded as `truncated_without_commitment` on **25% of solver,
46% of critic and 45% of refiner** generations. The p90 completion length sat at
exactly 2048 in all three stages -- the cap, not the model. Of the 100
completions that finished naturally, the length distribution was
**p50 = 1003, p90 = 1538, max = 2013**, so the requirement has a real ceiling
around 2000 rather than an unbounded tail. Prompts are short (p50 ~293 tokens,
max ~427), leaving roughly **3670 tokens of headroom** against
`max_model_len` 4096. 3072 therefore sits inside the context window with margin
and clears the observed tail. For contrast, `llama32_3b` truncated 0/157 and
`ministral3_14b` 8/392 at their own budgets.

**Consequence for Gate 6, stated because it changes the pass criterion's
meaning.** MX phi4 will now be **less truncated than the main-study cache**,
which was generated at 2048. Exact reproduction is therefore no longer the
expectation for that model. Agreement within the measurement floor remains the
pass bar, and any residual delta is to be attributed, explicitly, to the
truncation differential rather than treated as unexplained. The other two models
are unaffected and exact reproduction remains the expectation for them --
`ministral3_14b` already reproduced its cached accuracy to +0.000 on matched task
ids.

**Follow-up required before the combined paste.** The affected phi4 stage rows
are regenerated under the new budget and Gate 6 re-reported with full
denominators. In the failed run, phi4's Gate 6 denominator was 15 tasks of 20
because five tasks had every sample discarded.

## 1. Prior-work numbers, with ledger IDs

Every number below is quoted from `results/claims.md` or recomputed from the
parquet it cites. The work-package prompt's figures were shorthand and are
**not** used.

| quantity | ledger value | ID |
|---|---|---|
| Within-model error correlation (GSM8K) | **0.652** | `C1` |
| Cross-model error correlation (GSM8K) | **0.444** | `C1` |
| N_eff, homogeneous 10-agent swarm | **1.50** (range **1.07–1.91**) | `C2` |
| N_eff, best heterogeneous | **2.46** | `C2` |
| N_eff, cross-model-only basis | 6.86 — **upper bound, not comparable** | `C2` |
| Inversion gain at f=0.7 | **+0.645** | `I1` |
| Inversion cost at f=0 | **0.001** (below floor) | `I2` |
| Attenuation law exact at C=2 | MAE **0.000** | `C3` |
| Measurement floor | **0.068** GSM8K / **0.100** elsewhere | — |

**Honest-attractor rate — corrected.** The prompt quoted 0.806. That is the MMLU
**all-pairs** mean of `q_true`, of which **1 of 21 pairs** clears R2's
`n_both_wrong ≥ 30`. Powered values:

| benchmark | all-pairs | **powered** | powered pairs |
|---|---|---|---|
| MMLU | 0.806 | **0.594** | 1 / 21 |
| MedQA | 0.563 | **0.556** | 8 / 21 |
| MATH-500 | 0.015 | **0.015** | 21 / 21 |
| ARC / BoolQ / GSM8K | 0.880 / 1.000 / 0.418 | — | **0 / 21** |

**MX uses 0.556 (MedQA, 8 powered pairs, 337 jointly-wrong items).** Not 0.806.

> **Escalation.** `\numHonestQMmlu` = **0.806** is live in `paper/numbers.tex:57`
> and used in four manuscript sites — `abstract.tex:18`, `introduction.tex:52`,
> `design_laws.tex:150`, `reproducibility.tex:41`. The generator
> (`paper_numbers.py`) computes it with **no power filter**. Reported in full in
> the accompanying message; not fixed unilaterally because choosing the
> replacement is a judgement call.

---

## 2. Pipeline definition

**Stages:** `solver → critic → refiner`, MedQA, C=4 multiple choice.
Answer extraction and equivalence reuse `src/aip/tasks/medqa.py` unchanged.

**Per stage:** k replicas produce candidate stage outputs. The stage aggregator
applies the arm's rule. The aggregator **also produces its own independent answer
to the stage task**, used as the receiver anchor — the same receiver-anchoring
convention as the paper (`G1`), and the reason the gate survives past f=0.5.

**Stage inputs:** stage 2 receives the stage-1 aggregate; stage 3 receives the
stage-2 aggregate. This sequential dependency is the cascade channel and is why
stage-2/3 generations cannot be shared across arms.

**All generation goes through `aip.models.inference.ModelRunner`.** Direct
`vllm.LLM(...)` construction fails with `TypeError: '>' not supported between
instances of 'dict' and 'int'` in vLLM input validation; the harness is immune
because it pins `attention_backend="TRITON_ATTN"` and normalises tokenizer
output. Recorded in RUNLOG 2026-09-20.

---

## 3. Arms and roster

| arm | structure | aggregation |
|---|---|---|
| **P0** | k=1 | none (vulnerability baseline) |
| **P1** | k=3 same-model | plurality |
| **P2** | k=3 same-model | AIP gate |
| **P3** | k=3 cross-family | AIP gate |
| **P4** | k=3 cross-family | plurality |

P1/P2 share generations; P3/P4 share generations. Aggregation is offline, so the
rule contrast costs no GPU time.

### Roster, with MedQA accuracies from the cache

Only five models are in the local HF cache; the 27–32B roster members would need
re-download (~60 GB each against 61 GB free disk), so MX is restricted to the
cached 3–14B band — which is the band the spec preferred anyway.

| model | family | MedQA acc | mean completion tokens | usable as a pipeline stage? |
|---|---|---|---|---|
| `ministral3_14b` | Mistral | **0.785** | 237 | yes |
| `phi4_mini_reasoning` | Microsoft | **0.580** | 1299 | yes |
| `llama32_3b` | Meta lineage | **0.540** | 95 | yes |
| `ministral_8b` | Mistral | 0.535 | **8** | **no** — emits a bare letter |
| `olmo2_7b` | AI2 | 0.450 | **4** | **no** — emits a bare letter |

`ministral_8b` and `olmo2_7b` produce no reasoning text for a critic to act on;
they are excluded on those grounds, which also avoids pooling the weak-tier arm
with the frozen roster.

**Cross-family trio (P3/P4):** `ministral3_14b` + `phi4_mini_reasoning` +
`llama32_3b`. Three distinct lineages, all frozen-roster, all cached.

**Same-model arm (P1/P2):** `ministral3_14b`, the strongest usable cached model
(0.785). Replicas drawn at T=0.7 from a sampled pool.

> **Declared confound.** The spec asks for a comparable accuracy band. The trio
> spans **0.540–0.785**, a 0.245 spread, because the cached band offers nothing
> tighter that also generates reasoning text. MX-H2 is therefore confounded
> between cross-family *decorrelation* and *mean competence*, in the same way S1
> found N_eff confounded with roster weakness. **Mitigation:** report
> containment against the stage-level mean accuracy as a covariate, and treat a
> positive MX-H2 as suggestive, never as isolating diversity. If this cannot be
> disentangled, MX-H2 is reported as confounded — not as a win.
>
> `llama32_3b` is the paper's designated primary adversary. Using it as an
> *honest* replica here is a deliberate role change and is stated wherever MX
> results appear.

---

## 4. Fault injection

Exactly **1 of the 3 critic-stage replicas** is replaced by a coherent liar.
Both variants are **offline transforms of the cached honest generations** — no
adversarial inference, the same discipline as `falsified_confidence` and `burst`.

- **L1 attractor-rider.** The liar endorses the honest replicas' most common
  **wrong** answer, computed from the cached honest replica answers for that
  task. Where the honest replicas are not both wrong, the liar falls back to a
  uniform wrong choice; the fallback rate is logged and reported.

  > **Threat model — L1 is an INFORMED adversary.** L1 reads the honest replicas'
  > answers on the same task before choosing its own, which is what "riding the
  > attractor" means operationally. It is informed in exactly the sense of the
  > gate-aware section: it conditions on the defence's own inputs. L2 is
  > **uninformed** — a fixed lie chosen without reference to what the honest
  > replicas said. Proposition 1 distinguishes precisely these two, and MX-H4 is
  > the pipeline-scale test of that distinction: an uninformed lie sits above the
  > honest coherence ceiling and the gate fires; an informed one can sit inside
  > it.
- **L2 fixed-lie.** The existing shared-lie rule: a designated wrong answer,
  constant across replicas and tasks.

**Secondary config:** inject at the **solver** stage instead, to test
stage-position sensitivity. Run only after the primary config completes.

---

## 5. Negative control — HARD invariant, not a threshold

The spec's "≤ 0.005 accuracy cost" is **below the 0.100 MedQA floor** and cannot
be evaluated as an accuracy comparison. Replaced with decision-equality:

With **no liar firing**, all of the following must hold, or the run is an
instrumentation failure and stops:

1. **P2 stage outputs equal P1's exactly**, and P3's equal P4's exactly, on every
   stage of every problem. Not "within a tolerance" — identical.
2. **Zero INVERT on any honest channel.**
3. **DISCARD only with a logged regime-change flag.**
4. Any deviation is traced to a specific gate firing, then fixed or explicitly
   justified in the ledger before Phase 3 proceeds.

**Reported separately either way:** the honest-channel INVERT count as the
**false-inversion rate**. That number matters to the paper in its own right
(`G1` currently reports ≤ `numHonestInvertedMax`% of honest channels inverted at
single-question scale; the pipeline figure is new).

---

## 6. Metrics and power

**Cascade rate** = P(final answer wrong | fault injected AND the no-fault
pipeline solved that problem correctly). **Containment** = 1 − cascade rate.

Also: end-task accuracy delta vs negative control; per-arm INVERT/TRUST/DISCARD
rates; false-inversion rate on honest channels; stage-level φ and N_eff computed
from the actual replicas (reusing `subset_swarm_rho_bar` / `effective_swarm_size`).

### Pre-registered power

Using the project's own MDE formula, `MDE = √2 × 1.96 × σ`, `σ = √(p(1−p)/n)` at
the worst case p=0.5:

| eligible cells n | MDE |
|---|---|
| 80 | 0.155 |
| 150 | 0.113 |
| **193** | **0.100** ← floor |
| 300 | 0.080 |
| 700 | 0.052 |
| 1000 | 0.044 |

- **Whole-sample containment** (all problems the no-fault pipeline solved):
  5 seeds × 200 problems ≈ 1000 instances, of which ≈70% are eligible ⇒ n≈700,
  **MDE ≈ 0.052**. Comfortably powered.
- **Attractor-cell-restricted containment** (the cells L1 actually exploits, rate
  0.0786): n ≈ 1000 × 0.0786 ≈ **79**, **MDE ≈ 0.155**. **Underpowered.** An L1
  effect restricted to attractor cells must exceed 0.155 to be reportable.

**Pre-registered legitimate outcome:** a containment differential below its
cell's floor is reported as **"undetectable at this scale"** — never as "small",
"no effect", or a win for either arm. This is an outcome, not a failure.

---

## 7. Hypotheses

Each becomes ledger entry `MX1..MX5`, marked confirmed / refuted / undetectable /
withdrawn.

### MX-H1 — P2/P3 contain more cascades than P1 under L1

*Derivation from measured quantities, not assumption.* Plurality with 2 honest +
1 liar fails only when the liar's vote is decisive. Given the MedQA attractor rate
**0.0786** (§0) and P(both honest wrong) **0.150**, the liar is decisive in two
regimes: honest replicas disagree (one right, one wrong) — the liar breaks the tie;
or both honest are wrong and agree (0.0786) — already lost, liar irrelevant. The
exploitable regime is therefore **disagreement**, frequency ≈ P(exactly one wrong)
≈ 2·p·(1−p) with p≈0.3 ⇒ **≈0.42** of problems, of which the liar flips those the
no-fault pipeline got right.

**Prediction:** containment(P2) − containment(P1) > 0 under L1, magnitude bounded
above by the disagreement rate. Reportable only if ≥ **0.100**.

### MX-H2 — containment(P3) > containment(P2)

*Derivation.* Cross-family φ **0.444** vs within-model **0.652** (`C1`) ⇒ N_eff
**2.46** vs **1.50** (`C2`). Lower correlation ⇒ honest replicas fail together
less often ⇒ fewer attractor cells for L1 ⇒ higher containment. **Predicted sign:
positive.** Rough magnitude: the attractor rate scales with pairwise agreement on
wrong answers; a φ drop of 0.208 is a ~32% relative reduction, so if the
same-model attractor rate is ~0.08, cross-family should be ~0.054, a differential
of order **0.03** — **below the 0.100 floor**.

> **Pre-registered expectation of undetectability.** MX-H2 is predicted to be
> *directionally* positive but **too small to detect at this scale**. It is
> registered so the null is interpretable, not because we expect to confirm it.
> S1 in the main paper already found N_eff anti-correlated with swarm accuracy
> (−0.702), so a diversity benefit is not assumed here.

### MX-H3 — gate overhead < 5% wall-clock per stage

*Grounded in `E3`:* AIP-gated costs 79.6 µs/decision against 77.8 µs for its
inversion-free ablation — **+1.7 µs, +2.2%**, at single-question scale, and
generation is ~5073× the aggregation cost. **Prediction: overhead well under 5%,
likely under 0.1% of stage wall-clock**, because the stage is generation-bound.

### MX-H4 — L2 easier to contain than L1

*Rationale.* L2's fixed lie sits **above** the honest coherence ceiling and the
gate fires on it. L1 rides the honest attractor and therefore approaches
Proposition-1 territory: coherence indistinguishable from honest coincidence.
**Prediction: containment(L2) > containment(L1) in every AIP arm.**

### MX-H5 — the MX-H4 gap widens on MedQA relative to MATH-500

*Registered per the spec's item 7, with the direction corrected by measurement.*
The spec expected the gap to widen on MATH-500 "because attractor cells are
frequent where honest error is high". **The measurement says the opposite**:
MATH-500 has the most honest error (P(both wrong) 0.334) and the fewest attractor
cells (0.0050), because the open answer space prevents shared error. MedQA has
less honest error (0.150) and 16× the attractor rate (0.0786).

**Corrected prediction:** the MX-H4 gap is **larger on MedQA than on MATH-500**,
and on MATH-500 L1 should be nearly inert — approaching L2 from below, or
producing no measurable cascade at all. Attractor frequency, not honest error
rate, is the governing quantity.

---

## 8. Phases, seeds, stop conditions

| phase | content | gate |
|---|---|---|
| **1** | Harness + probe tests: 20 problems, all arms, no fault | per-stage outputs logged, gate decisions logged, liar firings logged, determinism (same seed → identical outputs) |
| **2** | Negative control at full N | §5 invariant holds exactly, or STOP |
| **3** | Main runs: 2 fault configs × 5 arms × 5 seeds | — |
| **3b** | Secondary config (solver-stage injection) | only if Phase 3 clean |
| **4** | Analysis, ledger entries MX1–MX5, report | — |

**Seeds:** 5 fixed seeds derived from `stable_seed` over the canonical MX config,
same SHA-256 discipline as every other phase. Seeds control replica draws from the
sampled pool and which replica is the liar — **not** generation, which is cached
once.

**Stop conditions.** Blocked > 2 h wall-clock → stop and report. Negative control
fails → stop and fix. Any hypothesis's data missing → reported as missing, never
inferred. Phase 0 estimate > 60 GPU-h → propose a cut (drop P4 first, then reduce
to 3 seeds, then drop MX-H5) — never silently shrink scope.

---

## 9. GPU budget

Re-estimated for MedQA generation lengths from the cache (`ministral3_14b` 237
mean tokens, `phi4_mini_reasoning` 1299, `llama32_3b` 95; trio mean ≈ 544).

- Distinct generation contexts ≈ 60 per problem across arms, fault configs and
  stages, after exploiting: stage-1 sharing, offline fault injection, and
  P1/P2 + P3/P4 aggregation sharing.
- Pool factor ×2 for seed variation without regenerating.
- 200 problems × 60 × 2 ≈ **24,000 generations** ≈ **13M output tokens**.
- H100 at 0.90 utilisation, 3–14B models, batched: 3,000–8,000 output tok/s.

**Estimate: 0.5–1.2 GPU-hours of generation**, plus model load/swap overhead
(3 models, ~45–90 s per load, many swaps) ≈ **1–3 GPU-hours** for the primary
config. MX-H5's MATH-500 contrast arm adds ~25%. **Total ≈ 2–4 GPU-hours.**

Far under the 60 h cap. The margin is large enough that the earlier 10–25 h
figure I gave is withdrawn — it assumed GSM8K's N=500 and did not account for the
frozen MedQA list being N=200.

---

## 10. What lands in the paper

Proposed: a new section, *AIP as a trust layer for agentic pipelines*, after the
gate-aware section. One figure (containment by arm × liar variant, with the floor
drawn), one table (gate telemetry and false-inversion rate per arm). If MX-H1 is
refuted or undetectable, the section still lands — as the negative result that
stage-local gating does not contain cascades at this scale, reported at the same
length as a positive one would be.

---

## Sign-off

Nothing in this document has been run. Phase 1 begins only on explicit go-ahead,
and only after the three deviations in §0 are accepted or overruled.
