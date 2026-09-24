# Phase B2 — the multiclass attenuation law

Cache-only; no model was loaded. Derivation in `docs/attenuation_law.md`.
Bootstrap: 1000 resamples over tasks, seed 20250825, 95% percentile CIs.
Artefacts: `results/correlation/multiclass_validation.parquet`,
`results/correlation/q_pairwise.parquet`.

## 0. A correction to Phase B, found by doing the derivation

Deriving the law properly exposed a bug in the Phase B validation. The binary
estimator was fed `correctness_agreement` — the rate at which two agents are
*both right or both wrong*. **That is not a blind observable**: computing it
requires knowing who was right, which is the very label the estimator is
supposed to do without. The observable an agent actually has is **answer
agreement**. The two coincide only for a binary answer space, which is exactly
why the mistake was invisible on GSM8K and loud on MATH-500.

Corrected, the binary baseline is *better* than originally reported, so the
Phase B conclusions strengthen — but the bar for this phase is correspondingly
higher:

| benchmark | binary law, as reported in Phase B | binary law, corrected |
|---|---|---|
| gsm8k | 0.0498 | **0.0298** |
| math500 | 0.2058 | **0.1275** |
| mmlu | 0.1909 | **0.0809** |

Everything below is measured against the corrected baseline, which is the
honest comparison. The `phi` correlation results in Phase B are unaffected —
those are ground-truth measurements and were always meant to use labels.

## 1. Does estimating q beat assuming q = 1?

| benchmark | binary law | multiclass law | paired Δ MAE [95% CI] |
|---|---|---|---|
| gsm8k | 0.0298 | **0.0274** | −0.0024 [−0.0147, +0.0098] |
| math500 | 0.1275 | **0.0205** | −0.1070 [−0.1537, +0.0243] |
| mmlu | **0.0809** | 0.0870 | +0.0061 [−0.0141, +0.0203] |

The paired difference is the right test — both estimators see the identical
resampled tasks, so comparing their two marginal CIs (which overlap heavily)
would badly understate the power.

**Against the stated success criterion, one of two targets is met.**

* **GSM8K must not regress: met.** Δ = −0.0024, CI spans zero; no regression.
* **MATH-500 must beat the binary law: met on the point estimate,** by 6.2×
  (0.0205 vs 0.1275), but the paired CI reaches +0.024 and so does not exclude
  zero. The distribution is heavily right-skewed: on most resamples the
  multiclass law is far better, on a minority the EM fits badly and it is worse.
  I would report this as a large but not yet statistically certified improvement
  at n = 100 tasks.
* **MMLU must beat the binary law: not met.** The multiclass law is slightly
  *worse* (0.0870 vs 0.0809), well within noise. §4 explains why, and the
  explanation is the interesting part.

Marginal error rates are a cleaner win, on all three benchmarks:

| benchmark | binary law | multiclass law |
|---|---|---|
| gsm8k | 0.0283 | **0.0275** |
| math500 | **undefined** | **0.0607** |
| mmlu | 0.1325 | **0.1179** |

MATH-500 is the notable cell. The binary triplet identity needs
`2·agreement − 1 > 0`, and open-ended answer agreement sits below 0.5, so the
binary estimator correctly refuses to return a number at all. The multiclass
route recovers every marginal to within 0.06 — it turns "cannot be estimated"
into "estimated well", which is the practical difference for a swarm that has to
bootstrap trust with no labels.

## 2. How well is q itself recovered?

| benchmark | MAE(q) [95% CI] | mean q true | mean q blind |
|---|---|---|---|
| gsm8k | 0.238 [0.063, 0.523] | 0.518 | 0.281 |
| math500 | 0.147 [0.095, 0.279] | 0.016 | 0.163 |
| mmlu | 0.130 [0.091, 0.282] | 0.469 | 0.339 |

The blind estimate **understates** q on GSM8K and MMLU and overstates it on
MATH-500. The understatement is the direction predicted in
`docs/attenuation_law.md` §6: the Dawid–Skene fit assumes conditional
independence given the truth, so shared attractors — the exact thing q measures —
are partly absorbed into the latent-label posterior instead of appearing in q.
For a detector of shared mechanism this is the conservative direction: a blind q
that already looks high is, if anything, an underestimate.

## 3. Honest q per pair (ground truth)

| benchmark | pair | both wrong | q [95% CI] | chance |
|---|---|---|---|---|
| math500 | llama32_3b + ministral_8b | 37 | 0.054 [0.000, 0.139] | 0 |
| math500 | llama32_3b + olmo2_7b | 48 | 0.042 [0.000, 0.102] | 0 |
| math500 | llama32_3b + phi4_mini_reasoning | 25 | 0.000 [0.000, 0.000] | 0 |
| math500 | ministral_8b + olmo2_7b | 38 | 0.000 [0.000, 0.000] | 0 |
| math500 | ministral_8b + phi4_mini_reasoning | 23 | 0.000 [0.000, 0.000] | 0 |
| math500 | olmo2_7b + phi4_mini_reasoning | 25 | 0.000 [0.000, 0.000] | 0 |
| mmlu | ministral_8b + olmo2_7b | 25 | **0.560** [0.368, 0.750] | 1/3 |
| mmlu | ministral_8b + phi4_mini_reasoning | 16 | 0.562 [0.312, 0.812] | 1/3 |
| mmlu | llama32_3b + ministral_8b | 24 | 0.542 [0.321, 0.759] | 1/3 |
| mmlu | llama32_3b + olmo2_7b | 27 | 0.444 [0.250, 0.630] | 1/3 |
| mmlu | olmo2_7b + phi4_mini_reasoning | 17 | 0.412 [0.187, 0.667] | 1/3 |
| mmlu | llama32_3b + phi4_mini_reasoning | 17 | 0.294 [0.091, 0.533] | 1/3 |

GSM8K pairs are omitted from the per-pair table as uninterpretable: the models
are jointly wrong on only 1–8 tasks each, so every CI spans essentially [0, 1].

Pooling every jointly-wrong event on a benchmark gives the powered test:

| benchmark | events | pooled q [95% CI] | chance | verdict |
|---|---|---|---|---|
| math500 | 196 | **0.020** [0.004, 0.042] | 0 | as predicted, ≈ 0 |
| mmlu | 126 | **0.476** [0.364, 0.589] | 0.333 | **above chance** |
| gsm8k | 23 | 0.478 [0.143, 0.775] | 0 | far above 0, underpowered |

## 4. The finding: honest models share wrong-answer attractors

**The MATH-500 prediction holds exactly.** Pooled honest q is 0.020 — two honest
models that are both wrong on a competition-maths problem essentially never
produce the *same* wrong expression. Four of the six pairs have q identically
zero across 23–38 joint errors. The wrong-answer space is large and honest
failures scatter across it, exactly as the derivation assumes.

**The MMLU prediction fails, and this is the headline.** Pooled honest q is
**0.476 with CI [0.364, 0.589]**, entirely above the 1/3 chance level — 43%
above it. Five of six pairs sit above chance on the point estimate. These are
four models from four different families with different pretraining corpora, and
when two of them are both wrong on a multiple-choice question they pick the
*same* distractor about half the time rather than a third of the time. Honest
agents share wrong-answer attractors: some distractors are simply more seductive
than others, and that seductiveness is shared across model families.

**GSM8K deserves a flag despite being underpowered.** With only 23 joint errors
the CI is wide, but pooled q = 0.478 on a benchmark whose answer space is
*unbounded integers* is striking — coincidence has essentially no chance mass to
hide behind, so 0.478 cannot be luck. When two models both miss a grade-school
word problem they tend to make the *same* arithmetic mistake. This is the same
phenomenon as MMLU seen without the floor of a fixed option set, and it deserves
a properly powered follow-up on more tasks.

### Why this matters for inversion, and why it explains §1

The paper's inversion argument reads high `q_ij` as evidence of a shared
mechanism, and a shared mechanism as something invertible. This result says
**high q is not sufficient evidence of adversarial coherence**: on MMLU an
entirely honest swarm reaches q ≈ 0.48, and on GSM8K roughly the same. An
inversion rule keyed on coherence alone would invert honest agents on exactly
the questions they find hardest — the worst possible failure mode, since those
are the questions where the swarm most needs its honest members. **Inversion
therefore needs a threshold calibrated against the honest-q baseline for the
answer space in use, not blind trust in coherence.** Phase C should treat honest
q as a per-benchmark nuisance parameter to be subtracted, not as zero.

It also explains the MMLU result in §1. The multiclass estimator is built on
Dawid–Skene, which assumes conditional independence given the truth; shared
attractors violate that assumption directly. So the blind q on MMLU comes back
at 0.339 — essentially chance — when the truth is 0.476. The multiclass law
therefore behaves on MMLU as if it had assumed `q = 1/3`, while the binary law
assumes `q = 1`; both are wrong in opposite directions and land at comparable
MAE. The failure to beat the baseline on MMLU is not a coding defect but the
same finding measured from the other side: **the benchmark where honest agents
share attractors is exactly the benchmark where an independence-assuming blind
estimator cannot measure that sharing.** Breaking it would need a model that
admits correlated errors — a coupled or hierarchical Dawid–Skene variant — which
is a Phase C decision, not something to slip in here.
