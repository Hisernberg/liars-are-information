# Phase B — error correlation and blind-estimator validation

Cache-only. No model was loaded; the Phase A parquet caches were read read-only.
Bootstrap: 2000 resamples over tasks, seed 20250825, 95% percentile CIs.
Artefacts: `results/correlation/*.parquet`, `figures/correlation_heatmap.pdf`.

## Q1 — Is within-model correlation higher than cross-model correlation?

Yes, decisively, and by about 0.4 in phi. On GSM8K, where both quantities are
measurable on the identical 100 tasks, within-model correlation (the same model
resampled at T=0.7 against its own greedy answer) averages **phi = 0.627**
(range 0.501–0.836, n=4), while cross-model correlation averages **phi = 0.227**
(range 0.029–0.480, n=6) — a difference of **+0.399**, a ratio of **2.76×**.
Every within-model point estimate sits above every benchmark's cross-model mean.
Swapping in a different model therefore decorrelates errors far more than
resampling the same model does, which is the concrete sense in which a
homogeneous swarm is closer to one epistemic agent than a heterogeneous one.

The effective-size table makes the cost legible. A 10-agent homogeneous swarm is
worth between **1.17 and 1.82** independent agents; the best two-model mix
reaches **2.62–2.74**. Heterogeneity buys roughly 1.5× more effective agents,
not 5×, and *no* composition gets close to 10. Phi-4-mini-reasoning is the
sharpest case: it has the highest within-model correlation (0.836) and agrees
with its own greedy answer on 96% of tasks even at T=0.7, so a homogeneous
reasoning swarm collapses hardest, to N_eff = 1.17. Sampling temperature is a
weak decorrelator for a model trained to reason its way to one answer.

## Q2 — Does correlation track accuracy gap, benchmark, or family?

Benchmark and difficulty, strongly; accuracy gap, not at all. Across the 18
cross-model pairs, correlation with the pair's accuracy gap is **+0.098** —
essentially nothing — while correlation with the pair's mean error rate is
**+0.672**. Benchmark means follow the same ordering: GSM8K 0.227, MMLU 0.429,
MATH-500 0.446, so correlation roughly doubles on the benchmarks where models
fail more often. The mechanism is shared task difficulty rather than mismatched
ability: where models are accurate (GSM8K, 87–93%) the few errors they make are
idiosyncratic, and where they are often wrong (MATH-500, 28–66% error) they fail
on the *same* problems. For the paper this is the more useful framing — swarm
redundancy is worst exactly where a swarm would be most valuable.

Two pairs deserve naming. **Ministral-8B + OLMo-2 7B on MMLU** is the most
correlated pair measured (phi = +0.584, CI [0.425, 0.740]) despite an accuracy
gap of only 0.05, sitting **+0.227 above** what a regression on accuracy gap
predicts. They are the closest pair in the registry by scale and role — 8.0B and
7.3B general-purpose instruct models — which suggests scale and training role
couple errors more tightly than model family does; note that they come from
entirely different families and pretraining corpora and are still the most
redundant pair we have. **OLMo-2 7B + Phi-4-mini-reasoning on GSM8K** is the
opposite extreme and the only effectively independent pair (phi = +0.029, CI
[-0.115, 0.293], spanning zero): different families, different scales, and one
is a reasoning model. If a heterogeneous swarm is to be assembled, that axis —
reasoning versus non-reasoning — buys more independence than family alone.

## Blind estimator validation (attenuation law, no gold labels)

> **Correction (Phase B2).** The first version of this section fed the estimator
> `correctness_agreement` — the rate at which two agents are *both right or both
> wrong*. That is not a blind observable: computing it requires knowing who was
> right. The genuinely observable quantity is **answer agreement**, what an agent
> actually hears on the wire. The two coincide only for a binary answer space,
> which is precisely why the mistake was invisible on GSM8K and loud on
> MATH-500. The table below is the corrected result; the corrected estimator is
> *better* than originally reported, so the Phase B conclusions strengthen rather
> than weaken. The `phi` correlations in §Q1 and §Q2 are unaffected — those are
> ground-truth measurements and were always meant to use labels.

| benchmark | pairs | MAE [95% CI] | mean blind | mean truth |
|---|---|---|---|---|
| gsm8k | 6 | **0.0298** [0.0124, 0.0549] | 0.3217 | 0.2919 |
| math500 | 6 | 0.1275 [0.0796, 0.1822] | −0.1400 | −0.0125 |
| mmlu | 6 | 0.0809 [0.0457, 0.1172] | 0.1300 | 0.0491 |
| **overall** | 18 | **0.0794** [0.0520, 0.1097] | 0.1039 | 0.1095 |

*(Previously reported, using the non-blind input: 0.0498 / 0.2058 / 0.1909 /
0.1488.)*

The instrument still works far better on GSM8K than elsewhere, and the reason is
structural rather than a coding error. The law is derived for a **binary** answer
space, where two wrong agents necessarily agree; then
`2 P(agree) − 1 = (1−2e_i)(1−2e_j)` exactly. With a K-ary space and wrong answers
spread out, `P(agree) = (1−e_i)(1−e_j) + e_i e_j q_ij`, so applying the binary
form understates the product by exactly `e_i e_j (1 − q_ij)` — a bias second
order in the error rates. GSM8K escapes because error rates there are 7–16%, so
`e_i e_j ≈ 0.01`. On MATH-500, where `e_i e_j ≈ 0.2` and honest `q ≈ 0`, the
blind estimate reads −0.14 against a truth of −0.01. Phase B2 derives the
generalized law and estimates `q_ij` rather than assuming it.

Induced marginal error rates show the same split: mean absolute error **0.080**,
but 0.002–0.052 on GSM8K, 0.057–0.233 on MMLU, and **undefined on MATH-500** —
there the triplet identity requires `2·agreement − 1 > 0`, and open-ended answer
agreement sits below 0.5, so the estimator correctly refuses to return a number
rather than inventing one.

The unit tests confirm the estimator is correct as specified: on synthetic binary
channels it recovers both the pairwise product and each agent's marginal error
rate inside theory-derived binomial bands (`tests/test_correlation.py`). So this
is a limitation of the law's assumption, not of its implementation: **the binary
attenuation law is trustworthy only where agents are individually strong and the
answer space is effectively binary.**

## Effective swarm size (N = 10)

| composition | benchmark | rho_bar | N_eff |
|---|---|---|---|
| homogeneous: Phi-4-mini-reasoning | gsm8k | +0.836 | **1.17** |
| homogeneous: Llama-3.2 3B | gsm8k | +0.604 | 1.55 |
| homogeneous: Ministral 8B | gsm8k | +0.565 | 1.64 |
| homogeneous: OLMo-2 7B | gsm8k | +0.501 | 1.82 |
| mixed: Ministral + OLMo-2 | gsm8k | +0.373 | 2.30 |
| mixed: Ministral + Phi-4-mini-R | gsm8k | +0.382 | 2.25 |
| mixed: OLMo-2 + Phi-4-mini-R | gsm8k | +0.313 | **2.62** |
| mixed: Llama-3.2 + OLMo-2 | gsm8k | +0.402 | 2.16 |
| mixed: Llama-3.2 + Phi-4-mini-R | gsm8k | +0.432 | 2.05 |
| mixed: Llama-3.2 + Ministral | gsm8k | +0.527 | 1.74 |

GSM8K mixes use the true 5+5 pair census — 20 same-model pairs and 25
cross-model pairs — rather than the cross-model correlation alone, which would
flatter the mix. MATH-500 and MMLU mixes are in the parquet but use cross-model
correlation only, because within-model correlation was measured on GSM8K alone
(it needs the T=0.7 second sample); those figures are therefore **upper bounds**
on N_eff and are labelled as such in `effective_swarm_size.parquet`.

## Data policy

* The 7 known non-reproducing task ids are **retained**, never dropped. Two fall
  inside the frozen 100 for each benchmark; every pair row carries
  `n_flagged_nonreproducing = 2`.
* Phi-4-mini-reasoning × MMLU has 30/100 NaN self-reported confidences. Phase B
  uses no self-reports, so no row is affected. Missing-data handling remains a
  Phase C decision.
* Llama-3.2 3B × MMLU has 1/100 failed extraction. It is scored incorrect, kept
  in the analysis, and counts as disagreement with every other agent.
* phi is reported as undefined, not zero, when either error vector is constant.
