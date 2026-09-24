# Phase D — real-LLM adversarial evaluation

Adversarial generation: **19.4 min actual against a 24.1 min projection and a
3.0 GPU-hour cap** — 0.32 GPU-hours used, no cuts needed. 2 adversary models
(llama32_3b, ministral_8b) x 4 inference attacks x 3 benchmarks x 100 tasks,
written to `data/cache_adversarial/`. Phase A honest caches were read-only
throughout. Sweep: 13,104 rows, 500 bootstrap resamples, CIs everywhere.

## 0. The MATH-500 gate fix (before/after)

Three changes, in the order they were needed:

1. **Binomial significance test** replaces the fixed-ceiling comparison. A
   channel is inversion-eligible only if its coincidence count is significantly
   above the calibrated ceiling under a one-sided exact binomial test at
   `alpha = 0.05/9` (Bonferroni over the channels one receiver judges). The
   fixed-ceiling rule is retained behind `coherence_test="fixed_ceiling"`.
2. **Minimum-support floor, derived not chosen**: the smallest `n` at which even
   the strongest possible evidence — every joint dissent a coincidence — clears
   alpha, i.e. the smallest `n` with `q0**n <= alpha`. That gives **4** events
   for open-text, **6** for open-numeric, **10** for multiple choice under the
   original ceilings. Classes where honest agents coincide more need more
   evidence to overturn, which is the right ordering.
3. **The ceilings were calibrated on the wrong quantity**, and this was the
   actual cause. Phase B2 measured `q = P(same | both WRONG)`, which needs gold
   labels. The gate has none, so it conditions on `P(same | both dissent from
   ME)`. The two coincide only when the receiver is almost always right. On
   MATH-500, where models are wrong 28–66% of the time, honest peers score
   **0.143** on the receiver-conditioned statistic against the **0.020** measured
   for the gold-conditioned one. `scripts/calibrate_receiver_coherence.py` now
   measures the statistic the gate actually uses; config is v2.

A fourth change was needed for the gate to work at all at high `f`: the honest
agreement reference is now found by **largest-gap cluster splitting** rather than
a quantile, because no fixed quantile survives the range — at `f = 0.7` seven of
nine peers are adversarial and even the upper-half median has gone over to them.

**Before/after, MATH-500, AIP-gated, mix_all4, coherent adversary:**

| f | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|---|
| before (Phase C) | 0.305 | 0.162 | 0.235 | 0.321 | 0.478 | 0.490 | 0.502 | 0.503 |
| after | **0.499** | **0.469** | **0.445** | 0.266 | 0.210 | 0.000 | 0.160 | 0.000 |
| honest channels wrongly inverted, before | 82% | 83% | 86% | 86% | 87% | 90% | 100% | 100% |
| honest channels wrongly inverted, after | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** |

The low-`f` bleed is fixed: +0.194 at `f=0`, +0.307 at `f=0.1`, and honest
inversion goes to zero at every `f` on **all three** benchmarks. The cost is
real and I am not hiding it: MATH-500 high-`f` inversion is lost, because the
symbolic adversary there draws its lie from honest wrong answers, so the
receiver itself sometimes produces the same value and the agreement clusters
overlap. GSM8K and MMLU are unaffected and stay flat in `f`.

## 1. Pilot gate

First pilot **failed** on one criterion: `semantic_hallucination` returned the
*correct* answer on 4 of 5 tasks. One fix applied — a two-step prompt that makes
the model commit to C internally, then requires a reported value that differs
from C. Second pilot passed all criteria and was not re-run again:

| attack | wrong | structure |
|---|---|---|
| always_wrong | 5/5 | — |
| semantic_negation | 5/5 | **ratio to gold = 2.0 on all five** (perfectly deterministic) |
| semantic_hallucination | 5/5 | ratios 1.11, 1.09, 0.0, 0.75, 0.75 (idiosyncratic) |
| rushing | 4/5 | conditions visibly on the peer answers it was shown |

The negation completion states the rule inline ("Applying the negation rule, we
get: 2 * 366 = 732"); the hallucination completion invents a question-specific
misread ("I thought it was 200 instead of 180"). That is exactly the
structured/unstructured split the theory needs.

## 2. Measured adversarial channel coherence

Cross-model coherence `q̂` — how often the two adversary models give the *same*
wrong answer — is the quantity that decides invertibility:

| attack | gsm8k | math500 | mmlu |
|---|---|---|---|
| **semantic_negation** | **0.620** | 0.230 | 0.290 |
| always_wrong | 0.030 | 0.020 | 0.380 |
| **semantic_hallucination** | **0.040** | 0.010 | 0.350 |
| rushing | 0.100 | 0.210 | 0.220 |

On GSM8K, negation is **15x more coherent than hallucination** (0.620 vs 0.040).
That single contrast is the paper's invertibility claim in one number.

## 3. Per-attack verdicts (mean over f >= 0.5, mix_all4)

| benchmark | attack | q̂ | AIP | best discard | gain | verdict |
|---|---|---|---|---|---|---|
| gsm8k | semantic_negation | 0.620 | **0.930** | 0.630 | **+0.300** | inverts |
| gsm8k | falsified_confidence | 0.620 | **0.926** | 0.567 | **+0.359** | inverts |
| gsm8k | semantic_hallucination | 0.040 | 0.890 | 0.859 | +0.031 | no gain |
| gsm8k | always_wrong | 0.030 | 0.863 | 0.913 | −0.050 | loses |
| gsm8k | rushing | 0.100 | 0.826 | 0.899 | −0.073 | loses |
| gsm8k | burst | 0.620 | 0.577 | 0.783 | −0.207 | loses |
| gsm8k | noise | — | 0.713 | 0.967 | −0.253 | loses |
| math500 | semantic_negation | 0.230 | **0.462** | 0.253 | **+0.208** | inverts |
| math500 | falsified_confidence | 0.230 | **0.462** | 0.335 | **+0.127** | inverts |
| math500 | semantic_hallucination | 0.010 | 0.433 | 0.333 | +0.099 | inverts |
| mmlu | *(every attack)* | 0.22–0.38 | — | — | −0.03 to −0.32 | loses |

**The theory holds where it predicts it should, and the failures are
interpretable rather than random.** Inversion pays exactly when the adversarial
channel is coherent (negation, and falsified_confidence which is negation plus a
rewritten self-report) and does not pay when it is not (hallucination,
always_wrong, rushing, noise).

## 4. The honest negative result: hallucination is not invertible

`semantic_hallucination` produces answers that are wrong 72–94% of the time and
fluent enough to pass, but with `q̂ = 0.010–0.040` on the open-ended benchmarks:
two adversaries essentially never make the *same* mistake. AIP's gain over the
best discard baseline is **+0.031 on GSM8K** — within noise — against +0.300 for
negation on the same benchmark and the same swarm.

This is the result the paper should state plainly. **Adversarial coherence, not
adversarial success, is what AIP converts into signal.** An adversary that is
merely effective is not exploitable; an adversary that is effective *and
consistent* is. A deployment facing idiosyncratic hallucination should expect
AIP to perform like a discard method, because that is what it degrades to.

## 5. Falsified confidence

The attack rewrites only the self-reported 0–100 confidence on the cached
negation outputs; answers and logprob confidences pass through byte-identical,
so any difference is attributable to the self-report channel alone.

| method | negation | falsified | delta |
|---|---|---|---|
| gsm8k confidence_weighted_selfreport | 0.364 | 0.356 | −0.008 |
| gsm8k confidence_weighted (logprob) | 0.687 | 0.673 | −0.014 |
| gsm8k **aip_gated** | 0.930 | 0.929 | **−0.000** |
| mmlu confidence_weighted_selfreport | 0.530 | 0.498 | **−0.032** |
| mmlu confidence_weighted (logprob) | 0.500 | 0.500 | 0.000 |
| mmlu **aip_gated** | 0.571 | 0.571 | **0.000** |

The prediction is confirmed in direction and exactly for AIP: **AIP's accuracy is
invariant to 0.000 on all three benchmarks**, and the self-report reader is the
only method that moves. The magnitude is small (worst case −0.032) because the
manipulation reweights identical votes rather than changing any answer — the
self-report channel was already the weakest input in the sweep (0.364 versus
0.687 for the logprob channel on GSM8K under plain negation). The cleaner
statement is that **the logprob surface is unfalsifiable by construction**, which
is why Phase A cached the two surfaces separately.

## 6. Deterrence

An epsilon-greedy adversary chose between coherent lying and noise for 40 epochs
against each defence (gsm8k, f = 0.6, heterogeneous swarm):

| defence | value(coherent lie) | value(noise) | mean swarm accuracy |
|---|---|---|---|
| SAC filter-refine | **0.957** | 0.350 | **0.073** |
| AIP-gated | **0.195** | 0.167 | **0.807** |

Against the discard baseline, coherent lying is worth 0.96 and the adversary
destroys the swarm — accuracy 0.073. Against AIP the same arm is worth 0.195,
barely above noise, and accuracy holds at 0.807. The timeseries shows the
mechanism: against AIP the estimated value of coherent lying **decays from 1.0 to
0.19 and crosses below noise around epoch 28** as the gate accumulates evidence,
while against SAC it climbs monotonically to 0.96.

**The predicted collapse onto noise did not happen, and I am reporting that
rather than the headline.** Within 40 epochs the bandit still plays coherent lie
95% of the time against AIP, because the two arms end up nearly equal (0.195 vs
0.167) and epsilon-greedy has little to separate. The honest claim is weaker than
"AIP forces the adversary into noise" and still substantial: **AIP flattens the
adversary's payoff landscape**, removing the 2.7x advantage coherence enjoys
against a discard defence. Deterrence in the payoff sense, not in the observed
behaviour, over this horizon.

## 7. What surprised me

**The negation attack does not cohere on multiple choice.** Its cross-model
`q̂` is 0.620 on GSM8K but only 0.290 on MMLU — *below* the 0.535 rate at which
honest models coincide there. The reason is that the negation rule is unambiguous
for numbers ("report 2*V") and subjective for options ("the most contrary
option"): llama and ministral simply disagree about which distractor is contrary.
**A structured lie needs a rule the liars can compute identically**, and closed
label spaces make that harder, not easier, which is the opposite of what I
expected.

> **Correction (Phase E).** I originally wrote that MMLU's loss followed because
> "the gate correctly never fires". The Phase E channel-decision decomposition
> shows that is **wrong**: on MMLU the gate inverts **89% of adversary channels
> and 0% of honest channels**, the same targeting it achieves on GSM8K. The
> confusion was between two different coherence measures — the *cross-model* `q̂`
> of 0.290 (do llama and ministral tell the same lie?) and the *within-swarm*
> coherence the gate actually computes, which is near 1.0 between Byzantine
> agents replaying the same model's cache. The gate fires correctly; what fails
> is the payoff. With the adversarial bloc split across two models telling
> *different* lies, inverting those channels pushes evidence away from several
> different wrong answers and spreads the positive mass over the remaining
> options — including other wrong ones — instead of concentrating it on the
> truth. Inversion is informative only when the liars agree with each other, and
> on MMLU they do not. See `results/phase_e_report.md` §3.

**Ministral resists the always_wrong injection on GSM8K**, complying only 29% of
the time versus 100% for llama. Prompt-injected wrongness is model-dependent in a
way that materially changes the attack's strength.

**Burst defeats the gate completely** (−0.207 on GSM8K) even though its coherent
epochs use the same negation outputs that AIP inverts at +0.300. AIP-gated and
AIP-trust-only are *identical* under burst, so the gate never fires: pooling
statistics across alternating regimes averages a coherent channel and a noisy one
into something that clears neither threshold. A non-stationary adversary beats a
stationary detector, and the fix — windowed or change-point statistics — is a
Phase E design question, not something to patch in silently.

**AIP loses to SAC under pure noise** (−0.253 on GSM8K). With nothing coherent
present, AIP's inversion machinery contributes variance while SAC's filtering
contributes signal. The control behaves as it should, but the size of the gap is
a genuine cost of the method that belongs in the paper.
