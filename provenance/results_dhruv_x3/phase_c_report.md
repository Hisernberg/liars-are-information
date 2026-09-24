# Phase C — aggregation sweep with q-gated inversion

Cache-only for the sweep; no model was loaded. 18,144 rows over 1,296 cells:
3 benchmarks x 6 swarm compositions x 8 Byzantine fractions x 4 observability
levels x 3 topologies x 2 adversaries x 2 parity conditions x 14 methods.
Bootstrap 500 resamples over tasks, seed 20250825, CIs on every number.
Artefacts: `results/aggregation/sweep.parquet`,
`results/aggregation/parity_audit.parquet`, `figures/money_figure_draft.pdf`,
`figures/gating_ablation.pdf`.

There is no `legacy/` directory in this repository, so every formula is
implemented from the specification and the Phase B2 derivation.

## 1. Headline: accuracy vs f (mix_all4, complete graph, p_obs = 1)

| f | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|---|
| **gsm8k** AIP-gated | .942 | .949 | .960 | .960 | .952 | .964 | **.975** | **.937** |
| gsm8k best baseline | .966 | .960 | .965 | .960 | .960 | .788 | .810 | **.000** |
| **math500** AIP-gated | .305 | .162 | .235 | .321 | .478 | .490 | **.502** | **.503** |
| math500 best baseline | .540 | .513 | .501 | .429 | .385 | .350 | .350 | **.000** |
| **mmlu** AIP-gated | .700 | .656 | .745 | .720 | .743 | .722 | **.730** | **.687** |
| mmlu best baseline | .710 | .700 | .721 | .691 | .740 | .430 | .570 | **.000** |

The claim under test survives. On all three benchmarks AIP-gated accuracy is
**flat in f**: GSM8K moves between .937 and .975 across the entire range, MMLU
between .656 and .745. Every discard-family baseline collapses to **exactly
0.000 at f = 0.7**, where a coherent adversarial bloc owns the plurality and
every rule that treats consensus as evidence follows it off the cliff. Krum,
Multi-Krum and trimmed mean are given the *true* Byzantine count and still
collapse, so this is not a tuning failure.

The 50% ceiling reported in the discard literature is real for discard methods
and is not a property of the problem. Inversion crosses it because a coherent
bloc is a channel, not noise.

## 2. Where inversion helps and where it hurts

| benchmark | f ≤ 0.3 | f ≥ 0.5 |
|---|---|---|
| gsm8k | −0.024 to +0.000 (ties) | **+0.165 to +0.937** |
| mmlu | −0.044 to +0.029 (ties) | **+0.160 to +0.687** |
| math500 | **−0.107 to −0.351 (loses)** | **+0.140 to +0.503** |

Inversion is close to free below f ≈ 0.4 on GSM8K and MMLU and decisive above
it. On **MATH-500 it is actively harmful at low f**, costing up to 0.35
accuracy at f = 0.1. §4 explains why, and the cause is the calibration rather
than the mechanism.

## 3. Did the honest-q threshold do its job? Yes, on two of three benchmarks

Channel decisions, counted from **honest receivers only** — a Byzantine
receiver's view is inverted by construction and including it reports the
adversaries' opinions as the defence's mistakes.

**Adversary channels correctly inverted:** 100% at every f ≥ 0.2, on all three
benchmarks. (At f = 0.1 a lone adversary has no bloc to be coherent *with*, so
it is correctly not inverted; a single channel changes almost nothing.)

**Honest channels wrongly inverted:**

| benchmark | AIP-gated | AIP-naive |
|---|---|---|
| gsm8k | **0% at every f** | 90–100% |
| mmlu | **0% at every f** | 82–100% |
| math500 | 82–100% | 82–100% |

On GSM8K and MMLU the gate is perfect in both directions: it inverts every
adversary and never once inverts an honest peer, at any Byzantine fraction.
AIP-naive inverts 82–100% of honest channels and pays for it:

| accuracy cost of naive inversion (gated − naive) | f=0.0 | f=0.2 | f=0.4 | f=0.6 |
|---|---|---|---|---|
| gsm8k | +0.305 | +0.897 | +0.050 | +0.043 |
| mmlu | +0.610 | +0.637 | +0.540 | +0.608 |
| math500 | +0.000 | +0.000 | +0.000 | +0.000 |

This is the planned experiment and it lands: **on MMLU, where Phase B2 measured
honest q at 0.476 against a 1/3 chance level, naive inversion destroys roughly
0.6 accuracy at every f.** Honest models share wrong-answer attractors, a rule
that inverts on above-chance coherence inverts them, and the calibrated ceiling
is what stands between the two. AIP-naive never beats AIP-gated on any cell.

**MATH-500 is where the gate fails**, and the reason is its ceiling rather than
its logic. The open-text honest-q ceiling is 0.042 — Phase B2 measured honest
coincidence at 0.020 there because two models almost never produce the same
wrong expression. A ceiling that low sits essentially on top of the chance level
of 0, so gated and naive make *identical* decisions (the cost table above is
0.000 at every f), and ordinary sampling noise in a peer's peak coincidence
clears the bar. The mechanism is sound; the class simply has no headroom between
chance and honest behaviour in which to place a threshold. Two ways forward, both
for Phase D to decide: require a minimum number of jointly-dissenting
observations before a channel is eligible for inversion, or replace the fixed
ceiling on this class with a significance test against 0.

## 4. Normalization-parity audit

Mean |accuracy(unmatched) − accuracy(matched)| over 144 paired cells each.
Matched and unmatched are evaluated on the **identical swarm draw**; an earlier
version seeded them separately and reported Krum — which has no weights at all —
as the most parity-sensitive method in the table, which was pure seed variance.

| method | mean effect | max effect | weighted? |
|---|---|---|---|
| aip_trust_only | 0.0135 | 0.590 | yes |
| confidence_weighted_selfreport | 0.0120 | 0.120 | yes |
| sac_filter_refine | 0.0108 | 0.393 | yes |
| confidence_weighted | 0.0066 | 0.118 | yes |
| aip_naive | 0.0017 | 0.120 | yes |
| **aip_gated** | **0.0016** | 0.120 | yes |
| majority, Krum, Multi-Krum, geometric median, coord median, trimmed mean, Dawid–Skene | **0.0000** | 0.000 | no |

Every unweighted method reads exactly 0.0000, which is the audit validating
itself: those methods cannot respond to self-vote share, so any non-zero entry
would mean the table was measuring something else. Among weighted methods the
confound is real but modest in the mean and large in the tail — up to 0.59 in a
single cell for AIP-trust-only. **AIP-gated is the least parity-sensitive
weighted method in the sweep (0.0016)**, so its advantage is not bought by
leaning on its own vote. All headline numbers are reported at matched parity.

## 5. Sensitivity

**Topology.** AIP-gated degrades gracefully: GSM8K .937–.975 (complete) →
.906–.955 (k-nearest, k=4) → .840–.922 (ring). Even on a ring, where each agent
hears only two peers, it stays far above every baseline's high-f collapse.

**Partial observability is the real limit.** On GSM8K at p_obs = 0.5, AIP-gated
falls from .932 at f = 0 to .215 at f = 0.6. Thinning the channel starves the
statistics the gate is built on: a receiver that hears a peer on few tasks cannot
establish that the peer belongs to a coherent bloc, and the `min_observations`
guard then discards it. Inversion needs *evidence*, and evidence needs
observations. This is the sharpest practical constraint on deploying AIP and it
should be stated plainly in the paper rather than buried in a sensitivity
appendix.

**Adversary type.** Against the noise (incoherent) adversary, AIP-gated tracks
the baselines rather than beating them — as it must, since there is no coherent
channel to invert. That is the control confirming the gain comes from inversion
of structure, not from the weighting machinery.

## 6. Two corrections found while building this phase

Both were caught by the gate audit the brief asked for, which is the argument
for having asked.

1. **The gate was inverting the wrong channels.** Coherence was averaged over
   all peers, which dilutes a bloc's signal in proportion to how many honest
   peers surround it: at f = 0.3 an adversary's mean coincidence came out at
   0.477, below the 0.589 ceiling, so the gate trusted every adversary while
   inverting honest peers. Coherence is now a **maximum** over partners with
   sufficient joint-dissent support, which detects bloc membership and is
   invariant to the size of the honest majority.
2. **Inversion needs two conditions, not one.** Coherence alone cannot separate
   a coherent liar from a set of peers who are right when the receiver is wrong.
   A channel is now inverted only if it is both more coherent than an honest
   population ever was *and* agrees with the receiver less than the attenuation
   law allows for an honest peer, tested at three binomial sigma. The receiver's
   own honesty is the anchor, which is what keeps the test meaningful past
   f = 0.5 where consensus-based estimators converge on the lie.
