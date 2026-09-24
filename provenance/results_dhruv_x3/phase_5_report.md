# Phase 5 — sleeper (E1) and Sybil (E2)

162 E1 rows, 144 E2 rows. Headline benchmarks (math500, medqa, mmlu), six defenders,
500 bootstrap resamples. Both experiments target *admitted* limitations of published
systems rather than weaknesses invented for them.

**One prediction held with a caveat that guts its stated mechanism. The other failed
outright.** Both are reported as measured.

---

## E1 — sleeper infiltration

Adversary honest for k ∈ {10, 25, 50} rounds, then switches to coherent lying
(cached `semantic_negation` output). f ∈ {0.1, 0.3, 0.5}.

**Predicted:** reputation keeps trusting the sleeper because standing is earned
before it is spent; AIP-windowed reclassifies within ~W rounds.

**Measured — the opposite ordering:**

| method | cover acc | betrayal acc | drop |
|---|---|---|---|
| `majority` | 0.709 | 0.677 | **0.032** |
| `reputation_decay` | 0.730 | 0.690 | **0.040** |
| `confidence_weighted` | 0.829 | 0.738 | 0.091 |
| `sac_filter_refine` | 0.865 | 0.748 | 0.117 |
| `aip_gated_w20` | 0.830 | 0.701 | 0.129 |
| `aip_gated` | 0.862 | 0.711 | **0.150** |

Reputation decay has the *smallest* drop of any memory-carrying defender and AIP the
largest. **The prediction is not supported.**

### The drop metric flatters weak defenders

`majority` has the smallest drop of all and has no memory whatsoever — it cannot be
betrayed because it never extended trust. Drop is not normalised by what a method had
to lose, and cover accuracy ranges from 0.709 to 0.865 across these rows. Ranked by
what actually matters, absolute accuracy *during* the betrayal:

| method | betrayal accuracy |
|---|---|
| `sac_filter_refine` | **0.748** |
| `confidence_weighted` | 0.738 |
| `aip_gated` | 0.711 |
| `aip_gated_w20` | 0.701 |
| `reputation_decay` | 0.690 |
| `majority` | 0.677 |

AIP sits mid-pack. It loses more than reputation decay but ends up above it, because
it started higher. Neither framing rescues the prediction: **AIP does not handle the
sleeper better than the baseline it was supposed to beat**, and the honest conclusion
is that E1 does not support the claim it was built to test.

Drop is flat in k (AIP-gated 0.151 / 0.154 / 0.146 at k = 10 / 25 / 50), so a longer
cover story buys the adversary nothing measurable here — itself evidence against the
"reputation is earned before it is spent" mechanism, which predicts the advantage
should grow with k.

### The recovery metric is too loose to cite

`rounds_to_recover` never returned −1 (nothing failed to recover inside the horizon)
but returned 0 for **91 of 162** configurations. A 5-round window at 0.05 tolerance
against a noisy cover baseline is cleared too easily. The range is real (0–16 rounds)
so the metric is not broken, but it is underpowered and **should not be quoted as a
finding** without a tighter tolerance and a longer window.

---

## E2 — Sybil coherence amplification

Adversary controls s ∈ {1, 2, 4, 8} of ten identities, all broadcasting the same lie,
against a control where the same identities each lie differently.

**Predicted:** bloc defences degrade with s; AIP flat or *improving*, because a larger
bloc is a more coherent channel and coherence is what inversion feeds on.

**Measured — coherent bloc:**

| method | s=1 | s=2 | s=4 | s=8 | Δ(s8−s1) |
|---|---|---|---|---|---|
| `aip_gated_w20` | 0.738 | 0.749 | 0.683 | **0.672** | **−0.066** |
| `aip_gated` | 0.778 | 0.765 | 0.495 | 0.521 | −0.257 |
| `sac_filter_refine` | 0.781 | 0.816 | 0.669 | 0.265 | −0.516 |
| `majority` | 0.783 | 0.787 | 0.583 | 0.265 | −0.518 |
| `reputation_decay` | 0.786 | 0.798 | 0.519 | 0.265 | −0.521 |
| `confidence_weighted` | 0.790 | 0.803 | 0.714 | 0.265 | −0.525 |

**The robustness half holds, strongly.** Every bloc defence collapses to 0.265 at
s=8 — the point where the bloc is the plurality and a consensus-following method
follows it. AIP-windowed holds **0.672**, about 2.5×, and degrades by 0.066 where the
others lose 0.52.

### The stated mechanism does not hold

The disagreeing-Sybil control is what makes this a test rather than a demonstration:

| method | coherent Δ | disagreeing Δ |
|---|---|---|
| `aip_gated_w20` | −0.066 | **−0.077** |
| `aip_gated` | −0.257 | −0.263 |

AIP-windowed behaves **the same whether the bloc is coherent or not**. If bloc size
were amplifying `q̂` and feeding inversion, the coherent condition should separate
from the control; it does not. So the measured advantage is *general resilience to a
plurality-sized bloc*, not coherence amplification.

That distinction matters for what may be claimed. Supportable: **AIP-windowed retains
2.5× the accuracy of every bloc defence at s=8, where they all collapse to the
adversary's answer.** Not supportable: that this happens *because* the bloc is
coherent. The control rules that explanation out.

---

## Verdicts

| claim | verdict |
|---|---|
| E1: reputation trusts the sleeper, AIP reclassifies | **NOT SUPPORTED** — reputation drops least; AIP most |
| E1: advantage grows with cover length k | **NOT SUPPORTED** — drop is flat in k |
| E2: bloc defences degrade with s | **SUPPORTED** — all collapse to 0.265 at s=8 |
| E2: AIP flat or improving in s | **PARTIALLY** — flat-ish (−0.066), never improving |
| E2: because bloc size amplifies coherence | **NOT SUPPORTED** — control behaves identically |

Two of five predictions survive, and the one clean positive result has a mechanism
different from the one proposed. That is a usable section — "AIP-windowed is the only
defence that survives a plurality-sized Sybil bloc, and not for the reason we
expected" is a stronger paper than a clean confirmation would have been, because the
control is what makes it credible.
