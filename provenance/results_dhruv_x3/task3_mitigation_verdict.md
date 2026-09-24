# Task 3 — mitigation verdict (corrected, second re-run)

Evaluated against `results/task3_preregistration.md` (commit `84341b0`), committed
before the experiment. No criterion was adjusted.

**Two earlier passes of this experiment were void** and are retracted. In the
first, M1 fell through to the hard gate for any channel whose blind error
exceeded `max_blind_error_rate` — which is every adversarial channel — so it never
executed in the sub-ceiling band it exists to defend. In the second, M2's jitter
was wired only into the fixed-ceiling ablation while the study's default rule is
a binomial tail test, leaving M2 inert. Both produced "identical to the hard gate"
results that looked like findings about the statistic and were findings about the
wiring. The probe tests in `tests/test_mitigation_variants.py` now pin activity,
and are checked before any verdict is read.

## Activity check (must pass before the verdict means anything)

| variant | cells differing from hard | max abs difference |
|---|---|---|
| M1 soft | **48 / 81** | 0.860 |
| M2 random | **20 / 81** | 0.600 |
| M3 both | **51 / 81** | 0.718 |

All three are demonstrably live.

## Verdict

| variant | C1 worst-case gain (≤0.15) | C3 retention (≥80%) | **VERDICT** |
|---|---|---|---|
| hard (baseline) | 0.915 | 100% | — |
| M1 soft | **0.915** ❌ | ✅ | **FAIL** |
| M2 random | **0.915** ❌ | ✅ | **FAIL** |
| M3 both | **0.915** ❌ | ✅ | **FAIL** |

C2 (cost at f=0) was not evaluable: f=0 is not on this sweep's grid. Recorded as
not-reached, and immaterial — C1 alone disqualifies.

## What the worst case hides, and why it matters

C1 is a max over the whole coherence grid, so it reports only where each defence
is weakest. The full curve says something the max does not. MedQA, f=0.7,
accuracy (lower = attack succeeds):

| q realised | hard | M1 | M2 | M3 |
|---|---|---|---|---|
| 0.253 | **0.000** | **0.000** | **0.000** | **0.000** |
| 0.280 | **0.000** | **0.000** | **0.000** | 0.032 |
| 0.333 | 0.050 | 0.284 | 0.050 | 0.312 |
| 0.413 | 0.143 | **0.747** | 0.567 | 0.600 |
| 0.520 | 0.005 | **0.865** | 0.005 | 0.723 |
| 0.653 | 0.910 | 0.890 | 0.310 | 0.888 |
| 1.000 | 0.832 | 0.832 | 0.832 | 0.853 |
| **mean over grid** | **0.350** | **0.563** | **0.291** | **0.536** |

Three things follow.

**M1 works, and substantially.** It lifts mean accuracy across the attack space
from 0.350 to 0.563, a gain of 0.213 against a 0.100 floor. It nearly closes the
mid-band: at q=0.52 the hard gate is destroyed (0.005) and M1 holds 0.865. The
soft gate is a real defence over most of the adversary's operating range, and
reporting it as a failure without this table would be misleading.

**M2 makes things worse.** Mean 0.291 against the hard gate's 0.350. Randomizing
the threshold costs calibration precision and buys nothing, and M3's 0.536
against M1's 0.563 says M2 detracts there too. This is a clean negative result
about a plausible idea.

**No variant survives the low-coherence end.** At q = 0.253 and 0.280 every
defence, including the baseline, is driven to 0.000. That is where C1's worst case
lives, and it does not move.

## Conclusion

The pre-committed reading — that the evasion band is fundamental — holds, but the
corrected data locates it far more precisely than "threshold-based gating fails".

The surviving attack sits at **q ≈ 0.25–0.28, at or below the 1/3 chance
coherence for a four-option space**. At that operating point the adversarial
channel's coherence is not merely close to honest disagreement, it is
indistinguishable from independent error by construction: two agents lying
independently on a C=4 space collide at 1/3, and the adversary is colliding at
0.25–0.28. There is no coherence statistic that separates those populations,
because they have the same coherence.

So the limit is real and the mitigations are informative rather than failed:
**smoothing the gate closes most of the band (M1, +0.213 mean), randomizing it
does not (M2, −0.059), and neither reaches the residual attack because the
residual attack lives where the statistic itself carries no signal.** A defence
that closes it must use information coherence does not contain.
