# PRE-REGISTRATION — Task 3 mitigation experiment

Committed BEFORE the experiment is run. Success criteria are fixed here and are
not to be revised after seeing results.

## The vulnerability

The gate-aware adversary parks at coherence just below the published honest-q
ceiling. Measured best attack sits at q ~ 0.28-0.33 with attacker gain up to
+0.92 accuracy (medqa/mmlu, f=0.7, accuracy driven to 0.000). Inversion gain is
exactly 0.000 for every q below the ceiling. The hard, published threshold is the
vulnerability: it is a cliff with a known location.

## Variants under test (aggregation-only; no inference)

* **M1 soft gate** — inversion weight continuous in q_hat rather than a step at
  the ceiling: a logistic in q_hat / ceiling. Removes the cliff, so there is no
  q at which behaviour changes discontinuously.
* **M2 randomized threshold** — the ceiling is resampled per epoch from a private
  distribution. The attacker knows the DISTRIBUTION but not the draw, so it
  cannot park just below a value it can compute.
* **M3** — M1 and M2 combined.

## Pre-registered success criteria (ALL three required)

1. **Worst-case attacker gain <= 0.15** over the full cached q-grid, per
   benchmark, versus the current +0.92.
2. **Cost at f=0 <= 0.01** relative to aip_trust_only -- the mitigation must not
   buy robustness by inverting when there is nothing to invert.
3. **Retains >= 80% of the hard gate's f=0.7 fully-coherent inversion gain**, so
   the fix does not destroy the mechanism it protects.

## Pre-committed interpretation

* Any variant meeting all three becomes the paper's proposed defence.
* If NO variant passes, the reported conclusion is that **the evasion band is
  fundamental to threshold-based gating**, not that the mitigations were badly
  tuned. That is a citable negative result and will be reported as the primary
  outcome rather than buried.
* Criteria will not be loosened after seeing the numbers. If a variant misses on
  one criterion it is a failure, and any near-miss is reported with its margin.
