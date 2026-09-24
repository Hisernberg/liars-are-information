# MX Phase 1 — state at hard stop, 2026-09-20

**Chain halted.** Three gates red after the one authorised fix attempt
(`fit_stage`). Phase 2 not started. Phase 3 not started. No liar code path has
executed at any scale.

## Gate status: 14 of 17 green

Green: G1, G1b, G2, G2b, G3, NC-P2, NC-zero-INVERT, extraction-on-stage-rows,
G4, G4b, G6 (all three models).

The `fit_stage` fix worked and is confirmed by the gates it was aimed at:
gate decisions **0/120 → 120/120**, decision vocabulary now the gate's own
(`trust`/`discard`, previously empty), false inversions 0, stage-row extraction
failures **2/300 → 0/300**.

## The three red gates share one root cause

| gate | before fix | after fix |
|---|---|---|
| NC P3 == P4 | 48/60 differ | **4/60 differ** |
| every cached generation parsed | 63/706 unparsed | **27/690 unparsed** |
| G5 discards <= 1% per cell | phi4 25–46% | **phi4 15–28%** |

All three trace to `phi4_mini_reasoning` still truncating, now at the 3072
budget Amendment A2 set. The chain is: phi4 hits the cap, the truncation guard
correctly refuses to manufacture an answer, the row is discarded, and an arm
that should be identical to another diverges because one member has no answer.

The four NC-differing rows confirm it exactly — `medqa-test-00077` and
`medqa-test-00009` each carry exactly one phi4 discard, and they are the only
tasks that differ:

    solver   medqa-test-00077   P3='C'  P4='B'
    critic   medqa-test-00077   P3='C'  P4='A'
    refiner  medqa-test-00009   P3='C'  P4='A'
    refiner  medqa-test-00077   P3='C'  P4='A'

## Finding: Amendment A2's budget evidence was self-referential

A2 justified 3072 from phi4's finished-completion lengths **measured at a 2048
cap**: p50/p90/max = 1003/1538/2013, read as "a real ceiling around 2000".

At 3072 the same statistics are **p50/p90/max = 1051/1981/2935**.

The ceiling moved with the cap. The tail was never a property of the model; it
was the cap truncating the distribution that was used to justify the cap. Raising
the budget again would chase the same moving target, so it is not proposed —
and the urge to do it is on the standing stop list.

## Defect: two stale rows from the failed run survive in the MX cache

`data/mx_cache/phase1.parquet` still holds **2 truncated rows capped at 2048** —
generated before A2 and never invalidated when the budget changed. The standing
rule is that nothing from a failed attempt may be reachable by the verifier, the
analysis code, or the seed-comparison path. These are reachable.

The A2 override changed the budget but did not invalidate cache rows whose
`n_tokens` equals the *old* cap. Not fixed here: it is a cache-invalidation
change, and the stop is in force.

## What is NOT done

- `llama32_3b` answer-flip measurement — Phase A item 5, blocked by the hard
  stop's "run nothing further". Needs ~2 minutes of GPU. Not run.
