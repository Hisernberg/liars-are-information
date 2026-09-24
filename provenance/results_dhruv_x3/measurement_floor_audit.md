# Measurement floor and claim audit

## The floor

Minimum detectable effect, two-sided 95%, on a difference of two means. Built
from resample noise (greedy vs T=0.7 on the same model and tasks) combined in
quadrature with binomial aggregation noise over the benchmark's task count.

| benchmark | tasks | resample sigma | floor (MDE) |
|---|---|---|---|
| gsm8k | 500 | 0.0104 measured (answer-level) | **0.068** |
| gsm8k | 500 | 0.0050 measured (path-level) | **0.064** |
| math500, mmlu, medqa, boolq, arc | 200 | 0.0077 proxy | **0.100** |

Only GSM8K carries a T=0.7 second pass, so it is the only benchmark with a
directly measured resample sigma. The others inherit it as a flagged proxy;
their floors are dominated by the binomial term at n=200 regardless.

The floor is **flat in N**, and that is not an oversight. A swarm decision is one
outcome per task however many agents vote, so adding agents does not add
independent trials at the level the accuracy is measured.

## HARD RULE

Any effect whose confidence interval crosses its cell's floor is
**"undetectable at this scale"**. It is never "small", never "no effect", and
never "essentially zero".

## Audit of every existing claim

| claim | magnitude | floor | verdict |
|---|---|---|---|
| Inversion gain at f=0.7 | **0.65** | 0.068 | **DETECTABLE** — 9.6x the floor |
| Inversion cost at f=0 | 0.001 | 0.068 | **UNDETECTABLE** — see below |
| Gain invariant across N=10/20/50 | range 0.017 | 0.068 | **UNDETECTABLE** — see below |
| Gate-aware attacker gain | **0.92** | 0.100 | **DETECTABLE** — 9.2x |
| Sybil s=8: AIP 0.672 vs bloc defences 0.265 | **0.407** | 0.100 | **DETECTABLE** — 4.1x |
| E2 coherent vs disagreeing bloc | 0.011 | 0.100 | **UNDETECTABLE** |
| E1 reputation drop 0.040 vs AIP 0.150 | 0.110 | 0.100 | **DETECTABLE, MARGINAL** — 1.1x |
| Flat-in-f refutation (medqa −0.44) | **0.44** | 0.100 | **DETECTABLE** — 4.4x |
| N_eff homogeneous 1.50 vs heterogeneous 2.31 | 0.81 agents | n/a | different units; CI-based |
| BoolQ q = 1.000, MAE 0.000 | exact | n/a | analytic identity, not an estimate |

## Three claims that must be rewritten

**1. "Inversion costs 0.001 at f=0" → "no detectable cost at f=0".**
0.001 is 1/68th of the floor. The honest statement is that the gate's cost at
f=0 is below what this study can resolve, which supports "we could not detect a
cost" and does **not** support "the cost is zero". The direction of the claim is
unchanged and the strength of it is not.

**2. "The gain is invariant across N=10 to 50" → "any N-dependence is below the
floor".** Measured 0.643 / 0.638 / 0.655, a range of 0.017 against a floor of
0.068. Invariance was not demonstrated; failure to detect variation was. This
also means the ORIGINAL prediction -- that the gain grows with N -- is not
refuted either. Both "grows" and "flat" are consistent with this data, and the
experiment was underpowered to separate them.

**3. "AIP-windowed behaves identically whether the bloc is coherent or not"**
(−0.066 vs −0.077). The difference is 0.011 against a 0.100 floor. The conclusion
drawn from it -- that coherence amplification is not the mechanism -- stands,
because the burden was on the mechanism to produce a separation and none is
detectable. But the phrasing must be "no detectable difference", not "identical".

## One marginal call

E1's comparative result -- reputation decay degraded least (0.040) and AIP most
(0.150) -- has a difference of 0.110 against a 0.100 floor. It clears, but by
10%. It should be reported with the floor stated alongside it, and it is not
strong enough to carry a headline on its own.
