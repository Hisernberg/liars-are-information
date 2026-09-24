# Gate-aware adversary — the threat model AIP does not survive

Every result before this one assumes the adversary does not know AIP exists. That
is the assumption a reviewer removes first, and it is the one that matters,
because AIP's threshold is not a secret: the calibrated honest-q ceiling is
published in `configs/inversion_thresholds.yaml`. An adversary who reads it can
aim just underneath — coordinated enough to move the vote, too incoherent to be
inverted.

**It works, and it works better than lying coherently.**

---

## 1. The adversary

One knob: `p`, the probability each Byzantine agent joins the bloc's shared lie
rather than lying alone. It never emits the correct answer — an evading adversary
has no reason to be right, only to be hard to characterise. Realised pairwise
coherence is

```
q = p² + (1-p)²/(C-1)      closed space of size C
q = p²                     open space
```

so a target `q` is reached by inverting that relation.

## 2. The gate is silent below its ceiling — exactly as designed, and that is the hole

Inversion gain (`aip_gated` − `aip_trust_only`) on MedQA, ceiling 0.677:

| q realised | f=0.3 | f=0.5 | f=0.7 |
|---|---|---|---|
| 0.253 | 0.000 | 0.000 | 0.000 |
| 0.280 | 0.000 | 0.000 | 0.000 |
| 0.333 | 0.000 | 0.000 | 0.000 |
| 0.413 | 0.000 | 0.000 | 0.000 |
| 0.520 | 0.000 | 0.000 | 0.000 |
| **0.653** | 0.432 | 0.847 | **0.908** |
| 0.813 | 0.454 | 0.845 | 0.862 |
| 1.000 | 0.502 | 0.878 | 0.832 |

Below the ceiling the gate never fires and AIP degrades to trust-only. Above it,
inversion is worth up to 0.91. The transition is sharp. Nothing here is a bug —
the gate is behaving exactly as specified, and the specification is what the
adversary exploits.

## 3. The optimal adversary is deliberately mediocre at coordinating

`aip_gated` accuracy; lower means a more damaging attack:

| benchmark | f | best attack `q` | accuracy | fully coherent (q=1.0) | adversary gains |
|---|---|---|---|---|---|
| medqa | 0.3 | 0.28 | 0.525 | 0.871 | +0.346 |
| medqa | 0.5 | 0.28 | **0.082** | 0.878 | **+0.796** |
| medqa | 0.7 | 0.33 | **0.000** | 0.832 | **+0.832** |
| mmlu | 0.3 | 0.33 | 0.805 | 0.891 | +0.086 |
| mmlu | 0.5 | 0.28 | 0.374 | 0.903 | +0.529 |
| mmlu | 0.7 | 0.33 | **0.000** | 0.915 | **+0.915** |
| math500 | 0.3 | 0.00 | 0.206 | 0.340 | +0.134 |
| math500 | 0.7 | 0.00 | 0.000 | 0.175 | +0.175 |

**At f=0.7 on MedQA and MMLU a semi-coherent adversary drives accuracy to exactly
0.000 while a fully coherent one leaves AIP at 0.83–0.92.** Coordinating *badly*
is worth up to +0.92 accuracy to the attacker. The naive intuition — that a more
coordinated adversary is a stronger adversary — is inverted by the presence of a
threshold, which is precisely what a threshold-based defence should have been
checked against before being claimed.

Note where the optimum sits: `q ≈ 0.28–0.33`, well below the 0.677 ceiling and
close to the 1/3 chance level for a 4-way space. The most effective adversary is
one whose coherence is nearly indistinguishable from honest disagreement.

## 4. The low-ceiling prediction was wrong

Going in, the expectation was that a low ceiling would be a *structural defence*:
MATH-500's open-text ceiling of 0.198 demands `p ≤ 0.44` to evade, fewer than half
the bloc agreeing, which is barely a bloc.

The data says the opposite. On MATH-500 the inversion gain is **0.000 across the
entire f=0.5 row**, at every coherence level. The gate is not hard to evade there;
it is close to inert, and a coherent adversary wins outright (accuracy 0.000 at
f=0.5 and f=0.7). A low ceiling does not mean evasion is hard — it means the class
is barely gated at all.

So the answer-space story is the reverse of the proposed one. Inversion is useful
where honest models *do* share attractors and the ceiling is high enough to leave
a usable band above it (the C=4 benchmarks), and useless where the ceiling sits
near zero.

## 5. What survives and what does not

| claim | status |
|---|---|
| Inversion is a real, large mechanism (gain 0.65 at f=0.7, 0.001 at f=0, flat in N) | **SUPPORTED** |
| AIP is robust to a coherent Byzantine bloc | **SUPPORTED** (0.83–0.92 at f=0.7) |
| AIP is robust to an adversary that knows the threshold | **REFUTED** (0.000 at f=0.7) |
| A low honest-q ceiling is a structural defence | **REFUTED** — it makes the gate inert instead |

The mechanism claim survives; the **threat model does not**. AIP defends against
adversaries who do not read the paper.

## 6. Where this points

The evasion band is not a mystery — it is a known interval of coherence,
`q ∈ (chance, ceiling)`, and the attack requires the adversary to sit inside it.
That is a defensible target for a mechanism fix rather than an admission of
defeat:

* **Estimate `f` instead of fixing a threshold.** `aip_naive` (inverts on any
  above-chance coherence) is monotone *increasing* in `f` and beats `aip_gated`
  at high `f`; `aip_gated` wins at low `f`. A defender that estimated `f` from
  observable channel statistics and interpolated would close the band from above.
* **Two-sided gating.** The current gate asks "is this channel more coherent than
  honest?" A channel sitting persistently *at* the honest boundary while
  dissenting from the receiver is itself anomalous, and that is a second statistic
  the gate does not currently use.

Neither is claimed here. Both are motivated by a measured attack rather than
proposed in the abstract, which is the difference between future work and
speculation.
