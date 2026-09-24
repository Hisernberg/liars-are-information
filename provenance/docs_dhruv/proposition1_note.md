# Proposition 1: setup, proof sketch, and the finite-$n$ conditions

This note carries the material cut from the manuscript's appendix for length. The
manuscript (Section "Limits of Coherence-Based Gating") states the proposition,
gives the proof sketch and the total-variation bound, and reports the empirical
anchor with its resolution limits. What is here is the formal setup, the
exchangeability condition in full, and the finite-$n$ statement.

## Setup

Fix a question with a closed answer space of size `C` and a receiver that observes
a channel's answers. AIP's gate is a decision rule computed from the channel's
per-question coherence

    q = Pr(a_i = a_j | a_i != y, a_j != y),

the probability that two agents on the channel, both wrong, are wrong the same
way. Honest agents that fail independently over a closed space collide at the
chance rate `q_ind = 1/(C-1)`.

## Exchangeability, stated in full

The sufficiency argument requires that, conditional on the channel's coherence
parameter `q`, the joint distribution of the dissenting peers' answers is
invariant under permutation of the peers. This is the assumption the blind
estimator already makes when it pools pairwise agreement across peers rather than
modelling each peer separately. It is violated by a channel whose members have
systematically different error rates -- a bloc mixing a strong and a weak model,
for instance -- in which case the collision count is no longer sufficient and the
bound applies only to the exchangeable sub-family. We do not evaluate a
non-exchangeable bloc; it is a gap, not a result.

## Finite-$n$

For `n` jointly-dissenting observations the collision count is Binomial(n, q), so
for two coherence parameters `q_A` and `q_ind` the total variation distance is
bounded by the standard Binomial TV bound, which is `O(|q_A - q_ind| * sqrt(n))`
for small differences. Two consequences the manuscript states without deriving:

1. The bound degrades as `sqrt(n)`, so a defence that accumulates more
   jointly-dissenting observations per question does gain separating power --
   but per-question `n` is bounded by the swarm size, which is why the escape
   route is longitudinal (across questions) rather than per-question.
2. At `q_A = q_ind` exactly, `d_TV = 0` for every `n`. No amount of data
   separates the hypotheses, because they are the same hypothesis.

## Scope

The limit applies to defences whose statistic is per-question channel coherence.
It does not apply to defences drawing on information that statistic does not
contain. The manuscript identifies two such classes -- external information and
longitudinal statistics -- and states both as untested hypotheses.
