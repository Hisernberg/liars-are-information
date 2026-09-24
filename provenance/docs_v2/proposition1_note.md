# Conditional limit of observable-based channel classification

Correction dated 2026-09-12. This note replaces the historical proof sketch.
Equality of a mean coherence value is insufficient to establish
indistinguishability. The previous claims that exchangeability makes collision
counts sufficient and binomial are withdrawn. Historical manuscript/result
files are preserved as evidence of the original artifact, not revalidated by
this corrected note.

## The observation and decision must be specified

Let `H` denote an honest channel and `A` an adversarial channel. Let `Z` be the
entire observable statistic supplied to a proposed decision rule, including its
sample count and any history, peer selection, receiver agreement, or additional
features. For a rule limited to a scalar estimate, `Z` could be that estimate;
for AIP, it is richer. `AIPAggregator.fit` pools observations across a task
stream, computes receiver-conditioned and partner-selected channel statistics,
and the default gate uses a binomial-tail rule alongside other conditions.
The channel is not estimated from one question alone.

Distinguish three quantities:

1. `q_wrong = P(a_i=a_j | a_i != y, a_j != y)`, a gold-conditioned
   wrong-answer collision probability used in theoretical analysis.
2. `q_receiver = P(a_i=a_j | a_i != a_r, a_j != a_r)`, conditioned on dissent
   from receiver `r`. This need not equal `q_wrong` when the receiver is wrong.
3. The finite-sample, selected-partner coherence actually stored in `ChannelStats`,
   accompanied by its coincidence and joint-dissent counts.

Neither the attack's theoretical pair expectation nor its empirical all-pair
average is automatically the statistic supplied to AIP's gate.

## Conditional proposition

Assume two admissible data-generating worlds induce distributions `P_H` and
`P_A` on **all of Z**, and require different channel-classification actions.
For any possibly randomized classifier `delta(Z)` in `{H,A}`,

    P_H(delta=A) + P_A(delta=H) >= 1 - TV(P_H, P_A).

Thus with equal prior probabilities, classification error is at least
`(1-TV(P_H,P_A))/2`. If `P_H=P_A`, its average error is at least `1/2`.
The distributions here must cover the complete observation history used by
the rule, not just a single observation or the mean of a statistic.

Proof: for a deterministic classifier let `B={z: delta(z)=A}`. The error sum
is `P_H(B)+1-P_A(B)`, at least `1-TV(P_H,P_A)` by the definition of total
variation. Randomization corresponds to a bounded decision probability and
obeys the same bound (or follows by averaging deterministic rules).

This is a conditional testing statement. It does not establish that two such
worlds exist in the implemented attack family or at a measured sweep point.
It is also not directly a lower bound on final answer accuracy. A three-way
TRUST/DISCARD/INVERT rule can abstain; a guarantee about its utility needs a
specified loss for abstention and a link between channel decisions and answer
correctness.

## What equal coherence and exchangeability do not establish

Two distributions can have equal `E[Z]` and still be perfectly distinguishable:
`Z=1/2` always and `Z` equally likely to be zero or one have the same mean
and disjoint supports. Similarly, exchangeability of agents does not make
pairwise equality counts binomial or sufficient for their joint response law.
Pairs sharing agents are generally dependent; a common task difficulty or
shared lie adds further dependence. A fixed-coherence family needs its full
likelihood specified before any sufficiency claim is possible.

If an explicitly different experiment supplies `n` independent Bernoulli
collision indicators, each with constant parameter `q`, then its count is
Binomial. For `q_H` in `(0,1)`, Pinsker's inequality and the IID likelihood give

    TV(Bin(n,q_A), Bin(n,q_H))
      <= min(1, sqrt(n * KL(Ber(q_A) || Ber(q_H)) / 2)).

For a fixed interior `q_H` this is locally of order
`sqrt(n)*abs(q_A-q_H)`. It is not an assertion that the repository's selected
receiver/partner counts satisfy that experiment. At equal Bernoulli parameters
the specified IID laws are equal, but equal marginal parameters without IID
assumptions do not imply equal history distributions. More tasks may supply
additional information; the sample count is not bounded solely by swarm size.

## Actual gate-aware sampler and historical q axis

For a task, `_wrong_answer` draws uniformly from a pool of `K` wrong candidates.
The shared lie is itself in that pool. Each Byzantine agent selects the shared
lie with probability `p`; otherwise it independently draws from the same pool.
Conditional on the shared lie, one label has probability `p+(1-p)/K`, and
each other label has probability `(1-p)/K`. Therefore

    E[pair agreement] = (p+(1-p)/K)^2 + (K-1)*((1-p)/K)^2
                      = p^2 + (1-p^2)/K.

For a closed `C`-class task, `K=C-1`. The expectation rises monotonically
from `1/(C-1)` to one, and cannot be below that chance level. In particular,
`C=4, p=0.4` gives `0.44`, while the historical formula gives `0.28`.
The old expression `p^2+(1-p)^2/(C-1)` omitted mixed-branch collisions and
some collisions involving the shared label. Its apparent interior minimum is
not a property of the implemented sampler.

For open tasks, `K` is the number of distinct wrong answers among the assigned
honest broadcasts. If no such answer exists, `_wrong_answer` returns one
deterministic synthetic fallback, so `K=1` and all Byzantine answers coincide
at every `p`. An open answer space does not imply that independent sampler
draws cannot collide. A finite realized pair estimate may fall below the
population chance level due to sampling, and should be reported separately.

Original `results/adversarial/gate_aware*.parquet` files and related figures
retain the historical `q_realised` values. Those values were formula outputs,
not empirical receiver-gate measurements, and their q axes are mislabelled.
Preserve the original accuracy values and `coherence_p` as historical records;
do not claim that correcting an axis independently validates those accuracies.
The corrected script saves expected and empirical Byzantine-pair statistics
with explicit scope metadata. Adaptive harm must be evaluated on a held-out
split and can be stated as an empirical boundary, without asserting the
conditional proposition's premises from one sweep.
