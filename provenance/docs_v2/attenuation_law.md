# Agreement, multiclass errors, and the attenuation identity

Mathematical clarification dated 2026-09-12. Historical numerical claims are not
reproduced by this derivation. It corrects the covariance identity, the sign of
the binary approximation's bias, and overbroad identifiability/open-space claims.

## 1. Definitions and exact decomposition

For a task with a single normalized true answer `y`, agent `i` emits `a_i`.
Let `E_i=1[a_i != y]`, `e_i=P(E_i=1)`, and define

    m_ij = P(a_i=a_j),
    q_ij = P(a_i=a_j | E_i=1, E_j=1),
    c_ij = Cov(E_i,E_j).

When the jointly wrong event has zero probability, its agreement contribution
is zero and the conditional `q_ij` itself is undefined. With one normalized
correct label, agents agree either when both are correct or when both are
wrong and choose the same label. Hence

    m_ij = P(E_i=0,E_j=0) + P(E_i=1,E_j=1)*q_ij.

Since `P(E_i=1,E_j=1)=e_i*e_j+c_ij` and
`P(E_i=0,E_j=0)=(1-e_i)*(1-e_j)+c_ij`, the exact identity is

    m_ij = (1-e_i)*(1-e_j) + e_i*e_j*q_ij + (1+q_ij)*c_ij.

For nondegenerate error marginals,
`c_ij=phi_ij*sqrt(e_i*(1-e_i)*e_j*(1-e_j))`.
The covariance enters **both** jointly correct and jointly wrong probabilities.
The previous expression canceled one covariance term and was incorrect.

Independent correctness sets `c_ij=0`, giving the commonly used special case

    m_ij = (1-e_i)*(1-e_j) + e_i*e_j*q_ij.             (1)

This assumption concerns whether agents err; it says nothing by itself about
the distribution of their wrong labels. Equality of differently formatted or
semantically equivalent answers needs an explicit normalization rule before
this categorical decomposition can be interpreted as task correctness.

## 2. Binary reduction and approximation bias

With exactly two labels, both wrong agents must choose the sole wrong label,
so `q_ij=1` whenever that conditional probability is defined. Under independent
correctness, (1) gives

    2*m_ij - 1 = (1-2*e_i)*(1-2*e_j).

With dependence, the right side instead has the additional term `4*c_ij`.
Binary `q=1` does not imply that agents' error events are identical.

Using the binary equation on multiclass data where `q_ij<1`, while still
assuming independent correctness, gives

    (m_ij - 1/2) - (1-2*e_i)*(1-2*e_j)/2
       = e_i*e_j*(q_ij-1) <= 0.

Thus that approximation **underestimates** the half-scaled signed-quality
product. The previous prose said “overstates” despite showing a negative bias.
Correlated correctness contributes `(1+q_ij)*c_ij` as well, so the same bias
direction is not guaranteed after dropping independence.

## 3. Chance collision is a distributional assumption

For independent wrong responses uniform over `C-1` alternatives,

    q_chance = sum over k != y of 1/(C-1)^2 = 1/(C-1).

This is a specified null model, not a universal description of honest errors.
Independent agents can have concentrated, overlapping wrong-answer
distributions and therefore high agreement without adversarial coordination.
For independent draws with wrong-label distributions `r_i` and `r_j`,
`q_ij=sum_k r_i(k)*r_j(k)`. If the two distributions coincide and have support
size `K`, their collision probability is at least `1/K`. Distinct agent
distributions need not satisfy that same lower bound.

Open numeric or text answers can also concentrate on a small candidate pool;
openness alone does not justify `q=0`. The approximation
`m_ij approximately (1-e_i)*(1-e_j)` is warranted only when both the
wrong-answer collision term and the covariance contribution are negligible.
Shared attractors can be honest misconceptions; they are not sufficient
evidence that inversion is safe.

The repository's gate-aware symbolic sampler uses a uniform wrong pool,
including its shared lie, and has expected malicious-pair agreement
`p^2+(1-p^2)/K`. Closed spaces have `K=C-1`; open spaces use the distinct wrong
honest-answer pool or a singleton fallback. See [the corrected sampler and
observability note](proposition1_note.md).

## 4. What pairwise agreement can identify

Agreement alone cannot distinguish arbitrary error dependence and wrong-answer
structure. The following algebra presumes independent correctness and a **known
common** value `q_ij=q` across all pairs. Let

    mu_i = 1-(1+q)*e_i,
    S_ij = (1+q)*m_ij-q.

Then (1) gives `S_ij=mu_i*mu_j`. For three distinct agents with nonzero
denominator,

    mu_i^2 = S_ij*S_ik/S_jk.

This can recover magnitudes and relative signs in a nondegenerate, correctly
specified population model. A global orientation ambiguity remains unless
probability constraints or an additional assumption resolves it. Requiring
`e_i<1/(1+q)` fixes positive signs; for uniform multiclass wrong labels this
corresponds to accuracy above `1/C`. A receiver's known honest identity does
not, on its own, establish that competence assumption.

| Setting | What the pairwise equations justify |
|---|---|
| Binary, known `q=1`, independent correctness | Rank-one recovery under nondegeneracy, with an orientation constraint. |
| Known common multiclass `q` | The same transformed rank-one identity, with valid probability/orientation constraints. |
| Known `q=0` | Products of accuracies; nonnegative accuracies can resolve orientation, except degenerate cases. |
| Unknown common `q` | Counting equations is insufficient to prove identifiability. Extra variation and an injectivity argument are needed; homogeneous-error populations are a counterexample to unconditional recovery. |
| Free pair-specific `q_ij` or unknown `c_ij` | Error rates and channel structure are generally not separately identified from agreements alone. |

For example, if all agents have one error rate `e`, every observed pair can
share `m=(1-e)^2+q*e^2`. Distinct admissible `(e,q)` values give the same `m`,
regardless of how many agents are added. More equations repeating the same
relationship do not establish identifiability.

A full latent confusion-matrix model uses response patterns beyond pairwise
agreement, but requires its own conditional-independence, data coverage, and
label-orientation conditions. The original `blind_estimate_mc` implementation
uses a **one-coin** Dawid–Skene model, not a full per-class confusion matrix.
Neither model should be claimed identifiable merely because EM converges.
Ground truth is available for offline calibration/validation when declared;
using it to choose a test-set label permutation or best restart would leak
evaluation labels.

## 5. Connection to the implemented gate

The gold-conditioned `q_ij` in (1) is not the gate's receiver-conditioned
`P(same | both peers dissent from receiver)`. They can differ when the receiver
is incorrect. Moreover, the implementation selects a partner and pools task
history; its finite-sample statistic is not a randomly selected pair probability.
The current hard-gate implementation also uses an empirical agreement-gap
reference. The attenuation identity is explanatory algebra and does not by
itself validate that implementation's calibration or statistical test.

Measure honest error covariance and the relevant conditional agreement on a
declared calibration split, then assess transfer and false inversions on held-out
tasks. Misspecifying conditional independence does not guarantee a conservative
bias direction for an EM estimate. Claims about that direction need evidence.
The source's frozen thresholds and historical results remain unchanged by this
documentation correction.
