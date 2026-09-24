# The attenuation law, generalized to multiclass answers

*Phase B2. This section derives the estimator AIP uses to measure pair quality
from broadcasts alone, and states exactly what it can and cannot identify.*

## 1. Setup

Fix a task with true answer $y$. Agent $i$ broadcasts an answer $a_i$ drawn from
an answer space $\mathcal{A}$ of size $C = |\mathcal{A}|$. Write

$$e_i \;=\; \Pr(a_i \neq y)$$

for agent $i$'s marginal error rate. We assume throughout that **correctness is
independent across agents**,

$$\Pr(a_i \neq y,\; a_j \neq y) \;=\; e_i e_j ,$$

and we relax this in §6. Note carefully what is *not* assumed: nothing is said
about *which* wrong answer an agent gives. That is the whole point.

## 2. The generalized law

Let $m_{ij} = \Pr(a_i = a_j)$ be the agreement rate — an observable, requiring no
labels. Partition the agreement event by the correctness of the two agents:

$$
m_{ij} = \underbrace{\Pr(a_i = a_j,\ \text{both correct})}_{(A)}
       + \underbrace{\Pr(a_i = a_j,\ \text{both wrong})}_{(B)}
       + \underbrace{\Pr(a_i = a_j,\ \text{exactly one correct})}_{(C)} .
$$

Term $(C)$ vanishes identically. If $i$ is correct then $a_i = y$, and if $j$ is
wrong then $a_j \neq y$, so $a_i \neq a_j$. Agreement and disagreement-about-
correctness cannot co-occur.

Term $(A)$ is $(1-e_i)(1-e_j)$: both correct means both answered $y$, so they
agree automatically.

Term $(B)$ is where the structure lives. Define

$$q_{ij} \;=\; \Pr\big(a_i = a_j \;\big|\; a_i \neq y,\ a_j \neq y\big),$$

the probability that two agents who are *both wrong* are wrong **in the same
way**. Then $(B) = e_i e_j q_{ij}$, and

$$\boxed{\;m_{ij} \;=\; (1-e_i)(1-e_j) \;+\; e_i e_j\, q_{ij}\;}
\tag{$\star$}$$

This is the generalized attenuation law. The binary law of Phase B is the special
case $q_{ij} \equiv 1$.

$q_{ij}$ is not a nuisance parameter. It is the paper's inversion signal:

* honest agents that fail independently scatter across the wrong-answer space,
  giving small $q_{ij}$;
* a **coherent adversary** — one that lies the *same* way every time — drives
  $q_{ij}$ toward 1 with whichever agents share its lie.

A high $q_{ij}$ between two broadcasters is therefore evidence of a shared
mechanism, and a shared mechanism is precisely what can be inverted.

## 3. Reduction to the binary law ($C = 2$)

**Claim.** When $C = 2$, $q_{ij} = 1$ identically, and $(\star)$ reduces to the
Phase B law.

*Proof.* With $\mathcal{A} = \{y, y'\}$ there is exactly one wrong answer, $y'$.
Conditioning on both agents being wrong forces $a_i = y'$ and $a_j = y'$, hence
$a_i = a_j$ with probability 1. So $q_{ij} = 1$ regardless of $e_i, e_j$ or of any
dependence between the agents' choices — there is nothing to choose. Substituting
into $(\star)$:

$$m_{ij} = (1-e_i)(1-e_j) + e_i e_j = 1 - e_i - e_j + 2 e_i e_j .$$

Then

$$2m_{ij} - 1 = 1 - 2e_i - 2e_j + 4 e_i e_j = (1-2e_i)(1-2e_j),$$

so

$$\frac{(1-2e_i)(1-2e_j)}{2} \;=\; m_{ij} - \tfrac{1}{2},$$

which is exactly the estimator validated in Phase B. $\blacksquare$

This also explains the Phase B failure pattern quantitatively. Using the binary
law when $q_{ij} < 1$ estimates the product as $m_{ij} - 1/2$ while the truth is
$\big((1-2e_i)(1-2e_j)\big)/2$; subtracting, the bias is

$$\text{bias} \;=\; e_i e_j\,(q_{ij} - 1) \;\le\; 0 \ \text{ in } m,$$

i.e. the binary law **overstates** the quality product by $e_i e_j (1 - q_{ij})$.
The bias is second order in the error rates, which is why GSM8K (error rates
0.07–0.16, so $e_i e_j \approx 0.01$) escaped with MAE 0.05, while MATH-500
(error rates up to 0.66, so $e_i e_j \approx 0.2$) did not.

## 4. Chance level for $q$

**Multiple choice, $C$ known.** If a wrong agent picks uniformly among the $C-1$
wrong options, and the two agents choose independently,

$$q_{ij}^{\text{chance}} = \sum_{k \neq y} \Pr(a_i = k)\Pr(a_j = k)
= (C-1)\cdot\frac{1}{(C-1)^2} = \frac{1}{C-1}.$$

For MMLU, $C = 4$ and $q^{\text{chance}} = 1/3$. This is the **null hypothesis**
for honest agents: $q_{ij}$ materially above $1/3$ means the two models are drawn
to the *same* distractor, which is a shared error mechanism, not bad luck.

**Open-ended answers.** MATH-500 and GSM8K have no fixed $C$; the reachable
wrong-answer space is effectively unbounded and non-uniform. For independent
errors $q_{ij} \approx 0$, and $(\star)$ collapses to $m_{ij} \approx
(1-e_i)(1-e_j)$ — a *product of accuracies* rather than of signed qualities. Any
$q_{ij}$ appreciably above 0 on an open-ended benchmark is strong evidence of a
shared attractor, because coincidence has no chance mass to hide behind.

## 5. Identifiability

Everything below concerns what can be recovered from **agreement statistics
alone**, with no labels. Take $n$ agents, so $\binom{n}{2}$ observable agreement
rates.

| case | assumption on $q$ | unknowns | equations | identifiable? |
|---|---|---|---|---|
| binary, $C=2$ | $q_{ij} = 1$, known | $n$ ($e_i$) | $\binom{n}{2}$ | **yes** for $n \ge 3$ |
| MC, chance-$q$ | $q_{ij} = \tfrac{1}{C-1}$, known | $n$ | $\binom{n}{2}$ | **yes** for $n \ge 3$ |
| open-ended | $q_{ij} = 0$, known | $n$ | $\binom{n}{2}$ | **yes** for $n \ge 3$ |
| MC, homogeneous $q$ | $q_{ij} = q$, unknown scalar | $n+1$ | $\binom{n}{2}$ | yes for $n \ge 4$ |
| MC, free $q_{ij}$ | none | $n + \binom{n}{2}$ | $\binom{n}{2}$ | **no**, short by $n$ |

**When $q$ is known**, the three known-$q$ rows all reduce to the same algebra.
Writing $\mu_i$ for the relevant per-agent quantity ($\mu_i = 1-2e_i$ when $q=1$;
$\mu_i = 1-e_i$ when $q=0$), $(\star)$ becomes $\mu_i \mu_j = $ (an affine
function of $m_{ij}$), and any triple $(i,j,k)$ gives

$$\mu_i^2 = \frac{(\mu_i\mu_j)(\mu_i\mu_k)}{\mu_j\mu_k},$$

so $\mu_i$ is determined up to sign; the sign is fixed by assuming agents are
better than chance. Three agents suffice, and averaging over all triples
containing $i$ damps the noise. This is the third-moment identity behind spectral
crowdsourcing estimators, and it is what Phase B implemented.

**When $q_{ij}$ is free, pairwise agreement can never identify it.** Each pair
contributes one number $m_{ij}$ but introduces one new unknown $q_{ij}$ on top of
the shared $e_i$'s, so the system is short by exactly $n$ equations for every
$n$. No amount of data fixes this: it is a structural deficiency, not a sampling
one. Two agents alone are hopeless in every case — one equation, three unknowns.

**What breaks the tie.** Three sources of extra information, in increasing order
of assumption:

1. **A third agent plus a structural assumption** ($q$ constant across pairs, or
   equal to chance) restores identifiability, as the table shows.
2. **The full answer-pattern distribution**, rather than only pairwise agreement.
   Pairwise rates are second-order summaries of a distribution over $C^n$ joint
   answer patterns. A Dawid–Skene model fits that full distribution: it posits a
   latent true label per task and a per-agent error model, and recovers a
   posterior over the latent labels. Given that posterior, $e_i$ and $q_{ij}$ are
   both computable as posterior-weighted empirical frequencies — no labels used.
   This is the route implemented in `blind_estimate_mc`, and the identifiability
   it buys comes from the conditional-independence assumption, which is an
   assumption and is stated as one.
3. **Ground truth**, which identifies everything trivially and is available only
   for validation, never to an agent at run time.

The honest summary for the paper: *a swarm can measure its own pairwise quality
blindly, but only by assuming something about how its members fail.* The binary
law assumes they fail identically; the chance-$q$ law assumes they fail
uniformly; Dawid–Skene assumes they fail independently given the truth. AIP's
inversion machinery is interesting precisely because a coherent adversary
violates all three in the same direction, and the violation is measurable.

## 6. What is assumed away

Two assumptions are load-bearing and both are testable against a labelled cache,
which is why Phase B2 validates rather than asserts.

**Independent correctness.** $(\star)$ used $\Pr(\text{both wrong}) = e_i e_j$.
Phase B measured this to be false: cross-model error correlation $\phi$ averages
$0.23$ on GSM8K and $0.43$–$0.45$ on MATH-500 and MMLU. With correlated
correctness, $\Pr(\text{both wrong}) = e_i e_j + \phi_{ij}\sqrt{e_i(1-e_i)e_j(1-e_j)}$,
and $(\star)$ becomes

$$m_{ij} = (1-e_i)(1-e_j) + \phi_{ij}\sqrt{e_i(1-e_i)e_j(1-e_j)} + \big(e_i e_j + \phi_{ij}\sqrt{\cdots}\big)\,q_{ij} - \phi_{ij}\sqrt{\cdots}\,,$$

which is to say: correlation in *whether* agents err and correlation in *how*
they err are separate channels, and $q_{ij}$ isolates the second. Reporting both
$\phi_{ij}$ (Phase B) and $q_{ij}$ (Phase B2) is therefore not redundant.

**Conditional independence given the truth**, used by the Dawid–Skene route.
Correlated errors violate it, so the blind estimator will *under*-estimate
$q_{ij}$ exactly when agents share attractors — a conservative direction for a
detector of shared mechanism, and one to state when interpreting a blind $q$ that
already looks high.
