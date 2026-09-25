# Theory for receiver-anchored channel estimation

Notation. A task has latent truth $Y$ in a finite candidate set $\mathcal S$ with
$|\mathcal S| = K$. Receiver $i$ is honest. It observes its own report $R_i$
and the reports $R_j$ of the peers it hears. $H$ denotes the honest reports
(including $R_i$) and $B$ the Byzantine reports. An adversary strategy $\sigma$
is any conditional law $B \sim \sigma(\cdot \mid Y, H)$: it may know the truth
(gold-aware) and see the honest reports (rushing).

---

## Theorem 1 (a lie can only add information; the minimax attack is silence)

Let $\mathrm{acc}^*(\sigma) = \mathbb E\,[\max_y P_\sigma(Y=y \mid H, B)]$ be
the accuracy of the Bayes receiver that knows the joint law induced by
$\sigma$, and $\mathrm{acc}^*_H = \mathbb E\,[\max_y P(Y=y \mid H)]$ the Bayes
accuracy from honest reports alone. Then for every $\sigma$

$$\mathrm{acc}^*(\sigma) \;\ge\; \mathrm{acc}^*_H,$$

with equality whenever $B \perp Y \mid H$. Hence
$\inf_\sigma \mathrm{acc}^*(\sigma) = \mathrm{acc}^*_H$ and the infimum is
attained, e.g. by answering uniformly at random (`uninformative`) or by
echoing a random honest report (`echo`).

*Proof.* By the tower property $P(Y=y\mid H) = \mathbb E[P(Y=y\mid H,B)\mid H]$.
Since the maximum is convex, $\max_y \mathbb E[\,\cdot\mid H] \le
\mathbb E[\max_y \cdot \mid H]$ (Jensen). Taking expectations gives the
inequality. If $B\perp Y\mid H$ then $P(Y=y\mid H,B)=P(Y=y\mid H)$ and equality
holds. $\square$

**Corollary (no breakdown point for a known-channel receiver).** Because
$\mathrm{acc}^*_H$ is at least the receiver's own accuracy, the Bayes receiver
never falls below "trust yourself" at any Byzantine fraction $f<1$. By
contrast every plurality-type rule reaches 0 against a coherent bloc once
$f > 1/2$. The whole practical problem is therefore *estimating* the channel
without labels; the adversary's only levers are (i) estimation error,
(ii) violating the model's assumptions (conditional independence given $Y$,
stationarity), and (iii) removing its own information. The experiments are
organised by exactly these three levers.

---

## Proposition 2 (label switching; why label-free estimators break at $f\ge 1/2$, and what the anchor buys)

(a) *Binary one-coin model.* With $Y$ uniform on $\{0,1\}$ and conditionally
independent reports with accuracies $\theta=(a_1,\dots,a_n)$, the report law
under $\theta$ equals the law under $\theta'=(1-a_1,\dots,1-a_n)$ (relabel
$Y\mapsto 1-Y$). No function of the reports alone separates $\theta$ from
$\theta'$. If a coherent always-wrong bloc ($a=0$) holds a majority, $\theta'$
makes the bloc perfect, and any rule that picks the labelling in which the
plurality is accurate (majority vote, plurality-initialised EM, the one-coin
Dawid–Skene estimator) outputs the lie on every task.

(b) *Full Dawid–Skene model.* For any permutation $\tau$ of the latent classes,
$(\rho, \{\pi_j\}) \mapsto (\rho\circ\tau^{-1}, \{\pi_j[\tau^{-1}(\cdot),\cdot]\})$
leaves the report law unchanged; Kruskal-type uniqueness identifies the model
only up to such $\tau$.

(c) *Anchor.* Suppose the receiver's confusion matrix $\pi_i$ is strictly
row-diagonally dominant: $\pi_i[k,k] > \pi_i[k,l]$ for all $l\ne k$ ("given the
truth, the receiver's most likely report is the truth"). Then among all
relabellings of the true parameters, only $\tau=\mathrm{id}$ keeps $\pi_i$
row-diagonally dominant.

*Proof of (c).* Under $\tau\neq\mathrm{id}$ pick $k$ with $m=\tau^{-1}(k)\neq k$.
Row $k$ of the relabelled receiver matrix is row $m$ of $\pi_i$, whose largest
entry sits in column $m\neq k$; hence $\pi'_i[k,m] > \pi'_i[k,k]$ and
dominance fails in row $k$. $\square$

In the binary case, (c) reduces to $a_i > 1/2$. RACE enforces a
margin-constrained version of (c) (prior + projection $a_i \ge 1/K +
\delta$) and initialises EM from the receiver's own answers.

**Necessity.** If $a_i = 1/2$ in the binary case, $\theta$ and $\theta'$ both
satisfy any anchor constraint that the truth satisfies, so no receiver-local
label-free method can identify the truth. This is the *weak-anchor capture*
observed on open-answer tasks where a near-chance receiver faces a rushing
attacker that re-uses the receiver's own errors (see `docs/RACE_NOTES.md`).

---

## Proposition 3 (AIP's gate is a coarse quantisation of the Bayes weight, and Proposition 1 does not bind RACE)

Under the symmetric one-coin model, a report $R_j=c$ changes the log-odds of
candidate $c$ against any other candidate by

$$\lambda_j = \log\frac{(K-1)\,a_j}{1-a_j},\qquad \operatorname{sign}\lambda_j=\operatorname{sign}(a_j-1/K).$$

TRUST / DISCARD / INVERT is the three-level quantisation
$\{\lambda>0,\ \lambda=0,\ \lambda<0\}$, but AIP *triggers* INVERT on a
coherence test rather than on $a_j$. For the implemented gate-aware attacker
(join the shared lie with probability $p$, otherwise lie independently; never
state the truth), the Byzantine pair collision is
$q(p)=p^2+(1-p^2)/(K-1)$, which sweeps $[1/(K-1),1]$, while $a_j=0$ and
$\lambda_j=-\infty$ for **every** $p$. Coherence therefore carries no
information about the quantity that determines the Bayes weight.
Proposition 1 of the original work (no rule measurable w.r.t. per-question
coherence separates sub-chance coordination from honest disagreement) concerns
coherence-measurable rules. RACE's statistic depends on the joint law of
$(R_j, Y)$, estimated through the anchored latent model, so it is outside that
class. Empirically RACE inverts the gate-aware bloc at every $p$ (zoo study).
RACE has its own limits: Theorem 1's silence strategy, violations of
conditional independence (camouflage/attractor attackers whose lies depend on
$H$), and non-stationarity (sleepers).

---

## Proposition 4 (how much history is needed to invert correctly)

Fix the channel grouping and let $\hat y$ be the receiver's hard truth estimate on
$n_j$ history tasks where peer $j$ was heard, with error rate $\delta$. Let
$\hat a_j$ be $j$'s agreement with $\hat y$. Then $|\hat a_j - \bar a_j|\le\delta$,
where $\bar a_j$ is $j$'s empirical accuracy, and by Hoeffding

$$P\big(\operatorname{sign}(\hat a_j-1/K)\ne\operatorname{sign}(a_j-1/K)\big)\le
2\exp\!\big(-2n_j(|a_j-1/K|-\delta)^2\big)\quad\text{whenever } |a_j-1/K|>\delta.$$

For an always-wrong liar on a four-way question ($|a_j-1/K|=0.25$) with
$\delta=0.1$: $n_j=80$ gives $\le 0.055$ and $n_j=160$ gives $\le 0.003$. The
anchored EM starts at $\delta = 1-a_i$ and decreases it as honest peers are
identified. The history-length study (E5) measures the realised curve.

---

## Proposition 5 (raw agreement confuses competence with dependence; error-conditioned agreement does not)

For two channels that are conditionally independent given $Y$ with accuracies
$a,b$ and symmetric errors over $K-1$ alternatives:

$$P(R_1=R_2) = ab + \frac{(1-a)(1-b)}{K-1} \xrightarrow{a,b\to1} 1,\qquad
P(R_1=R_2 \mid R_1\ne Y \text{ or } R_2\ne Y) = \frac{(1-a)(1-b)}{(K-1)(1-ab)} \le \frac{1}{K-1},$$

where the last step uses $1-ab-(1-a)(1-b) = a(1-b)+b(1-a)\ge 0$.

The second quantity tends to $0$ as $a,b\to1$ and never exceeds $1/(K-1)$.
For replicas of one source, and for a perfectly coordinated bloc, it equals $1$.
Thresholding raw agreement therefore merges distinct *accurate* channels (the
GSM8K failure in `docs/RACE_NOTES.md`), while thresholding error-conditioned
agreement isolates conditional dependence. This is the quantity the latent-class
independence assumption ignores. It is the same confound behind the refuted
"maximise $N_\mathrm{eff}$" recipe in the original ledger (S1), where
agreement-based effective size largely restated roster weakness.

*Proof.* $P(R_1=R_2, \text{not both correct}) = P(\text{both wrong and equal}) =
(1-a)(1-b)/(K-1)$ and $P(\text{not both correct}) = 1-ab$. $\square$

## The information budget (an empirical companion to Theorem 1)

Theorem 1 compares the Bayes receiver that sees the liars' reports $B$ with the one that sees only the honest reports $H$. Study E9 measures that comparison directly with two oracles fitted on labeled HISTORY questions:

* the **liar-removal oracle** knows the Byzantine set, drops it, and decodes $H$ with the true one-coin accuracies and true source groups;
* the **known-channel oracle** decodes $(H,B)$ with the true accuracies and source groups.

The **information budget** of an attack is the known-channel accuracy minus the liar-removal accuracy. Both oracles use the same one-coin, conditionally independent decoder. So:

* under the one-coin model with stationary channels, the budget is the finite-sample analogue of $\mathrm{acc}^*(\sigma)-\mathrm{acc}^*_H\ge 0$. It is zero for an uninformative attacker (the minimax attack of Theorem 1) and grows with $f$ for truth-dependent liars (E9: +1.9, +2.8, +5.7 points at $f=0.3,0.5,0.7$ for independent liars);
* a **negative** budget cannot happen for the Bayes receiver. When the one-coin decoder shows one, the attack lies outside the one-coin conditionally independent stationary model. In E9 this happens for echo, camouflage and the sleeper, which are exactly RACE's failure modes.

A rule that only filters suspected liars sees at most $H$, so in expectation it cannot beat the Bayes receiver on $H$. The liar-removal oracle is our proxy for that receiver. RACE is label-free and does not know the Byzantine set, yet it exceeds that bound in 17 of 54 pooled (attack, $f$) cells, all of them truth-dependent attacks.

## Theorem 2 (anchored estimation is identified and consistent)

**Setting.** The class-conditional (Dawid–Skene) model: questions are i.i.d., the class prior ρ has full support on K classes, and G ≥ 3 channel groups are conditionally independent given Y, each group counted once after clone tempering.

**Assumptions.**
- Three of the groups have *invertible* confusion matrices. For a one-coin channel this means accuracy a ≠ 1/K, so a consistent liar counts as informative.
- The receiver's confusion matrix is row-diagonally dominant with margin δ > 0.

**Conclusions.**
- (a) The parameters are identified *exactly*, not only up to a relabelling of the classes, within the set Θ_δ where the receiver is δ-dominant.
- (b) The maximum-likelihood estimator over Θ_δ, with entries clipped to [ε, 1−ε], is strongly consistent as the history length T → ∞. So is the anchored MAP estimator, whose fixed-strength priors are O(1/T).
- (c) The plug-in decision converges to the known-channel receiver's decision. RACE's accuracy therefore converges to acc* of Theorem 1 under the model.

**Proof.**
- (a) The three groups' joint law is a three-way array Σ_y ρ_y π₁[y,·]⊗π₂[y,·]⊗π₃[y,·] whose factor matrices have full Kruskal rank K, and 3K ≥ 2K + 2. Kruskal's theorem (Kruskal, 1977; Allman et al., 2009 for latent-class models) identifies ρ and the three matrices up to one simultaneous permutation τ; row-stochasticity fixes the scales. Each further group g follows from π₁ᵀ diag(ρ) π_g, because π₁ and diag(ρ) are invertible. Proposition 2(c) leaves only τ = id.
- (b) Θ_δ is compact, the log-likelihood is continuous, and by (a) the truth is its unique maximiser in expectation; the standard consistency argument for maximum likelihood applies.
- (c) The posterior is continuous in the parameters.

**Consequences.**
- *Three informative groups are needed.* With the receiver and only one informative peer, the binary class-conditional model is not identified, even with the anchor. `tests/test_race.py::test_binary_identification_needs_three_informative_channels` shows exactly this: the peer's confusion matrix is recovered to within 0.1 with three informative channels, and is not recovered with two.
- *Misspecification.* A misspecified model converges to the KL projection of the truth. This is the one-coin model on option-biased live BoolQ answers, and it is why v3.1 uses class-conditional channels on binary questions.
- *The algorithm.* EM converges to a stationary point. The anchor initialisation places it in the basin of the dominant solution; the theorem is about the estimator, not about EM.
