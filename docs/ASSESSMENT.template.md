# Assessment: how good is this work, and where should it go?

This is an internal review written the way an external referee would read the work, before submission. The scores are our judgement. Every number comes from `results/` through the same generator as the paper, and every qualitative claim it relies on is re-checked by `experiments/verify_claims.py` ({{claimsHold}} of {{claimsTotal}} hold).

## 1. The contribution in four sentences

Honest agents in a multi-agent LLM system can use what liars say, as long as the lies depend on the truth. RACE lets every honest agent estimate that dependence without labels, by anchoring a latent-truth model on the one fact it knows: that it is itself honest. The anchor is what breaks label switching, so RACE survives liar majorities that defeat majority vote, Dawid–Skene and four other classical crowdsourcing estimators (E11: at most {{crowdClassicalMaxSeven}}% at 70% liars, against {{crowdRaceSeven}}% for RACE). The claim is backed by a theorem (identification and consistency), {{nWorldsReplay}} replayed swarms, two live six-model swarms (one pre-registered), and an evidence ledger that re-derives every sentence from the data.

## 2. Scorecard

| Criterion | Earlier release (v3.1, before this round) | **Now** | What moved it |
|---|:-:|:-:|---|
| Originality | 3 / 5 | **3.5 / 5** | E11 shows empirically that the anchor, not the latent-class model, is what matters. RACE-informed debate is a new protocol that closes the loop from estimation back to interaction. |
| Technical soundness | 4 / 5 | **4.5 / 5** | Theorem 2: identification via Kruskal's condition, with consistency of the anchored MLE/MAP and of the plug-in decision. A two-agent counterexample test shows why three informative agents are needed. |
| Baselines | 3 / 5 | **4.5 / 5** | IWMV, MACE, GLAD and KOS were added under the identical protocol, with {{crowdClassicalCells}} paired comparisons and {{crowdClassicalLosses}} losses. Missing: spectral Dawid–Skene and robust-aggregation rules adapted to categorical answers. |
| Empirical breadth | 5 / 5 | **5 / 5** | 11 studies, including best-response and adaptive attackers, LLM-written deception, online sleepers, two oracles and two live swarms. |
| Statistical rigour and reproducibility | 5 / 5 | **5 / 5** | Paired tests with Holm correction; the test and the interval must agree. Manifests with hashes, one-command reproduction, {{nTests}} tests. New: pre-registration (E10) and a claims ledger in the paper. |
| Honesty about limits | 5 / 5 | **5 / 5** | Camouflage, sleepers, weak anchors, the RACE-D negative result, the post-hoc binary rule and every E10 outcome are reported, whatever their direction. |
| Realism of the multi-agent setting | 2 / 5 | **2.5 / 5** | A second live swarm on a new benchmark, with a debate protocol that uses RACE. Still 1–2B models, closed-form answers and one debate round. |
| Clarity | 4 / 5 | **4 / 5** | Dense. The README, videos and figures help, but a 9-page conference version still has to be cut from 32 pages. |
| **Overall** | **≈ 7 / 10** | **≈ 8 / 10** | For the recommended journals (TMLR, JAAMAS) and the multi-agent venues (AAMAS). For NeurIPS, ICML or ICLR main tracks it is ≈ 6.5–7 / 10 until the realism gap closes. |

## 3. Strengths a referee will see

1. **A clean idea with a proof.** Label switching is *the* reason label-free estimators fail at f ≥ ½ (Proposition 2). A receiver anchor provably fixes it, and a near-chance anchor provably cannot. With three informative agents the anchored estimator is consistent (Theorem 2).
2. **Wins where it should, loses where the theory says it must.**
   - RACE scores {{zooGateRaceSeven}}% against the attack built to beat AIP ({{zooGateAipSeven}}% for AIP) and {{llmRaceSeven}}% against real LLM deceivers at 70%.
   - It loses to camouflage and sleepers, which break exactly the assumptions of Theorem 1.
   - The information budget (E9) certifies those violations from data.
3. **It can do better than removing the liars.** Against independent liars at 70%, it scores {{extRaceIndepSeven}}% vs {{extRemovalIndepSeven}}% for an oracle that knows who lies and removes them. No filtering rule can do that in expectation.
4. **The baselines are strong and fair.** E11 runs six classical estimators through the same harness as RACE. They match RACE with a liar minority, so the comparison is not stacked.
5. **Reproducibility is unusually strong.**
   - Every number is generated.
   - {{claimsTotal}} prose claims are re-derived automatically.
   - Each study carries a manifest.
   - E10 was pre-registered in a public commit before any answer existed.

## 4. Weaknesses a referee will raise, and our answer

| Likely objection | Our answer | Residual risk |
|---|---|---|
| "This is Dawid–Skene with one trusted worker." | Partly true, and the paper says so. What is new: every receiver anchors on itself with no shared trust; the proof that this is what breaks label switching; E11's demonstration; clone tempering by dependent errors; the information budget; the analysis against AIP's impossibility result. | Medium: novelty will be judged as incremental on the estimator side and stronger on the multi-agent and adversarial side. |
| "The binary rule was chosen after seeing live data." | Disclosed. E10 tests it on fresh answers under a pre-registered rule. H1a (v3.1 > v3.0 on BoolQ) is **{{hypHOneAVerdict}}** ({{hypHOneATarget}}% vs {{hypHOneABase}}%, Δ = {{hypHOneADelta}}). H1b (v3.1 ≥ receiver alone) is **{{hypHOneBVerdict}}** ({{hypHOneBTarget}}% vs {{hypHOneBBase}}%). | Low: the amendment is now tested on data that did not exist when it was adopted. |
| "Tiny models, closed-form answers." | True. Both live swarms use 1–2B models on CPU, and answers are option letters. The replay studies use 3B–32B models. | **High: this is the main gap to a 9 / 10.** |
| "Few test questions per live world." | E8 has {{nLiveTestTasks}} test questions per world and E10 {{nLiveTwoTestTasks}}. Verdicts need the Wilcoxon test and the bootstrap interval to agree, which is conservative at this size. | Medium: live effects are real but imprecise. |
| "Does RACE help the *interaction*, not just the vote?" | E10's H3 tests it. RACE-informed debate changes individual post-debate accuracy by **{{informedDelta}}** points ({{informedPlain}}% → {{informedInformed}}%, 95% CI [{{informedCiLow}}, {{informedCiHigh}}], Wilcoxon p = {{informedP}}). Verdict: **{{hypHThreeVerdict}}**. | See §6: a larger debate study is the natural next step. |
| "An adaptive attacker who knows RACE wins." | Camouflage does, and we show it. Against a best-response attacker over the stationary zoo, RACE scores {{brRaceSeven}}% at 70% liars. | Medium: a formal game-theoretic bound is future work. |

## 5. Where to submit, and how likely it is to land

| Venue | Type | Fit | Our estimate now | What would raise it |
|---|---|:-:|---|---|
| **TMLR** | journal, rolling | ★★★★★ | Good: its criteria are correctness and evidence, which is where this work is strongest | Shorter main text |
| **JAAMAS** | journal | ★★★★★ | Good: multi-agent trust and robustness are core topics | A GPU live swarm |
| **AAMAS** (main track) | conference | ★★★★★ | Competitive | Cut to 8 pages around E1–E3, E8–E11 |
| **IEEE SaTML** | conference | ★★★★☆ | Competitive with a security framing | A sharper threat-model section |
| **UAI / AISTATS** | conference | ★★★★☆ | Competitive with the identifiability framing | Finite-sample rates for anchored EM |
| **JAIR / Machine Learning (Springer)** | journal | ★★★★☆ | Good | — |
| **AAAI / IJCAI** | conference | ★★★★☆ | Borderline to competitive | Realism |
| **COLM / ACL venues** | conference | ★★★☆☆ | Borderline | Open-ended tasks, larger models |
| **NeurIPS / ICML / ICLR** (main) | conference | ★★★☆☆ | Borderline | Everything in §6 |
| Workshops (multi-agent LLMs, agent safety, trustworthy ML) | workshop | ★★★★★ | Very likely | — |

**Recommended plan.**
1. Post an arXiv preprint.
2. Submit a 4-page workshop version for early feedback.
3. Submit the full paper to **TMLR** (or **JAAMAS**).
4. In parallel, submit an 8-page version to **AAMAS**.
5. Agree authorship with the authors of the prior AIP work and the bucket release before any submission.

## 6. What would move it from 8 to 9 / 10

1. **A GPU-scale live swarm:**
   - 7–14B open models;
   - ≥ 500 questions per benchmark;
   - two or three debate rounds;
   - at least one open-ended task (free-form QA with an equivalence judge, or code with unit tests).

   This closes the realism gap, which is the largest single risk.
2. **Finite-sample theory:** a rate for anchored EM under row-diagonal dominance, and a bound on what a camouflage attacker can extract.
3. **Two more baselines:** spectral Dawid–Skene initialisation, and a robust-aggregation rule (trimmed or median-style) adapted to categorical answers.
4. **An external replication:** another group running `experiments/reproduce_all.sh` and the live swarm on their own hardware.

## 7. Verdict

The work is now a solid, honest, unusually reproducible paper with a clear idea, a correct and relevant theory, strong baselines, and a pre-registered confirmation. We rate it **≈ 8 / 10** for its natural venues (TMLR, JAAMAS, AAMAS). What keeps it below 9 is not correctness but scale and realism: small models and closed-form answers. Every other known weakness is disclosed with numbers, which is the property referees at these venues reward most.
