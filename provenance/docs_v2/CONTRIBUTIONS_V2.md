# Six contributions, with all previous work retained

This version responds to a request for three **additional** contributions. Contributions 1–3 retain and organize the original project and the first DGX verification release. Contributions 4–6 add new software and new experiments. Organizing earlier work is not a claim that it was first introduced in v2. The extensions are research artifact contributions; novelty relative to all prior literature is not established here.

## Complete crosswalk from the original manuscript

The five bullets in `paper/sections/introduction.tex` remain in that historical file. None is deleted or silently replaced.

| Original contribution | Where it appears in the six-theme organization | What remains, and what was qualified |
|---|---|---|
| Mechanism: local invert/trust/discard and windowed variant | **1. Receiver-local inversion**; extended by **6. Causal online adaptation** | Original AIP and historical window code remain. v1 repairs static alignment and random thresholds. v2 introduces an independently tested causal wrapper; it does not retroactively validate the old window implementation. |
| Theory: attenuation, coordinatability, Proposition 1 | **2. Attack geometry and identifiability limits** | Preserve the original manuscript as historical evidence. Use the corrected covariance identity and conditional full-observation testing statement in the maintained notes. The implemented sampler has no below-chance population collision minimum of the kind claimed by the old formula. |
| Design laws: correlation ceiling, replicated models, blind estimation, defense premise | **3. Dependence, protocol and reproducible evaluation**; extended by **4. Calibration** and **5. Clone-aware caps** | Preserve all original correlation/replica outputs. v1 audits dependence and population transfer. v2 measures new ceiling transfer and controlled clone amplification without counting repeated outputs as fresh samples. |
| Protocol: confidence surfaces and sender control | **3. Dependence, protocol and reproducible evaluation** | Preserve original confidence-AUC and forged-self-report experiments. Their numeric claims are historical. A log-probability value is trustworthy only when supplied through a trusted collection/transport mechanism; arbitrary Byzantine metadata is not authenticated merely by being called logprob. New extensions do not use confidence to establish trust. |
| Evaluation, including failures, mitigations and cost | **3. Dependence, protocol and reproducible evaluation**, with new evaluation in **4–6** | Preserve historical sweeps, failed claims and cost tables. Keep the independent v1 results and add separate v2 simulations, task-level evidence, limitations and CPU receipts. New replay does not reproduce the original H100 inference campaign. |

The original ledger identifiers in `results/claims.md` keep their original definitions. File names such as `C4_CALIBRATION.md` abbreviate the new **contribution number**; they are not replacements for legacy Claim C4.

## 1. Receiver-local inversion and fair comparators — retained

**RQ1:** Under which answer spaces and attack conditions does inversion improve receiver-local trust/discard?

**Method:** Estimate channels from each receiver's own task-aligned history. Compare hard gating, trust-only, naive inversion, receiver-only, majority, confidence and a SAC-style approximation on identical broadcasts. Add full class-confusion Dawid–Skene for fixed closed label spaces, including unseen-task prediction. Requested self weighting is matched for the AIP mechanism ablations; the other methods retain their own definitions.

**Evidence:** `results/dgx/standard/`, `results/dgx/analysis/`, `src/aip/aggregation/dawid_skene_full.py`, and regression tests. At 50% coherent attackers/full observation, v1 AIP beats receiver-only on MMLU by 6.2 percentage points (paired 95% interval +3.7 to +8.8), but loses 9.8 points on MATH-500 and 89.3 on BoolQ. See the exact comparison tables for unrounded estimates and intervals. This answers RQ1 conditionally; it does not support universal robustness to a malicious majority.

**Limits:** Binary coherence has no room above a ceiling of one. Open-answer inversion need not create or select the true answer. Dawid–Skene has conditional independence, stationary-channel, initialization and latent-label ambiguity limitations and is not intrinsically unable to use anti-experts.

## 2. Corrected attack geometry and empirical boundary — retained

**RQ2:** What coherence does the implemented attacker generate, and can a gate-aware attacker defeat the fitted rule?

**Method:** For the shared/uniform-wrong-pool mixture, derive and test `q_pair = p² + (1−p²)/K` with the actual candidate-pool size K. Distinguish this expectation from empirical Byzantine pair collisions and receiver-conditioned selected-partner coherence. Correct the covariance term in the agreement identity. Restrict the indistinguishability statement to specified distributions over every observation used by the decision rule.

**Evidence:** `tests/test_coherence_formula.py`, `docs/attenuation_law.md`, `docs/proposition1_note.md`, and the v1 validation-selected attack records. At f=0.7, selected stationary gate-aware attacks reduce hard-gated MMLU accuracy to zero; soft variants recover only part of the loss. This is an empirical failure under the tested family, not a proof that every history-aware defense fails.

**Limits:** Each candidate attack was fitted on its own stationary history. Grid selection is finite and intervals condition on the selected attack. Contribution 6 separately examines actual regime changes with a common past-only state.

## 3. Dependence, confidence provenance and reproducible DGX evaluation — retained and expanded

**RQ3:** Which sources of apparent swarm size and confidence represent independent or trustworthy information, and what evidence makes an evaluation auditable?

**Method:** Retain original correlation and confidence analyses; audit all 54 honest caches (nine models, six benchmarks, 13,500 stored predictions); compare receiver- and gold-conditioned coincidence; document prior calibration overlap. Freeze agent-model identities before corruption. Use local observations, correct scoring, paired task inference, bounded CPU runs and saved trace/state evidence.

**Evidence:** `results/dgx/cache_audit/` contains 216 pair estimates, explicit undefined correlations and calibration-transfer diagnostics. Zero duplicate task rows and zero inconsistent gold labels were found; three missing answers and six source-marked nonreproducing rows remain disclosed. The v1 main replay used 144 worlds, 75.76 seconds and 504.20 MiB peak sampled RSS. Original confidence results remain under `results/correlation/`, with their provenance retained.

**Limits:** Same-cache replicas add dependent decisions, not independent generations. Freshly computed fingerprints do not prove historical byte identity where old manifests lack hashes. Trusted measurement requires trusted collection; sender-supplied fields can be forged. All new experiments are symbolic cache replay, without new model inference.

## 4. Disjoint clean calibration and transfer — new

**RQ4:** Do ceilings learned from separate clean tasks reduce honest-channel inversion while retaining useful adversary inversion?

**Implementation:** `src/aip/extensions/calibration.py` and `scripts/extension_calibration.py`. The calibrator consumes local public answers on explicit CAL IDs; the study separates CAL, HISTORY, VALIDATION and TEST. It compares legacy, pooled-bootstrap and conservative partner ceilings, with channel decisions and answer accuracy reported together.

**Evidence and conclusion:** See `extensions/C4_CALIBRATION.md` and `results/extensions/calibration/REPORT.md` for the completed run, exact split sizes, uncertainty, false-inversion diagnostics and measured answer to RQ4. An uncertainty-aware ceiling can become so conservative that useful inversion disappears; calibration quality and task utility are distinct endpoints.

**Limits:** A clean reference population is an explicit assumption. This is new task separation inside an existing artifact, not proof that the cached tasks were never examined historically. A bootstrap ceiling is not a finite-sample false-discovery or familywise error guarantee.

## 5. Clone-aware peer evidence caps — new

**RQ5:** Can response-stream grouping reduce duplicate vote amplification without removing useful inversion?

**Implementation:** `src/aip/extensions/clone_aware.py` and `scripts/extension_clone.py`. Learn groups from aligned historical response agreement, then cap each group's peer mass at its strongest member before the same requested self parity. Original AIP gate fitting remains, allowing a direct comparison of capped and uncapped aggregation.

**Evidence and conclusion:** See `extensions/C5_CLONE_AWARE.md` and `results/extensions/clone/REPORT.md`. Controlled additions of 0, 5 and 15 exact source replicas preserve the base answers and evaluate only original honest receivers. The report includes realized corruption fractions, grouping diagnostics and paired accuracy contrasts.

**Limits:** Caps constrain aggregation mass, not the complete fitted estimator. Replicas may alter gate statistics, identity-count corrections and self parity. Highly accurate honest agents may have similar histories too. Exact clones are narrower than paraphrasing, strategically diverging or independent model attacks.

## 6. Causal online adaptation under changing attacks — new

**RQ6:** Does a strictly past-only rolling or cumulative gate improve performance when attackers change behavior?

**Implementation:** `src/aip/extensions/online.py` and `scripts/extension_online.py`. A two-phase API requires prediction before observation/update. Fixed, cumulative and rolling histories share the same channel algorithm. Tests enforce prefix invariance and prevent current or future observations from influencing earlier predictions.

**Evidence and conclusion:** See `extensions/C6_ONLINE.md` and `results/extensions/online/REPORT.md`. Sleeper, regime-switch and toggle schedules are compared with fixed-regime streams on the same tasks. Chronological traces, channel states, window accuracy and paired block intervals show when adaptation helps or harms.

**Limits:** A short rolling window has less statistical support. A failed observable can remain uninformative after a refresh. Time blocks also differ in task difficulty. Finite dependent sequences and symbolic schedules support a bounded empirical answer, not a general regret or adversarial recovery theorem.

## Research status and external foundations

The contribution is a transparent experimental artifact with implementation, controls, evidence and explicit failure cases. It is not a claim of six new theoretical results or state-of-the-art superiority. The original manuscript remains a historical document whose stronger claims must be revised before submission as a current paper. The detailed v2 README and extension reports are the maintained account of the new experiments.

Full-confusion Dawid–Skene follows the latent categorical error-rate model of [Dawid and Skene, 1979](https://academic.oup.com/jrsssc/article/28/1/20/6953573). The local implementation, initialization and prediction API are documented separately. Byzantine gradient methods such as [Krum, Blanchard et al., 2017](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) concern a different observation and optimization problem; their guarantees do not transfer to this categorical cache replay. The temporal study uses block resampling as a dependence-aware descriptive procedure; the original work is listed in [Künsch's publication record, 1989, entry 56](https://people.math.ethz.ch/~hkuensch/papers/). None of these citations establishes novelty or a guarantee for the new extensions.
