# Independent evaluation review — 12 September 2026

Scope: the supplied diagram report, downloaded README, decision rules, inversion thresholds, formal notes, and manuscript. This review proposes inexpensive, reproducible checks; it does not independently establish the original H100 results. No model inference, credentials, or publication were performed for this review. A subsequent, separately assigned implementation added a closed-label full Dawid–Skene baseline to the working copy; see the implementation note below.

## Three contributions and the evidence each needs

The report does not assign literal C1/C2/C3 contribution identifiers. The downloaded manuscript has five contribution bullets (mechanism, theory, design laws, protocol, evaluation), while its literal `[C1]`, `[C2]`, and `[C3]` are claims-ledger identifiers: within/cross-model correlation, effective swarm size, and attenuation/calibration, respectively. Preserve those existing identifiers. A three-part release can group the evidence as follows, using “Contribution 1/2/3” rather than overwriting claims C1/C2/C3.

| Contribution | Operational claim | Matched evidence | Allowed conclusion |
|---|---|---|---|
| Contribution 1 — Receiver-local adversarial information pooling | The invert action can improve an otherwise identical trust/discard mechanism on identifiable coherent attacks. | Same broadcasts, tasks, receivers, topology, seeds, and calibration for `aip_gated` versus `aip_trust_only`; show the f=0 control and a corrupted slice. | An observed mechanism benefit on the tested cache/attack slice, with its paired uncertainty. |
| Contribution 2 — Answer-space and correlation calibration | Gold-conditioned wrong-answer coherence, receiver-conditioned coherence, and correctness covariance are different statistics; the appropriate calibration matters. | Binary exact-law control, four-class control, covariance-corrected identity, and cache pair statistics with denominators. | Validate an algebraic identity and quantify a cache-conditioned calibration gap; do not claim all honest errors are independent. |
| Contribution 3 — Adaptive boundary and transparent failure analysis | An attacker can change coordination to evade a coherence gate; mitigations may improve only part of the attack space. | Coherence sweep; p selected on calibration tasks only; fixed p evaluated on held-out tasks; soft-gate and normalization controls; saved failure trajectories. | Conditional robustness and observed failure regions, not arbitrary-Byzantine tolerance or a universal impossibility theorem. |

These are evidence bundles for the existing project, not three invented algorithms or claims of novelty. The README's three correlation-ceiling extensions are compatible with this arc. Original claims C1/C2/C3 belong in the calibration contribution's claim map, with cached within-model resampling needed to independently revisit C1 and measured-versus-upper-bound estimator distinctions preserved for C2.

## Critical validity observations already visible in the sources

1. `README.md` treats Dawid–Skene as discarding adversarial information. Full confusion-matrix latent-label models can represent systematic anti-expertise. Use a narrower comparison, and label a one-coin or implemented approximation precisely if a full baseline is absent.
2. The README's broad statement that every discard defense has a structural ceiling at f=0.5 is too general. Scope it to the tested response-level rules, their assumptions, and the coherent attack construction.
3. `configs/decision_rules.yaml` retains a hard stop requiring flatness in f although the README says flatness was refuted on four benchmarks. A low-resource runner should not silently change a prior decision rule or advertise that historical rule as satisfied. Preserve it and provide a separate current protocol.
4. `configs/inversion_thresholds.yaml` retains old model-population provenance, superseded numeric comments, and a statement that no benchmark is binary despite BoolQ being configured. Source calibration must be versioned separately from any new synthetic calibration; do not overwrite it with an in-sample fitted threshold.
5. `docs/proposition1_note.md` defines gold-conditioned coherence while the deployed gate uses receiver-conditioned coherence. Those are unequal when the receiver errs. A result about one statistic does not automatically establish a result about the other.
6. Exchangeability alone does not imply a Binomial law for all pair collisions. Collisions sharing a peer are dependent, even with independent peer labels; equal mean q also need not imply equal distributions. A defensible indistinguishability statement assumes equality of the complete observable law (or specifies independent Bernoulli collision observations as a separate model). Do not infer a theorem from an empirical q below a reference line.
7. The correlated-correctness expression in `docs/attenuation_law.md` §6 incorrectly cancels a covariance term. With δ = Cov(1[agent i wrong], 1[agent j wrong]), the exact identity is `m = (1-e_i)(1-e_j) + δ + (e_i*e_j + δ)*q`. Both-correct probability contains +δ and both-wrong probability contains +δ. This deserves a direct analytical check.
8. The binary identity q=1 when both peers are wrong is exact, but an attenuation identity formed by replacing joint error probability with the product of marginals still requires independent correctness or its covariance correction.
9. The measured covariance sign/bias claims need to be separated from algebra. A nonpositive expression cannot be described as overstatement of the same quantity without identifying the transformed estimator.

## Feasible low-resource protocol

Use the supplied cache rather than loading a language model. Limit the process to two CPU cores and two native math-library threads; set `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `NUMEXPR_NUM_THREADS=2`, and `VECLIB_MAXIMUM_THREADS=2`. Prefer process affinity to two allowed cores and one Python worker. No GPU inference is needed for this audit. Record the host/GPU inventory as context without claiming the GPU was used.

Start with existing cached MMLU (C=4), MedQA (C=4), MATH-500 (open answer), and BoolQ (binary control). Use the available frozen tasks and models actually present; incomplete coverage must appear in the manifest and tables. For genuinely independent replicates, a small synthetic harness can supplement these tasks, but report it as a synthetic mechanism check separately from cached LLM evaluation.

An affordable cache sweep is f in {0, 0.3, 0.5, 0.7}, N=10, three fixed composition/attack seeds, fully observed topology initially, and the following rules: receiver-only, majority, `aip_trust_only`, `aip_gated`, and any existing soft-gate variant. Add the repository's confusion-matrix baseline if it is implemented and compatible. Do not invent baseline names for a different algorithm. Add one smaller N slice and one partial-observation slice only after the base run succeeds within the budget. The original caches have only 200/500 tasks per benchmark; additional agent seeds do not create additional independent task observations.

### Required controls and ablations

| Control | What it resolves | Required record |
|---|---|---|
| No adversaries | Whether honest channels are inverted and whether the defense harms baseline behavior. | Accuracy difference, false inversion count/eligible honest decisions, gate decision counts. |
| Receiver-only | Whether apparent robustness comes from the honest self-anchor alone. | Same receiver/task accuracy and self-vote share. |
| Trust-only versus gated | Effect attributable to enabling inversion. | Paired task/receiver predictions from identical broadcasts. |
| Binary C=2 | Degenerate wrong-label coherence and gate limitations. | q denominator and gate should never cross a ceiling of 1. |
| C=4 independent wrong labels | Calibration against the known wrong-label chance reference. | Wrong-label support and measured versus expected q. |
| Correlated honest wrong answers | False detection caused by shared honest errors. | Honest phi, gold-conditioned q, receiver-conditioned q. |
| Constant/modified confidence | Whether a rule unexpectedly consumes self-reported confidence. | Identical answers with only confidence changed. |
| Gate-aware coordination sweep | Failure as coherence changes. | Full p curve; no selection from test accuracy. |
| Self-vote normalization at f=0.5 | Tie/normalization effects previously disclosed by the project. | Actual integer Byzantine count, receiver exclusion, vote weights, matched variant. |
| Variant activity probe | Whether a named mitigation changes behavior on a designed trigger. | At least one differing gate/weight/decision before reading effectiveness. |

For every finite N, record actual `n_byzantine/N`; do not label a rounded fraction as exactly the requested f. Receiver evaluation must be restricted to agents explicitly honest in the experimental generator, without leaking those labels to the deployable defense. An oracle baseline may receive labels only when plainly identified as an oracle.

For attacks, keep the semantics of p explicit. The expression `p^2 + (1-p)^2/(C-1)` describes one label of mass p and C-1 disjoint other labels sharing the rest. If the focal label is a lie, those other labels include the truth. It is not the law for an always-wrong distribution spread among C-1 wrong labels. Receiver conditioning and within-swarm finite pair sampling can further change the measured statistic. Validate the actual generator's distribution, then label any theory curve with that exact distribution.

## Seeds, splits, and uncertainty

1. Canonicalize JSON with sorted keys and stable separators, then derive seeds from SHA-256 of explicit semantic identifiers. Never use Python's salted `hash()`.
2. Separate random streams for task partition, honest broadcast generation, attacker generation, observation topology, tie-breaking, and bootstrap. Share the underlying broadcast/attack/topology draws across compared methods. An extra random call in one method must not change another method's inputs.
3. Freeze task identifiers, model roster, p candidates, primary comparisons, and split definition before the new run. Hash task IDs into calibration/test sets. No task may be in both. Keep all receivers and model responses for one task in the same split.
4. Select adaptive p on calibration data only, using a deterministic tie rule, then freeze it for test evaluation. Report both the calibration-selected attack and the whole test p curve as exploratory. Do not relabel the best test p as a held-out attack.
5. Evaluate primary accuracy over honest receivers but aggregate to one value per task before resampling. For a paired comparison use the vector of task-level accuracy differences, sampling task IDs with replacement, preserving all seeds/receivers/methods within each sampled task. A 1,000-resample percentile interval is adequate for this audit; record its bootstrap seed and sample count.
6. Report the number of unique tasks and number of composition/attack seeds separately. Do not treat sweep rows, agent pairs, receivers, and cached resamples as independent examples. If enough independently sampled compositions exist, a second stage of hierarchical resampling can quantify that component; otherwise state that the interval is conditional on the selected cache/compositions.
7. Give mean paired difference and 95% interval, baseline/defense accuracies, task count, disagreement count, and timing. Report negative and inconclusive findings. Separate the historical measurement floor from new paired intervals; `sqrt(2)*1.96*sigma` is not a conventional powered MDE because it has no specified power term.
8. Predeclare one primary comparison per contribution. Treat remaining curves as exploratory; if p-values are used across a family, correct multiplicity (for example Holm). A descriptive release does not require significance stars or a new best-method leaderboard.

## Saved simulations and trajectories

Save compact JSONL or compressed JSONL traces for deterministic task IDs chosen before looking at outcomes, plus explicitly labeled diagnostic traces selected for observed failures. Record which selection rule produced each trace. Retain all per-task prediction rows in Parquet/CSV so figures and paired intervals can be regenerated without running aggregation again.

Each trajectory should include run/config identifier, task ID or synthetic truth, split, seed streams, model/agent IDs, experimental honest/Byzantine status, received answer and confidence, observation mask, receiver's self-answer, actual attack parameters, measured gate statistics with denominators, calibrated threshold source, trust/discard/invert decision, pre/post transformation weights, final prediction, correctness, and method name. Gold and attacker identities are offline evaluation fields and must not be passed to the deployable decision function.

If the source implementation is one-shot cached aggregation, call these receiver decision traces. Do not invent conversational rounds, simulated dialogue, or live agent trajectories that did not happen. A synthetic example should state `data_origin=synthetic` and never carry a real model name.

Manifest fields: UTC start/end, host and CPU affinity, thread caps, Python/package versions, source file hashes or Git commit, exact command/config, cache hashes and row counts, task/model counts, all seeds, row schemas, environment mode, peak memory, wall time, and artifact SHA-256 hashes. Exclude tokens and credentials. A cache file alone should not be described as new DGX model inference.

## Acceptance checks before the release is called ready

- Input task IDs are unique per model/sample, answer spaces match the task configuration, and all methods compare the same records. Missing cache coverage is explicit.
- Gold-answer lookup is isolated from the deployed aggregation path. Labels are permitted only for calibration on the calibration split, attack construction when oracle knowledge is specified, oracle baselines, and scoring.
- All probability estimates lie in [0,1]; undefined denominators are represented as missing with counts, not silently set to evidence of safety. Accuracy denominators exclude no difficult tasks after outcomes are known.
- A tiny deterministic rerun yields the same predictions, gate decisions, metrics, and trace content; timing/environment timestamps may differ.
- f=0, binary q, covariance identity, attack-support law, and mitigation-activity controls behave as specified, with any failure written to the claims table instead of suppressed.
- Every result table has matching method/task/model/seed bases. No L4-era result is presented as a new H100/DGX result. No original report number is described as independently reproduced merely because it was copied.
- Every new claim has evidence paths, exact slice, uncertainty, and status (supported within slice / contradicted / inconclusive / not tested). The status does not assume a positive scientific outcome is needed for a successful run.
- Three contribution narratives map to actual checks and outputs. Novelty, external validity to live LLM swarms, and broad robustness remain bounded by the executed evidence.
- Saved plots read result files, saved examples have stated selection rules, and the release manifest verifies all required artifacts. Test source, runnable command, and low-resource configuration are included.

## Subsequent implementation note

The separately assigned `work/src/aip/aggregation/dawid_skene_full.py` implements `DawidSkeneFullAggregator(label_space, max_iter=100, tol=1e-7, smoothing=0.5, init_accuracy=0.7)`. Each receiver learns a class prior and one C×C confusion matrix per observed worker from history only. Its held-out prediction computes a stable log-likelihood from those frozen parameters, including off-diagonal anti-expert likelihoods. It ignores Byzantine flags, gold labels, confidence, missing answers, and workers unseen in history. It requires an explicit fixed closed label space and documents conditional-independence, stationarity, label-orientation, and local-optimum limits.

`work/tests/test_dawid_skene_full.py` passed 23 checks in 0.10 seconds with native thread caps of two. Checks include cyclic anti-expert recovery with competent majority workers, a genuinely unseen task whose predicted truth is not reported by any present worker, per-receiver learning, missing-only training neutrality, hidden-field and Byzantine-flag invariance, deterministic bounded iterations, and numerical stability. These are synthetic software/mechanism checks, not benchmark results. The original scalar baseline remains untouched; it caches fitted-task predictions and falls back to majority on unseen task IDs, so it must not be presented as a fitted held-out full confusion-matrix comparator.

Source references: supplied `Liars_Are_Information_AIP_Detailed_Diagram_Report_2026.pdf`, especially §§2, 6, 7, 11–12; downloaded `README.md`, `configs/decision_rules.yaml`, `configs/inversion_thresholds.yaml`, `docs/proposition1_note.md`, `docs/attenuation_law.md`, `paper/sections/introduction.tex`, `paper/sections/design_laws.tex`, `paper/sections/theory.tex`, and the new baseline/test files above.
