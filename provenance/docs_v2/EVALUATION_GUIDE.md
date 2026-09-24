# Reading and reproducing the research evidence

This guide connects a question to code, an experiment, a statistical unit and a claim. Use the maintained README and v2 contribution notes for the new study. Historical result files and manuscript claims remain available for provenance.

## Three levels of evidence

| Evidence level | What it establishes | What it does not establish |
|---|---|---|
| Synthetic unit/regression tests | Behavior under constructed inputs: correct algebra, task alignment, cap bounds, missingness, no hidden-field use, causal ordering | Accuracy on natural questions, a statistical guarantee, or a novel research result |
| Cache-based multi-agent simulations | Conditional performance on the saved model answers, task sets, symbolic attacks and stated observation protocol | Fresh independent inference, naturally occurring model attacks, deliberation or tool-using agent performance |
| Historical model-inference artifacts | What the upstream artifact records from its original generation campaign | A rerun or independent verification by this DGX session; v1/v2 verify selected downstream behavior only |

The engineering agents used to implement and review the code are not the agents counted in the simulation. A ten-identity simulation can reuse the same cached model at several identities. A clone simulation adds identical source broadcasts. Neither creates fresh model samples.

## End-to-end observation contract

1. Load the verified cache intersection and one consistent gold label per task. Freeze each base agent's model assignment before selecting corrupted identities.
2. The generator replaces corrupted broadcasts according to the documented symbolic attack. It may use gold to ensure a proposed lie is wrong under the evaluation scorer. Save this fact as part of the threat model.
3. Apply a visibility mask to create each receiver's local observation. Own-answer visibility is required. Preserve common task positions when an observed peer or extracted answer is missing.
4. Remove `is_byzantine`, `source_model` and `attack` annotations before fitting or predicting. Defense code has no gold-answer argument. Calibration uses explicit CAL observations; static fitting uses HISTORY only; online fitting uses committed past observations only.
5. Evaluate the frozen prediction using gold. Hidden labels may be used to measure honest inversion, malicious trust, or grouping quality after the prediction is formed.
6. Save task-level decisions, fits or diagnostic summaries, input hashes, source hashes, configuration, figures and a successful resource receipt. Treat incomplete output as a failed/partial run.

The separation is auditable, not cryptographically enforced against a malicious experiment script. Tests and independent trace review verify the concrete release implementation. A downstream change to the generator or fitting path requires new validation.

## Comparisons and denominators

| Study | Main independent resampling unit | Why other row counts are larger | Primary restrictions |
|---|---|---|---|
| v1 mechanism and stationary attack | Held-out task after averaging original honest receivers and the fixed swarm seeds | Multiple methods, visibility settings, attack candidates and corruption fractions reuse a task | 80 test tasks per benchmark; historical calibration overlap; selected-attack intervals condition on selection |
| v2 clean calibration | Whole CAL task for ceiling bootstrap; held-out TEST task for utility comparisons | Many receiver/pair events inside each task; several ceilings and target conditions | Task separation for new calibration; clean full-observation reference; no simultaneous confidence guarantee |
| v2 clones | Same held-out task under matched base swarm/source and method | Added clone identities, two source types, several clone counts and methods | Only original honest receivers are scored; actual corruption fraction changes with added identities |
| v2 online | Paired consecutive task blocks within the reported regime/horizon | Repeated methods, attack schedules and fits share a chronological task sequence | Short dependent stream, fixed chronology, finite block size, fixed seeds; intervals are descriptive |

Do not pool the studies' accuracies into one leaderboard: their task splits, histories, horizons and attacks differ. A larger number of simulations is broader coverage, not a corresponding increase in independent sample size. Pointwise exploratory intervals are not multiplicity-corrected discoveries. A confidence interval for a paired difference is more directly relevant than overlap between two separate method intervals.

## What each metric answers

| Metric | Question answered | Common misinterpretation to avoid |
|---|---|---|
| Mean task accuracy | How often did the tested method answer correctly under this protocol? | Higher average across a changed task mix does not isolate an algorithm improvement. |
| Paired method difference | How much did two methods differ on the same tasks/conditions? | Conditional intervals do not cover new model populations or unrestricted attackers. |
| Honest-channel inversion rate | How often did the fitted gate invert channels known by the evaluator to be honest? | More than zero does not automatically imply answer-level harm, and a zero estimate is not a population error bound. |
| Malicious trust/discard/invert | What action did the gate take on the symbolic attacker channels? | Labeling a channel INVERT does not guarantee the correct answer wins. |
| Receiver-conditioned coincidence | How often did peers agree while both dissented from the receiver? | It is not gold-conditioned wrong-answer coincidence when the receiver is wrong. |
| Learned group support | Which streams were sufficiently alike in the observed history to share an evidence cap? | Similar answers do not prove common ownership or malicious identity. |
| Before/after cap weights | Was duplicate peer evidence constrained as intended? | A pooling cap does not make the upstream channel estimator clone invariant. |
| Post-switch/window accuracy | What happened after a stated regime change? | A chronological block can be harder; compare same-task fixed-regime controls. |
| Cumulative loss relative to self | How many task-equivalent mistakes did cooperation add or avoid relative to the receiver? | This descriptive loss difference is not a proven online regret bound. |
| Runtime and peak RSS | What did this CPU replay cost on the measured host? | It excludes model inference and is not the original GPU campaign's cost. |

## Baselines and assumptions

| Method | Historical fit? | Key evidence/assumption | Scope of comparison |
|---|---|---|---|
| Receiver-only | No | Receiver's own cached answer | Essential competence baseline |
| Majority | No | Current observed labels; identities count as votes | Sensitive to identity replication and corruption |
| Confidence weighting | No local channel fit | Stored confidence field plus chosen self parity | Historical collection assumptions; sender-controlled values are not authentic by default |
| SAC-style comparator | Yes | Repository's similarity filter/refine approximation | An implementation-level comparator, not a reproduction of every external SAC method |
| AIP trust-only | Yes | Same local estimates as gated AIP with inversion disabled | Closest mechanism ablation |
| AIP naive inversion | Yes | Inversion without the full calibrated gate | Tests the need for gating |
| AIP hard/soft/randomized | Yes | Receiver agreement plus selected-partner coincidence and heuristics | Finite empirical controls; dependent counts do not supply automatic Binomial error control |
| Full-matrix Dawid–Skene | Yes | Fixed categorical labels, stationary worker confusion, latent class prior | Closed labels only; EM ambiguity, independence violations and local optima matter |
| New calibration variants | CAL then HISTORY | Clean reference tasks and fixed ceiling transfer | Changes calibration, not available truth information |
| Clone-aware AIP | HISTORY | Near-identical aligned stream groups and group mass cap | Changes aggregation mass, preserving baseline fit within each matched world |
| Causal AIP variants | Past observations | Same AIP rule with fixed/cumulative/rolling retention | Changes temporal evidence; no future/current-task fit |

The full Dawid–Skene model can use an anti-expert through an off-diagonal confusion matrix. Claims that it intrinsically discards all liars are incorrect. See [Dawid and Skene's original error-rate model](https://academic.oup.com/jrsssc/article/28/1/20/6953573) and the release's implementation/tests for the distinction between the classical idea and this concrete comparator.

## Scoring and semantic sensitivity

Closed-label benchmarks use their explicit answer labels. MATH-500 uses the repository's approximate `math_match`; the new GSM8K calibration study uses its documented numeric scorer. Where an equivalent string differs from gold, exact-string correctness and semantic correctness may disagree. Inspect sensitivity fields on the same predictions.

Semantic correctness belongs to the scorer and the symbolic wrong-pool construction. Defenders still aggregate observed strings without a gold-assisted canonicalization step. For open answers, a shared/uniform lie pool must exclude scorer-equivalent answers and check its synthetic fallback. `K` in the collision equation is the realized number of distinct candidate strings, not the size of an unbounded theoretical answer space.

## Low-resource DGX operation

Use `requirements-dgx.lock.txt` in an isolated Python 3.12 environment. The original `pyproject.toml` also supports the historical inference stack; installing its GPU extra is unnecessary for the new replay. The lock records the actual replay environment, not universal wheel availability on every operating system.

The extension scripts serialize standard jobs using `/tmp/aip-dgx-v2.lock`. They restrict affinity to two available CPU cores, native numerical threads to one, nice priority to 10, and CUDA visibility to none. Each script documents its own memory/time mechanism in its resource receipt. A process RSS sampler and an address-space limit are different controls; refer to the actual receipt before asserting a hard limit. Experiment time begins after lock acquisition, so queue waiting is separate from measured scientific runtime.

Run new output directories, retain logs and check for a completed manifest plus a successful receipt. Do not delete the supplied standard results to make a command succeed. The initial legacy `scripts/dgx_run_bounded.py` is for v1 replay; each v2 extension script implements its own bounded launch. Running a second outer `flock` on the same lock can deadlock a script that already acquires it.

## Research questions that remain open

The release provides measured answers to six bounded questions. Stronger claims would need untouched external calibration/evaluation tasks, fresh repeated model generations, a wider independent roster, live model-driven attacks and interactive communication. Clone results do not cover arbitrary Sybils that deliberately diverge during history. Online results do not cover unrestricted reactive attacks or arbitrary drift. None of these gaps is hidden by a large world count, a narrow conditional interval, or a passing unit suite.
