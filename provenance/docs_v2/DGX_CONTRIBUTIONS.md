# Three contributions in the DGX verification release

This release adds a controlled verification layer to the existing Liars Are Information artifact. Its three contributions are an evaluation baseline and inversion controls, a corrected account of the adaptive attack, and reproducible low-resource replay of cached multi-agent decisions. They do not establish three new algorithms or reproduce the original GPU inference campaign.

The original claims ledger uses identifiers such as **C1** for within/cross-model error correlation, **C2** for effective swarm size, and **C3/C4** for attenuation and calibration. Those identifiers retain their original meanings. “Contribution 1”, “Contribution 2”, and “Contribution 3” below are release groupings, not replacements for the old claims.

## Contribution 1 — Inversion controls and a full confusion-matrix comparator

The scientific question is whether enabling inversion improves the same receiver-local trust/discard mechanism on the same observations. `scripts/dgx_low_resource.py` compares `aip_gated` with `aip_trust_only`, a naive inversion variant, receiver-only predictions, majority vote, confidence weighting, and the existing SAC-style filter/refine approximation. The gated/trust-only comparison is the primary mechanism ablation. The tested weighted methods use the same requested self-vote share; majority and probabilistic Dawid–Skene retain their native aggregation definitions and should not be described as matched causal ablations.

`src/aip/aggregation/dawid_skene_full.py` adds a full class-conditional Dawid–Skene baseline. A receiver learns a class prior and one C×C worker confusion matrix from its own historical observations. At prediction time, it combines the learned likelihoods on genuinely unseen tasks, so a consistently wrong worker can contribute evidence for another label. The legacy one-coin aggregator stores fitted-task answers and falls back to majority for unseen task IDs; it remains historical code and is not used as the new full-matrix comparator.

The new baseline requires an explicit fixed closed label space. The replay uses it on MMLU and BoolQ; MATH-500 does not receive a full-confusion-matrix result. Missing answers and workers unseen in history contribute no likelihood. The code reads no gold labels, Byzantine flags, model identities, or confidence scores. `posterior()` and `models_` expose learned probabilities and convergence diagnostics for inspection.

This comparator assumes stable class-conditional worker behavior and conditional independence given truth. Replaying the same model at several agent identities violates that independence. Diagonal initialization selects a better-than-chance label orientation; it does not identify truth under an adversarial majority. Sparse history, latent-label symmetry, nonstationary attacks, and EM local optima can all limit performance. Its synthetic anti-expert tests demonstrate software behavior under specified conditions, not universal recovery from malicious majorities.

Evidence paths:

- Implementation: `src/aip/aggregation/dawid_skene_full.py`.
- Validity checks: `tests/test_dawid_skene_full.py`, including unseen-task prediction, anti-expert recovery with competent majority workers, receiver-local fitting, missingness, hidden-field invariance, bounded deterministic iteration, and numerical stability.
- Replay outcomes, when a run completes: its `per_task.parquet`, `paired_comparisons.csv`, `channel_diagnostics.parquet`, and `traces.jsonl.gz`.

No improvement magnitude or winner is asserted in this contribution description. The measured result can support, contradict, or leave the inversion comparison unresolved.

## Contribution 2 — Corrected adaptive-attack geometry and limits

The gate-aware mixture either follows a shared lie with probability p or draws independently from a pool of K candidate lies. The independent branch can draw the shared lie too. For a uniform candidate pool, the expected agreement of two Byzantine agents is therefore

```text
q_pair(p, K) = p² + (1 − p²) / K.
```

For closed C-class tasks, K=C−1. At p=0 the expectation is 1/(C−1), and increasing p increases this expectation. The old expression `p² + (1−p)²/(C−1)` describes a different disjoint-support distribution; it cannot justify a below-chance expectation for the implemented always-wrong sampler. Open-answer tasks have a task-specific finite pool of distinct candidate strings, including a one-candidate fallback. They cannot be assigned zero chance collision merely because their nominal answer space is open. K describes the actual string pool; different strings may still denote equivalent mathematical answers.

`scripts/phase_gate_aware.py` now labels expected and empirical Byzantine-pair coincidence with the relevant pool/count/scope metadata. Its native legacy sampler is unchanged. That sampler excludes answers by exact string inequality, which can leave a mathematically correct but differently written answer in an open-answer “wrong” pool. The new DGX harness therefore uses a separately identified MATH-500 pool filtered by the benchmark's `math_match` scorer and checks the fallback with the same scorer. This changes the new open-answer attack distribution intentionally; it is not a byte-for-byte replay of the legacy open-answer sampler. Closed label spaces retain the existing sampler semantics. The mixture equation applies using the actual pool for each implementation.

Historical result files and paper numbers remain historical outputs; correcting their theoretical axis or scoring protocol does not silently regenerate or validate their conclusions. Actual receiver-conditioned, partner-selected coherence differs from this Byzantine-pair expectation and must be inspected separately.

The formal notes also correct the covariance term in the attenuation identity. With binary error indicators E_i and E_j, error rates e_i and e_j, wrong-answer coincidence q, and δ=Cov(E_i,E_j),

```text
P(answers agree) = (1−e_i)(1−e_j) + e_i e_j q + (1+q)δ.
```

The identity q=1 for two wrong binary answers does not make δ vanish. Empirical dependence and receiver-conditioned calibration are part of the evaluation, not assumptions justified by the binary identity alone.

`docs/proposition1_note.md` withdraws the inference that equal mean coherence, or exchangeability by itself, implies an indistinguishable observable distribution or a Binomial count of overlapping pair collisions. A valid total-variation testing bound concerns the complete specified observable distributions. The deployed gate also uses history and receiver-agreement features, so a result about one scalar coincidence statistic does not establish impossibility for the complete defense. Finite-sample uncertainty and a finite attack grid are not standalone proofs of robustness or impossibility.

The new adaptive replay selects p using validation tasks only, separately for each defender and swarm seed, with an ascending-p tie break. It then reports test accuracy at that selected p. **Each candidate p has its own stationary p-specific historical observations and independently fitted defender.** This evaluates stationary attack regimes. It does not simulate an attacker changing p after one common defender has been trained. For randomized thresholds, validation and test share the same seeded defender realization; the experiment does not independently test secrecy of the random draw or unrestricted attacker adaptation.

Evidence paths:

- Corrections: `src/aip/swarm/broadcast.py` documentation, `scripts/phase_gate_aware.py`, `docs/attenuation_law.md`, and `docs/proposition1_note.md`; scorer-valid open-answer attack construction in `scripts/dgx_low_resource.py`.
- Checks: `tests/test_coherence_formula.py`, including branch/label enumeration and agreement with the actual sampler.
- Completed replay evidence: `selected_attacks.csv`, `selected_attack_predictions.parquet`, `worlds.json`, and saved decision traces in the run directory.

## Contribution 3 — Reproducible low-resource cache-based multi-agent evaluation

The replay uses ten simulated agent identities. Honest agents emit frozen greedy model answers already present in the repository; symbolic attackers emit task-specific answers checked as wrong under the evaluation scorer and are explicitly allowed to know gold. The simulation evaluates receiver decisions under complete-graph communication with full or partial observation. It does not call live LLMs, generate conversations, or execute tool-using agents. Same-model replicas replay the same answer and must not be counted as independent generations. Saved records are receiver decision traces, not newly generated dialogues.

The model identities are assigned before corruption, with a seed-fixed shuffled roster. Increasing f masks a nested set of corrupted identities while preserving the remaining identities' models. This avoids changing the honest roster merely by truncating its alphabetical ordering. Separate seeded streams generate corruption, attacks, and visibility. All methods within a world receive the same broadcasts and visibility masks. Static AIP statistics preserve task positions under missing observations; the replay does not use the historical windowed mode.

The task IDs are split deterministically into disjoint history/validation/test sets in a 40/20/40 ratio. Each honest receiver's stateful defense fits only that receiver's history. Gold answers and true Byzantine identities are used by the experimental generator and scorer; the defense receives broadcasts with hidden identity/model/attack metadata removed. Predictions are frozen across validation and test; validation selects finite-grid attacks, and test tasks do not enter the new fits or attack selection.

**Scoring protocol:** the original Phase C aggregation evaluates exact string equality, which can reject equivalent MATH-500 expressions such as `1/2` and `0.5`. The canonical new DGX run uses memoized `math_match` for MATH-500 scoring and retains exact-string accuracy as a sensitivity column on the same predictions. MMLU and BoolQ remain exact closed-label comparisons. The open-answer attack pool is filtered using the same MATH-500 scorer so that a proposed lie is incorrect under the scorer that will evaluate it. Honest answer strings, defense comparison rules, and defense inputs are not canonicalized against gold; semantic matching stays in the experimental generator and scorer. Therefore this release does not add a semantic-equivalence-aware defense.

`math_match` uses normalization plus a limited symbolic/numeric fallback. It is an approximate benchmark scorer, not a general mathematical verifier, and it can miss or misclassify equivalences. “Wrong” in the new attack-validity check means wrong under that scorer. The exact-string sensitivity column changes only scoring of the new predictions; it does not recreate a legacy run with its different open-answer attack pool. An earlier local exact-string run is retained outside the canonical results as audit evidence rather than published as the canonical reproduction. Canonical run identity and completion must come from the final manifest after the scorer and sampler changes.

**Calibration caveat:** the existing honest-coherence thresholds were calibrated before this release. Their historical calibration tasks may overlap the current cached test tasks, and their model population differs from the current roster. The new split prevents leakage from new fitting and selection; it does not make these results wholly independent of historical calibration. `results/dgx/cache_audit/calibration_task_overlap.csv` documents available overlap evidence. This is a transfer diagnostic, not a new independent recalibration.

Accuracy is first averaged over honest receivers for each task. For paired comparisons, the same task's accuracies are averaged across the fixed swarm seeds, then task-level method differences are bootstrapped. The resulting intervals are conditional on the cached tasks, roster, seed set, and historical calibration. Receivers, worker pairs, repeated seeds, and sweep rows are not treated as additional independent benchmark examples. The intervals are pointwise and exploratory, without multiplicity correction; the historical measurement floor is not substituted for these new intervals. Adaptive selected-attack point estimates require the same caution.

Use the bounded launcher from the repository root with the prepared Python environment:

```bash
python scripts/dgx_run_bounded.py --profile smoke --out results/dgx/smoke_review --timeout 120
python scripts/dgx_run_bounded.py --profile standard --out results/dgx/standard_review --timeout 600
```

Choose a new output directory for each run. The launcher constrains the child to two available CPU cores, sets native math-library threads to one, runs at nice 10, hides CUDA devices, and monitors a 2 GiB RSS limit and wall deadline. RSS is sampled rather than enforced as an address-space reservation. It writes `resource_receipt.json` on success or failure and preserves the log. A directory or partial Parquet alone does not establish successful completion; inspect both the resource receipt and completed run manifest.

The separate cache audit checks cached-answer coverage and dependence, receiver- versus gold-conditioned coincidence, and historical calibration overlap. Its accuracy/dependence tables use the stored cache correctness fields, as documented in its manifest; the replay recomputes correctness of aggregate predictions under the stated scoring protocol. Its source file hashes fingerprint the current inputs; where historical sidecars lack content hashes, they cannot prove historical byte identity.

Expected evidence for each completed replay is `splits.json`, `worlds.json`, `per_task.parquet`, `summary.csv`, `paired_comparisons.csv`, `selected_attacks.csv`, `selected_attack_predictions.parquet`, `channel_diagnostics.parquet`, `traces.jsonl.gz`, figures, `REPORT.md`, `run_manifest.json`, and `resource_receipt.json`. Representative trace IDs are fixed before method outcomes are inspected. The manifest records the executed profile, seeds/configuration evidence, inputs, script identity, and resource information. Report actual completion and outcomes from these artifacts rather than transferring numbers from the historical manuscript.

The engineering agents who reviewed and built this release are documented separately in `docs/MULTI_AGENT_REVIEW.md`. They are distinct from the ten simulated swarm identities.
