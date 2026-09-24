# Six-contribution expansion: analysis plan

Recorded during implementation on 2026-09-12 (Asia/Seoul), before the new standard experiment results were integrated. This is an internal design record, not an externally timestamped preregistration. Smoke checks can inform runtime and software repairs; changes to the stated design must be disclosed in the final reports. Negative and mixed results remain deliverables.

## Scope and provenance

Retain the original five manuscript contribution bullets, all original caches/result tables, and every completed v1 DGX experiment. Organize that material into three retained research themes, then add three implemented extensions, giving six main contributions. The crosswalk in `CONTRIBUTIONS_V2.md` explains the grouping. This numbering does not rename the historical C1–C10 claims ledger.

The original artifact is a cache-based research project on Adversarial Informativeness Pooling (AIP). The new experiments replay existing model answers at simulated communicating agent identities. They add no model inference, dialogue generation, independently sampled completions, or tool-use trials. Four engineering agents build and review the artifact; their work is separate from the experimental swarm.

## Questions and evidence required

| Contribution | Research question | Required comparison or check | Interpretation boundary |
|---|---|---|---|
| 1, retained and verified in v1 | When does receiver-local inversion improve a matched trust/discard rule? | Same broadcasts, receiver histories, requested self weighting; trust-only, naive inversion, self, majority, confidence, SAC-style comparator, full-matrix Dawid–Skene on closed labels | AIP versus trust-only is a mechanism ablation. Other methods have different assumptions. Gains may reverse across benchmarks. |
| 2, retained and corrected in v1 | What attack-coherence law is actually implemented, and where does the gate fail? | Enumerate the shared/uniform wrong-pool sampler; correct covariance identity; validation-select a stationary attack per defender; report conditional task intervals | Scalar coincidence does not prove full-history indistinguishability. Candidate-specific histories are not one pretrained defense under a changing attack. |
| 3, retained and expanded | What does correlation, confidence provenance, visibility, and reproducibility permit us to claim? | Audit all 54 caches; retain confidence/protocol analysis as historical; use matched identities, local observations, saved traces, source hashes and measured CPU resources | Cache copies are dependent. Confidence is trustworthy only if an authenticated collection path supplies it. No new confidence-forgery experiment is planned. |
| 4, new | Does calibration from separate clean tasks transfer better than the historical ceiling? | Six benchmarks, three swarm seeds, four corruption fractions, full/half observation; legacy gate versus pooled and conservative partner ceiling estimates, trust-only and self | Clean calibration is an explicit assumption. Task bootstrap does not give a simultaneous channel error guarantee. Binary coincidence is degenerate. |
| 5, new | Can grouping repeated response streams and capping their evidence reduce clone amplification? | Base ten-agent swarm; 0/5/15 exact replicas of an honest or Byzantine source; hold base answers and source fixed; compare capped/uncapped gated and trust-only rules, self and majority | Only original honest receivers count. Report actual corruption fraction. Gate fitting can still change with clones; do not claim full clone invariance. |
| 6, new | Do strictly causal cumulative/rolling histories adapt to changed attackers better than fixed histories? | Predict, then update; fixed, cumulative, rolling32 and rolling64; sleeper, coherent-to-independent and toggle schedules; same-task fixed-regime controls | History length and refit interval are chosen before the standard results. Regime blocks differ in difficulty. Use temporal dependence-aware descriptive intervals, without claiming a deployment guarantee. |

## Planned new experiment designs

### Contribution 4: clean calibration transfer

Allocate task IDs into CAL/HISTORY/VALIDATION/TEST proportions 20/30/20/30 before observing outcomes. Fit per-receiver ceilings on the CAL IDs of an all-honest reference population with full observation, then use those fixed ceilings at target visibility 1.0 and 0.5. Calibrators receive public local answers and explicit CAL IDs, no gold labels. The experiment generator may know gold to construct scorer-valid attacks; the evaluator may use gold and Byzantine labels for diagnostics. The defense may not.

The alternatives are the frozen legacy ceiling, an upper bootstrap percentile of pooled receiver-conditioned coincidence, and a conservative maximum of partner estimates with bootstrap upper percentiles. Preserve every peer response on a task within a bootstrap cluster. Sparse or degenerate estimates need an explicit conservative fallback. The maximum strategy is a heuristic ceiling, not a familywise confidence construction. Record thresholds, support, task membership, false inversion of honest channels, malicious channel decisions, answer accuracy, paired task intervals and selected representative traces. There is no validation-based tuning in the initial fixed-method comparison; unused validation tasks remain reserved and documented.

### Contribution 5: repeated evidence and clones

Learn conservative groups from aligned nonmissing past observations, with minimum support and near-identity agreement. Grouping may not use model identities, source-clone IDs, gold, or adversarial labels. Cap each peer group's aggregate mass at its strongest single member before the requested self parity is applied. A peer in the receiver's own response group should not count the receiver's evidence again. Keep ordinary AIP channel fitting unchanged to isolate the additional aggregation cap and disclose the remaining estimator dependence.

Use MMLU, BoolQ and MATH-500; base corruption fractions 0.3/0.5; three seeds; exact source replicas 0/5/15, with full and half visibility if the 600-second budget permits. Keep source broadcasts, base identities and corruption fixed as clones are added. Evaluate original honest receivers only, average within tasks, and use paired task differences across matched conditions. Grouping precision/recall can use hidden clone ancestry in the evaluator only. Actual population size and corruption fraction change when a Byzantine or honest source is cloned; these are distinct from base corruption fraction and must be reported. Exact replicas test a narrow attack family.

### Contribution 6: causal nonstationarity

Use three benchmarks, base corruption 0.5/0.7, three seeds, a fixed chronology of 40 warmup tasks plus 160 distinct evaluation tasks. A receiver predicts the current task from the state fitted before that task, then may observe the current public broadcasts. Any refresh excludes the current prediction's answer from its fit. Methods: initial-history fixed gate, cumulative history refreshed every eight observations, rolling history of 32 or 64 observations refreshed every eight, self and majority.

Schedules change after 40 evaluation tasks; toggles also change after 80 and 120. Compare sleeper honest-to-coherent, coherent-to-independent wrong answers, and a toggle schedule. Run fixed honest/coherent/independent schedules over the same task chronology as difficulty controls. Preserve model identities before corruption. Use scorer-valid MATH-500 lies and omit hidden metadata from the defense. Save chronological predictions, public histories, channel diagnostics, window accuracy, cumulative loss relative to the receiver, timing, and within-regime paired moving-block intervals (planned block length 16). These intervals are conditional descriptive summaries with a short horizon, not asymptotic certification.

## Statistical and engineering rules

1. Do not treat peer pairs, honest receivers, clone replicas, world rows, or repeated swarm seeds as independent benchmark tasks. Average receivers and fixed seeds within a task before a task bootstrap; respect temporal blocks for streaming. Report the unit and horizon next to every interval.
2. Do not combine extension accuracy tables as if they used the same split, denominator, attack, or fitted state. Contribution 4 has a four-way split; Contribution 5 uses static held-out evaluation; Contribution 6 predicts a chronological stream.
3. Gold and adversary labels are permissible for the symbolic attack generator and the scorer, never for defense fitting or decisions. Save trace evidence of this separation. Check hidden-field invariance and missing-answer alignment.
4. Open-answer correctness uses the repository's approximate math/numeric scorer. Honest strings remain unchanged in the defense. Save exact-string sensitivity where applicable and identify the scorer. A scorer-valid lie is not a formal proof of mathematical incorrectness.
5. Run each standard study under the same `/tmp/aip-dgx-v2.lock` to avoid competing jobs. CPU affinity: two available cores; native threads: one; nice: 10; CUDA hidden; wall limit: 600 seconds; sampled process-tree RSS budget: 2 GiB. Save success/failure receipts. A sampled RSS monitor is not a hard kernel reservation.
6. Keep pilots and any failed attempts distinguishable from canonical standard runs. Runtime-driven design changes must be documented. Do not silently overwrite historical results or use a result directory alone as proof of completion.
7. Verify implementation with substantive tests, then independently reconcile manifests, source/input hashes, task splits, row counts, representative predictions, statistics and conclusions. Publish an additive dated release and read back representative objects after upload.

## Required delivery

A detailed README; six-theme crosswalk; a runnable module, script, tests and technical note for each new extension; machine-readable per-task data and comparison tables; saved representative simulations; standalone figures; actual host/resource receipts; integration review; reproducible commands; a complete SHA-256 manifest; and an updated bucket landing page linking the immutable versioned release and its predecessor.
