# Multi-agent contribution and review record: v2

The user asked for three additional implemented contributions while retaining all previous work. The coordinating agent split the independent implementation/evaluation tasks among three engineering agents. Four agents worked in total. Scientific experiments were serialized on the DGX; parallel code review did not imply parallel heavy jobs.

These engineering agents are distinct from the simulated experimental swarm. The latter consists of identities replaying cached answers and symbolic attackers. Engineering-agent messages, implementation activity and test counts are not model-inference data and do not count toward benchmark sample size.

## Ownership and concrete work

| Engineering agent | Bounded ownership | Concrete delivered work |
|---|---|---|
| `/root` | Integration, preservation, research framing, independent reconciliation, release delivery | Copy v1 into a separate v2 workspace; retain every prior scientific artifact; write the original-five-to-six crosswalk, internal analysis plan, detailed README/evaluation guide; independently check C4 score arrays, intervals and weighted trace decisions; integrate all study evidence; run combined checks; fingerprint and publish a versioned HF release; verify uploaded objects. |
| `/root/research_audit` (Dewey) | Contribution 4: clean-task calibration | Implement receiver-local task-separated pooled/partner calibrators and threshold transfer; add leakage, missingness and binary tests; run six-benchmark DGX simulations; save raw CAL inputs, thresholds, receiver predictions, channel diagnostics, comparisons and plots; independently review Contribution 5. |
| `/root/evaluation_review` (Epicurus) | Contribution 5: clone-aware evidence caps | Implement aligned complete-link grouping, group mass caps and matched fitted-state reuse; add grouping/cap/input tests; run controlled honest/Byzantine replica studies at two visibility levels; save grouping/ancestry evaluation, task decisions, actual fractions and plots; independently review Contribution 6. |
| `/root/dgx_audit` (Heisenberg) | Contribution 6: causal online evaluation | Implement explicit predict/commit API and fixed/cumulative/rolling retention; add temporal leakage and metadata tests; run changed and fixed-regime streams on identical tasks; save every broadcast/prediction, exact fit history, channel state, block intervals, recovery diagnostics and resources. |

Each agent owned separate modules, scripts, tests, notes and result directories. Shared v1 code was treated as frozen input. The original source snapshot and the prior published release were preserved outside the editable v2 tree. The [v1 multi-agent ledger](MULTI_AGENT_REVIEW.md) and [v1 final review](DGX_FINAL_REVIEW.md) retain the earlier work in full.

## Review findings and disposition

| Finding or design risk | Resolution and remaining boundary |
|---|---|
| The prior delivery grouped work into three contributions, while the user intended three additions. | Preserve the original five manuscript bullets through an explicit crosswalk; retain v1 Contributions 1–3 and implement new Contributions 4–6. No claim that regrouping old work is new research. |
| Historical calibration overlaps available cached task lists. | Add explicit CAL/HISTORY/VALIDATION/TEST separation for the new calibrators, and disclose historical exposure. Legacy-threshold comparisons retain the overlap caveat. |
| Partner-selected calibration can become excessively conservative. | Measure it as an endpoint. Every standard partner ceiling saturated at one; the report describes the loss of useful inversion, rather than claiming that larger ceilings necessarily improve robustness. |
| Peer pairs and receivers are dependent within a task. | Resample whole CAL or TEST tasks for static studies; average receiver scores and fixed seeds before paired static inference. Count channels only descriptively. |
| Replication changes both voting mass and the fitted gate. | Copy the same baseline fitted state for each within-world cap comparison. State the cap's limited weight guarantee; report estimator changes and actual-N corrections across clone counts. |
| Similar streams do not prove common provenance. | Group only from public histories. Report two evaluator-only recovery targets: intentional copy ancestry and shared cached/attack source; disclose false merges and the narrow exact-replica attack family. |
| Partial visibility can become a confound when population size changes. | Keep original base-edge visibility draws fixed while adding clone edges; preserve base answers, source selection and original scoring receivers across clone counts. |
| A temporal gate may accidentally use current or future observations. | Enforce predict-before-update, reject inconsistent commits/duplicate tasks, fit only committed history, and save exact fit-task IDs and trace state references. Never invoke the old window path. |
| Raw pre/post accuracy can reflect task difficulty. | Run fixed-regime controls over the identical chronology and current broadcasts; interpret their gap as history exposure under this controlled generator. |
| Temporal bootstrap may underrepresent long dependence. | Use paired circular blocks within regime segments with seed resampling, disclose the 16-task block and 200-draw standard limit, and avoid a formal regret or recovery guarantee. |
| Resource receipts need to distinguish sampled RSS from address-space controls. | Each script records its actual mechanism. Use the same shared experiment lock and record actual runtime, CPU affinity, thread count and GPU exclusion. |
| Scientific-looking row counts can inflate apparent evidence. | Separate world, task/method, receiver-prediction and benchmark-task counts in the overview. Do not pool accuracy across different studies. |

## Independent evidence checks

The integration agent's [C4 reconciliation](../results/extensions/calibration/independent_review.json) verifies 67 source/input/output hashes, all 144 world definitions, 90,000 task/method rows, 562,500 receiver correctness values, 1,440 summary rows and 192 paired pointwise intervals. It also independently reconstructs weighted aggregation for 9,000 saved predictions across 288 test traces, validates 1,080 Byzantine trace broadcasts under the declared scorer, and checks 9,000 clean CAL observations. The reconciliation script reuses the stated scorer; it does not independently adjudicate mathematical truth or refit every channel.

Contribution 5 receives a [separate scientific source review](../results/extensions/clone/independent_review.md) by the calibration agent. Contribution 6 receives a [separate causal/protocol review](../results/extensions/online/independent_review.md) by the clone agent. These reviews state their scope and findings; a source review is not a full independent rerun. The combined [v2 integration review](DGX_V2_REVIEW.md) records result-level checks, tests, provenance preservation and publication verification boundaries.

Smoke reproducibility comparisons use fresh processes and compare scientific records, treating compressed-container timestamps, plot metadata, timing and resource statistics separately. Successful smoke reproduction does not claim a second complete standard run. Canonical standard outputs remain unchanged after they are fingerprinted.

## Publication and authorship

The versioned v2 release is additive under `releases/dgx-research-20260912-v2`. The prior `releases/dgx-verification-20260912-v1` remains available. The bucket root README points readers to the detailed current content. The release contains no user authentication credential. Upload verification checks all expected file sizes and representative content digests after readback.

This ledger records engineering assistance and artifact provenance. It does not assign human academic authorship, replace the original authors, or claim novelty beyond the measured implementations and experiments. Original benchmark/model terms and the original project license remain applicable.
