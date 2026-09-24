# New extension 5: history-derived clone evidence caps

This extension asks whether pooling dependent answer channels as one source can reduce harmful copy amplification without removing the benefit of inversion. It adds a concrete AIP-derived aggregation method, an exact-copy stress experiment, and diagnostics that distinguish intentional copies from other observationally identical channels. It retains the existing mechanism, theory, design-law, protocol, and evaluation contributions; “extension 5” is a new release theme, not a replacement for an original claims-ledger identifier.

## Method

`src/aip/extensions/clone_aware.py` defines `CloneAwareAIP`. Its static AIP gate is fitted on the same receiver-local history as the native control. The experiment reuses a deep copy of that fitted state for the capped variant, so a within-world capped/native comparison changes pooling rather than the original channel estimator.

For each receiver, grouping uses only task IDs, peer IDs, and reported answer strings. Missing reports retain their task positions. A pair qualifies only when it has at least **20 jointly observed nonmissing historical answers** and at least **98% exact agreement**. A deterministic complete-link partition requires every pair within a group to qualify; an A–B–C chain cannot merge A and C merely because each resembles B. The group placement rule follows agent-ID order and does not claim to find a unique latent source partition. Neither model identity, source family, gold answers, nor Byzantine flags are used for grouping.

Let w_j be the original gate's nonnegative channel weight and G a learned group of peers currently observed by the receiver. Before self-vote parity, the cap is

```text
w'_j = w_j × max_{k in G}(w_k) / sum_{k in G}(w_k),  when group mass is positive.
```

Thus the group's total weight equals its strongest currently observed member's weight, preserving within-group weight ratios. A singleton is unchanged. If a group contains the receiver, its other members receive zero extra weight: the trusted self channel already represents that history. The receiver's own weight is retained, then matched self-vote parity of **0.1** is applied. If no positive peer mass remains, that target share is unattainable and the original self-only behavior is retained. The group's original trust/discard/invert decisions and the answer-space inversion transform are otherwise preserved.

This provides a **fixed-statistics pooling guarantee**: copies that remain in the same learned group, have the same gate decisions/current answers, and do not change its maximum weight cannot multiply that group's pre-parity evidence. It does **not** establish clone invariance of the whole defense. Adding copies before fitting can change AIP's blind error estimates, agreement reference, selected coherence partner, gate decisions, and the actual-N multiple-testing threshold. Partial observation can also fragment one true source into several learned groups. Similar histories establish an observed dependency pattern, not common ownership or Byzantine identity.

The interfaces are:

```python
from aip.extensions.clone_aware import CloneAwareAIP

method = CloneAwareAIP(
    "mmlu", mode="gated", label_space=["A", "B", "C", "D"],
    agreement_threshold=0.98, min_support=20,
)
method.fit(history)
answer = method.aggregate(observed_broadcasts, receiver_id)

# Paired experiment: preserve exactly the already-fitted native gate.
capped = CloneAwareAIP.from_fitted(native_aip, same_history)
details = capped.weight_diagnostics(observed_broadcasts, receiver_id)
```

`groups_`, `pair_support_`, and `training_task_ids_` expose the fitted grouping evidence. `aggregate` does not update these structures. The extension supports the static gate only; the legacy windowed path is outside this study.

## Controlled experiment

The experiment in `scripts/extension_clone.py` uses existing MMLU, BoolQ, and MATH-500 caches, base N=10, base Byzantine fractions **0.3 and 0.5**, and seeds **20260912, 20260913, and 20260914**. The native v1 assignment and semantically valid symbolic base broadcasts are fixed before cloning. A seeded original honest or Byzantine identity supplies **5 or 15 exact additional replicas**; the unmodified base world supplies the zero-copy control. Every replica preserves its source's answer on each task. No new model is loaded.

Only the **original honest receivers** are evaluated. Added honest receivers are not counted as new independent observations. The original receiver set and underlying base corruption remain fixed within a clone comparison. Actual N and realized Byzantine fraction are recorded, since cloning a Byzantine identity increases f and cloning an honest identity decreases it. Original communication edges use a task/receiver/peer-specific visibility seed, so increasing N does not redraw visibility for the original identities. Full visibility is mandatory; observation probability 0.5 is included if the declared resource budget permits it.

Six methods receive identical history and test observations:

- receiver-only and majority vote;
- native gated AIP and native trust-only AIP;
- capped gated AIP and capped trust-only AIP.

Both native and capped AIP use a Bonferroni peer count of **actual N−1**, including added identities. The correction is matched within a world but deliberately not frozen across clone counts. The matched weighted controls use self-vote share 0.1. Native majority vote retains its ordinary vote definition.

The task split reuses the deterministic v1 40/20/40 partition: 80 historical tasks, 40 reserved validation tasks, and 80 test tasks from a 200-item cache. Validation is unused because the grouping parameters are fixed before examining outcomes. Every fit sees only one original receiver's history. Hidden source/model/Byzantine metadata is removed before fitting and prediction. The symbolic attack generator and offline scorer are allowed to use gold; the defense is not.

MMLU and BoolQ use exact closed labels. MATH-500 uses the repository's approximate `math_match` scorer and the v1 scorer-valid wrong-answer pool; exact-string sensitivity is also saved. The defense continues to group and aggregate the unmodified reported strings, so this is not a semantic-answer-clustering method.

## Measurements and interpretation

The primary contrast is capped gated AIP minus native gated AIP for each benchmark, base f, clone count/source, and observation probability. Capped trust-only versus native trust-only checks whether dependence caps help without inversion. Capped gated versus capped trust-only measures whether inversion remains useful after capping. `clone_effects.csv` additionally compares each method's cloned world against the same seed/task's zero-copy world.

Receiver correctness is averaged within a task, then fixed seeds are averaged within that same task. Paired task bootstrap intervals use 1,000 draws. Replicas, receivers, and seeds do not create additional benchmark sample size. Intervals are exploratory and pointwise, without multiplicity correction; they remain conditional on the selected caches, seeds, and historically calibrated AIP thresholds.

Source-group diagnostics score two distinct hidden targets:

1. **Intentional copy origin:** a new replica and its original identity share a family; different original identities remain distinct.
2. **Shared cached-model or coherent-attack source:** original agents replaying the same model share a source, and the coherent symbolic bloc shares an attack source.

A merger can be a false positive under the first target while correctly consolidating identical broadcast streams under the second. Precision/recall are undefined when their relevant denominator is zero and are saved as missing. These diagnostics do not prove the learned groups are malicious or independently generated.

Inspect the completed [measured overview](../../results/extensions/clone/REPORT.md) for all-grid context, zero-copy controls, and paired difference-in-differences. The frozen `standard/REPORT.md` headline table specifically shows 15 additional copies at full visibility; machine outputs include both visibility settings and all copy counts. The reports include unfavorable as well as favorable paired cells and separate the cap's algebraic guarantee from observed end-to-end accuracy.

## Measured conclusion

The canonical 180-world experiment completed both visibility settings in 99.96 seconds with 536.28 MiB sampled peak RSS and no GPU/inference. With full visibility and 15 Byzantine copies, BoolQ at base f=0.3 falls from 0.869 to zero under native AIP while capped AIP stays near 0.898; the paired difference-in-differences is +0.868 [0.795, 0.932]. MATH-500 at base f=0.3 has a +0.216 [0.176, 0.257] difference-in-differences. However, native BoolQ at base f=0.5 is already at zero without added copies, so its large cap gain cannot be attributed to new copy amplification. At that setting capped accuracy itself falls by 0.010 after 15 Byzantine copies.

The cap also removes useful evidence: at base f=0.5 and full visibility, 15 honest MATH-500 copies yield native accuracy 0.704 versus capped accuracy 0.639, a −0.065 [−0.085, −0.046] paired effect. Capped inversion retains positive paired gains over capped trust-only in several MMLU/MATH-500 cells, but the binary BoolQ gate does not invert. With observation probability 0.5, broader-source group recall averages only about 0.20 in the 15-copy slice, compared with 1.0 at full visibility. These are conditional, pointwise results under the declared task bootstrap, not a general cap or grouping guarantee.

The added analysis in `results/extensions/clone/clone_amplification_contrasts.csv` evaluates `(cap_cloned−native_cloned)−(cap_zero−native_zero)` on matching task/seed records before averaging fixed seeds and bootstrapping tasks. Its separate receipt fingerprints the untouched canonical per-task input and specifies the deterministic seed protocol. This separates existing cached-stream dependence from amplification caused by additional copies.

## Execution and artifacts

From the repository root with the prepared environment:

```bash
python scripts/extension_clone.py --profile smoke --out results/extensions/clone/smoke --timeout 180
python scripts/extension_clone.py --profile standard --out results/extensions/clone/standard --visibility both --timeout 600
```

The launcher serializes execution with `/tmp/aip-dgx-v2.lock`, constrains the child to CPU cores 0 and 1, sets native numerical threads to one and nice to 10, hides CUDA devices, and monitors 2 GiB RSS and the wall deadline. A new output directory is required. The receipt records completion or failure; a partial directory is not a completed study. The RSS monitor samples the child process rather than providing a hard allocation reservation.

Outputs include `per_task.parquet`, `paired_comparisons.csv`, `clone_effects.csv`, `summary.csv`, `group_recovery.csv`, `group_pairs.parquet`, `groups.jsonl.gz`, `channel_diagnostics.parquet`, `traces.jsonl.gz`, `worlds.json`, `splits.json`, `run_manifest.json`, `resource_receipt.json`, `artifact_manifest.json`, a human-readable report, and `accuracy_vs_clones` PNG/PDF figures. Three predetermined test tasks per world retain raw broadcasts, original-receiver visibility, method predictions, learned groups, and pre/post-cap/parity weights. Source and input hashes identify the executed code and cache bytes; the script checks for changes to its declared source dependencies during execution.

`tests/test_extension_clone.py` checks missing-task alignment, complete-link behavior, group weight caps, receiver-group treatment, fixed-statistics copy behavior, unchanged fitted native gates, observational-only access, metadata invariance, and the absence of group updates from test calls.

Historical threshold calibration may overlap the cached tasks and was measured on a different model population. Greedy model replicas are dependent; finite answer spaces can create coincident histories without common provenance. The MATH scorer is approximate. This study covers exact-copy attacks against a static cached-answer defense, not paraphrased Sybils, strategic identity churn, general online adversaries, or live tool-using systems.
