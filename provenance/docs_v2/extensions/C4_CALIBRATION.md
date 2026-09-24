# C4 — Independent clean calibration and transfer

This is an additional empirical extension. The original method, manuscript contributions, frozen configuration, and historical results remain preserved. C4 estimates new constants in a separate result directory and compares them with the inherited constants; it does not replace the original contribution with a documentation change or claim a new aggregation algorithm.

## Research question

**RQ4:** How do frozen legacy coherence ceilings and ceilings estimated from an independent sample of current-population clean responses change honest-channel inversion and adversarial task accuracy, when corruption and observation probability change?

The practical issue is that the inherited gate compares a selected partner's receiver-conditioned coherence with a threshold whose calibration population and selection procedure differ. A small upper bound on an average coherence statistic need not be an appropriate upper bound after partner selection. Increasing that bound can reduce inversion, including useful inversion of a coherent adversary. Both effects must be measured.

The internal analysis plan fixes three variants before test scoring:

1. **Legacy gated AIP:** the existing benchmark-specific constants, unchanged.
2. **Pooled independent calibration:** a receiver-specific upper bootstrap bound on pooled coincidence probability, measured only on separate clean calibration tasks.
3. **Partner independent calibration:** a conservative upper bootstrap heuristic for the maximum eligible partner-pair rate, bounded below by the pooled ceiling.

Trust-only AIP and the receiver's own answer are controls. No variant is selected using test performance. This document is an internal protocol description, not an external preregistration.

## Separate calibration and evaluation

The same task-ID hash split is used across all swarm seeds and methods:

| Partition | Fraction | Use |
|---|---:|---|
| CAL | 20% | Estimate new thresholds from clean receiver observations only. |
| HISTORY | 30% | Fit each receiver's channel statistics separately, under the target corruption and visibility. |
| VALIDATION | 20% | Save an independent diagnostic score; no threshold selection is performed. |
| TEST | 30% | Final paired comparisons and representative traces. |

Each standard 200-task benchmark therefore has 40/60/40/60 items; the 500-task GSM8K cache has 100/150/100/150. Unique task identities, exact memberships, and hashes are saved. The code rejects empty, overlapping, duplicated, or incomplete calibration membership.

Clean calibration uses the frozen seven-model roster distributed across ten agent identities, with three independently seeded identity assignments. It runs at complete visibility. The resulting per-receiver thresholds transfer unchanged to target visibility 1.0 and 0.5 and Byzantine fractions 0, 0.3, 0.5, and 0.7. This measures both corruption transfer and a visibility change; it does not recalibrate after seeing the target test set.

The standard experiment covers ARC, MMLU, BoolQ, MATH-500, GSM8K, and MedQA: **144 worlds**, five methods, three seeds. It uses cached greedy responses and symbolic coherent wrong-answer attackers. The original v1 `build_world` fixes identities before corruption and separates visibility and attack randomness. MATH-500 keeps v1 semantic scoring and wrong-pool filtering. The extension uses the repository GSM8K numeric scorer and excludes numerically correct strings from the symbolic attack pool.

## Estimator and uncertainty

For a receiver `r`, calibration retains its own answer and each visible peer's answer. For every unordered pair of distinct peers `(j,k)` and task `t`, it records

```text
D[t,j,k] = 1 if both peers have nonmissing answers different from receiver r
K[t,j,k] = D[t,j,k] * 1[answer_j == answer_k]
```

The pooled estimate is `sum(K)/sum(D)`. Each bootstrap draw resamples complete tasks, preserving the full vector of dependent peer-pair events. The upper ceiling is the 97.5th percentile of bootstrap pooled estimates.

The partner statistic is the maximum pair rate with at least five observed joint dissents. Its bootstrap recomputes support and the maximum after each task resample. The partner ceiling is the larger of its upper percentile and the pooled upper percentile. A resample with no eligible pair uses one for the conservative upper calculation; an original calibration with no eligible partner assigns a partner ceiling of one. No joint dissent at all assigns both ceilings one. Binary receiver-conditioned disagreement has a single alternative, so both ceilings are structurally one. A numerical lower clip of `1e-6` avoids the inherited gate's degenerate zero-threshold branch.

These are **calibration heuristics**, not simultaneous confidence limits, per-channel predictive bounds, or family-wise error guarantees. In particular, a bootstrap upper bound on an average does not validate the native gate's selected-partner binomial p-value. Calibration follows the gate's exact extracted-answer equivalence; task utility separately uses semantic/numeric scoring on the open benchmarks.

## Information available to each receiver

Calibration receives only CAL observations; it has no gold-label argument. Before calibration and defender fitting, `is_byzantine`, `source_model`, and `attack` annotations are stripped. A defender is fitted using singleton sequences containing only its own receiver observations from HISTORY. Missing peer responses remain missing. Neither TEST nor VALIDATION observations enter a fit.

Gold is available to the offline symbolic attacker and evaluator. Raw annotated traces are saved for auditing, separately from the public observations supplied to defenders. Calibration input traces contain the sanitized clean CAL observations without gold.

## Measurements and interpretation

The saved per-task table includes the receiver IDs, individual predictions and correctness flags, and the mean honest-receiver accuracy. Paired method comparisons average seeds within each original task, then bootstrap original test tasks. Receivers and seeds are not counted as new independent benchmark items. Intervals are pointwise exploratory percentile intervals, without multiplicity correction.

Channel diagnostics report the frozen HISTORY decision for each receiver/peer. False inversion is the fraction of observed honest-peer channels classified INVERT, excluding receiver self channels. Adversarial false trust and inversion rates use observed adversarial-peer channels. These channel counts are descriptive and dependent across variants and seeds.

The completed standard run found the partner ceiling at one in all 180 receiver/seed calibrations, suppressing useful inversion. On clean MMLU at full visibility, pooled calibration reduced honest-channel false inversion from 10/270 to 2/270, while mean test accuracy across corruption/visibility worlds fell from 0.857 to 0.672. At ARC f=0.7 and full visibility, pooled-minus-legacy accuracy was −0.987037 with a pointwise paired interval [−1.000000, −0.970370]. A small positive MATH-500 cell also occurred; it is exploratory and not multiplicity-adjusted. These observations demonstrate a calibration/utility tradeoff in this experiment, not a uniform improvement. See the [measured overview](../../results/extensions/calibration/REPORT.md) and [full standard report](../../results/extensions/calibration/standard/REPORT.md).

## Reproduction

From the repository root, with the offline dependencies installed:

```bash
python -m pytest tests/test_extension_calibration.py
python scripts/extension_calibration.py --profile smoke --out results/extensions/calibration/smoke_new --resamples 100 --calibration-resamples 80
python scripts/extension_calibration.py --profile standard --out results/extensions/calibration/standard_new
```

The standard command acquires `/tmp/aip-dgx-v2.lock` internally and spawns a monitored child pinned to CPUs 0 and 1, nice level 10, with one numerical thread and no visible GPU. Defaults are 600 seconds and a 2 GiB sampled-RSS limit. Do not acquire the same lock externally around this command. Existing output directories are rejected. The hidden `--worker` mode is an internal controller implementation detail, not the reproducibility entry point.

The controller records wall time, sampled CPU time, peak RSS, affinity, priority, exit status, timeout/memory stop reason, and host. Worker manifests include package versions, source/input SHA-256 checksums, and checksums for completed outputs. The experiment uses the actual connected ARM host; it does not imply fresh GPU inference merely because the host provides accelerator hardware.

| Artifact | Contents |
|---|---|
| `config.json`, `splits.json` | Protocol settings and exact four-way task partitions. |
| `calibration_thresholds.csv` | Legacy and independent thresholds, bootstrap intervals, support, fallback status, and CAL ID hash per receiver/seed. |
| `calibration_inputs.jsonl.gz` | All sanitized CAL receiver observations needed to independently recompute thresholds. |
| `per_task.parquet` | Paired task scores plus individual receiver predictions/correctness. |
| `channel_diagnostics.parquet`, `channel_error_rates.csv` | HISTORY gate decisions and honest/adversarial decision error rates. |
| `summary.csv`, `paired_comparisons.csv` | World-level results and paired test differences against legacy AIP. |
| `worlds.json`, `traces.jsonl.gz` | Assignments, representative raw/public observations, and method predictions. |
| `threshold_transfer.*`, `accuracy_transfer.*` | Figures generated from saved results. |
| `run_manifest.json`, `resource_receipt.json` | Source/input/output provenance and measured execution limits. |

Tests cover deterministic disjoint splitting, calibration invariance to altered TEST responses and labels, binary ceiling, task-cluster/partner calculations, missing observations, empty/duplicate/forbidden calibration IDs, metadata stripping, and immutable legacy constants. The committed [test log](../../results/extensions/calibration/tests.log) records the executed checks.

A separate bounded smoke process also reproduced the eight-world scientific outputs exactly: 13 comparisons covered tables, receiver arrays, JSON traces, and figure pixels, with matching source/input hashes. Only execution and file-container metadata were excluded. See the [scientific equivalence receipt](../../results/extensions/calibration/scientific_equivalence.json); this does not substitute for a second complete standard study.

## Scope limits

The legacy calibration may overlap all four current partitions and may use another model population. Only the new calibration is task-separated in this extension. Small CAL samples and repeated cached models can make maximum-partner calibration nearly degenerate. Ten agent votes are not ten independent model samples. Threshold replacement also changes the inherited minimum-support and fallback calculations; it is not a test of a threshold comparison in isolation.

All results concern one fixed cache collection and coherent symbolic attacks over final QA answers. They do not establish algorithmic novelty, uniform improvement, robustness to arbitrary adaptive attacks, or action safety in interactive multi-agent workflows.
