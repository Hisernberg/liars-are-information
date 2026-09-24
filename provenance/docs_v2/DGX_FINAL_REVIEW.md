# Independent final review of the canonical DGX replay

**Status: passed the artifact checks listed below.** Reviewed on 12 September 2026 by `/root/evaluation_review`, independently of the coordinating agent that implemented and executed the replay. Canonical run directory: `results/dgx/standard/`.

The review loaded the completed artifacts, recomputed trace correctness and paired bootstrap intervals, checked attack selection against validation data, and verified input/script hashes. It did not rerun the simulations or infer that a positive scientific result is required for an internally consistent artifact. The canonical run uses the revised MATH-500 scoring and attack-pool protocol; the earlier local exact-string run is separate outer audit evidence.

## Completed-run and coverage checks

| Check | Verified evidence |
|---|---|
| Completion | Both `run_manifest.json` and `resource_receipt.json` report complete; child exit code is 0 with no recorded stop reason. |
| World coverage | 144 distinct worlds: 72 mechanism, 12 diversity, and 60 adaptive worlds. |
| Swarm seeds | Exactly 20260912, 20260913, and 20260914. |
| Benchmarks and task splits | MMLU, BoolQ, and MATH-500 each have 200 task IDs divided into 80 history, 40 validation, and 80 test tasks. No overlap or duplicated IDs within a benchmark's partition. |
| Cache agreement | The seven selected model caches per benchmark agree on task coverage and gold strings, with no duplicate task IDs; the splits cover those exact IDs. |
| Prediction-table coverage | 152,640 distinct `(world, method, split, task)` records. Every method has the complete expected validation/test task set for its world. |
| Scoring consistency | Accuracy values are finite and within [0,1], and each value corresponds to an integer hit count over the recorded honest-receiver count. Closed-label primary and exact-string scores agree. |
| Assignment consistency | Recorded Byzantine count matches f for N=10; model entries are absent exactly for Byzantine identities. Every record's honest-receiver count and world metadata match `worlds.json`. |
| Summary consistency | All 2,544 rows of `summary.csv` reproduce the means and counts obtained from `per_task.parquet`. |

The resource receipt records **75.76 seconds**, **504.20 MiB peak sampled RSS**, affinity to CPU cores **0 and 1**, one native math-library thread, and nice 10. The run stayed below its 600-second deadline and 2,048 MiB monitored RSS limit. Both manifests identify it as cache replay without new GPU/model inference. RSS is sampled; this check confirms the recorded monitored peak, not an independently measured continuous peak.

## Saved-trace and attack checks

The saved `traces.jsonl.gz` contains **432 traces**, exactly three predetermined test tasks per world. Every trace task belongs to the first three IDs of that benchmark's saved test partition, rather than being selected by the observed outcome.

The reviewer checked all trace broadcasts for task/agent identity and world assignment, verified honest answers against their source-model caches, and checked visibility masks for self inclusion and valid peer IDs. Fully observed worlds expose all ten agents to every receiver.

Across **3,816 method/task trace groups**, the reviewer independently re-scored **19,935 honest-receiver predictions** using the repository's `math_match` for MATH-500 and exact labels for MMLU/BoolQ. Their average hit fractions match the corresponding canonical `accuracy` values. Exact-string hit fractions also match `exact_match_accuracy`.

All **1,980 Byzantine broadcasts present in the saved traces** score incorrect under the applicable primary scorer. This check covers the saved examples; it is not a reconstruction of every unlogged task's broadcasts. Source review and protocol tests separately establish that the new open-answer pool is filtered with the same scorer and that its fallback is checked. The original native sampler remains separate legacy behavior.

## Attack selection and uncertainty checks

For all **126 rows** of `selected_attacks.csv`, the selected p equals the minimum validation accuracy among `{0, 0.25, 0.5, 0.75, 1}`, with ascending-p tie breaking. The reviewer recomputed the selected test accuracy and the paired receiver-only/trust-only reference accuracies from the canonical per-task table. Every selection uses the expected 80 test tasks.

All **60,480 rows** of `selected_attack_predictions.parquet` match their source per-task records, include only test tasks, and carry the p selected for their target defender. This file intentionally includes different methods evaluated under each target defender's selected attack; comparisons must retain that target-method identifier.

The reviewer recomputed all **186 paired comparisons**. For each comparison, receiver-averaged scores are averaged across the three fixed swarm seeds within each task; then the 80 task-level AIP-minus-baseline differences are resampled. All saved point estimates and 95% percentile interval endpoints match an independent recomputation using **1,000 seeded bootstrap draws** per comparison.

These intervals are pointwise and exploratory. They are conditional on the current cache, roster, fixed seed set, and historical calibration; they do not treat receivers or seeds as new independent benchmark items, correct multiple comparisons, or prove universal robustness.

## Hash verification

All **23 input hashes** in the run manifest match the current two configuration files and 21 selected cache Parquets. The current runner script matches the hash recorded by the canonical run:

```text
scripts/dgx_low_resource.py
ba115e8f2f38e6e7be1ee9cf503ef49006f9e0ad1ca90c94ced539553a0aeff8
```

The following reviewed artifact hashes bind this review to the saved canonical outputs. Final release packaging should retain these bytes and include its own complete file manifest.

| Artifact under `results/dgx/standard/` | SHA-256 |
|---|---|
| `per_task.parquet` | `7a42a6e40bde090f899d4ee8a0af3a5890546c1d7c661a7d4f0fc2a0572b8a20` |
| `worlds.json` | `6ede345394eba12fdbed160f8a328aeb44f6e46107009394b457da9ca2f64ada` |
| `splits.json` | `f99c9bcf031b58eed6aca2c0d595be238790eb07dcc2320e69ba81058ca7af12` |
| `traces.jsonl.gz` | `8f4b6f0c2c31c359b8b875438cf8135bcfea440ce946abe378c6b020b74ac868` |
| `paired_comparisons.csv` | `0f8d50f3c827089a78528b7a01006bf5919c3393e7a44156a72f79fce639e814` |
| `selected_attacks.csv` | `6275cce24f6f5afdf93bf478474a3d1d6c7081612d0dd5b240d79c50a56d39c3` |
| `selected_attack_predictions.parquet` | `dd8ec3bd701460a67e27964b754a12e751a3299d53d789de7e75c716eb07ed45` |
| `summary.csv` | `fb6739c4929121bd2497f00f64b3b0757b9e1de6d47e3afece1268ff834cd34e` |
| `run_manifest.json` | `e7b76ebc0b1610f2da28b1d520369b01cb51838670db2f8dcbbe13d6ecde9b93` |
| `resource_receipt.json` | `b09ec801e8cc1ba735f4f7c45f12c35c7a2175c5fa2a72aef7dee0325c3aa978` |

## Remaining scientific and validation limits

- Historical coherence-threshold calibration can overlap the new test tasks and comes from a different model population. New split isolation does not remove that historical dependence.
- Every adaptive p candidate receives its own stationary p-specific history and fitted defender. Validation/test share the same seeded randomized-defense realization. This is not a common fixed defender exposed to a later online attack, nor a test of secret randomness.
- `math_match` is an approximate normalization/symbolic scorer. Re-scoring agrees with that implementation, not with an independently adjudicated mathematical oracle. Exact-string sensitivity uses the revised predictions and attack pool; it does not recreate the historical Phase C experiment.
- Honest outputs are greedy cache replays; identical model replicas remain dependent. The full Dawid–Skene baseline assumes conditional independence and stationary worker channels, and can encounter label non-identifiability or EM local optima. It is not applied to open-answer MATH-500.
- The replay evaluates static AIP gates, three benchmarks, a complete communication graph with observation thinning, and a finite symbolic attack grid. It does not validate the historical windowed path, live LLM attacks, conversational trajectories, or tool actions.
- Per-task-table integrity and bootstrap calculations were checked exhaustively. Correctness was reconstructed from saved prediction labels only for the three retained trace tasks per world; the complete set of individual receiver predictions was not saved.
- The runtime manifest directly hashes the runner and specified inputs. Final packaging must fingerprint the remaining source and dependencies as well. New hashes cannot prove historical byte identity where original cache sidecars lacked content hashes.

The three release contribution descriptions and the engineering-agent responsibilities remain in `docs/DGX_CONTRIBUTIONS.md` and `docs/MULTI_AGENT_REVIEW.md`. This review supports the canonical artifact's consistency within the checked scope; scientific claims still need their exact measured comparison and stated limitations.
