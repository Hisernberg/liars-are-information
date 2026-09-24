# Final integration review: six-contribution DGX artifact

Date: 2026-09-12, Asia/Seoul. This record covers the local scientific package prepared for the versioned v2 Hugging Face release. Remote upload/readback verification is recorded separately after publication so that the scientific manifest does not recursively depend on its own upload receipt.

## Outcome and preserved scope

The release contains the original project, all v1 scientific artifacts and three implemented additions. The original five manuscript contribution bullets map explicitly into six maintained themes. No historical data table or paper result was regenerated and silently substituted. All 660 entries in the prior release's manifest are preserved byte-for-byte: the old README and release identity are archived under `provenance/v1/`, and the scientific files remain at their original relative paths. The previous manifest itself is also archived unchanged.

The new standard experiments completed 432 worlds: calibration 144, clones 180, and online 108. They add 306,000 task/step-method records, with different splits and online warmup included. Together with the retained 144-world v1 standard study, the maintained artifact covers 576 simulation worlds. World and row counts are not independent benchmark sample sizes.

## Software and execution validation

The combined suite passed **548 tests**, with zero failures, errors or skips: 516 retained tests plus 13 calibration, 10 clone and 9 online tests. The initial README check mistook a backticked mathematical fraction for a file path; the README formatting was corrected and the full suite passed. No scientific source or result changed for that formatting repair. See `results/extensions/pytest.xml`, `pytest.log` and `validation.json`.

Critical Ruff checks (E9 and F) cover the new extension implementations, experiment runners, tests, review scripts, verification script and additional-copy analysis. The extension owners also recorded targeted style checks. Historical source formatting is not presented as newly rewritten or exhaustively lint-clean.

All three standard studies ran on host `aitopatom-bb03`, an aarch64 DGX/GB10 system, using cores 0 and 1, nice 10, one native numerical thread and no GPU inference. The shared lock serialized scientific jobs. Calibration took 66.12 seconds at 486.93 MiB sampled peak RSS; clones took 99.96 seconds at 536.28 MiB; online took 207.94 seconds at 423.42 MiB process high-water RSS. Each study stayed within its recorded 600-second and 2 GiB controls. Sampled RSS and address-space caps are identified separately in the receipts. Standard runtime sums to 374.02 seconds; development, tests, audits, smoke runs and upload are excluded.

## Independent checks of results

| Study | Independent result reconciliation | What was not inferred |
|---|---|---|
| C4 calibration | 67 source/input/output hashes; 144 worlds; all 90,000 rows and 562,500 receiver correctness values; 1,440 summaries; all 192 paired intervals; 288 traces with 9,000 independently weighted/decoded predictions; 1,080 scorer-wrong Byzantine trace broadcasts; 9,000 clean CAL observations; all 180 ceiling records | Did not independently refit every channel or externally adjudicate mathematical correctness. |
| C5 clones | 52 hashes; all 86,400 rows; 1,080 summaries; all 180 paired intervals; 540 traces with 19,440 independent prediction reconstructions; 54,450 group-cap checks; 4,320 scorer-wrong Byzantine broadcasts; invariant base answers/visibility, fixed clone source and original-only scoring receivers | Saved gate estimates were reused for decoding; no claim of full-estimator clone invariance or authenticated ancestry. |
| C5 extra-copy analysis | Independently reproduced all 48 four-score difference-in-differences cells and paired task intervals; read-only check matches published numbers | Post-standard analysis is exploratory, not an externally preregistered primary endpoint. |
| C6 online | 54 declared hashes; all 26,352 fits with exact past/window support and cadence; 21,600 traces; 518,400 prediction scores against 129,600 rows; 263,520 channel rows; 345,600 fitted-state references; 99,360 scorer-wrong active Byzantine reports; 8,640 identical dynamic/control current-broadcast pairs and 51,840 control rows | Did not independently refit channel coefficients or reconstruct C6 bootstrap intervals; source review inspected the paired block algorithm. |

Runnable evidence: `scripts/review_calibration_results.py`, `scripts/review_clone_results.py`, `scripts/analyze_clone_amplification.py`, and `scripts/review_online_results.py`. Receipts are in the corresponding extension result directories. The C4/C5 reconstructions implement weighted decoding separately from the extension aggregation classes, using the saved channel coefficients. All scorer checks inherit the repository's approximate math/numeric equivalence assumptions.

Separate engineering agents reviewed the clone and online methods. They checked local-only grouping, hidden metadata, exact-copy pairing, actual fractions, past-only fitting, fixed scoring receivers, same-task controls and uncertainty units. Minor input validation, output overwrite protection, source-freeze checks and portable CPU-affinity selection were resolved before each applicable canonical source was frozen. See `results/extensions/clone/independent_review.md`, `results/extensions/online/independent_review.md` and the [multi-agent ledger](MULTI_AGENT_V2.md).

Fresh smoke-process comparisons are recorded separately in each extension directory. Scientific tables, predictions and decompressed traces are compared while timing, compressed-container metadata and PDF timestamps are treated separately. These checks are not second full standard runs. Initial pilots remain identifiable; canonical results have complete manifests and receipts.

## Claim review

- **Calibration:** cleaner task separation is implemented, but conservative ceilings often suppress useful inversion. Every partner ceiling saturated at one. Honest false-inversion reduction is not interchangeable with answer-accuracy improvement.
- **Clones:** caps reduce some added-copy harm, but important baseline benefits already occur with zero copies. The additional-copy analysis subtracts that baseline advantage. Honest-copy harms and partial-visibility fragmentation are reported alongside favorable conditions.
- **Online:** rolling history helps MMLU sleepers and can severely harm the coherent-to-independent regime. The same-task controls account for current task/broadcast differences. Their comparison remains a simulated history intervention, not a natural-deployment recovery estimate.
- **Earlier work:** corrected sampler/covariance laws and the conditional testing statement remain authoritative over historical stronger claims. Original manuscript numbers, confidence studies and old q-axis tables remain historical, with their limitations preserved. No new inference or literature-wide novelty claim is made.
- **Uncertainty:** static analyses average receivers and fixed seeds inside tasks; temporal analysis resamples seeds and shared blocks inside regimes. Pointwise exploratory intervals do not establish multiplicity-controlled discoveries, untouched-data generalization or arbitrary-adversary robustness.

## Package and publication checks

`scripts/verify_research_release.py` uses the standard library to verify all manifest file sizes/hashes, retained v1 content, completed canonical study identities and declared source/input/output hashes. It also checks packaged text for credential-like strings without printing any matched value. The complete file manifest excludes itself; the file count in `RELEASE.json` includes it. Reproduction-generated environments and caches are not part of the released scientific file set.

The delivery uses an additive `releases/dgx-research-20260912-v2` bucket prefix. The prior dated release and all original non-index objects remain. The bucket landing README is updated to the full detailed account with links into the new prefix. Post-upload checks compare every expected remote file/size, read back representative content for SHA-256 verification, and compare previous remote objects to the pre-upload inventory. The external upload receipt is the evidence for those publication checks; this local review does not substitute for it.
