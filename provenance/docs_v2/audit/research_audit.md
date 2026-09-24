# Scientific contribution and evidence audit

Date: 2026-09-12. Scope: the supplied diagram report and downloaded source, including broadcast sampling, the gate-aware phase, AIP fitting, calibration, and mathematical notes. The original `source/` snapshot was not changed. Corrections listed below were made only in the isolated `work/` copy. This report separates inherited claims, verified implementation findings, and proposed extensions.

## Evidence boundary

The supplied `Liars_Are_Information_AIP_Detailed_Diagram_Report_2026.pdf` is a secondary review of `Dhruv1000/Liars_Are_Information`, not a reproduction. Its pages 1 and 5 explicitly state that gated source, cache, result, and manuscript bodies were inaccessible. Its favorable scores, novelty judgments, test counts, accuracy values, and hardware statements cannot certify this bucket. The current task targets `Nabidnur/Liars_Are_Information-bucket`; provenance linking these two artifacts must be retained and checked.

The downloaded `source/README.md` contains source-artifact claims including H100 execution, complete phases, 54,432 symbolic rows, 26,208 adversarial rows, and 9 cached models. These are inherited claims until the corresponding files, run manifests, and selected outputs are verified. This audit did not independently reproduce the source paper's benchmark accuracies. Its analytical and sampler verification results are documented below; new benchmark replay is a separate workstream.

## Verified corrections and validation

| Source evidence | Finding | Correction delivered in `work/` |
|---|---|---|
| `src/aip/swarm/broadcast.py::_wrong_answer` and `build_broadcasts` | Shared and independent attack branches draw from the same wrong-answer pool, so mixed-branch collisions were omitted by the historical formula. | Sampler unchanged. Docstrings corrected; `scripts/phase_gate_aware.py::realised_q` now returns `p²+(1-p²)/K`. |
| `scripts/phase_gate_aware.py` historical `q_realised` assignment | The old q axis was a formula output, not measured gate coherence. At C=4,p=0.4 it was 0.28; the actual sampler's population pair expectation is 0.44. | New outputs separate expected and empirical Byzantine-pair agreement, report pair counts and open-pool sizes, and tag statistic scope/version. Historical results and manuscript untouched. |
| `docs/proposition1_note.md` | Equal mean q and exchangeability were treated as sufficient for identical observation laws and binomial collision counts. Neither implication holds generally. | Note now states only the conditional total-variation testing bound on the complete observable distribution. It separates channel classification, abstention, and final answer accuracy. |
| `docs/attenuation_law.md` | One error-covariance contribution was canceled incorrectly; the approximation-bias prose also had the wrong sign. | Exact identity is `m=(1-e_i)(1-e_j)+e_i e_j q+(1+q)Cov(E_i,E_j)`. The independent-correctness binary approximation underestimates the signed-quality product when q<1. |
| `src/aip/aggregation/aip.py::fit`, `_coherence_fields`, `_honest_agreement_reference` | The native gate pools task history, selects partner evidence, defaults to a binomial rule, and uses an empirical agreement-gap reference. It is not the report's simple per-question threshold-only rule. | Corrected mathematical notes describe that scope; attack-pair expectations are not equated to receiver-gate statistics. Runtime fit fixes are owned by the root implementation workstream. |

Validation: `tests/test_coherence_formula.py` passed **42 tests in 0.67 seconds**, locally under `/home/urad/liars_information_20260912/.venv/bin/python` with two BLAS/OpenMP threads. The tests exhaustively enumerate shared/independent branch probabilities, exercise the actual unchanged sampler by Monte Carlo, cover closed binary/multiclass and open deduplicated/fallback pools, and reject partial-observation input to complete-pool metadata. This validates the formula and metadata path; it is not a DGX benchmark-accuracy reproduction.

Helper interface: `realised_q(p, n_options, *, n_wrong_candidates=None)`. Closed spaces supply `n_options>=2`; open spaces require a positive integer `n_wrong_candidates`, including one for the synthetic fallback. Missing open-pool information raises `ValueError`. The helper name is retained for compatibility but returns a theoretical expectation. `pairwise_q_metadata` additionally computes an empirical all-Byzantine-pair estimate on complete broadcasts. `q_realised` remains an explicitly tagged compatibility alias of `q_expected_pairwise` in new native phase output.

## Three contributions that can be made precise

The PDF gives six suggested paper contributions (page 21), rather than exactly three established contributions. The following is a proposed consolidation, not a claim that three new discoveries were made during this session.

| Contribution | Narrow scientific statement | Evidence necessary | Suitable low-resource deliverable |
|---|---|---|---|
| C1: Receiver-local negative-evidence aggregation | A receiver-conditioned TRUST/DISCARD/INVERT rule can exploit identifiable structured adversarial responses under stated calibration and answer-space assumptions. | Inspect actual gate and inversion implementation; show matched AIP versus trust-only/discard ablations on identical broadcasts; include self-only, majority, a relevant confusion-matrix model, and an explicitly labeled oracle control. Report false inversion and clean-task cost. | Cached replay for a few benchmarks, fractions, and compositions, plus tiny synthetic controls with known channels. |
| C2: Adaptive failure and observability | A gate-aware attacker can reduce the usefulness of coherence-based detection in particular measured regimes. A general non-identifiability theorem requires stronger assumptions and a proof about observable distributions. | Inspect attack sampler and exact statistic; select attack parameters on development tasks, freeze them, and evaluate on disjoint tasks. Report gate actions, uncertainty, attack accuracy, and honest overlap. Separate empirical counterexample from impossibility theorem. | Held-out attack sweep, hard/soft-gate comparison, and an attack geometry figure with explicit support and conditioning. |
| C3: Reproducible multi-agent evaluation and diversity analysis | A cache-first evaluation exposes how answer-space structure, error dependence, model composition, and resource use affect this aggregation rule. | Verified cache/task manifests, matched populations, task-level outputs and intervals, method/runtime provenance, deterministic rerun, and a limitations ledger. Swarm members must be actual available cached model outputs or clearly labeled synthetic agents. | Small replay bundle, model correlation/diversity tables, paired-bootstrap results, DGX CPU resource log, and an artifact contribution ledger. |

C3 is an empirical and reproducibility contribution. It should not be presented as a new defense algorithm unless a separate implemented method and supporting evidence exist. A small simulation validates a pipeline and probes mechanisms; it does not reproduce the original large-model benchmark claims.

## Positioning corrections

1. **Do not claim that all prior work discards liars.** The downloaded README includes Dawid–Skene in such a list at line 62. Dawid–Skene jointly estimates latent labels and observer error parameters; true labels need not be available. A class-conditional confusion matrix can represent structured anti-expertise. Whether its orientation is identifiable is a separate issue. Distinguish AIP by its receiver-local decision rule, specific calibration/statistics, and deployment assumptions. Do not claim that the distinction is simply absence of gold labels. [Original Dawid–Skene article](https://academic.oup.com/jrsssc/article/28/1/20/6953573).
2. **Do not frame adaptive disguise as unprecedented.** Earlier adversarial crowdsourcing work explicitly optimizes damage while preserving apparent worker reliability against Dawid–Skene aggregation. The current attack's specific gate, observables, and decentralized setting may differ; novelty requires a focused comparison. [Miao et al., Attack under Disguise](https://cse.buffalo.edu/~lusu/papers/WWW2018.pdf).
3. **Keep algorithmic scope narrow.** Cached final-answer pooling does not establish resilience for tool calls, interactive debate, long-horizon actions, or Byzantine equivocation. A receiver's trusted identity is not a guarantee its answer is correct. State whether all recipients see the same broadcast.
4. **Credit upstream work.** A derived bucket should preserve upstream authorship, license, and citation; describe this session's additions as audits, reproducibility work, or new evaluated experiments. Three agent workstreams are not three scientific discoveries.

Suggested compact thesis: “We evaluate receiver-local negative-evidence pooling when adversarial answer channels are statistically distinguishable, characterize an adaptive failure regime, and provide reproducible cached analyses of the role of agent dependence.” The verb “introduce” should be reserved for authors of the original method or a separately identified new method.

## Important validity checks

### Calibration population and stale documentation

`source/configs/inversion_thresholds.yaml` distinguishes gold-conditioned wrong-answer agreement from the operative receiver-conditioned agreement. This is essential: `P(same | both wrong)` is not generally `P(same | both dissent from receiver)`.

The same file contains historical provenance that needs explicit labeling:

- Top-level `n_models: 4` names `llama32_3b`, `ministral_8b`, `olmo2_7b`, and `phi4_mini_reasoning`, while the README describes a seven-model frozen roster. Transfer to a changed population requires evidence or a declared limitation.
- Operative receiver ceilings are 0.677, 0.8, and 0.198; older gold-conditioned prose still refers to ceilings 0.589, 0.416, and 0.042. Historical figures should not be silently interpreted as operative thresholds.
- Binary provenance says that no benchmark is binary; the same file maps BoolQ to the binary class. This is stale explanatory text.
- A bootstrap upper confidence bound on mean coherence is not a guarantee that all future honest channels lie beneath it. Measure false-inversion frequency on independent clean data.

These are verified documentation/provenance concerns. They do not establish that runtime code uses incorrect constants. Preserve original operational values for faithful replay and keep any recalibrated experiment separate.

### Coherence law and chance reference

The README and PDF state `q(p) = p² + (1-p)²/(C-1)` and discuss a value below `1/(C-1)`. Source inspection confirms that this is not the implemented sampler's formula. Conditional on the shared lie, one wrong category has probability `p+(1-p)/K`, while every other wrong category has probability `(1-p)/K`. Summing squared probabilities gives `p²+(1-p²)/K`, with K=C-1 for closed spaces. This expectation is monotone in p and has no below-chance interior minimum.

If two draws are independent from a common distribution over only `C-1` incorrect labels, their population collision probability is at least `1/(C-1)`. That bound is not automatically a lower bound on an empirical finite-pair estimate, on correlated coordinated agents, or on a differently conditioned statistic. For example, balanced counts 2/2/1 across five agents and three categories yield finite-pair agreement `(2+2+0)/(5*4)=0.2`.

The historical 0.280 was a theoretical formula value, not a sampled estimate. It therefore cannot be rescued by the finite-pair caveat; the old q axis is mislabelled. The sampler always emits incorrect labels in gate-aware mode. The original p settings and accuracy values remain historical records and require independent replay to validate; changing the formula does not recompute or certify them. For open tasks, actual candidate pools are finite and may be singleton fallbacks, so claiming q=p² is also incorrect in general.

To establish a threshold impossibility claim, define the admissible honest/adversarial distributions and the complete statistic available to the decision rule. Construct two admissible worlds with matching observable distributions and conflicting correct actions, or prove an overlap/error lower bound. One optimized attack point below a reference rate demonstrates a limitation of a selected gate, not a universal impossibility theorem.

### Statistical design

- Resampled compositions, seeds, receivers, and sweep rows sharing task answers are dependent. Use original tasks as the principal sampling unit. Keep seed/composition effects paired within each task; use hierarchical uncertainty if generalizing across compositions.
- Separate threshold calibration, attacker selection, and final testing by task identity. Reusing labels during calibration is legitimate when declared; testing on those same items does not establish held-out performance.
- Freeze selected attack parameters before final evaluation. Show selection curves as exploratory diagnostics and held-out results as validation.
- Pair method outputs at identical task/model/seed/fraction/observation settings. Equal aggregate row counts do not prove a matched comparison.
- The PDF's `sqrt(2) * 1.96 * sigma` is a 95%-style uncertainty scale under specific independence assumptions. It is not a conventional powered MDE without a target power and alternative. Prefer a paired confidence interval and an explicitly named measurement floor.
- Large bootstrap replicate counts do not increase the number of independent benchmark tasks. More resampling of a 200-task slice cannot replace more tasks.
- Below-floor or non-significant differences support “not resolved at this sample size,” not equivalence, invariance, or zero overhead.
- If many cells are scanned, label exploratory results and predefine a small primary claim family. Avoid selecting only positive benchmarks for summaries.

### Baseline fairness

For a confusion-matrix comparison, distinguish unsupervised Dawid–Skene, a supervised calibration-only estimate, and an oracle known-channel bound. These have different information access. A supervised confusion matrix is not an implementation of unsupervised Dawid–Skene. Fit parameters on allowed tasks only. Report unsupervised label-permutation ambiguity and initialization sensitivity; do not use test gold labels to select orientation or restarts.

Response-level majority/filtering methods and iterative LLM evaluators have different budgets. Use genuine native baselines where available. A numeric toy imitation of SAC, CP-WBFT, or other named systems should not be labeled a reproduction of those systems.

## Low-resource simulation and replay plan

Use DGX CPUs for cached and synthetic analyses. A DGX host run is not automatically a GPU run. Do not allocate a GPU merely to prove DGX usage; save hostname, execution time, process resource limits, versions, and observed resource use. Keep GPU inference optional and separately budgeted.

### Minimal independent synthetic control

Use small categorical channels with explicit true labels and agent identities. Begin with `C in {2,4}`, `N=10`, `f in {0,0.3,0.5,0.7}`, competence levels such as 0.6/0.8, independent versus common-mode honest errors, and honest/coherent/mixed adversaries. These settings are proposed controls, not statements about real LLM behavior. A few hundred to a few thousand tasks per fixed cell can be processed in batches with only NumPy-scale CPU memory.

Primary metrics: paired accuracy difference versus trust-only, false inversion of honest channels, false trust of adversarial channels, classwise accuracy, gate decision counts, and per-run runtime. Preserve task-level predictions. Implement one trust/invert sanity fixture with an analytically known answer before interpreting aggregate outputs.

### Cached replay with existing code

Select a bounded common set from existing cache files and use the actual aggregation implementation. Prefer one binary benchmark, one multiple-choice benchmark, and one open-answer benchmark, provided matching cache coverage exists. Start at `N=10`, a few fractions, full observation, and a single topology. Reuse original model outputs; changing seeds alone must not be described as new LLM samples.

Avoid synthetic duplicates of the same cached model being counted as independent models. If larger swarms resample a model or a finite response cache, disclose this and interpret `N` as agents/votes, not independent evidence.

Run the matched ablation first; then a small held-out adaptive sweep; then diversity/correlation summaries at fixed model-count/composition rules. Expand only to address an identified concern. If a cache or dependency is missing, save a verified synthetic-only result and mark native replay as incomplete instead of filling a table from inherited README numbers.

### Resource envelope

An initial operating envelope is 2–4 CPU threads, one process per run, chunked task arrays, and no model loading. Set BLAS/OpenMP thread caps and record actual limits in the run manifest. Choose a modest memory cap after inspecting cache sizes. Save a pilot's elapsed time and peak RSS before choosing a larger sweep; do not promise a runtime without measurements. Restrict new model inference to a separate explicitly identified necessity and budget.

## Saved artifacts and release review

Keep artifacts under a dated extension directory with:

- `config.json`: all settings, task partitions, seeds, calibration/attack-selection protocol, source revision or source-file hashes.
- `task_outputs.parquet` or compressed CSV: task ID, method, composition, attack, seed, prediction, truth, correctness, and gate diagnostics where supported.
- `summary.csv` and `paired_differences.csv`: sample counts, metric definitions, uncertainty method, and denominators.
- `manifest.json`: hardware host, CPU/GPU allocation, package versions, wall time, peak RSS, command, input/output checksums, and completion state.
- Figures generated only from saved tables, with labels distinguishing synthetic simulation, cached replay, and inherited reported results.
- `CONTRIBUTIONS.md`: C1/C2/C3, original authorship versus session additions, evidence paths, agent responsibilities, and status (`verified`, `exploratory`, `not evaluated`, or `blocked`).

For repository readiness, verify that displayed commands actually run from the packaged layout and target the correct Hugging Face object type. A bucket URL is not itself a dataset repository ID for `load_dataset`. Preserve licenses and upstream provenance; do not bundle access credentials. A complete audit-lite release can contain code, tests, configs, small synthetic outputs, and new analyses without copying all large caches.

## Claim wording checklist

Acceptable after a corresponding run: “On this saved synthetic/cached test split, AIP changed accuracy by X relative to the matched trust-only ablation, with interval Y.”

Unsupported without broader evidence: universal Byzantine-majority tolerance; novelty of systematic-error exploitation; robustness of real tool-using agents; invariance to swarm size; zero clean-task cost; source-paper replication from a toy simulator; superiority over an unimplemented baseline; or a formal impossibility conclusion from one numerical sweep.

The next decisive evidence is a matched, independently saved native replay and a held-out gate-aware attack analysis. The audit supports organizing those as three clear contributions while keeping inherited claims and new evidence separate.
