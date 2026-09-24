# C6 — Strictly causal online AIP

## Research question and contribution

**RQ6. Can an AIP receiver using only past public broadcasts adapt to sleeping or changing adversaries, and what does adaptation cost in accuracy, history, and computation?**

This extension adds a receiver API with an explicit prediction/commit boundary, four history policies, and a streaming experiment with contemporaneous fixed-regime controls. The question is distinct from whether a gate works after fitting an entire completed stream. A deployment must choose an answer before seeing future tasks, and its current answer cannot become evidence for the weights used to produce that answer.

The original static AIP gate, calibrated thresholds, and decoder remain the mechanism being studied. C6 adds a causal execution contract around that mechanism. It does not invoke the historical `AIPAggregator(window=...)` path, whose per-task and pooled statistics are unsuitable for this causal comparison.

## Falsifiable expectations

1. A fixed initial history can become stale when previously honest channels become coherent liars. A cumulative or rolling history may reduce the resulting post-switch damage, but an improvement is not assumed.
2. Rolling history can forget obsolete channel behavior more quickly, while reducing statistical support for the gate and increasing estimation variability. The 32-task and 64-task windows therefore need not dominate either cumulative history or self.
3. Changes from coherent to independent wrong draws can invalidate learned inversion decisions. This behavior depends on the answer space: for a binary question there is only one wrong label, so the two wrong-answer schedules are behaviorally identical.
4. A task sequence can become easier or harder near a switch. An apparent raw-accuracy recovery is insufficient evidence of adaptation unless the same-task comparisons support that interpretation.

The standard run answers these questions with measured results in [REPORT.md](../../results/extensions/online/REPORT.md), paired estimates in [paired_comparisons.csv](../../results/extensions/online/paired_comparisons.csv), and history controls in [control_summary.csv](../../results/extensions/online/control_summary.csv). Negative or inconclusive comparisons remain part of the result.

## Measured answer

**Past-only rolling history recovered much of the damage from MMLU sleepers, but continual refitting also caused a large failure when coherent liars changed to independent wrong draws.** A useful initial inversion decision can remain valuable even after current coherence disappears. Shortening the memory is therefore a tradeoff, not a general robustness improvement.

These representative predeclared comparisons use all 160 evaluation tasks, including the first 40 tasks before a scheduled switch. Differences are accuracy units; intervals are the exploratory paired seed/task-block intervals described below.

| Benchmark and schedule | f | Method | Method accuracy | Fixed40 accuracy | Difference [95% block interval] |
| --- | ---: | --- | ---: | ---: | ---: |
| MMLU, sleeper | 0.5 | Rolling32 | 0.8375 | 0.3554 | +0.4821 [0.3155, 0.6198] |
| MMLU, sleeper | 0.7 | Rolling32 | 0.8014 | 0.2458 | +0.5556 [0.4195, 0.6633] |
| MMLU, coherent → independent | 0.7 | Rolling32 | 0.3750 | 0.9528 | −0.5778 [−0.6841, −0.4111] |
| MATH-500, sleeper | 0.7 | Rolling64 | 0.3667 | 0.1826 | +0.1840 [0.0477, 0.3453] |
| BoolQ, sleeper | 0.7 | Rolling32 | 0.2375 | 0.2375 | 0.0000 [0.0000, 0.0000] |

There are 54 dynamic-schedule online-method versus Fixed40 contrasts in the full table: 15 intervals lie above zero and 10 below zero. This count is descriptive, with no multiplicity correction. Performance relative to self is a stricter check. MMLU Rolling32 at f = 0.7 under sleepers differs from self by −0.0056 [−0.1941, 0.1216], so the run does not establish an improvement over self. MATH-500 Rolling64 under that sleeper setting remains 0.2340 below self, interval [−0.3473, −0.1144]. Every AIP history variant scores 0.2375 on the BoolQ f = 0.7 sleeper run while self scores 0.8882. More frequent fitting does not solve that binary failure.

The channel records expose the adverse switch mechanism. On MMLU with f = 0.7 and a coherent → independent switch, all originally Byzantine peer channels are inverted in the pre-switch fitted snapshots. In the late online snapshots (evaluation steps 120–159), all are trusted for Cumulative, Rolling32, and Rolling64. Fixed40 retains its initial inversion weights. The attacker still always gives a wrong answer; its independent draws remove the coherence signal used by the refitted gate. Thus fresh history can remove useful negative evidence. This is a failure of the evaluated gate/history combination, not proof that every causal online defense must fail.

The same-task controls also distinguish memory adaptation from task difficulty. For the MMLU f = 0.7 sleeper, Rolling32's first-16-task gap to the corresponding coherent-history control is −0.8681, while its final-16-task gap within the 64-task post-switch horizon is 0.0000. Its descriptive binned gap-closure lag is 48 tasks; Rolling64's is 64 tasks. Cumulative and Fixed40 do not close that gap within the same horizon. These are the ends of nonoverlapping display/test bins, not exact first-recovery timestamps or guarantees of sustained recovery. A zero control gap can also mean two methods are equally poor, as the BoolQ controls demonstrate.

The standard run completed **108 worlds, 129,600 task/method records, and 263,520 channel-state rows in 207.94 seconds**, with **423.42 MiB peak RSS** and no GPU use. Fixed40 performed 432 receiver fits; each online policy performed 8,640, a 20-fold increase. Total measured evaluation-phase method time was 14.97 seconds for Fixed40, 60.08 for Cumulative, 42.44 for Rolling32, and 48.15 for Rolling64. Those timings include state logging and are specific to this instrumented CPU replay; they are not model-inference latency estimates. Canonical output files total about 100.6 MiB excluding the manifest and separate development smoke directories. See [resource_receipt.json](../../results/extensions/online/resource_receipt.json) and [summary.csv](../../results/extensions/online/summary.csv).

## Receiver API and causal boundary

The implementation is [online.py](../../src/aip/extensions/online.py):

```python
policy = CausalAIPReceiver(
    receiver_id=0,
    benchmark="mmlu",
    retention="rolling",
    history_size=32,
    warmup=40,
    refresh_interval=8,
    thresholds=thresholds,
    label_space=["A", "B", "C", "D"],
)
prediction = policy.predict(current_public_observation)
# The caller can publish/score the answer here.
policy.update(current_public_observation)
```

`predict` may refresh the fitted gate from already committed observations, then aggregates the current task with those fixed weights. `update` commits the observation without refitting or revising the prediction. The API rejects commits before prediction, another prediction while a commit is pending, repeated task IDs, wrong receiver IDs, mixed task IDs, duplicate peer IDs, and changes to public content between prediction and commit.

The API accepts no gold-label argument. It strips hidden Byzantine status, source model identity, and attack metadata. Those fields are retained only in separate evaluator traces. Current public answers may affect the current vote, as they should; they cannot affect the current fitted channel statistics.

Before a valid fitted history exists, the predeclared prediction is the receiver's own answer. Missing answers supply no agreement evidence. If the receiver's past answer is missing, that complete task row is excluded from fitting all peer series together. If a peer is missing, the peer remains missing at that task position. Independently compacting peer streams would compare answers to different questions and is prohibited.

`fit_snapshot()` exposes the retained past task IDs, the subset contributing valid receiver rows, a task-list hash, a fit generation, and each channel's action, weight, and support. Thus a prediction can be linked to the exact evidence available before it. The rolling observation buffer is bounded, while the duplicate-task ledger still grows with the number of committed tasks; C6 does not claim constant total memory over an unbounded stream.

## Predeclared comparison

| Method | Fitting history | Refresh rule | Cold start |
| --- | --- | --- | --- |
| Self | None | None | Self |
| Majority | Current public votes | None | Current majority |
| Fixed40 | First 40 committed tasks | Once, before evaluation | Self |
| Cumulative | All committed past tasks | Every 8 committed tasks | Self |
| Rolling32 | Last 32 task positions | Every 8 committed tasks | Self |
| Rolling64 | Last 64 task positions | Every 8 committed tasks | Self |

All AIP variants use self-vote parity 0.1, the inherited binomial gate, unchanged threshold constants, and the same per-receiver static fitting procedure with `window=None`. Rolling32 begins its first evaluation fit with 32 retained tasks; Rolling64 and Cumulative begin with 40. That support difference is explicit in the saved diagnostics. No drift detector, label-based selection, post-hoc window tuning, or self fallback beyond cold start is added.

## Stream and attack design

The standard grid is 3 benchmarks × 2 corrupted fractions × 3 seeds × 6 schedules = **108 worlds**. MMLU, BoolQ, and MATH-500 each provide 40 warmup plus 160 distinct evaluation tasks. The seven `arm=frozen` model caches form the model population. Ten agent slots receive stable model identities before corruption; the corrupted sets are nested between fractions 0.5 and 0.7. A slot that changes behavior retains its original cached model when behaving honestly.

Only the **original honest receivers**, selected before the stream begins, enter the evaluation mean. Latent sleepers are excluded from that scoring population even while they behave honestly. This prevents an apparent accuracy change caused simply by changing which receivers are counted.

| Schedule | Warmup and evaluation 0–39 | Evaluation 40–79 | Evaluation 80–119 | Evaluation 120–159 |
| --- | --- | --- | --- | --- |
| Sleeper | Honest | Coherent wrong | Coherent wrong | Coherent wrong |
| Coherent → independent | Coherent wrong | Independent wrong, p=0 | Independent wrong, p=0 | Independent wrong, p=0 |
| Toggle | Coherent wrong | Independent wrong, p=0 | Coherent wrong | Independent wrong, p=0 |
| Fixed honest control | Honest | Honest | Honest | Honest |
| Fixed coherent control | Coherent wrong | Coherent wrong | Coherent wrong | Coherent wrong |
| Fixed independent control | Independent wrong, p=0 | Independent wrong, p=0 | Independent wrong, p=0 | Independent wrong, p=0 |

Here “independent” describes independent draws from the symbolic wrong-answer pool, not independent agent information. Such draws can coincide. For MATH-500 the pool uses the v1 semantic-valid sampler, which excludes answers matching gold under `math_match`; scoring also uses `math_match`. Exact-string scores are saved as a sensitivity check. Symbolic attackers know gold; receivers never do. These attackers follow a fixed schedule and are not optimizing against a newly learned online policy.

Tasks follow a predeclared deterministic hash ordering, shared across seeds, methods, fractions, and schedules for each benchmark. This is a simulated chronology of static benchmark questions, not a natural deployment time series. The chronology and warmup/evaluation lists are saved and hashed. Every control with the same benchmark, fraction, seed, task, and current regime receives the exact same broadcasts as the corresponding dynamic task.

## Outcomes and inference

The elementary score is the mean accuracy over original honest receivers for one task. The reported regret relative to self is `self accuracy − method accuracy`, so positive regret is harmful. Receiver votes, duplicated model slots, and receiver-pair events are never treated as independent samples.

The outputs include:

- Per-task accuracy, exact-string sensitivity, regret, fit support, and refit count.
- Sixteen-task time windows and eight-task display bins, with switch boundaries marked.
- Paired accuracy differences against self and Fixed40.
- Same-task differences against the corresponding complete fixed-regime control.
- First-16-task post-switch damage and the final-16-task control gap within an observation horizon of at most 64 tasks.
- A descriptive gap-closure lag: after an initial deficit worse than −0.02, the end of the first later nonoverlapping block of up to 16 tasks whose mean control gap is at least −0.02. Zero means no initial deficit; null means no observed closure. Closure can be temporary. Toggle phases can censor the horizon after 40 tasks.

The fixed-regime control is essential. For example, the Sleeper and fixed-coherent runs have identical current broadcasts after the switch, but different histories. Their paired difference then quantifies the consequences of historical exposure on those same tasks. Their raw post-switch accuracies can still move with task difficulty.

Intervals use 200 exploratory hierarchical bootstrap draws. Each draw resamples the three seeds and circular task blocks of length 16 within contiguous regime segments. Task-block choices are shared across seeds and methods, retaining task-sharing dependence. This avoids an IID step/receiver analysis, but it is not a guarantee of coverage: cumulative histories can induce dependence longer than 16 tasks, only three seeds are available, and the model roster and chronology are fixed. There is no multiplicity adjustment across the many pointwise comparisons.

The legacy calibration task IDs overlap current benchmark caches. C6 does not independently recalibrate thresholds or claim independent threshold validation. It also provides no formal regret bound, universal robustness guarantee, independent-generation diversity estimate, or real-world recovery time.

## DGX resource envelope and reproducibility

Run from the repository root:

```bash
../.venv/bin/python scripts/extension_online.py \
  --profile standard --out results/extensions/online --resamples 200

PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_extension_online.py -q
```

The first command is the canonical run command. Reproduction after downloading completed outputs requires a fresh output path, for example `--out results/extensions/online/reproduction`; the runner rejects existing output files. Child directories alone do not count as conflicting outputs.

The runner acquires `/tmp/aip-dgx-v2.lock` with `flock`, then starts a 600-second experiment alarm. It uses the first two available CPU cores (`[0, 1]` on this DGX), nice 10, one numerical/Arrow thread, no visible GPU, and a 2 GiB address-space limit. The address-space cap is stricter than a 2 GiB resident-memory target. No models, services, credentials, or published repositories are changed. The actual resource receipt records runtime, CPU time, peak RSS, and library versions. Full traces and CSVs stream to disk rather than accumulating channel histories from every world in memory.

`run_manifest.json` records the design, source/input/output hashes, completed world counts, causal-boundary assertions, and limitations. Source hashes are captured before execution and verified again afterward; inputs are also checked for changes. Success and failure receipts carry explicit status. `chronology.json` and `worlds.json` pin tasks, model identities, corruption sets, and scored receivers. `complete_traces.jsonl.gz` contains all current broadcasts and predictions; `fits.jsonl.gz` and `channel_states.csv` reconstruct the fitted state behind every prediction. Traces include benchmark answers for auditing, but no question text or credentials.

The targeted tests check predict-before-update, absence of current/future tasks in fits, prefix invariance when future broadcasts or evaluator labels change, hidden identity invariance, missing-peer alignment, fixed/rolling retention, and invalid-observation rejection. These tests establish the execution contract; the measured experiment determines whether a particular history policy improves accuracy.

## Completed verification

All nine C6 tests passed, including the analytical constant-difference block-bootstrap case and the binary coherent/independent behavior check. Ruff passed for the new module, runner, and tests.

An independent reconciliation checked all 26,352 fitted histories and generation references, 21,600 complete traces, 129,600 task/method records, and 263,520 channel-state rows. It also checked 99,360 active adversarial reports against the scorer and 8,640 dynamic/control pairs for identical current-regime broadcasts. The declared hashes passed. That review verifies stored evidence and chronology; it does not independently re-estimate every AIP coefficient or establish statistical coverage. See [result_reconciliation.json](../../results/extensions/online/result_reconciliation.json) and [independent_review.md](../../results/extensions/online/independent_review.md).

A fresh smoke repeat in a separate directory outside the repository completed in 3.61 seconds with 300.58 MiB peak RSS under the same lock and limits. Its 1,872 task/method rows, 312 complete traces, 375 fit records, scientific summaries, and paired intervals exactly matched the finalized smoke run after excluding runtime and timestamp fields. All declared file hashes from both runs were separately verified. The standard run was not repeated. [smoke_equivalence.json](../../results/extensions/online/smoke_equivalence.json) records the exact comparisons and exclusions.
