# RACE development log (what changed, when, and why)

This file records every design decision taken **after** looking at evaluation
data, so a reader can judge the risk of overfitting. The rule followed: a
change is admissible only if it (a) fixes a mechanism that can be stated in
one sentence without reference to a benchmark number, (b) keeps every
previously-fixed hyper-parameter unchanged, and (c) the superseded variant stays
in every results table as an ablation.

Fixed a priori and never changed: `anchor_mean=0.75`, `anchor_strength=8`,
`anchor_margin=0.05`, `peer_strength=2` (Beta prior centred at chance),
`clone_threshold=0.95`, `clone_min_support=8`, `eps=1e-3`, one-coin model,
receiver-initialised EM, tie-break toward the receiver's own answer.

| # | Change | Trigger (data seen) | Mechanism | Kept as ablation |
|---|---|---|---|---|
| 1 | Initial RACE: anchored one-coin EM + clone tempering by **raw agreement ≥ 0.95** | — | — | `race_rawclone` |
| 2 | Tried multi-start MAP (receiver init + plurality init, keep higher log-posterior) | MATH-500, f=0.1: a near-chance receiver (llama32_3b, 44%) is captured by a rushing liar that re-uses the receiver's own wrong answers | Did **not** help: with a near-chance anchor the two labellings are likelihood-equivalent (binary-like open tasks), and when replicated channels inflate the likelihood of the adversarial labelling, likelihood selection picks it (0% on MMLU/BoolQ at f=0.7 without tempering). **Rejected as default.** | `race_ms` |
| 3 | Clone detection = **same-source links** (raw agreement ≥ 0.99, applied before any fit) ∪ **dependent-error links** (agreement ≥ 0.95 on tasks where at least one member is wrong w.r.t. the stage-1 truth estimate) | GSM8K, f=0.7, uninformative / partial liars: RACE 64.8% vs 97.2% without tempering | Raw agreement between two *distinct* ≥95%-accurate models is ≥0.9 whatever their dependence, so accuracy was mistaken for duplication and honest evidence collapsed. Conditioning on errors measures conditional dependence (what the independence assumption actually ignores). Exact-identity links are kept and applied first because a receiver's own replica would otherwise "confirm" the receiver. | `race_rawclone`, `race_noclone` |

After change 3 the method was **frozen**. Two evaluation sets had not been
inspected at freeze time and serve as confirmation sets:

1. the replayed **real-LLM deceivers** (`results/llm`, four deception prompts ×
   two attacker models, six benchmarks);
2. the **live multi-agent LLM swarm** (`results/live`, new inference with six
   small open models; generated after the freeze).

All studies were re-run from scratch on the frozen code; `run_manifest.json`
in each study directory records the code hash.

## Known limitation found during development (not fixed, reported)

**Weak-anchor capture.** RACE's identification rests on the receiver being
better than chance on the *effective* answer space. On open-answer tasks where
the candidate set is typically {consensus, one alternative}, a receiver near
50% is effectively at chance, and a rushing attacker who lies with the
receiver's own wrong answers produces a labelling that is observationally
equivalent to the truth from that receiver's point of view. Strong receivers in
the same swarm are unaffected (92.5% vs 71.9% mean in the diagnostic world).
This is an identifiability limit, not an optimisation failure; the paper
quantifies it as a function of receiver competence.

## Post-freeze extension: RACE-D (negative result)

RACE-D (`RACEAggregator(condition="disagreement")`, method key `race_d`) fits
the channels only on history questions where the heard answers disagree,
falling back to all questions if fewer than five qualify. It targets the
camouflage attacker, which earns trust on unanimous questions. It was
specified and implemented before study E9 (`results/ext`) was run, as a
separate method; the frozen RACE code path is unchanged (`condition="all"`).

E9 outcome (324 per-benchmark cells, Holm-corrected, test and bootstrap
interval must agree): 14 wins, 31 losses. It never differs from RACE on the 14
attack settings other than camouflage, echo and sleeper. Against camouflage it
helps at f = 0.7 (+4.9 points) but hurts at f = 0.3 (−3.8); against echo it
costs about 5 points. Dropping the unanimous questions removes exactly the
evidence that pins down the honest peers' accuracies. RACE stays the method.

## Verdict rule tightened (no effect on E1–E3)

`lai.stats.compare` now requires the Holm-corrected Wilcoxon test *and* the
paired bootstrap interval of the mean to agree before calling a win or loss.
In the per-benchmark cells of E1–E3 the two criteria never disagree (checked
on `results/tables/paired_comparisons.csv`); they can disagree when a cell
pools heterogeneous benchmarks (E9's pooled comparison), where the signed-rank
test measures location rather than the mean.
