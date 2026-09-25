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

## Confirmation set 2 (live swarm, E8): an observation, not a change

The live six-model swarm (`results/live/`) was evaluated with the frozen code.
On MMLU, RACE lifts every honest receiver (+5.6 to +27.9 points). On binary
BoolQ with LLM saboteurs it falls below the receiver alone (56.2% vs 60.1% at
f = 0.5). The per-receiver breakdown (`results/live/receivers.csv`) shows that
the strongest receiver loses most, which rules out weak-anchor capture. The
answer-bias table (`results/live/answer_bias.csv`) shows the cause: small
models answer binary questions with a fixed option bias (SmolLM2 says "yes"
89% of the time as an honest agent; Granite's "lies" are "no" 98% of the time).
A symmetric one-coin channel cannot represent that asymmetry. The full-confusion
ablation (`race_full`, frozen before the run) recovers 69.5%, and it is not
worse than RACE on the replayed BoolQ worlds. With four options and 48 history
questions it is over-parameterised (45.7% on MMLU).

A defensible amendment is class-conditional channels for binary answer spaces
and the one-coin model otherwise. Because this was found on a confirmation set,
it is reported as a post-hoc observation, and the frozen method is unchanged.

## v3.1: class-conditional channels on binary questions (adopted after E8)

Decision: `RACEAggregator(model="auto")` is now the default. Binary label
spaces get class-conditional (2x2 confusion) channels with the same anchor:
a Dirichlet prior on the receiver's diagonal (mean 0.75, strength 8), and each
diagonal entry kept >= 0.55. Everything else uses the one-coin model, as in
v3.0. The rule was adopted on the owner's decision after the live-BoolQ
observation above. v3.0 is kept in every study as `race_onecoin`, and every
study was re-run from scratch.

Consequences we state openly:
- The live BoolQ result for v3.1 is not a confirmation. It is the observation
  that motivated the rule.
- The replayed BoolQ cells of E1–E3 were already known when the rule was
  adopted, so they cannot confirm it either. They only show that it does no harm
  there: the macro `binaryReplayNonTies` counts the cells with a significant
  v3.1–v3.0 difference.
- A fresh confirmation set is needed before the rule is claimed as validated.
- The known-channel oracle now uses the same channel family (class-conditional
  on binary questions), so it stays a fair ceiling for v3.1.

## E11: classical crowdsourcing baselines (no change to RACE)

Added IWMV (Li and Yu, 2014), MACE (Hovy et al., 2013), GLAD (Whitehill et al.,
2009) and KOS (Karger, Oh and Shah, 2014; binary only) in `src/lai/crowd.py`.
They run under exactly RACE's protocol: per honest receiver, fitted on its own
HISTORY window, and ties go to the receiver's own answer. One bug was found in
our MACE M-step (the spam responsibility was mis-normalised) and fixed before
the study ran; `tests/test_crowd.py` covers each estimator on toy swarms with a
known answer. RACE itself is unchanged (v3.1). Result: the classical estimators
match RACE with a liar minority and collapse with a liar majority, which is
Proposition 2 in action; RACE loses no paired comparison to them.

## E10: pre-registered fresh live run (no change to RACE)

Registered in `docs/PREREGISTRATION_E10.md` (commit 494aa28) before any answer
existed. The evaluation code (`experiments/eval_live2.py`) implements the
registered decision rule and was written while the run was in progress, before
any E10 answer was inspected. RACE v3.1 is evaluated exactly as frozen.

Outcome (analysis run once, exactly as registered; `results/live2/hypotheses.csv`):
all 5 of 5 hypotheses supported.
- H1a v3.1 > v3.0 on unseen BoolQ: 81.9% vs 69.1% (7 wins, 0 losses in 12 cells).
- H1b v3.1 >= receiver alone: 81.9% vs 74.0%.
- H2a/H2b ARC, f >= 0.5: RACE 58.2% vs majority 41.0% and alone 49.7%.
- H3 informed vs plain debate, individual accuracy: 68.2% vs 61.7%
  (+6.5 points, 95% CI [+3.9, +9.1]).
Exploratory (not registered, reported as such): informed debaters adopt RACE's
suggested option 87.6% of the time (84.5% when it is wrong), and are
less accurate (68.5%) than the suggestion itself (74.4%), so the gain is
deference, not deliberation. Two draft sentences were corrected against the data
before publication: clone tempering does not merge a saboteur with its honest
twin (0 of 36 pairs; their answers agree 16-81% of the time), and the effect of
debate on pooling reversed between ARC and BoolQ.
