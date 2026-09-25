# Pre-registration: E10, a fresh live confirmation of RACE v3.1 and RACE-informed debate

**Registered:** 2026-09-25, before any E10 answer was generated. This file is committed together with the frozen code; the commit hash and its timestamp on GitHub are the registration record.

**Method under test:** RACE v3.1 exactly as in `src/lai/race.py` at the commit that adds this file. It uses class-conditional channels on binary questions and one-coin channels otherwise. Every hyper-parameter is unchanged from v3.0 (paper, Appendix D). Until the analysis below is complete, the only changes allowed are bug fixes that do not alter any method's output; each would be logged in `docs/RACE_NOTES.md`.

## Why this study exists

RACE v3.1's binary rule was adopted after the live swarm E8 showed that small models answer yes/no questions with an option bias. E8 therefore cannot confirm v3.1. E10 is a new live run whose answers did not exist when the rule was adopted.

## Data (generated after this registration)

- **Agents:** the six E8 models (Qwen2.5-1.5B, SmolLM2-1.7B, Granite-3.3-2B, OLMo-2-1B, Llama-3.2-1B, Gemma-3-1B), same decoding (arg-max over option letters, float32, CPU).
- **Questions:**
  - **ARC-Challenge items 0–119.** A four-option benchmark never used in the live swarm.
  - **BoolQ items 120–199.** Binary questions never shown to any live model; E8 used items 0–119.
  - The replay studies used *other* models' cached answers to these items. The questions are not new to the project, but every live answer is.
- **Roles.** The prompts for honest, solo saboteur, rushing saboteur and debate are identical to E8. One role is new: **informed debate**.
  - An informed debater sees the same 12 round-1 votes as a plain debater, anonymised and shuffled.
  - Each vote carries the panelist's reliability ("reliable", "no better than chance", or "usually wrong: its answer is probably incorrect").
  - The reliabilities come from the debater's own RACE fit on the panel's round-1 answers to *earlier* questions only.
  - A final line gives the option favoured by the reliability-weighted evidence.
  - No gold answer is ever used. The first 8 questions show "no track record yet".
- **Output:** `data/live_cache_v2/`.

## Hypotheses

- **H1 (binary rule, confirmation).** On fresh BoolQ answers against LLM saboteurs (solo, rushing, colluding; f ∈ {0.1, 0.3, 0.5, 0.7}; 5 seeds), mean honest-agent accuracy is ordered RACE v3.1 > RACE v3.0. It is also RACE v3.1 ≥ receiver alone.
- **H2 (four options, confirmation).** On ARC against LLM saboteurs at f ∈ {0.5, 0.7}, RACE v3.1 is at least as accurate as majority vote and as the receiver alone.
- **H3 (RACE-informed debate).** Honest agents' *individual* post-debate accuracy is higher with RACE-informed prompts than with plain debate prompts. The test is paired over (model, question) and Wilcoxon two-sided.
- **H4 (exploratory).** Pooled accuracy (majority vote, RACE) over informed-debate answers, compared with independent and plain-debate answers.

## Analysis plan

- **Harness and statistics.** We use the same harness and statistics as E8: 10 identities from the six-model roster, a HISTORY/VALIDATION/TEST split of 40/20/40, 5 seeds, and paired bootstrap intervals with Holm-corrected Wilcoxon tests. A verdict needs both the test and the interval to agree.
- **Reporting.** Every outcome is reported whatever its direction, in the paper (E10), the README and `results/live2/`.
- **Decision rule.**
  - A hypothesis is *supported* if its stated ordering holds in the mean and no cell shows a significant reversal.
  - It is *contradicted* if the ordering is reversed in the mean.
  - Otherwise it is *inconclusive*.
