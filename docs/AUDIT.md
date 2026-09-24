# Audit of the existing "Liars Are Information" work (as of 2026-09-24)

Three public sources were analysed. Each finding below was verified by running
code or diffing files; nothing is taken from prose alone.

| Source | What it is | Date |
|---|---|---|
| `Dhruv1000/Liars_Are_Information` (HF **dataset**) | Dhruv Jyoti Das's upstream artifact: caches, results, claims ledger, 13-page IEEE manuscript, **X3 truncation repair**, **MX** pipeline | 2026-09-23 |
| `Dhruv1000/Liars_Are_Information` (HF **model**) | Same artifact + `data/benchmarks/`; **gated (manual)** — only the card is readable, byte-identical to the dataset card | 2026-09-23 |
| `Nabidnur/Liars_Are_Information-bucket` (HF **bucket**) | A derivative release of the Sep-11 upstream with code fixes and three extensions (clean calibration, clone caps, causal online) | 2026-09-12 |
| `Dhruv1000/adversarial-negotiation-env` (HF **Space**) | OpenEnv REST environment with scripted liars | 2026-04-13 |

The other 13 Dhruv1000 repositories (AgriFM/PASTIS, CET-ViT, TSRDA,
PhenoProto-SSL, Nemotron LoRA, TinyLlama fine-tune, a Gemma-4 API Space, a
Qwen3 assistant Space) are unrelated to this line of work; see the inventory at
the end.

## What we verified as sound

* **The cached answers are internally consistent.** 9 models × 6 benchmarks,
  13,500 greedy rows: zero duplicate task rows, identical gold across models,
  task ids aligned. After the X3 repair our loader reproduces Dhruv's published
  post-X3 single-model table to the digit (e.g. MATH-500: gemma 94.0, qwen 88.5,
  ministral 80.5, llama 44.5).
* **The test suite runs.** 548 tests pass on the bucket release; after merging
  the MX module and the repaired caches, all functional tests pass (the three
  artifact-integrity test files that check the historical README/manuscript are
  preserved under `provenance/tests_legacy_artifact/`).
* **The qualitative AIP story reproduces.** Inversion works against a coherent
  bloc on 4-way questions; AIP is 0% on binary BoolQ at f ≥ 0.5 (the ceiling is
  1.0 by construction, so the gate cannot fire); the gate-aware attacker defeats
  it; majority and label-free Dawid–Skene collapse at f ≥ 1/2.

## Defects found

### D1 — The X3 recompute is incomplete (upstream, Sep 23)
The upstream README states every number is post-repair. Byte comparison of the
result parquets shows `n_scaling.parquet`, `minimax_regret.parquet`,
`accuracy_landscape.parquet` and `gate_aware_mitigations.parquet` are
**byte-identical** before and after X3, and all 54,432 frozen-roster rows of
`sweep.parquet` are value-identical. The sweep driver resumed and skipped every
existing cell (`phase_c.resume cells_done=3888` in `results/x3_recompute.log`),
`phase_n_scaling.py` is not in `x3_recompute.sh`, and the manuscript reads a
gate-aware file the recompute never rewrites (the recomputed
`gate_aware.parquet` gives 0.087 / 0.889 where the paper prints 0.082 / 0.878).
Headline macros 0.645, 0.001, 0.082, 0.280, 0.796 are therefore pre-repair.
**Our v3 evaluation re-derives everything from the repaired cache.**

### D2 — MATH-500 results in the bucket release rest on truncated generations
The bucket (Sep 12) predates X3: 632/1,800 MATH-500 rows were *manufactured*
answers (extractor fallback on truncated completions). Several MATH-500
conclusions there (e.g. "receiver-only 58.6% beats AIP") do not survive
(repaired single-model accuracies rise by up to +0.50).

### D3 — Upstream AIP code lacks four fixes made in the bucket fork
Confirmed by diffing `aip.py`: (a) with p_obs < 1, `fit` compacts each peer's
history independently and pairs answers from *different tasks*; (b) the M2
randomised-threshold jitter is seeded with Python's salted `hash()` — not
reproducible across processes, contradicting the "SHA-256 seeds" claim;
(c) `fit` does not clear state between refits; (d) missing answers count as
disagreements. **v3 uses the fixed code with the repaired data.**

### D4 — MX same-model replicas are byte-identical clones
`scripts/mx_phase1.py` passes one seed to a single `SamplingParams` for the
batch; the sample index only enters the cache key. In all 138 (stage, model,
prompt) groups the "replicas" are byte-identical, so P1/P2 replicas equal the
anchor in 60/60 rows and the negative control and determinism gate pass
trivially. MX is paused at Phase 1 (14/17 gates) and no liar code path ever ran.

### D5 — llama32_3b regenerations are not deterministic
Upstream measured 52/300 (17.3%, Wilson CI [0.135, 0.220]) answer flips on
regeneration of naturally finished rows. The manuscript does not disclose this.
Any cell that mixes regenerated and original llama rows mixes two samples.

### D6 — Documentation drift
The README mixes pre- and post-X3 values; "113 of 231 numbers changed" misreads
the diff table (113 is the *unchanged* count); page count is stated as both 12
and 13; test counts as 377, 527 and 530; "three cells moved beyond the floor" but
the re-audit table shows four (olmo3 +0.120).

### D7 — The negotiation Space cannot host a deceptive-agent experiment
All three tasks are solved by trivial heuristics (1.000 / 1.000 / 0.964 mean
grade over 200 seeds); the liars are scripted and uncoordinated; the
"episode already done" branch raises a `ValidationError`; `uvicorn server:app`
from the README does not resolve. Only the OpenEnv/Docker shell is reusable.

### D8 — Scientific limits acknowledged by the authors that v3 addresses
1. **Binary collapse**: AIP 0% on BoolQ at f ≥ 0.5 (coherence is degenerate at C=2).
2. **Gate-aware defeat**: Proposition 1 — no coherence-measurable rule separates
   sub-chance coordination from honest disagreement.
3. **Dawid–Skene beat AIP** at f = 0.5 on MedQA/MMLU in the upstream sweep, but
   collapses past f = 1/2 (label switching).
4. **No live multi-agent inference** in any release after Phase 4.

## What v3 changes in response

| Finding | v3 response |
|---|---|
| D1, D2 | All results re-derived from the repaired cache; every study re-run from scratch with code hashes in `run_manifest.json` |
| D3 | Built on the fixed AIP code; RACE uses SHA-256 seeds only; bootstrap seeds are salt-free |
| D4 | Not reused; the live experiment uses *distinct* models and explicit role prompts, and reports compliance |
| D8.1–3 | RACE (receiver-anchored channel estimation) — theory in `docs/THEORY.md`, results in the paper |
| D8.4 | Live six-model CPU swarm with LLM-written deception and a debate-contagion condition |

## Inventory of Dhruv1000 repositories (relevance to this project)

| Repo | Type | Relevance |
|---|---|---|
| Liars_Are_Information | dataset / model (gated) | **core** |
| adversarial-negotiation-env | Space | medium (theme only; see D7) |
| autotrain-advanced ("GhostShip" Gemma-4 API), Qwen3 ("AIPA") | Space | low (utility) |
| tinyllama-finetuned, nemotron-submission (model + empty dataset) | model | none |
| AgriFM-PASTIS, AgriFM_PASTIS, Swin-PASTIS, PhenoProto-SSL, TSRDA, PASTIS-FOLD-5 | model / dataset | none (crop remote sensing) |
| cet-vit-cifar100, cet-vit-v4-cifar100, cet-vit-source | model / dataset | none (vision; note the same self-correcting reporting style) |
