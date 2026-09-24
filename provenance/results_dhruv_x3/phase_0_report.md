# Phase 0 — integrity, hardware, and model preflight

**Node** `ib-bom-dev-gpu02` · **GPU** NVIDIA H100 NVL, 95830 MiB board / 93.10 GiB usable
· **Driver** 570.195.03 (CUDA 12.8) · **Date** 2026-09-02

Supersedes the L4 regime (`ib-bom-dev-gpu0`, NVIDIA L4 22.49 GiB, driver 550.127.08).
Under RULE 1 every L4-era result is void; nothing below is carried forward from it.

---

## 1. Test suite

| suite | result |
|---|---|
| fast (`pytest`) | **348 passed**, 24 deselected |
| slow (`pytest -m slow`) | **24 passed** — dataset downloads + real-gold round-trip |
| `ruff check src scripts tests` | clean |

**Code failures: none.** **Missing-data failures: none** — the suite is self-contained and
does not read the lost caches. Two failures appeared after the registry rewrite and were
migration artefacts, not code faults; both are fixed (§3).

The lost artefacts (`data/cache*/`, `results/**/*.parquet`, `results/logs/`) are confirmed
absent from the HuggingFace remote as well: the repo holds 193 files and no parquet. There is
no copy of the L4 caches on this machine. Regeneration is the only path, which is what this
plan does.

## 2. Config inventory — survivors

| file | sha256[:16] | status |
|---|---|---|
| `configs/task_lists.json` | `e3674c571c550744` | **intact** |
| `configs/task_lists_gsm8k_ext.json` | `a70035fdde1d3650` | **intact** |
| `configs/inversion_thresholds.yaml` | v2, receiver-conditioned | intact; recalibrated to v3 in Phase 2 |
| `configs/experiments/phase_a_cache.yaml` | — | intact |

**The GSM8K 100 + 400 lists are verified verbatim and reusable:**

* frozen list — 100 ids, 100 unique; ext list — 400 ids, 400 unique
* **overlap 0**, union **500** — exactly the Phase 1 GSM8K target, no reselection needed
* identical seed (`20250825`) and identical `prompt_template_hash` (`f9b756633e80`) across both

MATH-500 (100, 7 subjects) and MMLU (100, stratified 57 subjects) are intact and will be
*extended* to 200 under the same seed in Phase 1, leaving the existing ids untouched.

## 3. Registry feasibility on the H100

`tests/test_registry.py` asserted the shipped registry against a hardcoded `L4_GIB = 22.49`.
That constant did two unrelated jobs, so it was split rather than simply retargeted:

* `SMALL_CARD_GIB = 22.49` — retained for the synthetic fit-arithmetic tests, where a small
  budget is what makes "does a 14B model fit" a meaningful question.
* `H100_GIB = 93.10` — used by the two tests that check the *shipped* registry against the
  *real* device (`test_every_enabled_model_fits_this_h100`, `test_oversized_entries_are_disabled`).

Sequential residency is unchanged and remains the only mode.

## 4. Final roster — 7 models, 7 labs, 5 tiers

Revised on instruction: Ministral 8B → **Ministral 3 14B**, **Granite 4.2-30B** added (IBM,
the seventh lab), Qwen 3.8-27B and OLMo 3-32B-Think confirmed, and the existing Gemma 4,
Phi-4-mini-reasoning and Llama-3.2-3B retained. All seven fit, all pass Hub architecture
agreement, **all ungated** (no `HF_TOKEN` required):

| model | hf_id | tier · lab | params | wt GiB | KV GiB | total | slack | fit |
|---|---|---|---|---|---|---|---|---|
| `qwen38_27b` | Qwen/Qwen3.8-27B | strong · Alibaba | 27.78B | 51.75 | 1.00 | 52.75 | 40.35 | PASS |
| `gemma4_31b` | google/gemma-4-31B-**it** | strong · Google | 31.27B | 58.25 | 3.75 | 62.00 | 31.09 | PASS |
| `granite42_30b` | ibm-granite/granite-4.2-30b | strong · IBM | 29.28B | 54.53 | 1.00 | 55.53 | 37.56 | PASS |
| `olmo3_32b_think` | allenai/Olmo-3-32B-Think | open-data reasoner · AI2 | 32.23B | 60.04 | 1.00 | 61.04 | 32.06 | PASS |
| `ministral3_14b` | mistralai/Ministral-3-14B-Instruct-2512-**BF16** | mid · Mistral | 13.95B | 25.97 | 0.62 | 26.60 | 66.50 | PASS |
| `phi4_mini_reasoning` | microsoft/Phi-4-mini-reasoning | compact reasoner · Microsoft | 3.84B | 7.15 | 0.50 | 7.65 | 85.45 | PASS |
| `llama32_3b` | unsloth/Llama-3.2-3B-Instruct | weak/fast · Meta lineage | 3.21B | 5.98 | 0.44 | 6.42 | 86.67 | PASS |

Disabled and retained on record as superseded: `ministral_8b`, `olmo2_7b`. Disabled as gated:
`llama32_3b_official`.

### 4a. Two checkpoint traps caught before any download

Both would have silently violated a stated rule of the study rather than failing loudly.

1. **Ministral 14B: the headline repo is FP8.**
   `mistralai/Ministral-3-14B-Instruct-2512` reports Hub safetensors
   `{BF16: 1.78B, F8_E4M3: 12.16B}` — an FP8 checkpoint, excluded outright by the bf16-only
   rule, since quantisation shifts the token logprob distribution that is *both* the
   confidence surface the attacks target and the error-correlation structure Phase 2 measures.
   The roster uses **`…-2512-BF16`**, which is 13.945B wholly BF16.

2. **Gemma 4: the headline repo is the base model.**
   `google/gemma-4-31B` ships **no chat template at all** — absent from `tokenizer_config`,
   absent from `processor_config`, and there is no `chat_template.jinja` in the repo
   (HTTP 404). It cannot be prompted in the instruction form every benchmark here uses. The
   roster uses **`google/gemma-4-31B-it`**, which has a template and is, as a bonus, wholly
   BF16 (31.273B) where the base carried a non-bf16 remainder.

### 4b. vLLM architecture support — the pin had to move

`Gemma4ForConditionalGeneration` **does not exist** in the previously pinned `vllm==0.17.0`.
Bisecting vLLM's `model_executor/models/registry.py` by release tag:

| vllm | Gemma4 | Olmo3 | Qwen3_5 | Mistral3 | Granite | torch |
|---|---|---|---|---|---|---|
| 0.17.0 (old pin) | ✗ | ✓ | ✓ | — | — | 2.10.0 |
| 0.18.1 | ✗ | ✓ | ✓ | — | — | 2.10.0 |
| **0.19.1** | **✓** | **✓** | **✓** | **✓** | **✓** | **2.10.0 (cu12 12.8.90)** |
| 0.20.0+ | ✓ | ✓ | ✓ | ✓ | ✓ | 2.11.0 (CUDA 13 — **will not init on a 12.8 driver**) |

`vllm==0.19.1` is the unique release satisfying both constraints: the oldest that loads
Gemma 4, and the newest still resolving `torch==2.10.0` → `nvidia-cuda-runtime-cu12==12.8.90`,
exactly at this driver's cap. Verified in-process:

```
vllm 0.19.1 · torch 2.10.0+cu128 · cuda 12.8 · H100 NVL · capability (9,0) · cuda available True
all 7 roster architectures present in the installed registry (323 total); vLLM raised nothing
```

### 4c. transformers had to move too

`google/gemma-4-31B-it`'s `tokenizer_config` uses a schema (list-valued `extra_special_tokens`,
`backend: "tokenizers"`) that transformers 4.x cannot parse: 4.57.6 falls back to the Gemma-1
`GemmaTokenizerFast` and dies with `AttributeError: 'list' object has no attribute 'keys'`.
vLLM 0.19.1 excludes transformers 5.0–5.4 and 5.5.0, so the usable floor is **5.5.1**; the
pin is now `transformers>=5.5.1` and 5.16.1 is installed. This is a major-version jump, and
the 24 slow tests — which download the real datasets and round-trip every gold answer through
the extractor and normalizer — pass on it.

### 4d. Hybrid attention and multimodality

**Five of seven are hybrid-attention**, which the registry's KV formula does not model: it
charges every layer at full-attention cost. Qwen3.8 is 48 linear + 16 full of 64; Gemma 4 is
50 sliding (window 1024) + 10 full of 60; OLMo 3 is 48 sliding (window 4096) + 16 full of 64.
The formula therefore **overestimates**, so the fit check stays conservative, never optimistic.
OLMo 3's window equals `max_model_len`, so its figure is exact. Every model clears the slack
floor by ≥ 31 GiB under the conservative number.

**Three are multimodal** (Qwen 3.8: vision; Gemma 4: vision + audio; Ministral 3: vision).
Text-only prompts throughout; the towers' weights are already inside each `params_b`.

## 5. Explicit generation parameters — five of seven ship truncating defaults

`GenerationDefaults` had **no `top_k` field at all**, and `inference.py` built `SamplingParams`
without one, so vLLM's default applied silently. Checking every `generation_config.json`:

| model | shipped default | now overridden to |
|---|---|---|
| `gemma4_31b` | temperature 1.0, top_p 0.95, **top_k 64** | 0.0 / 1.0 / -1 |
| `qwen38_27b` | temperature 1.0, top_p 0.95, **top_k 20** | 0.0 / 1.0 / -1 |
| `granite42_30b` | temperature 1.0, top_p 0.95 | 0.0 / 1.0 / -1 |
| `olmo3_32b_think` | temperature 0.6, top_p 0.95 | 0.0 / 1.0 / -1 |
| `llama32_3b` | temperature 0.6, top_p 0.90 | 0.0 / 1.0 / -1 |
| `ministral3_14b` | none | 0.0 / 1.0 / -1 |
| `phi4_mini_reasoning` | none | 0.0 / 1.0 / -1 |

At `temperature=0` vLLM decodes greedily and both nucleus parameters are inert, so the main
cache is unaffected either way. They bind on the **temperature-0.7 GSM8K second pass** — the
run that feeds the within-model φ in Phase 2 — where an inherited `top_k=64` would truncate the
resample and bias the correlation estimate.

### 5a. Thinking modes — four reasoning models, three suppressible

`encode_messages` already passes `enable_thinking=False`; every roster template accepts the
kwarg. Rendered each one to confirm what the model is actually asked to do:

| model | rendered tail | reasons? | budget |
|---|---|---|---|
| `qwen38_27b` | `<think>\n\n</think>` | suppressed | 640 / 16 |
| `granite42_30b` | `<think></think>` | suppressed | 640 / 16 |
| `gemma4_31b` | `<\|channel>thought<channel\|>` | suppressed | 640 / 16 |
| `olmo3_32b_think` | `<think>` **left open** | **not suppressible** | 2048 / 512 |
| `phi4_mini_reasoning` | no thought block | reasons in completion | 2048 / 512 |
| `ministral3_14b`, `llama32_3b` | — | no | 640 / 16 |

This mattered: **Qwen 3.8's template defaults to `reasoning_effort='xhigh'`** when thinking is
left on ("think carefully… validate key assumptions… consider plausible alternatives"), which
at a 640-token cap would truncate mid-thought — the exact failure that cost Phi-4-mini 3/5 of
its GSM8K answers on the L4 run. `enable_thinking=False` closes it. OLMo-3-*-Think is a
dedicated reasoning checkpoint where thinking is the model rather than a mode, so it keeps the
reasoning-tier budget.

**Dry-run watch item:** Gemma 4's template is channel-structured (`<|channel>thought`), unlike
every other entry. The extraction gate is the check that its answers parse.

## 6. Preflight — PASSED

`python scripts/preflight.py --all` exits 0. Fit, architecture agreement (every `arch` block
re-fetched from the Hub and compared: all `yes`), and access (all seven ungated) pass for the
whole enabled roster. Tests: **348 fast + 24 slow pass; ruff clean.**

## 7. BLOCKED — dry-run not executed

Step 6 (5 tasks × GSM8K per new model, with the five gates) **could not run.**

### 7a. The GPU is occupied — stop authorised, but the command is blocked for me

```
PID 2149086  VLLM::EngineCore  87778 MiB / 95830 MiB
  └─ docker container a486234a645f  image vllm/vllm-openai:latest  name qwen3-32b  up 23 h
     vllm serve --model Qwen/Qwen3-32B --host 0.0.0.0 --port 8000
                --gpu-memory-utilization 0.90 --max-model-len 32768
```

Live and listening on `0.0.0.0:8000`; leaves **6.86 GiB** free. Even `llama32_3b` (6.42 GiB)
cannot load, because vLLM sizes its pool as a fraction of *total* memory: `0.15 × 93.10 =
13.97 GiB` requested against 6.86 GiB available. It is unrelated to this study, which cannot
use a served endpoint at all — an OpenAI-compatible API does not expose the answer-span token
logprobs the confidence surface is built on.

**Stopping it was authorised, but `sudo docker stop qwen3-32b` was refused by the sandbox
policy.** The owner must run it:

```
! sudo docker stop qwen3-32b
```

### 7b. Disk is short for the full roster

| | GB |
|---|---|
| weights to download (7 models, dedup'd) | **283.2** |
| free on `/` | **205.7** |

The fix is routine and fits the design: download and **evict per model**. Sequential residency
already generates each model's Phase 1 cells in one residency block, so the four large
checkpoints never need to coexist. Retaining the three small models (42.0 GB — they are Phase 4's
adversarial generators) plus the largest single big model (64.5 GB) peaks at ≈ **107 GB**,
comfortably inside 205.7 GB. To be implemented in Phase 1.

## 8. Changes committed in this phase

* `pyproject.toml` — `vllm` `0.17.0` → `0.19.1`; `transformers` floor `>=4.44` → `>=5.5.1`
* `configs/models.yaml` — rewritten for the H100 and the 7-model / 7-lab roster
* `src/aip/models/registry.py` — explicit `top_k` on `GenerationDefaults`
* `src/aip/models/inference.py` — `top_k` wired into `SamplingParams`
* `tests/test_registry.py` — `L4_GIB` split into `SMALL_CARD_GIB` / `H100_GIB`
* `results/phase_0_report.md` — this file

## 9. Open item for the owner

**`HF_TOKEN`** is absent and **not needed** — the entire enabled roster is ungated. Only
`llama32_3b_official` (disabled) would want one.
