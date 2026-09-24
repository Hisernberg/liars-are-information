# Phase 1 gate report

Grid: 7 models x 6 benchmarks = 42 cells, 10500 cached predictions.

Rules applied mechanically: **R1** ceiling at accuracy >= 0.93; **R2** AUC interpreted only at n_wrong >= 30.


## Gates a-f, per cell

| model | tier | benchmark | n | acc | **head** | extract | srep | resample | flag | AUC power |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| gemma4_31b | strong | arc | 200 | 1.00 | **0.00** | 100% | 100% | answer_level | HEADROOM-LOW | underpowered (n_wrong=0) |
| granite42_30b | strong | arc | 200 | 0.95 | **0.04** | 100% | 100% | answer_level | HEADROOM-LOW | underpowered (n_wrong=9) |
| llama32_3b | weak_fast | arc | 200 | 0.82 | **0.18** | 100% | 100% | answer_level | ok | ok |
| ministral3_14b | mid | arc | 200 | 0.94 | **0.06** | 100% | 100% | answer_level | HEADROOM-LOW | underpowered (n_wrong=12) |
| olmo3_32b_think | reasoning | arc | 200 | 0.95 | **0.04** | 100% | 90% | path_level | HEADROOM-LOW | underpowered (n_wrong=9) |
| phi4_mini_reasoning | reasoning | arc | 200 | 0.87 | **0.13** | 99% | 66% | path_level | ok | underpowered (n_wrong=26) |
| qwen38_27b | strong | arc | 200 | 0.98 | **0.02** | 100% | 100% | answer_level | HEADROOM-LOW | underpowered (n_wrong=4) |
| gemma4_31b | strong | boolq | 200 | 0.91 | **0.10** | 100% | 100% | answer_level | ok | underpowered (n_wrong=19) |
| granite42_30b | strong | boolq | 200 | 0.92 | **0.09** | 100% | 100% | answer_level | ok | underpowered (n_wrong=17) |
| llama32_3b | weak_fast | boolq | 200 | 0.83 | **0.17** | 100% | 100% | answer_level | ok | ok |
| ministral3_14b | mid | boolq | 200 | 0.91 | **0.10** | 100% | 100% | answer_level | ok | underpowered (n_wrong=19) |
| olmo3_32b_think | reasoning | boolq | 200 | 0.91 | **0.10** | 100% | 98% | path_level | ok | underpowered (n_wrong=19) |
| phi4_mini_reasoning | reasoning | boolq | 200 | 0.83 | **0.17** | 100% | 92% | path_level | ok | ok |
| qwen38_27b | strong | boolq | 200 | 0.92 | **0.09** | 100% | 100% | answer_level | ok | underpowered (n_wrong=17) |
| gemma4_31b | strong | gsm8k | 500 | 0.97 | **0.03** | 100% | 100% | answer_level | HEADROOM-LOW | underpowered (n_wrong=14) |
| granite42_30b | strong | gsm8k | 500 | 0.93 | **0.07** | 100% | 100% | answer_level | HEADROOM-LOW | ok |
| llama32_3b | weak_fast | gsm8k | 500 | 0.80 | **0.20** | 100% | 100% | answer_level | ok | ok |
| ministral3_14b | mid | gsm8k | 500 | 0.96 | **0.04** | 100% | 100% | answer_level | HEADROOM-LOW | underpowered (n_wrong=20) |
| olmo3_32b_think | reasoning | gsm8k | 500 | 0.95 | **0.05** | 100% | 100% | path_level | HEADROOM-LOW | underpowered (n_wrong=25) |
| phi4_mini_reasoning | reasoning | gsm8k | 500 | 0.92 | **0.08** | 100% | 100% | path_level | ok | ok |
| qwen38_27b | strong | gsm8k | 500 | 0.97 | **0.03** | 100% | 100% | answer_level | HEADROOM-LOW | underpowered (n_wrong=17) |
| gemma4_31b | strong | math500 | 200 | 0.61 | **0.39** | 100% | 100% | answer_level | ok | ok |
| granite42_30b | strong | math500 | 200 | 0.48 | **0.52** | 100% | 100% | answer_level | ok | ok |
| llama32_3b | weak_fast | math500 | 200 | 0.42 | **0.57** | 100% | 100% | answer_level | ok | ok |
| ministral3_14b | mid | math500 | 200 | 0.30 | **0.69** | 100% | 100% | answer_level | ok | ok |
| olmo3_32b_think | reasoning | math500 | 200 | 0.65 | **0.35** | 100% | 100% | path_level | ok | ok |
| phi4_mini_reasoning | reasoning | math500 | 200 | 0.61 | **0.39** | 100% | 100% | path_level | ok | ok |
| qwen38_27b | strong | math500 | 200 | 0.80 | **0.20** | 100% | 100% | answer_level | ok | ok |
| gemma4_31b | strong | medqa | 200 | 0.93 | **0.07** | 100% | 100% | answer_level | ok | underpowered (n_wrong=15) |
| granite42_30b | strong | medqa | 200 | 0.66 | **0.34** | 100% | 100% | answer_level | ok | ok |
| llama32_3b | weak_fast | medqa | 200 | 0.54 | **0.46** | 100% | 100% | answer_level | ok | ok |
| ministral3_14b | mid | medqa | 200 | 0.79 | **0.21** | 100% | 100% | answer_level | ok | ok |
| olmo3_32b_think | reasoning | medqa | 200 | 0.72 | **0.28** | 100% | 98% | path_level | ok | ok |
| phi4_mini_reasoning | reasoning | medqa | 200 | 0.58 | **0.42** | 100% | 95% | path_level | ok | ok |
| qwen38_27b | strong | medqa | 200 | 0.92 | **0.08** | 100% | 98% | answer_level | ok | underpowered (n_wrong=16) |
| gemma4_31b | strong | mmlu | 200 | 0.93 | **0.07** | 100% | 100% | answer_level | ok | underpowered (n_wrong=15) |
| granite42_30b | strong | mmlu | 200 | 0.82 | **0.18** | 100% | 100% | answer_level | ok | ok |
| llama32_3b | weak_fast | mmlu | 200 | 0.61 | **0.39** | 100% | 100% | answer_level | ok | ok |
| ministral3_14b | mid | mmlu | 200 | 0.84 | **0.16** | 100% | 100% | answer_level | ok | ok |
| olmo3_32b_think | reasoning | mmlu | 200 | 0.87 | **0.13** | 100% | 86% | path_level | ok | underpowered (n_wrong=26) |
| phi4_mini_reasoning | reasoning | mmlu | 200 | 0.74 | **0.26** | 100% | 66% | path_level | ok | ok |
| qwen38_27b | strong | mmlu | 200 | 0.90 | **0.10** | 100% | 100% | answer_level | ok | underpowered (n_wrong=21) |

## Gate f: headroom summary (R1)

**10 of 42 cells are HEADROOM-LOW** (accuracy >= 0.93). Kept in cache and appendix tables; excluded from headline figures.

| benchmark | HEADROOM-LOW cells | models |
|---|---:|---|
| arc | 5 | gemma4_31b, granite42_30b, ministral3_14b, olmo3_32b_think, qwen38_27b |
| gsm8k | 5 | gemma4_31b, granite42_30b, ministral3_14b, olmo3_32b_think, qwen38_27b |

**Headline-eligible cells: 32 of 42.**

| benchmark | headline-eligible | median headroom |
|---|---:|---:|
| arc | 2 | 0.04 |
| boolq | 7 | 0.10 |
| gsm8k | 2 | 0.05 |
| math500 | 7 | 0.39 |
| medqa | 7 | 0.28 |
| mmlu | 7 | 0.16 |

## Gate b power (R2)

**20 cells are underpowered for AUC** (n_wrong < 30). No direction-of-effect language may be attached to their confidence AUC.

| cell | n_wrong |
|---|---:|
| gemma4_31b x arc | 0 |
| qwen38_27b x arc | 4 |
| olmo3_32b_think x arc | 9 |
| granite42_30b x arc | 9 |
| ministral3_14b x arc | 12 |
| gemma4_31b x gsm8k | 14 |
| gemma4_31b x medqa | 15 |
| gemma4_31b x mmlu | 15 |
| qwen38_27b x medqa | 16 |
| qwen38_27b x boolq | 17 |
| granite42_30b x boolq | 17 |
| qwen38_27b x gsm8k | 17 |
| ministral3_14b x boolq | 19 |
| gemma4_31b x boolq | 19 |
| olmo3_32b_think x boolq | 19 |
| ministral3_14b x gsm8k | 20 |
| qwen38_27b x mmlu | 21 |
| olmo3_32b_think x gsm8k | 25 |
| olmo3_32b_think x mmlu | 26 |
| phi4_mini_reasoning x arc | 26 |

## Gate a: extraction

All cells at 100% extraction.


## Gate c: self-report parse rate (R3 -- finding, not a defect)

| cell | self-report rate |
|---|---:|
| phi4_mini_reasoning x arc | 66% |
| phi4_mini_reasoning x mmlu | 66% |
| olmo3_32b_think x mmlu | 86% |
| olmo3_32b_think x arc | 90% |
| phi4_mini_reasoning x boolq | 92% |
| phi4_mini_reasoning x medqa | 95% |
| olmo3_32b_think x boolq | 98% |
| olmo3_32b_think x medqa | 98% |
| qwen38_27b x medqa | 98% |

## Gate d: reproducibility

4 rows flagged `nonreproducing` across 4 cells (recorded in configs/experiments/phase_a_cache.yaml).

