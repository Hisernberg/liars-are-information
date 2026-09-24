# X3 re-audit: `data/cache`

`pre-x3-cache` -> working tree.

| | count |
|---|---|
| rows | 13500 |
| pinned before | 1036 |
| pinned after | 436 |
| manufactured before | 944 |
| manufactured after | 356 |

Cap-pinned is generation hygiene; manufactured is the defect. A row that
emitted its answer marker and then kept rambling past it is pinned but
not manufactured.

|                                    |   n_after |   pinned_before |   pinned_after |   manufactured_before |   manufactured_after |   manuf_delta |   accuracy_before |   accuracy_after |   acc_delta |
|:-----------------------------------|----------:|----------------:|---------------:|----------------------:|---------------------:|--------------:|------------------:|-----------------:|------------:|
| ('math500', 'ministral3_14b')      |       200 |             136 |             28 |                   135 |                   27 |          -108 |             0.305 |            0.805 |       0.5   |
| ('math500', 'gemma4_31b')          |       200 |              74 |              3 |                    74 |                    3 |           -71 |             0.615 |            0.94  |       0.325 |
| ('math500', 'granite42_30b')       |       200 |             100 |             47 |                    99 |                   42 |           -57 |             0.485 |            0.745 |       0.26  |
| ('math500', 'olmo2_7b')            |       200 |              62 |              4 |                    60 |                    4 |           -56 |             0.295 |            0.315 |       0.02  |
| ('math500', 'llama32_3b')          |       200 |              51 |             16 |                    51 |                   16 |           -35 |             0.425 |            0.445 |       0.02  |
| ('math500', 'olmo3_32b_think')     |       200 |              83 |             52 |                    68 |                   39 |           -29 |             0.645 |            0.765 |       0.12  |
| ('math500', 'ministral_8b')        |       200 |              36 |              8 |                    33 |                    8 |           -25 |             0.515 |            0.55  |       0.035 |
| ('medqa', 'phi4_mini_reasoning')   |       200 |              60 |             34 |                    54 |                   29 |           -25 |             0.58  |            0.595 |       0.015 |
| ('medqa', 'olmo3_32b_think')       |       200 |              31 |             10 |                    31 |                   10 |           -21 |             0.725 |            0.78  |       0.055 |
| ('math500', 'qwen38_27b')          |       200 |              36 |             18 |                    36 |                   18 |           -18 |             0.8   |            0.885 |       0.085 |
| ('gsm8k', 'olmo3_32b_think')       |       500 |              24 |              8 |                    23 |                    6 |           -17 |             0.95  |            0.964 |       0.014 |
| ('math500', 'phi4_mini_reasoning') |       200 |              89 |             64 |                    76 |                   60 |           -16 |             0.61  |            0.69  |       0.08  |
| ('gsm8k', 'phi4_mini_reasoning')   |       500 |              73 |             60 |                    31 |                   19 |           -12 |             0.918 |            0.93  |       0.012 |
| ('mmlu', 'phi4_mini_reasoning')    |       200 |              36 |             25 |                    36 |                   24 |           -12 |             0.745 |            0.775 |       0.03  |
| ('gsm8k', 'granite42_30b')         |       500 |              18 |              7 |                    18 |                    7 |           -11 |             0.934 |            0.952 |       0.018 |
| ('medqa', 'granite42_30b')         |       200 |              22 |             12 |                    22 |                   12 |           -10 |             0.655 |            0.675 |       0.02  |
| ('arc', 'phi4_mini_reasoning')     |       200 |              16 |              7 |                    15 |                    6 |            -9 |             0.87  |            0.895 |       0.025 |
| ('mmlu', 'granite42_30b')          |       200 |              10 |              2 |                    10 |                    2 |            -8 |             0.82  |            0.84  |       0.02  |
| ('gsm8k', 'qwen38_27b')            |       500 |              12 |              6 |                    12 |                    5 |            -7 |             0.966 |            0.968 |       0.002 |
| ('mmlu', 'olmo3_32b_think')        |       200 |              12 |              6 |                    12 |                    6 |            -6 |             0.87  |            0.9   |       0.03  |
| ('gsm8k', 'gemma4_31b')            |       500 |               6 |              0 |                     5 |                    0 |            -5 |             0.972 |            0.978 |       0.006 |
| ('medqa', 'qwen38_27b')            |       200 |               6 |              2 |                     6 |                    1 |            -5 |             0.92  |            0.93  |       0.01  |
| ('gsm8k', 'ministral3_14b')        |       500 |               6 |              2 |                     5 |                    1 |            -4 |             0.96  |            0.96  |       0     |
| ('medqa', 'gemma4_31b')            |       200 |               3 |              0 |                     3 |                    0 |            -3 |             0.925 |            0.925 |       0     |
| ('mmlu', 'qwen38_27b')             |       200 |               3 |              0 |                     3 |                    0 |            -3 |             0.895 |            0.905 |       0.01  |
| ('medqa', 'ministral3_14b')        |       200 |               4 |              1 |                     4 |                    1 |            -3 |             0.785 |            0.79  |       0.005 |
| ('arc', 'olmo3_32b_think')         |       200 |               3 |              1 |                     3 |                    1 |            -2 |             0.955 |            0.96  |       0.005 |
| ('mmlu', 'ministral3_14b')         |       200 |               3 |              1 |                     2 |                    0 |            -2 |             0.84  |            0.845 |       0.005 |
| ('boolq', 'granite42_30b')         |       200 |               2 |              0 |                     2 |                    0 |            -2 |             0.915 |            0.92  |       0.005 |
| ('arc', 'granite42_30b')           |       200 |               2 |              0 |                     2 |                    0 |            -2 |             0.955 |            0.96  |       0.005 |
| ('gsm8k', 'ministral_8b')          |       500 |               3 |              2 |                     3 |                    2 |            -1 |             0.896 |            0.896 |       0     |
| ('gsm8k', 'llama32_3b')            |       500 |               1 |              0 |                     1 |                    0 |            -1 |             0.802 |            0.802 |       0     |
| ('mmlu', 'ministral_8b')           |       200 |               3 |              2 |                     3 |                    2 |            -1 |             0.67  |            0.675 |       0.005 |
| ('mmlu', 'llama32_3b')             |       200 |               4 |              3 |                     4 |                    3 |            -1 |             0.615 |            0.615 |       0     |
| ('gsm8k', 'olmo2_7b')              |       500 |               1 |              1 |                     1 |                    1 |             0 |             0.846 |            0.846 |       0     |
| ('boolq', 'qwen38_27b')            |       200 |               1 |              0 |                     0 |                    0 |             0 |             0.915 |            0.915 |       0     |
| ('arc', 'ministral3_14b')          |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.94  |            0.94  |       0     |
| ('arc', 'llama32_3b')              |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.82  |            0.82  |       0     |
| ('arc', 'olmo2_7b')                |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.79  |            0.79  |       0     |
| ('boolq', 'llama32_3b')            |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.83  |            0.83  |       0     |
| ('arc', 'ministral_8b')            |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.85  |            0.85  |       0     |
| ('arc', 'qwen38_27b')              |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.98  |            0.98  |       0     |
| ('boolq', 'gemma4_31b')            |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.905 |            0.905 |       0     |
| ('boolq', 'olmo3_32b_think')       |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.905 |            0.905 |       0     |
| ('boolq', 'ministral3_14b')        |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.905 |            0.905 |       0     |
| ('boolq', 'ministral_8b')          |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.855 |            0.855 |       0     |
| ('boolq', 'olmo2_7b')              |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.82  |            0.82  |       0     |
| ('arc', 'gemma4_31b')              |       200 |               0 |              0 |                     0 |                    0 |             0 |             1     |            1     |       0     |
| ('boolq', 'phi4_mini_reasoning')   |       200 |               4 |              4 |                     1 |                    1 |             0 |             0.83  |            0.83  |       0     |
| ('medqa', 'ministral_8b')          |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.535 |            0.535 |       0     |
| ('medqa', 'olmo2_7b')              |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.45  |            0.45  |       0     |
| ('medqa', 'llama32_3b')            |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.54  |            0.54  |       0     |
| ('mmlu', 'gemma4_31b')             |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.925 |            0.925 |       0     |
| ('mmlu', 'olmo2_7b')               |       200 |               0 |              0 |                     0 |                    0 |             0 |             0.605 |            0.605 |       0     |

## Measurement floor: clean floor adopted

97 T=0.7 rows remain cap-pinned after X3 and are excluded from `resample_noise`
(phi4_mini_reasoning 44, olmo3_32b_think 24, granite42_30b 13, qwen38_27b 11,
gemma4_31b 4, ministral3_14b 1 — all GSM8K). Adopted floor: **0.066** GSM8K,
**0.099** elsewhere, from 0.068 / 0.100.

A cap-pinned row is wrong in both passes, so it registers as agreement and
suppresses `sigma_resample`. The bias is signed by class: `answer_level`
0.0645 → 0.0628 (keeping was conservative), `path_level` 0.0642 → **0.0664**
(keeping was anti-conservative). Truncation concentrates in `path_level`, whose
two models are both reasoning models.

No verdict flips. See `results/claims.md` for the full re-check.
