# Pre-camera-ready checklist

Deferred work that is optional for correctness but would tighten the artifact.
Nothing here blocks a submission; each item names what it would change and what
it costs.

## Open

### C-1. Re-download the four evicted checkpoints to clean the floor by construction

**Status:** deferred, 2026-09-20. Not scheduled — the GPU is committed to MX.

The measurement floor excludes 97 T=0.7 rows that are still cap-pinned after X3.
52 of them belong to `gemma4_31b`, `granite42_30b`, `olmo3_32b_think` and
`qwen38_27b`, whose checkpoints were evicted before `data/cache_t07` could be
repaired; the other 45 belong to models that were resident but whose rows need
more than the 3072-token budget.

Re-downloading the four (~225 GB, roughly one hour, GPU mostly idle) and
repairing their 52 rows would make the floor clean **by construction** rather
than **by exclusion**. The exclusion is sound without it — the contaminant is
identified, counted, pinned by a test, and its direction is measured per
resample class — so this is tidiness, not correctness.

It would not repair the remaining 45, which are a genuine budget limit rather
than an eviction accident.

### C-2. Answer-level drift is unmeasured for `qwen38_27b`

`qwen38_27b` fails the byte-identity control (19 of 40 rows) and was evicted
before the answer-level gate existed, so whether its drift reaches the extracted
answer is unknown. `olmo3_32b_think` changed the answer once in 37 drifted rows;
`llama32_3b` changed it more often. Measuring qwen needs the same download as
C-1 and should be done in the same pass if C-1 is ever taken.
