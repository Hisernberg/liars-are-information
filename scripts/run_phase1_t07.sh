#!/usr/bin/env bash
# Phase 1, second pass: a temperature-0.7 resample of GSM8K, sample_index=1.
#
# MANDATORY, not optional. Phase 2's within-model error correlation phi is
# estimated from two independent samples of the SAME model on the SAME tasks;
# with only the greedy pass there is no second sample and the within-model half
# of the correlation table cannot be computed at all.
#
# ALL SEVEN MODELS, reasoning ones included. A reasoning model's answer is the
# tail of a sampled chain of thought, so resampling it varies the whole reasoning
# PATH and only then the answer -- a different quantity from answer-level
# resampling. That difference is not a contaminant to be excluded; it is the
# measurement. Path-level resampling is exactly what produced the finding that
# temperature is a weak decorrelator for reasoning models (phi4-mini agreed with
# itself 96% of the time at T=0.7), and dropping these models would silently
# delete that comparison from Phase 2.
#
# The two quantities are kept apart in the data instead: every row carries
# resample_kind, "path_level" for the reasoning tier and "answer_level"
# otherwise, so Phase 2 analyses them separately rather than pooling them.
#
# BUDGET CAP: 45 minutes per model, checked against that model's MEASURED
# greedy GSM8K wall time rather than a fresh projection. Temperature does not
# change the token budget, so the T=0 cell time is the honest estimate for the
# T=0.7 cell, and it costs nothing to obtain. Over cap: skipped and reported,
# never silently truncated.
#
# --no-self-report: the self-reported confidence is elicited once, in the greedy
# pass. Re-eliciting it here would produce a second, differently-sampled value
# for the same task and make "the" self-report ambiguous downstream.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PATH="$PWD/.venv/bin:$PATH"   # ninja, for Qwen's GDN JIT (see run_phase1.sh)
LOG=results/phase1_t07_run.log
CAP_SECONDS=2700
mkdir -p results

# DISK. The greedy pass evicts the large checkpoints after their cells, so this
# pass re-downloads them -- and the roster is 283 GB of weights against ~200 GB
# of disk. Re-downloading without reclaiming filled the filesystem and killed the
# engine with "No space left on device" mid-run.
#
# So before each model, evict any checkpoint this pass has finished with, except
# the three kept deliberately: llama32_3b and ministral3_14b are Phase 4's
# adversarial generators and phi4_mini_reasoning is small enough that re-fetching
# it costs more than the space it holds.
KEEP_ALWAYS="unsloth/Llama-3.2-3B-Instruct mistralai/Ministral-3-14B-Instruct-2512-BF16 microsoft/Phi-4-mini-reasoning"
MIN_FREE_GB=70

free_gb() { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

hf_id() { $PY - "$1" <<'EOF'
import sys, yaml
print(yaml.safe_load(open("configs/models.yaml"))["models"][sys.argv[1]]["hf_id"])
EOF
}

reclaim_for() {
  # $1 = hf_id of the model about to run; keep it and the always-keeps.
  local wanted="$1"
  [ "$(free_gb)" -ge "$MIN_FREE_GB" ] && return 0
  echo "=== $(date -Is) disk $(free_gb)G < ${MIN_FREE_GB}G, reclaiming ===" | tee -a "$LOG"
  $PY scripts/hf_evict.py --keep-only $KEEP_ALWAYS "$wanted" >>"$LOG" 2>&1
  echo "=== $(date -Is) disk now $(free_gb)G ===" | tee -a "$LOG"
}

MODELS=("llama32_3b" "ministral3_14b" "phi4_mini_reasoning" "granite42_30b" \
        "gemma4_31b" "qwen38_27b" "olmo3_32b_think")

for m in "${MODELS[@]}"; do
  if ! $PY -c "
import sys,yaml
r=yaml.safe_load(open('configs/models.yaml'))['models']
sys.exit(0 if r.get('$m',{}).get('enabled') else 1)"; then
    echo "=== $(date -Is)  $m not enabled -- skipped ===" | tee -a "$LOG"; continue
  fi

  est=$($PY - "$m" <<'EOF'
import json, sys, pathlib
m = sys.argv[1]
p = pathlib.Path(f"results/phase_a_full_{m}_report.json")
if not p.exists():
    print(-1); raise SystemExit
cells = [c for c in json.load(p.open())["cells"] if c["benchmark"] == "gsm8k"]
print(round(cells[0]["wall_seconds"], 1) if cells and cells[0].get("wall_seconds") else -1)
EOF
)
  if [ "${est%%.*}" -lt 0 ] 2>/dev/null; then
    echo "=== $(date -Is)  $m has no measured greedy GSM8K cell -- skipped ===" | tee -a "$LOG"; continue
  fi
  if [ "${est%%.*}" -gt "$CAP_SECONDS" ]; then
    echo "=== $(date -Is)  $m OVER CAP: measured greedy GSM8K ${est}s > ${CAP_SECONDS}s -- skipped, reported not truncated ===" | tee -a "$LOG"
    continue
  fi

  # Resume without paying for it. phase_a_cache skips cached (task_id, model)
  # cells, but only AFTER loading the model -- and the large checkpoints have
  # been evicted by now, so a naive re-run would re-download 60 GB to discover
  # there is nothing to do. Check the artifact on disk first.
  if [ -f "data/cache_t07/gsm8k/$m.parquet" ]; then
    echo "=== $(date -Is)  $m T=0.7 already cached -- skipped ===" | tee -a "$LOG"
    continue
  fi

  hf="$(hf_id "$m")"
  reclaim_for "$hf"
  echo "=== $(date -Is)  $m  T=0.7 SECOND PASS (measured greedy GSM8K ${est}s, cap ${CAP_SECONDS}s, disk $(free_gb)G) ===" | tee -a "$LOG"
  $PY scripts/phase_a_cache.py --benchmarks gsm8k --models "$m" \
      --temperature 0.7 --sample-index 1 --cache-root data/cache_t07 \
      --no-self-report --tag "t07_$m" >>"$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "T=0.7 PASS FAILED for $m (exit $rc) -- STOPPING." | tee -a "$LOG"; exit $rc
  fi
done
echo "=== $(date -Is)  T=0.7 SECOND PASS COMPLETE ===" | tee -a "$LOG"
