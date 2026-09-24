#!/usr/bin/env bash
# Phase 1 driver: per model, gate then cache, then reclaim the disk.
#
# The dry-run gate is folded into each model's residency block rather than run
# as a separate pass over the whole roster. That is not a shortcut past the
# gate -- every model still faces the same five gates on 5 GSM8K tasks before a
# single full cell is written, and a red gate stops the run. It is a disk
# decision: the roster is 283 GB of weights against ~200 GB of free space, so a
# roster-wide dry run followed by a roster-wide cache run would download every
# large checkpoint twice.
#
# Ordering is cheapest-first so a harness fault surfaces on a 6 GB model rather
# than a 64 GB one.
#
# Optional argument: a model name to resume from, skipping everything before it.
# The prediction cache is resumable per (task_id, model) cell, so re-running a
# completed model is harmless -- but it still costs a full weight load, and on a
# 32B checkpoint that is minutes of nothing.
set -uo pipefail
cd "$(dirname "$0")/.."
START_FROM="${1:-}"
PY=.venv/bin/python
export HF_HUB_DISABLE_PROGRESS_BARS=1

# The venv's bin/ must be on PATH, not just its python. Qwen 3.8's Gated Delta
# Net layers JIT-compile a kernel and shell out to `ninja`; invoking
# .venv/bin/python directly leaves .venv/bin off PATH, so the compile fails with
# FileNotFoundError: 'ninja' inside the vLLM EngineCore subprocess and the whole
# engine dies. It surfaces first as a "GDN prefill kernel warmup failed" warning
# at load, then fatally on first inference.
export PATH="$PWD/.venv/bin:$PATH"
LOG=results/phase1_run.log
mkdir -p results

# model:evict_after  -- the three small models stay resident on disk; they are
# Phase 4's adversarial generators and re-downloading them would cost more than
# the space they hold.
MODELS=(
  "llama32_3b:0"
  "phi4_mini_reasoning:0"
  "ministral3_14b:0"
  "qwen38_27b:1"
  "granite42_30b:1"
  "gemma4_31b:1"
  "olmo3_32b_think:1"
)

hf_id() { $PY - "$1" <<'EOF'
import sys, yaml
print(yaml.safe_load(open("configs/models.yaml"))["models"][sys.argv[1]]["hf_id"])
EOF
}

started=0
for entry in "${MODELS[@]}"; do
  m="${entry%%:*}"; evict="${entry##*:}"
  if [ -n "$START_FROM" ] && [ "$started" = "0" ]; then
    if [ "$m" = "$START_FROM" ]; then started=1; else
      echo "=== skipping $m (resuming from $START_FROM) ===" | tee -a "$LOG"; continue
    fi
  fi
  # Resume without paying for it. phase_a_cache skips cached (task_id, model)
  # cells, but only AFTER loading the model, and large checkpoints are evicted
  # once their block finishes -- so re-running a finished model would re-download
  # ~60 GB, re-run its gate, find nothing to do, and then EVICT it again, forcing
  # the T=0.7 pass to fetch it a third time. Check the artifacts on disk first.
  n_cells=$(ls data/cache/*/"$m".parquet 2>/dev/null | wc -l)
  n_benchmarks=$($PY -c "import json; print(len(json.load(open('configs/task_lists.json'))['benchmarks']))")
  if [ "$n_cells" -ge "$n_benchmarks" ]; then
    echo "=== $(date -Is)  $m all $n_cells cells cached -- skipped (no reload, no evict) ===" | tee -a "$LOG"
    continue
  fi

  echo "=== $(date -Is)  $m  DRY-RUN GATE ===" | tee -a "$LOG"
  $PY scripts/phase_a_cache.py --dry-run --benchmarks gsm8k --models "$m" \
      --tag "dryrun_$m" >>"$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "GATE FAILED for $m (exit $rc) -- STOPPING, no substitutions." | tee -a "$LOG"
    exit $rc
  fi
  echo "=== $(date -Is)  $m  GATE PASSED -> FULL CACHE (6 benchmarks) ===" | tee -a "$LOG"
  $PY scripts/phase_a_cache.py --models "$m" --tag "full_$m" >>"$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FULL RUN FAILED for $m (exit $rc) -- STOPPING." | tee -a "$LOG"
    exit $rc
  fi
  if [ "$evict" = "1" ]; then
    $PY scripts/hf_evict.py --model "$(hf_id "$m")" >>"$LOG" 2>&1
    echo "=== $(date -Is)  $m evicted; free: $(df -h / | awk 'NR==2{print $4}') ===" | tee -a "$LOG"
  fi
done
echo "=== $(date -Is)  PHASE 1 COMPLETE ===" | tee -a "$LOG"
