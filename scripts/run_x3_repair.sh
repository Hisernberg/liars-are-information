#!/usr/bin/env bash
# X3 driver: regenerate every cap-truncated cache row, one model at a time.
#
# Ordering is disk-driven, not preference. Five checkpoints are still in the hub
# cache and cost nothing to run; the four large ones were evicted after Phase 1
# and must be re-fetched at ~60 GB each against ~60 GB of free space, so each is
# downloaded, run, and evicted before the next arrives. The small models run
# first so a harness fault surfaces before an hour of downloading.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# The venv's bin/ must be on PATH, not merely its python: Qwen 3.8's Gated Delta
# Net layers JIT-compile through `ninja`, and the vLLM EngineCore subprocess
# dies without it. Same reason as scripts/run_phase1.sh.
export PATH="$PWD/.venv/bin:$PATH"
export HF_HUB_DISABLE_PROGRESS_BARS=1
LOG=results/x3_repair_run.log
mkdir -p results

RESIDENT=(phi4_mini_reasoning llama32_3b olmo2_7b ministral_8b ministral3_14b)
DOWNLOAD=(granite42_30b qwen38_27b gemma4_31b olmo3_32b_think)

hf_id() { $PY - "$1" <<'PYEOF'
import sys, yaml
print(yaml.safe_load(open("configs/models.yaml"))["models"][sys.argv[1]]["hf_id"])
PYEOF
}

run_model() {
  local m="$1"
  echo "=== $(date -Is)  X3 $m  (free: $(df -h / | awk 'NR==2{print $4}')) ===" | tee -a "$LOG"
  $PY scripts/x3_repair.py --models "$m" >>"$LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "X3 FAILED for $m (exit $rc) -- STOPPING." | tee -a "$LOG"
    exit $rc
  fi
}

for m in "${RESIDENT[@]}"; do run_model "$m"; done

# Reclaim before the downloads start. olmo2_7b and ministral_8b are the retired
# weak-tier arm and nothing later needs them on disk; ministral3_14b stays (it
# is Phase 4's secondary adversarial generator) but its consolidated
# safetensors copy is redundant -- vLLM loads the shards.
$PY scripts/hf_evict.py --model "$(hf_id olmo2_7b)" >>"$LOG" 2>&1
$PY scripts/hf_evict.py --model "$(hf_id ministral_8b)" >>"$LOG" 2>&1
$PY scripts/hf_evict.py --drop-consolidated >>"$LOG" 2>&1
echo "=== $(date -Is)  reclaimed; free: $(df -h / | awk 'NR==2{print $4}') ===" | tee -a "$LOG"

for m in "${DOWNLOAD[@]}"; do
  run_model "$m"
  $PY scripts/hf_evict.py --model "$(hf_id "$m")" >>"$LOG" 2>&1
  echo "=== $(date -Is)  $m evicted; free: $(df -h / | awk 'NR==2{print $4}') ===" | tee -a "$LOG"
done

echo "=== $(date -Is)  X3 REPAIR COMPLETE ===" | tee -a "$LOG"
