#!/usr/bin/env bash
# X3 closing pass: the work that needs the four evicted checkpoints back.
#
# Three separate debts, all blocked on the same ~225 GB of downloads, so they
# are paid per model while it is resident rather than in three passes:
#
#   1. data/cache_t07 still has 52 truncated rows belonging to these four
#      models. The resident-model pass repaired 94 of its 146.
#   2. qwen38_27b's answer-level drift was never measured -- it was evicted
#      before the gate that measures it existed.
#   3. qwen38_27b needs a full-cell regeneration for the same reason
#      llama32_3b does: 19 of 40 control rows differed, so its repaired cells
#      are a mixture of two batch conditions.
#
# Free space is ~60 GB against ~60 GB per checkpoint, so each is downloaded,
# used for everything it owes, and evicted before the next arrives.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PATH="$PWD/.venv/bin:$PATH"
export HF_HUB_DISABLE_PROGRESS_BARS=1
LOG=results/x3_finish.log
mkdir -p results

MODELS=(gemma4_31b granite42_30b olmo3_32b_think qwen38_27b)

hf_id() { $PY - "$1" <<'PYEOF'
import sys, yaml
print(yaml.safe_load(open("configs/models.yaml"))["models"][sys.argv[1]]["hf_id"])
PYEOF
}

for m in "${MODELS[@]}"; do
  echo "=== $(date -Is)  X3-finish $m  (free: $(df -h / | awk 'NR==2{print $4}')) ===" | tee -a "$LOG"

  # 1. the T=0.7 rows this model owes
  $PY scripts/x3_repair.py --models "$m" --cache-root data/cache_t07 \
      --temperature 0.7 --sample-index 1 --out results/x3_t07_finish >>"$LOG" 2>&1 \
      || { echo "T07 FAILED for $m -- STOPPING" | tee -a "$LOG"; exit 1; }

  if [ "$m" = "qwen38_27b" ]; then
    # 2. answer-level drift, at the repair's own batch composition
    $PY scripts/x3_repair.py --models "$m" --no-write \
        --out results/x3_probe_qwen >>"$LOG" 2>&1 \
        || { echo "PROBE FAILED for $m -- STOPPING" | tee -a "$LOG"; exit 1; }
    # 3. whole cells in one batch, so the cell stops being a mixture
    $PY scripts/x3_fullcell.py --models "$m" >>"$LOG" 2>&1 \
        || { echo "FULLCELL FAILED for $m -- STOPPING" | tee -a "$LOG"; exit 1; }
  fi

  $PY scripts/hf_evict.py --model "$(hf_id "$m")" >>"$LOG" 2>&1
  echo "=== $(date -Is)  $m evicted; free: $(df -h / | awk 'NR==2{print $4}') ===" | tee -a "$LOG"
done

echo "=== $(date -Is)  X3 FINISH COMPLETE ===" | tee -a "$LOG"
