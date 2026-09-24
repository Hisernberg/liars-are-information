#!/usr/bin/env bash
# X3: recompute every downstream surface from the repaired cache.
#
# No GPU. Every step here reads the parquet cache and loads no model.
# phase_d_adversarial is deliberately absent: scripts/x3_repair_adversarial.py
# has already rewritten the rows it would have generated, and re-running it
# would only resume-skip.
#
# Order is the one run_all.py documents the hard way -- the mechanism
# experiments and the tables must precede the ledger that cites them, and the
# measurement floor must precede numbers.tex, which reads it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PATH="$PWD/.venv/bin:$PATH"
LOG=results/x3_recompute.log
: > "$LOG"

step() {
  echo "=== $(date -Is)  $* ===" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1 || { echo "FAILED: $*" | tee -a "$LOG"; exit 1; }
}

# measurement_floor.json is an INPUT to phase_composition.py and
# task6_audit_pending.py. It used to run at position 22, after both, so a
# recompute fed them the PREVIOUS run's floor and their 'detectable' columns
# were decided by a superseded threshold. It runs first now.
step $PY scripts/measurement_floor.py
step $PY scripts/phase_b_correlation.py
step $PY scripts/phase_b2_multiclass.py
step $PY scripts/calibrate_receiver_coherence.py
step $PY scripts/phase_c_aggregate.py
step $PY scripts/phase_c_report.py
step $PY scripts/phase_d_sweep.py
step $PY scripts/phase_d_bandit.py
step $PY scripts/phase_gate_aware.py
step $PY scripts/phase_e_sleeper_sybil.py
step $PY scripts/phase_e_burst.py
step $PY scripts/phase_e_decomposition.py
step $PY scripts/phase_e_tables.py
step $PY scripts/phase_e_claims.py
step $PY scripts/task6_audit_pending.py
step $PY scripts/phase_e3_cost.py
step $PY scripts/phase_composition.py
step $PY scripts/phase_confidence_auc.py
# The floor is itself downstream of the T=0.7 cache, which X3 repairs, so it is
# recomputed BEFORE the macros that quote it.
step $PY scripts/make_numbers.py
step $PY scripts/phase_c_figures.py
step $PY scripts/phase_d_figures.py
step $PY scripts/phase_e_figures.py
step $PY scripts/fig_evasion_band.py
step $PY scripts/fig_sybil.py
step $PY -m pytest tests/test_paper.py -q
echo "=== $(date -Is)  X3 RECOMPUTE COMPLETE ===" | tee -a "$LOG"
