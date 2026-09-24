#!/bin/sh
# Full v3 evaluation suite on cached answers (CPU only). ~30-60 min on 4 cores.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p results/logs
for s in main zoo llm history swarm; do
  $PY experiments/run_suite.py "$s" > "results/logs/$s.log" 2>&1
done
$PY experiments/run_online.py > results/logs/online.log 2>&1
echo done > results/logs/SUITE_DONE
