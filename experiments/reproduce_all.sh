#!/bin/sh
# Reproduce every number, figure, table, video, the README and the manuscript from the
# cached answers in data/ (CPU only, no GPU, no API key). About 1.5-2 h on 4 cores.
# The live six-model swarms (E8, E10) are re-evaluated from data/live_cache and
# data/live_cache_v2; to regenerate the live answers themselves run
# experiments/live_swarm.py first (several hours each, see docs/PREREGISTRATION_E10.md).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
P=${PROCESSES:-4}
export PYTHONPATH=src
mkdir -p results/logs
stamp() { echo "$(date -u +%H:%M:%S) $*" | tee -a results/logs/reproduce.log; }
: > results/logs/reproduce.log
for s in main zoo llm history swarm ext crowd; do
  stamp "study $s"
  $PY experiments/run_suite.py "$s" --processes "$P" > "results/logs/$s.log" 2>&1
done
stamp "E6 online";          $PY experiments/run_online.py --processes "$P" > results/logs/online.log 2>&1
stamp "E7 diagnostics";     $PY experiments/analysis_extra.py > results/logs/extra.log 2>&1
stamp "E8 live evaluation"; $PY experiments/eval_live.py > results/logs/eval_live.log 2>&1
if [ -d data/live_cache_v2/cache ]; then
  stamp "E10 live evaluation"; $PY experiments/eval_live2.py > results/logs/eval_live2.log 2>&1
fi
stamp "figures";            $PY experiments/make_figures.py > results/logs/figures.log 2>&1
stamp "numbers";            $PY experiments/make_numbers.py > results/logs/numbers.log 2>&1
stamp "late figures";       $PY experiments/make_figures.py --late >> results/logs/figures.log 2>&1
stamp "film";               $PY experiments/make_film.py > results/logs/film.log 2>&1
stamp "videos";             $PY experiments/make_videos.py > results/logs/videos.log 2>&1
stamp "label switching";    $PY experiments/make_switching.py > results/logs/switching.log 2>&1
stamp "showcase video";     $PY experiments/make_showcase.py > results/logs/showcase.log 2>&1
stamp "gifs";               $PY experiments/make_gifs.py > results/logs/gifs.log 2>&1
stamp "claims";             $PY experiments/verify_claims.py > results/logs/claims.log 2>&1 || \
  { stamp "a prose claim no longer holds: see results/logs/claims.log"; exit 1; }
stamp "readme";             $PY experiments/make_readme.py > results/logs/readme.log 2>&1
if [ -n "$TECTONIC" ] || command -v tectonic > /dev/null; then
  stamp "paper"; (cd paper && ./build.sh > ../results/logs/paper_build.log 2>&1)
fi
stamp "done"
