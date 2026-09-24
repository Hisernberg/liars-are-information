#!/bin/sh
# Build the manuscript: regenerate numbers, copy figures/tables, compile with tectonic.
set -e
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
TECTONIC=${TECTONIC:-tectonic}
$PY ../experiments/make_numbers.py
rm -rf figures tables && mkdir -p figures tables
cp ../figures/*.pdf figures/ 2>/dev/null || true
cp ../results/tables/*.tex tables/ 2>/dev/null || true
$TECTONIC -X compile main.tex
