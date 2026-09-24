# Build the manuscript. `make numbers` regenerates the macro file from the
# result parquets first; never edit numbers.tex by hand.
#
# Either engine works. latexmk is the conventional choice; tectonic is a single
# static binary that fetches the packages it needs, which is what this project
# actually builds with -- there is no TeX Live installation on the run host, and
# a Makefile that names only latexmk made `make` fail with exit 127 while
# claiming in the README that the manuscript builds.
# `python` is not a command on every host -- this one has python3 and a venv,
# and `make numbers` died with "python: not found" before this was derived.
PY := $(shell test -x ../.venv/bin/python && echo ../.venv/bin/python || command -v python3)
TEX := $(shell command -v latexmk 2>/dev/null)
ifeq ($(TEX),)
  BUILD := tectonic -X compile main.tex --keep-logs
else
  BUILD := latexmk -pdf main.tex
endif

.PHONY: all numbers clean check

all: numbers
	$(BUILD)

numbers:
	cd .. && $(subst ../,,$(PY)) scripts/make_numbers.py

check:
	cd .. && $(subst ../,,$(PY)) -m pytest tests/test_paper.py -q

clean:
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.out main.log main.xdv
