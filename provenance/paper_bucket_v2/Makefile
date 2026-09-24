# Build the manuscript. `make numbers` regenerates the macro file from the
# result parquets first; never edit numbers.tex by hand.
.PHONY: all numbers clean check

all: numbers
	latexmk -pdf main.tex

numbers:
	cd .. && python scripts/make_numbers.py

check:
	cd .. && python -m pytest tests/test_paper.py -q

clean:
	latexmk -C
	rm -f main.bbl main.blg
