.PHONY: help install test quick all paper clean

help:
	@echo "make install | test | quick | all | paper | clean"

install:
	pip install -r requirements.txt pytest

test:
	pytest -q tests

quick:
	python ptopofl_v2.py --exp all --quick --no-heart --out results_quick

all:
	python ptopofl_v2.py --exp all --seeds 10 --out results_v2

paper:
	cd paper && latexmk -pdf ptopofl.tex

clean:
	rm -rf results_quick results_ci .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	cd paper && latexmk -C ptopofl.tex 2>/dev/null || true
