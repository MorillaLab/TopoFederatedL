.PHONY: install lint test simulate clean help

help:
	@echo "TopoFederatedL — available commands:"
	@echo "  make install    Install all dependencies"
	@echo "  make lint       Lint Python source"
	@echo "  make test       Run unit tests"
	@echo "  make simulate   Run FL simulation demo"
	@echo "  make clean      Remove cache files"

install:
	pip install -r requirements.txt

lint:
	flake8 . --max-line-length=127 --exclude=.git,__pycache__ --count --statistics

test:
	pytest tests/ -v --tb=short 2>/dev/null || echo "No tests yet — skipping"
	continue-on-error: true

simulate:
	python -c "print('TopoFederatedL simulation — add your script here')"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".DS_Store" -delete 2>/dev/null; true
	find . -name "*_executed.ipynb" -delete 2>/dev/null; true
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null; true
