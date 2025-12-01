.PHONY: create-env activate-env smoke-test run-evaluation

PYTHON ?= python3
VENV_DIR ?= .venv
PY ?= $(VENV_DIR)/bin/python
PIP ?= $(VENV_DIR)/bin/pip
OUTPUT_DIR ?= ./results
HMA_HOST ?= localhost
HMA_PORT ?= 5005

create-env:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

activate-env: create-env
	@echo "Environment ready in $(VENV_DIR)."
	@echo "To activate in your shell: source $(VENV_DIR)/bin/activate"
	@echo "Or run commands via: $(PY) script.py"

smoke-test: activate-env
	@echo "Running smoke test locally against HMA at $(HMA_HOST):$(HMA_PORT)..."
	OUTPUT_DIR=$(OUTPUT_DIR) HMA_HOST=$(HMA_HOST) HMA_PORT=$(HMA_PORT) POSTGRES_HOST=localhost POSTGRES_PORT=55432 BANK_NAME=SMOKE_TEST $(PY) evaluate.py

run-evaluation: activate-env
	@echo "Running full evaluation locally against HMA at $(HMA_HOST):$(HMA_PORT)..."
	OUTPUT_DIR=$(OUTPUT_DIR) HMA_HOST=$(HMA_HOST) HMA_PORT=$(HMA_PORT) POSTGRES_HOST=localhost POSTGRES_PORT=55432 EVAL_MODE=test $(PY) evaluate.py


