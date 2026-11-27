.PHONY: create-env activate-env lint unit-test smoke-test run-evaluation fetch-video-dataset video-smoke-test run-video-evaluation

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
	$(PIP) install --no-deps -r requirements-scenedetect.txt

activate-env: create-env
	@echo "Environment ready in $(VENV_DIR)."
	@echo "To activate in your shell: source $(VENV_DIR)/bin/activate"
	@echo "Or run commands via: $(PY) script.py"

lint:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -q -r hma/requirements-dev.txt
	$(VENV_DIR)/bin/ruff check .

# The HMA and Sentinel suites share module names, so they run as separate pytest sessions.
unit-test: activate-env
	$(PIP) install -q -r hma/requirements-dev.txt -r sentinel/requirements-dev.txt
	status=0; \
	PYTHONPATH=. $(PY) -m pytest hma/unit_tests -v || status=$$?; \
	PYTHONPATH=. $(PY) -m pytest sentinel/tests -v || status=$$?; \
	exit $$status

smoke-test: activate-env
	@echo "Running smoke test locally against HMA at $(HMA_HOST):$(HMA_PORT)..."
	PYTHONPATH=. OUTPUT_DIR=$(OUTPUT_DIR) HMA_HOST=$(HMA_HOST) HMA_PORT=$(HMA_PORT) POSTGRES_HOST=localhost POSTGRES_PORT=55432 BANK_NAME=SMOKE_TEST $(PY) hma/image/evaluate.py

run-evaluation: activate-env
	@echo "Running full evaluation locally against HMA at $(HMA_HOST):$(HMA_PORT)..."
	PYTHONPATH=. OUTPUT_DIR=$(OUTPUT_DIR) HMA_HOST=$(HMA_HOST) HMA_PORT=$(HMA_PORT) POSTGRES_HOST=localhost POSTGRES_PORT=55432 EVAL_MODE=test $(PY) hma/image/evaluate.py

fetch-video-dataset:
	bash hma/video/resources/fetch_dataset.sh

video-smoke-test: activate-env
	@echo "Running video smoke test locally against HMA at $(HMA_HOST):$(HMA_PORT)..."
	PYTHONPATH=. OUTPUT_DIR=$(OUTPUT_DIR) HMA_HOST=$(HMA_HOST) HMA_PORT=$(HMA_PORT) POSTGRES_HOST=localhost POSTGRES_PORT=55432 $(PY) hma/video/smoke_test.py

run-video-evaluation: activate-env
	@echo "Running video evaluation locally against HMA at $(HMA_HOST):$(HMA_PORT)..."
	PYTHONPATH=. OUTPUT_DIR=$(OUTPUT_DIR) HMA_HOST=$(HMA_HOST) HMA_PORT=$(HMA_PORT) POSTGRES_HOST=localhost POSTGRES_PORT=55432 $(PY) hma/video/evaluate.py
