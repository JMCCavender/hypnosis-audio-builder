# Hypnosis Audio Builder - Makefile
# Run 'make help' to see available commands

SHELL := /bin/bash
VENV  := venv
BIN   := $(VENV)/bin
PYTHON := $(BIN)/python
PIP   := $(BIN)/pip

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Full install (venv + deps + ffmpeg check)
	@./install.sh

.PHONY: venv
venv: $(VENV)/bin/activate ## Create virtual environment only
$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -e ".[dev]" --quiet
	@echo "Virtual environment ready. Run: source $(VENV)/bin/activate"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

.PHONY: test-audio
test-audio: venv ## Run built-in audio generation test
	$(PYTHON) hypnosis_audio_builder.py --test

.PHONY: presets
presets: venv ## List available session presets
	$(PYTHON) hypnosis_audio_builder.py --list-presets

.PHONY: frequency-guide
frequency-guide: venv ## Open the interactive frequency explorer
	$(PYTHON) hypnosis_audio_builder.py --open-frequency-guide

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

.PHONY: test
test: venv ## Run test suite
	$(BIN)/pytest tests/ -v

.PHONY: check
check: ## Check prerequisites (Python, ffmpeg) without installing
	@./install.sh --check

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove virtual environment and build artifacts
	rm -rf $(VENV) build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

.PHONY: reinstall
reinstall: clean install ## Clean and reinstall everything

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "  Hypnosis Audio Builder"
	@echo "  ────────────────────────────────"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
