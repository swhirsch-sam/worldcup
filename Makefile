.DEFAULT_GOAL := help
PYTHON := python3
SRC := src
TESTS := tests

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Data / model targets:"
	@echo "  ingest           Fetch and cache all data sources"
	@echo "  strength         Build data/processed/strength.csv (ingest + ensemble)"
	@echo "  fit              Fit the goals model on historical data"
	@echo "  simulate         Run the 1k Monte Carlo simulation"
	@echo "  backtest         Run backtests for 2018 and 2022"
	@echo "  convergence      Run convergence analysis and save plot"
	@echo "  sensitivity      Run sensitivity analysis and save tornado plot"
	@echo ""
	@echo "App targets:"
	@echo "  app              Launch the Streamlit report (reads results/)"
	@echo ""
	@echo "Dev targets:"
	@echo "  install          Install Python dependencies"
	@echo "  install-dev      Install dev dependencies and pre-commit hooks"
	@echo "  lint             Run ruff + black check"
	@echo "  typecheck        Run mypy"
	@echo "  test             Run pytest"
	@echo "  ci               lint + typecheck + test (mirrors CI)"
	@echo "  refresh          Re-fetch all data sources ignoring cache TTL"
	@echo "  clean            Remove generated outputs"

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: install-dev
install-dev: install
	pre-commit install

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
.PHONY: ingest
ingest:
	$(PYTHON) -m src.ingest.elo
	$(PYTHON) -m src.ingest.fifa
	$(PYTHON) -m src.ingest.odds

.PHONY: refresh
refresh:
	$(PYTHON) -m src.ingest.elo --refresh
	$(PYTHON) -m src.ingest.fifa --refresh
	$(PYTHON) -m src.ingest.odds --refresh

.PHONY: strength
strength:
	$(PYTHON) -m src.model.strength

.PHONY: fit
fit:
	$(PYTHON) -m src.model.fit

.PHONY: simulate
simulate:
	$(PYTHON) -m src.model.montecarlo

.PHONY: backtest
backtest:
	$(PYTHON) -m src.eval.backtest

.PHONY: convergence
convergence:
	$(PYTHON) -m src.eval.convergence

.PHONY: sensitivity
sensitivity:
	$(PYTHON) -m src.eval.sensitivity

# Full pipeline (excluding app)
.PHONY: all
all: ingest fit simulate backtest convergence sensitivity

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
.PHONY: app
app:
	streamlit run app/streamlit_app.py

# ---------------------------------------------------------------------------
# Dev
# ---------------------------------------------------------------------------
.PHONY: lint
lint:
	ruff check $(SRC) $(TESTS)
	black --check --diff $(SRC) $(TESTS)

.PHONY: typecheck
typecheck:
	mypy $(SRC) --ignore-missing-imports

.PHONY: test
test:
	pytest $(TESTS) -q

.PHONY: ci
ci: lint typecheck test

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf results/simulation_summary.json results/run_manifest.json \
	       results/convergence.png results/sensitivity_tornado.png \
	       results/calibration.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
