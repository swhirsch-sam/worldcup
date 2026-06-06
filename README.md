# World Cup 2026 Bracket Predictor

Probabilistic simulation of the 2026 FIFA World Cup. Predicts group advancement,
the eight best third-place qualifiers, and the full knockout bracket, with
per-team probabilities for reaching every round — each with a 95% confidence interval.

## Quick start

```bash
pip install -r requirements.txt
make ingest      # fetch and cache Elo / FIFA / odds
make fit         # fit goals model on historical data
make simulate    # run 50k Monte Carlo → results/simulation_summary.json
make app         # launch Streamlit report
```

## Correctness risks

> **bracket_map.json is the top correctness risk.**
> It encodes which group positions and best-third combinations fill which R32 slots.
> Verify `data/bracket_map.json` against the official FIFA 2026 bracket allocation
> table before running the simulation. A startup assertion in `src/tournament/bracket.py`
> checks structural integrity (every group referenced the right number of times,
> no slot double-filled), but the table itself must be supplied and verified manually.

## Required data files

| File | Status | Source |
|---|---|---|
| `data/groups.json` | **AWAITING CONFIRMATION** | Dec 5 2025 FIFA draw |
| `data/bracket_map.json` | **AWAITING CONFIRMATION** | Official FIFA R32 allocation |
| `data/raw/historical_results.csv` | Download separately | See below |

### Historical results

Download international match results from [international_results.csv on Kaggle]
or the [Mart Jürisoo dataset on GitHub] covering matches from 1872 onward.
Place at `data/raw/historical_results.csv`. The pipeline filters by
`config.yaml:data.historical.earliest_date` (default `2000-01-01`).

## Configuration

All parameters live in `config.yaml`. No magic numbers in source code.

Key parameters:

| Key | Default | Description |
|---|---|---|
| `simulation.iterations` | 50000 | Monte Carlo iterations |
| `simulation.seed` | 42 | RNG seed (reproducibility) |
| `ensemble.weights.elo` | 0.60 | Elo weight in strength ensemble |
| `host.elo_bump` | 75.0 | Host team Elo bonus (group stage) |
| `dixon_coles.rho` | 0.10 | Low-score correction (fitted) |
| `knockout.caution_factor` | 0.85 | Expected goals reduction in KO rounds |

Run `make fit` to update the goals model coefficients and overwrite the
`goals_model.intercept`, `goals_model.slope`, and `dixon_coles.rho` fields.

## Repo structure

```
data/          Raw and processed data, groups.json, bracket_map.json
src/
  ingest/      Data fetching, caching, name resolution, schema validation
  model/       Strength ensemble, Dixon-Coles, bivariate Poisson, fitter, MC
  tournament/  Standings, tiebreakers, best-third, bracket allocation
  eval/        Backtesting, calibration, convergence, sensitivity
app/           Streamlit report (reads results/; never runs sim)
notebooks/     methodology.ipynb — calibration plots, backtest summary
results/       simulation_summary.json, run_manifest.json, plots
tests/         pytest + Hypothesis
```

## Development

```bash
make install-dev   # install deps + pre-commit hooks
make ci            # lint + typecheck + test (mirrors GitHub Actions)
make test          # pytest only
make lint          # ruff + black check
make typecheck     # mypy
```

## Reproducibility

`results/run_manifest.json` records: git commit SHA, RNG seed, config hash,
data source URLs and snapshot dates, iteration count, library versions, and
wall-clock timestamp. Fixed seed (`42`) by default. Pass `--seed N` to
override.

## Build order (phases)

- [x] Phase 1: Scaffold, config, tooling, CI, stubs
- [ ] Phase 2: Name registry + ingest + schema validation ← **current phase requires groups.json**
- [ ] Phase 3: Strength-to-goals fit + Dixon-Coles model
- [ ] Phase 4: Group simulation + tiebreakers + best-third
- [ ] Phase 5: Bracket allocation + knockout
- [ ] Phase 6: Vectorized Monte Carlo + invariant checks + manifest
- [ ] Phase 7: Eval suite (backtest, calibration, convergence, sensitivity)
- [ ] Phase 8: Streamlit report
- [ ] Phase 9: Methodology notebook

## License

MIT
