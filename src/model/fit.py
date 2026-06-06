"""Fit the strength-to-goals mapping and Dixon-Coles parameters on historical data.

Outputs fitted coefficients to stdout and writes them back into config.yaml.
Running `make fit` invokes this module.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict:  # type: ignore[type-arg]
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_historical(cfg: dict) -> pd.DataFrame:  # type: ignore[type-arg]
    """Load and filter historical results from the processed data directory."""
    raise NotImplementedError("load_historical: implement in Phase 3.")


def fit_goals_model(historical_df: pd.DataFrame, strength_df: pd.DataFrame) -> dict[str, float]:
    """Fit the Poisson/MLE goals model: lambda = exp(intercept + slope * delta).

    Args:
        historical_df: Validated historical match results.
        strength_df: Processed strength ratings per team.

    Returns:
        dict with keys: intercept, slope (printed and written to config).
    """
    raise NotImplementedError("fit_goals_model: implement in Phase 3.")


def fit_dixon_coles(
    historical_df: pd.DataFrame,
    strength_df: pd.DataFrame,
    xi: float,
) -> dict[str, float]:
    """Fit Dixon-Coles rho and time-decay xi via MLE on historical data.

    Args:
        historical_df: Historical matches with dates.
        strength_df: Team strengths.
        xi: Initial time-decay value (from config, or derived from half-life).

    Returns:
        dict with keys: rho (and optionally xi if jointly fitted).
    """
    raise NotImplementedError("fit_dixon_coles: implement in Phase 3.")


def write_fitted_coefficients(goals_coef: dict[str, float], dc_coef: dict[str, float]) -> None:
    """Write fitted coefficients back to config.yaml."""
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["goals_model"]["intercept"] = float(goals_coef["intercept"])
    cfg["goals_model"]["slope"] = float(goals_coef["slope"])
    cfg["dixon_coles"]["rho"] = float(dc_coef["rho"])

    with open("config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print("\n=== Fitted coefficients written to config.yaml ===")
    print(f"  goals_model.intercept : {goals_coef['intercept']:.6f}")
    print(f"  goals_model.slope     : {goals_coef['slope']:.6f}")
    print(f"  dixon_coles.rho       : {dc_coef['rho']:.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Fit model parameters on historical data")
    parser.add_argument(
        "--historical", default=None, help="Path to historical CSV (overrides config)"
    )
    args = parser.parse_args()

    cfg = _load_config()
    hist_df = load_historical(cfg)
    print(f"Loaded {len(hist_df)} historical matches for fitting.")
    # Phase 3: call fit functions and write results
    raise SystemExit("fit.py: full implementation in Phase 3.")
