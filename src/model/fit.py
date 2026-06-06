"""Fit the strength-to-goals mapping and Dixon-Coles parameters on historical data.

Outputs fitted coefficients to stdout and writes them back into config.yaml.
Running `make fit` invokes this module.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize, minimize_scalar

from src.ingest.elo import compute_elo_with_records
from src.ingest.historical import fetch_historical

logger = logging.getLogger(__name__)


def _load_config() -> dict:  # type: ignore[type-arg]
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_historical_with_elo(cfg: dict, cutoff_date: str | None = None) -> pd.DataFrame:  # type: ignore[type-arg]
    """Return match records with pre-match Elo, filtered to earliest_date <= date <= cutoff_date.

    Calls fetch_historical() then compute_elo_with_records().
    Applies cfg["data"]["historical"]["earliest_date"] lower bound.

    Returns:
        DataFrame with: date, home_team, away_team, home_score, away_score,
        home_elo, away_elo, neutral, tournament
    """
    hist_df = fetch_historical()
    _ratings, _match_counts, match_records = compute_elo_with_records(hist_df, cfg)

    earliest_str: str = cfg["data"]["historical"]["earliest_date"]
    earliest = pd.Timestamp(earliest_str)
    mask = match_records["date"] >= earliest

    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        mask = mask & (match_records["date"] <= cutoff)

    return match_records.loc[mask].reset_index(drop=True)


def _compute_elo_diff(match_records: pd.DataFrame, home_adv: float) -> pd.Series:
    """elo_diff = home_elo - away_elo + home_adv * (1 - neutral)"""
    neutral_flag = match_records["neutral"].astype(float)
    return match_records["home_elo"] - match_records["away_elo"] + home_adv * (1.0 - neutral_flag)


def fit_goals_model(match_records: pd.DataFrame, cfg: dict) -> dict[str, float]:  # type: ignore[type-arg]
    """Fit lambda = exp(alpha + beta * elo_diff) via Poisson MLE.

    Both home and away goals contribute to the log-likelihood.

    Model:
      elo_diff_i = home_elo_i - away_elo_i + home_advantage * (1 - neutral_i)
      lambda_home_i = exp(alpha + beta * elo_diff_i)
      lambda_away_i = exp(alpha - beta * elo_diff_i)

    Maximize: sum_i [home_score_i * log(lambda_home_i) - lambda_home_i
                   + away_score_i * log(lambda_away_i) - lambda_away_i]

    Use scipy.optimize.minimize with method='L-BFGS-B',
    x0=[log(1.35), 0.001], bounds=[(-2,2),(-0.1, 0.1)]

    Returns: {"intercept": float, "slope": float}
    """
    home_adv: float = float(cfg["elo_computation"]["home_advantage_elo"])
    elo_diff = _compute_elo_diff(match_records, home_adv).to_numpy()
    home_goals = match_records["home_score"].to_numpy(dtype=float)
    away_goals = match_records["away_score"].to_numpy(dtype=float)

    def neg_log_likelihood(params: np.ndarray) -> float:
        alpha, beta = params
        lam_h = np.exp(alpha + beta * elo_diff)
        lam_a = np.exp(alpha - beta * elo_diff)
        ll = home_goals * np.log(lam_h) - lam_h + away_goals * np.log(lam_a) - lam_a
        return -ll.sum()

    x0 = np.array([np.log(1.35), 0.001])
    bounds = [(-2.0, 2.0), (-0.1, 0.1)]
    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)

    alpha_fit, beta_fit = result.x
    logger.info(
        "Goals model fit: alpha=%.6f, beta=%.6f (converged=%s)",
        alpha_fit,
        beta_fit,
        result.success,
    )
    return {"intercept": float(alpha_fit), "slope": float(beta_fit)}


def fit_dixon_coles(
    match_records: pd.DataFrame, goals_coef: dict, cfg: dict  # type: ignore[type-arg]
) -> dict[str, float]:
    """Fit Dixon-Coles rho with time-decay weighting (sequential after goals model fit).

    Time decay: weight_i = exp(-xi * days_since_match)
    where xi = log(2) / halflife_days (from config)

    tau correction:
      tau(0,0) = 1 - lambda_h * lambda_a * rho
      tau(1,0) = 1 + lambda_a * rho
      tau(0,1) = 1 + lambda_h * rho
      tau(1,1) = 1 - rho
      tau(x,y) = 1 otherwise

    IMPORTANT: rho is NEGATIVE in football (more 0-0 and 1-1 than independent
    Poisson predicts). Optimize over rho in bounds [-0.5, 0.0].
    Use scipy.optimize.minimize_scalar, method='bounded'.

    Returns: {"rho": float}
    """
    halflife: float = float(cfg["dixon_coles"]["time_decay_halflife_days"])
    xi: float = np.log(2.0) / halflife

    home_adv: float = float(cfg["elo_computation"]["home_advantage_elo"])
    alpha: float = float(goals_coef["intercept"])
    beta: float = float(goals_coef["slope"])

    elo_diff = _compute_elo_diff(match_records, home_adv).to_numpy()
    home_goals = match_records["home_score"].to_numpy(dtype=int)
    away_goals = match_records["away_score"].to_numpy(dtype=int)

    # Compute lambdas from fitted goals model
    lam_h = np.exp(alpha + beta * elo_diff)
    lam_a = np.exp(alpha - beta * elo_diff)

    # Time decay weights: most recent date is reference
    dates = pd.to_datetime(match_records["date"])
    most_recent = dates.max()
    days_since = (most_recent - dates).dt.days.to_numpy(dtype=float)
    weights = np.exp(-xi * days_since)

    # Identify low-score cells
    is_00 = (home_goals == 0) & (away_goals == 0)
    is_10 = (home_goals == 1) & (away_goals == 0)
    is_01 = (home_goals == 0) & (away_goals == 1)
    is_11 = (home_goals == 1) & (away_goals == 1)

    # Poisson log-likelihoods (score-independent part)
    # log P(x|lam) = x*log(lam) - lam - log(x!)  — constant log(x!) drops out in optimization
    log_p_h = home_goals * np.log(lam_h) - lam_h
    log_p_a = away_goals * np.log(lam_a) - lam_a

    def neg_weighted_ll(rho: float) -> float:
        # Tau corrections (vectorized)
        log_tau = np.zeros(len(home_goals))
        log_tau[is_00] = np.log(np.maximum(1.0 - lam_h[is_00] * lam_a[is_00] * rho, 1e-10))
        log_tau[is_10] = np.log(np.maximum(1.0 + lam_a[is_10] * rho, 1e-10))
        log_tau[is_01] = np.log(np.maximum(1.0 + lam_h[is_01] * rho, 1e-10))
        log_tau[is_11] = np.log(np.maximum(1.0 - rho, 1e-10))

        ll = weights * (log_p_h + log_p_a + log_tau)
        return -ll.sum()

    result = minimize_scalar(neg_weighted_ll, bounds=(-0.5, 0.0), method="bounded")
    rho_fit = float(result.x)
    logger.info("Dixon-Coles fit: rho=%.6f (converged=%s)", rho_fit, result.success)
    return {"rho": rho_fit}


def write_fitted_coefficients(goals_coef: dict[str, float], dc_coef: dict[str, float]) -> None:
    """Write fitted coefficients back to config.yaml and print a clear summary."""
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["goals_model"]["intercept"] = float(goals_coef["intercept"])
    cfg["goals_model"]["slope"] = float(goals_coef["slope"])
    cfg["dixon_coles"]["rho"] = float(dc_coef["rho"])

    with open("config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print()
    print("=" * 60)
    print("  FITTED COEFFICIENTS — written to config.yaml")
    print("=" * 60)
    print(f"  goals_model.intercept : {goals_coef['intercept']:+.6f}")
    print(f"  goals_model.slope     : {goals_coef['slope']:+.8f}")
    print(f"  dixon_coles.rho       : {dc_coef['rho']:+.6f}")
    print("=" * 60)
    print()
    # Derived statistics
    mean_lambda = np.exp(goals_coef["intercept"])
    print(f"  Implied mean goals per team per match : {mean_lambda:.4f}")
    print(
        f"  Elo-diff effect at +100 pts          : "
        f"exp(slope*100) = {np.exp(goals_coef['slope']*100):.4f}x"
    )
    print()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fit model parameters on historical data")
    parser.add_argument("--cutoff", default=None, help="Optional cutoff date (YYYY-MM-DD)")
    args = parser.parse_args()

    cfg = _load_config()

    print("Loading historical match records with pre-match Elo ratings...")
    match_records = load_historical_with_elo(cfg, cutoff_date=args.cutoff)
    print(f"Loaded {len(match_records):,} match records for fitting.")
    date_min = match_records["date"].min().date()
    date_max = match_records["date"].max().date()
    print(f"  Date range : {date_min} to {date_max}")
    print()

    print("Fitting goals model (Poisson MLE)...")
    goals_coef = fit_goals_model(match_records, cfg)
    print(f"  alpha (intercept) = {goals_coef['intercept']:+.6f}")
    print(f"  beta  (slope)     = {goals_coef['slope']:+.8f}")
    print()

    print("Fitting Dixon-Coles rho (time-decay weighted MLE)...")
    dc_coef = fit_dixon_coles(match_records, goals_coef, cfg)
    print(f"  rho = {dc_coef['rho']:+.6f}")
    print()

    write_fitted_coefficients(goals_coef, dc_coef)
