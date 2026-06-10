"""Closed-form Dixon-Coles match outcome math (numpy-only).

This module is intentionally dependency-light — it imports *only* numpy so the
Streamlit app (whose runtime has just numpy/pandas/streamlit/plotly) can import
it directly for live "head-to-head" predictions, while the offline precompute
in ``src/model/match_predict.py`` uses the same functions to generate
``results/match_predictions.json``.

The scoreline grid mirrors ``DixonColesModel`` in ``poisson.py`` exactly (same
0..10 grid, same tau low-score correction). The difference is purpose: this
module evaluates the *analytic* distribution (win/draw/loss, expected goals,
most-likely scores) rather than drawing random samples. ``test_match_predict``
asserts the two stay numerically in sync.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Goal grid 0..10, matching DixonColesModel's truncation.
MAX_GOALS = 10
_GRID: NDArray[np.int64] = np.arange(MAX_GOALS + 1)
# Precomputed log-factorial for a numpy-only Poisson PMF (avoids scipy).
_LOG_FACTORIAL: NDArray[np.float64] = np.concatenate(
    ([0.0], np.cumsum(np.log(np.arange(1, MAX_GOALS + 1))))
)


def _poisson_pmf(lam: float) -> NDArray[np.float64]:
    """Poisson PMF over the 0..10 goal grid, computed with numpy only."""
    safe_lam = max(lam, 1e-12)
    logp = -safe_lam + _GRID * np.log(safe_lam) - _LOG_FACTORIAL
    return np.exp(logp)


def match_lambdas(
    strength_home: float,
    strength_away: float,
    *,
    intercept: float,
    slope: float,
    lambda_floor: float = 0.1,
    lambda_cap: float = 6.0,
) -> tuple[float, float]:
    """Expected goals for each side from the fitted goals model.

    Mirrors ``simulate_group`` (neutral venue): ``lambda = exp(alpha +/- beta *
    elo_diff)`` where ``elo_diff`` is the composite-strength gap, clipped to
    ``[lambda_floor, lambda_cap]``.
    """
    elo_diff = strength_home - strength_away
    lam_h = float(np.clip(np.exp(intercept + slope * elo_diff), lambda_floor, lambda_cap))
    lam_a = float(np.clip(np.exp(intercept - slope * elo_diff), lambda_floor, lambda_cap))
    return lam_h, lam_a


def dixon_coles_matrix(lambda_home: float, lambda_away: float, rho: float) -> NDArray[np.float64]:
    """Normalized 11x11 scoreline distribution; ``matrix[i, j] = P(home i, away j)``."""
    p_h = _poisson_pmf(lambda_home)
    p_a = _poisson_pmf(lambda_away)
    joint = np.outer(p_h, p_a)

    # Dixon-Coles low-score correction (same cells as poisson.DixonColesModel).
    joint[0, 0] *= 1.0 - lambda_home * lambda_away * rho
    joint[1, 0] *= 1.0 + lambda_away * rho
    joint[0, 1] *= 1.0 + lambda_home * rho
    joint[1, 1] *= 1.0 - rho

    total = joint.sum()
    return joint / total


def outcome_probabilities(matrix: NDArray[np.float64]) -> dict[str, Any]:
    """Derive human-friendly outcome stats from a scoreline matrix.

    Returns win/draw/loss probabilities (home perspective), expected goals for
    each side, and the three most-likely scorelines.
    """
    p_home = float(np.tril(matrix, -1).sum())  # home goals > away goals
    p_away = float(np.triu(matrix, 1).sum())  # away goals > home goals
    p_draw = float(np.trace(matrix))

    n_cols = matrix.shape[1]
    order = np.argsort(matrix, axis=None)[::-1]
    top_scores: list[dict[str, Any]] = []
    for flat_idx in order[:3]:
        home_goals, away_goals = divmod(int(flat_idx), n_cols)
        top_scores.append(
            {
                "home": home_goals,
                "away": away_goals,
                "prob": float(matrix[home_goals, away_goals]),
            }
        )

    exp_home = float((matrix.sum(axis=1) * _GRID).sum())
    exp_away = float((matrix.sum(axis=0) * _GRID).sum())

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "exp_home": exp_home,
        "exp_away": exp_away,
        "top_scores": top_scores,
    }


def predict_outcome(
    strength_home: float,
    strength_away: float,
    *,
    intercept: float,
    slope: float,
    rho: float,
    lambda_floor: float = 0.1,
    lambda_cap: float = 6.0,
) -> dict[str, Any]:
    """Full closed-form prediction for a single (neutral-venue) match."""
    lam_h, lam_a = match_lambdas(
        strength_home,
        strength_away,
        intercept=intercept,
        slope=slope,
        lambda_floor=lambda_floor,
        lambda_cap=lambda_cap,
    )
    matrix = dixon_coles_matrix(lam_h, lam_a, rho)
    return outcome_probabilities(matrix)
