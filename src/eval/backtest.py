"""Backtesting the pipeline on 2018 and 2022 World Cups.

For each tournament, uses only pre-tournament data to produce probabilities,
then scores them against actual outcomes using:
  - Ranked Probability Score (RPS) — standard for football
  - Brier score
  - Log loss
  - Reliability diagram (predicted vs observed frequency)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def run_backtest(year: int, cutoff_date: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run full pipeline on *year* using data up to *cutoff_date*.

    Returns a dict of calibration metrics for this tournament.
    """
    raise NotImplementedError(f"run_backtest({year}): implement in Phase 7.")


def ranked_probability_score(
    predicted: NDArray[Any],
    observed_outcome: int,
    n_outcomes: int = 3,
) -> float:
    """Compute the Ranked Probability Score for a single match.

    Args:
        predicted: Array of shape (n_outcomes,) summing to 1.
                   [P(home win), P(draw), P(away win)]
        observed_outcome: 0=home win, 1=draw, 2=away win.
        n_outcomes: Number of ordinal outcome bins.

    Returns:
        RPS ∈ [0, 1] — lower is better.
    """
    raise NotImplementedError("ranked_probability_score: implement in Phase 7.")


def brier_score(predicted_prob: float, outcome: int) -> float:
    """Binary Brier score for a single event."""
    return (predicted_prob - outcome) ** 2


def log_loss_single(predicted_prob: float, outcome: int, eps: float = 1e-15) -> float:
    """Log loss for a single event."""
    p = max(min(predicted_prob, 1 - eps), eps)
    return -(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit("backtest.py: full implementation in Phase 7.")
