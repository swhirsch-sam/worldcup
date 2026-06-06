"""Calibration metrics and reliability diagram.

Compares predicted probabilities to observed frequencies in bins.
Used by both the backtest module and the Streamlit app summary section.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from numpy.typing import NDArray


def reliability_diagram_data(
    predicted: NDArray[Any],
    observed: NDArray[Any],
    n_bins: int = 10,
    min_bin_size: int = 10,
) -> pd.DataFrame:
    """Compute binned mean predicted vs mean observed for a reliability diagram.

    Args:
        predicted: Array of predicted probabilities ∈ [0, 1].
        observed: Array of binary outcomes (0 or 1), same length.
        n_bins: Number of equal-width bins across [0, 1].
        min_bin_size: Bins with fewer samples are excluded.

    Returns:
        DataFrame with columns: bin_midpoint, mean_predicted, mean_observed, count
    """
    raise NotImplementedError("reliability_diagram_data: implement in Phase 7.")


def calibration_summary(
    predicted: NDArray[Any],
    observed: NDArray[Any],
) -> dict[str, float]:
    """Return a dict with overall_brier, overall_log_loss, and expected_calibration_error."""
    raise NotImplementedError("calibration_summary: implement in Phase 7.")
