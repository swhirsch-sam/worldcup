"""Convergence analysis: run the simulation at increasing N and track SE.

Plots how key probabilities and their standard errors stabilize across
iteration checkpoints defined in config.yaml.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def run_convergence_analysis(
    strength_df: object,
    model: object,
    cfg: dict[str, Any],
) -> None:
    """Run simulations at each checkpoint and save a convergence plot.

    Saves to cfg["output"]["convergence_plot"].
    """
    raise NotImplementedError("run_convergence_analysis: implement in Phase 7.")


def mc_standard_error(p: float, n: int) -> float:
    """Monte Carlo standard error: sqrt(p(1-p)/n)."""
    return float(np.sqrt(p * (1 - p) / max(n, 1)))


def confidence_interval_95(p: float, n: int) -> tuple[float, float]:
    """Wilson score 95% CI — more reliable than normal approx near 0/1."""
    from scipy.stats import norm

    z = norm.ppf(0.975)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, float(centre - half)), min(1.0, float(centre + half)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit("convergence.py: full implementation in Phase 7.")
