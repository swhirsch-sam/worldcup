"""Team strength computation: weighted ensemble of Elo, FIFA, and market signals.

Weights are read from config.yaml and renormalized when signals are absent.
Provisional teams are shrunk toward the global mean.
Host teams receive a configurable Elo bump.
"""

from __future__ import annotations

import logging

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict:  # type: ignore[type-arg]
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def build_strength_table(
    elo_df: pd.DataFrame,
    fifa_df: pd.DataFrame | None,
    odds_df: pd.DataFrame | None,
    *,
    hosts: list[str] | None = None,
    group_stage: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame with one row per canonical team and a composite *strength* column.

    The strength score is on the Elo scale (global mean ≈ 1500).

    Args:
        elo_df: Validated Elo DataFrame (team, elo_rating).
        fifa_df: Validated FIFA DataFrame or None.
        odds_df: Validated market odds DataFrame or None.
        hosts: Canonical names of host teams for the bump.
        group_stage: Whether host bump applies (True = group stage).

    Returns:
        DataFrame with columns: team, elo_rating, strength, [fifa_points], [implied_probability]
    """
    raise NotImplementedError("build_strength_table: implement in Phase 3.")


def _renormalize_weights(
    weights: dict[str, float],
    available: set[str],
) -> dict[str, float]:
    """Return weights restricted to *available* signals and renormalized to sum to 1."""
    active = {k: v for k, v in weights.items() if k in available}
    total = sum(active.values())
    if total == 0:
        raise ValueError("No signals available; cannot build strength table.")
    return {k: v / total for k, v in active.items()}


def _apply_shrinkage(
    ratings: pd.Series,  # type: ignore[type-arg]
    provisional_mask: pd.Series,  # type: ignore[type-arg]
    global_mean: float,
    shrinkage: float,
) -> pd.Series:  # type: ignore[type-arg]
    """Shrink provisional team ratings toward *global_mean*."""
    result = ratings.copy()
    result[provisional_mask] = (1 - shrinkage) * ratings[provisional_mask] + shrinkage * global_mean
    return result
