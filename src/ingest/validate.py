"""Schema validation for ingested data sources.

Each validator raises SchemaValidationError with a clear message on failure.
Callers must not catch these exceptions silently.
"""

from __future__ import annotations

import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when an ingested data frame does not match its expected schema."""


# Expected columns and dtypes per source
_ELO_SCHEMA: dict[str, type] = {
    "team": str,
    "elo_rating": float,
}

_FIFA_SCHEMA: dict[str, type] = {
    "team": str,
    "fifa_points": float,
    "fifa_rank": int,
}

_ODDS_SCHEMA: dict[str, type] = {
    "team": str,
    "implied_probability": float,
}

_HISTORICAL_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"date", "home_team", "away_team", "home_score", "away_score", "neutral"}
)


def validate_elo(df: pd.DataFrame) -> None:
    """Validate Elo ratings data frame."""
    _check_columns(df, _ELO_SCHEMA, "Elo")
    _check_nonempty(df, "Elo")
    if df["elo_rating"].isna().any():
        raise SchemaValidationError("Elo: null values found in elo_rating column.")


def validate_fifa(df: pd.DataFrame) -> None:
    """Validate FIFA rankings data frame."""
    _check_columns(df, _FIFA_SCHEMA, "FIFA")
    _check_nonempty(df, "FIFA")


def validate_odds(df: pd.DataFrame) -> None:
    """Validate market odds data frame."""
    _check_columns(df, _ODDS_SCHEMA, "Odds")
    _check_nonempty(df, "Odds")
    bad = df[(df["implied_probability"] < 0) | (df["implied_probability"] > 1)]
    if not bad.empty:
        raise SchemaValidationError(
            f"Odds: implied_probability out of [0,1] for {bad['team'].tolist()}"
        )


def validate_historical(df: pd.DataFrame) -> None:
    """Validate historical match results data frame."""
    missing = _HISTORICAL_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Historical: missing columns {sorted(missing)}. Got: {sorted(df.columns.tolist())}"
        )
    _check_nonempty(df, "Historical")
    if (df["home_score"] < 0).any() or (df["away_score"] < 0).any():
        raise SchemaValidationError("Historical: negative scoreline found.")


# ---------------------------------------------------------------------------
# Bracket map validation (run at startup)
# ---------------------------------------------------------------------------
def validate_bracket_map(bracket_map: dict, groups: dict) -> None:  # type: ignore[type-arg]
    """Assert that every group label is referenced the correct number of times
    and that no bracket slot is double-filled.

    This is flagged in the README as the top correctness risk.
    """
    # Placeholder: full implementation in Phase 5 once bracket_map.json is confirmed.
    raise NotImplementedError("validate_bracket_map: implement after bracket_map.json is supplied.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _check_columns(df: pd.DataFrame, schema: dict[str, type], source: str) -> None:
    missing = set(schema) - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"{source}: missing expected columns {sorted(missing)}. "
            f"Got: {sorted(df.columns.tolist())}"
        )


def _check_nonempty(df: pd.DataFrame, source: str) -> None:
    if df.empty:
        raise SchemaValidationError(f"{source}: data frame is empty after ingestion.")
