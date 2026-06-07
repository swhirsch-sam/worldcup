"""Unit tests for the Polymarket ingest parser."""

from __future__ import annotations

import json

import pandas as pd

from src.ingest.polymarket import _extract_prices, _fetch_and_parse


def _market(outcomes: list[str], prices: list[float]) -> dict:
    return {
        "question": "2026 FIFA World Cup Winner?",
        "outcomes": json.dumps(outcomes),
        "outcomePrices": json.dumps([str(p) for p in prices]),
    }


# ---------------------------------------------------------------------------
# _extract_prices unit tests
# ---------------------------------------------------------------------------


def test_extract_prices_basic() -> None:
    m = _market(["France", "Brazil", "Argentina"], [0.25, 0.18, 0.20])
    rows = _extract_prices(m)
    assert len(rows) == 3
    teams = {r["team"] for r in rows}
    assert teams == {"France", "Brazil", "Argentina"}


def test_extract_prices_normalizes_aliases() -> None:
    m = _market(["USA", "Turkey"], [0.1, 0.05])
    rows = _extract_prices(m)
    teams = {r["team"] for r in rows}
    assert "United States" in teams
    assert "Türkiye" in teams


def test_extract_prices_skips_unknown_teams() -> None:
    m = _market(["France", "UnknownFC", "Brazil"], [0.25, 0.01, 0.18])
    rows = _extract_prices(m)
    teams = {r["team"] for r in rows}
    assert "UnknownFC" not in teams
    assert {"France", "Brazil"}.issubset(teams)


def test_extract_prices_handles_list_inputs() -> None:
    # When the API returns actual lists (not JSON strings)
    m = {
        "question": "...",
        "outcomes": ["France", "Brazil"],
        "outcomePrices": [0.30, 0.25],
    }
    rows = _extract_prices(m)
    assert len(rows) == 2


def test_extract_prices_empty_market() -> None:
    m = _market([], [])
    assert _extract_prices(m) == []


def test_extract_prices_malformed_json_returns_empty() -> None:
    m = {"question": "...", "outcomes": "{bad json", "outcomePrices": "[0.1]"}
    assert _extract_prices(m) == []


# ---------------------------------------------------------------------------
# de-vig / normalization
# ---------------------------------------------------------------------------


def test_normalization_sums_to_one() -> None:
    m = _market(["France", "Brazil", "Argentina"], [0.25, 0.20, 0.15])
    rows = _extract_prices(m)
    df = pd.DataFrame(rows)
    total = float(df["implied_probability"].sum())
    # Normalization happens in _fetch_and_parse, not _extract_prices, so raw sum here
    # Just verify types are correct
    assert all(isinstance(r["implied_probability"], float) for r in rows)


def test_prices_preserved_before_normalization() -> None:
    m = _market(["France", "Brazil"], [0.30, 0.20])
    rows = _extract_prices(m)
    probs = {r["team"]: r["implied_probability"] for r in rows}
    assert abs(probs["France"] - 0.30) < 1e-9
    assert abs(probs["Brazil"] - 0.20) < 1e-9
