"""Unit tests for The Odds API parser and de-vigging logic."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.ingest.odds import _parse_and_devige


def _make_odds_response(
    bookmakers: list[dict],
) -> str:
    return json.dumps(
        [
            {
                "id": "abc123",
                "sport_key": "soccer_world_cup_winner",
                "bookmakers": bookmakers,
            }
        ]
    )


def _bm(outcomes: list[tuple[str, float]]) -> dict:
    return {
        "key": "pinnacle",
        "title": "Pinnacle",
        "markets": [
            {
                "key": "outrights",
                "outcomes": [{"name": t, "price": p} for t, p in outcomes],
            }
        ],
    }


def test_sums_to_one() -> None:
    raw = _make_odds_response([_bm([("France", 5.0), ("Argentina", 6.0), ("Brazil", 8.0)])])
    df = _parse_and_devige(raw)
    assert abs(df["implied_probability"].sum() - 1.0) < 1e-9


def test_correct_teams_returned() -> None:
    raw = _make_odds_response([_bm([("France", 5.0), ("Argentina", 6.0), ("Brazil", 8.0)])])
    df = _parse_and_devige(raw)
    assert set(df["team"]) == {"France", "Argentina", "Brazil"}


def test_higher_odds_lower_probability() -> None:
    # France at 5.0 (shorter odds) should have higher implied prob than Brazil at 8.0
    raw = _make_odds_response([_bm([("France", 5.0), ("Brazil", 8.0)])])
    df = _parse_and_devige(raw).set_index("team")
    assert df.loc["France", "implied_probability"] > df.loc["Brazil", "implied_probability"]


def test_averages_across_bookmakers() -> None:
    # Two bookmakers disagree on France: 5.0 and 4.0
    raw = _make_odds_response(
        [
            _bm([("France", 5.0), ("Brazil", 8.0)]),
            _bm([("France", 4.0), ("Brazil", 8.0)]),
        ]
    )
    df = _parse_and_devige(raw).set_index("team")
    # Raw average for France: (1/5 + 1/4) / 2 = 0.225
    # Raw average for Brazil: (1/8 + 1/8) / 2 = 0.125
    # After normalising: France = 0.225/0.35 ≈ 0.643
    expected = 0.225 / (0.225 + 0.125)
    assert abs(df.loc["France", "implied_probability"] - expected) < 1e-9


def test_empty_response_raises() -> None:
    with pytest.raises(ValueError, match="No outright outcomes"):
        _parse_and_devige(json.dumps([{"bookmakers": []}]))


def test_non_outright_market_ignored() -> None:
    raw = json.dumps(
        [
            {
                "id": "x",
                "bookmakers": [
                    {
                        "key": "bwin",
                        "markets": [
                            {
                                "key": "h2h",  # not outrights
                                "outcomes": [{"name": "France", "price": 1.5}],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    with pytest.raises(ValueError, match="No outright outcomes"):
        _parse_and_devige(raw)


def test_output_schema() -> None:
    raw = _make_odds_response([_bm([("France", 5.0), ("Brazil", 8.0)])])
    df = _parse_and_devige(raw)
    assert isinstance(df, pd.DataFrame)
    assert {"team", "implied_probability"}.issubset(df.columns)
    assert (df["implied_probability"] >= 0).all()
    assert (df["implied_probability"] <= 1).all()
