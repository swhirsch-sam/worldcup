"""Precompute per-match model predictions for the Streamlit app.

Closed-form (no Monte Carlo): for every fixed group-stage match we evaluate the
Dixon-Coles scoreline distribution and read off win/draw/loss probabilities,
expected goals, and the most-likely scorelines. We also emit the composite team
strength table and goals-model parameters so the app can predict *any*
hypothetical head-to-head (e.g. a knockout matchup) live, using numpy only.

Strengths are built from the same signal ensemble the Monte Carlo simulation
uses (Elo + FIFA + bookmaker odds + Polymarket, whichever are available), via
the same fetchers — run ``make predict`` in the same session as
``make simulate`` so both read identical cached signals. The signals actually
used are recorded in the output metadata.

Output: ``results/match_predictions.json``.
Run via ``make predict`` or ``python3 -m src.model.match_predict``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.model.scoreline import dixon_coles_matrix, match_lambdas, outcome_probabilities

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "results/match_predictions.json"
GROUPS_PATH = "data/groups.json"


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def _load_groups() -> dict[str, list[str]]:
    with open(GROUPS_PATH) as f:
        data: dict[str, Any] = json.load(f)
    groups: dict[str, list[str]] = data["groups"]
    return groups


def extract_params(cfg: dict[str, Any]) -> dict[str, float]:
    """Pull the goals-model + Dixon-Coles parameters the app needs."""
    goals = cfg["goals_model"]
    return {
        "intercept": float(goals["intercept"]),
        "slope": float(goals["slope"]),
        "rho": float(cfg["dixon_coles"]["rho"]),
        "lambda_floor": float(goals.get("lambda_floor", 0.1)),
        "lambda_cap": float(goals.get("lambda_cap", 6.0)),
    }


def predict_match(
    home: str,
    away: str,
    strength: dict[str, float],
    params: dict[str, float],
) -> dict[str, Any]:
    """Closed-form prediction for one neutral-venue match.

    Returns win/draw/loss probabilities, expected goals, the three most-likely
    scorelines, and a convenience ``favorite`` label.
    """
    lam_h, lam_a = match_lambdas(
        strength[home],
        strength[away],
        intercept=params["intercept"],
        slope=params["slope"],
        lambda_floor=params["lambda_floor"],
        lambda_cap=params["lambda_cap"],
    )
    matrix = dixon_coles_matrix(lam_h, lam_a, params["rho"])
    stats = outcome_probabilities(matrix)

    if stats["p_home"] >= stats["p_away"]:
        favorite, favorite_prob = home, stats["p_home"]
    else:
        favorite, favorite_prob = away, stats["p_away"]

    return {
        "home": home,
        "away": away,
        "p_home": stats["p_home"],
        "p_draw": stats["p_draw"],
        "p_away": stats["p_away"],
        "exp_home": stats["exp_home"],
        "exp_away": stats["exp_away"],
        "top_scores": stats["top_scores"],
        "favorite": favorite,
        "favorite_prob": favorite_prob,
    }


def build_group_matches(
    groups: dict[str, list[str]],
    strength: dict[str, float],
    params: dict[str, float],
) -> list[dict[str, Any]]:
    """All round-robin matches for every group (6 per 4-team group)."""
    matches: list[dict[str, Any]] = []
    for group_id, teams in groups.items():
        for home, away in itertools.combinations(teams, 2):
            match = predict_match(home, away, strength, params)
            match["group"] = group_id
            matches.append(match)
    return matches


def build_payload(
    strength_df: pd.DataFrame,
    groups: dict[str, list[str]],
    cfg: dict[str, Any],
    *,
    signals_used: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the full match-predictions payload (pure; no I/O)."""
    params = extract_params(cfg)
    strength: dict[str, float] = dict(
        zip(strength_df["team"], strength_df["strength"].astype(float), strict=False)
    )

    return {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": "dixon_coles",
            "signals_used": signals_used or ["elo"],
            **params,
        },
        "model_params": params,
        "team_strength": {team: round(value, 1) for team, value in strength.items()},
        "groups": groups,
        "group_matches": build_group_matches(groups, strength, params),
    }


def build_and_save(*, output_path: str | None = None) -> dict[str, Any]:
    """Build strengths from the full signal ensemble and write the JSON.

    Mirrors ``strength.build_and_save``: same fetchers, same
    ``build_strength_table`` call, so match predictions share their strength
    source with the Monte Carlo simulation. Signals that fall back to None are
    omitted from ``metadata.signals_used``.
    """
    from src.ingest.elo import fetch_elo
    from src.ingest.fifa import fetch_fifa
    from src.ingest.odds import fetch_odds
    from src.ingest.polymarket import fetch_polymarket
    from src.model.strength import build_strength_table

    cfg = _load_config()
    groups = _load_groups()

    elo_df = fetch_elo()
    fifa_df = fetch_fifa()
    odds_df = fetch_odds()
    polymarket_df = fetch_polymarket()

    strength_df, fallbacks = build_strength_table(
        elo_df, fifa_df, odds_df, polymarket_df, cfg=cfg, group_stage=True
    )
    for msg in fallbacks:
        logger.info("Strength fallback: %s", msg)

    signals_used = ["elo"] + [
        name
        for name, df in (("fifa", fifa_df), ("market", odds_df), ("polymarket", polymarket_df))
        if df is not None
    ]

    payload = build_payload(strength_df, groups, cfg, signals_used=signals_used)

    out = Path(output_path or cfg.get("output", {}).get("match_predictions", DEFAULT_OUTPUT))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(
        "Wrote %d match predictions to %s (signals: %s)",
        len(payload["group_matches"]),
        out,
        ", ".join(signals_used),
    )

    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Precompute per-match predictions.")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = build_and_save(output_path=args.output)
    matches = result["group_matches"]
    print(f"\nGenerated {len(matches)} group-stage match predictions.\n")
    sample = matches[0]
    top = sample["top_scores"][0]
    print(
        f"  e.g. {sample['home']} vs {sample['away']} (Group {sample['group']}): "
        f"{sample['p_home']:.0%} / {sample['p_draw']:.0%} / {sample['p_away']:.0%}  "
        f"— likely {top['home']}-{top['away']}"
    )
