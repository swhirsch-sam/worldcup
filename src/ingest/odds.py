"""Market odds ingestion via The Odds API.

Optional signal gated by config. De-vigs implied probabilities using the
multiplicative (proportional) method. Falls back gracefully when disabled
or when ODDS_API_KEY is not set.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.ingest.names import resolve
from src.ingest.validate import validate_odds

logger = logging.getLogger(__name__)


def _load_config() -> dict:  # type: ignore[type-arg]
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def fetch_odds(*, refresh: bool = False) -> pd.DataFrame | None:
    """Return de-vigged implied probability DataFrame, or None if disabled.

    Returns:
        DataFrame with columns: team, implied_probability
        None if odds are disabled or API key is absent (caller records fallback).
    """
    cfg = _load_config()
    if not cfg["data"]["market"]["enabled"]:
        logger.info("Market odds disabled in config; skipping.")
        return None

    odds_cfg = cfg["data"]["market"]
    api_key = os.getenv(odds_cfg["api_key_env"])
    if not api_key:
        logger.warning(
            "Env var %s not set; market odds unavailable. Falling back to Elo-only.",
            odds_cfg["api_key_env"],
        )
        return None

    cache_path = Path(cfg["data"]["cache_dir"]) / odds_cfg["filename"]

    if not refresh and cache_path.exists():
        logger.info("Loading market odds from cache: %s", cache_path)
        df = pd.read_json(cache_path)
        validate_odds(df)
        df["team"] = df["team"].map(resolve)
        return df

    url = f"{odds_cfg['url']}?apiKey={api_key}&regions=eu&markets=outrights"
    logger.info("Fetching market odds from The Odds API")
    raw = _fetch_with_retry(url, cfg["data"])
    df = _parse_and_devige(raw)
    validate_odds(df)
    df["team"] = df["team"].map(resolve)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(cache_path, orient="records")
    logger.info("Saved market odds to %s", cache_path)
    return df


def _parse_and_devige(raw: str) -> pd.DataFrame:
    """Parse Odds API JSON and remove bookmaker overround via multiplicative method.

    The Odds API outrights endpoint returns a list of events, each with a list of
    bookmakers, each with an outrights market whose outcomes carry decimal odds.
    We average raw implied probs across bookmakers then normalise to sum to 1.

    De-vigging: p_fair_i = mean_raw_i / sum(mean_raw_j)
    """
    import json as _json

    data = _json.loads(raw)

    # Aggregate 1/decimal_odds per team across all bookmakers
    team_raw: dict[str, list[float]] = {}
    for event in data:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    name: str = str(outcome.get("name", ""))
                    price = float(outcome.get("price", 0))
                    if price > 1.0:  # decimal odds only
                        team_raw.setdefault(name, []).append(1.0 / price)

    if not team_raw:
        raise ValueError("No outright outcomes found in Odds API response.")

    rows = [
        {"team": team, "implied_probability": sum(ps) / len(ps)} for team, ps in team_raw.items()
    ]
    df = pd.DataFrame(rows)

    # Multiplicative de-vig: normalise to sum to 1
    total = float(df["implied_probability"].sum())
    if total < 1e-9:
        raise ValueError("Sum of implied probabilities is zero after de-vigging.")
    df["implied_probability"] = df["implied_probability"] / total
    return df


def _fetch_with_retry(url: str, data_cfg: dict) -> str:  # type: ignore[type-arg]
    max_attempts: int = data_cfg["retry_max_attempts"]
    backoff: float = data_cfg["retry_backoff_base_seconds"]
    timeout: int = data_cfg["request_timeout_seconds"]

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Failed to fetch odds after {max_attempts} attempts: {exc}"
                ) from exc
            wait = backoff * (2 ** (attempt - 1))
            logger.warning("Attempt %d failed (%s). Retrying in %.0fs.", attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")  # pragma: no cover


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest market odds")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch ignoring cache")
    args = parser.parse_args()
    result = fetch_odds(refresh=args.refresh)
    if result is not None:
        print(result.head(20).to_string(index=False))
