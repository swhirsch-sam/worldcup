"""Historical international match results ingestion.

Downloads Mart Jürisoo's public international results dataset from GitHub,
caches to disk with TTL, validates schema.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.ingest.validate import validate_historical

logger = logging.getLogger(__name__)

_HISTORICAL_FILENAME = "historical_results.csv"


def _load_config() -> dict:  # type: ignore[type-arg]
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def fetch_historical(*, refresh: bool = False) -> pd.DataFrame:
    """Return validated historical match results DataFrame.

    Downloads from the URL in config (elo_computation.historical_url)
    on first call; subsequent calls use the disk cache unless refresh=True.

    Returns:
        DataFrame with columns: date, home_team, away_team, home_score,
        away_score, tournament, neutral
    """
    cfg = _load_config()
    cache_path = Path(cfg["data"]["cache_dir"]) / _HISTORICAL_FILENAME

    if not refresh and cache_path.exists():
        logger.info("Loading historical results from cache: %s", cache_path)
        df = _load_and_clean(cache_path)
        validate_historical(df)
        return df

    url: str = cfg["elo_computation"]["historical_url"]
    logger.info("Downloading historical results from %s", url)
    raw = _fetch_with_retry(url, cfg["data"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(raw, encoding="utf-8")
    logger.info("Cached historical results to %s (%d bytes)", cache_path, len(raw))

    df = _load_and_clean(cache_path)
    validate_historical(df)
    logger.info("Loaded %d historical matches", len(df))
    return df


def _load_and_clean(path: Path) -> pd.DataFrame:
    """Parse the CSV and standardize column types."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"home_score": "home_score", "away_score": "away_score"})
    df["neutral"] = (
        df["neutral"].map({"TRUE": True, "FALSE": False, True: True, False: False}).astype(bool)
    )
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # Keep only needed columns; extra columns (city, country) ignored downstream
    return df[
        ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    ].copy()


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
                    f"Failed to fetch {url} after {max_attempts} attempts: {exc}"
                ) from exc
            wait = backoff * (2 ** (attempt - 1))
            logger.warning("Attempt %d failed (%s). Retrying in %.0fs.", attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")  # pragma: no cover
