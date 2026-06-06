"""World Football Elo ratings ingestion.

Fetches from eloratings.net TSV with retry/backoff, caches to disk,
validates schema, and resolves team names to canonical form.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.ingest.names import resolve
from src.ingest.validate import validate_elo

logger = logging.getLogger(__name__)


def _load_config() -> dict:  # type: ignore[type-arg]
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def fetch_elo(*, refresh: bool = False) -> pd.DataFrame:
    """Return a validated Elo ratings DataFrame with canonical team names.

    Args:
        refresh: Ignore disk cache and re-fetch from the network.

    Returns:
        DataFrame with columns: team, elo_rating
    """
    cfg = _load_config()
    elo_cfg = cfg["data"]["elo"]
    cache_path = Path(elo_cfg["local_csv"] or f"data/raw/{elo_cfg['filename']}")

    if not refresh and cache_path.exists():
        logger.info("Loading Elo ratings from cache: %s", cache_path)
        df = pd.read_csv(cache_path)
        validate_elo(df)
        df["team"] = df["team"].map(resolve)
        return df

    logger.info("Fetching Elo ratings from %s", elo_cfg["url"])
    raw = _fetch_with_retry(elo_cfg["url"], cfg["data"])
    df = _parse_elo_tsv(raw)
    validate_elo(df)
    df["team"] = df["team"].map(resolve)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Saved Elo ratings to %s", cache_path)
    return df


def _parse_elo_tsv(raw: str) -> pd.DataFrame:
    """Parse the eloratings.net TSV response into a clean DataFrame."""
    raise NotImplementedError("_parse_elo_tsv: implement in Phase 2 after confirming TSV format.")


def _fetch_with_retry(url: str, data_cfg: dict) -> str:  # type: ignore[type-arg]
    """HTTP GET with exponential backoff retry."""
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest Elo ratings")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch ignoring cache")
    args = parser.parse_args()
    df = fetch_elo(refresh=args.refresh)
    print(df.head(20).to_string(index=False))
