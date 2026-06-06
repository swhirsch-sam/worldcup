"""FIFA World Rankings ingestion.

Optional secondary signal; disabled falls back to Elo-only with a warning
recorded in the run manifest.
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
from src.ingest.validate import validate_fifa

logger = logging.getLogger(__name__)


def _load_config() -> dict:  # type: ignore[type-arg]
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def fetch_fifa(*, refresh: bool = False) -> pd.DataFrame | None:
    """Return validated FIFA rankings DataFrame, or None if disabled.

    Returns:
        DataFrame with columns: team, fifa_points, fifa_rank
        None if fifa is disabled in config (caller records fallback).
    """
    cfg = _load_config()
    if not cfg["data"]["fifa"]["enabled"]:
        logger.warning("FIFA rankings disabled in config; skipping.")
        return None

    fifa_cfg = cfg["data"]["fifa"]
    cache_path = Path(cfg["data"]["cache_dir"]) / fifa_cfg["filename"]

    if not refresh and cache_path.exists():
        logger.info("Loading FIFA rankings from cache: %s", cache_path)
        df = pd.read_json(cache_path)
        validate_fifa(df)
        df["team"] = df["team"].map(resolve)
        return df

    logger.info("Fetching FIFA rankings from %s", fifa_cfg["url"])
    try:
        raw = _fetch_with_retry(fifa_cfg["url"], cfg["data"])
        df = _parse_fifa_json(raw)
        validate_fifa(df)
        df["team"] = df["team"].map(resolve)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(cache_path, orient="records")
        logger.info("Saved FIFA rankings to %s", cache_path)
        return df
    except Exception as exc:
        logger.warning(
            "FIFA rankings fetch/parse failed (%s). Signal unavailable; "
            "Elo-only ensemble will be used.",
            exc,
        )
        return None


def _parse_fifa_json(raw: str) -> pd.DataFrame:
    """Parse the FIFA rankings JSON response into a clean DataFrame."""
    raise NotImplementedError("_parse_fifa_json: implement in Phase 2 after confirming API shape.")


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest FIFA rankings")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch ignoring cache")
    args = parser.parse_args()
    result = fetch_fifa(refresh=args.refresh)
    if result is not None:
        print(result.head(20).to_string(index=False))
