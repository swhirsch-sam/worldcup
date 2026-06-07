"""Polymarket prediction market ingestion for WC 2026 winner odds.

Uses the public Gamma API (gamma-api.polymarket.com) — no API key required.
Multi-outcome markets encode team names in `outcomes` and prices in
`outcomePrices`, both as JSON-encoded strings inside the market object.

Falls back gracefully when the market is not found or the network is
unavailable; the strength ensemble simply drops the polymarket weight.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from src.ingest.names import NameResolutionError, resolve
from src.ingest.validate import validate_odds

logger = logging.getLogger(__name__)

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_SEARCH_TERMS = [
    "2026 FIFA World Cup winner",
    "FIFA World Cup 2026",
    "World Cup 2026",
]


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def fetch_polymarket(*, refresh: bool = False) -> pd.DataFrame | None:
    """Return de-vigged implied probabilities from Polymarket WC 2026 winner market.

    Returns:
        DataFrame with columns [team, implied_probability], or None if disabled,
        market not found, or API unavailable (caller records the fallback).
    """
    cfg = _load_config()
    pm_cfg: dict[str, Any] = cfg.get("data", {}).get("polymarket", {})

    if not pm_cfg.get("enabled", False):
        logger.info("Polymarket disabled in config; skipping.")
        return None

    cache_path = Path(cfg["data"]["cache_dir"]) / pm_cfg.get(
        "filename", "polymarket_odds.json"
    )

    if not refresh and cache_path.exists():
        logger.info("Loading Polymarket odds from cache: %s", cache_path)
        df = pd.read_json(cache_path)
        validate_odds(df)
        return df

    logger.info("Fetching Polymarket WC 2026 winner odds")
    try:
        df = _fetch_and_parse(cfg["data"])
    except Exception as exc:
        logger.warning("Polymarket fetch failed (%s); signal unavailable.", exc)
        return None

    if df is None or df.empty:
        logger.warning("Polymarket: no data extracted from market; signal unavailable.")
        return None

    validate_odds(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(cache_path, orient="records")
    logger.info("Saved Polymarket odds to %s (n=%d teams)", cache_path, len(df))
    return df


def _fetch_and_parse(data_cfg: dict[str, Any]) -> pd.DataFrame | None:
    timeout: int = int(data_cfg.get("request_timeout_seconds", 15))
    max_attempts: int = int(data_cfg.get("retry_max_attempts", 4))
    backoff: float = float(data_cfg.get("retry_backoff_base_seconds", 2.0))

    market = _find_wc_market(timeout, max_attempts, backoff)
    if market is None:
        return None

    rows = _extract_prices(market)
    if not rows:
        logger.warning("Polymarket: no valid team prices in market.")
        return None

    df = pd.DataFrame(rows)
    total = float(df["implied_probability"].sum())
    if total < 1e-9:
        return None
    df["implied_probability"] = df["implied_probability"] / total
    return df


def _find_wc_market(
    timeout: int, max_attempts: int, backoff: float
) -> dict[str, Any] | None:
    """Search Gamma API for the active 2026 WC winner multi-outcome market."""
    for term in _SEARCH_TERMS:
        url = f"{_GAMMA_BASE}/markets"
        params: dict[str, str] = {
            "q": term,
            "active": "true",
            "closed": "false",
            "limit": "20",
        }
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                candidates: list[dict[str, Any]] = resp.json()
                for m in candidates:
                    q = m.get("question", "").lower()
                    slug = m.get("slug", "").lower()
                    if ("world cup" in q or "world-cup" in slug) and any(
                        kw in q for kw in ("winner", "win", "champion")
                    ):
                        logger.info(
                            "Found Polymarket WC market: %r", m.get("question")
                        )
                        return m
                break  # response received, no match for this term
            except requests.RequestException as exc:
                if attempt == max_attempts:
                    logger.debug("Polymarket request failed for %r: %s", term, exc)
                    break
                time.sleep(backoff * (2 ** (attempt - 1)))

    logger.warning("Polymarket: no active 2026 WC winner market found.")
    return None


def _extract_prices(market: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse outcomes and outcomePrices from a Gamma API market object.

    Both fields arrive as JSON-encoded strings (e.g. '["France","Brazil"]').
    """
    outcomes_raw = market.get("outcomes", "[]")
    prices_raw = market.get("outcomePrices", "[]")

    if isinstance(outcomes_raw, str):
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except json.JSONDecodeError:
            return []
    if isinstance(prices_raw, str):
        try:
            prices_raw = json.loads(prices_raw)
        except json.JSONDecodeError:
            return []

    rows: list[dict[str, Any]] = []
    for outcome, price_val in zip(outcomes_raw, prices_raw, strict=False):
        try:
            prob = float(price_val)
            canonical = resolve(str(outcome))
            rows.append({"team": canonical, "implied_probability": prob})
        except (ValueError, TypeError, NameResolutionError) as exc:
            logger.debug("Polymarket: skipping outcome %r (%s)", outcome, exc)
    return rows


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest Polymarket WC 2026 odds")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = fetch_polymarket(refresh=args.refresh)
    if result is not None:
        print(result.to_string(index=False))
    else:
        print("No Polymarket data available.")
