"""World Football Elo ratings ingestion and computation.

Primary path: fetch TSV from eloratings.net (fast, authoritative).
Fallback path: compute Elo iteratively from full historical results using
the eloratings.net methodology (same K-factors, goal-weight formula, and
home-advantage convention).  The fallback is triggered automatically when
the primary URL is unreachable; the choice is recorded in the run manifest.

K-factors and home-advantage values come from config.yaml so they are
adjustable without touching code.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from src.ingest.historical import fetch_historical
from src.ingest.names import _ALIAS_MAP, CANONICAL_TEAMS
from src.ingest.validate import SchemaValidationError, validate_elo

logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# K-factor tournament classification
# Follows eloratings.net methodology:
#   World Cup (tournament): 60  Continental championships: 50
#   WC qualifiers / major qualifiers / Nations Leagues: 40
#   Other competitive: 30   Friendlies: 20
# ---------------------------------------------------------------------------
def _k_factor(tournament: str, cfg: dict[str, Any]) -> float:
    """Return the K-factor for a given tournament string."""
    kf = cfg["elo_computation"]["k_factors"]
    t = tournament.lower()

    # World Cup final tournament (not qualifiers)
    if "world cup" in t and "qual" not in t and "conifa" not in t:
        return float(kf["world_cup_final"])

    # Continental championships (final tournament only)
    continental_keywords = [
        "uefa euro",
        "european championship",
        "copa america",
        "copa américa",
        "african cup of nations",
        "africa cup of nations",
        "afc asian cup",
        "gold cup",
        "oceania nations cup",
        "ofc nations cup",
        "concacaf championship",
        "confederations cup",
        "nations cup",
    ]
    if any(kw in t for kw in continental_keywords) and "qual" not in t:
        return float(kf["continental_championship"])

    # Qualifiers and competitive league tournaments
    qualifier_keywords = [
        "qual",
        "nations league",
        "concacaf series",
        "fifa series",
    ]
    if any(kw in t for kw in qualifier_keywords):
        return float(kf["world_cup_qualifier"])

    # Friendlies
    if "friendly" in t:
        return float(kf["friendly"])

    # All other official matches
    return float(kf["other_official"])


def _goal_weight(goal_diff: int) -> float:
    """eloratings.net goal-difference multiplier G."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11.0 + goal_diff) / 8.0


# ---------------------------------------------------------------------------
# Core Elo computation
# ---------------------------------------------------------------------------
def compute_elo_from_historical(
    hist_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute current Elo ratings and match counts from all historical results.

    Iterates over every match in chronological order, applying the
    eloratings.net update formula.

    Args:
        hist_df: Full historical results (all matches, not filtered by date).
        cfg: Parsed config dict.

    Returns:
        (ratings, match_counts): dicts keyed by raw team name as it
        appears in the historical dataset.
    """
    ec = cfg["elo_computation"]
    home_adv: float = float(ec["home_advantage_elo"])
    start: float = float(ec["starting_rating"])

    ratings: dict[str, float] = {}
    match_counts: dict[str, int] = {}

    df = hist_df.sort_values("date").reset_index(drop=True)

    for row in df.itertuples(index=False):
        home: str = row.home_team
        away: str = row.away_team

        r_h = ratings.get(home, start)
        r_a = ratings.get(away, start)

        adv = 0.0 if row.neutral else home_adv
        e_h = 1.0 / (1.0 + 10.0 ** (-(r_h + adv - r_a) / 400.0))

        if row.home_score > row.away_score:
            a_h = 1.0
        elif row.home_score == row.away_score:
            a_h = 0.5
        else:
            a_h = 0.0

        k = _k_factor(row.tournament, cfg)
        g = _goal_weight(abs(row.home_score - row.away_score))
        delta_h = k * g * (a_h - e_h)

        ratings[home] = r_h + delta_h
        ratings[away] = r_a - delta_h  # symmetric update

        match_counts[home] = match_counts.get(home, 0) + 1
        match_counts[away] = match_counts.get(away, 0) + 1

    logger.info("Elo computation complete. %d unique teams rated.", len(ratings))
    return ratings, match_counts


def compute_elo_with_records(
    hist_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[dict[str, float], dict[str, int], pd.DataFrame]:
    """Compute Elo ratings while saving pre-match ratings for each match.

    Iterates over every match in chronological order, records the pre-match
    Elo for both teams, then applies the same update as
    compute_elo_from_historical().

    Args:
        hist_df: Full historical results (all matches, not filtered by date).
        cfg: Parsed config dict.

    Returns:
        (ratings, match_counts, match_records_df) where match_records_df has
        columns: date (datetime), home_team, away_team, home_score, away_score,
        home_elo, away_elo, neutral (bool), tournament.
    """
    ec = cfg["elo_computation"]
    home_adv: float = float(ec["home_advantage_elo"])
    start: float = float(ec["starting_rating"])

    ratings: dict[str, float] = {}
    match_counts: dict[str, int] = {}

    df = hist_df.sort_values("date").reset_index(drop=True)

    records: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        home: str = row.home_team
        away: str = row.away_team

        r_h = ratings.get(home, start)
        r_a = ratings.get(away, start)

        # Save pre-match ratings
        records.append(
            {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "home_score": row.home_score,
                "away_score": row.away_score,
                "home_elo": r_h,
                "away_elo": r_a,
                "neutral": bool(row.neutral),
                "tournament": row.tournament,
            }
        )

        adv = 0.0 if row.neutral else home_adv
        e_h = 1.0 / (1.0 + 10.0 ** (-(r_h + adv - r_a) / 400.0))

        if row.home_score > row.away_score:
            a_h = 1.0
        elif row.home_score == row.away_score:
            a_h = 0.5
        else:
            a_h = 0.0

        k = _k_factor(row.tournament, cfg)
        g = _goal_weight(abs(row.home_score - row.away_score))
        delta_h = k * g * (a_h - e_h)

        ratings[home] = r_h + delta_h
        ratings[away] = r_a - delta_h  # symmetric update

        match_counts[home] = match_counts.get(home, 0) + 1
        match_counts[away] = match_counts.get(away, 0) + 1

    match_records_df = pd.DataFrame(records)
    logger.info(
        "Elo-with-records complete. %d unique teams, %d match records.",
        len(ratings),
        len(match_records_df),
    )
    return ratings, match_counts, match_records_df


def ratings_to_dataframe(
    ratings: dict[str, float],
    match_counts: dict[str, int],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Convert raw rating dict to a validated DataFrame for the 48 WC teams.

    For each canonical WC team, tries all known aliases to find the team's
    entry in the historical ratings dict. Teams not found in history receive
    the global mean (provisional).
    """
    global_mean: float = float(cfg["ensemble"]["global_mean_rating"])
    threshold: int = int(cfg["ensemble"]["provisional_cap_threshold"])

    # Build reverse alias lookup: canonical → list of source names
    reverse: dict[str, list[str]] = {}
    for src, canon in _ALIAS_MAP.items():
        if canon in CANONICAL_TEAMS:
            reverse.setdefault(canon, []).append(src)
    # Always include the canonical name itself
    for canon in CANONICAL_TEAMS:
        reverse.setdefault(canon, []).append(canon)

    rows = []
    for canon in sorted(CANONICAL_TEAMS):
        # Try all aliases to find the rating
        rating = global_mean
        count = 0
        for src_name in reverse.get(canon, [canon]):
            if src_name in ratings:
                rating = ratings[src_name]
                count = match_counts.get(src_name, 0)
                break

        rows.append(
            {
                "team": canon,
                "elo_rating": round(rating, 1),
                "match_count": count,
                "provisional": count < threshold,
            }
        )

    df = pd.DataFrame(rows).sort_values("elo_rating", ascending=False).reset_index(drop=True)
    validate_elo(df)
    return df


# ---------------------------------------------------------------------------
# Public fetch entry point
# ---------------------------------------------------------------------------
def fetch_elo(*, refresh: bool = False) -> pd.DataFrame:
    """Return a validated Elo ratings DataFrame for all 48 WC teams.

    Tries eloratings.net TSV first; falls back to computing from the
    historical results dataset.  Fallback is logged at WARNING level and
    will be recorded in the run manifest.

    Args:
        refresh: Ignore disk cache and re-fetch/recompute.

    Returns:
        DataFrame with columns: team, elo_rating, match_count, provisional
    """
    cfg = _load_config()
    elo_cfg = cfg["data"]["elo"]
    cache_path = Path(cfg["data"]["cache_dir"]) / elo_cfg["filename"]

    if not refresh and cache_path.exists():
        logger.info("Loading Elo ratings from cache: %s", cache_path)
        df = pd.read_csv(cache_path)
        df["provisional"] = df["provisional"].astype(bool)
        validate_elo(df)
        return df

    # --- Primary: try eloratings.net ---
    try:
        logger.info("Attempting to fetch Elo ratings from eloratings.net...")
        raw = _fetch_with_retry(elo_cfg["url"], cfg["data"])
        df = _parse_eloratings_tsv(raw, cfg)
        validate_elo(df)
        logger.info("Successfully fetched %d Elo ratings from eloratings.net.", len(df))
    except Exception as exc:
        logger.warning(
            "eloratings.net fetch failed (%s). Falling back to historical computation.",
            exc,
        )
        df = _compute_from_history(cfg)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Saved Elo ratings to %s", cache_path)
    return df


def _compute_from_history(cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute Elo ratings from the full historical results dataset."""
    logger.info("Computing Elo from historical results (fallback path)...")
    hist_df = fetch_historical()
    ratings, match_counts = compute_elo_from_historical(hist_df, cfg)
    return ratings_to_dataframe(ratings, match_counts, cfg)


def _parse_eloratings_tsv(raw: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Parse eloratings.net World.tsv into a clean DataFrame.

    The TSV has no header; columns are (by community documentation):
      0: rank  1: country_code  2: team_name  3: elo_rating  4: ...
    """
    lines = [line for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        raise SchemaValidationError("eloratings.net returned an empty file.")

    rows = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            # Try common column layouts: rank, name, elo  OR  rank, code, name, elo
            if len(parts) >= 4:
                team_name = parts[2].strip()
                elo = float(parts[3].strip())
            else:
                team_name = parts[1].strip()
                elo = float(parts[2].strip())
            rows.append({"team": team_name, "elo_rating": elo})
        except (ValueError, IndexError):
            continue

    if not rows:
        raise SchemaValidationError("Could not parse any rows from eloratings.net TSV.")

    from src.ingest.names import NameResolutionError, resolve

    result = []
    for r in rows:
        try:
            canon = resolve(str(r["team"]))
            if canon in CANONICAL_TEAMS:
                result.append(
                    {
                        "team": canon,
                        "elo_rating": r["elo_rating"],
                        "match_count": 0,
                        "provisional": False,
                    }
                )
        except NameResolutionError:
            pass  # Not a WC team

    return pd.DataFrame(result).drop_duplicates(subset="team")


def _fetch_with_retry(url: str, data_cfg: dict[str, Any]) -> str:
    max_attempts: int = data_cfg["retry_max_attempts"]
    backoff: float = float(data_cfg["retry_backoff_base_seconds"])
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ingest Elo ratings")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch/recompute")
    args = parser.parse_args()
    result = fetch_elo(refresh=args.refresh)
    print(f"\nTop 20 Elo ratings ({len(result)} total):\n")
    print(result.head(20).to_string(index=False))
