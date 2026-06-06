"""Team strength computation: weighted ensemble of Elo, FIFA, and market signals.

Each signal is z-score normalized against the 48 WC teams, then blended with
config weights, then re-expressed on the Elo scale (mean~1500, std~Elo std).
Missing signals trigger automatic weight renormalization and a fallback warning.

Host teams receive an additive Elo bump (group-stage only, from config).
Provisional teams (few caps) are shrunk toward the global mean.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.ingest.names import CANONICAL_TEAMS

logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def build_strength_table(
    elo_df: pd.DataFrame,
    fifa_df: pd.DataFrame | None,
    odds_df: pd.DataFrame | None,
    *,
    cfg: dict[str, Any] | None = None,
    group_stage: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build composite team strength ratings for all 48 WC teams.

    Args:
        elo_df: Validated Elo DataFrame (team, elo_rating, match_count, provisional).
        fifa_df: Validated FIFA DataFrame or None (team, fifa_points, fifa_rank).
        odds_df: Validated market odds DataFrame or None (team, implied_probability).
        cfg: Parsed config dict; loaded from disk if None.
        group_stage: Whether host bump applies (True = group stage, False = KO).

    Returns:
        (strength_df, fallback_log): DataFrame sorted by strength desc; list of
        any fallback messages recorded for the run manifest.
    """
    if cfg is None:
        cfg = _load_config()

    fallback_log: list[str] = []
    global_mean: float = float(cfg["ensemble"]["global_mean_rating"])

    # --- 1. Base table: one row per canonical WC team ---
    df = pd.DataFrame({"team": sorted(CANONICAL_TEAMS)})

    # --- 2. Merge available signals ---
    df = df.merge(
        elo_df[["team", "elo_rating", "match_count", "provisional"]],
        on="team",
        how="left",
    )
    df["elo_rating"] = df["elo_rating"].fillna(global_mean)
    df["match_count"] = df["match_count"].fillna(0).astype(int)
    df["provisional"] = df["provisional"].fillna(True).astype(bool)

    available: set[str] = {"elo"}

    if fifa_df is not None:
        df = df.merge(fifa_df[["team", "fifa_points"]], on="team", how="left")
        available.add("fifa")
    else:
        fallback_log.append("FIFA rankings unavailable; Elo-only ensemble used.")

    if odds_df is not None:
        df = df.merge(odds_df[["team", "implied_probability"]], on="team", how="left")
        available.add("market")
    else:
        fallback_log.append("Market odds unavailable; Elo-only ensemble used.")

    if fallback_log:
        for msg in fallback_log:
            logger.warning(msg)

    # --- 3. Renormalize weights to available signals ---
    raw_weights: dict[str, float] = cfg["ensemble"]["weights"]
    active_weights = _renormalize_weights(raw_weights, available)
    logger.info(
        "Ensemble weights (after renormalization): %s",
        {k: f"{v:.3f}" for k, v in active_weights.items()},
    )

    # --- 4. Z-score blend (each signal standardized against WC-team pool) ---
    elo_std = float(df["elo_rating"].std())
    if elo_std < 1e-6:
        elo_std = 1.0  # degenerate guard

    elo_mean = float(df["elo_rating"].mean())
    blended_z: pd.Series = active_weights["elo"] * (df["elo_rating"] - elo_mean) / elo_std

    if "fifa" in active_weights and "fifa_points" in df.columns:
        fifa_vals = df["fifa_points"].fillna(df["fifa_points"].mean())
        fmean, fstd = float(fifa_vals.mean()), float(fifa_vals.std())
        if fstd < 1e-6:
            fstd = 1.0
        blended_z = blended_z + active_weights["fifa"] * (fifa_vals - fmean) / fstd

    if "market" in active_weights and "implied_probability" in df.columns:
        mkt_vals = df["implied_probability"].fillna(df["implied_probability"].mean())
        mmean, mstd = float(mkt_vals.mean()), float(mkt_vals.std())
        if mstd < 1e-6:
            mstd = 1.0
        blended_z = blended_z + active_weights["market"] * (mkt_vals - mmean) / mstd

    # Re-express on Elo scale: mean = global_mean, std = Elo std
    df["strength"] = global_mean + blended_z * elo_std

    # --- 5. Shrink provisional teams toward global mean ---
    shrinkage: float = float(cfg["ensemble"]["provisional_shrinkage"])
    mask = df["provisional"].astype(bool)
    df.loc[mask, "strength"] = (1.0 - shrinkage) * df.loc[
        mask, "strength"
    ] + shrinkage * global_mean
    if mask.any():
        logger.info(
            "Applied %.0f%% shrinkage to %d provisional teams.",
            shrinkage * 100,
            int(mask.sum()),
        )

    # --- 6. Host bump (group stage only) ---
    host_bump: float = float(cfg["host"]["elo_bump"])
    group_stage_only: bool = bool(cfg["host"].get("group_stage_only", True))
    hosts: list[str] = list(cfg["host"]["teams"])
    if group_stage or not group_stage_only:
        for host in hosts:
            df.loc[df["team"] == host, "strength"] += host_bump
        logger.info("Applied %.0f Elo host bump to: %s", host_bump, hosts)

    # --- 7. Sort and round ---
    df = df.sort_values("strength", ascending=False).reset_index(drop=True)
    df["strength"] = df["strength"].round(1)

    return df, fallback_log


def _renormalize_weights(
    weights: dict[str, float],
    available: set[str],
) -> dict[str, float]:
    """Return weights restricted to *available* signals, summing to 1."""
    active = {k: v for k, v in weights.items() if k in available}
    total = sum(active.values())
    if total < 1e-9:
        raise ValueError("No signals available; cannot build strength table.")
    return {k: v / total for k, v in active.items()}


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def build_and_save(*, refresh: bool = False) -> pd.DataFrame:
    """Run the full ingest + strength pipeline and save to processed/strength.csv."""
    from src.ingest.elo import fetch_elo
    from src.ingest.fifa import fetch_fifa
    from src.ingest.odds import fetch_odds

    cfg = _load_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "strength.csv"

    elo_df = fetch_elo(refresh=refresh)
    fifa_df = fetch_fifa(refresh=refresh)
    odds_df = fetch_odds(refresh=refresh)

    strength_df, fallbacks = build_strength_table(
        elo_df, fifa_df, odds_df, cfg=cfg, group_stage=True
    )

    strength_df.to_csv(out_path, index=False)
    logger.info("Saved strength table to %s", out_path)

    if fallbacks:
        logger.warning("Fallbacks recorded: %s", fallbacks)

    return strength_df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build team strength table")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch all data sources")
    args = parser.parse_args()

    df = build_and_save(refresh=args.refresh)
    print("\n=== Top 20 by composite strength ===\n")
    cols = ["team", "strength", "elo_rating", "match_count", "provisional"]
    print(df[cols].head(20).to_string(index=False))
    print(f"\nTotal teams: {len(df)}")
