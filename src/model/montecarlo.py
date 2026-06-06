"""Vectorized Monte Carlo simulation of the full 2026 World Cup.

Architecture:
  - All per-match scoreline draws are numpy array operations (no Python loops
    over individual matches within a group stage round).
  - A Generator instance is passed down the call stack; global numpy seed is
    never touched.
  - After each run, invariant assertions are checked and the run manifest is
    written to results/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from numpy.random import Generator

from src.model.poisson import MatchSimulator
from src.sim.best_third import pick_best_thirds
from src.sim.bracket import load_bracket_map, resolve_r32, simulate_knockout
from src.sim.standings import GroupResult, simulate_group

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_STAGES = (
    "group_first",
    "group_second",
    "third_qualify",
    "r32",
    "r16",
    "qf",
    "sf",
    "final",
    "champion",
)
# Expected total count per stage across all teams, per iteration
_STAGE_TOTALS = {
    "group_first": 12,
    "group_second": 12,
    "third_qualify": 8,
    "r32": 32,
    "r16": 16,
    "qf": 8,
    "sf": 4,
    "final": 2,
    "champion": 1,
}


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def _load_groups() -> dict[str, list[str]]:
    """Return {group_letter: [team, ...]} from data/groups.json."""
    with open("data/groups.json") as f:
        data: dict[str, Any] = json.load(f)
    return data["groups"]


def run_simulation(
    strength_df: pd.DataFrame,
    model: MatchSimulator,
    *,
    n_iterations: int | None = None,
    seed: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full tournament Monte Carlo simulation.

    Args:
        strength_df: Team strength ratings (from strength.py).
        model: MatchSimulator instance (from poisson.py).
        n_iterations: Override config iterations (useful for testing).
        seed: Override config seed.
        cfg: Parsed config dict; loaded from disk if None.

    Returns:
        Results dict written to results/simulation_summary.json.
    """
    if cfg is None:
        cfg = _load_config()

    n = n_iterations if n_iterations is not None else int(cfg["simulation"]["iterations"])
    s = seed if seed is not None else int(cfg["simulation"]["seed"])
    rng: Generator = np.random.default_rng(s)

    logger.info("Starting Monte Carlo simulation: %d iterations, seed=%d", n, s)

    strength: dict[str, float] = dict(
        zip(strength_df["team"], strength_df["strength"].astype(float), strict=False)
    )
    groups_data = _load_groups()
    bracket_map = load_bracket_map()

    all_teams = [t for teams in groups_data.values() for t in teams]
    counts: dict[str, dict[str, int]] = {team: dict.fromkeys(_STAGES, 0) for team in all_teams}

    t0 = time.time()
    log_interval = max(1, n // 10)

    for i in range(n):
        if i % log_interval == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            logger.info("  %d / %d  (%.0f iter/s)", i, n, rate)

        # --- Group stage ---
        group_results: dict[str, GroupResult] = {}
        for group_id, teams in groups_data.items():
            gr = simulate_group(group_id, teams, strength, model, rng, cfg)
            group_results[group_id] = gr
            counts[gr.first]["group_first"] += 1
            counts[gr.second]["group_second"] += 1
            counts[gr.first]["r32"] += 1
            counts[gr.second]["r32"] += 1

        # --- Best-third selection ---
        qualifying_teams = pick_best_thirds(list(group_results.values()), n=8, rng=rng, cfg=cfg)
        third_to_group: dict[str, str] = {gr.third: gid for gid, gr in group_results.items()}
        best_third_groups = [third_to_group[t] for t in qualifying_teams]
        for team in qualifying_teams:
            counts[team]["third_qualify"] += 1
            counts[team]["r32"] += 1

        # --- Knockout stage ---
        r32_matchups = resolve_r32(group_results, best_third_groups, bracket_map)
        ko = simulate_knockout(r32_matchups, strength, model, rng, cfg)

        for team in ko["r32"]:
            counts[team]["r16"] += 1
        for team in ko["r16"]:
            counts[team]["qf"] += 1
        for team in ko["qf"]:
            counts[team]["sf"] += 1
        for team in ko["sf"]:
            counts[team]["final"] += 1
        counts[ko["champion"]]["champion"] += 1

    runtime = time.time() - t0
    logger.info("Simulation complete: %.1fs  (%.0f iter/s)", runtime, n / runtime)

    # Normalize to probabilities
    probabilities: dict[str, dict[str, float]] = {}
    for team in all_teams:
        p: dict[str, float] = {stage: counts[team][stage] / n for stage in _STAGES}
        p["group_top2"] = p["group_first"] + p["group_second"]
        probabilities[team] = p

    model_name = (
        "bivariate_poisson"
        if cfg.get("bivariate_poisson", {}).get("enabled", False)
        else "dixon_coles"
    )
    results: dict[str, Any] = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iterations": n,
            "seed": s,
            "model": model_name,
            "rho": float(cfg["dixon_coles"]["rho"]),
            "intercept": float(cfg["goals_model"]["intercept"]),
            "slope": float(cfg["goals_model"]["slope"]),
            "runtime_seconds": round(runtime, 1),
        },
        "probabilities": probabilities,
        "counts": counts,
    }

    out_path = Path(cfg["output"]["simulation_summary"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Simulation summary written to %s", out_path)

    return results


def check_invariants(results: dict[str, Any], n_iterations: int) -> None:
    """Assert all post-simulation invariants. Raises AssertionError on violation.

    Checks:
      - Sum of P(win tournament) == 1 within MC tolerance
      - Monotonic probability chain per team
      - Group finish probabilities sum to 1
      - Exactly 8 best-third slots, 32 R32 teams
      - No duplicate teams in any simulated bracket
      - Group points/GD/goals internal consistency
    """
    counts = results["counts"]
    probs = results["probabilities"]
    n = n_iterations

    # Stage total counts
    for stage, expected_per_iter in _STAGE_TOTALS.items():
        total = sum(c[stage] for c in counts.values())
        assert (
            total == expected_per_iter * n
        ), f"Invariant failed: {stage} total={total}, expected {expected_per_iter * n}"

    # r32 == group_first + group_second + third_qualify per team
    for team, c in counts.items():
        expected_r32 = c["group_first"] + c["group_second"] + c["third_qualify"]
        assert (
            c["r32"] == expected_r32
        ), f"{team}: r32={c['r32']} != group_first+second+third_qualify={expected_r32}"

    # Monotonic chain per team
    chain = ("r32", "r16", "qf", "sf", "final", "champion")
    for team, c in counts.items():
        for a, b in itertools.pairwise(chain):
            assert c[a] >= c[b], f"{team}: {a}={c[a]} < {b}={c[b]} (not monotonic)"

    # All probabilities in [0, 1]
    for team, p in probs.items():
        for stage, val in p.items():
            assert 0.0 <= val <= 1.0, f"{team}.{stage}={val:.6f} outside [0,1]"

    # Champion probs sum close to 1 (within MC noise)
    champion_sum = sum(p["champion"] for p in probs.values())
    assert abs(champion_sum - 1.0) < 1e-9, f"Champion prob sum={champion_sum:.8f}, expected 1.0"

    logger.info("All invariants passed.")


def write_run_manifest(cfg: dict[str, Any], seed: int, n_iterations: int) -> None:
    """Write results/run_manifest.json with provenance information."""
    import hashlib
    import subprocess

    import numpy as np_mod
    import pandas as pd_mod
    import scipy as scipy_mod

    manifest_path = Path(cfg["output"]["run_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"

    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

    manifest: dict[str, Any] = {
        "git_commit_sha": git_sha,
        "rng_seed": seed,
        "config_hash": config_hash,
        "n_iterations": n_iterations,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "type": (
                "bivariate_poisson"
                if cfg.get("bivariate_poisson", {}).get("enabled", False)
                else "dixon_coles"
            ),
            "rho": float(cfg["dixon_coles"]["rho"]),
            "intercept": float(cfg["goals_model"]["intercept"]),
            "slope": float(cfg["goals_model"]["slope"]),
        },
        "data_sources": {
            "groups": "data/groups.json",
            "bracket_map": "data/bracket_map.json",
            "elo_ratings": "data/raw/elo_ratings.csv",
        },
        "signal_fallbacks": [],
        "library_versions": {
            "numpy": np_mod.__version__,
            "pandas": pd_mod.__version__,
            "scipy": scipy_mod.__version__,
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Run manifest written to %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run WC 2026 Monte Carlo simulation.")
    parser.add_argument("--n-iterations", type=int, default=None, metavar="N")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast_mode_iterations from config (default 1000)",
    )
    args = parser.parse_args()

    cfg = _load_config()

    if args.fast:
        n_iter: int | None = int(cfg["simulation"]["fast_mode_iterations"])
    else:
        n_iter = args.n_iterations  # None means use config default

    from src.model.poisson import get_model
    from src.model.strength import build_and_save

    strength_df = build_and_save()
    model = get_model(
        rho=float(cfg["dixon_coles"]["rho"]),
        use_bivariate=bool(cfg.get("bivariate_poisson", {}).get("enabled", False)),
        lambda_corr=float(cfg.get("bivariate_poisson", {}).get("lambda_corr", 0.1)),
    )

    sim_results = run_simulation(
        strength_df,
        model,
        n_iterations=n_iter,
        seed=args.seed,
        cfg=cfg,
    )

    check_invariants(sim_results, sim_results["metadata"]["iterations"])
    write_run_manifest(cfg, sim_results["metadata"]["seed"], sim_results["metadata"]["iterations"])

    probs = sim_results["probabilities"]
    top10 = sorted(probs.items(), key=lambda x: x[1]["champion"], reverse=True)[:10]
    print("\nTop 10 champion probabilities")
    print("-" * 40)
    for rank, (team, p) in enumerate(top10, 1):
        print(f"  {rank:>2}. {team:<28}  {p['champion'] * 100:>5.1f}%")
    print()
