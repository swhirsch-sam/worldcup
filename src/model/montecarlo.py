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

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.random import Generator

logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def run_simulation(
    strength_df: object,
    model: object,
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

    n = n_iterations if n_iterations is not None else cfg["simulation"]["iterations"]
    s = seed if seed is not None else cfg["simulation"]["seed"]
    _rng: Generator = np.random.default_rng(s)

    logger.info("Starting Monte Carlo simulation: %d iterations, seed=%d", n, s)

    raise NotImplementedError("run_simulation: implement in Phase 6.")


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
    raise NotImplementedError("check_invariants: implement in Phase 6.")


def write_run_manifest(cfg: dict[str, Any], seed: int, n_iterations: int) -> None:
    """Write results/run_manifest.json with provenance information."""
    import hashlib
    import subprocess
    import time as _time

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
        "timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "data_sources": {},  # populated in Phase 6 by ingest layer
        "signal_fallbacks": [],
        "library_versions": {},  # populated in Phase 6
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Run manifest written to %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit("montecarlo.py: full implementation in Phase 6.")
