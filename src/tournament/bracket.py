"""Bracket allocation and knockout simulation.

Loads bracket_map.json (the official R32 allocation table) and validates
it at startup. Runs the full knockout tree: R32 → R16 → QF → SF → Final.
Handles extra time and penalty shootouts.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import yaml
from numpy.random import Generator

logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_and_validate_bracket_map(path: str = "data/bracket_map.json") -> dict[str, Any]:
    """Load bracket_map.json and assert structural integrity.

    Checks that every group label is referenced exactly the right number of
    times and that no slot is double-filled. This is the top correctness risk
    flagged in the README — raises immediately on any violation.
    """
    with open(path) as f:
        bracket_map: dict[str, Any] = json.load(f)

    from src.ingest.validate import validate_bracket_map
    from src.tournament.best_third import THIRD_PLACE_GROUP_COMBINATIONS

    validate_bracket_map(bracket_map, THIRD_PLACE_GROUP_COMBINATIONS)
    return bracket_map


def simulate_knockout(
    r32_bracket: list[tuple[str, str]],
    strength_lookup: dict[str, float],
    model: object,
    rng: Generator,
    cfg: dict[str, Any],
) -> str:
    """Simulate the full knockout tree and return the champion.

    Args:
        r32_bracket: 16 pairs (team_a, team_b) for the Round of 32.
        strength_lookup: Canonical team → composite strength.
        model: MatchSimulator instance.
        rng: Seeded Generator.
        cfg: Parsed config dict.

    Returns:
        Canonical name of the champion.
    """
    raise NotImplementedError("simulate_knockout: implement in Phase 5.")


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    strength_lookup: dict[str, float],
    model: object,
    rng: Generator,
    cfg: dict[str, Any],
    *,
    apply_caution: bool = True,
) -> str:
    """Simulate a single knockout match including ET and penalties if needed.

    Returns the winner's canonical name.
    """
    raise NotImplementedError("simulate_knockout_match: implement in Phase 5.")


def simulate_penalty_shootout(
    strength_a: float,
    strength_b: float,
    rng: Generator,
    cfg: dict[str, Any],
) -> bool:
    """Return True if team_a wins the shootout.

    Probability is base rate ± a bounded strength tilt from config.
    """
    raise NotImplementedError("simulate_penalty_shootout: implement in Phase 5.")
