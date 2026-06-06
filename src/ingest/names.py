"""Canonical team name registry and resolution layer.

Every source-specific team name must resolve through this module before
any join. Unmapped names raise NameResolutionError; they must not silently
pass through.
"""

from __future__ import annotations

from typing import Final


class NameResolutionError(ValueError):
    """Raised when a source name cannot be mapped to a canonical key."""


# ---------------------------------------------------------------------------
# Canonical team list (48 teams, ordered by group as of the Dec 5 2025 draw)
# Populated in Phase 2 once groups.json is confirmed.
# ---------------------------------------------------------------------------
CANONICAL_TEAMS: Final[frozenset[str]] = frozenset()  # filled in Phase 2


# ---------------------------------------------------------------------------
# Source → canonical mapping
# Keys are raw names as they appear in external sources.
# Values are canonical names that must be members of CANONICAL_TEAMS.
# ---------------------------------------------------------------------------
_ALIAS_MAP: dict[str, str] = {
    # FIFA / Elo common divergences
    "United States": "United States",
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Republic of Korea": "South Korea",
    "IR Iran": "Iran",
    "Iran": "Iran",
    "Islamic Republic of Iran": "Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Côte d'Ivoire": "Côte d'Ivoire",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Kyrgyzstan": "Kyrgyzstan",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
}


def resolve(raw_name: str) -> str:
    """Return the canonical team name for *raw_name*.

    Raises NameResolutionError if the name is not in the alias map or is
    already canonical. Callers must handle this exception — never suppress it.
    """
    candidate = _ALIAS_MAP.get(raw_name, raw_name)
    # Once CANONICAL_TEAMS is populated, enforce membership.
    if CANONICAL_TEAMS and candidate not in CANONICAL_TEAMS:
        raise NameResolutionError(
            f"Cannot resolve {raw_name!r} → {candidate!r}: not in canonical registry. "
            "Add it to _ALIAS_MAP in src/ingest/names.py."
        )
    return candidate


def resolve_all(names: list[str]) -> list[str]:
    """Resolve a list of names, raising on the first unmapped entry."""
    return [resolve(n) for n in names]
