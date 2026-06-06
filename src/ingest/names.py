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
# Canonical team list - all 48 teams from the Dec 5 2025 draw (groups A-L)
# ---------------------------------------------------------------------------
CANONICAL_TEAMS: Final[frozenset[str]] = frozenset(
    {
        # Group A
        "Mexico",
        "South Korea",
        "South Africa",
        "Czechia",
        # Group B
        "Canada",
        "Switzerland",
        "Qatar",
        "Bosnia and Herzegovina",
        # Group C
        "Brazil",
        "Morocco",
        "Haiti",
        "Scotland",
        # Group D
        "United States",
        "Paraguay",
        "Australia",
        "Türkiye",
        # Group E
        "Germany",
        "Curaçao",
        "Côte d'Ivoire",
        "Ecuador",
        # Group F
        "Netherlands",
        "Japan",
        "Tunisia",
        "Sweden",
        # Group G
        "Belgium",
        "Egypt",
        "Iran",
        "New Zealand",
        # Group H
        "Spain",
        "Cabo Verde",
        "Saudi Arabia",
        "Uruguay",
        # Group I
        "France",
        "Senegal",
        "Norway",
        "Iraq",
        # Group J
        "Argentina",
        "Algeria",
        "Austria",
        "Jordan",
        # Group K
        "Portugal",
        "Uzbekistan",
        "Colombia",
        "DR Congo",
        # Group L
        "England",
        "Croatia",
        "Ghana",
        "Panama",
    }
)


# ---------------------------------------------------------------------------
# Source → canonical mapping
# Keys are raw names as they appear in external sources.
# Values are canonical names that must be members of CANONICAL_TEAMS.
# ---------------------------------------------------------------------------
_ALIAS_MAP: dict[str, str] = {
    # United States
    "United States": "United States",
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    # South Korea
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Republic of Korea": "South Korea",
    # Iran
    "IR Iran": "Iran",
    "Iran": "Iran",
    "Islamic Republic of Iran": "Iran",
    # Côte d'Ivoire
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Côte d'Ivoire": "Côte d'Ivoire",
    # DR Congo
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Congo, DR": "DR Congo",
    # Curaçao
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
    # Czechia
    "Czechia": "Czechia",
    "Czech Republic": "Czechia",
    "Czechia Republic": "Czechia",
    # Türkiye
    "Türkiye": "Türkiye",
    "Turkey": "Türkiye",
    # Cabo Verde
    "Cabo Verde": "Cabo Verde",
    "Cape Verde": "Cabo Verde",
    # Bosnia and Herzegovina
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    # Other passthrough canonicals (names that are already correct)
    "Mexico": "Mexico",
    "South Africa": "South Africa",
    "Canada": "Canada",
    "Switzerland": "Switzerland",
    "Qatar": "Qatar",
    "Brazil": "Brazil",
    "Morocco": "Morocco",
    "Haiti": "Haiti",
    "Scotland": "Scotland",
    "Paraguay": "Paraguay",
    "Australia": "Australia",
    "Germany": "Germany",
    "Ecuador": "Ecuador",
    "Netherlands": "Netherlands",
    "Japan": "Japan",
    "Tunisia": "Tunisia",
    "Sweden": "Sweden",
    "Belgium": "Belgium",
    "Egypt": "Egypt",
    "New Zealand": "New Zealand",
    "Spain": "Spain",
    "Saudi Arabia": "Saudi Arabia",
    "Uruguay": "Uruguay",
    "France": "France",
    "Senegal": "Senegal",
    "Norway": "Norway",
    "Iraq": "Iraq",
    "Argentina": "Argentina",
    "Algeria": "Algeria",
    "Austria": "Austria",
    "Jordan": "Jordan",
    "Portugal": "Portugal",
    "Uzbekistan": "Uzbekistan",
    "Colombia": "Colombia",
    "England": "England",
    "Croatia": "Croatia",
    "Ghana": "Ghana",
    "Panama": "Panama",
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
