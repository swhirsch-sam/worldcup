"""Projected bracket with head-to-head matchup odds.

simulation_summary.json answers "how likely is team X to reach stage Y?" but not
"who do they face at each stage, and what are the odds of *that* matchup?" —
the question you need answered to fill out an actual bracket.

This module answers it in two steps:

  1. Project the single most-likely outcome of the group stage (and best-third
     race) from the per-team marginal probabilities already in
     simulation_summary.json, then resolve it into 16 R32 matchups via the
     existing bracket_map machinery (resolve_r32) — so slot assignment exactly
     matches the simulator's logic.
  2. For every matchup at every stage, compute the *exact* head-to-head
     advancement probability in closed form from the fitted Dixon-Coles
     parameters: the same joint scoreline PMF simulate_match samples from,
     evaluated analytically rather than drawn from, chained through the same
     regulation -> extra-time -> penalties phases as simulate_ko_match. No
     extra Monte Carlo iterations required — these are the model's exact
     odds for that pairing, not empirical counts from however often two
     specific teams happened to meet across simulated tournaments.

The projected bracket is necessarily a single point estimate: real bracket
slots depend on group results that haven't been played, and marginal
probabilities can't capture every joint correlation. Treat it as "the most
likely single storyline," not a guarantee — exactly like a bracket you'd fill
out by hand, but with odds attached to every matchup.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.stats import poisson as scipy_poisson

from src.sim.bracket import load_bracket_map, resolve_r32
from src.sim.standings import GroupResult

logger = logging.getLogger(__name__)

_GRID = np.arange(11)
_KO_STAGES: tuple[str, ...] = ("r32", "r16", "qf", "sf", "final")
_STAGE_LABELS: dict[str, str] = {
    "r32": "Round of 32",
    "r16": "Round of 16",
    "qf": "Quarterfinals",
    "sf": "Semifinals",
    "final": "Final",
}


def _load_config() -> dict[str, Any]:
    with open("config.yaml") as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def _load_groups() -> dict[str, list[str]]:
    with open("data/groups.json") as f:
        data: dict[str, Any] = json.load(f)
    return data["groups"]


def _load_simulation_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    path = Path(cfg["output"]["simulation_summary"])
    with open(path) as f:
        result: dict[str, Any] = json.load(f)
    return result


# ---------------------------------------------------------------------------
# Strength loading (composite ensemble if cached; Elo + host bump fallback)
# ---------------------------------------------------------------------------
def load_strength(cfg: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Return (team -> strength, provenance note).

    Prefers the cached composite ensemble table (data/processed/strength.csv,
    the exact input the simulator used). Falls back to cached Elo ratings with
    the host bump applied — Elo carries the dominant ensemble weight (>=60%),
    so this closely approximates the composite for non-provisional teams, but
    will not exactly reproduce FIFA/market contributions.
    """
    processed_path = Path(cfg["data"]["processed_dir"]) / "strength.csv"
    if processed_path.exists():
        with open(processed_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {r["team"]: float(r["strength"]) for r in rows}, str(processed_path)

    logger.warning(
        "data/processed/strength.csv not found; approximating composite strength "
        "from cached Elo ratings + host bump (FIFA/market contributions not reproduced)."
    )
    elo_path = Path(cfg["data"]["cache_dir"]) / "elo_ratings.csv"
    with open(elo_path, newline="", encoding="utf-8") as f:
        strength = {r["team"]: float(r["elo_rating"]) for r in csv.DictReader(f)}

    host_bump = float(cfg["host"]["elo_bump"])
    for host in cfg["host"]["teams"]:
        strength[host] = (
            strength.get(host, float(cfg["ensemble"]["global_mean_rating"])) + host_bump
        )
    return strength, f"{elo_path} (Elo + host bump approximation)"


# ---------------------------------------------------------------------------
# Closed-form scoreline / outcome probabilities
#
# Mirrors DixonColesModel.simulate_match's joint-PMF construction exactly
# (src/model/poisson.py) but returns the probability mass directly instead of
# sampling from it -- the analytical odds *are* what simulate_match draws from.
# ---------------------------------------------------------------------------
def joint_scoreline_pmf(lambda_h: float, lambda_a: float, rho: float) -> NDArray[Any]:
    """Exact 11x11 Dixon-Coles joint scoreline probability matrix."""
    p_h = scipy_poisson.pmf(_GRID, lambda_h)
    p_a = scipy_poisson.pmf(_GRID, lambda_a)
    joint = np.outer(p_h, p_a)
    joint[0, 0] *= 1.0 - lambda_h * lambda_a * rho
    joint[1, 0] *= 1.0 + lambda_a * rho
    joint[0, 1] *= 1.0 + lambda_h * rho
    joint[1, 1] *= 1.0 - rho
    joint = joint / joint.sum()
    return joint


def outcome_probs(lambda_h: float, lambda_a: float, rho: float) -> tuple[float, float, float]:
    """Return (P(home win), P(draw), P(away win)) for a single 90-minute match."""
    joint = joint_scoreline_pmf(lambda_h, lambda_a, rho)
    # joint[i, j] = P(home scores i, away scores j): home win is row > col (lower
    # triangle), away win is row < col (upper triangle).
    p_home = float(np.tril(joint, k=-1).sum())
    p_draw = float(np.trace(joint))
    p_away = float(np.triu(joint, k=1).sum())
    return p_home, p_draw, p_away


def group_match_odds(
    team_a: str, team_b: str, strength: dict[str, float], cfg: dict[str, Any]
) -> dict[str, float]:
    """P(A win) / P(draw) / P(B win) for a single neutral-venue group match."""
    gm = cfg["goals_model"]
    alpha, beta = float(gm["intercept"]), float(gm["slope"])
    lam_cap, lam_floor = float(gm.get("lambda_cap", 6.0)), float(gm.get("lambda_floor", 0.1))
    rho = float(cfg["dixon_coles"]["rho"])

    elo_diff = strength[team_a] - strength[team_b]
    lam_a = float(np.clip(np.exp(alpha + beta * elo_diff), lam_floor, lam_cap))
    lam_b = float(np.clip(np.exp(alpha - beta * elo_diff), lam_floor, lam_cap))
    p_a, p_draw, p_b = outcome_probs(lam_a, lam_b, rho)
    return {"p_a_win": p_a, "p_draw": p_draw, "p_b_win": p_b}


def knockout_odds(
    team_a: str, team_b: str, strength: dict[str, float], cfg: dict[str, Any]
) -> dict[str, float]:
    """Exact P(advance) for both sides of a neutral knockout match.

    Analytically chains the same three phases simulate_ko_match plays out:
      regulation -> extra time (if level) -> penalties (if still level).
    P(A advances) = P(A wins reg)
                  + P(level after reg)  * [ P(A wins ET)
                                           + P(level after ET) * P(A wins pens) ]
    """
    gm, ko = cfg["goals_model"], cfg["knockout"]
    alpha, beta = float(gm["intercept"]), float(gm["slope"])
    lam_cap, lam_floor = float(gm.get("lambda_cap", 6.0)), float(gm.get("lambda_floor", 0.1))
    caution = float(ko["caution_factor"])
    et_frac = float(ko["extra_time_duration_fraction"])
    p_base = float(ko["penalty_base"])
    p_tilt = float(ko["penalty_strength_tilt"])
    p_tilt_max = float(ko["penalty_strength_tilt_max"])
    rho = float(cfg["dixon_coles"]["rho"])

    elo_diff = strength[team_a] - strength[team_b]

    lam_a = float(np.clip(np.exp(alpha + beta * elo_diff) * caution, lam_floor, lam_cap))
    lam_b = float(np.clip(np.exp(alpha - beta * elo_diff) * caution, lam_floor, lam_cap))
    a_reg, draw_reg, _ = outcome_probs(lam_a, lam_b, rho)

    lam_a_et = float(np.clip(lam_a * et_frac, lam_floor, lam_cap))
    lam_b_et = float(np.clip(lam_b * et_frac, lam_floor, lam_cap))
    a_et, draw_et, _ = outcome_probs(lam_a_et, lam_b_et, rho)

    p_a_pens = float(p_base + np.clip(elo_diff / 100.0 * p_tilt, -p_tilt_max, p_tilt_max))

    p_a = a_reg + draw_reg * (a_et + draw_et * p_a_pens)
    return {
        "p_a_advance": p_a,
        "p_b_advance": 1.0 - p_a,
        "p_drawn_after_90": draw_reg,
        "p_drawn_after_120": draw_reg * draw_et,
        "p_a_win_penalties": p_a_pens,
    }


# ---------------------------------------------------------------------------
# Most-likely group stage projection
# ---------------------------------------------------------------------------
def project_group_outcomes(
    probs: dict[str, dict[str, float]], groups: dict[str, list[str]]
) -> dict[str, GroupResult]:
    """Project the single most-likely 1st/2nd/3rd/4th finish for each group.

    Greedy selection straight from the simulator's marginal probabilities:
    most-probable group winner first, then most-probable runner-up among
    those remaining, then rank the last two by their combined top-2 odds
    (a proxy for "which of these two is the stronger side") to settle 3rd
    vs. 4th. This is a projection, not a forecast of the joint outcome —
    marginal probabilities can't capture how group results correlate.
    """
    projected: dict[str, GroupResult] = {}
    for group_id, teams in groups.items():
        remaining = list(teams)
        first = max(remaining, key=lambda t: probs[t]["group_first"])
        remaining.remove(first)
        second = max(remaining, key=lambda t: probs[t]["group_second"])
        remaining.remove(second)
        third, fourth = sorted(remaining, key=lambda t: probs[t]["group_top2"], reverse=True)
        projected[group_id] = GroupResult(
            group_id=group_id, first=first, second=second, third=third, fourth=fourth
        )
    return projected


def project_best_thirds(
    group_results: dict[str, GroupResult],
    probs: dict[str, dict[str, float]],
    n: int = 8,
) -> list[str]:
    """Pick the n group letters whose projected 3rd-place team is likeliest to qualify."""
    ranked = sorted(
        group_results.items(), key=lambda kv: probs[kv[1].third]["third_qualify"], reverse=True
    )
    return sorted(group_id for group_id, _ in ranked[:n])


# ---------------------------------------------------------------------------
# Full projected bracket
# ---------------------------------------------------------------------------
def build_projected_bracket(
    strength: dict[str, float],
    probs: dict[str, dict[str, float]],
    groups: dict[str, list[str]],
    bracket_map: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Project the most-likely bracket end to end, with odds on every matchup.

    Returns:
        {
          "group_results": {group_id: GroupResult},
          "best_third_groups": [8 group letters],
          "rounds": {stage: [{"team_a", "team_b", "odds", "favorite"}, ...]},
          "champion": str,
        }
    """
    group_results = project_group_outcomes(probs, groups)
    best_third_groups = project_best_thirds(group_results, probs)
    matchups = resolve_r32(group_results, best_third_groups, bracket_map)

    rounds: dict[str, list[dict[str, Any]]] = {}
    for stage_idx, stage in enumerate(_KO_STAGES):
        records = []
        winners = []
        for team_a, team_b in matchups:
            odds = knockout_odds(team_a, team_b, strength, cfg)
            favorite = team_a if odds["p_a_advance"] >= odds["p_b_advance"] else team_b
            records.append({"team_a": team_a, "team_b": team_b, "odds": odds, "favorite": favorite})
            winners.append(favorite)
        rounds[stage] = records
        if stage_idx + 1 < len(_KO_STAGES):
            matchups = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

    return {
        "group_results": group_results,
        "best_third_groups": best_third_groups,
        "rounds": rounds,
        "champion": rounds["final"][0]["favorite"],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def render_report(
    bracket: dict[str, Any], strength_source: str, groups: dict[str, list[str]]
) -> str:
    """Render the projected bracket + matchup odds as a Markdown report."""
    team_to_group = {t: g for g, ts in groups.items() for t in ts}
    lines: list[str] = []
    lines.append("# Projected World Cup 2026 Bracket — with head-to-head matchup odds")
    lines.append("")
    lines.append(
        "A single 'most likely' bracket built by taking the most probable occupant of "
        "every slot from the simulation's per-team probabilities, then computing the "
        "*exact* head-to-head advancement odds for each resulting matchup directly from "
        "the fitted Dixon-Coles model (regulation -> extra time -> penalties, in closed "
        "form -- not re-sampled)."
    )
    lines.append("")
    lines.append(f"Strength basis: `{strength_source}`")
    lines.append(
        "\n> **Read this as one plausible storyline, not a forecast of the joint outcome.** "
        "Both the projected group results and the bracket built on top of them are point "
        "estimates from marginal probabilities -- real tournaments branch at every match."
    )
    lines.append("")

    lines.append("## Projected group-stage results")
    lines.append("")
    lines.append("| Group | 1st | 2nd | 3rd | 4th |")
    lines.append("|---|---|---|---|---|")
    for group_id in sorted(bracket["group_results"]):
        gr = bracket["group_results"][group_id]
        lines.append(f"| {group_id} | {gr.first} | {gr.second} | {gr.third} | {gr.fourth} |")
    lines.append("")
    lines.append(
        "**Best third-place qualifiers projected from groups:** "
        + ", ".join(bracket["best_third_groups"])
    )
    lines.append("")

    for stage in _KO_STAGES:
        lines.append(f"## {_STAGE_LABELS[stage]}")
        lines.append("")
        lines.append("| Matchup | Advance odds | Favorite | Path to a draw |")
        lines.append("|---|---|---|---|")
        for rec in bracket["rounds"][stage]:
            a, b, odds, fav = rec["team_a"], rec["team_b"], rec["odds"], rec["favorite"]
            ga, gb = team_to_group.get(a, "?"), team_to_group.get(b, "?")
            matchup = f"**{a}** ({ga}) vs **{b}** ({gb})"
            advance = f"{a} {_fmt_pct(odds['p_a_advance'])} — {_fmt_pct(odds['p_b_advance'])} {b}"
            draw_path = (
                f"P(level after 90)={_fmt_pct(odds['p_drawn_after_90'])}, "
                f"P(to penalties)={_fmt_pct(odds['p_drawn_after_120'])}, "
                f"P({a} wins pens)={_fmt_pct(odds['p_a_win_penalties'])}"
            )
            # Round to display precision before crowning a "favorite" -- a coin-flip
            # pairing shouldn't read as a pick when the two sides display identically.
            favorite_label = (
                "Pick'em"
                if round(odds["p_a_advance"], 3) == round(odds["p_b_advance"], 3)
                else f"**{fav}**"
            )
            lines.append(f"| {matchup} | {advance} | {favorite_label} | {draw_path} |")
        lines.append("")

    lines.append(f"## Projected champion: {bracket['champion']}")
    lines.append("")

    # Road to the title: the champion's advance odds in each round they played,
    # and the compound probability of running this exact projected gauntlet.
    champ = bracket["champion"]
    compound = 1.0
    road: list[str] = []
    for stage in _KO_STAGES:
        for rec in bracket["rounds"][stage]:
            if champ not in (rec["team_a"], rec["team_b"]):
                continue
            opp = rec["team_b"] if rec["team_a"] == champ else rec["team_a"]
            p = rec["odds"]["p_a_advance"] if rec["team_a"] == champ else rec["odds"]["p_b_advance"]
            compound *= p
            road.append(f"- {_STAGE_LABELS[stage]}: beat {opp} — {_fmt_pct(p)}")
            break
    lines.append(f"**{champ}'s road to the title** (advance odds each round):")
    lines.append("")
    lines.extend(road)
    lines.append("")
    lines.append(
        f"Compounding those five results, the probability of {champ} running *this exact* "
        f"projected gauntlet is **{_fmt_pct(compound)}** — a reminder that even the single "
        "most-likely path is unlikely in absolute terms, because every round is a fresh coin "
        "weighted by these odds."
    )
    lines.append("")
    lines.append(
        "*Methodology: matchup odds are the model's exact analytical probabilities for "
        "that specific pairing -- P(advance) = P(win in regulation) + P(level after 90) x "
        "[P(win in ET) + P(level after 120) x P(win on penalties)] -- evaluated from the "
        "same joint Dixon-Coles scoreline distribution `simulate_match` samples from. "
        "They are not empirical counts of how often these two sides happened to meet "
        "across simulated tournaments, so they carry no Monte Carlo sampling noise.*"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = _load_config()
    groups = _load_groups()
    bracket_map = load_bracket_map()
    summary = _load_simulation_summary(cfg)
    probs: dict[str, dict[str, float]] = summary["probabilities"]
    strength, strength_source = load_strength(cfg)

    bracket = build_projected_bracket(strength, probs, groups, bracket_map, cfg)
    report = render_report(bracket, strength_source, groups)

    out_path = Path("results") / "projected_bracket.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote projected bracket report to %s", out_path)
    print(report)


if __name__ == "__main__":
    main()
