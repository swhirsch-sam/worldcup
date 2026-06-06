"""World Cup 2026 Bracket Predictor — Streamlit report.

Reads results/simulation_summary.json. Never runs the simulation on page load.
Run `python3 -m src.model.montecarlo` first to generate the results file,
then `streamlit run app/streamlit_app.py` (or `make app`) to launch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (relative to repo root, resolved from this file's location)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
SIMULATION_SUMMARY = _ROOT / "results" / "simulation_summary.json"
RUN_MANIFEST = _ROOT / "results" / "run_manifest.json"
GROUPS_JSON = _ROOT / "data" / "groups.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_ORDER = ("r32", "r16", "qf", "sf", "final", "champion")
STAGE_LABELS: dict[str, str] = {
    "r32": "R32",
    "r16": "R16 (Top 16)",
    "qf": "QF (Top 8)",
    "sf": "SF (Top 4)",
    "final": "Final",
    "champion": "Champion",
}
STAGE_SHORT: dict[str, str] = {
    "r32": "R32",
    "r16": "R16",
    "qf": "QF",
    "sf": "SF",
    "final": "Final",
    "champion": "Win",
}

HOSTS = {"United States", "Canada", "Mexico"}

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_results() -> dict[str, Any]:
    if not SIMULATION_SUMMARY.exists():
        st.error(
            f"No simulation results found at `{SIMULATION_SUMMARY}`. "
            "Run `python3 -m src.model.montecarlo` first."
        )
        st.stop()
    with open(SIMULATION_SUMMARY) as f:
        return json.load(f)


@st.cache_data
def load_manifest() -> dict[str, Any]:
    if not RUN_MANIFEST.exists():
        return {}
    with open(RUN_MANIFEST) as f:
        return json.load(f)


@st.cache_data
def load_groups() -> dict[str, list[str]]:
    with open(GROUPS_JSON) as f:
        return json.load(f)["groups"]


# ---------------------------------------------------------------------------
# Derived data helpers
# ---------------------------------------------------------------------------


def build_team_df(probs: dict[str, Any], groups: dict[str, list[str]]) -> pd.DataFrame:
    """One row per team with group assignment and all stage probabilities."""
    team_to_group = {team: g for g, teams in groups.items() for team in teams}
    rows = []
    for team, p in probs.items():
        row: dict[str, Any] = {
            "team": team,
            "group": team_to_group.get(team, "?"),
            "host": team in HOSTS,
        }
        for stage in STAGE_ORDER:
            row[stage] = p.get(stage, 0.0)
        row["group_first"] = p.get("group_first", 0.0)
        row["group_second"] = p.get("group_second", 0.0)
        row["group_top2"] = p.get("group_top2", 0.0)
        row["third_qualify"] = p.get("third_qualify", 0.0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)


def ci_half(p: float, n: int) -> float:
    """Half-width of 95% Wilson-style CI; clamped to avoid negatives."""
    return 1.96 * math.sqrt(max(p * (1.0 - p) / max(n, 1), 0.0))


def _cond_prob(probs: dict[str, Any], team: str, from_s: str, to_s: str) -> float:
    """P(reach to_s | reached from_s) from simulation counts."""
    p_from = probs.get(team, {}).get(from_s, 0.0)
    p_to = probs.get(team, {}).get(to_s, 0.0)
    return p_to / p_from if p_from > 1e-9 else 0.0


def _pick_prob(probs: dict[str, Any], team: str, role: str) -> float:
    """Joint probability that a pre-tournament pick succeeds.

    For a pick to survive a given round the team must (a) reach that round
    AND (b) win it.  Using raw P(reach next stage) captures both factors and
    is stable even with few simulation iterations.
    """
    stage_map = {
        "gs": "r32",      # must qualify for R32
        "r32": "r16",     # must win R32 match → reach R16
        "r16": "qf",      # must win R16 match → reach QF
        "qf": "sf",       # must win QF → reach SF
        "sf": "final",    # must win SF → reach Final
        "champ": "champion",  # must win Final
    }
    return probs.get(team, {}).get(stage_map[role], 0.0)


def _recommend_picks(probs: dict[str, Any]) -> dict[str, Any]:
    """Greedy survivor picks maximising joint survival probability.

    Uses P(reach next stage) for each role — this is the true probability that
    a pre-tournament pick survives that round (team must both reach AND win it).
    Fills most-constrained roles first: champion odds are lowest (~5-20%) so
    we assign the best available team there before filling Group Stage slots
    where top teams have 80-95% P(advance).

    Assignment order: Championship -> SF -> QF -> R16 -> R32 (x2) -> GS (x4).
    """
    all_teams = list(probs.keys())
    used: set[str] = set()

    def _pick(role: str, n: int) -> list[str]:
        ranked = sorted(
            (t for t in all_teams if t not in used),
            key=lambda t: _pick_prob(probs, t, role),
            reverse=True,
        )[:n]
        used.update(ranked)
        return ranked

    champ_picks = _pick("champ", 1)
    sf_picks = _pick("sf", 1)
    qf_picks = _pick("qf", 1)
    r16_picks = _pick("r16", 1)
    r32_picks = _pick("r32", 2)
    gs_picks = _pick("gs", 4)

    return {
        "gs": gs_picks,
        "r32": r32_picks,
        "r16": r16_picks[0] if r16_picks else "",
        "qf": qf_picks[0] if qf_picks else "",
        "sf": sf_picks[0] if sf_picks else "",
        "champ": champ_picks[0] if champ_picks else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    results = load_results()
    manifest = load_manifest()
    groups = load_groups()

    meta = results["metadata"]
    probs: dict[str, Any] = results["probabilities"]
    n_iter: int = int(meta["iterations"])

    team_df = build_team_df(probs, groups)
    top_team = team_df.iloc[0]["team"]
    top_prob = team_df.iloc[0]["champion"]

    # -- Header --
    st.title("FIFA World Cup 2026 — Bracket Predictor")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projected Champion", top_team)
    c2.metric("Title Odds", f"{top_prob:.1%}")
    c3.metric("Simulations Run", f"{n_iter:,}")
    c4.metric("Last Updated", meta.get("generated_at", "")[:10])

    st.markdown(
        f"**How it works:** We simulated the entire 2026 World Cup **{n_iter:,} times** using a "
        "statistical model trained on **49,000+ historical international matches** (1872-present). "
        "Each match outcome is determined by team strength learned from decades of real results - "
        "stronger teams win more often, but upsets still happen. After all those simulations, we "
        "count how often each team reached each round. Those frequencies become the percentages "
        "you see throughout this app. For the technical details, see the "
        "**Model Stats** and **Methodology** tabs.\n\n"
        "**Data sources:** Historical match results (49k+ games, 1872-present) · "
        "Elo strength ratings · FIFA rankings · WC 2026 official bracket format"
    )

    if n_iter < 10_000:
        st.warning(
            f"Only {n_iter:,} simulations run — probabilities will sharpen with more. "
            "Run `python3 -m src.model.montecarlo` for 50k iterations."
        )

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Group Stage",
            "Bracket",
            "Champion Odds",
            "Best Third",
            "Pick 'Em",
            "Model Stats",
            "Methodology",
        ]
    )

    with tab1:
        _render_group_stage(probs, groups)
    with tab2:
        _render_bracket(team_df, n_iter)
    with tab3:
        _render_champion_odds(team_df, n_iter)
    with tab4:
        _render_best_third(team_df, groups, n_iter)
    with tab5:
        _render_pickem(team_df, probs, groups, n_iter)
    with tab6:
        _render_model_stats(team_df, meta, n_iter)
    with tab7:
        _render_methodology(manifest, meta)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_group_stage(probs: dict[str, Any], groups: dict[str, list[str]]) -> None:
    st.header("Group Stage — Who Advances?")
    st.markdown(
        "The **top 2 teams** in each group automatically qualify for the knockout round. "
        "The best 8 third-place teams also advance as wildcards (see the **Best Third** tab). "
        "Percentages show each team's estimated chance of finishing in that position. "
        "Teams marked **HOST** represent the United States, Canada, or Mexico."
    )

    cols = st.columns(3)
    for idx, (group_id, teams) in enumerate(sorted(groups.items())):
        with cols[idx % 3]:
            st.subheader(f"Group {group_id}")
            rows = []
            for team in teams:
                p = probs.get(team, {})
                flag = "HOST " if team in HOSTS else ""
                rows.append(
                    {
                        "Team": flag + team,
                        "Finish 1st": p.get("group_first", 0.0),
                        "Finish 2nd": p.get("group_second", 0.0),
                        "Qualify %": p.get("r32", 0.0),
                    }
                )
            df = (
                pd.DataFrame(rows)
                .sort_values("Qualify %", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(
                df.style.format(
                    {"Finish 1st": "{:.1%}", "Finish 2nd": "{:.1%}", "Qualify %": "{:.1%}"}
                ).background_gradient(subset=["Qualify %"], cmap="YlGn"),
                hide_index=True,
                use_container_width=True,
            )


def _render_best_third(
    team_df: pd.DataFrame,
    groups: dict[str, list[str]],
    n_iter: int,
) -> None:
    st.header("The Wildcard Spots — Best Third-Place Teams")
    st.markdown(
        "This World Cup has **48 teams split into 12 groups of 4**. Only the top 2 from each "
        "group automatically advance — but **8 of the 12 third-place finishers** also qualify "
        "based on their combined points, goal difference, and goals scored across all groups. "
        "Gold bars show the teams most likely to grab one of those 8 wildcard spots."
    )

    keep_cols = ["team", "group", "group_first", "group_second", "third_qualify", "r32"]
    third_df = team_df[keep_cols].copy()
    third_df = third_df.sort_values("third_qualify", ascending=False).reset_index(drop=True)
    third_df["rank"] = range(1, len(third_df) + 1)

    # Only show teams with meaningful best-third probability
    plot_df = third_df[third_df["third_qualify"] > 0.001].head(20)

    fig = go.Figure(
        go.Bar(
            x=plot_df["third_qualify"] * 100,
            y=plot_df["team"] + " (" + plot_df["group"] + ")",
            orientation="h",
            marker_color=["#f0a500" if i < 8 else "#adb5bd" for i in range(len(plot_df))],
            text=[f"{v:.1f}%" for v in plot_df["third_qualify"] * 100],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Estimated chance of qualifying as a best third-place team — gold = likely top 8",
        xaxis_title="Estimated probability (%)",
        height=max(400, len(plot_df) * 28),
        margin={"l": 220, "r": 80, "t": 50, "b": 40},
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full best-third table"):
        show_df = third_df[third_df["third_qualify"] > 0.001][
            ["team", "group", "group_first", "group_second", "third_qualify", "r32"]
        ]
        st.dataframe(
            show_df.style.format(
                {
                    "group_first": "{:.1%}",
                    "group_second": "{:.1%}",
                    "third_qualify": "{:.1%}",
                    "r32": "{:.1%}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def _render_bracket(team_df: pd.DataFrame, n_iter: int) -> None:
    st.header("How Far Will Each Team Go?")
    st.markdown(
        "Each number shows a team's estimated chance of **reaching** that round. "
        "Darker color = higher probability. Columns go left to right: "
        "R32 → Round of 16 → Quarterfinals → Semifinals → Final → Champion. "
        "Use the slider to compare more or fewer teams."
    )

    # Heatmap: teams x stages
    top_n = st.slider("How many teams to show", 10, 48, 24)
    df = team_df.head(top_n)[["team", *STAGE_ORDER]].set_index("team")
    df.columns = [STAGE_SHORT[s] for s in STAGE_ORDER]

    fig = px.imshow(
        df.values * 100,
        x=list(df.columns),
        y=list(df.index),
        color_continuous_scale="YlOrRd",
        labels={"color": "%"},
        text_auto=".0f",
        aspect="auto",
    )
    fig.update_layout(
        title=f"Estimated chance of reaching each round (%) — top {top_n} teams",
        height=max(400, top_n * 22 + 100),
        coloraxis_colorbar_title="%",
        xaxis_side="top",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full advancement table"):
        display_cols: dict[str, str] = {
            "group": "Group",
            "r32": "R32",
            "r16": "R16",
            "qf": "QF",
            "sf": "SF",
            "final": "Final",
            "champion": "Champion",
        }
        show = team_df[["team", *display_cols.keys()]].rename(
            columns={"team": "Team", **display_cols}
        )
        fmt = {v: "{:.1%}" for v in display_cols.values() if v != "Group"}
        st.dataframe(
            show.style.format(fmt).background_gradient(subset=["Champion"], cmap="YlGn"),
            hide_index=True,
            use_container_width=True,
        )


def _render_champion_odds(team_df: pd.DataFrame, n_iter: int) -> None:
    st.header("Who Wins the World Cup?")
    st.markdown(
        "Each bar shows a team's estimated chance of **winning the tournament outright**. "
        "Football is unpredictable, but decades of match data give the model a strong signal "
        "on which teams are legitimate contenders. Hover over any bar for the exact percentage."
    )

    top_n = st.slider("How many teams to show", 10, 48, 20, key="champ_n")
    df = team_df.head(top_n).copy()
    df["ci"] = df["champion"].apply(lambda p: ci_half(p, n_iter))
    df["pct"] = df["champion"] * 100
    df["ci_pct"] = df["ci"] * 100
    df["label"] = df["champion"].apply(lambda p: f"{p:.1%}")
    df = df.sort_values("champion", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["pct"],
            y=df["team"],
            orientation="h",
            error_x={"array": df["ci_pct"], "color": "rgba(0,0,0,0.4)", "thickness": 1.5},
            marker_color=df["champion"].apply(
                lambda p: f"rgba(240, 165, 0, {0.4 + 0.6 * p / df['champion'].max()})"
            ),
            text=df["label"],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Estimated win probability (%)",
        height=max(400, top_n * 28 + 80),
        margin={"l": 200, "r": 80, "t": 40, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Group-level champion totals
    st.subheader("Win probability by group")
    group_champ = (
        team_df.groupby("group")["champion"]
        .sum()
        .reset_index()
        .sort_values("champion", ascending=False)
    )
    fig2 = px.bar(
        group_champ,
        x="group",
        y="champion",
        text=group_champ["champion"].apply(lambda p: f"{p:.1%}"),
        labels={"champion": "P(Champion)", "group": "Group"},
        color="champion",
        color_continuous_scale="YlOrRd",
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(coloraxis_showscale=False, height=350)
    st.plotly_chart(fig2, use_container_width=True)


def _render_model_stats(
    team_df: pd.DataFrame,
    meta: dict[str, Any],
    n_iter: int,
) -> None:
    st.header("Model & Simulation Statistics")
    st.markdown(
        "For the technically curious — here's what's powering the numbers throughout this app. "
        "The model is a **Dixon-Coles Poisson regression** fitted on 49k+ historical matches. "
        "It learns how Elo rating differences translate into expected goals, then applies a "
        "low-score correction (rho) since scorelines like 0-0 and 1-1 happen more often in "
        "football than a naive Poisson model would predict. "
        "See the **Methodology** tab for a full walkthrough."
    )
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model parameters")
        st.table(
            pd.DataFrame(
                {
                    "Parameter": [
                        "Model",
                        "rho",
                        "Intercept (alpha)",
                        "Slope (beta)",
                        "Iterations",
                        "Seed",
                    ],
                    "Value": [
                        meta.get("model", "dixon_coles"),
                        f"{meta.get('rho', 0):.6f}",
                        f"{meta.get('intercept', 0):.6f}",
                        f"{meta.get('slope', 0):.8f}",
                        f"{n_iter:,}",
                        str(meta.get("seed", "")),
                    ],
                }
            ).set_index("Parameter")
        )

    with col2:
        st.subheader("Champion distribution")
        champ_probs = team_df["champion"].sort_values(ascending=False).values
        hhi = float((champ_probs**2).sum())
        effective_n = 1.0 / hhi if hhi > 0 else 0
        top3_share = float(champ_probs[:3].sum())
        top8_share = float(champ_probs[:8].sum())
        st.table(
            pd.DataFrame(
                {
                    "Statistic": [
                        "HHI (market concentration)",
                        "Effective # of contenders",
                        "Top-3 combined P(champion)",
                        "Top-8 combined P(champion)",
                        "Runtime",
                    ],
                    "Value": [
                        f"{hhi:.4f}",
                        f"{effective_n:.1f}",
                        f"{top3_share:.1%}",
                        f"{top8_share:.1%}",
                        f"{meta.get('runtime_seconds', 0):.0f}s",
                    ],
                }
            ).set_index("Statistic")
        )

    st.subheader("Champion probability distribution")
    fig = px.histogram(
        team_df,
        x="champion",
        nbins=30,
        labels={"champion": "P(Champion)"},
        title="Distribution of champion probabilities across 48 teams",
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Formal backtesting against 2018 and 2022 World Cup results will be "
        "added in Phase 7 (calibration & eval suite)."
    )


def _render_methodology(manifest: dict[str, Any], meta: dict[str, Any]) -> None:
    st.header("Methodology & Data Provenance")

    st.subheader("Model overview")
    st.markdown(
        """
**Elo ratings** — computed from 49k+ historical international matches (Mart Jurisoo dataset)
using eloratings.net methodology: K-factors by tournament type (60/50/40/30/20),
goal-weight G = (goal_diff + 1)^0.5, home advantage 100 Elo pts.

**Dixon-Coles Poisson model** — each match draws from an 11x11 joint PMF with
a low-score correction factor tau(x,y; rho). rho is negative for football (more
0-0/1-1 than independent Poisson predicts). Both teams' lambdas are derived from
`lambda = exp(alpha + beta * elo_diff)`, fitted by Poisson MLE on the same dataset.

**Strength ensemble** — z-score blend of Elo, FIFA rankings (when available), and
market implied probabilities (when available). Provisional teams are shrunk 30%
toward the mean; host nations receive an Elo bump.

**Group stage** — 12 groups of 4, round-robin (6 matches each). Rankings use FIFA
8-level tiebreakers: points, GD, GF, H2H points, H2H GD, H2H GF, fair-play, lots.

**Best-third selection** — 8 of the 12 third-place teams qualify based on pts/GD/GF/FP.
Slot assignments follow a bipartite matching (Hungarian algorithm) over the Annex C
bracket map, pre-computed for all C(12,8)=495 combinations.

**Knockout simulation** — 90-min Poisson draw (caution factor 0.85), then extra time
(lambda x 0.333), then penalties (Bernoulli p = 0.5 +/- Elo tilt, capped at 0.1).
        """
    )

    st.subheader("Key fitted values")
    st.code(
        f"alpha (intercept) = {meta.get('intercept', 0):.6f}\n"
        f"beta  (Elo slope)  = {meta.get('slope', 0):.8f}\n"
        f"rho   (DC rho)     = {meta.get('rho', 0):.6f}",
        language="text",
    )

    if manifest:
        st.subheader("Run manifest")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Git commit:** `{manifest.get('git_commit_sha', 'n/a')[:12]}`")
            st.markdown(f"**Config hash:** `{manifest.get('config_hash', 'n/a')}`")
            st.markdown(f"**RNG seed:** {manifest.get('rng_seed', 'n/a')}")
            st.markdown(f"**Iterations:** {manifest.get('n_iterations', 'n/a'):,}")
        with col2:
            st.markdown(f"**Timestamp (UTC):** {manifest.get('timestamp_utc', 'n/a')}")
            libs = manifest.get("library_versions", {})
            if libs:
                st.markdown("**Library versions:**")
                for lib, ver in libs.items():
                    st.markdown(f"  - {lib}: {ver}")
    else:
        st.info("Run manifest not found. Run `python3 -m src.model.montecarlo` to generate it.")


def _render_pickem(
    team_df: pd.DataFrame,
    probs: dict[str, Any],
    groups: dict[str, list[str]],
    n_iter: int,
) -> None:
    st.header("Pick 'Em — Survivor Contest")

    hdr, reset_col = st.columns([6, 1])
    with hdr:
        st.caption(
            "**Phase 1 — Group Stage:** Pick 4 teams to advance to the R32 (all 4 must survive). "
            "**Phase 2 — Knockout:** R32 (2 picks) → R16 → QF → SF → Championship (1 pick each). "
            "All picks in a round must win to advance. "
            "Each team usable **once** in the entire contest."
        )
    with reset_col:
        if st.button("↺ Reset", use_container_width=True):
            for k in ("pk_gs", "pk_r32", "pk_r16", "pk_qf", "pk_sf", "pk_champ"):
                st.session_state.pop(k, None)
            st.rerun()

    team_to_group = {t: g for g, ts in groups.items() for t in ts}
    teams_by_r32 = team_df.sort_values("r32", ascending=False)["team"].tolist()

    # ── Model Recommendations ─────────────────────────────────────────────────
    rec = _recommend_picks(probs)

    with st.expander("💡 Model Recommendations", expanded=True):
        st.markdown(
            "The model picks the team with the **highest P(survive that round)** "
            "for each role — the joint probability of both reaching and winning it. "
            "Most-constrained roles are filled first "
            "(Championship → SF → QF → R16 → R32 → Group Stage). "
            "Each team is used at most once."
        )

        # Build recommendation table
        rec_rows: list[dict[str, Any]] = []
        for t in rec["gs"]:
            rec_rows.append(
                {
                    "Round": "Group Stage",
                    "Team": t,
                    "Grp": team_to_group.get(t, "?"),
                    "P(survive)": probs[t].get("r32", 0.0),
                    "Rationale": "Highest P(advance to R32)",
                }
            )
        for t in rec["r32"]:
            rec_rows.append(
                {
                    "Round": "R32",
                    "Team": t,
                    "Grp": team_to_group.get(t, "?"),
                    "P(survive)": _pick_prob(probs, t, "r32"),
                    "Rationale": "Highest P(reach R16)",
                }
            )
        for pick, rnd, role, rationale in [
            (rec["r16"], "R16", "r16", "Highest P(reach QF)"),
            (rec["qf"], "QF", "qf", "Highest P(reach SF)"),
            (rec["sf"], "SF", "sf", "Highest P(reach Final)"),
            (rec["champ"], "Championship", "champ", "Highest P(win title)"),
        ]:
            if pick:
                rec_rows.append(
                    {
                        "Round": rnd,
                        "Team": pick,
                        "Grp": team_to_group.get(pick, "?"),
                        "P(survive)": _pick_prob(probs, pick, role),
                        "Rationale": rationale,
                    }
                )

        rec_df = pd.DataFrame(rec_rows)
        overall_rec = math.prod(r["P(survive)"] for r in rec_rows)
        st.dataframe(
            rec_df.style.format({"P(survive)": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.metric("Recommended entry survival probability", f"{overall_rec:.4%}")

        if st.button("Apply recommendations to my picks", type="primary"):
            st.session_state["pk_gs"] = rec["gs"]
            st.session_state["pk_r32"] = rec["r32"]
            st.session_state["pk_r16"] = rec["r16"] or "(none)"
            st.session_state["pk_qf"] = rec["qf"] or "(none)"
            st.session_state["pk_sf"] = rec["sf"] or "(none)"
            st.session_state["pk_champ"] = rec["champ"] or "(none)"
            st.rerun()

    def _lbl(team: str, role: str) -> str:
        g = team_to_group.get(team, "?")
        p = _pick_prob(probs, team, role)
        return f"{team} [{g}]  ·  {p:.1%}"

    # Initialise session state on first visit
    for k, v in [
        ("pk_gs", []),
        ("pk_r32", []),
        ("pk_r16", "(none)"),
        ("pk_qf", "(none)"),
        ("pk_sf", "(none)"),
        ("pk_champ", "(none)"),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Phase 1: Group Stage ─────────────────────────────────────────────────
    st.subheader("Phase 1 — Group Stage")
    st.markdown(
        "Pick **4 teams** you believe will advance to the Round of 32. "
        "All 4 must advance or you are eliminated. "
        "Probability shown = P(qualify for R32)."
    )

    gs_picks: list[str] = st.multiselect(
        "Group Stage picks (choose exactly 4)",
        options=teams_by_r32,
        max_selections=4,
        format_func=lambda t: _lbl(t, "gs"),
        key="pk_gs",
    )

    if len(gs_picks) == 4:
        gs_prob = math.prod(probs[t].get("r32", 0.0) for t in gs_picks)
        st.success(f"P(all 4 advance to R32) = **{gs_prob:.3%}**")
    else:
        rem = 4 - len(gs_picks)
        st.info(f"Select {rem} more team{'s' if rem > 1 else ''} to complete Phase 1.")

    # ── Phase 2: Knockout Stage ──────────────────────────────────────────────
    st.divider()
    st.subheader("Phase 2 — Knockout Stage")
    st.markdown(
        "Teams used in the Group Stage are excluded. Probability = P(win that specific match)."
    )

    used: set[str] = set(gs_picks)

    def _avail(exclude: set[str]) -> list[str]:
        return [t for t in teams_by_r32 if t not in exclude]

    def _safe_select(label_str: str, key: str, opts: list[str], role: str) -> str:
        choices = ["(none)", *opts]
        cur = st.session_state.get(key, "(none)")
        if cur not in choices:
            cur = "(none)"
            st.session_state[key] = "(none)"
        result = st.selectbox(
            label_str,
            options=choices,
            index=choices.index(cur),
            format_func=lambda t: "— not yet picked —" if t == "(none)" else _lbl(t, role),
            key=key,
        )
        return result or "(none)"

    # R32 — 2 picks
    st.markdown("##### Round of 32 — Pick 2 teams to win their R32 match")
    r32_opts = _avail(used)
    prev_r32 = [t for t in st.session_state.get("pk_r32", []) if t in r32_opts]
    if prev_r32 != st.session_state.get("pk_r32", []):
        st.session_state["pk_r32"] = prev_r32
    r32_picks: list[str] = st.multiselect(
        "R32 picks (choose exactly 2)",
        options=r32_opts,
        max_selections=2,
        format_func=lambda t: _lbl(t, "r32"),
        key="pk_r32",
    )
    used.update(r32_picks)

    # R16 — 1 pick
    st.markdown("##### Round of 16 — Pick 1 team to win their R16 match")
    r16_pick: str = _safe_select("R16 pick", "pk_r16", _avail(used), "r16")
    if r16_pick != "(none)":
        used.add(r16_pick)

    # QF — 1 pick
    st.markdown("##### Quarterfinals — Pick 1 team to win their QF match")
    qf_pick: str = _safe_select("QF pick", "pk_qf", _avail(used), "qf")
    if qf_pick != "(none)":
        used.add(qf_pick)

    # SF — 1 pick
    st.markdown("##### Semifinals — Pick 1 team to win their SF match")
    sf_pick: str = _safe_select("SF pick", "pk_sf", _avail(used), "sf")
    if sf_pick != "(none)":
        used.add(sf_pick)

    # Championship — 1 pick
    st.markdown("##### Championship — Pick 1 team to win the tournament")
    champ_pick: str = _safe_select("Championship pick", "pk_champ", _avail(used), "champ")

    # ── Summary ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Your Picks Summary")

    rows: list[dict[str, Any]] = []
    for t in gs_picks:
        rows.append(
            {
                "Round": "Group Stage",
                "Team": t,
                "Grp": team_to_group.get(t, "?"),
                "P(survive)": probs.get(t, {}).get("r32", 0.0),
                "Must": "Advance to R32",
            }
        )
    for t in r32_picks:
        rows.append(
            {
                "Round": "R32",
                "Team": t,
                "Grp": team_to_group.get(t, "?"),
                "P(survive)": _pick_prob(probs, t, "r32"),
                "Must": "Win R32 match",
            }
        )
    for pick, rnd, role, must in [
        (r16_pick, "R16", "r16", "Win R16 match"),
        (qf_pick, "QF", "qf", "Win QF match"),
        (sf_pick, "SF", "sf", "Win SF match"),
        (champ_pick, "Championship", "champ", "Win the Final"),
    ]:
        if pick and pick != "(none)":
            rows.append(
                {
                    "Round": rnd,
                    "Team": pick,
                    "Grp": team_to_group.get(pick, "?"),
                    "P(survive)": _pick_prob(probs, pick, role),
                    "Must": must,
                }
            )

    if rows:
        sdf = pd.DataFrame(rows)
        st.dataframe(
            sdf.style.format({"P(survive)": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )
        overall = math.prod(r["P(survive)"] for r in rows)
        n_made = len(rows)
        st.metric(
            f"Estimated survival probability  ({n_made} / 10 picks made)",
            f"{overall:.4%}",
        )
        st.caption(
            "Survival probability is the product of each pick's P(advance to next stage). "
            "Assumes picks are independent — actual probability differs slightly due to "
            "bracket correlations."
        )
    else:
        st.info("Make your picks above to see your summary here.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
else:
    # Streamlit runs the module directly; execute main at import time
    main()
