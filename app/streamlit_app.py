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
ACTUAL_RESULTS_JSON = _ROOT / "data" / "actual_results.json"

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


@st.cache_data
def load_actual_results() -> dict[str, Any]:
    if not ACTUAL_RESULTS_JSON.exists():
        return {"group_stage": [], "knockout": {}, "_metadata": {}}
    with open(ACTUAL_RESULTS_JSON) as f:
        return json.load(f)


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


def compute_group_standings(
    matches: list[dict[str, Any]],
    groups: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {
        team: {"team": team, "group": g, "played": 0, "wins": 0, "draws": 0,
               "losses": 0, "gf": 0, "ga": 0, "pts": 0}
        for g, teams in groups.items()
        for team in teams
    }
    for m in matches:
        home, away = m["home"], m["away"]
        hg, ag = m["home_goals"], m["away_goals"]
        if home not in records or away not in records:
            continue
        records[home]["played"] += 1
        records[home]["gf"] += hg
        records[home]["ga"] += ag
        records[away]["played"] += 1
        records[away]["gf"] += ag
        records[away]["ga"] += hg
        if hg > ag:
            records[home]["wins"] += 1
            records[home]["pts"] += 3
            records[away]["losses"] += 1
        elif ag > hg:
            records[away]["wins"] += 1
            records[away]["pts"] += 3
            records[home]["losses"] += 1
        else:
            records[home]["draws"] += 1
            records[home]["pts"] += 1
            records[away]["draws"] += 1
            records[away]["pts"] += 1

    standings: dict[str, list[dict[str, Any]]] = {}
    for group_id, teams in groups.items():
        group_recs = [dict(records[t], gd=records[t]["gf"] - records[t]["ga"]) for t in teams]
        group_recs.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
        for i, r in enumerate(group_recs):
            r["rank"] = i + 1
        standings[group_id] = group_recs
    return standings


def compute_match_predictions(
    matches: list[dict[str, Any]],
    probs: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for m in matches:
        home, away = m["home"], m["away"]
        hg, ag = m["home_goals"], m["away_goals"]
        home_p1 = probs.get(home, {}).get("group_first", 0.0001)
        away_p1 = probs.get(away, {}).get("group_first", 0.0001)
        total = home_p1 + away_p1
        implied_home_p = home_p1 / total

        if hg > ag:
            actual = "home_win"
        elif ag > hg:
            actual = "away_win"
        else:
            actual = "draw"

        model_fav = home if implied_home_p >= 0.5 else away
        model_p = implied_home_p if implied_home_p >= 0.5 else 1.0 - implied_home_p

        if actual == "draw":
            verdict = "⚠️ Draw"
        elif (actual == "home_win" and implied_home_p >= 0.5) or (actual == "away_win" and implied_home_p < 0.5):
            verdict = "✅ Correct"
        else:
            verdict = "❌ Upset"

        results.append({
            **m,
            "score": f"{hg}–{ag}",
            "implied_home_p": implied_home_p,
            "model_fav": model_fav,
            "model_p": model_p,
            "actual": actual,
            "verdict": verdict,
        })
    return results


# ---------------------------------------------------------------------------
# Actual results tab
# ---------------------------------------------------------------------------


def _render_actual_results(
    probs: dict[str, Any],
    groups: dict[str, list[str]],
    actual_data: dict[str, Any],
) -> None:
    st.header("Actual Results & Model Check")

    matches: list[dict[str, Any]] = actual_data.get("group_stage", [])
    meta_ar = actual_data.get("_metadata", {})

    if not matches:
        st.info("No results have been entered yet. Add them to `data/actual_results.json`.")
        return

    preds = compute_match_predictions(matches, probs)
    n_correct = sum(1 for p in preds if p["verdict"].startswith("✅"))
    n_upsets = sum(1 for p in preds if p["verdict"].startswith("❌"))
    n_draws = sum(1 for p in preds if p["verdict"].startswith("⚠️"))
    accuracy = n_correct / len(preds) if preds else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches Played", len(matches))
    c2.metric("Model Accuracy", f"{accuracy:.0%}", f"{n_correct} / {len(preds)} favorites won")
    c3.metric("Upsets", n_upsets, delta="underdog won", delta_color="inverse")
    c4.metric("Draws", n_draws)

    if meta_ar.get("notes"):
        st.caption(f"📅 Last updated: {meta_ar.get('last_updated', '')} — {meta_ar['notes']}")

    st.divider()

    res_tab, stand_tab, check_tab = st.tabs(["Match Results", "Group Standings", "Model vs Reality"])

    # ---- Match Results ----
    with res_tab:
        st.subheader("Results by Group")
        group_matches: dict[str, list[dict[str, Any]]] = {}
        for m in matches:
            group_matches.setdefault(m["group"], []).append(m)

        cols = st.columns(3)
        for idx, (group_id, gmatches) in enumerate(sorted(group_matches.items())):
            with cols[idx % 3]:
                st.markdown(f"**Group {group_id}**")
                rows = [
                    {
                        "Date": m["date"][5:],
                        "Home": m["home"],
                        "Score": f"{m['home_goals']}–{m['away_goals']}",
                        "Away": m["away"],
                    }
                    for m in sorted(gmatches, key=lambda x: x["date"])
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ---- Group Standings ----
    with stand_tab:
        st.subheader("Current Group Standings")
        st.caption(
            "Sorted by Pts → GD → GF. **Pred.Rank** = model's predicted finishing order "
            "(by P(1st)). Green = auto-qualify top 2; yellow = potential best-third."
        )
        standings = compute_group_standings(matches, groups)

        cols = st.columns(3)
        for idx, (group_id, recs) in enumerate(sorted(standings.items())):
            with cols[idx % 3]:
                st.markdown(f"**Group {group_id}**")
                pred_order = sorted(groups[group_id], key=lambda t: -probs.get(t, {}).get("group_first", 0))
                rows = [
                    {
                        "Team": r["team"],
                        "Pts": r["pts"],
                        "GD": r["gd"],
                        "GF": r["gf"],
                        "Pred.Rank": pred_order.index(r["team"]) + 1 if r["team"] in pred_order else "-",
                    }
                    for r in recs
                ]
                df_stand = pd.DataFrame(rows)

                def _highlight_stand(df: pd.DataFrame) -> pd.DataFrame:
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    if len(styles) > 0:
                        styles.iloc[0] = "background-color: #c8f7c5"
                    if len(styles) > 1:
                        styles.iloc[1] = "background-color: #c8f7c5"
                    if len(styles) > 2:
                        styles.iloc[2] = "background-color: #fff3cd"
                    return styles

                st.dataframe(
                    df_stand.style.apply(_highlight_stand, axis=None),
                    hide_index=True,
                    use_container_width=True,
                )

    # ---- Model vs Reality ----
    with check_tab:
        st.subheader("Match-by-Match Model Check")
        st.caption(
            "Model picks the team with higher P(group_first). "
            "Confidence = implied win probability from relative strength. "
            "✅ = favorite won · ⚠️ = draw · ❌ = underdog won (upset)"
        )

        rows = [
            {
                "Date": p["date"][5:],
                "Grp": p["group"],
                "Home": p["home"],
                "Away": p["away"],
                "Score": p["score"],
                "Model Pick": p["model_fav"],
                "Confidence": p["model_p"],
                "Result": p["verdict"],
            }
            for p in preds
        ]
        df_check = pd.DataFrame(rows)

        def _color_verdict(col: pd.Series) -> list[str]:
            return [
                "background-color: #d4edda" if "✅" in v else
                "background-color: #f8d7da" if "❌" in v else
                "background-color: #fff3cd"
                for v in col
            ]

        st.dataframe(
            df_check.style
                .apply(_color_verdict, subset=["Result"])
                .format({"Confidence": "{:.0%}"}),
            hide_index=True,
            use_container_width=True,
        )

        # Biggest surprises chart
        surprises = sorted(
            [p for p in preds if not p["verdict"].startswith("✅")],
            key=lambda x: -x["model_p"],
        )
        if surprises:
            st.subheader("Biggest Surprises (favorites that didn't win)")
            rows2 = []
            for p in surprises:
                label = (
                    f"{p['home']} {p['score']} {p['away']}"
                    f"  ({p['verdict']})"
                )
                rows2.append({"match_label": label, "model_confidence": p["model_p"] * 100})
            df_surp = pd.DataFrame(rows2)
            fig = go.Figure(
                go.Bar(
                    x=df_surp["model_confidence"],
                    y=df_surp["match_label"],
                    orientation="h",
                    marker_color=[
                        "#dc3545" if "❌" in r["verdict"] else "#ffc107"
                        for r in surprises
                    ],
                    text=[f"{v:.0f}%" for v in df_surp["model_confidence"]],
                    textposition="outside",
                )
            )
            fig.add_vline(x=50, line_dash="dash", line_color="grey", annotation_text="50%")
            fig.update_layout(
                title="Model's confidence in the non-winning team (higher = bigger surprise)",
                xaxis_title="Model Confidence (%)",
                xaxis_range=[0, 105],
                height=max(300, len(surprises) * 36 + 80),
                margin={"l": 320, "r": 60, "t": 50, "b": 40},
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    results = load_results()
    manifest = load_manifest()
    groups = load_groups()
    actual_data = load_actual_results()

    meta = results["metadata"]
    probs: dict[str, Any] = results["probabilities"]
    n_iter: int = int(meta["iterations"])

    team_df = build_team_df(probs, groups)

    # -- Header --
    st.title("FIFA World Cup 2026 — Bracket Predictor")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Simulations", f"{n_iter:,}")
    c2.metric("Model", meta.get("model", "dixon_coles").replace("_", "-").title())
    c3.metric("Rho (Dixon-Coles)", f"{meta.get('rho', 0):.4f}")
    c4.metric("Generated", meta.get("generated_at", "")[:10])

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Group Stage", "Best Third", "Bracket", "Champion Odds",
         "Actual Results ⚽", "Model Stats", "Methodology"]
    )

    with tab1:
        _render_group_stage(probs, groups)
    with tab2:
        _render_best_third(team_df, groups, n_iter)
    with tab3:
        _render_bracket(team_df, n_iter)
    with tab4:
        _render_champion_odds(team_df, n_iter)
    with tab5:
        _render_actual_results(probs, groups, actual_data)
    with tab6:
        _render_model_stats(team_df, meta, n_iter)
    with tab7:
        _render_methodology(manifest, meta)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_group_stage(probs: dict[str, Any], groups: dict[str, list[str]]) -> None:
    st.header("Group Stage Advancement Probabilities")
    st.caption(
        "P(1st) / P(2nd) / P(Advance R32) for each group. "
        "P(Advance) includes automatic top-2 qualification plus best-third selection."
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
                        "P(1st)": p.get("group_first", 0.0),
                        "P(2nd)": p.get("group_second", 0.0),
                        "P(R32)": p.get("r32", 0.0),
                    }
                )
            df = pd.DataFrame(rows).sort_values("P(R32)", ascending=False).reset_index(drop=True)
            st.dataframe(
                df.style.format(
                    {"P(1st)": "{:.1%}", "P(2nd)": "{:.1%}", "P(R32)": "{:.1%}"}
                ).background_gradient(subset=["P(R32)"], cmap="YlGn"),
                hide_index=True,
                use_container_width=True,
            )


def _render_best_third(
    team_df: pd.DataFrame,
    groups: dict[str, list[str]],
    n_iter: int,
) -> None:
    st.header("Best-Third Qualification Probabilities")
    st.caption(
        "8 of the 12 third-place teams qualify for the R32. "
        "P(Best Third) is the probability of finishing 3rd AND being selected among the top 8."
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
        title="P(Qualify as Best Third) — gold = likely top-8",
        xaxis_title="Probability (%)",
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
    st.header("Knockout Bracket Advancement Probabilities")
    st.caption(
        "Probability of reaching each knockout round, conditional on being in the tournament. "
        "All 48 teams shown; teams that rarely qualify have near-zero values."
    )

    # Heatmap: teams x stages
    top_n = st.slider("Show top N teams by champion probability", 10, 48, 24)
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
        title=f"Advancement probabilities (%) — top {top_n} by champion odds",
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
    st.header("Champion Probability")
    st.caption(
        "Horizontal bar chart with 95% confidence intervals (normal approximation). "
        "Wider bars indicate less certainty; run more iterations to tighten them."
    )

    top_n = st.slider("Show top N teams", 10, 48, 20, key="champ_n")
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
        xaxis_title="Probability (%)",
        height=max(400, top_n * 28 + 80),
        margin={"l": 200, "r": 80, "t": 40, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Group-level champion totals
    st.subheader("Champion probability by group")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
else:
    # Streamlit runs the module directly; execute main at import time
    main()
