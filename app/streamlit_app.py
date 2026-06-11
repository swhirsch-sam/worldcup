"""World Cup 2026 Bracket Predictor — Streamlit report.

Reads results/simulation_summary.json. Never runs the simulation on page load.
Run `python3 -m src.model.montecarlo` first to generate the results file,
then `streamlit run app/streamlit_app.py` (or `make app`) to launch.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (relative to repo root, resolved from this file's location)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
SIMULATION_SUMMARY = _ROOT / "results" / "simulation_summary.json"
MATCH_PREDICTIONS = _ROOT / "results" / "match_predictions.json"
RUN_MANIFEST = _ROOT / "results" / "run_manifest.json"
GROUPS_JSON = _ROOT / "data" / "groups.json"
MATCH_TRACKER = _ROOT / "data" / "match_tracker.json"
MY_BRACKET = _ROOT / "data" / "my_bracket.json"

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

# Match-card result-bar colors: left team wins / draw / right team wins.
HOME_COLOR = "#2e7d32"  # green
DRAW_COLOR = "#9e9e9e"  # grey
AWAY_COLOR = "#1565c0"  # blue

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

# Clean, consistent styling for every Plotly chart in the app (cosmetic only).
pio.templates.default = "plotly_white"
try:
    _tpl = pio.templates["plotly_white"].layout
    _tpl.font.family = "Inter, -apple-system, 'Segoe UI', sans-serif"
    _tpl.paper_bgcolor = "rgba(0,0,0,0)"
    _tpl.colorway = ["#1D6FB8", "#F0A500", "#0E9F6E", "#C0392B", "#6C5CE7", "#16A085"]
except Exception:  # pragma: no cover - styling must never break the app
    pass


def _inject_theme() -> None:
    """Inject web fonts, a navy/gold palette, and component styling for polish."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--wc-navy:#0B1F3A;--wc-blue:#1D6FB8;
--wc-gold:#F0A500;--wc-line:#E5EAF1;}
html, body, [class*="css"]{font-family:'Inter',-apple-system,BlinkMacSystemFont,
'Segoe UI',sans-serif;}
.stApp{background:linear-gradient(180deg,#FBFCFE 0%,#F2F6FB 100%);}
.block-container{padding-top:1.6rem;}
.wc-hero{background:linear-gradient(120deg,#0B1F3A 0%,#14346B 52%,#1D6FB8 100%);
  border-radius:18px;padding:30px 34px;margin-bottom:24px;
  box-shadow:0 12px 32px rgba(11,31,58,.28);position:relative;overflow:hidden;}
.wc-hero:after{content:"\\26BD";position:absolute;right:26px;top:50%;transform:translateY(-50%);
  font-size:128px;opacity:.10;line-height:1;}
.wc-hero h1{color:#fff;font-weight:800;font-size:2.15rem;margin:0;letter-spacing:-.5px;}
.wc-hero h1 span{color:#FFC94D;}
.wc-hero p{color:#CBD9EC;margin:.45rem 0 0;font-size:1.03rem;}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--wc-line);border-radius:14px;
  padding:16px 18px;box-shadow:0 1px 3px rgba(16,24,40,.06);
  transition:transform .15s ease,box-shadow .15s ease;}
[data-testid="stMetric"]:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(16,24,40,.10);}
[data-testid="stMetricLabel"] p{font-weight:600;color:#5A6B85;font-size:.80rem;
  text-transform:uppercase;letter-spacing:.045em;}
[data-testid="stMetricValue"]{color:var(--wc-navy);font-weight:800;}
h2,h3{color:var(--wc-navy);font-weight:700;letter-spacing:-.3px;}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:none;flex-wrap:wrap;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}
.stTabs [data-baseweb="tab"]{background:#EAF1F8;border-radius:10px;padding:8px 16px;
  font-weight:600;color:#33455F;border:1px solid transparent;}
.stTabs [aria-selected="true"]{background:var(--wc-blue);color:#fff;
  box-shadow:0 4px 12px rgba(29,111,184,.30);}
[data-testid="stExpander"]{border:1px solid var(--wc-line);border-radius:12px;}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;border:1px solid var(--wc-line);}
</style>
        """,
        unsafe_allow_html=True,
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
def load_match_predictions() -> dict[str, Any] | None:
    if not MATCH_PREDICTIONS.exists():
        return None
    with open(MATCH_PREDICTIONS, encoding="utf-8") as f:
        return json.load(f)


def _load_tracker() -> dict[str, Any]:
    """Group-stage pick tracker. Uncached so a freshly committed file shows up."""
    if not MATCH_TRACKER.exists():
        return {"meta": {}, "matches": []}
    with open(MATCH_TRACKER, encoding="utf-8") as f:
        return json.load(f)


def _load_bracket() -> dict[str, Any]:
    """Personal knockout bracket projection. Uncached so commits show up."""
    if not MY_BRACKET.exists():
        return {}
    with open(MY_BRACKET, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_predictor() -> Callable[..., dict[str, Any]] | None:
    """Lazily import the numpy-only closed-form predictor for live head-to-head.

    Returns None if the import fails (e.g. ``src`` not importable), so the rest
    of the page keeps working from the precomputed file.
    """
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    try:
        from src.model.scoreline import predict_outcome

        return predict_outcome
    except Exception:  # pragma: no cover - defensive import guard
        return None


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


# ---------------------------------------------------------------------------
# Match-card rendering (Match Predictions tab + head-to-head)
# ---------------------------------------------------------------------------


def _wdl_bar_html(p_home: float, p_draw: float, p_away: float) -> str:
    """A single stacked horizontal bar showing win/draw/loss probabilities."""

    def seg(pct: float, color: str) -> str:
        text = f"{pct * 100:.0f}%" if pct >= 0.08 else ""
        return f'<div style="flex:0 0 {pct * 100:.2f}%;background:{color};">{text}</div>'

    return (
        '<div style="display:flex;width:100%;height:24px;border-radius:6px;overflow:hidden;'
        'font-size:12px;font-weight:600;color:#fff;text-align:center;line-height:24px;">'
        + seg(p_home, HOME_COLOR)
        + seg(p_draw, DRAW_COLOR)
        + seg(p_away, AWAY_COLOR)
        + "</div>"
    )


def _legend_html() -> str:
    return (
        '<div style="font-size:12px;color:#666;margin-bottom:6px;">'
        f'<span style="color:{HOME_COLOR};font-weight:700;">■</span> left team wins'
        "&nbsp;&nbsp;&nbsp;"
        f'<span style="color:{DRAW_COLOR};font-weight:700;">■</span> draw'
        "&nbsp;&nbsp;&nbsp;"
        f'<span style="color:{AWAY_COLOR};font-weight:700;">■</span> right team wins</div>'
    )


def _render_match_card(
    home: str, away: str, stats: dict[str, Any], group: str | None = None
) -> None:
    """Render one match prediction as a name row + result bar + summary caption."""
    p_home = float(stats["p_home"])
    p_draw = float(stats["p_draw"])
    p_away = float(stats["p_away"])

    # Bold the more-likely-to-win side.
    home_lbl = f"**{home}**" if p_home >= p_away else home
    away_lbl = f"**{away}**" if p_away > p_home else away
    group_tag = f"  —  Group {group}" if group else ""
    st.markdown(f"{home_lbl} &nbsp;vs&nbsp; {away_lbl}{group_tag}")

    st.markdown(_wdl_bar_html(p_home, p_draw, p_away), unsafe_allow_html=True)

    top = stats["top_scores"][0]
    st.caption(
        f"Likely score: {home} {int(top['home'])}-{int(top['away'])} {away}"
        f" &nbsp;|&nbsp; chances {p_home:.0%} / {p_draw:.0%} / {p_away:.0%}"
        f" &nbsp;|&nbsp; expected goals {stats['exp_home']:.1f}-{stats['exp_away']:.1f}"
    )
    st.write("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _inject_theme()
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
    st.markdown(
        f'<div class="wc-hero">'
        f"<h1>FIFA World Cup <span>2026</span> &mdash; Bracket Predictor</h1>"
        f"<p>Probabilistic forecasts from <b>{n_iter:,}</b> full-tournament "
        f"Monte&nbsp;Carlo simulations &middot; updated {meta.get('generated_at', '')[:10]}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Projected Champion",
        top_team,
        help=(
            "The team with the highest estimated probability of winning the 2026 World Cup, "
            "based on the simulation results. This is the model's best guess — not a guarantee."
        ),
    )
    c2.metric(
        "Title Odds",
        f"{top_prob:.1%}",
        help=(
            "Estimated probability that the projected champion wins the tournament outright. "
            "Even the top favourite rarely exceeds 20-25% — football is unpredictable."
        ),
    )
    c3.metric(
        "Simulations Run",
        f"{n_iter:,}",
        help=(
            "Number of complete tournament simulations used to calculate these probabilities. "
            "More simulations produce tighter, more reliable estimates."
        ),
    )
    c4.metric(
        "Last Updated",
        meta.get("generated_at", "")[:10],
        help=(
            "Date the simulation was last run. "
            "Run `python3 -m src.model.montecarlo` to refresh with the latest data."
        ),
    )

    st.markdown(
        f"**How it works:** We simulated the entire 2026 World Cup **{n_iter:,} times** using a "
        "statistical model trained on **49,000+ historical international matches** (1872-present). "
        "Each match outcome is determined by team strength learned from decades of real results — "
        "stronger teams win more often, but upsets still happen. After all those simulations, we "
        "count how often each team reached each round. Those frequencies become the percentages "
        "you see throughout this app. For the technical details, see the "
        "**Model Stats** and **Methodology** tabs.\n\n"
        "**Data sources:** Historical match results (49k+ games, 1872-present) · "
        "Elo strength ratings · FIFA rankings · Polymarket predictions · "
        "WC 2026 official bracket format"
    )

    st.divider()

    tabs = st.tabs(
        [
            "Model Picks (Group Stage)",
            "🗳️ My Bracket",
            "Match Predictions",
            "Group Stage",
            "Advancement",
            "Champion Odds",
            "Model Stats",
            "Methodology",
        ]
    )

    with tabs[0]:
        _render_model_picks()
    with tabs[1]:
        _render_my_bracket()
    with tabs[2]:
        _render_matches(load_match_predictions())
    with tabs[3]:
        _render_group_stage(probs, groups)
    with tabs[4]:
        _render_advancement(team_df)
    with tabs[5]:
        _render_champion_odds(team_df, n_iter)
    with tabs[6]:
        _render_model_stats(team_df, meta, n_iter)
    with tabs[7]:
        _render_methodology(manifest, meta)


# ---------------------------------------------------------------------------
# Shared footer
# ---------------------------------------------------------------------------


def _render_footer() -> None:
    st.markdown(
        """
<div style="
  background: #f0f4f8;
  border-top: 3px solid #1a6eb5;
  border-radius: 6px;
  padding: 14px 20px;
  margin-top: 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
">
  <span style="color: #4a4a4a; font-size: 0.82em; flex: 1;">
    <a href="https://www.eloratings.net/" target="_blank"
    style="color:#1a6eb5;">Elo strength ratings</a>
    &nbsp;&middot;&nbsp;
    <a href="https://www.rsssf.org/miscellaneous/intlrecs.html"
    target="_blank" style="color:#1a6eb5;">49,000+ historical matches</a>
    &nbsp;&middot;&nbsp;
    <a href="https://polymarket.com/event/2026-fifa-world-cup-winner"
    target="_blank" style="color:#1a6eb5;">Polymarket predictions</a>
    &nbsp;&middot;&nbsp;
    <a href="https://www.fifa.com/en/ranking/men"
    target="_blank" style="color:#1a6eb5;">FIFA rankings</a>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    See the <b>Methodology</b> tab for details
  </span>
  <span style="white-space: nowrap; font-size: 0.88em; font-weight: 700; color: #1a6eb5;">
    Created by&nbsp;<a href="https://samhirsch.com" target="_blank"
      style="color: #1a6eb5; text-decoration: none;">Sam Hirsch</a>
  </span>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_model_picks() -> None:
    """Read-only group-stage tracker: the model's pick vs. the actual result."""
    st.header("Model Picks — Group Stage")

    base = _load_tracker()
    matches: list[dict[str, Any]] = base.get("matches", [])
    if not matches:
        st.info("No tracker data yet — `data/match_tracker.json` is missing.")
        return

    decided = [m for m in matches if m.get("actual")]
    n_dec = len(decided)
    mod_hits = sum(1 for m in decided if m["model_pick"] == m["actual"])
    mod_acc = mod_hits / n_dec if n_dec else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Results in", f"{n_dec}/{len(matches)}")
    c2.metric("Model correct", f"{mod_hits}/{n_dec}" if n_dec else "—")
    c3.metric("Model accuracy", f"{mod_acc:.0%}" if n_dec else "—")
    st.progress(n_dec / len(matches), text=f"{n_dec} of {len(matches)} group matches played")
    st.divider()

    def _hit(pick: str | None, actual: str | None) -> str:
        if not pick or not actual:
            return ""
        return "✅" if pick == actual else "❌"

    df = pd.DataFrame(
        {
            "MD": m["matchday"],
            "Grp": m["group"],
            "Match": f"{m['home']} v {m['away']}",
            "Model pick": m["model_pick"],
            "Result": m.get("actual") or "—",
            "✓": _hit(m["model_pick"], m.get("actual")),
        }
        for m in matches
    )
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        height=560,
        column_config={
            "MD": st.column_config.NumberColumn("MD", width="small"),
            "Grp": st.column_config.TextColumn("Grp", width="small"),
            "✓": st.column_config.TextColumn("✓", width="small"),
        },
    )
    _render_footer()


def _render_my_bracket() -> None:
    """Read-only knockout bracket: my projection vs. the model vs. actual."""
    st.header("🗳️ My Bracket — Knockout Projection")

    b = _load_bracket()
    if not b or not b.get("rounds"):
        st.info("No bracket yet — `data/my_bracket.json` is missing.")
        return

    champ = b.get("champion", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("My champion", champ.get("mine") or "—")
    c2.metric("Model champion", champ.get("model") or "—")
    c3.metric("Actual champion", champ.get("actual") or "TBD")

    all_m = [m for r in b["rounds"].values() for m in r]
    deviations = sum(1 for m in all_m if m.get("mine") and m.get("mine") != m.get("model"))
    decided = [m for m in all_m if m.get("actual")]
    my_hits = sum(1 for m in decided if m.get("mine") == m["actual"])
    mod_hits = sum(1 for m in decided if m.get("model") == m["actual"])
    d1, d2, d3 = st.columns(3)
    d1.metric("Ties played", f"{len(decided)}/{len(all_m)}")
    d2.metric("My calls right", f"{my_hits}/{len(decided)}" if decided else "—")
    d3.metric("Model calls right", f"{mod_hits}/{len(decided)}" if decided else "—")
    if deviations:
        st.caption(f"You've flipped **{deviations}** call(s) from the model (marked ⚡).")
    st.divider()

    labels = {
        "Final": "🏆 Final",
        "SF": "Semifinals",
        "QF": "Quarterfinals",
        "R16": "Round of 16",
        "R32": "Round of 32",
    }
    for rk in ["Final", "SF", "QF", "R16", "R32"]:
        matches = b["rounds"].get(rk, [])
        if not matches:
            continue
        st.subheader(labels.get(rk, rk))
        rows = []
        for m in matches:
            mine, model, actual = m.get("mine"), m.get("model"), m.get("actual")
            mark = "" if not actual else ("✅" if mine == actual else "❌")
            flip = " ⚡" if (mine and model and mine != model) else ""
            rows.append(
                {
                    "Match": f"{m['team_a']} v {m['team_b']}",
                    "My pick": (mine or "—") + flip,
                    "Model pick": model or "—",
                    "Actual": actual or "—",
                    "✓": mark,
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            column_config={"✓": st.column_config.TextColumn("✓", width="small")},
        )
    _render_footer()


def _render_matches(match_data: dict[str, Any] | None) -> None:
    st.header("Match-by-Match Predictions")

    if match_data is None:
        st.info(
            "Match predictions haven't been generated yet. Run "
            "`python3 -m src.model.match_predict` (or `make predict`) and reload."
        )
        return

    groups: dict[str, list[str]] = match_data["groups"]
    matches: list[dict[str, Any]] = match_data["group_matches"]

    # --- Group-stage matches ---
    st.subheader("Group-stage matches")
    group_ids = sorted(groups.keys())
    options = [f"Group {g}" for g in group_ids] + ["All groups"]
    choice = st.selectbox(
        "Choose a group",
        options,
        index=0,
        help="All 72 group-stage matchups are fixed by the official draw.",
    )

    if choice == "All groups":
        selected = matches
    else:
        wanted = choice.split()[-1]
        selected = [m for m in matches if m["group"] == wanted]

    st.markdown(_legend_html(), unsafe_allow_html=True)
    show_group = choice == "All groups"
    for m in selected:
        _render_match_card(m["home"], m["away"], m, group=m["group"] if show_group else None)
    _render_footer()


def _render_group_stage(probs: dict[str, Any], groups: dict[str, list[str]]) -> None:
    st.header("Group Stage — Who Advances?")

    cols = st.columns(3)
    for idx, (group_id, teams) in enumerate(sorted(groups.items())):
        with cols[idx % 3]:
            st.subheader(f"Group {group_id}")
            rows = []
            for team in teams:
                p = probs.get(team, {})
                rows.append(
                    {
                        "Team": team,
                        "Finish 1st": p.get("group_first", 0.0),
                        "Finish 2nd": p.get("group_second", 0.0),
                        "Qualify %": p.get("r32", 0.0),
                    }
                )
            df = pd.DataFrame(rows).sort_values("Qualify %", ascending=False).reset_index(drop=True)
            st.dataframe(
                df.style.format(
                    {"Finish 1st": "{:.1%}", "Finish 2nd": "{:.1%}", "Qualify %": "{:.1%}"}
                ).background_gradient(subset=["Qualify %"], cmap="YlGn"),
                hide_index=True,
                use_container_width=True,
            )
    _render_footer()


def _render_advancement(team_df: pd.DataFrame) -> None:
    st.header("Advancement Probabilities")

    cols: dict[str, str] = {
        "group": "Group",
        "r32": "R32",
        "r16": "R16",
        "qf": "QF",
        "sf": "SF",
        "final": "Final",
        "champion": "Champion",
    }
    show = team_df[["team", *cols.keys()]].rename(columns={"team": "Team", **cols})
    pct_cols = [v for v in cols.values() if v != "Group"]
    st.dataframe(
        show.style.format(dict.fromkeys(pct_cols, "{:.1%}")).background_gradient(
            subset=pct_cols, cmap="YlGn"
        ),
        hide_index=True,
        use_container_width=True,
        height=600,
    )
    _render_footer()


def _render_champion_odds(team_df: pd.DataFrame, n_iter: int) -> None:
    st.header("Who Wins the World Cup?")

    top_n = st.slider("How many teams to show", 5, 48, 15, key="champ_n")
    chart_df = team_df.head(top_n).sort_values("champion", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=chart_df["champion"] * 100,
            y=chart_df["team"],
            orientation="h",
            marker_color="#1D6FB8",
            text=[f"{p:.1%}" for p in chart_df["champion"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Chance of winning (%)",
        height=max(360, top_n * 26 + 80),
        margin={"l": 160, "r": 60, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)

    table = team_df[["team", "champion"]].sort_values("champion", ascending=False)
    st.dataframe(
        table.rename(columns={"team": "Team", "champion": "Win %"}).style.format(
            {"Win %": "{:.1%}"}
        ),
        hide_index=True,
        use_container_width=True,
        height=400,
    )
    _render_footer()


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
        ma, mb = st.columns(2)
        ma.metric(
            "Effective contenders",
            f"{effective_n:.1f}",
            help=(
                "How many teams are genuinely in contention. "
                "If one team had 100% odds this would be 1. "
                "With 48 equal teams it would be 48. "
                "Derived from the HHI: 1 / sum(p²)."
            ),
        )
        mb.metric(
            "HHI (concentration)",
            f"{hhi:.4f}",
            help=(
                "Herfindahl-Hirschman Index: sum of squared win probabilities. "
                "Higher = more concentrated around a few favorites. "
                "0.02 = perfectly equal field; 1.0 = one team is a certainty."
            ),
        )
        ma.metric(
            "Top-3 title share",
            f"{top3_share:.1%}",
            help="Combined win probability of the three highest-ranked teams.",
        )
        mb.metric(
            "Top-8 title share",
            f"{top8_share:.1%}",
            help="Combined win probability of the top eight teams — the realistic contender pool.",
        )
        st.metric(
            "Simulation runtime",
            f"{meta.get('runtime_seconds', 0):.0f}s",
            help="Wall-clock time to run the full simulation on the generation machine.",
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
    _render_footer()


def _render_methodology(manifest: dict[str, Any], meta: dict[str, Any]) -> None:
    st.header("Methodology & Data Provenance")

    st.subheader("How the model works")
    st.markdown(
        """
**Step 1 — Measuring team strength**

Each team gets a single strength number built from four sources, blended together:

- **Elo ratings (50%)** — A running score updated after every international match since 1872.
  Big tournament wins move it more than friendly wins.
- **Betting market odds (20%)** — Implied win probabilities from bookmakers like Pinnacle,
  with the house margin mathematically removed. Sharp money tends to be well-informed.
- **Polymarket (20%)** — Crowd-sourced prediction market prices. People put real money on
  outcomes, creating a live consensus independent of the bookmakers.
- **FIFA rankings (10%)** — FIFA's own monthly ranking of national teams.

When a source isn't available its weight is automatically redistributed to the others.
Host nations (USA, Canada, Mexico) receive a small home-crowd boost. Teams with limited
match history are nudged toward average to avoid overconfidence.

---

**Step 2 — Predicting match scores**

With team strengths in hand, we predict how many goals each side is likely to score.
The stronger the team and the bigger the gap, the higher their expected goal tally.
We use a model called **Dixon-Coles** (a well-known football statistics method) which
includes a correction for a quirk in the sport: low-scoring draws like 0-0 and 1-1
happen *more often* than basic probability would suggest, so we adjust for that.

---

**Step 3 — Simulating the tournament 1,000 times**

We run the entire World Cup from first group game to the final, 1,000 times.
Each simulation draws a realistic scoreline for every match, determines group standings
(using official FIFA tiebreaker rules), selects the 8 best third-place wildcards, and
plays out the full knockout bracket. After all 1,000 runs, we count how often each team
reached each round — those frequencies become the percentages you see in this app.

---

**Step 4 — Tiebreakers & wildcards**

When teams finish a group level on points, we apply the official FIFA rules in order:
goal difference, goals scored, head-to-head record, and so on down to a coin flip.
For the 8 wildcard spots, the 12 third-place teams are ranked by the same criteria and
the best 8 advance. Bracket slot assignments follow FIFA's official Annex C rules.

---

**Step 5 — Knockout extra time and penalties**

If a knockout match is still tied after 90 minutes, we simulate 30 minutes of extra time
(teams score fewer goals when tired). Still level? We simulate a penalty shootout —
roughly 50/50 for either side, with a small tilt toward the stronger team.
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
    _render_footer()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
else:
    # Streamlit runs the module directly; execute main at import time
    main()
