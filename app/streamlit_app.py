"""World Cup 2026 Bracket Predictor — Streamlit report.

Reads results/simulation_summary.json. Never runs the simulation on page load.
Run `make simulate` first, then `make app` to launch this.

Sections:
  1. Group advancement tables with uncertainty bands
  2. The 8 most likely best-third qualifiers
  3. Most-likely bracket + probabilistic bracket
  4. Champion odds chart with 95% intervals
  5. Calibration / backtest summary
  6. Methodology and data provenance from the run manifest
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

SIMULATION_SUMMARY = Path("results/simulation_summary.json")
RUN_MANIFEST = Path("results/run_manifest.json")

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)


def load_results() -> dict:  # type: ignore[type-arg]
    if not SIMULATION_SUMMARY.exists():
        st.error(
            "No simulation results found. Run `make simulate` first to generate "
            f"`{SIMULATION_SUMMARY}`."
        )
        st.stop()
    with open(SIMULATION_SUMMARY) as f:
        return json.load(f)


def load_manifest() -> dict:  # type: ignore[type-arg]
    if not RUN_MANIFEST.exists():
        return {}
    with open(RUN_MANIFEST) as f:
        return json.load(f)


def main() -> None:
    st.title("⚽ 2026 FIFA World Cup Predictor")
    st.caption("Probabilistic bracket simulation with uncertainty quantification.")

    results = load_results()
    manifest = load_manifest()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Group Stage",
            "Best Third",
            "Bracket",
            "Champion Odds",
            "Calibration",
            "Methodology",
        ]
    )

    with tab1:
        _render_group_stage(results)

    with tab2:
        _render_best_third(results)

    with tab3:
        _render_bracket(results)

    with tab4:
        _render_champion_odds(results)

    with tab5:
        _render_calibration(results)

    with tab6:
        _render_methodology(manifest)


def _render_group_stage(results: dict) -> None:  # type: ignore[type-arg]
    st.header("Group Stage Advancement Probabilities")
    st.info("Full implementation in Phase 8.")


def _render_best_third(results: dict) -> None:  # type: ignore[type-arg]
    st.header("8 Most Likely Best-Third Qualifiers")
    st.info("Full implementation in Phase 8.")


def _render_bracket(results: dict) -> None:  # type: ignore[type-arg]
    st.header("Knockout Bracket")
    st.info("Full implementation in Phase 8.")


def _render_champion_odds(results: dict) -> None:  # type: ignore[type-arg]
    st.header("Champion Odds with 95% Confidence Intervals")
    st.info("Full implementation in Phase 8.")


def _render_calibration(results: dict) -> None:  # type: ignore[type-arg]
    st.header("Calibration & Backtesting Summary")
    st.info("Full implementation in Phase 8.")


def _render_methodology(manifest: dict) -> None:  # type: ignore[type-arg]
    st.header("Methodology & Data Provenance")
    if manifest:
        st.json(manifest)
    else:
        st.info("Run manifest not found. Run `make simulate` to generate it.")


if __name__ == "__main__":
    main()
