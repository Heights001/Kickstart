"""Streamlit dashboard (SCOPE §4 P4). Run via ``engine dashboard``.

Reads processed artifacts; the match explainer builds the promoted-rung sampler
once per session (cached — expect a tuning wait on first use with rung 2).
"""

import json
import os
from pathlib import Path

import plotly.express as px
import streamlit as st

from engine.api.app import AppContext, build_context
from engine.simulation.monte_carlo import ROUNDS
from engine.simulation.state import build_state

PACK_DIR = Path(os.environ.get("ENGINE_PACK", "packs/world_cup_2026"))
DATA_DIR = Path(os.environ.get("ENGINE_DATA_DIR", "data"))
CONFIG_PATH = Path(os.environ.get("ENGINE_CONFIG", "configs/default.yaml"))

st.set_page_config(page_title="predictor-engine", layout="wide")
st.title("World Cup 2026 — predictor-engine")


@st.cache_data
def load_history() -> list[dict]:  # type: ignore[type-arg]
    path = DATA_DIR / "processed" / "prob_history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@st.cache_resource(show_spinner="Fitting the promoted model as-of the data horizon…")
def context() -> AppContext:
    return build_context(PACK_DIR, DATA_DIR, CONFIG_PATH)


def champion_tab() -> None:
    history = load_history()
    if not history:
        st.info("No simulations yet — run `engine simulate` first.")
        return
    latest_gen = max(r["generated_at"] for r in history)
    latest = sorted(
        (r for r in history if r["generated_at"] == latest_gen),
        key=lambda r: -r["champion"],
    )
    st.subheader(f"Champion probabilities (as of {latest[0]['as_of']}, {latest[0]['model']})")
    st.caption("MC standard error only — model uncertainty is not included.")
    table = [{"team": r["team"], **{k: r[k] for k in ROUNDS}} for r in latest]
    st.dataframe(table, width="stretch")
    st.download_button(
        "Download CSV",
        "team,"
        + ",".join(ROUNDS)
        + "\n"
        + "\n".join(f"{r['team']}," + ",".join(str(r[k]) for k in ROUNDS) for r in latest),
        file_name=f"champion_probs_{latest[0]['as_of']}.csv",
    )

    st.subheader("Probability over time")
    top_teams = [r["team"] for r in latest[:10]]
    selected = st.multiselect("Teams", sorted({r["team"] for r in history}), default=top_teams)
    rows = [
        {"as_of": r["as_of"], "team": r["team"], "champion": r["champion"], "model": r["model"]}
        for r in history
        if r["team"] in selected
    ]
    if rows:
        fig = px.line(
            rows, x="as_of", y="champion", color="team", hover_data=["model"], markers=True
        )
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="P(champion)")
        st.plotly_chart(fig, width="stretch")


def bracket_tab() -> None:
    ctx = context()
    state = build_state(ctx.matches, ctx.spec, ctx.sampler.as_of)
    st.subheader(f"Tournament state as of {state.as_of} ({state.completed_count} matches played)")
    left, right = st.columns(2)
    with left:
        st.markdown("**Group results (completed)**")
        for group, results in state.group_results.items():
            if results:
                st.markdown(f"*Group {group}*")
                st.table(
                    [{"home": h, "away": a, "score": f"{hg}-{ag}"} for h, a, hg, ag in results]
                )
    with right:
        st.markdown("**Knockout results (facts)**")
        if state.knockout_facts:
            st.table(
                [
                    {
                        "date": f.date,
                        "match": f"{f.home} vs {f.away}",
                        "score": f"{f.home_goals}-{f.away_goals}",
                        "winner": f.winner,
                    }
                    for f in state.knockout_facts
                ]
            )
        else:
            st.caption("No knockout matches played before the as-of date.")


def explainer_tab() -> None:
    ctx = context()
    teams = sorted(ctx.spec.teams)
    left, right = st.columns(2)
    home = left.selectbox("Home / first team", teams, index=teams.index("argentina"))
    away = right.selectbox("Away / second team", teams, index=teams.index("france"))
    if home == away:
        st.warning("Pick two different teams.")
        return
    probs = ctx.sampler.outcome_probs(home, away)
    st.metric(f"{home} win", f"{probs[0]:.1%}")
    st.metric("draw", f"{probs[1]:.1%}")
    st.metric(f"{away} win", f"{probs[2]:.1%}")
    st.caption(
        f"Model {ctx.model_name}, as-of {ctx.sampler.as_of}. Venue: "
        + ("neutral" if ctx.sampler.is_neutral(home, away) else "host advantage applies")
    )
    x = ctx.sampler.feature_row(home, away)
    st.bar_chart(dict(zip(ctx.sampler.rung.feature_names, x[0], strict=True)))


def report_tab() -> None:
    registry_path = DATA_DIR / "processed" / "model_registry.json"
    if registry_path.exists():
        st.subheader("Promoted model")
        st.json(json.loads(registry_path.read_text()))
    eval_path = DATA_DIR / "processed" / "eval_rung0.json"
    if eval_path.exists():
        st.subheader("Rung 0 backtest windows")
        st.json(json.loads(eval_path.read_text()))
    if not registry_path.exists() and not eval_path.exists():
        st.info("Run `engine train` to produce evaluation artifacts.")


tabs = st.tabs(["Champion odds", "Bracket", "Match explainer", "Model report"])
with tabs[0]:
    champion_tab()
with tabs[1]:
    bracket_tab()
with tabs[2]:
    explainer_tab()
with tabs[3]:
    report_tab()
